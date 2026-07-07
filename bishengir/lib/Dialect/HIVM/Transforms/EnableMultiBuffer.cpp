//===------------------------ EnableMultiBuffer.cpp -----------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/MultiBufferLoopAdapter.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hivm-enable-multi-buffer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace mlir {
#define GEN_PASS_DEF_ENABLEMULTIBUFFER
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// MultiBufferHelper
//===----------------------------------------------------------------------===//

// Membase-only: LoopInfo and getLoopsInfo
struct LoopInfo_membase {
  Value inductionVar;
  OpFoldResult lowerBound;
  OpFoldResult upperBound;
  OpFoldResult singleStep;
};

/// Get info of parent and ancestor loop of ptrCastOp.
/// The info is stored from inner to outer parent loop.
std::vector<LoopInfo_membase> getLoopsInfo_membase(LoopLikeOpInterface ptrParentLoop) {
  std::vector<LoopInfo_membase> loopInfoVec;
  LoopLikeOpInterface curOp = ptrParentLoop;
  while (isa_and_nonnull<scf::ForOp>(curOp)) {
    std::optional<Value> inductionVar = curOp.getSingleInductionVar();
    std::optional<OpFoldResult> lowerBound = curOp.getSingleLowerBound();
    std::optional<OpFoldResult> upperBound = curOp.getSingleUpperBound();
    std::optional<OpFoldResult> singleStep = curOp.getSingleStep();

    assert((inductionVar.has_value() && lowerBound.has_value() &&
            upperBound.has_value() && singleStep.has_value()) &&
           "iv, lb, ub and step shouldn't be null.");

    loopInfoVec.push_back(
        {*inductionVar, *lowerBound, *upperBound, *singleStep});

    curOp = curOp->getParentOfType<LoopLikeOpInterface>();
  }

  return loopInfoVec;
}

/// Index of yielded value where is alias of targetVal.
std::optional<int> getYieldValueIdx(Value targetVal, ValueRange yieldedValues) {
  auto it = std::find(yieldedValues.begin(), yieldedValues.end(), targetVal);
  if (it != yieldedValues.end()) {
    return it - yieldedValues.begin();
  }

  return std::nullopt;
}

// Regbase-only helper functions
/// Returns true if `val` is genuinely consumed at `loop`'s own body level --
/// i.e. it has at least one non-terminator, non-annotation use whose nearest
/// parent loop is `loop`. Uses nested in deeper loops belong to those inner
/// loops and must not make the enclosing loop the multi-buffer anchor.
static bool isConsumedInLoop_regbase(Value val, LoopLikeOpInterface loop) {
  Operation *loopOp = loop.getOperation();
  for (Operation *user : val.getUsers()) {
    if (user->hasTrait<OpTrait::IsTerminator>())
      continue;
    if (isa<annotation::MarkOp>(user))
      continue;
    bool nestedInDeeperLoop = false;
    for (Operation *ancestor = user; ancestor;
         ancestor = ancestor->getParentOp()) {
      if (ancestor == loopOp)
        return true;
      if (isa<scf::ForOp, scf::WhileOp>(ancestor)) {
        nestedInDeeperLoop = true;
        break;
      }
    }
    if (nestedInDeeperLoop)
      continue;
  }
  return false;
}

/// Follow loop/if yielded values outward and keep the outermost loop that
/// actually consumes the tracked value. Pure loop-carried values do not update
/// `consumerLoop`; they only bridge the search to an enclosing loop result.
static LoopLikeOpInterface getParentLoop_regbase(Value val,
                                                 LoopLikeOpInterface consumerLoop) {
  auto *valDefOp = val.getDefiningOp();
  if (!valDefOp)
    llvm::report_fatal_error("val should have defining op.");

  // Firstly, get parent loop
  LoopLikeOpInterface parentLoop =
      valDefOp->getParentOfType<LoopLikeOpInterface>();
  if (!parentLoop) {
    return consumerLoop;
  }

  if (isConsumedInLoop_regbase(val, parentLoop))
    consumerLoop = parentLoop;

  // Need to determine whether val is yielded by the loop.
  auto yieldedValues = parentLoop.getYieldedValues();
  if (yieldedValues.empty())
    return consumerLoop ? consumerLoop : parentLoop;

  auto idxLoopRes = getYieldValueIdx(val, yieldedValues);
  if (idxLoopRes.has_value()) {
    // Continue tracking the loop result. If an enclosing loop consumes that
    // result, it becomes the new anchor. If the enclosing loops only forward
    // the value via yields, keep the innermost consumer found so far.
    //
    // Some loop ops (e.g. scf.while) do not expose their results through the
    // LoopLikeOpInterface -- getLoopResults() returns std::nullopt. In that
    // case we cannot follow the value outward through a loop result, so stop
    // here and keep the anchor we already found. (For scf.while the yielded
    // value maps to a before-region iter_arg rather than a loop result, so
    // there is nothing to track further outward anyway.)
    auto loopResults = parentLoop.getLoopResults();
    if (!loopResults ||
        *idxLoopRes >= static_cast<int>(loopResults->size()))
      return consumerLoop ? consumerLoop : parentLoop;
    auto res = (*loopResults)[*idxLoopRes];
    return getParentLoop_regbase(res, consumerLoop);
  }

  // Need to determine whether val is yielded by if/else.
  auto parentIf = valDefOp->getParentOfType<scf::IfOp>();
  if (!parentIf || parentIf.getResults().empty())
    return consumerLoop ? consumerLoop : parentLoop;

  auto thenYieldOp = parentIf.thenYield();
  auto thenYieldOpers = thenYieldOp.getOperands();

  auto idxThenYielded = getYieldValueIdx(val, thenYieldOpers);
  if (idxThenYielded.has_value()) {
    // The val is yielded by ifOp, need to find parent loop of ifOp's result
    auto res = parentIf.getResults()[*idxThenYielded];
    return getParentLoop_regbase(res, consumerLoop);
  }

  auto elseYieldOp = parentIf.elseYield();
  auto elseYieldOpers = elseYieldOp.getOperands();
  auto idxElseYielded = getYieldValueIdx(val, elseYieldOpers);
  if (idxElseYielded.has_value()) {
    auto res = parentIf.getResults()[*idxElseYielded];
    return getParentLoop_regbase(res, consumerLoop);
  }

  return consumerLoop ? consumerLoop : parentLoop;
}

LoopLikeOpInterface mlir::hivm::getParentLoop_regbase(Value val) {
  return getParentLoop_regbase(val, nullptr);
}

Value mlir::hivm::createNestedIndexForOp_regbase(OpBuilder &builder,
                                                 Operation *operation) {
  // Historical name (kept for call-site stability in SyncCodegen.cpp and
  // SyncSolverCodeGen.cpp); semantically equivalent to
  // `createNestedIndexModular(op, /*modular=*/-1)` but tolerates missing
  // parent loops by returning nullptr instead of asserting.
  LoopLikeOpInterface parentLoop =
      operation->getParentOfType<LoopLikeOpInterface>();
  if (!parentLoop)
    return nullptr;
  auto adapterOr = MultiBufferLoopAdapter::create(parentLoop);
  if (failed(adapterOr))
    return nullptr;
  return adapterOr->getIterationCounter(builder);
}

Value mlir::hivm::createNestedIndexModular_regbase(OpBuilder &builder,
                                                   Operation *op,
                                                   int modular) {
  // Unified path: same as the (Operation*, int) overload but with the
  // parent loop supplied directly (used by GraphSyncSolver's SetWait
  // codegen).
  LoopLikeOpInterface parentLoop = getParentLoop_regbase(op->getResult(0));
  assert(parentLoop && " op has no proper parent loop to do multi buffer");
  return createNestedIndexModular_regbase(builder, parentLoop, modular);
}

Value mlir::hivm::createNestedIndexModular_regbase(OpBuilder &builder,
                                                   LoopLikeOpInterface loopOp,
                                                   int modular) {
  // Unified path: same as the (Operation*, int) overload but with the
  // parent loop supplied directly (used by GraphSyncSolver's SetWait
  // codegen).
  auto adapterOr = MultiBufferLoopAdapter::create(loopOp);
  assert(succeeded(adapterOr) &&
         "createNestedIndexModular: loop is neither scf.for nor scf.while");
  return modular == -1 ? adapterOr->getIterationCounter(builder)
                       : adapterOr->getModuloIndex(builder, modular);
}

class MultiBufferHelper {
public:
  explicit MultiBufferHelper(hivm::PointerCastOp &ptrCastOp)
      : ptrCastOp_(ptrCastOp) {}

  /// Transformation to do multi-buffering/array expansion to remove
  /// dependencies on the temporary pointerCastOp between consecutive loop
  /// iterations. It returns the new pointerCastOp if the original pointerCastOp
  /// was multi-buffered and returns failure() otherwise. Example:
  /// ```
  /// scf.for %iv = %c0 to %c16 step %c4 {
  ///   %0 = hivm.hir.pointer_cast(addr1, addr2) [] : memref<4x128xf32>
  ///   annotation.mark %0 {hivm.multi_buffer = 2 : i32}
  ///   "some_use"(%0) : (memref<4x128xf32>) -> ()
  /// }
  /// ```
  /// into:
  /// ```
  /// #map = affine_map<()[s0] -> ((s0 floordiv 4) mod 2)>
  /// %0 = hivm.hir.pointer_cast(addr1) [] : memref<4x128xf32>
  /// %1 = hivm.hir.pointer_cast(addr2) [] : memref<4x128xf32>
  /// scf.for %iv = %c0 to %c16 step %c4 {
  ///   %2 = affine.apply #map()[%iv]
  ///   %3 = arith.index_cast %2 : index to i64
  ///   %4 = ... // Select a value within [%0, %1] based on the value of %3.
  ///   "some_use"(%4) : (memref<4x128xf32, strided<...>) -> ()
  /// }
  /// ```
  LogicalResult extMultiBuffer_membase() {
    LLVM_DEBUG(DBGS() << "Try multi buffer: " << ptrCastOp_ << "\n");
    LLVM_DEBUG(DBGS() << "Enable multi-buffer in split buffer mode\n");

    assert(ptrCastOp_ && "ptrCastOp can't be null.");
    if (!ptrCastOp_->getParentOfType<LoopLikeOpInterface>()) {
      LLVM_DEBUG(DBGS() << " ptrCastOp has no parent loop!\n");
      return failure();
    }

    OpBuilder builder(ptrCastOp_);
    auto newPtrCastOps = createPtrCastOps(builder);
    // Multi-buffer factor = number of physical buffers (one per addr)
    const unsigned factor = newPtrCastOps.size();
    if (factor < 2) {
      LLVM_DEBUG(DBGS() << "multi-buffer factor < 2, skip\n");
      return failure();
    }
    createMarkOp(builder, newPtrCastOps);

    Location loc = ptrCastOp_->getLoc();
    auto idxType = builder.getI64Type();
    Value modularIndex =
        createNestedIndexModular(builder, ptrCastOp_.getOperation(), factor);
    Value modularIdx =
        builder.create<arith::IndexCastOp>(loc, idxType, modularIndex);

    // Build N-way selection:
    //   selected = buf0;
    //   for i in 1..factor-1:
    //     if (idx == i) selected = bufi else keep previous
    Value selectedBuffer = newPtrCastOps[0];
    for (unsigned i = 1; i < factor; ++i) {
      Value iVal = builder.create<arith::ConstantIntOp>(
          loc, static_cast<int64_t>(i), idxType);
      Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 modularIdx, iVal);
      selectedBuffer = builder.create<arith::SelectOp>(
          loc, ptrCastOp_.getType(), cond, newPtrCastOps[i], selectedBuffer);
    }

    ptrCastOp_.replaceAllUsesWith(selectedBuffer);
    ptrCastOp_.erase();
    return success();
  }

  /// Transformation to do multi-buffering/array expansion to remove
  /// dependencies on the temporary pointerCastOp between consecutive loop
  /// iterations. It returns the new pointerCastOp if the original
  /// pointerCastOp was multi-buffered and returns failure() otherwise.
  /// Example (scf.for; scf.while is fully analogous, body block is
  /// `whileOp.getAfter().front()`):
  /// ```
  /// scf.for %iv = %c0 to %c16 step %c4 {
  ///   %0 = hivm.hir.pointer_cast(addr1, addr2) [] : memref<4x128xf32>
  ///   annotation.mark %0 {hivm.multi_buffer = 2 : i32}
  ///   "some_use"(%0) : (memref<4x128xf32>) -> ()
  /// }
  /// ```
  /// into (unified alloca-based counter, see MultiBufferLoopAdapter):
  /// ```
  /// %counter = memref.alloca() {hivm.multi_buffer_counter_for = 0 : i64} :
  ///                memref<1xi64>
  /// memref.store %c0_i64, %counter[%c0]
  /// %0 = hivm.hir.pointer_cast(addr1) [] : memref<4x128xf32>
  /// %1 = hivm.hir.pointer_cast(addr2) [] : memref<4x128xf32>
  /// scf.for %iv = %c0 to %c16 step %c4 {hivm.multi_buffer_loop_id = 0 : i64} {
  ///   %loaded = memref.load %counter[%c0] : memref<1xi64>
  ///   %idx    = arith.remui %loaded, %c2_i64 : i64
  ///   %cond   = arith.cmpi eq, %idx, %c1_i64 : i64
  ///   %sel    = arith.select %cond, %1, %0 : memref<4x128xf32>
  ///   "some_use"(%sel) : (memref<4x128xf32>) -> ()
  ///   %next   = arith.addi %loaded, %c1_i64 : i64
  ///   memref.store %next, %counter[%c0]
  /// }
  /// ```
  LogicalResult extMultiBuffer_regbase() {
    LLVM_DEBUG(DBGS() << "Try multi buffer: " << ptrCastOp_ << "\n");
    LLVM_DEBUG(DBGS() << "Enable multi-buffer in split buffer mode\n");

    assert(ptrCastOp_ && "ptrCastOp can't be null.");
    if (!ptrCastOp_->getParentOfType<LoopLikeOpInterface>()) {
      LLVM_DEBUG(DBGS() << " ptrCastOp has no parent loop!\n");
      return failure();
    }

    OpBuilder builder(ptrCastOp_);
    auto newPtrCastOps = createPtrCastOps(builder);
    // Multi-buffer factor = number of physical buffers (one per addr)
    const unsigned factor = newPtrCastOps.size();
    if (factor < 2) {
      LLVM_DEBUG(DBGS() << "multi-buffer factor < 2, skip\n");
      return failure();
    }
    createMarkOp(builder, newPtrCastOps);

    Location loc = ptrCastOp_->getLoc();
    auto idxType = builder.getI64Type();
    Value modularIndex =
        createNestedIndexModular_regbase(builder, ptrCastOp_.getOperation(), factor);
    Value modularIdx =
        builder.create<arith::IndexCastOp>(loc, idxType, modularIndex);

    // Build N-way selection:
    //   selected = buf0;
    //   for i in 1..factor-1:
    //     if (idx == i) selected = bufi else keep previous
    Value selectedBuffer = newPtrCastOps[0];
    for (unsigned i = 1; i < factor; ++i) {
      Value iVal = builder.create<arith::ConstantIntOp>(
          loc, idxType, static_cast<int64_t>(i));
      Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 modularIdx, iVal);
      selectedBuffer = builder.create<arith::SelectOp>(
          loc, ptrCastOp_.getType(), cond, newPtrCastOps[i], selectedBuffer);
    }

    ptrCastOp_.replaceAllUsesWith(selectedBuffer);
    ptrCastOp_.erase();
    return success();
  }

private:
  bool isPtrAddrsConstantIntOp() {
    auto addrs = ptrCastOp_.getAddrs();
    for (auto addr : addrs) {
      if (!isa<arith::ConstantIntOp>(addr.getDefiningOp())) {
        return false;
      }
    }

    return true;
  }

  SmallVector<hivm::PointerCastOp, 2> createPtrCastOps(OpBuilder &builder) {
    // Set insert point to the beginning of func body
    auto funcOp = ptrCastOp_->getParentOfType<FunctionOpInterface>();
    assert(funcOp && "no funcOp found!");
    auto &frontOpInFunc = funcOp->getRegions().front().front();
    builder.setInsertionPointToStart(&frontOpInFunc);

    // Insert point cast addrs
    assert(isPtrAddrsConstantIntOp() &&
           "ptrCastOp's addrs should be constantIntOp.");

    // Insert new point cast ops
    SmallVector<hivm::PointerCastOp, 2> newPtrCastOps;
    for (const auto &addr : ptrCastOp_.getAddrs()) {
      auto newPointCastOp = builder.create<hivm::PointerCastOp>(
          ptrCastOp_->getLoc(), ptrCastOp_.getType(), addr,
          ptrCastOp_.getDynamicSizes());
      newPtrCastOps.push_back(newPointCastOp);
    }

    // No need to move ptrCastOp. But need to hoist addrs of ptrCastOp,
    // otherwise new ptrCastOps can't find them.
    hoistPtrCastOpAddrs(frontOpInFunc);
    return newPtrCastOps;
  }

  void createMarkOp(OpBuilder &builder,
                    const SmallVector<hivm::PointerCastOp, 2> &newPtrCastOps) {
    // Find markOp which marks ptrCastOp.
    // Note that ptrCastOp may have more than one markOp users.
    auto ptrUsers = ptrCastOp_->getUsers();
    std::vector<annotation::MarkOp> markOps;
    for (auto user : ptrUsers) {
      if (isa<annotation::MarkOp>(user)) {
        markOps.push_back(cast<annotation::MarkOp>(user));
      }
    }

    // Create new markOp
    for (auto markOp : markOps) {
      for (auto newPtrCastOp : newPtrCastOps) {
        builder.setInsertionPointAfter(newPtrCastOp);
        auto newMarkOp = builder.create<annotation::MarkOp>(
            ptrCastOp_->getLoc(), markOp->getResultTypes(),
            newPtrCastOp->getResult(0));
        newMarkOp->setAttrs(markOp->getAttrDictionary());
      }

      markOp.erase();
    }
  }

  void hoistPtrCastOpAddrs(Block &frontOpInFunc) {
    auto addrs = ptrCastOp_.getAddrs();

    for (int i = (int)addrs.size() - 1; i >= 0; --i) {
      auto addr = addrs[i];
      auto *addrDefOp = addr.getDefiningOp();
      if (!addrDefOp)
        llvm::report_fatal_error("definingOp of addr shouldn't be null!");
      addrDefOp->moveBefore(&frontOpInFunc, frontOpInFunc.begin());
    }
  }

  hivm::PointerCastOp &ptrCastOp_;
};

//===----------------------------------------------------------------------===//
// EnableMultiBufferPass
//===----------------------------------------------------------------------===//
namespace {

/// This pass enable multi buffer
struct EnableMultiBufferPass
    : public impl::EnableMultiBufferBase<EnableMultiBufferPass> {
  using EnableMultiBufferBase<EnableMultiBufferPass>::EnableMultiBufferBase;

public:
  void runOnOperation() override;
};
} // end anonymous namespace

// Membase version of MultiBufferPattern
struct MultiBufferPattern_membase : public OpRewritePattern<hivm::PointerCastOp> {
  using OpRewritePattern<hivm::PointerCastOp>::OpRewritePattern;

  explicit MultiBufferPattern_membase(MLIRContext *ctx)
      : OpRewritePattern<hivm::PointerCastOp>(ctx) {}

  LogicalResult matchAndRewrite(hivm::PointerCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getAddrs().size() <= 1 || util::isGMPointerCastOp(op)) {
      return failure();
    }

    LoopLikeOpInterface loopOp = getParentLoop(op.getResult());
    /// When the outmost loopOp of PointerCastOp is null ptr, it means multi
    /// buffer can not be enabled directly. CopyOp is needed to copy yield value
    /// (which is also the result of PointerCastOp) to iter_arg of
    /// PointerCastOp's parent loop to enable multi buffer.
    if (!loopOp) {
      LoopLikeOpInterface parentLoop =
          op->getParentOfType<LoopLikeOpInterface>();
      if (!isa_and_nonnull<scf::ForOp>(parentLoop)) {
        LLVM_DEBUG(DBGS() << " PointerCastOp op has no parent for loop!\n");
        return failure();
      }
      auto yieldIndex = GetParentLoopYieldIndex_membase(parentLoop, op.getResult());
      assert(yieldIndex.has_value());
      InsertCopyBeforeLoopYield_membase(rewriter, parentLoop, yieldIndex.value());
    }
    while (loopOp) {
      if (!isa<scf::ForOp>(loopOp))
        return failure();
      loopOp = loopOp->getParentOfType<LoopLikeOpInterface>();
    }
    return OptMultiBuffer_membase(op);
  }

private:
  LogicalResult OptMultiBuffer_membase(hivm::PointerCastOp op) const;
  std::optional<size_t> GetParentLoopYieldIndex_membase(LoopLikeOpInterface parentLoop,
                                                        Value val) const;
  void InsertCopyBeforeLoopYield_membase(PatternRewriter &rewriter,
                                         LoopLikeOpInterface parentLoopOp,
                                         size_t yieldIndex) const;
};

std::optional<size_t>
MultiBufferPattern_membase::GetParentLoopYieldIndex_membase(LoopLikeOpInterface parentLoop,
                                                            Value val) const {
  auto *valDefOp = val.getDefiningOp();
  if (!valDefOp)
    llvm::report_fatal_error("val should have defining op.");

  // Need to determine whether val is yielded by the loop.
  auto yieldedValues = parentLoop.getYieldedValues();
  if (yieldedValues.empty())
    return std::nullopt;

  auto idxLoopRes = getYieldValueIdx(val, yieldedValues);
  if (idxLoopRes.has_value()) {
    return idxLoopRes.value();
  }

  // Need to determine whether val is yielded by if/else.
  auto parentIf = valDefOp->getParentOfType<scf::IfOp>();
  if (!parentIf || parentIf.getResults().empty())
    return std::nullopt;

  auto thenYieldOp = parentIf.thenYield();
  auto thenYieldOpers = thenYieldOp.getOperands();

  auto idxThenYielded = getYieldValueIdx(val, thenYieldOpers);
  if (idxThenYielded.has_value()) {
    // The val is yielded by ifOp, need to find yield index of ifOp's result
    auto res = parentIf.getResults()[*idxThenYielded];
    return GetParentLoopYieldIndex_membase(parentLoop, res);
  }

  auto elseYieldOp = parentIf.elseYield();
  auto elseYieldOpers = elseYieldOp.getOperands();
  auto idxElseYielded = getYieldValueIdx(val, elseYieldOpers);
  if (idxElseYielded.has_value()) {
    auto res = parentIf.getResults()[*idxElseYielded];
    return GetParentLoopYieldIndex_membase(parentLoop, res);
  }

  return std::nullopt;
}

void MultiBufferPattern_membase::InsertCopyBeforeLoopYield_membase(
    PatternRewriter &rewriter, LoopLikeOpInterface parentLoopOp,
    size_t yieldIndex) const {
  rewriter.setInsertionPoint(
      parentLoopOp.getLoopRegions()[0]->front().getTerminator());
  auto iterArgs = parentLoopOp.getRegionIterArgs();
  auto yieldVals = parentLoopOp.getYieldedValues();
  assert(yieldIndex < iterArgs.size() && iterArgs.size() == yieldVals.size());
  Value iterArg = iterArgs[yieldIndex];
  Value originYield = yieldVals[yieldIndex];
  rewriter.create<hivm::CopyOp>(parentLoopOp->getLoc(), TypeRange{},
                                /*src*/ originYield, /*dst*/ iterArg);
  assert(parentLoopOp.getYieldedValuesMutable().has_value());
  rewriter.modifyOpInPlace(parentLoopOp, [&]() {
    parentLoopOp.getYieldedValuesMutable().value()[yieldIndex].assign(iterArg);
  });
}

LogicalResult MultiBufferPattern_membase::OptMultiBuffer_membase(hivm::PointerCastOp op) const {
  auto status = MultiBufferHelper(op).extMultiBuffer_membase();
  if (failed(status)) {
    op.emitError("failed to multibuffer");
    return failure();
  }

  return success();
}

// Regbase version of MultiBufferPattern
struct MultiBufferPattern_regbase : public OpRewritePattern<hivm::PointerCastOp> {
  using OpRewritePattern<hivm::PointerCastOp>::OpRewritePattern;

  explicit MultiBufferPattern_regbase(MLIRContext *ctx)
      : OpRewritePattern<hivm::PointerCastOp>(ctx) {}

  LogicalResult matchAndRewrite(hivm::PointerCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getAddrs().size() <= 1 || util::isGMPointerCastOp(op)) {
      return failure();
    }

    LoopLikeOpInterface loopOp = getParentLoop_regbase(op.getResult());
    while (loopOp) {
      if (!isa<scf::ForOp, scf::WhileOp>(loopOp))
        return failure();
      loopOp = loopOp->getParentOfType<LoopLikeOpInterface>();
    }
    return OptMultiBuffer_regbase(op);
  }

private:
  LogicalResult OptMultiBuffer_regbase(hivm::PointerCastOp op) const;
};

LogicalResult MultiBufferPattern_regbase::OptMultiBuffer_regbase(hivm::PointerCastOp op) const {
  auto status = MultiBufferHelper(op).extMultiBuffer_regbase();
  if (failed(status)) {
    op.emitError("failed to multibuffer");
    return failure();
  }

  return success();
}

void EnableMultiBufferPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  RewritePatternSet patterns(&getContext());
  patterns.insert<MultiBufferPattern_membase>(patterns.getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createEnableMultiBufferPass() {
  return std::make_unique<EnableMultiBufferPass>();
}
