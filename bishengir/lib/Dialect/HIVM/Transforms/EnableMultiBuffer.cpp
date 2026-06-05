//===------------------------ EnableMultiBuffer.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

/// Index of yielded value where is alias of targetVal.
std::optional<int> getYieldValueIdx(Value targetVal, ValueRange yieldedValues) {
  auto it = std::find(yieldedValues.begin(), yieldedValues.end(), targetVal);
  if (it != yieldedValues.end()) {
    return it - yieldedValues.begin();
  }

  return std::nullopt;
}

/// Returns true if `val` is genuinely consumed at `loop`'s own body level --
/// i.e. it has at least one non-terminator, non-annotation use whose nearest
/// parent loop is `loop`. Uses nested in deeper loops belong to those inner
/// loops and must not make the enclosing loop the multi-buffer anchor.
static bool isConsumedInLoop(Value val, LoopLikeOpInterface loop) {
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
static LoopLikeOpInterface getParentLoop(Value val,
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

  if (isConsumedInLoop(val, parentLoop))
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
    return getParentLoop(res, consumerLoop);
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
    return getParentLoop(res, consumerLoop);
  }

  auto elseYieldOp = parentIf.elseYield();
  auto elseYieldOpers = elseYieldOp.getOperands();
  auto idxElseYielded = getYieldValueIdx(val, elseYieldOpers);
  if (idxElseYielded.has_value()) {
    auto res = parentIf.getResults()[*idxElseYielded];
    return getParentLoop(res, consumerLoop);
  }

  return consumerLoop ? consumerLoop : parentLoop;
}

LoopLikeOpInterface mlir::hivm::getParentLoop(Value val) {
  return ::getParentLoop(val, nullptr);
}

Value mlir::hivm::createNestedIndexModular(OpBuilder &builder, Operation *op,
                                           int modular) {
  // Resolve the parent loop via the value-following helper (which threads
  // through scf.if yields) and delegate to the LoopLikeOpInterface
  // overload for the actual codegen. Keeping a single source of truth
  // there reflects the unified strategy: scf.for and scf.while share the
  // same alloca-based counter materialized by MultiBufferLoopAdapter
  // (see bishengir/include/bishengir/Dialect/HIVM/Utils/MultiBufferLoopAdapter.h).
  LoopLikeOpInterface parentLoop = getParentLoop(op->getResult(0));
  assert(parentLoop && " op has no proper parent loop to do multi buffer");
  return createNestedIndexModular(builder, parentLoop, modular);
}

Value mlir::hivm::createNestedIndexForOp(OpBuilder &builder,
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

Value mlir::hivm::createNestedIndexModular(OpBuilder &builder,
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
  LogicalResult extMultiBuffer() {
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

struct MultiBufferPattern : public OpRewritePattern<hivm::PointerCastOp> {
  using OpRewritePattern<hivm::PointerCastOp>::OpRewritePattern;

  explicit MultiBufferPattern(MLIRContext *ctx)
      : OpRewritePattern<hivm::PointerCastOp>(ctx) {}

  LogicalResult matchAndRewrite(hivm::PointerCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getAddrs().size() <= 1 || util::isGMPointerCastOp(op)) {
      return failure();
    }

    LoopLikeOpInterface loopOp = getParentLoop(op.getResult());
    while (loopOp) {
      // scf.for and scf.while are both supported (while via alloca-based
      // counter in MultiBufferLoopAdapter); other LoopLike ops bail out.
      if (!isa<scf::ForOp, scf::WhileOp>(loopOp))
        return failure();
      loopOp = loopOp->getParentOfType<LoopLikeOpInterface>();
    }
    return OptMultiBuffer(op);
  }

private:
  LogicalResult OptMultiBuffer(hivm::PointerCastOp op) const;
};

LogicalResult MultiBufferPattern::OptMultiBuffer(hivm::PointerCastOp op) const {
  auto status = MultiBufferHelper(op).extMultiBuffer();
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
  patterns.insert<MultiBufferPattern>(patterns.getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createEnableMultiBufferPass() {
  return std::make_unique<EnableMultiBufferPass>();
}
