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

struct LoopInfo {
  Value inductionVar;
  OpFoldResult lowerBound;
  OpFoldResult upperBound;
  OpFoldResult singleStep;
};

/// Get info of parent and ancestor loop of ptrCastOp.
/// The info is stored from inner to outer parent loop.
std::vector<LoopInfo> getLoopsInfo(LoopLikeOpInterface ptrParentLoop) {
  std::vector<LoopInfo> loopInfoVec;
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

/// Flatten loops into one dimension and then modulo modular.
/// for example modular is 2, use all induction var of loops to generate
/// sequence data 0,1,2,3,... then mod 2 to get 0,1,0,1,...
///
/// IR:
/// for i, 0, upperI, step=1:
///   for j, 0, upperJ, step=1:
///     affine.apply #map()[j, i]
///
/// the sequence is: 0,1,2,..., i*upperJ + j, ...
///
/// \return IndexCastOp of affineApply
Value createNestedIndexModularUsingLoopInfo(
    OpBuilder &builder, Location loc, const std::vector<LoopInfo> &loopInfoVec,
    int modular) {
  auto ctx = builder.getContext();
  AffineExpr nElems = builder.getAffineConstantExpr(1);
  AffineExpr targetExpr = builder.getAffineConstantExpr(0);
  std::vector<OpFoldResult> symbolValueVec;

  // In order to create affineMap, bind symbols, affineExpr and values are
  // needed. loopInfoVec would be used, loop from inner to outer.
  for (size_t i = 0; i < loopInfoVec.size(); ++i) {
    AffineExpr iv;
    AffineExpr lb;
    AffineExpr ub;
    AffineExpr step;

    // bind symbols
    iv = getAffineSymbolExpr(i * 4, ctx);
    lb = getAffineSymbolExpr(i * 4 + 1, ctx);
    ub = getAffineSymbolExpr(i * 4 + 2, ctx);
    step = getAffineSymbolExpr(i * 4 + 3, ctx);

    // create affineExpr
    targetExpr = targetExpr + (iv - lb).floorDiv(step) * nElems;
    nElems = nElems * ((ub - lb + step - 1).floorDiv(step));

    // values
    auto info = loopInfoVec[i];
    Value ivVal = getValueOrCreateCastToIndexLike(
        builder, loc, builder.getIndexType(), info.inductionVar);
    Value lbVal = getValueOrCreateCastToIndexLike(
        builder, loc, builder.getIndexType(),
        getValueOrCreateConstantIndexOp(builder, loc, info.lowerBound));
    Value ubVal = getValueOrCreateCastToIndexLike(
        builder, loc, builder.getIndexType(),
        getValueOrCreateConstantIndexOp(builder, loc, info.upperBound));
    Value stepVal = getValueOrCreateCastToIndexLike(
        builder, loc, builder.getIndexType(),
        getValueOrCreateConstantIndexOp(builder, loc, info.singleStep));

    symbolValueVec.push_back(ivVal);
    symbolValueVec.push_back(lbVal);
    symbolValueVec.push_back(ubVal);
    symbolValueVec.push_back(stepVal);
  }

  if (modular == -1) {
    Value affineApply = mlir::affine::makeComposedAffineApply(
        builder, loc, targetExpr, ArrayRef(symbolValueVec));
    return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                              affineApply);
  }

  targetExpr = targetExpr % modular;

  Value affineApply = mlir::affine::makeComposedAffineApply(
      builder, loc, targetExpr, ArrayRef(symbolValueVec));
  return builder.create<arith::IndexCastOp>(loc, builder.getI1Type(),
                                            affineApply);
}

/// Index of yielded value where is alias of targetVal.
std::optional<int> getYieldValueIdx(Value targetVal, ValueRange yieldedValues) {
  auto it = std::find(yieldedValues.begin(), yieldedValues.end(), targetVal);
  if (it != yieldedValues.end()) {
    return it - yieldedValues.begin();
  }

  return std::nullopt;
}

LoopLikeOpInterface mlir::hivm::getParentLoop(Value val) {
  auto *valDefOp = val.getDefiningOp();
  if (!valDefOp)
    llvm::report_fatal_error("val should have defining op.");

  // Firstly, get parent loop
  LoopLikeOpInterface parentLoop =
      valDefOp->getParentOfType<LoopLikeOpInterface>();
  if (!parentLoop) {
    return nullptr;
  }

  // Need to determine whether val is yielded by the loop.
  auto yieldedValues = parentLoop.getYieldedValues();
  if (yieldedValues.empty())
    return parentLoop;

  auto idxLoopRes = getYieldValueIdx(val, yieldedValues);
  if (idxLoopRes.has_value()) {
    // The val is yielded by loop, so need to find parent of parent loop.
    auto res = parentLoop.getLoopResults().value()[*idxLoopRes];
    return getParentLoop(res);
  }

  // Need to determine whether val is yielded by if/else.
  auto parentIf = valDefOp->getParentOfType<scf::IfOp>();
  if (!parentIf || parentIf.getResults().empty())
    return parentLoop;

  auto thenYieldOp = parentIf.thenYield();
  auto thenYieldOpers = thenYieldOp.getOperands();

  auto idxThenYielded = getYieldValueIdx(val, thenYieldOpers);
  if (idxThenYielded.has_value()) {
    // The val is yielded by ifOp, need to find parent loop of ifOp's result
    auto res = parentIf.getResults()[*idxThenYielded];
    return getParentLoop(res);
  }

  auto elseYieldOp = parentIf.elseYield();
  auto elseYieldOpers = elseYieldOp.getOperands();
  auto idxElseYielded = getYieldValueIdx(val, elseYieldOpers);
  if (idxElseYielded.has_value()) {
    auto res = parentIf.getResults()[*idxElseYielded];
    return getParentLoop(res);
  }

  return parentLoop;
}

Value mlir::hivm::createNestedIndexModular(OpBuilder &builder, Operation *op,
                                           int modular) {
  LoopLikeOpInterface parentLoop = getParentLoop(op->getResult(0));
  assert(parentLoop &&
         " ptrCastOp has no proper parent loop to do multi buffer");

  auto loopInfoVec = getLoopsInfo(parentLoop);

  // Insert at the beginning of the For loop.
  auto forOp = cast<scf::ForOp>(parentLoop.getOperation());
  builder.setInsertionPointToStart(forOp.getBody());
  return createNestedIndexModularUsingLoopInfo(builder, forOp->getLoc(),
                                               loopInfoVec, modular);
}

Value mlir::hivm::createNestedIndexForOp(OpBuilder &builder,
                                         Operation *operation) {
  LoopLikeOpInterface parentLoop =
      operation->getParentOfType<LoopLikeOpInterface>();
  if (!parentLoop) {
    return nullptr;
  }
  assert(parentLoop &&
         " ptrCastOp has no proper parent loop to do multi buffer");
  if (!llvm::isa<scf::ForOp>(parentLoop)) {
    return nullptr;
  }
  auto loopInfoVec = getLoopsInfo(parentLoop);
  // Insert at the beginning of the For loop.
  auto forOp = cast<scf::ForOp>(parentLoop.getOperation());
  builder.setInsertionPointToStart(forOp.getBody());
  return createNestedIndexModularUsingLoopInfo(builder, forOp->getLoc(),
                                               loopInfoVec, -1);
}

Value mlir::hivm::createNestedIndexModular(OpBuilder &builder,
                                           LoopLikeOpInterface loopOp,
                                           int modular) {
  auto loopInfoVec = getLoopsInfo(loopOp);
  // Insert at the beginning of the For loop.
  auto forOp = dyn_cast<scf::ForOp>(loopOp.getOperation());
  assert(forOp != nullptr);
  builder.setInsertionPointToStart(forOp.getBody());
  return createNestedIndexModularUsingLoopInfo(builder, forOp->getLoc(),
                                               loopInfoVec, modular);
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
  //    %3 = arith.index_cast %2 : index to i1
  //    %4 = arith.select %3, %0, %1 : memref<16xf16, #hivm.address_space<ub>>
  ///   "some_use"(%4) : (memref<4x128xf32, strided<...>) -> ()
  /// }
  /// ```
  LogicalResult extMultiBuffer() {
    LLVM_DEBUG(DBGS() << "Try multi buffer: " << ptrCastOp_ << "\n");
    LLVM_DEBUG(DBGS() << "Currently only supports double buffering (factor = "
                         "2) in split buffer mode\n");

    assert(ptrCastOp_ && "ptrCastOp can't be null.");
    if (!ptrCastOp_->getParentOfType<LoopLikeOpInterface>()) {
      LLVM_DEBUG(DBGS() << " ptrCastOp has no parent loop!\n");
      return failure();
    }

    OpBuilder builder(ptrCastOp_);
    auto newPtrCastOps = createPtrCastOps(builder);
    createMarkOp(builder, newPtrCastOps);

    // create affineApply and indexCast
    Location loc = ptrCastOp_->getLoc();
    auto counter = createNestedIndexModular(builder, ptrCastOp_.getOperation());

    // TODO: currently only support double buffer.
    assert(newPtrCastOps.size() >= 2);
    auto ptrCastOp0 = newPtrCastOps[0];
    auto ptrCastOp1 = newPtrCastOps[1];
    Value selectedBuffer = builder.create<arith::SelectOp>(
        loc, ptrCastOp_.getType(), counter, ptrCastOp0, ptrCastOp1);

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
      if (!isa<scf::ForOp>(loopOp))
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
