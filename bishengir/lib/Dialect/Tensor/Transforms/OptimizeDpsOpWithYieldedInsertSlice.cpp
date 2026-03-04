//===- OptimizeDpsOpWithYieldedInsertSlice.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Transforms.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_OPTIMIZEDPSOPWITHYIELDEDINSERTSLICE
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::tensor;

namespace {
struct OptimizeDpsOpWithYieldedInsertSlicePass
    : public impl::OptimizeDpsOpWithYieldedInsertSliceBase<
          OptimizeDpsOpWithYieldedInsertSlicePass> {
public:
  void runOnOperation() override;
};

/// Pattern to modify dps op's inits to extract slice if the result

/// is being inserted and yielded.
///
/// For example:
/// ```
/// %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) {
///   %1 = linalg.add ins(...) outs(%empty)
///   %inserted_slice = tensor.insert_slice %1 into %arg1
///   scf.yield %inserted_slice
/// }
/// ```
/// will be optimized to
/// ```
/// %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) {
///   %extract_slice = tensor.extract_slice %arg1
///   %1 = linalg.add ins(...) outs(%extract_slice)
///   %inserted_slice = tensor.insert_slice %1 into %arg1
///   scf.yield %inserted_slice
/// }
/// ```
/// This is because one-shot-bufferize requires an extract and insert pair
/// for region iter args to be considered for inplace reuse. Otherwise there
/// will be extract copies.
struct ModifyDpsInitToSlicedIterArg : public OpRewritePattern<InsertSliceOp> {
  using OpRewritePattern<InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    auto insertSrc = insertSliceOp.getSource();
    auto *srcDefiningOp = insertSrc.getDefiningOp();
    if (!isa_and_nonnull<DestinationStyleOpInterface>(srcDefiningOp))
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "source is not destination style");

    auto dpsSrc = cast<DestinationStyleOpInterface>(srcDefiningOp);
    auto resultNumber = cast<OpResult>(insertSrc).getResultNumber();
    auto tyingInit = dpsSrc.getDpsInitOperand(resultNumber);
    // If the original semantic expects the dps op to write into some value, we
    // cannot do this optimization.
    if (!isa_and_nonnull<EmptyOp>(tyingInit->get().getDefiningOp()))
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "init is not empty tensor");

    auto enclosingFunc = insertSliceOp->getParentOfType<func::FuncOp>();
    // If the mixed offsets, sizes and strides are not dominated by the source,
    // we cannot create the extract slice.
    DominanceInfo domInfo(enclosingFunc);
    for (auto operand : insertSliceOp->getOperands()) {
      if (isa<BlockArgument>(operand))
        continue;

      auto *operandDef = operand.getDefiningOp();
      if (!domInfo.dominates(operandDef, srcDefiningOp))
        return rewriter.notifyMatchFailure(
            insertSliceOp, "insert slice operand doesn't dominate dps, cannot "
                           "create extract slice");
    }

    rewriter.setInsertionPoint(srcDefiningOp);
    auto extractSlice = rewriter.create<ExtractSliceOp>(
        insertSliceOp.getLoc(), insertSliceOp.getDest(),
        insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());

    rewriter.modifyOpInPlace(
        srcDefiningOp, [&dpsSrc, &resultNumber, &extractSlice]() {
          dpsSrc.getDpsInitsMutable()[resultNumber].set(extractSlice);
        });
    return success();
  }
};

} // namespace

void OptimizeDpsOpWithYieldedInsertSlicePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  RewritePatternSet patterns(funcOp.getContext());
  bishengir::tensor::populateOptimizeDpsOpWithYieldedInsertSlicePattern(
      patterns);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass>
mlir::tensor::createOptimizeDpsOpWithYieldedInsertSlicePass() {
  return std::make_unique<OptimizeDpsOpWithYieldedInsertSlicePass>();
}

void bishengir::tensor::populateOptimizeDpsOpWithYieldedInsertSlicePattern(
    mlir::RewritePatternSet &patterns) {
  patterns.insert<ModifyDpsInitToSlicedIterArg>(patterns.getContext());
}