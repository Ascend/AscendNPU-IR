//===-------------------- InsertConvertLayout.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "hivm-insert-convert-layout"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_INSERTCONVERTLAYOUT
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct InsertConvertLayoutAroundMmadL1 : public OpRewritePattern<MmadL1Op> {
  using OpRewritePattern<MmadL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    // Cast to interface to get layout info
    auto opWithLayout = dyn_cast<OpWithLayoutInterface>(op.getOperation());
    if (!opWithLayout) {
      return rewriter.notifyMatchFailure(op, "op doesn't implement OpWithLayoutInterface");
    }

    llvm::SmallDenseMap<Value, DataLayoutAttr> currentLayoutMap =
        opWithLayout.getOperandsCurrentLayout();
    LDBG("Checking " << op);
    llvm::SmallDenseMap<Value, DataLayoutAttr> targetLayoutMap =
        opWithLayout.getOperandsTargetFractalLayout();

    Location loc = op.getLoc();

    Value aMatrix = op.getA();
    Value bMatrix = op.getB();
    Value cMatrix = op.getC();

    // Check if already converted (rank 4 check is still a heuristic)
    if (isAlreadyConverted(aMatrix) && isAlreadyConverted(bMatrix) &&
        isAlreadyConverted(cMatrix)) {
      return rewriter.notifyMatchFailure(op, "already converted");
    }

    // Get layouts from the interface
    DataLayoutAttr srcLayoutA = currentLayoutMap.lookup(aMatrix);
    DataLayoutAttr dstLayoutA = targetLayoutMap.lookup(aMatrix);
    DataLayoutAttr srcLayoutB = currentLayoutMap.lookup(bMatrix);
    DataLayoutAttr dstLayoutB = targetLayoutMap.lookup(bMatrix);
    DataLayoutAttr srcLayoutC = currentLayoutMap.lookup(cMatrix);
    DataLayoutAttr dstLayoutC = targetLayoutMap.lookup(cMatrix);

    LDBG("A matrix - src: " << srcLayoutA << ", dst: " << dstLayoutA);
    LDBG("B matrix - src: " << srcLayoutB << ", dst: " << dstLayoutB);
    LDBG("C matrix - src: " << srcLayoutC << ", dst: " << dstLayoutC);

    // Validate we got all layouts
    if (!srcLayoutA || !dstLayoutA || !srcLayoutB || !dstLayoutB ||
        !srcLayoutC || !dstLayoutC) {
      return rewriter.notifyMatchFailure(op, "missing layout info for operands");
    }

    auto newOp = cast<MmadL1Op>(rewriter.clone(*op));
    rewriter.setInsertionPoint(newOp);

    // Convert operands to target layout if needed
    if (failed(convertAndAssignOperand(rewriter, loc, aMatrix,
                                       newOp.getAMutable(), srcLayoutA,
                                       dstLayoutA)))
      return rewriter.notifyMatchFailure(op, "failed to convert A matrix");

    if (failed(convertAndAssignOperand(rewriter, loc, bMatrix,
                                       newOp.getBMutable(), srcLayoutB,
                                       dstLayoutB)))
      return rewriter.notifyMatchFailure(op, "failed to convert B matrix");

    if (failed(convertAndAssignOperand(rewriter, loc, cMatrix,
                                       newOp.getCMutable(), srcLayoutC,
                                       dstLayoutC)))
      return rewriter.notifyMatchFailure(op, "failed to convert C matrix");

    // Update result type and convert back
    newOp.getResult(0).setType(newOp.getC().getType());
    rewriter.setInsertionPointAfter(newOp);

    srcLayoutC = normalizeToND(rewriter.getContext(),
                              srcLayoutC,
                              {hivm::DataLayout::DOTC_ND});

    // Convert result back: from target layout (zN) to source layout (dotC_ND)
    auto ndResult = rewriter.create<ConvertLayoutOp>(
        loc, cMatrix.getType(), newOp.getResult(0),
        dstLayoutC,  // from target layout (e.g., zN)
        srcLayoutC); // back to source layout (e.g., dotC_ND)

    rewriter.replaceOp(op, ndResult);

    LDBG("=== MmadL1Op conversion complete ===");
    return success();
  }

private:
  static bool isAlreadyConverted(Value val) {
    if (auto shapedType = dyn_cast<ShapedType>(val.getType()))
      return shapedType.getRank() == 4;
    return false;
  }

  /// Normalizes a DataLayoutAttr to ND if it matches any of the specified layouts.
  static DataLayoutAttr normalizeToND(MLIRContext *ctx,
                                      DataLayoutAttr layout,
                                      ArrayRef<hivm::DataLayout>
                                      layoutsToNormalize) {
    if (llvm::is_contained(layoutsToNormalize, layout.getDataLayout())) {
      auto newDataLayout = DataLayoutAttr::get(ctx,
                                               hivm::DataLayout::ND);
      LDBG("new data layout " << newDataLayout);
      return newDataLayout;
    }
    return layout;
  }

  /// Converts input to target layout if needed and assigns to the target operand.
  static LogicalResult convertAndAssignOperand(PatternRewriter &rewriter, Location loc,
                                               Value input, OpOperand &targetOperand,
                                               DataLayoutAttr srcLayout,
                                               DataLayoutAttr dstLayout) {
    if (isAlreadyConverted(input)) {
      LDBG("Input already in fractal layout, no conversion needed");
      targetOperand.assign(input);
      return success();
    }

    // Skip conversion if source and target layouts are the same
    if (srcLayout == dstLayout) {
      LDBG("Source and target layouts are the same, no conversion needed");
      targetOperand.assign(input);
      return success();
    }

    auto inputType = cast<ShapedType>(input.getType());
    auto inputShape = llvm::map_to_vector(inputType.getShape(), [&rewriter](auto val) -> OpFoldResult {
      return getAsIndexOpFoldResult(rewriter.getContext(), val);
    });

    auto mixedShape = computeMixedTargetLayoutShape(
        inputShape, srcLayout, dstLayout, rewriter, loc);
    if (failed(mixedShape)) {
      LDBG("Failed to infer fractal type");
      return mixedShape;
    }
    Type convertedType = RankedTensorType::get(decomposeMixedValues(*mixedShape).first, inputType.getElementType());

    srcLayout = normalizeToND(rewriter.getContext(),
                              srcLayout,
                              {hivm::DataLayout::DOTA_ND,
                               hivm::DataLayout::DOTB_ND,
                               hivm::DataLayout::DOTC_ND});

    LDBG("Creating ConvertLayoutOp: " << srcLayout << " -> " << dstLayout);
    auto converted = rewriter.create<ConvertLayoutOp>(
        loc, convertedType, input, srcLayout, dstLayout);
    targetOperand.assign(converted);
    return success();
  }
};

struct InsertConvertLayoutPass
    : public impl::InsertConvertLayoutBase<InsertConvertLayoutPass> {
  void runOnOperation() override {
    LDBG("=== InsertConvertLayoutPass starting ===");
    auto module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    // Add all transformation patterns
    patterns.add<InsertConvertLayoutAroundMmadL1>(context);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    LDBG("Applying patterns with greedy rewrite");
    // Apply patterns with greedy rewrite
    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      LDBG("Pattern application failed");
      signalPassFailure();
    }

    LDBG("=== InsertConvertLayoutPass complete ===");
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createInsertConvertLayoutPass() {
  return std::make_unique<InsertConvertLayoutPass>();
}