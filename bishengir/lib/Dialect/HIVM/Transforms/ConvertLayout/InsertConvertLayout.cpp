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

#include <optional>

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

/// Rank-4 tensors are already in fractal layout for InsertConvertLayout.
bool isAlreadyConverted(Value val) {
  if (!val)
    return false;
  if (auto shapedType = dyn_cast<ShapedType>(val.getType()))
    return shapedType.getRank() == 4;
  return false;
}

/// Collapse DOT*_ND layout aliases to the generic ND layout used by
/// ConvertLayoutOp for matrix operands. Scale layouts (SCALEA_ND / SCALEB_DN)
/// are also mapped here for callers that want a generic ND view; callers that
/// must preserve scale-specific src layouts for load_scale fusion should skip
/// this helper for those operands.
DataLayoutAttr normalizeToND(MLIRContext *ctx, DataLayoutAttr layout) {
  switch (layout.getDataLayout()) {
  case hivm::DataLayout::DOTA_ND:
  case hivm::DataLayout::DOTB_ND:
  case hivm::DataLayout::DOTC_ND:
  case hivm::DataLayout::SCALEA_ND:
  case hivm::DataLayout::SCALEB_DN:
    return DataLayoutAttr::get(ctx, hivm::DataLayout::ND);
  default:
    return layout;
  }
}

/// Insert convert_layout(srcLayout→dstLayout) on `input` when needed and assign
/// the converted value to `targetOperand`. Scale ND layouts are preserved on
/// the ConvertLayoutOp so downstream load_scale fusion can match them.
LogicalResult convertAndAssignOperand(PatternRewriter &rewriter, Location loc,
                                      Value input, OpOperand &targetOperand,
                                      DataLayoutAttr srcLayout,
                                      DataLayoutAttr dstLayout) {
  if (isAlreadyConverted(input)) {
    LDBG("Input already in fractal layout, no conversion needed");
    targetOperand.assign(input);
    return success();
  }

  if (srcLayout == dstLayout) {
    LDBG("Source and target layouts are the same, no conversion needed");
    targetOperand.assign(input);
    return success();
  }

  auto inputType = cast<ShapedType>(input.getType());
  auto inputShape = llvm::map_to_vector(
      inputType.getShape(), [&rewriter](auto val) -> OpFoldResult {
        return getAsIndexOpFoldResult(rewriter.getContext(), val);
      });

  auto mixedShape = computeMixedTargetLayoutShape(inputShape, srcLayout,
                                                  dstLayout, rewriter, loc);
  if (failed(mixedShape)) {
    LDBG("Failed to infer fractal type");
    return mixedShape;
  }
  Type convertedType = RankedTensorType::get(
      decomposeMixedValues(*mixedShape).first, inputType.getElementType());

  DataLayoutAttr convertSrcLayout = srcLayout;
  switch (srcLayout.getDataLayout()) {
  case hivm::DataLayout::SCALEA_ND:
  case hivm::DataLayout::SCALEB_DN:
    break;
  default:
    convertSrcLayout = normalizeToND(rewriter.getContext(), srcLayout);
    break;
  }

  LDBG("Creating ConvertLayoutOp: " << convertSrcLayout << " -> " << dstLayout);
  auto converted = rewriter.create<ConvertLayoutOp>(
      loc, convertedType, input, convertSrcLayout, dstLayout);
  targetOperand.assign(converted);
  return success();
}

/// Insert ND↔fractal convert_layout around local matmul-like ops (mmadL1 /
/// mmadmxL1) that implement OpWithLayoutInterface.
struct InsertConvertLayoutAroundLocalMatmul
    : public OpInterfaceRewritePattern<LocalMatmulLikeOpInterface> {
  using OpInterfaceRewritePattern<
      LocalMatmulLikeOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LocalMatmulLikeOpInterface op,
                                PatternRewriter &rewriter) const override {
    auto opWithLayout = dyn_cast<OpWithLayoutInterface>(op.getOperation());
    if (!opWithLayout) {
      return rewriter.notifyMatchFailure(
          op, "op doesn't implement OpWithLayoutInterface");
    }

    Value aMatrix = op.getMatmulA();
    Value bMatrix = op.getMatmulB();
    Value cMatrix = op.getMatmulC();

    std::optional<MmadMxL1Op> mxOp;
    if (auto casted = dyn_cast<MmadMxL1Op>(op.getOperation()))
      mxOp = casted;

    Value scaleA = mxOp ? mxOp->getScaleA() : Value{};
    Value scaleB = mxOp ? mxOp->getScaleB() : Value{};

    bool alreadyConverted =
        isAlreadyConverted(aMatrix) && isAlreadyConverted(bMatrix) &&
        isAlreadyConverted(cMatrix);
    if (mxOp)
      alreadyConverted = alreadyConverted && isAlreadyConverted(scaleA) &&
                         isAlreadyConverted(scaleB);
    if (alreadyConverted)
      return rewriter.notifyMatchFailure(op, "already converted");

    // TODO: Refactor away from Value→layout maps. Looking up by SSA value is
    // incorrect when the same value is used for more than one operand.
    llvm::SmallDenseMap<Value, DataLayoutAttr> currentLayoutMap =
        opWithLayout.getOperandsCurrentLayout();
    auto targetLayoutMap = opWithLayout.getOperandsTargetFractalLayout();

    DataLayoutAttr srcLayoutA = currentLayoutMap.lookup(aMatrix);
    DataLayoutAttr dstLayoutA =
        dyn_cast_or_null<DataLayoutAttr>(targetLayoutMap.a);
    DataLayoutAttr srcLayoutB = currentLayoutMap.lookup(bMatrix);
    DataLayoutAttr dstLayoutB =
        dyn_cast_or_null<DataLayoutAttr>(targetLayoutMap.b);
    DataLayoutAttr srcLayoutC = currentLayoutMap.lookup(cMatrix);
    DataLayoutAttr dstLayoutC =
        dyn_cast_or_null<DataLayoutAttr>(targetLayoutMap.c);

    DataLayoutAttr srcLayoutScaleA;
    DataLayoutAttr dstLayoutScaleA;
    DataLayoutAttr srcLayoutScaleB;
    DataLayoutAttr dstLayoutScaleB;
    if (mxOp) {
      srcLayoutScaleA = currentLayoutMap.lookup(scaleA);
      dstLayoutScaleA =
          dyn_cast_or_null<DataLayoutAttr>(targetLayoutMap.scaleA);
      srcLayoutScaleB = currentLayoutMap.lookup(scaleB);
      dstLayoutScaleB =
          dyn_cast_or_null<DataLayoutAttr>(targetLayoutMap.scaleB);
    }

    if (!srcLayoutA || !dstLayoutA || !srcLayoutB || !dstLayoutB ||
        !srcLayoutC || !dstLayoutC ||
        (mxOp && (!srcLayoutScaleA || !dstLayoutScaleA || !srcLayoutScaleB ||
                  !dstLayoutScaleB))) {
      llvm::report_fatal_error(
          "InsertConvertLayout: missing layout info for local matmul operands");
    }

    Operation *newOp = rewriter.clone(*op.getOperation());
    rewriter.setInsertionPoint(newOp);
    Location loc = op.getLoc();

    auto convertOperand = [&](OpOperand &operand, DataLayoutAttr src,
                              DataLayoutAttr dst,
                              StringRef name) -> LogicalResult {
      if (failed(convertAndAssignOperand(rewriter, loc, operand.get(), operand,
                                         src, dst)))
        return rewriter.notifyMatchFailure(op, "failed to convert " + name);
      return success();
    };

    if (mxOp) {
      auto newMx = cast<MmadMxL1Op>(newOp);
      if (failed(convertOperand(newMx.getAMutable(), srcLayoutA, dstLayoutA,
                                "A matrix")))
        return failure();
      if (failed(convertOperand(newMx.getBMutable(), srcLayoutB, dstLayoutB,
                                "B matrix")))
        return failure();
      if (failed(convertOperand(newMx.getScaleAMutable(), srcLayoutScaleA,
                                dstLayoutScaleA, "ScaleA")))
        return failure();
      if (failed(convertOperand(newMx.getScaleBMutable(), srcLayoutScaleB,
                                dstLayoutScaleB, "ScaleB")))
        return failure();
      if (failed(convertOperand(newMx.getCMutable(), srcLayoutC, dstLayoutC,
                                "C matrix")))
        return failure();
    } else if (auto newMmad = dyn_cast<MmadL1Op>(newOp)) {
      if (failed(convertOperand(newMmad.getAMutable(), srcLayoutA, dstLayoutA,
                                "A matrix")))
        return failure();
      if (failed(convertOperand(newMmad.getBMutable(), srcLayoutB, dstLayoutB,
                                "B matrix")))
        return failure();
      if (failed(convertOperand(newMmad.getCMutable(), srcLayoutC, dstLayoutC,
                                "C matrix")))
        return failure();
    } else {
      auto newBatch = cast<BatchMmadL1Op>(newOp);
      if (failed(convertOperand(newBatch.getAMutable(), srcLayoutA, dstLayoutA,
                                "A matrix")))
        return failure();
      if (failed(convertOperand(newBatch.getBMutable(), srcLayoutB, dstLayoutB,
                                "B matrix")))
        return failure();
      if (failed(convertOperand(newBatch.getCMutable(), srcLayoutC, dstLayoutC,
                                "C matrix")))
        return failure();
    }

    newOp->getResult(0).setType(
        cast<LocalMatmulLikeOpInterface>(newOp).getMatmulC().getType());
    rewriter.setInsertionPointAfter(newOp);

    srcLayoutC = normalizeToND(rewriter.getContext(), srcLayoutC);
    auto ndResult = rewriter.create<ConvertLayoutOp>(
        loc, cMatrix.getType(), newOp->getResult(0), dstLayoutC, srcLayoutC);

    rewriter.replaceOp(op, ndResult);
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
    patterns.add<InsertConvertLayoutAroundLocalMatmul>(context);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    LDBG("Applying patterns with greedy rewrite");
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
