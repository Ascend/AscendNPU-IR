//===-------------------- InsertConvertLayout.cpp -------------------------===//
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
/// are preserved for load_scale fusion downstream.
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

/// Insert convert_layout(srcLayout→dstLayout) on `input` when needed.
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

struct InsertConvertLayoutAroundMmadL1 : public OpRewritePattern<MmadL1Op> {
  using OpRewritePattern<MmadL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    // Cast to interface to get layout info
    auto opWithLayout = dyn_cast<OpWithLayoutInterface>(op.getOperation());
    if (!opWithLayout) {
      return rewriter.notifyMatchFailure(
          op, "op doesn't implement OpWithLayoutInterface");
    }

    Value aMatrix = op.getA();
    Value bMatrix = op.getB();
    Value cMatrix = op.getC();

    // Check if already converted (rank 4 check is still a heuristic)
    if (isAlreadyConverted(aMatrix) && isAlreadyConverted(bMatrix) &&
        isAlreadyConverted(cMatrix)) {
      return rewriter.notifyMatchFailure(op, "already converted");
    }

    llvm::SmallDenseMap<Value, DataLayoutAttr> currentLayoutMap =
        opWithLayout.getOperandsCurrentLayout();
    LDBG("Checking " << op);
    auto targetLayoutMap = opWithLayout.getOperandsTargetFractalLayout();

    // Get layouts from the interface
    DataLayoutAttr srcLayoutA = currentLayoutMap.lookup(aMatrix);
    DataLayoutAttr dstLayoutA =
        dyn_cast_or_null<DataLayoutAttr>(targetLayoutMap.a);
    DataLayoutAttr srcLayoutB = currentLayoutMap.lookup(bMatrix);
    DataLayoutAttr dstLayoutB =
        dyn_cast_or_null<DataLayoutAttr>(targetLayoutMap.b);
    DataLayoutAttr srcLayoutC = currentLayoutMap.lookup(cMatrix);
    DataLayoutAttr dstLayoutC =
        dyn_cast_or_null<DataLayoutAttr>(targetLayoutMap.c);

    LDBG("A matrix - src: " << srcLayoutA << ", dst: " << dstLayoutA);
    LDBG("B matrix - src: " << srcLayoutB << ", dst: " << dstLayoutB);
    LDBG("C matrix - src: " << srcLayoutC << ", dst: " << dstLayoutC);

    // Validate we got all layouts
    if (!srcLayoutA || !dstLayoutA || !srcLayoutB || !dstLayoutB ||
        !srcLayoutC || !dstLayoutC) {
      return rewriter.notifyMatchFailure(op,
                                         "missing layout info for operands");
    }

    auto newOp = cast<MmadL1Op>(rewriter.clone(*op));
    rewriter.setInsertionPoint(newOp);

    Location loc = op.getLoc();
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

    srcLayoutC = normalizeToND(rewriter.getContext(), srcLayoutC);

    // Convert result back: from target layout (zN) to source layout (dotC_ND)
    auto ndResult = rewriter.create<ConvertLayoutOp>(
        loc, cMatrix.getType(), newOp.getResult(0),
        dstLayoutC,  // from target layout (e.g., zN)
        srcLayoutC); // back to source layout (e.g., dotC_ND)

    rewriter.replaceOp(op, ndResult);

    LDBG("=== MmadL1Op conversion complete ===");
    return success();
  }
};

/// Insert ND↔fractal convert_layout around mmadmxL1 (A5/regbase only).
struct InsertConvertLayoutAroundMmadMxL1 : public OpRewritePattern<MmadMxL1Op> {
  using OpRewritePattern<MmadMxL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(MmadMxL1Op op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module || !hacc::utils::isRegBasedArch(module))
      return rewriter.notifyMatchFailure(op, "not regbase arch");

    auto opWithLayout = dyn_cast<OpWithLayoutInterface>(op.getOperation());
    if (!opWithLayout)
      return rewriter.notifyMatchFailure(
          op, "op doesn't implement OpWithLayoutInterface");

    Value aMatrix = op.getA();
    Value bMatrix = op.getB();
    Value scaleA = op.getScaleA();
    Value scaleB = op.getScaleB();
    Value cMatrix = op.getC();

    if (isAlreadyConverted(aMatrix) && isAlreadyConverted(bMatrix) &&
        isAlreadyConverted(scaleA) && isAlreadyConverted(scaleB) &&
        isAlreadyConverted(cMatrix))
      return rewriter.notifyMatchFailure(op, "already converted");

    llvm::SmallDenseMap<Value, DataLayoutAttr> currentLayoutMap =
        opWithLayout.getOperandsCurrentLayout();
    auto targetLayoutMap = opWithLayout.getOperandsTargetFractalLayout();

    DataLayoutAttr srcLayoutA = currentLayoutMap.lookup(aMatrix);
    DataLayoutAttr dstLayoutA =
        dyn_cast_or_null<DataLayoutAttr>(targetLayoutMap.a);
    DataLayoutAttr srcLayoutB = currentLayoutMap.lookup(bMatrix);
    DataLayoutAttr dstLayoutB =
        dyn_cast_or_null<DataLayoutAttr>(targetLayoutMap.b);
    DataLayoutAttr srcLayoutScaleA = currentLayoutMap.lookup(scaleA);
    DataLayoutAttr dstLayoutScaleA =
        dyn_cast_or_null<DataLayoutAttr>(targetLayoutMap.scaleA);
    DataLayoutAttr srcLayoutScaleB = currentLayoutMap.lookup(scaleB);
    DataLayoutAttr dstLayoutScaleB =
        dyn_cast_or_null<DataLayoutAttr>(targetLayoutMap.scaleB);
    DataLayoutAttr srcLayoutC = currentLayoutMap.lookup(cMatrix);
    DataLayoutAttr dstLayoutC =
        dyn_cast_or_null<DataLayoutAttr>(targetLayoutMap.c);

    if (!srcLayoutA || !dstLayoutA || !srcLayoutB || !dstLayoutB ||
        !srcLayoutScaleA || !dstLayoutScaleA || !srcLayoutScaleB ||
        !dstLayoutScaleB || !srcLayoutC || !dstLayoutC) {
      llvm::report_fatal_error(
          "InsertConvertLayout: missing layout info for mmadmxL1 operands");
    }

    auto newOp = cast<MmadMxL1Op>(rewriter.clone(*op));
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

    if (failed(convertOperand(newOp.getAMutable(), srcLayoutA, dstLayoutA,
                              "A matrix")))
      return failure();
    if (failed(convertOperand(newOp.getBMutable(), srcLayoutB, dstLayoutB,
                              "B matrix")))
      return failure();
    if (failed(convertOperand(newOp.getScaleAMutable(), srcLayoutScaleA,
                              dstLayoutScaleA, "ScaleA")))
      return failure();
    if (failed(convertOperand(newOp.getScaleBMutable(), srcLayoutScaleB,
                              dstLayoutScaleB, "ScaleB")))
      return failure();
    if (failed(convertOperand(newOp.getCMutable(), srcLayoutC, dstLayoutC,
                              "C matrix")))
      return failure();

    newOp.getResult(0).setType(newOp.getC().getType());
    rewriter.setInsertionPointAfter(newOp);

    srcLayoutC = normalizeToND(rewriter.getContext(), srcLayoutC);
    auto ndResult = rewriter.create<ConvertLayoutOp>(
        loc, cMatrix.getType(), newOp.getResult(0), dstLayoutC, srcLayoutC);

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

    // Add all transformation patterns
    patterns.add<InsertConvertLayoutAroundMmadL1>(context);
    patterns.add<InsertConvertLayoutAroundMmadMxL1>(context);
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
