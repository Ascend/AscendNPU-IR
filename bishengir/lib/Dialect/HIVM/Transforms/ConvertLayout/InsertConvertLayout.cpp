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

/// Return the FixpipeOp producing \p val when it is NZ2NZ (physically fractal).
static FixpipeOp getNz2NzFixpipe(Value val) {
  auto maybeFixpipe = traceDefOp<FixpipeOp>(val);
  if (!maybeFixpipe)
    return {};
  auto fixpipe = cast<FixpipeOp>(*maybeFixpipe);
  if (fixpipe.getDmaMode() != FixpipeDMAMode::NZ2NZ)
    return {};
  return fixpipe;
}

static bool isIgnorableFixpipeUser(Operation *user) {
  return isa<tensor::DimOp, hivm::DebugOp>(user);
}

/// True when every non-ignored user of \p val is \p allowedUser.
static bool hasOnlyAllowedUser(Value val, Operation *allowedUser) {
  for (Operation *user : val.getUsers()) {
    if (isIgnorableFixpipeUser(user))
      continue;
    if (user != allowedUser)
      return false;
  }
  return true;
}

/// Compute the rank-4 fractal tensor type that \p input would have under
/// \p dstLayout. Uses ND→Fractal shape math for the type only.
static FailureOr<RankedTensorType>
computeFractalTensorType(Value input, DataLayoutAttr dstLayout,
                         OpBuilder &builder, Location loc) {
  auto inputType = cast<ShapedType>(input.getType());
  auto inputShape = llvm::map_to_vector(
      inputType.getShape(), [&builder](auto val) -> OpFoldResult {
        return getAsIndexOpFoldResult(builder.getContext(), val);
      });
  auto ndLayout =
      DataLayoutAttr::get(builder.getContext(), hivm::DataLayout::ND);
  auto mixedShape = computeMixedNDToFractalShape(inputShape, ndLayout,
                                                 dstLayout, builder, loc);
  if (failed(mixedShape))
    return failure();
  return RankedTensorType::get(decomposeMixedValues(*mixedShape).first,
                               inputType.getElementType());
}

/// Rewrite an NZ2NZ Fixpipe so its tensor result is the fractal \p fractalType.
/// Replaces all uses of the old result and erases the old Fixpipe.
static FailureOr<Value>
retargetFixpipeToFractalType(PatternRewriter &rewriter, FixpipeOp fixpipe,
                             RankedTensorType fractalType) {
  if (!fixpipe.getResultTensor())
    return failure();

  rewriter.setInsertionPoint(fixpipe);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(
      fixpipe.getLoc(), fractalType.getShape(), fractalType.getElementType());

  auto newFixpipe = rewriter.create<FixpipeOp>(
      fixpipe.getLoc(), fractalType, fixpipe.getSrc(), emptyOp.getResult(),
      fixpipe.getDmaModeAttr(), fixpipe.getDualDstModeAttr(),
      fixpipe.getSubBlockIdxAttr(), fixpipe.getPreQuantAttr(),
      fixpipe.getPreReluAttr(), fixpipe.getChannelSplitAttr(),
      fixpipe.getC0PadEnAttr(), fixpipe.getQuantScale());
  if (fixpipe.getUnitFlagMode())
    newFixpipe.setUnitFlagModeAttr(fixpipe.getUnitFlagModeAttr());

  Value newResult = newFixpipe.getResultTensor();
  rewriter.replaceAllUsesWith(fixpipe.getResultTensor(), newResult);
  rewriter.eraseOp(fixpipe);
  LDBG("Retargeted NZ2NZ Fixpipe to fractal type " << fractalType);
  return newResult;
}

/// Materialize a rank-4 fractal view of an NZ2NZ Fixpipe result without
/// claiming an ND source. Prefers retargeting a single-use Fixpipe; otherwise
/// inserts convert_layout(Fractal→Fractal) for later Combine cleanup.
static FailureOr<Value> materializeNz2NzAsFractal(PatternRewriter &rewriter,
                                                  Location loc, Value input,
                                                  DataLayoutAttr dstLayout,
                                                  Operation *allowedUser) {
  FixpipeOp fixpipe = getNz2NzFixpipe(input);
  if (!fixpipe)
    return failure();

  if (isAlreadyConverted(input))
    return input;

  auto fractalTypeOr =
      computeFractalTensorType(input, dstLayout, rewriter, loc);
  if (failed(fractalTypeOr))
    return failure();
  RankedTensorType fractalType = *fractalTypeOr;

  // Preferred path: retarget the Fixpipe when it directly defines `input` and
  // only feeds this matmul (plus ignorable users).
  if (input.getDefiningOp() == fixpipe.getOperation() &&
      hasOnlyAllowedUser(input, allowedUser)) {
    return retargetFixpipeToFractalType(rewriter, fixpipe, fractalType);
  }

  // Fallback: Fractal→Fractal convert_layout as a type-only marker. Combine
  // folds this into a 4D Fixpipe later.
  LDBG("Creating Fractal→Fractal ConvertLayoutOp for NZ2NZ Fixpipe result");
  auto converted = rewriter.create<ConvertLayoutOp>(loc, fractalType, input,
                                                    dstLayout, dstLayout);
  return converted.getResult();
}

/// Insert convert_layout(srcLayout→dstLayout) on `input` when needed and
/// assign the converted value to `targetOperand`. Scale ND layouts are
/// preserved on the ConvertLayoutOp so downstream load_scale fusion can match
/// them.
static LogicalResult
convertAndAssignOperand(PatternRewriter &rewriter, Location loc, Value input,
                        OpOperand &targetOperand, DataLayoutAttr srcLayout,
                        DataLayoutAttr dstLayout, Operation *matmulOp) {
  if (isAlreadyConverted(input)) {
    LDBG("Input already in fractal layout, no conversion needed");
    targetOperand.assign(input);
    return success();
  }

  // NZ2NZ Fixpipe results are already physically fractal even when ranked 2D.
  // Do not insert ND→Fractal (that would later become a wrong vtranspose).
  if (getNz2NzFixpipe(input)) {
    auto fractal =
        materializeNz2NzAsFractal(rewriter, loc, input, dstLayout, matmulOp);
    if (failed(fractal)) {
      LDBG("Failed to materialize NZ2NZ Fixpipe as fractal");
      return failure();
    }
    targetOperand.assign(*fractal);
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

    // Retarget single-use NZ2NZ Fixpipe operands to rank-4 before cloning so
    // the clone sees fractal types and convertAndAssignOperand can no-op.
    auto maybeRetarget = [&](Value v, DataLayoutAttr dst) {
      if (!getNz2NzFixpipe(v) || isAlreadyConverted(v))
        return;
      if (v.getDefiningOp() == nullptr || !isa<FixpipeOp>(v.getDefiningOp()))
        return;
      if (!hasOnlyAllowedUser(v, op.getOperation()))
        return;
      auto fractalTypeOr =
          computeFractalTensorType(v, dst, rewriter, op.getLoc());
      if (failed(fractalTypeOr))
        return;
      (void)retargetFixpipeToFractalType(
          rewriter, cast<FixpipeOp>(v.getDefiningOp()), *fractalTypeOr);
    };
    maybeRetarget(aMatrix, dstLayoutA);
    maybeRetarget(bMatrix, dstLayoutB);
    maybeRetarget(cMatrix, dstLayoutC);

    // Refresh operands after possible Fixpipe retargeting.
    aMatrix = op.getMatmulA();
    bMatrix = op.getMatmulB();
    cMatrix = op.getMatmulC();

    // Retargeting moves the insertion point; restore it before cloning.
    rewriter.setInsertionPoint(op.getOperation());
    auto newOp = cast<MmadL1Op>(rewriter.clone(*op));
    rewriter.setInsertionPoint(newOp);

    Location loc = op.getLoc();
    // Convert operands to target layout if needed
    if (failed(convertAndAssignOperand(rewriter, loc, aMatrix,
                                       newOp.getAMutable(), srcLayoutA,
                                       dstLayoutA, op.getOperation())))
      return rewriter.notifyMatchFailure(op, "failed to convert A matrix");

    if (failed(convertAndAssignOperand(rewriter, loc, bMatrix,
                                       newOp.getBMutable(), srcLayoutB,
                                       dstLayoutB, op.getOperation())))
      return rewriter.notifyMatchFailure(op, "failed to convert B matrix");

    if (failed(convertAndAssignOperand(rewriter, loc, cMatrix,
                                       newOp.getCMutable(), srcLayoutC,
                                       dstLayoutC, op.getOperation())))
      return rewriter.notifyMatchFailure(op, "failed to convert C matrix");

    // Update result type and convert back
    newOp.getResult(0).setType(newOp.getC().getType());
    rewriter.setInsertionPointAfter(newOp);

    // if mmadL1Op->fixpipeOp(cbuf), no convert layout on mmadL1Op result.
    bool usedByFixpipeCbuf =
        llvm::all_of(op->getResults()[0].getUsers(), [](auto *user) {
          if (!isa<hivm::FixpipeOp>(user)) {
            return false;
          }
          hivm::FixpipeOp fixpipeOp = cast<hivm::FixpipeOp>(user);
          memref::AllocOp allocOp =
              fixpipeOp.getDst().getDefiningOp<memref::AllocOp>();
          if (!allocOp) {
            return false;
          }
          auto memorySpace = allocOp.getType().getMemorySpace();
          auto toAddrSpace =
              cast<hivm::AddressSpaceAttr>(memorySpace).getAddressSpace();
          return toAddrSpace == hivm::AddressSpace::L1;
        });

    if (usedByFixpipeCbuf) {
      rewriter.replaceOp(op, newOp);
    } else {
      srcLayoutC = normalizeToND(rewriter.getContext(), srcLayoutC);
      // Convert result back: from target layout (zN) to source layout (dotC_ND)
      auto ndResult = rewriter.create<ConvertLayoutOp>(
          loc, cMatrix.getType(), newOp.getResult(0),
          dstLayoutC,  // from target layout (e.g., zN)
          srcLayoutC); // back to source layout (e.g., dotC_ND)
      rewriter.replaceOp(op, ndResult);
    }

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

    // Retarget single-use NZ2NZ Fixpipe operands to rank-4 before cloning so
    // the clone sees fractal types and convertAndAssignOperand can no-op.
    auto maybeRetarget = [&](Value v, DataLayoutAttr dst) {
      if (!getNz2NzFixpipe(v) || isAlreadyConverted(v))
        return;
      if (v.getDefiningOp() == nullptr || !isa<FixpipeOp>(v.getDefiningOp()))
        return;
      if (!hasOnlyAllowedUser(v, op.getOperation()))
        return;
      auto fractalTypeOr =
          computeFractalTensorType(v, dst, rewriter, op.getLoc());
      if (failed(fractalTypeOr))
        return;
      (void)retargetFixpipeToFractalType(
          rewriter, cast<FixpipeOp>(v.getDefiningOp()), *fractalTypeOr);
    };
    maybeRetarget(aMatrix, dstLayoutA);
    maybeRetarget(bMatrix, dstLayoutB);
    maybeRetarget(cMatrix, dstLayoutC);
    maybeRetarget(scaleA, dstLayoutScaleA);
    maybeRetarget(scaleB, dstLayoutScaleB);

    // Refresh operands after possible Fixpipe retargeting.
    aMatrix = op.getMatmulA();
    bMatrix = op.getMatmulB();
    cMatrix = op.getMatmulC();
    scaleA = op.getScaleA();
    scaleB = op.getScaleB();

    // Retargeting moves the insertion point; restore it before cloning.
    rewriter.setInsertionPoint(op.getOperation());
    auto newOp = cast<MmadMxL1Op>(rewriter.clone(*op));
    rewriter.setInsertionPoint(newOp);
    Location loc = op.getLoc();

    auto convertOperand = [&](OpOperand &operand, DataLayoutAttr src,
                              DataLayoutAttr dst,
                              StringRef name) -> LogicalResult {
      if (failed(convertAndAssignOperand(rewriter, loc, operand.get(), operand,
                                         src, dst, op.getOperation())))
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
        loc, cMatrix.getType(), newOp->getResult(0), dstLayoutC, srcLayoutC);
    rewriter.replaceOp(op, ndResult);

    LDBG("=== MmadMxL1Op conversion complete ===");
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
