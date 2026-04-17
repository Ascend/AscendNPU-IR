//===- NormalizeConvOps.cpp - normalize hivm conv ops ---------------------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Casting.h"

#include <cstdint>
#include <memory>

namespace mlir {
#define GEN_PASS_DEF_NORMALIZECONVOPS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-normalize-convops"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
/// This pass normalizes Conv1D/Conv2D operations by adapting user-facing
/// tensors to the hardware-required layout and precision, and then
/// restoring results back to the user-visible form.
///
/// The transformation is organized into multiple stages, each handling
/// a specific responsibility:
///
///   1. Input / weight normalization
///        - Convert tensors from user layout to hardware layout
///        - Insert padding if channel dimensions do not meet hardware
///          alignment requirements
///
///   2. Convolution computation
///        - Performed by hardware
///        - Hardware may internally use higher precision than input type
///
///   3. Output normalization
///        - Convert results from hardware layout back to user layout
///        - Remove any padding introduced during computation
///
///   4. Bias handling (if present)
///        - Insert bias add at the correct stage depending on dtype
///
///   5. Cast insertion (if needed)
///        - Convert from hardware computation precision back to
///          user-expected element type
///
/// The exact ordering depends on dtype and bias:
///
///   - fp32 without bias:
///       normalize input/weight
///       → conv (fp32)
///       → normalize output
///
///   - fp32 with bias:
///       normalize input/weight
///       → conv (fp32)
///       → normalize output
///       → add bias
///
///   - fp16 without bias:
///       normalize input/weight
///       → conv (fp16 → fp32 hardware uses higher precision internally)
///       → cast back to fp16
///       → normalize output
///
///   - fp16 with bias:
///       normalize input/weight
///       → conv (fp16 → fp32 hardware uses higher precision internally)
///       → cast back to fp16
///       → normalize output
///       → add bias (in fp16)
///
///   - bf16 without bias:
///       normalize input/weight
///       → conv (bf16 → fp32 hardware uses higher precision internally)
///       → cast back to bf16
///       → normalize output
///
///   - bf16 with bias:
///       normalize input/weight
///       → conv (bf16 → fp32 hardware uses higher precision internally)
///       → normalize output
///       → add bias (in fp32)
///       → cast back to bf16
///
/// Key design points:
///   - "Normalization" bridges user layout and hardware layout
///   - Padding is introduced only for hardware constraints and removed later
///   - Cast is required because hardware computation precision may differ
///     from user-visible tensor types
///   - Bias placement depends on numerical and hardware considerations
///
struct NormalizeConvOpsPass
    : public impl::NormalizeConvOpsBase<NormalizeConvOpsPass> {
  using Base::Base;
  void runOnOperation() override;
};

/// Helper trait to get base dimensions from Conv Op type
template <typename ConvOpType> struct ConvBaseDims;

template <> struct ConvBaseDims<hivm::Conv1DL1Op> {
  // the base dimension for Conv1DL1Op is 2 (CW)
  static constexpr int64_t dim = 2;
};

template <> struct ConvBaseDims<hivm::Conv2DL1Op> {
  // the base dimension for Conv2DL1Op is 3 (CHW)
  static constexpr int64_t dim = 3;
};

static constexpr llvm::StringLiteral outputAlreadyNormalized =
    "outputAlreadyNormalized";

inline RoundModeAttr getRoundAttr(mlir::OpBuilder &b, Type srcType,
                                  Type dstType) {
  return hivm::RoundModeAttr::get(
      b.getContext(),
      mlir::utils::selectRoundMode<hivm::RoundMode>(srcType, dstType));
}

LogicalResult getElementsFor32ByteAlignment(Type elementType, int64_t &C0) {
  if (!elementType || !elementType.isIntOrFloat()) {
    return failure();
  }
  const int64_t bytesize = 8;
  auto elementSize = CEIL_DIV(elementType.getIntOrFloatBitWidth(), bytesize);
  if (elementSize == 0) {
    return failure();
  }

  constexpr int64_t k32ByteAlignment = 32;
  C0 = CEIL_DIV(k32ByteAlignment, elementSize);

  return success();
}

template <typename ConvOpType>
LogicalResult expandToBatch(ConvOpType op, PatternRewriter &rewriter) {
  auto input = op.getInput();
  auto inputType = cast<ShapedType>(input.getType());
  auto elementType = inputType.getElementType();
  const int64_t batch = 1;

  static constexpr int64_t baseDims = ConvBaseDims<ConvOpType>::dim;

  SmallVector<int64_t> originalDims;
  for (int64_t i = 0; i <= baseDims - 1; i++) {
    originalDims.push_back(inputType.getDimSize(i));
  }

  // 1D: expand [iC, iW] -> [N, iC, iW]
  // 2D: expand [iC, iH, iW] -> [N, iC, iH, iW]
  SmallVector<int64_t> newShapeInput;
  newShapeInput.push_back(batch);
  newShapeInput.insert(newShapeInput.end(), originalDims.begin(),
                       originalDims.end());

  // reassociation: {{0, 1}, {2}, {3}, ...}
  SmallVector<SmallVector<int64_t, 2>> reassoc;

  reassoc.push_back({0, 1});

  for (int64_t i = 2; i <= baseDims; i++) {
    reassoc.push_back({static_cast<int64_t>(i)});
  }

  auto newType = RankedTensorType::get(newShapeInput, elementType);
  auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
      op->getLoc(), newType, input, reassoc);

  op->replaceUsesOfWith(input, expandShapeOp->getResults()[0]);

  return success();
}

template <typename ConvOpType>
LogicalResult padForInput(ConvOpType op, PatternRewriter &rewriter,
                          int64_t groups, int64_t C) {
  auto input = op.getInput();
  auto inputType = cast<ShapedType>(input.getType());
  auto elementType = inputType.getElementType();

  static constexpr int64_t baseDims = ConvBaseDims<ConvOpType>::dim;

  int64_t batch = inputType.getDimSize(0);
  int64_t iC = inputType.getDimSize(1);

  SmallVector<int64_t> spatialSizes;
  for (int i = 0; i < baseDims - 1; i++) {
    spatialSizes.push_back(inputType.getDimSize(2 + i));
  }

  // Step 1: Expand [N, C, spatial...] -> [N, G, C/G, spatial...]
  SmallVector<int64_t> expandedShape;
  SmallVector<SmallVector<int64_t, 2>> expandReassoc;

  expandedShape.push_back(batch);
  expandedShape.push_back(groups);
  expandedShape.push_back(iC / groups);
  expandedShape.insert(expandedShape.end(), spatialSizes.begin(),
                       spatialSizes.end());

  // Reassociation: {0}, {1, 2}, {3}, {4}, ...
  expandReassoc.push_back({0});    // N
  expandReassoc.push_back({1, 2}); // G, C/G
  for (int i = 0; i < baseDims - 1; i++) {
    expandReassoc.push_back({static_cast<int64_t>(3 + i)}); // spatial dims
  }

  auto expandedType = RankedTensorType::get(expandedShape, elementType);
  auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
      op->getLoc(), expandedType, input, expandReassoc);

  // Step 2: Create padding tensor [N, G, (C-iC)/G, spatial...]
  SmallVector<int64_t> padShape;
  padShape.push_back(batch);
  padShape.push_back(groups);
  padShape.push_back((C - iC) / groups);
  padShape.insert(padShape.end(), spatialSizes.begin(), spatialSizes.end());

  auto emptyOp =
      rewriter.create<tensor::EmptyOp>(op->getLoc(), padShape, elementType);
  auto zero = rewriter.create<arith::ConstantOp>(
      op->getLoc(), rewriter.getZeroAttr(elementType));
  auto pad = rewriter.create<hivm::VBrcOp>(
      op->getLoc(), emptyOp->getResultTypes(), zero, emptyOp->getResults()[0],
      rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));

  // Step 3: Concat along channel dimension (dimension index 2: C/G)
  SmallVector<int64_t> concatShape;
  concatShape.push_back(batch);
  concatShape.push_back(groups);
  concatShape.push_back(C / groups);
  concatShape.insert(concatShape.end(), spatialSizes.begin(),
                     spatialSizes.end());

  auto dst =
      rewriter.create<tensor::EmptyOp>(op->getLoc(), concatShape, elementType);
  auto concatOp = rewriter.create<hivm::VConcatOp>(
      op->getLoc(), dst->getResultTypes(),
      rewriter.getI64IntegerAttr(2), // concat along C/G dimension
      ValueRange{expandShapeOp->getResults()[0], pad->getResults()[0]},
      dst->getResults()[0]);

  // Step 4: Collapse back to [N, C, spatial...]
  SmallVector<int64_t> collapsedShape;
  SmallVector<SmallVector<int64_t, 2>> collapseReassoc;

  collapsedShape.push_back(batch);
  collapsedShape.push_back(C);
  collapsedShape.insert(collapsedShape.end(), spatialSizes.begin(),
                        spatialSizes.end());

  // Reassociation: {0}, {1, 2}, {3}, {4}, ...
  collapseReassoc.push_back({0});    // N
  collapseReassoc.push_back({1, 2}); // G, C/G -> C
  for (int i = 0; i < baseDims - 1; i++) {
    collapseReassoc.push_back({static_cast<int64_t>(3 + i)}); // spatial dims
  }

  auto collapsedType = RankedTensorType::get(collapsedShape, elementType);
  auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
      op->getLoc(), collapsedType, concatOp->getResults()[0], collapseReassoc);

  op->replaceUsesOfWith(input, collapseOp.getResult());
  return success();
}

template <typename ConvOpType>
LogicalResult
insertPadExpandTransToFormatInput(ConvOpType op, PatternRewriter &rewriter,
                                  int64_t C0, int64_t C, int64_t groups) {
  auto input = op.getInput();
  auto inputType = cast<ShapedType>(input.getType());
  Type elementType = inputType.getElementType();

  static constexpr int64_t baseDims = ConvBaseDims<ConvOpType>::dim;

  auto batch = inputType.getDimSize(0);
  int64_t iC = inputType.getDimSize(1);

  SmallVector<int64_t> spatialSizes;
  for (int i = 0; i < baseDims - 1; i++) {
    spatialSizes.push_back(inputType.getDimSize(2 + i));
  }

  // Step 1: Padding if needed
  if (iC != C) {
    if (failed(padForInput<ConvOpType>(op, rewriter, groups, C))) {
      return failure();
    }
  }

  input = op.getInput();

  // Step 2: For 2D, collapse spatial dimensions [N, C, iH, iW] -> [N, C, iHW]
  Value currentInput = input;
  int64_t iHW = 1;
  if (baseDims == 3) {
    for (auto size : spatialSizes) {
      iHW *= size;
    }

    SmallVector<int64_t> collapsedShape{batch, C, iHW};
    SmallVector<SmallVector<int64_t, 2>> collapseReassoc = {{0}, {1}, {2, 3}};

    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        op->getLoc(), RankedTensorType::get(collapsedShape, elementType),
        currentInput, collapseReassoc);
    currentInput = collapseOp.getResult();
  } else {
    iHW = spatialSizes[0];
  }

  // Step 3: Expand [N, C, iHW] -> [N, C1, C0, iHW]
  int64_t C1 = C / C0;
  SmallVector<int64_t> newShapeInput{batch, C1, C0, iHW};
  SmallVector<SmallVector<int64_t, 2>> reassoc = {{0}, {1, 2}, {3}};
  auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
      op->getLoc(), RankedTensorType::get(newShapeInput, elementType),
      currentInput, reassoc);

  // Step 4: Transpose [N, C1, C0, iHW] -> [N, C1, iHW, C0]
  // (0, 1, 3, 2)
  SmallVector<int64_t, 4> newPerm{0, 1, 3, 2};
  auto dst = rewriter.create<tensor::EmptyOp>(
      op->getLoc(), ArrayRef<int64_t>{batch, C1, iHW, C0}, elementType);
  auto vtransposeOp = rewriter.create<hivm::VTransposeOp>(
      op->getLoc(), TypeRange{dst.getResult().getType()},
      expandShapeOp.getResult(), dst.getResult(),
      rewriter.getDenseI64ArrayAttr(newPerm));

  // Step 5: Expand spatial dimensions back
  SmallVector<int64_t> finalShape;
  if (baseDims == 2) {
    // For 1D: [N, C1, iHW, C0] -> [N, C1, 1, iW, C0]
    finalShape = {batch, C1, 1, iHW, C0};
  } else if (baseDims == 3) {
    // For 2D: [N, C1, iHW, C0] -> [N, C1, iH, iW, C0]
    finalShape.push_back(batch);
    finalShape.push_back(C1);
    finalShape.insert(finalShape.end(), spatialSizes.begin(),
                      spatialSizes.end());
    finalShape.push_back(C0);
  }
  SmallVector<SmallVector<int64_t, 2>> finalReassoc = {{0}, {1}, {2, 3}, {4}};
  auto expandShapeOp1 = rewriter.create<tensor::ExpandShapeOp>(
      op->getLoc(), RankedTensorType::get(finalShape, elementType),
      vtransposeOp->getResults()[0], finalReassoc);

  op->replaceUsesOfWith(input, expandShapeOp1->getResult(0));

  return success();
}

template <typename ConvOpType>
LogicalResult padForWeight(ConvOpType op, PatternRewriter &rewriter,
                           int64_t groups, int64_t C) {
  auto weight = op.getWeight();
  auto weightType = cast<ShapedType>(weight.getType());
  auto elementType = weightType.getElementType();

  static constexpr int64_t baseDims = ConvBaseDims<ConvOpType>::dim;

  // Weight shape for 1D: [oC, iC/G, kW]
  // Weight shape for 2D: [oC, iC/G, kH, kW]
  int64_t oC = weightType.getDimSize(0);
  int64_t channelsPerGroup = weightType.getDimSize(1);

  SmallVector<int64_t> kernelSizes;
  for (int i = 0; i < baseDims - 1; i++) {
    kernelSizes.push_back(weightType.getDimSize(2 + i));
  }

  int64_t newChannelsPerGroup = C / groups;

  // Step 1: Create padding tensor [oC, (C/G - iC/G), kernel_spatial...]
  SmallVector<int64_t> padShape;
  padShape.push_back(oC);
  padShape.push_back(newChannelsPerGroup - channelsPerGroup);
  padShape.insert(padShape.end(), kernelSizes.begin(), kernelSizes.end());

  auto emptyOp =
      rewriter.create<tensor::EmptyOp>(op->getLoc(), padShape, elementType);

  auto zero = rewriter.create<arith::ConstantOp>(
      op->getLoc(), rewriter.getZeroAttr(elementType));

  auto pad = rewriter.create<hivm::VBrcOp>(
      op->getLoc(), emptyOp->getResultTypes(), zero, emptyOp->getResults()[0],
      rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));

  // Step 2: Concat along input channel dimension (dimension 1)
  // Result shape: [oC, C/G, kernel_spatial...]
  SmallVector<int64_t> concatShape;
  concatShape.push_back(oC);
  concatShape.push_back(newChannelsPerGroup);
  concatShape.insert(concatShape.end(), kernelSizes.begin(), kernelSizes.end());

  auto dst =
      rewriter.create<tensor::EmptyOp>(op->getLoc(), concatShape, elementType);

  auto concat = rewriter.create<hivm::VConcatOp>(
      op->getLoc(), dst->getResultTypes(), rewriter.getI64IntegerAttr(1),
      ValueRange{weight, pad->getResults()[0]}, dst->getResults()[0]);

  op->replaceUsesOfWith(weight, concat->getResults()[0]);
  return success();
}

template <typename ConvOpType>
LogicalResult
insertPadExpandTransToFormatWeight(ConvOpType op, PatternRewriter &rewriter,
                                   int64_t C0, int64_t groups, int64_t C) {
  auto weight = op.getWeight();
  auto weightType = cast<ShapedType>(weight.getType());
  auto elementType = weightType.getElementType();

  static constexpr int64_t baseDims = ConvBaseDims<ConvOpType>::dim;

  auto oC = weightType.getDimSize(0);
  auto channelsPerGroup = weightType.getDimSize(1);

  SmallVector<int64_t> kernelSizes;
  for (int i = 0; i < baseDims - 1; i++) {
    kernelSizes.push_back(weightType.getDimSize(2 + i));
  }

  // Step 1: Padding if needed
  if (channelsPerGroup * groups != C) {
    if (failed(padForWeight<ConvOpType>(op, rewriter, groups, C)))
      return failure();
  }

  weight = op.getWeight();

  // Step 2: For 2D, collapse spatial dimensions [oC, C/groups, wH, wW] -> [oC,
  // C/groups, wHW]
  Value currentWeight = weight;
  int64_t wHW = 1;
  if (baseDims == 3) {
    for (auto size : kernelSizes) {
      wHW *= size;
    }

    SmallVector<int64_t> collapsedShape{oC, C / groups, wHW};
    SmallVector<SmallVector<int64_t, 2>> collapseReassoc = {{0}, {1}, {2, 3}};

    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        op->getLoc(), RankedTensorType::get(collapsedShape, elementType),
        currentWeight, collapseReassoc);
    currentWeight = collapseOp.getResult();
  } else {
    wHW = kernelSizes[0];
  }

  // Step 3: Expand [oC, C/groups, wHW] -> [oC, C1/groups, C0, wHW]
  int64_t newChannelsPerGroup = C / groups / C0;
  SmallVector<int64_t> newShape{oC, newChannelsPerGroup, C0, wHW};
  SmallVector<SmallVector<int64_t, 2>> reassoc = {{0}, {1, 2}, {3}};
  auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
      op->getLoc(), RankedTensorType::get(newShape, elementType), currentWeight,
      reassoc);

  // Step 4: Transpose [oC, C1/groups, C0, wHW] -> [C1/groups, oC, C0, wHW]
  SmallVector<int64_t, 4> newPerm{1, 0, 2, 3};
  auto dst = rewriter.create<tensor::EmptyOp>(
      op->getLoc(), ArrayRef<int64_t>{newChannelsPerGroup, oC, C0, wHW},
      elementType);
  auto vtransposeOp = rewriter.create<hivm::VTransposeOp>(
      op->getLoc(), TypeRange{dst.getResult().getType()},
      expandShapeOp.getResult(), dst.getResult(),
      rewriter.getDenseI64ArrayAttr(newPerm));

  // Step 5: Transpose [C1/groups, oC, C0, wHW] -> [C1/groups, oC, wHW, C0]
  SmallVector<int64_t, 4> newPerm1{0, 1, 3, 2};
  auto dst1 = rewriter.create<tensor::EmptyOp>(
      op->getLoc(), ArrayRef<int64_t>{newChannelsPerGroup, oC, wHW, C0},
      elementType);
  auto vtransposeOp1 = rewriter.create<hivm::VTransposeOp>(
      op->getLoc(), TypeRange{dst1.getResult().getType()},
      vtransposeOp.getResult()[0], dst1.getResult(),
      rewriter.getDenseI64ArrayAttr(newPerm1));

  // Step 6: Transpose [C1/groups, oC, wHW, C0] -> [C1/groups, wHW, oC, C0]
  SmallVector<int64_t, 4> newPerm2{0, 2, 1, 3};
  auto dst2 = rewriter.create<tensor::EmptyOp>(
      op->getLoc(), ArrayRef<int64_t>{newChannelsPerGroup, wHW, oC, C0},
      elementType);
  auto vtransposeOp2 = rewriter.create<hivm::VTransposeOp>(
      op->getLoc(), TypeRange{dst2.getResult().getType()},
      vtransposeOp1.getResult()[0], dst2.getResult(),
      rewriter.getDenseI64ArrayAttr(newPerm2));

  // Step 7: Expand spatial dimensions back
  SmallVector<int64_t> finalShape;
  if (baseDims == 2) {
    // For 1D: [C1/groups, wHW, oC, C0] -> [C1/groups, 1, wW, oC, C0]
    finalShape = {newChannelsPerGroup, 1, wHW, oC, C0};
  } else if (baseDims == 3) {
    // For 2D: [C1/groups, wHW, oC, C0] -> [C1/groups, wH, wW, oC, C0]
    finalShape.push_back(newChannelsPerGroup);
    finalShape.insert(finalShape.end(), kernelSizes.begin(), kernelSizes.end());
    finalShape.push_back(oC);
    finalShape.push_back(C0);
  }
  SmallVector<SmallVector<int64_t, 2>> finalReassoc = {{0}, {1, 2}, {3}, {4}};
  auto expandShapeOp1 = rewriter.create<tensor::ExpandShapeOp>(
      op->getLoc(), RankedTensorType::get(finalShape, elementType),
      vtransposeOp2.getResult()[0], finalReassoc);

  op->replaceUsesOfWith(weight, expandShapeOp1->getResult(0));
  return success();
}

/// This pattern normalizes the input and weight layout of Conv1D/Conv2D
/// operations by transforming them into a fractal format for efficient
/// hardware execution.
///
/// Original logical layouts:
///   Conv1D:
///     - Input:  [B, iC, iW] or [iC, iW] (without batch)
///     - Weight: [oC, iC, wW] or [oC, iC/groups, wW] (grouped)
///   Conv2D:
///     - Input:  [B, iC, iH, iW] or [iC, iH, iW] (without batch)
///     - Weight: [oC, iC, wH, wW] or [oC, iC/groups, wH, wW] (grouped)
///
/// Target fractal format (32-byte aligned):
///   - Input:  [B, C1, iH, iW, C0] where C0 = 32 / sizeof(element)
///   - Weight: [C1/groups, wH, wW, oC, C0]
///
/// Rewrite steps:
///   1. Calculate alignment factor C0 based on element type (32-byte alignment)
///   2. Add batch dimension if missing (rank == baseDims)
///   3. For 2D, collapse spatial dimensions:
///        Input:  [B, iC, iH, iW] -> [B, iC, iHW]
///        Weight: [oC, iC/groups, wH, wW] -> [oC, iC/groups, wHW]
///   4. Pad input channels to multiple of C0 * groups:
///        Expand [B, iC, ...] -> [B, groups, iC/groups, ...]
///        Pad to [B, groups, C/groups, ...]
///        Collapse back to [B, C, ...]
///   5. Pad weight input channels similarly:
///        Pad [oC, iC/groups, ...] -> [oC, C/groups, ...]
///   6. Expand channel dimension:
///        Input:  [B, C, iHW] -> [B, C1, C0, iHW] where C1 = C / C0
///        Weight: [oC, C/groups, wHW] -> [oC, C1/groups, C0, wHW]
///   7. Transpose input:
///        [B, C1, C0, iHW] -> [B, C1, iHW, C0]
///   8. Transpose weight (three steps):
///        [oC, C1/groups, C0, wHW] -> [C1/groups, oC, C0, wHW]
///        [C1/groups, oC, C0, wHW] -> [C1/groups, oC, wHW, C0]
///        [C1/groups, oC, wHW, C0] -> [C1/groups, wHW, oC, C0]
///   9. Expand spatial dimensions back:
///        For Conv1D: [B, C1, iHW, C0] -> [B, C1, 1, iW, C0]
///        For Conv2D: [B, C1, iHW, C0] -> [B, C1, iH, iW, C0]
///        For weight: [C1/groups, wHW, oC, C0] -> [C1/groups, wH, wW, oC, C0]
///
/// Semantics:
///   This pattern only changes the physical memory layout for hardware
///   efficiency. The convolution's numerical behavior remains unchanged.
template <typename ConvOpType>
struct NormalizeConvInputAndWeightPattern
    : public OpRewritePattern<ConvOpType> {
public:
  using OpRewritePattern<ConvOpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvOpType op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 3) {
      return op.emitError() << "Operation " << op->getName()
                            << " requires at least 3 operands, but got "
                            << op->getNumOperands();
    }
    auto input = op.getInput();
    auto weight = op.getWeight();

    if (!isa<ShapedType>(input.getType())) {
      return op.emitError("Expected ShapedType for input, but got: ")
             << input.getType();
    }

    if (!isa<ShapedType>(weight.getType())) {
      return op.emitError("Expected ShapedType for weight, but got: ")
             << weight.getType();
    }

    auto inputType = cast<ShapedType>(input.getType());
    if (!inputType.hasStaticShape() || inputType.getRank() == 5) {
      return failure();
    }

    auto weightType = cast<ShapedType>(weight.getType());
    if (!weightType.hasStaticShape() || weightType.getRank() == 5) {
      return failure();
    }

    if (inputType.getElementType() != weightType.getElementType()) {
      return op.emitError("Element type mismatch: input has ")
             << inputType.getElementType() << " but weight has "
             << weightType.getElementType();
    }

    int64_t C0 = 0;
    if (failed(getElementsFor32ByteAlignment(inputType.getElementType(), C0))) {
      return op.emitError(
          "Failed to calculate 32-byte alignment for element type");
    }

    if (!op->hasAttr("groups")) {
      return op.emitError("Expected Attr named groups for Conv1D op");
    }

    if (!isa<IntegerAttr>(op->getAttr("groups"))) {
      return op.emitError("Expected IntegerType for Attr 'groups'");
    }

    // set insert loc before the define of out operand
    auto *outDefOp = op.getInit().getDefiningOp();
    if (outDefOp) {
      rewriter.setInsertionPoint(outDefOp);
    }

    static constexpr int64_t baseDims = ConvBaseDims<ConvOpType>::dim;

    if (inputType.getRank() == baseDims) {
      if (failed(expandToBatch<ConvOpType>(op, rewriter))) {
        return rewriter.notifyMatchFailure(op, "Failed to expand to batch");
      }
    }

    int64_t groups = op.getGroups();
    input = op.getInput();
    inputType = cast<ShapedType>(input.getType());
    int64_t iC = inputType.getDimSize(1);
    int64_t C = CEIL_DIV(iC, C0 * groups) * (C0 * groups);

    LLVM_DEBUG(llvm::dbgs()
               << "start insert [pad expand trans] op for conv1D op"
               << "\n");
    if (failed(insertPadExpandTransToFormatInput<ConvOpType>(op, rewriter, C0,
                                                             C, groups))) {
      return rewriter.notifyMatchFailure(
          op, "Failed to insert pad/expand/trans for input");
    }
    if (failed(insertPadExpandTransToFormatWeight<ConvOpType>(op, rewriter, C0,
                                                              groups, C))) {
      return rewriter.notifyMatchFailure(
          op, "Failed to insert pad/expand/trans for weight");
    }

    LLVM_DEBUG(llvm::dbgs() << "insert op for conv1D op successfully!"
                            << "\n");
    return success();
  }
};

/// This pattern transforms bf16/fp16 output of ConvOps to fp32 and then cast
/// back
template <typename TargetType, typename ConvOpType>
struct NormalizeConvResultTypePattern : public OpRewritePattern<ConvOpType> {
public:
  using OpRewritePattern<ConvOpType>::OpRewritePattern;
  ~NormalizeConvResultTypePattern() override = default;

  LogicalResult matchAndRewrite(ConvOpType op,
                                PatternRewriter &rewriter) const override {

    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    auto resultType =
        dyn_cast<RankedTensorType>(op.getResultTensors()[0].getType());
    if (!resultType) {
      return failure();
    }

    auto elemType = resultType.getElementType();
    if (!isa<TargetType>(elemType)) {
      return failure();
    }

    Location loc = op.getLoc();
    auto fp32Ty = rewriter.getF32Type();

    Value bias = op.getBias();
    Value newBias = bias;

    if (bias && elemType.isBF16()) {
      auto biasType = mlir::cast<RankedTensorType>(bias.getType());
      auto biasShape = biasType.getShape();
      auto biasElemType = biasType.getElementType();

      auto biasCastType = RankedTensorType::get(biasShape, fp32Ty);
      auto biasCastInit =
          rewriter.create<tensor::EmptyOp>(loc, biasShape, fp32Ty);
      auto biasRoundAttr = getRoundAttr(rewriter, biasElemType, fp32Ty);

      newBias = rewriter
                    .create<hivm::VCastOp>(
                        loc, TypeRange{biasCastType}, ValueRange{bias},
                        ValueRange{biasCastInit.getResult()}, Value(),
                        biasRoundAttr, hivm::TypeFnAttr{})
                    ->getResult(0);
    }

    SmallVector<int64_t> shape(resultType.getShape().begin(),
                               resultType.getShape().end());

    auto fp32ResultType = RankedTensorType::get(shape, fp32Ty);
    auto init = rewriter.create<tensor::EmptyOp>(loc, shape, fp32Ty);

    auto input = op.getInput();
    auto weight = op.getWeight();
    auto initCondition = op.getInitCondition();
    auto padding = op.getPaddingAttr();
    auto groups = op.getGroupsAttr();

    auto newConv =
        rewriter.create<ConvOpType>(loc, TypeRange{fp32ResultType},
                                    input,         // input
                                    weight,        // weight
                                    newBias,       // bias
                                    init,          // init
                                    initCondition, // init_condition
                                    ValueRange{},  // sync_related_args
                                    padding,       // padding
                                    groups         // groups
        );

    auto castInit = rewriter.create<tensor::EmptyOp>(loc, shape, elemType);
    auto roundAttr = getRoundAttr(rewriter, fp32Ty, elemType);

    auto castResult =
        rewriter
            .create<hivm::VCastOp>(loc, TypeRange{castInit.getType()},
                                   ValueRange{newConv.getResultTensors()[0]},
                                   ValueRange{castInit.getResult()},
                                   /*temp_buffer=*/Value(), roundAttr,
                                   hivm::TypeFnAttr{})
            ->getResult(0);

    rewriter.replaceOp(op, castResult);

    return success();
  }
};

/// This pattern decomposes ConvOps with bias into ConvOps(no bias) + vadd.
/// Vadd is inserted to different position (after ConvOps + vcast or directly
/// after ConvOps) according to different dtype
template <typename TargetType, typename ConvOpType>
struct DecomposeConv1dWithBiasPattern : public OpRewritePattern<ConvOpType> {
public:
  using OpRewritePattern<ConvOpType>::OpRewritePattern;
  ~DecomposeConv1dWithBiasPattern() override = default;

  LogicalResult matchAndRewrite(ConvOpType op,
                                PatternRewriter &rewriter) const override {

    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    Value bias = op.getBias();
    if (!bias) {
      return failure();
    }

    auto biasType = dyn_cast<RankedTensorType>(bias.getType());
    if (!biasType) {
      return failure();
    }

    auto biasElemType = biasType.getElementType();
    if (!isa<TargetType>(biasElemType)) {
      return failure();
    }

    auto resultType =
        dyn_cast<RankedTensorType>(op.getResultTensors()[0].getType());
    if (!resultType) {
      return failure();
    }

    static constexpr int64_t baseDims = ConvBaseDims<ConvOpType>::dim;

    int64_t rank = resultType.getRank();
    // For 1D conv: rank should be 2 (CW) or 3 (NCW)
    // For 2D conv: rank should be 3 (CHW) or 4 (NCHW)
    int64_t expectedMinRank = baseDims;
    int64_t expectedMaxRank = baseDims + 1;
    if (rank != expectedMinRank && rank != expectedMaxRank) {
      return failure();
    }

    Location loc = op.getLoc();

    auto input = op.getInput();
    auto weight = op.getWeight();
    auto initCondition = op.getInitCondition();
    auto padding = op.getPaddingAttr();
    auto groups = op.getGroupsAttr();

    auto elemType = resultType.getElementType();
    SmallVector<int64_t> shape(resultType.getShape().begin(),
                               resultType.getShape().end());

    auto convNoBiasInit =
        rewriter.create<tensor::EmptyOp>(loc, shape, elemType);

    auto newConv =
        rewriter.create<ConvOpType>(loc, TypeRange{resultType},
                                    input,              // input
                                    weight,             // weight
                                    /* bias */ Value(), // remove bias
                                    convNoBiasInit,     // init
                                    initCondition,      // init_condition
                                    ValueRange{},       // sync_related_args
                                    padding,            // padding
                                    groups              // groups
        );

    Value convResult = newConv.getResultTensors()[0];

    // Output channel dimension position depends on rank
    // For rank = baseDims: format is oCxSpatial (no batch)
    // For rank = baseDims + 1: format is NxoCxSpatial (with batch)
    int64_t outputChannelDim = (rank == baseDims) ? 0 : 1;
    int64_t oC = resultType.getDimSize(outputChannelDim);

    SmallVector<int64_t> expandedShape;
    SmallVector<SmallVector<int64_t, 2>> reassoc;

    if (rank == baseDims) {
      // No batch dimension: [oC, spatial_dims...]
      // For 1D: expandedShape = {oC, 1}, reassoc = {{0, 1}}
      // For 2D: expandedShape = {oC, 1, 1}, reassoc = {{0, 1, 2}}
      expandedShape.push_back(oC);
      for (int64_t i = 0; i < baseDims - 1; i++) {
        expandedShape.push_back(1);
      }
      SmallVector<int64_t, 2> reassocIndices;
      for (int64_t i = 0; i <= baseDims - 1; i++) {
        reassocIndices.push_back(i);
      }
      reassoc.push_back(reassocIndices);
    } else {
      // With batch dimension: [N, oC, spatial_dims...]
      // For 1D: expandedShape = {1, oC, 1}, reassoc = {{0, 1, 2}}
      // For 2D: expandedShape = {1, oC, 1, 1}, reassoc = {{0, 1, 2, 3}}
      expandedShape.push_back(1);
      expandedShape.push_back(oC);
      for (int64_t i = 0; i < baseDims - 1; i++) {
        expandedShape.push_back(1);
      }
      SmallVector<int64_t, 2> reassocIndices;
      for (int64_t i = 0; i <= baseDims; i++) {
        reassocIndices.push_back(i);
      }
      reassoc.push_back(reassocIndices);
    }

    auto expandedType = RankedTensorType::get(expandedShape, biasElemType);
    auto expandedBias = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandedType, bias, reassoc);

    auto convResultElemType =
        dyn_cast<RankedTensorType>(convResult.getType()).getElementType();

    if (convResultElemType != biasElemType) {
      return rewriter.notifyMatchFailure(
          op, "bias type mismatch with convResult type");
    }

    SmallVector<int64_t> broadcastDims;
    if (rank == baseDims) {
      // No batch: broadcast over all spatial dimensions
      // For 1D: broadcastDims = {1}
      // For 2D: broadcastDims = {1, 2}
      for (int64_t i = 1; i <= baseDims - 1; i++) {
        broadcastDims.push_back(i); // spatial dimensions
      }
    } else {
      // With batch: broadcast over batch and spatial dimensions
      // For 1D: broadcastDims = {0, 2}
      // For 2D: broadcastDims = {0, 2, 3}
      broadcastDims.push_back(0); // batch dimension
      for (int64_t i = 2; i <= baseDims; i++) {
        broadcastDims.push_back(i); // spatial dimensions
      }
    }

    auto broadcastAttr = rewriter.getDenseI64ArrayAttr(broadcastDims);

    auto vaddInit =
        rewriter.create<tensor::EmptyOp>(loc, shape, convResultElemType);

    auto vadd = rewriter.create<hivm::VAddOp>(
        loc, TypeRange{vaddInit.getType()},
        ValueRange{convResult, expandedBias}, ValueRange{vaddInit}, Value(),
        nullptr, broadcastAttr);

    rewriter.replaceOp(op, vadd->getResult(0));
    return success();
  }
};

/// This pattern normalizes the output layout of hivm::Conv1DL1Op by fusing
/// batch and group dimensions and aligning width and per-group channels,
/// then restoring the result back to the user-visible layout.
///
/// Motivation:
///   The original Conv1DL1 produces output in logical layout:
///     - without batch: [oC, oW]
///     - with batch:    [B, oC, oW]
///   The original Conv2DL1 produces output in logical layout:
///     - without batch: [oC, oH, oW]
///     - with batch:    [B, oC, oH, oW]
///   For hardware execution, batch and group are fused together, oH and oW are
///   fused together and the output layout is required to be aligned:
///     - oHW aligned to FRACTAL_BLOCK_NUM
///     - oCPerGroup aligned to FRACTAL_BLOCK_NUM
///
/// Rewrite overview:
///   1. Rewrite ConvOps to produce a fused & aligned layout:
///        [oHWCeil, fusedOCCeil]
///      where:
///        - oHW         = oH * oW
///        - fusedOC     = B * oC
///        - fusedOCCeil = B * ceil(oC / groups, FRACTAL_BLOCK_NUM) * groups
///        - oHWCeil     = ceil(oHW, FRACTAL_BLOCK_NUM)
///
///   2. Post-process the aligned layout:
///      - Slice away padding on width and channels.
///      - Transpose the result to get [fusedOC, oHW].
///      - For grouped and unaligned channels, remove per-group padding by
///        per-group slice + insert.
///
///   3. Restore user-visible layout if batch exists or 2D convolution:
///        1D (fusedOC = oC, oHW = oW)): [fusedOC, oHW] -> [oC, oW]
///        1D + batch (oHW = oW)): [fusedOC, oHW] -> [B, oC, oW]
///        2D (fusedOC = oC): [fusedOC, oW] -> [oC, oH, oW]
///        2D + batch: [fusedOC, oW] -> [B, oC, oH, oW]
///
/// Semantics:
///   This pattern only changes the physical layout for alignment and
///   hardware execution, and then restores the logical layout expected by
///   users. The numerical semantics are unchanged.
///
/// In short:
///   user layout
///     -> fused & aligned layout (for hardware)
///     -> sliced / transposed / reshaped back to user layout
template <typename ConvOpType>
struct NormalizeConvOutputPattern : public OpRewritePattern<ConvOpType> {
public:
  using OpRewritePattern<ConvOpType>::OpRewritePattern;
  ~NormalizeConvOutputPattern() override = default;

  LogicalResult matchAndRewrite(ConvOpType op,
                                PatternRewriter &rewriter) const override {

    if (op->getAttr(outputAlreadyNormalized)) {
      return failure();
    }

    auto input = op.getInput();
    auto weight = op.getWeight();
    auto bias = op.getBias();
    auto init = op.getInit();
    auto initCondition = op.getInitCondition();
    auto padding = op.getPaddingAttr();
    auto groups = op.getGroupsAttr();
    int64_t groupsVal = groups.getInt();

    // For 1D conv: rank should be 2 (CW) or 3 (NCW)
    // For 2D conv: rank should be 3 (CHW) or 4 (NCHW)
    static constexpr int64_t baseDims = ConvBaseDims<ConvOpType>::dim;
    int64_t expectedMinRank = baseDims;
    int64_t expectedMaxRank = baseDims + 1;

    auto convResult = op.getResultTensors()[0];
    auto resultType = dyn_cast<RankedTensorType>(convResult.getType());
    if (!resultType || (resultType.getRank() != expectedMinRank &&
                        resultType.getRank() != expectedMaxRank)) {
      return failure();
    }

    auto initType = dyn_cast<RankedTensorType>(init.getType());
    if (!initType || (initType.getRank() != expectedMinRank &&
                      initType.getRank() != expectedMaxRank)) {
      return failure();
    }

    bool hasBatch = resultType.getRank() == expectedMaxRank;

    int64_t batch = 1;
    int64_t oH = 1;
    int64_t oC, oW;

    if (baseDims == 2) {
      // 1D convolution
      if (!hasBatch) {
        // [oC, oW]
        oC = resultType.getDimSize(0);
        oW = resultType.getDimSize(1);
      } else {
        // [B, oC, oW]
        batch = resultType.getDimSize(0);
        oC = resultType.getDimSize(1);
        oW = resultType.getDimSize(2);
      }
    } else if (baseDims == 3) {
      // 2D convolution
      if (!hasBatch) {
        // [oC, oH, oW]
        oC = resultType.getDimSize(0);
        oH = resultType.getDimSize(1);
        oW = resultType.getDimSize(2);
      } else {
        // [B, oC, oH, oW]
        batch = resultType.getDimSize(0);
        oC = resultType.getDimSize(1);
        oH = resultType.getDimSize(2);
        oW = resultType.getDimSize(3);
      }
    } else {
      // 3D convolution or other
      return failure();
    }

    int64_t oHW = oH * oW;

    int64_t oCPerGroup = oC / groupsVal;
    int64_t oCPerGroupCeil = CEIL_FACTOR(oCPerGroup, utils::FRACTAL_BLOCK_NUM);
    int64_t oCCeil = oCPerGroupCeil * groupsVal;

    bool isOCPerGroupAligned = (oCPerGroup % utils::FRACTAL_BLOCK_NUM == 0);
    int64_t oHWCeil = CEIL_FACTOR(oHW, utils::FRACTAL_BLOCK_NUM);
    int64_t fusedOC = batch * oC;
    int64_t fusedOCCeil = batch * oCCeil;
    int64_t fusedGroupsVal = batch * groupsVal;

    auto elementType = resultType.getElementType();
    Location loc = op.getLoc();

    SmallVector<int64_t> newShape{oHWCeil, fusedOCCeil};
    auto newResultType = RankedTensorType::get(newShape, elementType);
    Value newEmpty =
        rewriter.create<tensor::EmptyOp>(loc, newShape, elementType);

    // === create new ConvOp with result of new shape ===
    auto newConvOp = rewriter.create<ConvOpType>(
        loc,           // location
        newResultType, // result type: [oHWCeil, fusedOCCeil]
        input,         // input
        weight,        // weight
        bias,          // bias
        newEmpty,      // init: [oHWCeil, fusedOCCeil]
        initCondition, // init condition
        ValueRange{},  // sync_related_args
        padding,       // padding attribute
        groups         // groups attribute
    );

    Value newResult = newConvOp.getResultTensors()[0];

    // === rewrite vcast and update target if exist ===
    Value target = convResult;
    Value newTarget = newResult;
    hivm::VCastOp castOp = nullptr;

    if (convResult.hasOneUse()) {
      castOp = dyn_cast<hivm::VCastOp>(*convResult.user_begin());
    }

    if (castOp) {
      auto castResultType =
          mlir::cast<RankedTensorType>(castOp->getResult(0).getType());
      auto newCastResultType =
          RankedTensorType::get(newShape, castResultType.getElementType());
      auto newCastInit = rewriter.create<tensor::EmptyOp>(
          loc, newShape, castResultType.getElementType());

      auto newCastOp = rewriter.create<hivm::VCastOp>(
          loc, TypeRange{newCastResultType}, ValueRange{newResult},
          ValueRange{newCastInit.getResult()}, /*temp_buffer=*/Value(),
          castOp.getRoundModeAttr(), hivm::TypeFnAttr{});

      target = castOp->getResult(0);
      newTarget = newCastOp->getResult(0);
    }

    auto newTargetType =
        mlir::cast<RankedTensorType>(newTarget.getType()).getElementType();

    // === post process to [fusedOC, oHW] ===
    if (isOCPerGroupAligned || fusedGroupsVal == 1) {
      // case: aligned or fusedGroupsVal == 1
      // step 1: [oHWCeil, fusedOCCeil] -> [oHW, fusedOC]
      SmallVector<OpFoldResult, 2> offsets{
          rewriter.getIndexAttr(0),
          rewriter.getIndexAttr(0),
      };

      SmallVector<OpFoldResult, 2> sizes{
          rewriter.getIndexAttr(oHW),
          rewriter.getIndexAttr(fusedOC),
      };

      SmallVector<OpFoldResult, 2> strides{
          rewriter.getIndexAttr(1),
          rewriter.getIndexAttr(1),
      };

      auto sliceType = RankedTensorType::get({oHW, fusedOC}, newTargetType);
      auto slice = rewriter.create<tensor::ExtractSliceOp>(
          loc, sliceType, newTarget, offsets, sizes, strides);

      // step 2: transpose [oHW, fusedOC] -> [fusedOC, oHW]
      SmallVector<int64_t> tShape{fusedOC, oHW};
      Value tInit =
          rewriter.create<tensor::EmptyOp>(loc, tShape, newTargetType);
      SmallVector<int64_t, 2> perm{1, 0};

      newResult = rewriter
                      .create<hivm::VTransposeOp>(
                          loc, TypeRange{tInit.getType()}, slice, tInit,
                          rewriter.getDenseI64ArrayAttr(perm))
                      ->getResult(0);
    } else {
      // case: not aligned && fusedGroupsVal != 1
      // step 1: transpose [oHWCeil, fusedOCCeil] -> [fusedOCCeil, oHWCeil]
      SmallVector<int64_t> tShape{fusedOCCeil, oHWCeil};
      Value tInit =
          rewriter.create<tensor::EmptyOp>(loc, tShape, newTargetType);
      SmallVector<int64_t, 2> perm{1, 0};

      Value transposedOutput =
          rewriter
              .create<hivm::VTransposeOp>(loc, TypeRange{tInit.getType()},
                                          newTarget, tInit,
                                          rewriter.getDenseI64ArrayAttr(perm))
              ->getResult(0);

      // step 2: alloc [fusedOC, oHWCeil]
      SmallVector<int64_t> sShape{fusedOC, oHWCeil};
      Value slicedOutput =
          rewriter.create<tensor::EmptyOp>(loc, sShape, newTargetType);

      // step 3: for i in [0, fusedGroupsVal), extract_slice + insert_slice
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upper = rewriter.create<arith::ConstantIndexOp>(loc, fusedGroupsVal);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      auto forOp = rewriter.create<scf::ForOp>(loc, zero, upper, step,
                                               ValueRange{slicedOutput});

      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());

        Value iv = forOp.getInductionVar();
        Value slicedOutput = forOp.getRegionIterArg(0);

        auto oCPerGroupCeilVal =
            rewriter.create<arith::ConstantIndexOp>(loc, oCPerGroupCeil);
        auto oCPerGroupVal =
            rewriter.create<arith::ConstantIndexOp>(loc, oCPerGroup);

        Value srcOffset0 =
            rewriter.create<arith::MulIOp>(loc, iv, oCPerGroupCeilVal);
        Value dstOffset0 =
            rewriter.create<arith::MulIOp>(loc, iv, oCPerGroupVal);

        // subTransposedOutput = extract_slice transposedOutput
        // [oCPerGroup, oHWCeil]
        SmallVector<OpFoldResult, 2> srcOffsets{srcOffset0,
                                                rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult, 2> srcSizes{rewriter.getIndexAttr(oCPerGroup),
                                              rewriter.getIndexAttr(oHWCeil)};
        SmallVector<OpFoldResult, 2> srcStrides{rewriter.getIndexAttr(1),
                                                rewriter.getIndexAttr(1)};

        auto subTransposedOutput = rewriter.create<tensor::ExtractSliceOp>(
            loc, transposedOutput, srcOffsets, srcSizes, srcStrides);

        // newSlicedOutput = insert_slice subTransposedOutput into slicedOutput
        SmallVector<OpFoldResult, 2> dstOffsets{dstOffset0,
                                                rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult, 2> dstSizes{rewriter.getIndexAttr(oCPerGroup),
                                              rewriter.getIndexAttr(oHWCeil)};
        SmallVector<OpFoldResult, 2> dstStrides{rewriter.getIndexAttr(1),
                                                rewriter.getIndexAttr(1)};

        auto newSlicedOutput = rewriter.create<tensor::InsertSliceOp>(
            loc, subTransposedOutput.getResult(), slicedOutput, dstOffsets,
            dstSizes, dstStrides);

        rewriter.create<scf::YieldOp>(loc, newSlicedOutput.getResult());
      }

      Value slicedResult = forOp.getResult(0);

      // step 4: extract_slice slicedResult: [fusedOC, oHWCeil] -> [fusedOC,
      // oHW]
      SmallVector<OpFoldResult, 2> offsets{
          rewriter.getIndexAttr(0),
          rewriter.getIndexAttr(0),
      };
      SmallVector<OpFoldResult, 2> sizes{
          rewriter.getIndexAttr(fusedOC),
          rewriter.getIndexAttr(oHW),
      };
      SmallVector<OpFoldResult, 2> strides{
          rewriter.getIndexAttr(1),
          rewriter.getIndexAttr(1),
      };

      auto finalType = RankedTensorType::get({fusedOC, oHW}, newTargetType);
      auto extractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
          loc, finalType, slicedResult, offsets, sizes, strides);

      newResult = extractSliceOp.getResult();
    }

    // === batch reshape and oHW split ===
    if (baseDims == 2) {
      if (hasBatch) {
        auto finalType = RankedTensorType::get({batch, oC, oHW}, newTargetType);

        SmallVector<ReassociationIndices> reassoc = {
            {0, 1}, // B * oC
            {2}     // oHW
        };

        newResult = rewriter.create<tensor::ExpandShapeOp>(loc, finalType,
                                                           newResult, reassoc);
      }
    } else if (baseDims == 3) {
      if (hasBatch) {
        auto finalType =
            RankedTensorType::get({batch, oC, oH, oW}, newTargetType);

        SmallVector<ReassociationIndices> reassoc = {
            {0, 1}, // B * oC
            {2, 3}  // oH * oW
        };

        newResult = rewriter.create<tensor::ExpandShapeOp>(loc, finalType,
                                                           newResult, reassoc);
      } else {
        auto finalType = RankedTensorType::get({oC, oH, oW}, newTargetType);

        SmallVector<ReassociationIndices> reassoc = {
            {0},   // oC
            {1, 2} // oH * oW
        };

        newResult = rewriter.create<tensor::ExpandShapeOp>(loc, finalType,
                                                           newResult, reassoc);
      }
    } else {
      return failure();
    }

    rewriter.replaceOp(target.getDefiningOp(), newResult);
    newConvOp->setAttr(outputAlreadyNormalized, rewriter.getUnitAttr());
    return success();
  }
};

void populateNormalizeConvOpsPattern1(RewritePatternSet &patterns) {
  patterns.add<NormalizeConvResultTypePattern<BFloat16Type, hivm::Conv1DL1Op>>(
      patterns.getContext());
  patterns.add<NormalizeConvResultTypePattern<BFloat16Type, hivm::Conv2DL1Op>>(
      patterns.getContext());
  patterns.add<DecomposeConv1dWithBiasPattern<Float16Type, hivm::Conv1DL1Op>>(
      patterns.getContext());
  patterns.add<DecomposeConv1dWithBiasPattern<Float16Type, hivm::Conv2DL1Op>>(
      patterns.getContext());
}

void populateNormalizeConvOpsPattern2(RewritePatternSet &patterns) {
  patterns.add<NormalizeConvResultTypePattern<Float16Type, hivm::Conv1DL1Op>>(
      patterns.getContext());
  patterns.add<NormalizeConvResultTypePattern<Float16Type, hivm::Conv2DL1Op>>(
      patterns.getContext());
  patterns.add<DecomposeConv1dWithBiasPattern<Float32Type, hivm::Conv1DL1Op>>(
      patterns.getContext());
  patterns.add<DecomposeConv1dWithBiasPattern<Float32Type, hivm::Conv2DL1Op>>(
      patterns.getContext());
}

void populateNormalizeConvOpsPattern3(RewritePatternSet &patterns) {
  patterns.add<NormalizeConvInputAndWeightPattern<hivm::Conv1DL1Op>>(
      patterns.getContext());
  patterns.add<NormalizeConvInputAndWeightPattern<hivm::Conv2DL1Op>>(
      patterns.getContext());
  patterns.add<NormalizeConvOutputPattern<hivm::Conv1DL1Op>>(
      patterns.getContext());
  patterns.add<NormalizeConvOutputPattern<hivm::Conv2DL1Op>>(
      patterns.getContext());
}

void NormalizeConvOpsPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto *context = &getContext();
  auto *funcOp = getOperation();

  // First Round
  RewritePatternSet patterns1(context);
  populateNormalizeConvOpsPattern1(patterns1);
  GreedyRewriteConfig config1 = GreedyRewriteConfig();
  (void)applyPatternsGreedily(funcOp, std::move(patterns1), config1);

  // Second Round
  RewritePatternSet patterns2(context);
  populateNormalizeConvOpsPattern2(patterns2);
  GreedyRewriteConfig config2 = GreedyRewriteConfig();
  (void)applyPatternsGreedily(funcOp, std::move(patterns2), config2);

  // Third Round
  RewritePatternSet patterns3(context);
  populateNormalizeConvOpsPattern3(patterns3);
  GreedyRewriteConfig config3 = GreedyRewriteConfig();
  (void)applyPatternsGreedily(funcOp, std::move(patterns3), config3);
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createNormalizeConvOpsPass() {
  return std::make_unique<NormalizeConvOpsPass>();
}