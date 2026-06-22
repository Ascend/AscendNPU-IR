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
#include "bishengir/Dialect/Annotation/IR/Annotation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Casting.h"

#include <array>
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
/// This pass normalizes Conv1D/Conv2D/Conv3D operations by adapting
/// user-facing tensors to the hardware-required layout and precision, and
/// then restoring results back to the user-visible form.
///
/// The transformation is organized into multiple stages, each handling a
/// specific responsibility:
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
/// Note:
///   Conv1D/Conv2D/Conv3D share the same output normalization template.
///   Conv3D is mapped to an "effective batch" (B*D) and only keeps a small
///   D-axis restore branch at the end.
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

template <> struct ConvBaseDims<hivm::Conv3DL1Op> {
  // the base dimension for Conv3DL1Op is 4 (CDHW)
  static constexpr int64_t dim = 4;
};

static constexpr llvm::StringLiteral outputAlreadyNormalized =
    "outputAlreadyNormalized";
static constexpr llvm::StringLiteral conv3dDepthPadded =
    "conv3dDepthPadded";
// outputAlreadyNormalized:
//   Generic idempotence guard for output normalization rewrite. Once set, the
//   op will be skipped by NormalizeConvOutputPattern to avoid repeated
//   reshape/transpose cycles.
//
// conv3dDepthPadded:
//   Explicit contract between normalize and decompose for Conv3d. When D
//   padding > 0, normalize materializes depth padding and tags the op, while
//   decompose relies on this tag and keeps H/W padding on Conv2d attributes.

template <size_t Rank>
FailureOr<std::array<int64_t, Rank>> getConvIntArrayAttr(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    int64_t value = intAttr.getInt();
    std::array<int64_t, Rank> values;
    values.fill(value);
    return values;
  }

  if (auto denseAttr = dyn_cast<DenseI64ArrayAttr>(attr)) {
    if (denseAttr.size() != Rank)
      return failure();
    std::array<int64_t, Rank> values;
    for (size_t idx = 0; idx < Rank; ++idx)
      values[idx] = denseAttr[idx];
    return values;
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    if (arrayAttr.size() != Rank)
      return failure();

    std::array<int64_t, Rank> values;
    for (auto [idx, element] : llvm::enumerate(arrayAttr)) {
      auto intAttr = dyn_cast<IntegerAttr>(element);
      if (!intAttr)
        return failure();
      values[idx] = intAttr.getInt();
    }
    return values;
  }

  return failure();
}

inline RoundModeAttr getRoundAttr(mlir::OpBuilder &b, Type srcType,
                                  Type dstType) {
  return hivm::RoundModeAttr::get(
      b.getContext(),
      mlir::utils::selectRoundMode<hivm::RoundMode>(srcType, dstType));
}

// Expand a 1-D channel bias and add it to a channel-major tensor, e.g.
// [C, HW] + [C, 1] or [N, C, HW] + [1, C, 1].
FailureOr<Value> createChannelBiasAdd(PatternRewriter &rewriter, Location loc,
                                      Value input, Value bias,
                                      ArrayRef<int64_t> resultShape,
                                      int64_t channelDim, Type elemType) {
  auto biasType = dyn_cast<RankedTensorType>(bias.getType());
  if (!biasType || biasType.getRank() != 1 ||
      biasType.getElementType() != elemType) {
    return failure();
  }
  int64_t rank = static_cast<int64_t>(resultShape.size());
  if (channelDim < 0 || channelDim >= rank) {
    return failure();
  }

  int64_t channelSize = resultShape[channelDim];
  if (biasType.getDimSize(0) != ShapedType::kDynamic &&
      channelSize != ShapedType::kDynamic &&
      biasType.getDimSize(0) != channelSize) {
    return failure();
  }

  SmallVector<int64_t> biasShape(rank, 1);
  biasShape[channelDim] = channelSize;
  SmallVector<int64_t, 2> biasReassocIndices;
  for (int64_t i = 0; i < rank; ++i) {
    biasReassocIndices.push_back(i);
  }
  SmallVector<SmallVector<int64_t, 2>> biasReassoc = {biasReassocIndices};
  auto expandedBiasType = RankedTensorType::get(biasShape, elemType);
  Value expandedBias = rewriter.create<tensor::ExpandShapeOp>(
      loc, expandedBiasType, bias, biasReassoc);

  SmallVector<int64_t> broadcastDims;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != channelDim) {
      broadcastDims.push_back(i);
    }
  }

  auto vaddType = RankedTensorType::get(resultShape, elemType);
  Value vaddInit =
      rewriter.create<tensor::EmptyOp>(loc, vaddType.getShape(), elemType);
  return rewriter
      .create<hivm::VAddOp>(
          loc, TypeRange{vaddType}, ValueRange{input, expandedBias},
          ValueRange{vaddInit}, Value(), nullptr,
          rewriter.getDenseI64ArrayAttr(broadcastDims))
      ->getResult(0);
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

LogicalResult padDepthForConv3dInput(hivm::Conv3DL1Op op,
                                     PatternRewriter &rewriter,
                                     int64_t depthPadding) {
  // Depth padding is handled explicitly here (instead of in decompose) so the
  // rest of the conv3d->conv2d lowering can assume a clean sliding window over
  // depth with no extra control-flow branches.
  if (depthPadding <= 0) {
    return success();
  }

  Value input = op.getInput();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || !inputType.hasStaticShape() || inputType.getRank() != 5) {
    return failure();
  }

  int64_t batch = inputType.getDimSize(0);
  int64_t channel = inputType.getDimSize(1);
  int64_t depth = inputType.getDimSize(2);
  int64_t height = inputType.getDimSize(3);
  int64_t width = inputType.getDimSize(4);
  Type elementType = inputType.getElementType();
  Location loc = op.getLoc();

  auto zero = rewriter.create<arith::ConstantOp>(loc,
                                                 rewriter.getZeroAttr(elementType));
  auto emptyAttr = rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{});

  auto createZeroPad = [&](ArrayRef<int64_t> shape) -> Value {
    Value empty = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);
    return rewriter
        .create<hivm::VBrcOp>(loc, TypeRange{empty.getType()}, zero, empty,
                              emptyAttr)
        ->getResult(0);
  };

  Value frontPad = createZeroPad(
      ArrayRef<int64_t>{batch, channel, depthPadding, height, width});
  Value backPad = createZeroPad(
      ArrayRef<int64_t>{batch, channel, depthPadding, height, width});

  Value concatFrontInit = rewriter.create<tensor::EmptyOp>(
      loc, ArrayRef<int64_t>{batch, channel, depth + depthPadding, height, width},
      elementType);
  Value inputWithFront =
      rewriter
          .create<hivm::VConcatOp>(
              loc, TypeRange{concatFrontInit.getType()},
              rewriter.getI64IntegerAttr(2), ValueRange{frontPad, input},
              concatFrontInit)
          ->getResult(0);

  Value concatFullInit = rewriter.create<tensor::EmptyOp>(
      loc,
      ArrayRef<int64_t>{batch, channel, depth + 2 * depthPadding, height, width},
      elementType);
  Value depthPaddedInput =
      rewriter
          .create<hivm::VConcatOp>(
              loc, TypeRange{concatFullInit.getType()},
              rewriter.getI64IntegerAttr(2), ValueRange{inputWithFront, backPad},
              concatFullInit)
          ->getResult(0);

  op->replaceUsesOfWith(input, depthPaddedInput);
  // Persist the "depth already materialized" fact for downstream decompose.
  op->setAttr(conv3dDepthPadded, rewriter.getUnitAttr());
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

  // Step 2: Collapse spatial dimensions to reduce transpose cost.
  // 2D: [N, C, iH, iW] -> [N, C, iHW]
  // 3D: [N, C, iD, iH, iW] -> [N, C, iDHW]
  Value currentInput = input;
  int64_t iHW = 1;
  if (baseDims == 3 || baseDims == 4) {
    for (auto size : spatialSizes) {
      iHW *= size;
    }

    SmallVector<int64_t> collapsedShape{batch, C, iHW};
    SmallVector<ReassociationIndices> collapseReassoc;
    if (baseDims == 3) {
      collapseReassoc = {{0}, {1}, {2, 3}};
    } else {
      collapseReassoc = {{0}, {1}, {2, 3, 4}};
    }

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
  } else if (baseDims == 4) {
    // For 3D: [N, C1, iDHW, C0] -> [N, C1, iD, iH, iW, C0]
    finalShape.push_back(batch);
    finalShape.push_back(C1);
    finalShape.insert(finalShape.end(), spatialSizes.begin(),
                      spatialSizes.end());
    finalShape.push_back(C0);
  }
  SmallVector<ReassociationIndices> finalReassoc;
  if (baseDims == 4) {
    finalReassoc = {{0}, {1}, {2, 3, 4}, {5}};
  } else {
    finalReassoc = {{0}, {1}, {2, 3}, {4}};
  }
  auto expandShapeOp1 = rewriter.create<tensor::ExpandShapeOp>(
      op->getLoc(), RankedTensorType::get(finalShape, elementType),
      vtransposeOp->getResults()[0], finalReassoc);

  // Keep downstream contracts unchanged: Conv3d decompose currently expects
  // packed input as [N, D, C1, H, W, C0].
  Value finalVal = expandShapeOp1->getResult(0);
  if (baseDims == 4) {
    // Generic packed flow naturally produces [N, C1, D, H, W, C0] for 3D.
    // Swap to [N, D, C1, H, W, C0] to align with decompose slicing semantics
    // (depth is the loop-driving axis).
    auto finalValType = dyn_cast<RankedTensorType>(finalVal.getType());
    if (!finalValType || !finalValType.hasStaticShape()) {
      return failure();
    }
    auto shape = finalValType.getShape();
    SmallVector<int64_t> outputShape{shape[0], shape[2], shape[1],
                                     shape[3], shape[4], shape[5]};
    SmallVector<int64_t> permutation{0, 2, 1, 3, 4, 5};
    auto dst = rewriter.create<tensor::EmptyOp>(op->getLoc(), outputShape,
                                                finalValType.getElementType());
    auto vtranspose = rewriter.create<hivm::VTransposeOp>(
        op->getLoc(), TypeRange{dst.getType()}, finalVal, dst.getResult(),
        rewriter.getDenseI64ArrayAttr(permutation));
    finalVal = vtranspose.getResult()[0];
  }

  // Step 6: insert store + load
  Type type = finalVal.getType();
  Type elemType = getElementTypeOrSelf(type);

  Value storeInit = utils::createEmptyOp(rewriter, op->getLoc(), finalVal);

  auto storeOp = rewriter.create<hivm::StoreOp>(op->getLoc(), TypeRange{type},
                                                finalVal, storeInit);

  auto NC1HWC0Layout =
      DataLayoutAttr::get(rewriter.getContext(), hivm::DataLayout::NC1HWC0);

  auto markOp =
      rewriter.create<annotation::MarkOp>(op->getLoc(), storeOp->getResult(0));

  markOp->setAttr(hivm::kHIVMDataLayoutAttrName, NC1HWC0Layout);

  Value loadInit = mlir::utils::createEmptyOpWithTargetElemType(
      rewriter, op->getLoc(), finalVal, elemType, std::nullopt);

  auto loadOp = rewriter.create<hivm::LoadOp>(op->getLoc(), TypeRange{type},
                                              storeOp->getResult(0), loadInit);

  op->replaceUsesOfWith(input, loadOp->getResult(0));

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

  // Step 2: Collapse kernel spatial dimensions to reduce transpose cost.
  // 2D: [oC, C/groups, wH, wW] -> [oC, C/groups, wHW]
  // 3D: [oC, C/groups, wD, wH, wW] -> [oC, C/groups, wDHW]
  Value currentWeight = weight;
  int64_t wHW = 1;
  if (baseDims == 3 || baseDims == 4) {
    for (auto size : kernelSizes) {
      wHW *= size;
    }

    SmallVector<int64_t> collapsedShape{oC, C / groups, wHW};
    SmallVector<ReassociationIndices> collapseReassoc;
    if (baseDims == 3) {
      collapseReassoc = {{0}, {1}, {2, 3}};
    } else {
      collapseReassoc = {{0}, {1}, {2, 3, 4}};
    }

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
  } else if (baseDims == 4) {
    // For 3D: [C1/groups, wDHW, oC, C0] -> [C1/groups, wD, wH, wW, oC, C0]
    finalShape.push_back(newChannelsPerGroup);
    finalShape.insert(finalShape.end(), kernelSizes.begin(), kernelSizes.end());
    finalShape.push_back(oC);
    finalShape.push_back(C0);
  }
  SmallVector<ReassociationIndices> finalReassoc;
  if (baseDims == 4) {
    finalReassoc = {{0}, {1, 2, 3}, {4}, {5}};
  } else {
    finalReassoc = {{0}, {1, 2}, {3}, {4}};
  }
  auto expandShapeOp1 = rewriter.create<tensor::ExpandShapeOp>(
      op->getLoc(), RankedTensorType::get(finalShape, elementType),
      vtransposeOp2.getResult()[0], finalReassoc);

  // Keep downstream contracts unchanged: Conv3d decompose currently expects
  // packed weight as [wD, C1/groups, wH, wW, oC, C0].
  Value finalVal = expandShapeOp1->getResult(0);
  if (baseDims == 4) {
    // Generic packed flow yields [C1/groups, wD, wH, wW, oC, C0] for 3D.
    // Swap to [wD, C1/groups, wH, wW, oC, C0] so each kd iteration can take a
    // simple contiguous slice on axis 0.
    auto finalValType = dyn_cast<RankedTensorType>(finalVal.getType());
    if (!finalValType || !finalValType.hasStaticShape()) {
      return failure();
    }
    auto shape = finalValType.getShape();
    SmallVector<int64_t> outputShape{shape[1], shape[0], shape[2],
                                     shape[3], shape[4], shape[5]};
    SmallVector<int64_t> permutation{1, 0, 2, 3, 4, 5};
    auto dst = rewriter.create<tensor::EmptyOp>(op->getLoc(), outputShape,
                                                finalValType.getElementType());
    auto vtranspose = rewriter.create<hivm::VTransposeOp>(
        op->getLoc(), TypeRange{dst.getType()}, finalVal, dst.getResult(),
        rewriter.getDenseI64ArrayAttr(permutation));
    finalVal = vtranspose.getResult()[0];
  }

  // Step 8: insert store + load & mark with layout
  Type type = finalVal.getType();
  Type elemType = getElementTypeOrSelf(type);

  Value storeInit = utils::createEmptyOp(rewriter, op->getLoc(), finalVal);

  auto storeOp = rewriter.create<hivm::StoreOp>(op->getLoc(), TypeRange{type},
                                                finalVal, storeInit);

  auto C1HWNC0Layout =
      DataLayoutAttr::get(rewriter.getContext(), hivm::DataLayout::C1HWNC0);

  auto markOp =
      rewriter.create<annotation::MarkOp>(op->getLoc(), storeOp->getResult(0));

  markOp->setAttr(hivm::kHIVMDataLayoutAttrName, C1HWNC0Layout);

  Value loadInit = mlir::utils::createEmptyOpWithTargetElemType(
      rewriter, op->getLoc(), finalVal, elemType, std::nullopt);

  auto loadOp = rewriter.create<hivm::LoadOp>(op->getLoc(), TypeRange{type},
                                              storeOp->getResult(0), loadInit);

  op->replaceUsesOfWith(weight, loadOp->getResult(0));

  return success();
}

/// This pattern normalizes the input and weight layout of Conv1D/Conv2D
/// operations by transforming them into a fractal format for efficient
/// hardware execution.
///
/// Note:
///   Conv3D input/weight normalization is also handled by this template
///   (baseDims==4 branch), using collapse(DHW) -> transpose -> expand(DHW)
///   to reduce transpose count while preserving downstream packed contracts.
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
    static constexpr int64_t baseDims = ConvBaseDims<ConvOpType>::dim;

    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

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
    if (!inputType.hasStaticShape() ||
        (inputType.getRank() != baseDims &&
         inputType.getRank() != baseDims + 1)) {
      return failure();
    }

    auto weightType = cast<ShapedType>(weight.getType());
    if (!weightType.hasStaticShape() ||
        weightType.getRank() != baseDims + 1) {
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
      return op.emitError("Expected Attr named groups for Conv op");
    }

    if (!isa<IntegerAttr>(op->getAttr("groups"))) {
      return op.emitError("Expected IntegerType for Attr 'groups'");
    }

    // set insert loc before the define of out operand
    auto *outDefOp = op.getInit().getDefiningOp();
    if (outDefOp) {
      rewriter.setInsertionPoint(outDefOp);
    }

    if (inputType.getRank() == baseDims) {
      if (failed(expandToBatch<ConvOpType>(op, rewriter))) {
        return rewriter.notifyMatchFailure(op, "Failed to expand to batch");
      }
    }

    if constexpr (baseDims == 4) {
      auto padding = getConvIntArrayAttr<3>(op.getPaddingAttr());
      if (failed(padding)) {
        return rewriter.notifyMatchFailure(
            op, "Conv3d padding must be a scalar or 3-element array");
      }
      int64_t depthPadding = (*padding)[0];
      if (depthPadding < 0 || (*padding)[1] < 0 || (*padding)[2] < 0) {
        return rewriter.notifyMatchFailure(op, "Conv3d padding must be >= 0");
      }
      // Normalize consumes only D padding. H/W padding stays on Conv3dL1 and
      // is converted to Conv2dL1 [paddingH, paddingW] during decompose.
      if (!op->hasAttr(conv3dDepthPadded) && depthPadding > 0) {
        if (failed(
                padDepthForConv3dInput(cast<hivm::Conv3DL1Op>(op), rewriter,
                                       depthPadding))) {
          return rewriter.notifyMatchFailure(op,
                                             "Failed to pre-pad Conv3d depth");
        }
      }
    }

    int64_t groups = op.getGroups();
    input = op.getInput();
    inputType = cast<ShapedType>(input.getType());
    int64_t iC = inputType.getDimSize(1);
    int64_t C = CEIL_DIV(iC, C0 * groups) * (C0 * groups);

    LLVM_DEBUG(llvm::dbgs()
               << "start insert [pad expand trans] op for conv op"
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

    LLVM_DEBUG(llvm::dbgs() << "insert op for conv op successfully!"
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
struct DecomposeConvResultWithBiasPattern
    : public OpRewritePattern<ConvOpType> {
public:
  using OpRewritePattern<ConvOpType>::OpRewritePattern;
  ~DecomposeConvResultWithBiasPattern() override = default;

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
    if (biasType.getRank() != 1) {
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
    // For 3D conv: rank should be 4 (CDHW) or 5 (NCDHW)
    int64_t expectedMinRank = baseDims;
    int64_t expectedMaxRank = baseDims + 1;
    if (rank != expectedMinRank && rank != expectedMaxRank) {
      return failure();
    }

    if constexpr (baseDims == 4) {
      if (resultType.hasStaticShape()) {
        // Static Conv3D output normalization restores the fused result to the
        // same rank-3 channel-bias layout used by Conv2D, so handle it there.
        return failure();
      }
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
    bool hasBatch = rank == expectedMaxRank;

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
    SmallVector<int64_t> vaddShape(shape.begin(), shape.end());
    Value vaddInput = convResult;

    auto getCollapsedDim = [](int64_t lhs, int64_t rhs) {
      if (lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic) {
        return ShapedType::kDynamic;
      }
      return lhs * rhs;
    };

    if constexpr (baseDims == 3) {
      if (hasBatch) {
        // [N, oC, oH, oW] -> [N, oC, oHW]
        vaddShape = {shape[0], shape[1], getCollapsedDim(shape[2], shape[3])};
        SmallVector<ReassociationIndices> reassoc = {{0}, {1}, {2, 3}};
        auto collapsedType = RankedTensorType::get(vaddShape, elemType);
        vaddInput = rewriter.create<tensor::CollapseShapeOp>(
            loc, collapsedType, convResult, reassoc);
      } else {
        // [oC, oH, oW] -> [oC, oHW]
        vaddShape = {shape[0], getCollapsedDim(shape[1], shape[2])};
        SmallVector<ReassociationIndices> reassoc = {{0}, {1, 2}};
        auto collapsedType = RankedTensorType::get(vaddShape, elemType);
        vaddInput = rewriter.create<tensor::CollapseShapeOp>(
            loc, collapsedType, convResult, reassoc);
      }
    } else if constexpr (baseDims == 4) {
      if (hasBatch) {
        // [N, oC, oD, oH, oW] -> [N, oC, oD, oHW]
        vaddShape = {shape[0], shape[1], shape[2],
                     getCollapsedDim(shape[3], shape[4])};
        SmallVector<ReassociationIndices> reassoc = {{0}, {1}, {2}, {3, 4}};
        auto collapsedType = RankedTensorType::get(vaddShape, elemType);
        vaddInput = rewriter.create<tensor::CollapseShapeOp>(
            loc, collapsedType, convResult, reassoc);
      } else {
        // [oC, oD, oH, oW] -> [oC, oD, oHW]
        vaddShape = {shape[0], shape[1], getCollapsedDim(shape[2], shape[3])};
        SmallVector<ReassociationIndices> reassoc = {{0}, {1}, {2, 3}};
        auto collapsedType = RankedTensorType::get(vaddShape, elemType);
        vaddInput = rewriter.create<tensor::CollapseShapeOp>(
            loc, collapsedType, convResult, reassoc);
      }
    }
    auto convResultElemType =
        dyn_cast<RankedTensorType>(convResult.getType()).getElementType();

    if (convResultElemType != biasElemType) {
      return rewriter.notifyMatchFailure(
          op, "bias type mismatch with convResult type");
    }

    int64_t channelDim = hasBatch ? 1 : 0;
    auto biasAdd = createChannelBiasAdd(rewriter, loc, vaddInput, bias,
                                        vaddShape, channelDim,
                                        convResultElemType);
    if (failed(biasAdd)) {
      return failure();
    }
    Value result = *biasAdd;
    if constexpr (baseDims == 3) {
      auto expandedResultType =
          RankedTensorType::get(shape, convResultElemType);
      if (hasBatch) {
        SmallVector<ReassociationIndices> reassoc = {{0}, {1}, {2, 3}};
        result = rewriter.create<tensor::ExpandShapeOp>(loc, expandedResultType,
                                                        result, reassoc);
      } else {
        SmallVector<ReassociationIndices> reassoc = {{0}, {1, 2}};
        result = rewriter.create<tensor::ExpandShapeOp>(loc, expandedResultType,
                                                        result, reassoc);
      }
    } else if constexpr (baseDims == 4) {
      auto expandedResultType =
          RankedTensorType::get(shape, convResultElemType);
      if (hasBatch) {
        SmallVector<ReassociationIndices> reassoc = {{0}, {1}, {2}, {3, 4}};
        result = rewriter.create<tensor::ExpandShapeOp>(loc, expandedResultType,
                                                        result, reassoc);
      } else {
        SmallVector<ReassociationIndices> reassoc = {{0}, {1}, {2, 3}};
        result = rewriter.create<tensor::ExpandShapeOp>(loc, expandedResultType,
                                                        result, reassoc);
      }
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// This pattern normalizes the output layout of Conv1D/Conv2D by fusing
/// batch and group dimensions and aligning width/per-group channels, then
/// restoring the result back to user-visible layout.
///
/// Note:
///   Conv3D reuses the same fused/aligned flow by mapping to an effective
///   batch (B*D), then restores the final axis order with a small D-axis
///   transpose branch.
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
///   3. Restore user-visible layout if batch exists:
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

    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op->getAttr(outputAlreadyNormalized)) {
      // Idempotence guard: previous normalize output rewrite already produced
      // internalized layout + restored visible layout.
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
    int64_t oD = 1;
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
    } else if (baseDims == 4) {
      // 3D convolution
      if (!hasBatch) {
        // [oC, oD, oH, oW]
        oC = resultType.getDimSize(0);
        oD = resultType.getDimSize(1);
        oH = resultType.getDimSize(2);
        oW = resultType.getDimSize(3);
      } else {
        // [B, oC, oD, oH, oW]
        batch = resultType.getDimSize(0);
        oC = resultType.getDimSize(1);
        oD = resultType.getDimSize(2);
        oH = resultType.getDimSize(3);
        oW = resultType.getDimSize(4);
      }
    } else {
      return failure();
    }

    int64_t oHW = oH * oW;
    // Conv3D uses an effective batch of B*oD so it can share the same
    // fused/aligned/crop flow as Conv1D/Conv2D.
    int64_t effectiveBatch = batch * oD;

    int64_t oCPerGroup = oC / groupsVal;
    int64_t oCPerGroupCeil = CEIL_FACTOR(oCPerGroup, utils::FRACTAL_BLOCK_NUM);
    int64_t oCCeil = oCPerGroupCeil * groupsVal;

    bool isOCPerGroupAligned = (oCPerGroup % utils::FRACTAL_BLOCK_NUM == 0);
    int64_t oHWCeil = CEIL_FACTOR(oHW, utils::FRACTAL_BLOCK_NUM);
    int64_t fusedOC = effectiveBatch * oC;
    int64_t fusedOCCeil = effectiveBatch * oCCeil;
    int64_t fusedGroupsVal = effectiveBatch * groupsVal;

    auto elementType = resultType.getElementType();
    Location loc = op.getLoc();

    bool useFusedConv3DBiasAdd = false;
    if constexpr (baseDims == 4) {
      if (bias) {
        auto biasType = dyn_cast<RankedTensorType>(bias.getType());
        if (!biasType || biasType.getRank() != 1 ||
            !resultType.hasStaticShape()) {
          return failure();
        }
        if (biasType.getDimSize(0) != ShapedType::kDynamic &&
            biasType.getDimSize(0) != oC) {
          return failure();
        }
        useFusedConv3DBiasAdd = true;
      }
    }

    hivm::VCastOp castOp = nullptr;
    if (convResult.hasOneUse()) {
      castOp = dyn_cast<hivm::VCastOp>(*convResult.user_begin());
    }

    bool delayCastUntilAfterFusedBias = false;
    if (useFusedConv3DBiasAdd) {
      auto biasType = cast<RankedTensorType>(bias.getType());
      Type biasAddType = elementType;
      if (castOp) {
        auto castResultType =
            mlir::cast<RankedTensorType>(castOp->getResult(0).getType());
        delayCastUntilAfterFusedBias =
            biasType.getElementType() == elementType;
        if (!delayCastUntilAfterFusedBias) {
          biasAddType = castResultType.getElementType();
        }
      }
      if (biasType.getElementType() != biasAddType) {
        return failure();
      }
    }

    SmallVector<int64_t> newShape = {oHWCeil, fusedOCCeil};
    auto newResultType = RankedTensorType::get(newShape, elementType);
    Value newEmpty =
        rewriter.create<tensor::EmptyOp>(loc, newShape, elementType);

    // === create new ConvOp with result of new shape ===
    Value convBias = useFusedConv3DBiasAdd ? Value() : bias;
    auto newConvOp = rewriter.create<ConvOpType>(
        loc,           // location
        newResultType, // result type: [oHWCeil, fusedOCCeil]
        input,         // input
        weight,        // weight
        convBias,      // bias
        newEmpty,      // init: [oHWCeil, fusedOCCeil]
        initCondition, // init condition
        ValueRange{},  // sync_related_args
        padding,       // padding attribute
        groups         // groups attribute
    );
    if (op->hasAttr(conv3dDepthPadded)) {
      // Preserve depth-padding contract across op replacement.
      newConvOp->setAttr(conv3dDepthPadded, rewriter.getUnitAttr());
    }

    Value newResult = newConvOp.getResultTensors()[0];

    // === rewrite vcast and update target if exist ===
    Value target = convResult;
    Value newTarget = newResult;

    if (castOp) {
      target = castOp->getResult(0);
      if (!delayCastUntilAfterFusedBias) {
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

        newTarget = newCastOp->getResult(0);
      }
    }

    auto newTargetType =
        mlir::cast<RankedTensorType>(newTarget.getType()).getElementType();

    // === post process to [fusedOC, oHW] ===
    bool useDirectSlicePath = isOCPerGroupAligned || fusedGroupsVal == 1;
    if (useDirectSlicePath) {
      // [oHWCeil, fusedOCCeil] -> [oHW, fusedOC] -> [fusedOC, oHW]
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

      SmallVector<int64_t> tShape{fusedOC, oHW};
      Value tInit = rewriter.create<tensor::EmptyOp>(loc, tShape, newTargetType);
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

    // === batch reshape ===
    // Conv1D/Conv2D reach this point as [B * oC, oHW]. Conv3D carries depth
    // in the leading dimension as [B * oD * oC, oHW], so it must keep that
    // shape for the dedicated D-axis restore below.
    bool needsGenericBatchRestore = hasBatch && baseDims != 4;
    if (needsGenericBatchRestore) {
      auto batchReshapedType =
          RankedTensorType::get({batch, oC, oHW}, newTargetType);

      SmallVector<ReassociationIndices> reassoc = {
          {0, 1}, // B * oC
          {2}     // oHW
      };

      newResult = rewriter.create<tensor::ExpandShapeOp>(loc, batchReshapedType,
                                                         newResult, reassoc);
    }

    // === oHW split ===
    if (baseDims == 3) {
      if (hasBatch) {
        auto finalType =
            RankedTensorType::get({batch, oC, oH, oW}, newTargetType);

        SmallVector<ReassociationIndices> reassoc = {
            {0},   // B
            {1},   // oC
            {2, 3} // oH * oW
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
    } else if (baseDims == 4) {
      // First restore to the same rank-3 channel layout used by batched Conv2D:
      // [batchLike, oC, oHW], where batchLike is B*oD or just oD.
      auto channelLayoutType =
          RankedTensorType::get({effectiveBatch, oC, oHW}, newTargetType);
      SmallVector<ReassociationIndices> channelLayoutReassoc = {{0, 1}, {2}};
      newResult = rewriter.create<tensor::ExpandShapeOp>(
          loc, channelLayoutType, newResult, channelLayoutReassoc);

      if (useFusedConv3DBiasAdd) {
        auto biasAdded = createChannelBiasAdd(
            rewriter, loc, newResult, bias, channelLayoutType.getShape(),
            /*channelDim=*/1, newTargetType);
        if (failed(biasAdded)) {
          return failure();
        }
        newResult = *biasAdded;
      }

      // Step 1: [effectiveBatch, oC, oHW] -> [effectiveBatch, oC, oH, oW]
      auto internalizedType =
          RankedTensorType::get({effectiveBatch, oC, oH, oW}, newTargetType);
      SmallVector<ReassociationIndices> internalizedReassoc = {
          {0},
          {1},
          {2, 3} // oH * oW
      };
      newResult = rewriter.create<tensor::ExpandShapeOp>(
          loc, internalizedType, newResult, internalizedReassoc);

      if (hasBatch) {
        // Step 2a: [B*oD, oC, oH, oW] -> [B, oD, oC, oH, oW]
        auto splitBatchType =
            RankedTensorType::get({batch, oD, oC, oH, oW}, newTargetType);
        SmallVector<ReassociationIndices> splitBatchReassoc = {
            {0, 1}, // B * oD
            {2},
            {3},
            {4},
        };
        newResult = rewriter.create<tensor::ExpandShapeOp>(
            loc, splitBatchType, newResult, splitBatchReassoc);

        // Step 2b: [B, oD, oC, oH, oW] -> [B, oC, oD, oH, oW]
        auto newResultType = dyn_cast<RankedTensorType>(newResult.getType());
        if (!newResultType || !newResultType.hasStaticShape()) {
          return failure();
        }
        auto shape = newResultType.getShape();
        SmallVector<int64_t> outputShape{shape[0], shape[2], shape[1],
                                         shape[3], shape[4]};
        SmallVector<int64_t> permutation{0, 2, 1, 3, 4};
        auto dst = rewriter.create<tensor::EmptyOp>(loc, outputShape,
                                                    newResultType.getElementType());
        auto vtranspose = rewriter.create<hivm::VTransposeOp>(
            loc, TypeRange{dst.getType()}, newResult, dst.getResult(),
            rewriter.getDenseI64ArrayAttr(permutation));
        newResult = vtranspose.getResult()[0];
      } else {
        // Step 2: [oD, oC, oH, oW] -> [oC, oD, oH, oW]
        auto newResultType = dyn_cast<RankedTensorType>(newResult.getType());
        if (!newResultType || !newResultType.hasStaticShape()) {
          return failure();
        }
        auto shape = newResultType.getShape();
        SmallVector<int64_t> outputShape{shape[1], shape[0], shape[2], shape[3]};
        SmallVector<int64_t> permutation{1, 0, 2, 3};
        auto dst = rewriter.create<tensor::EmptyOp>(loc, outputShape,
                                                    newResultType.getElementType());
        auto vtranspose = rewriter.create<hivm::VTransposeOp>(
            loc, TypeRange{dst.getType()}, newResult, dst.getResult(),
            rewriter.getDenseI64ArrayAttr(permutation));
        newResult = vtranspose.getResult()[0];
      }
    } else if (baseDims == 2) {
      // Conv1D already has its final [oC, oW] or [B, oC, oW] shape here.
    } else {
      return failure();
    }

    if (delayCastUntilAfterFusedBias) {
      auto castResultType =
          mlir::cast<RankedTensorType>(castOp->getResult(0).getType());
      auto castInit = rewriter.create<tensor::EmptyOp>(
          loc, castResultType.getShape(), castResultType.getElementType());
      newResult =
          rewriter
              .create<hivm::VCastOp>(
                  loc, TypeRange{castResultType}, ValueRange{newResult},
                  ValueRange{castInit.getResult()}, /*temp_buffer=*/Value(),
                  castOp.getRoundModeAttr(), hivm::TypeFnAttr{})
              ->getResult(0);
    }

    rewriter.replaceOp(target.getDefiningOp(), newResult);
    // Mark the newly-created conv op to prevent re-entering this pattern.
    newConvOp->setAttr(outputAlreadyNormalized, rewriter.getUnitAttr());
    return success();
  }
};

void populateNormalizeConvOpsPattern1(RewritePatternSet &patterns) {
  patterns.add<NormalizeConvResultTypePattern<BFloat16Type, hivm::Conv1DL1Op>>(
      patterns.getContext());
  patterns.add<NormalizeConvResultTypePattern<BFloat16Type, hivm::Conv2DL1Op>>(
      patterns.getContext());
  patterns.add<NormalizeConvResultTypePattern<BFloat16Type, hivm::Conv3DL1Op>>(
      patterns.getContext());
  patterns.add<DecomposeConvResultWithBiasPattern<Float16Type, hivm::Conv1DL1Op>>(
      patterns.getContext());
  patterns.add<DecomposeConvResultWithBiasPattern<Float16Type, hivm::Conv2DL1Op>>(
      patterns.getContext());
  patterns.add<DecomposeConvResultWithBiasPattern<Float16Type, hivm::Conv3DL1Op>>(
      patterns.getContext());
}

void populateNormalizeConvOpsPattern2(RewritePatternSet &patterns) {
  patterns.add<NormalizeConvResultTypePattern<Float16Type, hivm::Conv1DL1Op>>(
      patterns.getContext());
  patterns.add<NormalizeConvResultTypePattern<Float16Type, hivm::Conv2DL1Op>>(
      patterns.getContext());
  patterns.add<NormalizeConvResultTypePattern<Float16Type, hivm::Conv3DL1Op>>(
      patterns.getContext());
  patterns.add<DecomposeConvResultWithBiasPattern<Float32Type, hivm::Conv1DL1Op>>(
      patterns.getContext());
  patterns.add<DecomposeConvResultWithBiasPattern<Float32Type, hivm::Conv2DL1Op>>(
      patterns.getContext());
  patterns.add<DecomposeConvResultWithBiasPattern<Float32Type, hivm::Conv3DL1Op>>(
      patterns.getContext());
}

void populateNormalizeConvOpsPattern3(RewritePatternSet &patterns) {
  patterns.add<NormalizeConvInputAndWeightPattern<hivm::Conv1DL1Op>,
               NormalizeConvInputAndWeightPattern<hivm::Conv2DL1Op>,
               NormalizeConvInputAndWeightPattern<hivm::Conv3DL1Op>>(
      patterns.getContext());
  patterns.add<NormalizeConvOutputPattern<hivm::Conv1DL1Op>,
               NormalizeConvOutputPattern<hivm::Conv2DL1Op>,
               NormalizeConvOutputPattern<hivm::Conv3DL1Op>>(
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
