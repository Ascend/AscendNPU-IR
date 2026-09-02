//===- Utils.cpp - Impl. of utilities for Torch to HFusion conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TorchToHFusion/Utils.h"
#include "bishengir/Conversion/TorchToHFusion/PopulatePatterns.h"
#include "bishengir/Conversion/TorchToHFusion/TorchToHFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

void mlir::getElementwiseResultShape(OpBuilder &b, Location loc,
                                     const ValueRange tensorOperands,
                                     SmallVector<Value> &resultShape) {
  SmallVector<int64_t> operandRanks;
  operandRanks.resize(tensorOperands.size());
  llvm::transform(tensorOperands, operandRanks.begin(), [](Value tensor) {
    return dyn_cast<RankedTensorType>(tensor.getType()).getRank();
  });

  auto resultRankIt =
      std::max_element(operandRanks.begin(), operandRanks.end());
  assert(resultRankIt != operandRanks.end() && "Unable to get result rank.");
  int64_t resultRank = *resultRankIt;

  // Initialize the resultShape to all 1's, as a fallback in case
  // all sizes along that result dimension are statically 1.
  auto c1 = b.create<arith::ConstantIndexOp>(loc, /*value=*/1);
  resultShape = SmallVector<Value>(resultRank, c1);
  for (Value tensorOperand : tensorOperands) {
    SmallVector<AffineExpr> exprs;
    auto type = cast<RankedTensorType>(tensorOperand.getType());
    for (auto size :
         llvm::enumerate(makeShapeTorchCompatible(type.getShape()))) {
      if (size.value() == 1)
        continue;

      // The rank of this operand might be smaller than the overall rank of
      // the broadcast. Add an offset to correlate it to the correct
      // dimension of the result.
      auto resultDim = size.index() + (resultRank - type.getRank());

      // Now, we need to ensure that such iteration is not going to trigger
      // undefined behavior, by doing appropriate checks against the current
      // dimension size.
      auto currentDimSize = getDimOp(b, loc, tensorOperand, size.index());

      // If the result size of this dimension has so far only hit the
      // statically-known-to-be-1 case above (i.e., we have not yet assigned a
      // new Value to `resultShape[resultDim]`), then we have no other dynamic
      // values to check against, and merely need to record the current
      // dimension size.
      if (resultShape[resultDim] == c1) {
        resultShape[resultDim] = currentDimSize;
        continue;
      }
    }
  }

  return;
}

static Value
convertScalarToTensorWithFillCast(OpBuilder &b, Location loc, Value srcScalar,
                                  Type srcType, Type dstType,
                                  SmallVector<Value> &resultShape) {
  // init scalar to tensor using fill, and insert cast after fill
  Type downgradeType = srcType;
  Value downgradeScalar = srcScalar;
  auto srcFloatType = dyn_cast_or_null<mlir::FloatType>(srcType);
  auto dstFloatType = dyn_cast_or_null<mlir::FloatType>(dstType);
  if (srcFloatType && dstFloatType && srcFloatType.isF64() &&
      dstFloatType.getWidth() <= 32) {
    // extra trunc for f64
    downgradeType = b.getF32Type();
    auto trunc2f32 = b.create<arith::TruncFOp>(loc, downgradeType, srcScalar);
    downgradeScalar = trunc2f32.getOut();
  }

  SmallVector<OpFoldResult> foldShape = getAsOpFoldResult(resultShape);
  Value emptyTensorIn =
      b.create<tensor::EmptyOp>(loc, foldShape, downgradeType);
  auto fillOp = b.create<linalg::FillOp>(loc, ValueRange{downgradeScalar},
                                         ValueRange{emptyTensorIn});
  Value fillResult = fillOp->getResult(0);
  assert(isa<ShapedType>(fillResult.getType()));

  if (downgradeType == dstType) {
    return fillResult;
  }

  Value emptyTensorOut = b.create<tensor::EmptyOp>(loc, foldShape, dstType);
  hfusion::RoundMode rounding =
      mlir::utils::selectRoundMode<hfusion::RoundMode>(downgradeType, dstType);
  auto roundingAttr = b.getAttr<hfusion::RoundModeAttr>(rounding);
  auto modeAttr =
      b.getNamedAttr(hfusion::RoundModeAttr::getMnemonic(), roundingAttr);
  auto castOp = b.create<hfusion::CastOp>(loc, ValueRange{fillResult},
                                          ValueRange{emptyTensorOut}, modeAttr);
  return castOp.getResult(0);
}

Value mlir::convertScalarToTensor(OpBuilder &b, Location loc, Value scalar,
                                  SmallVector<Value> &resultShape,
                                  Type resultElementType) {
  if (isa<RankedTensorType>(scalar.getType())) {
    return scalar;
  }

  assert(!isa<TensorType>(resultElementType) &&
         "resultElementType can't be tensor type.");

  Type scalarType = scalar.getType();
  if (scalarType != resultElementType) {
    return convertScalarToTensorWithFillCast(b, loc, scalar, scalarType,
                                             resultElementType, resultShape);
  }

  // Convert scalar type.
  auto resultScalar = mlir::torch::Torch::convertScalarToDtype(
      b, loc, scalar, resultElementType);
  // Init tensor with scalar.
  return Torch::createInitTensor(b, loc, resultShape, resultElementType,
                                 resultScalar);
}

// How to use this func (for dynDims):
// If broadcastType contains dynamic dim, there are two possible cases:
// (1) this dynamic dim is inherit from input's shape.
// (2) this dynamic dim is passed by !torch.int. e.g, func
// @torch.aten.broadcast_to_4
//     in test/Conversion/TorchToHFusion/datamovement.mlir.
// If your op only contains the 1st case, you don't need to pass dynDims (by
// default it will be a empty SmallVector), as it can be handled from input
// automatically. In your op conains the 2nd case, you need to manually handle
// dynDims all and pass them through a SmallVector<Value>.
FailureOr<Value> mlir::broadcastTensorToShape(PatternRewriter &rewriter,
                                              Location loc, Value input,
                                              RankedTensorType broadcastType,
                                              SmallVector<Value> dynDims) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  auto outputShape = broadcastType.getShape();
  int64_t outputRank = outputShape.size();
  if (outputRank < inputRank)
    return rewriter.notifyMatchFailure(loc,
                                       "Output's rank need larger than input.");

  if (inputType.getElementType() != broadcastType.getElementType())
    return failure();

  SmallVector<int64_t> broadcastDims;
  SmallVector<Value> inputDynDims;
  size_t diff = outputRank - inputRank;
  for (size_t i = 0, e = outputRank; i < e; i++) {
    if (i < diff) {
      if (outputShape[i] == ShapedType::kDynamic)
        return failure();
      broadcastDims.emplace_back(i);
      continue;
    }

    size_t j = i - diff;

    if (!dynDims.size() && outputShape[i] == ShapedType::kDynamic) {
      inputDynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, j));
    }

    // Same shape, not need broadcast.
    if (outputShape[i] == inputShape[j])
      continue;

    // If shape is not 1, we don't know how to broadcast.
    if (inputShape[j] != 1)
      return failure();

    broadcastDims.emplace_back(i);
  }
  // No need broadcast.
  if (broadcastDims.empty()) {
    return input;
  }

  Value emptyTensor = rewriter.create<tensor::EmptyOp>(
      loc, outputShape, broadcastType.getElementType(),
      dynDims.size() > 0 ? dynDims : inputDynDims);

  // Squeeze first
  SmallVector<int64_t> squeezedims;
  for (auto dim : broadcastDims) {
    auto d = static_cast<int64_t>(diff);
    if (dim >= d)
      squeezedims.emplace_back(dim - d);
  }

  Value arg = input;
  if (squeezedims.size() > 0) {
    auto squeezed = squeezeDims(rewriter, loc, arg, squeezedims);
    if (failed(squeezed))
      return failure();
    arg = *squeezed;
  }

  // Create broadcast.
  auto newOp = rewriter.create<linalg::BroadcastOp>(loc, arg, emptyTensor,
                                                    broadcastDims);
  return newOp->getResult(0);
}

FailureOr<Value> mlir::unsqueezeDims(PatternRewriter &rewriter, Location loc,
                                     Value operand,
                                     SmallVector<int64_t> &dims) {
  auto operandType = cast<RankedTensorType>(operand.getType());
  if (!operandType)
    return failure();

  llvm::SmallVector<int64_t> operandShape(operandType.getShape());

  mlir::DenseSet<int64_t> dimSet(dims.begin(), dims.end());

  SmallVector<ReassociationIndices> reassociation(operandShape.size());
  int64_t idx = -1;
  if (reassociation.size() > 0) {
    for (size_t i = 0; i < operandShape.size() + dims.size(); ++i) {
      if (!dimSet.contains(i))
        ++idx;
      reassociation[std::max<int64_t>(idx, 0)].push_back(i);
    }
  }

  SmallVector<int64_t> resultShape(operandType.getShape());
  for (size_t dim : dims) {
    assert(resultShape.size() >= dim);
    resultShape.insert(resultShape.begin() + dim, 1);
  }
  auto resultType =
      RankedTensorType::get(resultShape, operandType.getElementType());
  Value result = rewriter.create<tensor::ExpandShapeOp>(loc, resultType,
                                                        operand, reassociation);

  return result;
}

FailureOr<Value> mlir::createHFusionCastOp(PatternRewriter &rewriter,
                                           Location loc, Type dtype,
                                           Value input) {
  auto inputTy = cast<RankedTensorType>(input.getType());
  if (!inputTy)
    return failure();

  auto inputDtype = inputTy.getElementType();
  if (inputDtype == dtype) {
    return input;
  }

  // BF16 <-> F16 : need cast to f32 first;
  if ((inputDtype.isBF16() && dtype.isF16()) ||
      (inputDtype.isF16() && dtype.isBF16())) {
    auto f32Type = rewriter.getF32Type();

    auto tmpF32 = createHFusionCastOp(rewriter, loc, f32Type, input);
    if (failed(tmpF32))
      return failure();

    auto result = createHFusionCastOp(rewriter, loc, dtype, *tmpF32);
    if (failed(result))
      return failure();

    return *result;
  }

  auto inputShape = inputTy.getShape();
  SmallVector<Value> outputDims;
  for (uint32_t i = 0; i < inputShape.size(); i++)
    outputDims.push_back(getDimOp(rewriter, loc, input, i));
  auto emptyOp = rewriter.create<tensor::EmptyOp>(
      loc, getAsOpFoldResult(outputDims), dtype);

  hfusion::RoundMode rounding =
      mlir::utils::selectRoundMode<hfusion::RoundMode>(inputDtype, dtype);
  auto roundingAttr = rewriter.getAttr<hfusion::RoundModeAttr>(rounding);
  auto modeAttr = rewriter.getNamedAttr(hfusion::RoundModeAttr::getMnemonic(),
                                        roundingAttr);

  auto result = rewriter
                    .create<hfusion::CastOp>(loc, ValueRange{input},
                                             ValueRange{emptyOp}, modeAttr)
                    .getResult(0);
  return result;
}

FailureOr<Value> mlir::permuteTensor(Operation *op, PatternRewriter &rewriter,
                                     Location loc,
                                     SmallVector<int64_t> dimensions,
                                     Value input) {
  auto inType = cast<RankedTensorType>(input.getType());
  int64_t inputRank = inType.getRank();
  Type elementType = inType.getElementType();

  // Check if the dimensions are a valid constants.
  int64_t numDimensions = dimensions.size();
  if (inputRank != numDimensions)
    return rewriter.notifyMatchFailure(
        op, "size of `dims` must be equal to the rank of the input");
  for (uint32_t i = 0; i < numDimensions; i++) {
    if (dimensions[i] < 0)
      dimensions[i] = toPositiveDim(dimensions[i], inputRank);
    if (!isValidDim(dimensions[i], inputRank))
      return rewriter.notifyMatchFailure(op, "dimension out of range");
  }

  // Get output
  SmallVector<Value> outputDims;
  for (uint32_t i = 0; i < inputRank; i++)
    outputDims.push_back(getDimOp(rewriter, loc, input, dimensions[i]));
  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, getAsOpFoldResult(outputDims), elementType);

  auto result =
      rewriter.create<linalg::TransposeOp>(loc, input, empty, dimensions)
          ->getResult(0);

  return result;
}

FailureOr<Value> mlir::squeezeDims(PatternRewriter &rewriter, Location loc,
                                   Value operand,
                                   SmallVector<int64_t> &dimensions) {
  auto operandTy = llvm::cast<RankedTensorType>(operand.getType());
  if (!operandTy)
    return failure();

  ArrayRef<int64_t> operandShape = operandTy.getShape();
  SmallVector<int64_t> newOperandShape;

  // Process reassociation.
  auto resultRank = operandShape.size() - dimensions.size();
  if (resultRank == 0) {
    SmallVector<ReassociationIndices> reassociation;
    auto newOperandType =
        RankedTensorType::get(newOperandShape, operandTy.getElementType());
    operand = rewriter.create<tensor::CollapseShapeOp>(loc, newOperandType,
                                                       operand, reassociation);

    return operand;
  }
  SmallVector<ReassociationIndices> reassociation(resultRank);
  DenseSet<int64_t> dimSet;
  for (auto i : dimensions) {
    assert(operandShape[i] == 1);
    dimSet.insert(i);
  }
  auto idx = -1;
  for (size_t i = 0; i < operandShape.size(); ++i) {
    if (!dimSet.contains(i)) {
      ++idx;
      newOperandShape.push_back(operandShape[i]);
    }
    reassociation[std::max<int64_t>(idx, 0)].push_back(i);
  }

  auto newOperandType =
      RankedTensorType::get(newOperandShape, operandTy.getElementType());
  operand = rewriter.create<tensor::CollapseShapeOp>(loc, newOperandType,
                                                     operand, reassociation);

  return operand;
}

FailureOr<Type> mlir::getPromotionType(Operation *op, Type lhsDtype,
                                       Type rhsDtype) {
  if (!lhsDtype.isIntOrFloat() || !rhsDtype.isIntOrFloat()) {
    op->emitWarning() << "unsupported type" << lhsDtype << " or " << rhsDtype
                      << " , now support int and float";
    return failure();
  }

  if (isa<mlir::FloatType>(lhsDtype) && isa<mlir::IntegerType>(rhsDtype)) {
    return lhsDtype;
  } else if (isa<mlir::IntegerType>(lhsDtype) &&
             isa<mlir::FloatType>(rhsDtype)) {
    return rhsDtype;
  } else {
    // Both are either Integer or Float types, but the bit width might be
    // different.
    if (lhsDtype.getIntOrFloatBitWidth() > rhsDtype.getIntOrFloatBitWidth()) {
      return lhsDtype;
    } else {
      return rhsDtype;
    }
  }
}

LogicalResult
mlir::broadcastToGivenShape(Operation *op, PatternRewriter &rewriter,
                            Value input, SmallVector<Value> broadcastToShape,
                            RankedTensorType broadcastType,
                            bool ensureNoImplicitBroadcast, Value &result,
                            SmallVector<bool> useBroadcastToShape) {
  RankedTensorType inputType = cast<RankedTensorType>(input.getType());
  int64_t inputRank = inputType.getRank();
  int64_t outputRank = broadcastToShape.size();
  ArrayRef<int64_t> outputShape = broadcastType.getShape();
  SmallVector<int64_t> inputShape =
      makeShapeTorchCompatible(inputType.getShape());
  if (outputRank < inputRank) {
    return rewriter.notifyMatchFailure(
        op, "invalid shape: broadcastToShape size must not be smaller than the "
            "size of the input shape");
  }

  Type elementType = inputType.getElementType();
  Location loc = op->getLoc();
  SmallVector<OpFoldResult> outShape;
  // Vector indicating broadcasted status, `true` means needed broadcasted.
  SmallVector<bool> broadcastedStatus;

  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
  Value oneIndex =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

  size_t diff = outputRank - inputRank;
  for (size_t i = 0, e = outputRank; i < e; i++) {
    Value shapeValue = broadcastToShape[i];
    size_t j = i - diff;
    bool isDynamic = i >= diff && inputShape[j] == kUnknownSize;

    // Inherit static output shapes if present.
    if (outputShape[i] != ShapedType::kDynamic) {
      outShape.push_back(rewriter.getIndexAttr(outputShape[i]));
      if (i < diff) {
        if (outputShape[i] < 0) {
          return rewriter.notifyMatchFailure(
              op, "invalid shape: negative values not allowed in new broadcast "
                  "dimensions");
        }
        continue;
      }
      if (isDynamic) {
        return rewriter.notifyMatchFailure(
            op, "invalid shape: input shape should not be dynamic while output "
                "shape is static");
      }
      if (inputShape[j] != outputShape[i] && inputShape[j] != 1) {
        return rewriter.notifyMatchFailure(
            op, "invalid shape: static mismatch in input and output broadcast "
                "shapes");
      }

      // If strict symbolic shapes are assumed and the input shape is dynamic,
      // we can assume that dim is not broadcasted.
      broadcastedStatus.push_back(inputShape[j] != outputShape[i] &&
                                  !isDynamic);
      continue;
    }

    if (i < diff) {
      if (!ensureNoImplicitBroadcast) {
        Value isValid = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, shapeValue, zero);
        rewriter.create<cf::AssertOp>(
            loc, isValid,
            rewriter.getStringAttr(
                "negative values not allowed in new dimensions"));
      }
      outShape.push_back(castIntToIndex(rewriter, loc, shapeValue));
      continue;
    }

    if (inputShape[j] == 1) {
      // Broadcast singleton dimension
      if (!ensureNoImplicitBroadcast) {
        Value isNegative = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, shapeValue, zero);
        Value select = rewriter.create<arith::SelectOp>(
            loc, isNegative, oneIndex,
            castIntToIndex(rewriter, loc, shapeValue));
        outShape.push_back(select);
      } else {
        outShape.push_back(castIntToIndex(rewriter, loc, shapeValue));
      }
      broadcastedStatus.push_back(true);
      continue;
    }

    if (!ensureNoImplicitBroadcast && !useBroadcastToShape.empty() &&
        useBroadcastToShape[j]) {
      // For some cases when size[i] != 1:
      // (1) static => dynamic
      // (2) dynamic => dynamic
      return rewriter.notifyMatchFailure(
          op, "cannot detect how to broadcast without dynamo guard.");
    }
    // Dynamo will guard to ensure case ? -> ? is not broadcasted.
    Value dim = getDimOp(rewriter, loc, input, j);
    broadcastedStatus.push_back(false);
    outShape.push_back(dim);
  }

  Value outTensor =
      rewriter.create<tensor::EmptyOp>(loc, outShape, elementType);

  if (!llvm::any_of(broadcastedStatus, [](bool b) { return b; }) &&
      inputRank == outputRank) {
    result = rewriter.create<tensor::CastOp>(loc, outTensor.getType(), input);
    return success();
  }

  SmallVector<int64_t> broadcastDims;
  SmallVector<int64_t> squeezedims;
  for (int64_t i = 0; i < outputRank; ++i) {
    if (i < static_cast<int64_t>(diff)) {
      broadcastDims.push_back(i);
    } else {
      if (broadcastedStatus[i - diff]) {
        broadcastDims.push_back(i);
        squeezedims.push_back(i - diff);
      }
    }
  }

  Value arg = input;
  if (!squeezedims.empty()) {
    auto squeezed = squeezeDims(rewriter, loc, arg, squeezedims);
    if (failed(squeezed))
      return failure();
    arg = *squeezed;
  }
  result =
      rewriter.create<linalg::BroadcastOp>(loc, arg, outTensor, broadcastDims)
          ->getResult(0);
  return success();
}
