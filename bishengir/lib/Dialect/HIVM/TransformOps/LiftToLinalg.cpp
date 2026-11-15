//===- LiftToLinalg.cpp - Lift HIVM vector functions to linalg ------------===//
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

#include "bishengir/Dialect/HIVM/TransformOps/LiftToLinalg.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Interfaces/VectorizableOpInterface.h"
#include "bishengir/Dialect/HIVM/TransformOps/Partition.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::transform;

namespace {
struct LiftContext {
  std::string error;
};

struct ElementwiseForm {
  ValueRange inputs;
  OperandRange outputs;
  SmallVector<AffineMap> indexingMaps;
  unsigned rank;
};

std::string getOpName(Operation *op) {
  return op->getName().getStringRef().str();
}

void reportUnsupported(Operation *op, const Twine &reason, std::string &error) {
  error = (Twine("cannot lift '") + getOpName(op) + "': " + reason).str();
}

template <typename MathOp> struct UnaryMathEmitter {
  static constexpr unsigned numInputs = 1;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    return {builder.create<MathOp>(loc, inputs.front()).getResult()};
  }
};

struct AddEmitter {
  static constexpr unsigned numInputs = 2;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    Type type = getElementTypeOrSelf(inputs[0].getType());
    if (isa<FloatType>(type))
      return {
          builder.create<arith::AddFOp>(loc, inputs[0], inputs[1]).getResult()};
    return {
        builder.create<arith::AddIOp>(loc, inputs[0], inputs[1]).getResult()};
  }
};

struct MulEmitter {
  static constexpr unsigned numInputs = 2;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    Type type = getElementTypeOrSelf(inputs[0].getType());
    if (isa<FloatType>(type))
      return {
          builder.create<arith::MulFOp>(loc, inputs[0], inputs[1]).getResult()};
    return {
        builder.create<arith::MulIOp>(loc, inputs[0], inputs[1]).getResult()};
  }
};

struct SubEmitter {
  static constexpr unsigned numInputs = 2;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    if (isa<FloatType>(inputs[0].getType()))
      return {builder.create<arith::SubFOp>(loc, inputs[0], inputs[1])};
    return {builder.create<arith::SubIOp>(loc, inputs[0], inputs[1])};
  }
};

struct SelectEmitter {
  static constexpr unsigned numInputs = 3;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    auto condition = inputs[0];
    if (!condition.getType().isInteger(1)) {
      auto zero = builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(condition.getType(), 0));
      condition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                condition, zero);
    }
    return {
        builder.create<arith::SelectOp>(loc, condition, inputs[1], inputs[2])};
  }
};

struct IdentityEmitter {
  SmallVector<Value> operator()(OpBuilder &, Location,
                                ValueRange inputs) const {
    return {inputs.front()};
  }
};

struct ReciprocalEmitter {
  static constexpr unsigned numInputs = 1;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    auto type = inputs.front().getType();
    auto one =
        builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(type, 1.0));
    return {builder.create<arith::DivFOp>(loc, one, inputs.front())};
  }
};

template <typename SourceOp, typename FloatOp, typename SignedOp,
          typename UnsignedOp>
struct MinMaxEmitter {
  SourceOp operation;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) {
    if (isa<FloatType>(inputs[0].getType()))
      return {builder.create<FloatOp>(loc, inputs[0], inputs[1])};
    if (operation.getIsSigned())
      return {builder.create<SignedOp>(loc, inputs[0], inputs[1])};
    return {builder.create<UnsignedOp>(loc, inputs[0], inputs[1])};
  }
};

using MinEmitter =
    MinMaxEmitter<VMinOp, arith::MinimumFOp, arith::MinSIOp, arith::MinUIOp>;
using MaxEmitter =
    MinMaxEmitter<VMaxOp, arith::MaximumFOp, arith::MaxSIOp, arith::MaxUIOp>;

struct NotEmitter {
  static constexpr unsigned numInputs = 1;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    auto type = inputs.front().getType();
    auto bits = builder.getIntegerType(type.getIntOrFloatBitWidth());
    Value input = inputs.front();
    if (isa<FloatType>(type))
      input = builder.create<arith::BitcastOp>(loc, bits, input);
    auto ones = builder.create<arith::ConstantOp>(
        loc, builder.getIntegerAttr(bits, APInt::getAllOnes(bits.getWidth())));
    Value result = builder.create<arith::XOrIOp>(loc, input, ones);
    if (isa<FloatType>(type))
      result = builder.create<arith::BitcastOp>(loc, type, result);
    return {result};
  }
};

struct AbsEmitter {
  static constexpr unsigned numInputs = 1;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    Value input = inputs.front();
    if (isa<FloatType>(input.getType()))
      return {builder.create<math::AbsFOp>(loc, input)};
    return {builder.create<math::AbsIOp>(loc, input)};
  }
};

struct ReluEmitter {
  static constexpr unsigned numInputs = 1;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    Type type = inputs.front().getType();
    auto zero =
        builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(type));
    if (isa<FloatType>(type))
      return {builder.create<arith::MaximumFOp>(loc, inputs.front(), zero)};
    return {builder.create<arith::MaxSIOp>(loc, inputs.front(), zero)};
  }
};

struct ShlEmitter {
  static constexpr unsigned numInputs = 2;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    return {builder.create<arith::ShLIOp>(loc, inputs[0], inputs[1])};
  }
};

struct ShrEmitter {
  static constexpr unsigned numInputs = 2;
  static constexpr unsigned numResults = 1;

  VShROp operation;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) {
    if (operation.getIsSigned())
      return {builder.create<arith::ShRSIOp>(loc, inputs[0], inputs[1])};
    return {builder.create<arith::ShRUIOp>(loc, inputs[0], inputs[1])};
  }
};

struct DivEmitter {
  VDivOp operation;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) {
    if (isa<FloatType>(inputs[0].getType()))
      return {builder.create<arith::DivFOp>(loc, inputs[0], inputs[1])};
    if (!operation.getIsSigned())
      return {builder.create<arith::DivUIOp>(loc, inputs[0], inputs[1])};
    return {builder.create<arith::DivSIOp>(loc, inputs[0], inputs[1])};
  }
};

struct CompareEmitter {
  VCmpOp operation;

  arith::CmpIPredicate integerPredicate() {
    bool isSigned = operation.getIsSigned();
    switch (operation.getCompareMode()) {
    case CompareMode::EQ:
      return arith::CmpIPredicate::eq;
    case CompareMode::NE:
      return arith::CmpIPredicate::ne;
    case CompareMode::LT:
      return isSigned ? arith::CmpIPredicate::slt : arith::CmpIPredicate::ult;
    case CompareMode::LE:
      return isSigned ? arith::CmpIPredicate::sle : arith::CmpIPredicate::ule;
    case CompareMode::GT:
      return isSigned ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::ugt;
    case CompareMode::GE:
      return isSigned ? arith::CmpIPredicate::sge : arith::CmpIPredicate::uge;
    }
    llvm_unreachable("unknown HIVM comparison mode");
  }

  arith::CmpFPredicate floatPredicate() {
    switch (operation.getCompareMode()) {
    case CompareMode::EQ:
      return arith::CmpFPredicate::OEQ;
    case CompareMode::NE:
      return arith::CmpFPredicate::UNE;
    case CompareMode::LT:
      return arith::CmpFPredicate::OLT;
    case CompareMode::LE:
      return arith::CmpFPredicate::OLE;
    case CompareMode::GT:
      return arith::CmpFPredicate::OGT;
    case CompareMode::GE:
      return arith::CmpFPredicate::OGE;
    }
    llvm_unreachable("unknown HIVM comparison mode");
  }

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) {
    Value result;
    if (isa<FloatType>(inputs[0].getType()))
      result = builder.create<arith::CmpFOp>(loc, floatPredicate(), inputs[0],
                                             inputs[1]);
    else
      result = builder.create<arith::CmpIOp>(loc, integerPredicate(), inputs[0],
                                             inputs[1]);
    auto outputType =
        getElementTypeOrSelf(operation.getDst().front().getType());
    if (!outputType.isInteger(1))
      result = builder.create<arith::ExtUIOp>(loc, outputType, result);
    return {result};
  }
};

template <typename ScalarOp> struct BitwiseEmitter {
  static constexpr unsigned numInputs = 2;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    Type type = inputs[0].getType();
    if (!isa<FloatType>(type))
      return {builder.create<ScalarOp>(loc, inputs[0], inputs[1])};
    Type integerType = builder.getIntegerType(type.getIntOrFloatBitWidth());
    Value lhs = builder.create<arith::BitcastOp>(loc, integerType, inputs[0]);
    Value rhs = builder.create<arith::BitcastOp>(loc, integerType, inputs[1]);
    Value result = builder.create<ScalarOp>(loc, lhs, rhs);
    return {builder.create<arith::BitcastOp>(loc, type, result)};
  }
};

struct MulSIExtendedEmitter {
  static constexpr unsigned numInputs = 2;
  static constexpr unsigned numResults = 2;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    SmallVector<Type> resultTypes(2, inputs[0].getType());
    auto mul = builder.create<arith::MulSIExtendedOp>(
        loc, TypeRange(resultTypes), inputs[0], inputs[1]);
    return {mul.getLow(), mul.getHigh()};
  }
};

struct MulUIExtendedEmitter {
  static constexpr unsigned numInputs = 2;
  static constexpr unsigned numResults = 2;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    SmallVector<Type> resultTypes(2, inputs[0].getType());
    auto mul = builder.create<arith::MulUIExtendedOp>(
        loc, TypeRange(resultTypes), inputs[0], inputs[1]);
    return {mul.getLow(), mul.getHigh()};
  }
};

struct ModEmitter {
  static constexpr unsigned numInputs = 2;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    Type type = getElementTypeOrSelf(inputs[0].getType());
    if (isa<FloatType>(type))
      return {
          builder.create<arith::RemFOp>(loc, inputs[0], inputs[1]).getResult()};
    return {
        builder.create<arith::RemSIOp>(loc, inputs[0], inputs[1]).getResult()};
  }
};

struct ModUIEmitter {
  static constexpr unsigned numInputs = 2;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    return {
        builder.create<arith::RemUIOp>(loc, inputs[0], inputs[1]).getResult()};
  }
};

struct IlogbEmitter {
  static constexpr unsigned numInputs = 1;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    Value absolute = builder.create<math::AbsFOp>(loc, inputs.front());
    Value logarithm = builder.create<math::Log2Op>(loc, absolute);
    Value floored = builder.create<math::FloorOp>(loc, logarithm);
    return {floored};
  }
};

struct IsInfEmitter {
  static constexpr unsigned numInputs = 1;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    Value input = inputs.front();
    auto floatType = cast<FloatType>(getElementTypeOrSelf(input.getType()));
    APFloat infinity = APFloat::getInf(floatType.getFloatSemantics());
    Value inf = builder.create<arith::ConstantOp>(
        loc, builder.getFloatAttr(floatType, infinity));
    Value absolute = builder.create<math::AbsFOp>(loc, input);
    Value isInf = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                                absolute, inf);
    return {isInf};
  }
};

struct IsNanEmitter {
  static constexpr unsigned numInputs = 1;
  static constexpr unsigned numResults = 1;

  SmallVector<Value> operator()(OpBuilder &builder, Location loc,
                                ValueRange inputs) const {
    Value input = inputs.front();
    Value isNan = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO,
                                                input, input);
    return {isNan};
  }
};

struct InlineLayout {
  ArrayRef<int64_t> broadcast;
  ArrayRef<int64_t> transpose;
};

InlineLayout getInlineLayout(Operation *op) {
  auto broadcast = op->getAttrOfType<DenseI64ArrayAttr>("broadcast");
  auto transpose = op->getAttrOfType<DenseI64ArrayAttr>("transpose");
  return {broadcast ? broadcast.asArrayRef() : ArrayRef<int64_t>(),
          transpose ? transpose.asArrayRef() : ArrayRef<int64_t>()};
}

FailureOr<ElementwiseForm> getElementwiseOperands(Operation *op,
                                                  unsigned expectedInputs,
                                                  unsigned expectedOutputs,
                                                  std::string &error) {
  auto destinationStyle = dyn_cast<DestinationStyleOpInterface>(op);
  if (!destinationStyle) {
    reportUnsupported(op, "expected a destination-style operation", error);
    return failure();
  }

  OperandRange outputs = destinationStyle.getDpsInits();
  if (outputs.size() != expectedOutputs) {
    error = (Twine("cannot lift '") + getOpName(op) + "': expected " +
             Twine(expectedOutputs) + " outputs, got " + Twine(outputs.size()))
                .str();
    return failure();
  }

  unsigned firstOutput = outputs.getBeginOperandIndex();
  ValueRange inputs = op->getOperands().take_front(firstOutput);
  if (inputs.size() != expectedInputs) {
    error = (Twine("cannot lift '") + getOpName(op) + "': expected " +
             Twine(expectedInputs) + " inputs, got " + Twine(inputs.size()))
                .str();
    return failure();
  }
  if (firstOutput + outputs.size() != op->getNumOperands()) {
    reportUnsupported(op, "unexpected extra operands", error);
    return failure();
  }

  auto firstOutputType = dyn_cast<RankedTensorType>(outputs.front().getType());
  if (!firstOutputType) {
    reportUnsupported(op, "requires tensor results", error);
    return failure();
  }
  unsigned rank = firstOutputType.getRank();
  for (Value output : outputs) {
    auto tensor = dyn_cast<RankedTensorType>(output.getType());
    if (!tensor) {
      reportUnsupported(op, "requires tensor results", error);
      return failure();
    }
    if (tensor.getRank() != rank) {
      reportUnsupported(op, "output ranks must match", error);
      return failure();
    }
  }
  if (op->getNumResults() != outputs.size()) {
    error =
        (Twine("cannot lift '") + getOpName(op) + "': expected " +
         Twine(outputs.size()) + " results, got " + Twine(op->getNumResults()))
            .str();
    return failure();
  }
  for (Value input : inputs) {
    auto shaped = dyn_cast<ShapedType>(input.getType());
    if (!shaped)
      continue;
    if (!isa<RankedTensorType>(shaped)) {
      reportUnsupported(op, "requires tensor or scalar inputs", error);
      return failure();
    }
    if (shaped.getRank() != rank) {
      reportUnsupported(op, "input and output ranks must match", error);
      return failure();
    }
  }

  return ElementwiseForm{inputs, outputs, {}, rank};
}

LogicalResult checkBroadcastShapes(Operation *op, const ElementwiseForm &form,
                                   ArrayRef<int64_t> broadcastAxes,
                                   std::string &error) {
  auto outputType = cast<RankedTensorType>(form.outputs.front().getType());
  for (int64_t axis : broadcastAxes) {
    if (axis < 0 || static_cast<unsigned>(axis) >= form.rank) {
      reportUnsupported(op, "broadcast axis is outside the op rank", error);
      return failure();
    }
    for (Value input : form.inputs) {
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      if (!inputType)
        continue;
      int64_t inputDim = inputType.getDimSize(axis);
      int64_t outputDim = outputType.getDimSize(axis);
      if (inputDim == 1 || inputDim == outputDim)
        continue;
      if (ShapedType::isDynamic(inputDim) || ShapedType::isDynamic(outputDim)) {
        reportUnsupported(op, "dynamic inline broadcast is not supported",
                          error);
        return failure();
      }
      reportUnsupported(op, "broadcast dim must be 1 or match the output",
                        error);
      return failure();
    }
  }
  return success();
}

LogicalResult checkTransposeLayout(Operation *op, const ElementwiseForm &form,
                                   ArrayRef<int64_t> permutation,
                                   std::string &error) {
  if (permutation.size() != form.rank) {
    reportUnsupported(op, "transpose size must match the op rank", error);
    return failure();
  }
  SmallVector<bool> seen(form.rank, false);
  for (int64_t value : permutation) {
    if (value < 0 || static_cast<unsigned>(value) >= form.rank || seen[value]) {
      reportUnsupported(op, "transpose must be a permutation", error);
      return failure();
    }
    seen[value] = true;
  }

  auto outputType = cast<RankedTensorType>(form.outputs.front().getType());
  for (Value input : form.inputs) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      continue;
    for (unsigned dim = 0; dim < form.rank; ++dim) {
      int64_t inputDim = inputType.getDimSize(permutation[dim]);
      int64_t outputDim = outputType.getDimSize(dim);
      if (inputDim == outputDim || ShapedType::isDynamic(inputDim) ||
          ShapedType::isDynamic(outputDim))
        continue;
      reportUnsupported(op, "transpose shapes do not match", error);
      return failure();
    }
  }
  return success();
}

AffineMap buildIdentityInputMap(Value input, unsigned rank,
                                ArrayRef<int64_t> broadcastAxes,
                                MLIRContext *context) {
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<AffineExpr> expressions;
  for (unsigned dim = 0; dim < rank; ++dim) {
    if (llvm::is_contained(broadcastAxes, static_cast<int64_t>(dim)) &&
        inputType.getDimSize(dim) == 1)
      expressions.push_back(getAffineConstantExpr(0, context));
    else
      expressions.push_back(getAffineDimExpr(dim, context));
  }
  return AffineMap::get(rank, 0, expressions, context);
}

AffineMap buildTransposeInputMap(ArrayRef<int64_t> permutation, unsigned rank,
                                 MLIRContext *context) {
  SmallVector<AffineExpr> expressions(rank);
  for (unsigned dim = 0; dim < rank; ++dim)
    expressions[permutation[dim]] = getAffineDimExpr(dim, context);
  return AffineMap::get(rank, 0, expressions, context);
}

SmallVector<AffineMap> buildElementwiseMaps(const ElementwiseForm &form,
                                            Operation *op,
                                            ArrayRef<int64_t> broadcastAxes,
                                            ArrayRef<int64_t> permutation) {
  MLIRContext *context = op->getContext();
  SmallVector<AffineMap> maps;
  for (Value input : form.inputs) {
    if (!isa<ShapedType>(input.getType())) {
      maps.push_back(AffineMap::get(form.rank, 0, context));
      continue;
    }
    if (!permutation.empty())
      maps.push_back(buildTransposeInputMap(permutation, form.rank, context));
    else
      maps.push_back(
          buildIdentityInputMap(input, form.rank, broadcastAxes, context));
  }
  for (unsigned index = 0; index < form.outputs.size(); ++index)
    maps.push_back(AffineMap::getMultiDimIdentityMap(form.rank, context));
  return maps;
}

LogicalResult buildElementwiseIndexingMaps(Operation *op, ElementwiseForm &form,
                                           std::string &error) {
  InlineLayout layout = getInlineLayout(op);
  auto hivmOp = cast<HIVMStructuredOp>(op);
  if (!layout.broadcast.empty() && !hivmOp.isInlineBroadcastable()) {
    reportUnsupported(op, "op does not support inline broadcast", error);
    return failure();
  }
  if (!layout.transpose.empty() && !hivmOp.isInlineTransposable()) {
    reportUnsupported(op, "op does not support inline transpose", error);
    return failure();
  }
  if (!layout.broadcast.empty() && !layout.transpose.empty()) {
    reportUnsupported(op, "inline broadcast and transpose cannot be combined",
                      error);
    return failure();
  }

  if (!layout.transpose.empty() &&
      failed(checkTransposeLayout(op, form, layout.transpose, error)))
    return failure();
  if (!layout.broadcast.empty() &&
      failed(checkBroadcastShapes(op, form, layout.broadcast, error)))
    return failure();

  form.indexingMaps =
      buildElementwiseMaps(form, op, layout.broadcast, layout.transpose);
  return success();
}

FailureOr<ElementwiseForm> inspectElementwiseForm(Operation *op,
                                                  unsigned expectedInputs,
                                                  unsigned expectedOutputs,
                                                  std::string &error) {
  auto form =
      getElementwiseOperands(op, expectedInputs, expectedOutputs, error);
  if (failed(form))
    return failure();
  if (failed(buildElementwiseIndexingMaps(op, *form, error)))
    return failure();
  return *form;
}

template <typename EmitterT>
linalg::GenericOp createGeneric(Operation *op, const ElementwiseForm &form,
                                EmitterT emitter, PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  SmallVector<Type> resultTypes;
  for (Value output : form.outputs)
    resultTypes.push_back(output.getType());
  SmallVector<utils::IteratorType> iteratorTypes(form.rank,
                                                 utils::IteratorType::parallel);

  return rewriter.create<linalg::GenericOp>(
      loc, TypeRange(resultTypes), form.inputs, form.outputs, form.indexingMaps,
      iteratorTypes,
      [&](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange arguments) {
        ValueRange inputArguments = arguments.take_front(form.inputs.size());
        SmallVector<Value> yielded =
            emitter(bodyBuilder, bodyLoc, inputArguments);
        bodyBuilder.create<linalg::YieldOp>(bodyLoc, yielded);
      });
}

template <typename EmitterT>
LogicalResult liftElementwiseOp(Operation *op, PatternRewriter &rewriter,
                                std::string &error) {
  auto form = inspectElementwiseForm(op, EmitterT::numInputs,
                                     EmitterT::numResults, error);
  if (failed(form))
    return failure();

  rewriter.setInsertionPoint(op);
  linalg::GenericOp generic = createGeneric(op, *form, EmitterT{}, rewriter);
  rewriter.replaceOp(op, generic.getResults());
  return success();
}

class ShrLiftPattern final : public OpRewritePattern<VShROp> {
public:
  ShrLiftPattern(MLIRContext *context, LiftContext &liftContext)
      : OpRewritePattern<VShROp>(context), liftContext(liftContext) {}

  LogicalResult matchAndRewrite(VShROp op,
                                PatternRewriter &rewriter) const override {
    if (op.getRound())
      return rewriter.notifyMatchFailure(
          op, "rounded shift right is not supported");
    std::string error;
    auto form = inspectElementwiseForm(op, 2, 1, error);
    if (failed(form)) {
      liftContext.error = std::move(error);
      return rewriter.notifyMatchFailure(op, liftContext.error);
    }
    auto generic = createGeneric(op, *form, ShrEmitter{op}, rewriter);
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }

private:
  LiftContext &liftContext;
};

template <typename SourceOp, typename EmitterT>
class ElementwiseLiftPattern final : public OpRewritePattern<SourceOp> {
public:
  ElementwiseLiftPattern(MLIRContext *context, LiftContext &liftContext)
      : OpRewritePattern<SourceOp>(context), liftContext(liftContext) {}

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    std::string error;
    if (failed(
            liftElementwiseOp<EmitterT>(op.getOperation(), rewriter, error))) {
      liftContext.error = std::move(error);
      return rewriter.notifyMatchFailure(op, liftContext.error);
    }
    return success();
  }

private:
  LiftContext &liftContext;
};

template <typename SourceOp, typename EmitterT>
class AttributedElementwiseLiftPattern final
    : public OpRewritePattern<SourceOp> {
public:
  AttributedElementwiseLiftPattern(MLIRContext *context,
                                   LiftContext &liftContext)
      : OpRewritePattern<SourceOp>(context), liftContext(liftContext) {}

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    auto form = inspectElementwiseForm(op, 2, 1, liftContext.error);
    if (failed(form))
      return failure();
    auto generic = createGeneric(op, *form, EmitterT{op}, rewriter);
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }

private:
  LiftContext &liftContext;
};

class CastLiftPattern final : public OpRewritePattern<VCastOp> {
public:
  CastLiftPattern(MLIRContext *context, LiftContext &liftContext)
      : OpRewritePattern<VCastOp>(context), liftContext(liftContext) {}

  LogicalResult matchAndRewrite(VCastOp op,
                                PatternRewriter &rewriter) const override {
    auto form = inspectElementwiseForm(op, 1, 1, liftContext.error);
    if (failed(form))
      return failure();
    auto round =
        hfusion::symbolizeRoundMode(stringifyRoundMode(op.getRoundMode()));
    auto kind = hfusion::symbolizeTypeFn(stringifyTypeFn(op.getCast()));
    if (!round || !kind)
      return failure();
    SmallVector<NamedAttribute> attrs{
        rewriter.getNamedAttr("round_mode",
                              rewriter.getAttr<hfusion::RoundModeAttr>(*round)),
        rewriter.getNamedAttr("cast",
                              rewriter.getAttr<hfusion::TypeFnAttr>(*kind))};
    for (auto name : {"enable_overflow", "enable_saturate"})
      if (auto attr = op->getAttr(name))
        attrs.push_back(rewriter.getNamedAttr(name, attr));
    if (auto mode =
            op->getAttrOfType<UnsignedModeAttr>(UnsignedModeAttr::name)) {
      auto value = hfusion::symbolizeUnsignedMode(
          stringifyUnsignedMode(mode.getValue()));
      if (!value)
        return failure();
      attrs.push_back(rewriter.getNamedAttr(
          "unsigned_mode",
          rewriter.getAttr<hfusion::UnsignedModeAttr>(*value)));
    }
    auto castOp = rewriter.create<hfusion::CastOp>(op.getLoc(), form->inputs,
                                                   form->outputs, attrs);
    auto generic = linalg::generalizeNamedOp(rewriter, castOp);
    if (failed(generic))
      return failure();
    generic->setIndexingMapsAttr(
        rewriter.getAffineMapArrayAttr(form->indexingMaps));
    rewriter.replaceOp(op, generic->getResults());
    return success();
  }

private:
  LiftContext &liftContext;
};

class BroadcastLiftPattern final : public OpRewritePattern<VBrcOp> {
public:
  BroadcastLiftPattern(MLIRContext *context, LiftContext &liftContext)
      : OpRewritePattern<VBrcOp>(context), liftContext(liftContext) {}

  LogicalResult matchAndRewrite(VBrcOp op,
                                PatternRewriter &rewriter) const override {
    auto form = getElementwiseOperands(op, 1, 1, liftContext.error);
    if (failed(form) ||
        failed(checkBroadcastShapes(op, *form, op.getBroadcastDims(),
                                    liftContext.error)))
      return failure();
    form->indexingMaps =
        buildElementwiseMaps(*form, op, op.getBroadcastDims(), {});
    auto generic = createGeneric(op, *form, IdentityEmitter{}, rewriter);
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }

private:
  LiftContext &liftContext;
};

class ArangeLiftPattern final : public OpRewritePattern<VArangeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(VArangeOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getDst().getType());
    if (!type)
      return rewriter.notifyMatchFailure(op, "requires tensor output");
    auto rank = type.getRank();
    auto map = AffineMap::getMultiDimIdentityMap(rank, op.getContext());
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(
        op.getLoc(), TypeRange{type}, ValueRange{}, ValueRange{op.getDst()},
        ArrayRef<AffineMap>{map}, iterators,
        [&](OpBuilder &builder, Location loc, ValueRange) {
          Value value = op.getOffset();
          if (!value)
            value = builder.create<arith::ConstantIndexOp>(loc, 0);
          for (auto [axis, stride] : llvm::enumerate(op.getStrides())) {
            auto index = builder.create<linalg::IndexOp>(loc, axis);
            auto scaled = builder.create<arith::MulIOp>(loc, index, stride);
            value = builder.create<arith::AddIOp>(loc, value, scaled);
          }
          auto element = type.getElementType();
          if (isa<FloatType>(element)) {
            value = builder.create<arith::IndexCastOp>(
                loc, builder.getI64Type(), value);
            value = builder.create<arith::SIToFPOp>(loc, element, value);
          } else {
            value = builder.create<arith::IndexCastOp>(loc, element, value);
          }
          builder.create<linalg::YieldOp>(loc, value);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

SmallVector<ReassociationIndices>
reductionReassociation(int64_t rank, ArrayRef<int64_t> axes) {
  SmallVector<ReassociationIndices> groups;
  ReassociationIndices pending;
  for (int64_t axis = 0; axis < rank; ++axis) {
    pending.push_back(axis);
    if (!llvm::is_contained(axes, axis)) {
      groups.push_back(std::move(pending));
      pending.clear();
    }
  }
  if (!groups.empty())
    llvm::append_range(groups.back(), pending);
  return groups;
}

constexpr llvm::StringLiteral kScalarizeReductionAttr =
    "hivm.vectorize.scalarize";

bool hasNativeVectorReduction(ReduceOperation kind) {
  return kind == ReduceOperation::sum || kind == ReduceOperation::min ||
         kind == ReduceOperation::max || kind == ReduceOperation::xori;
}

ReduceOperation getValueReductionKind(VReduceOp op) {
  switch (op.getArith().getReduceOp()) {
  case ReduceOperation::min_with_index_left:
  case ReduceOperation::min_with_index_right:
    return ReduceOperation::min;
  case ReduceOperation::max_with_index_left:
  case ReduceOperation::max_with_index_right:
    return ReduceOperation::max;
  default:
    return op.getArith().getReduceOp();
  }
}

TypedAttr getReductionIdentity(VReduceOp op, OpBuilder &builder) {
  auto type = getElementTypeOrSelf(op.getSrc().getType());
  auto kind = getValueReductionKind(op);
  if (auto floating = dyn_cast<FloatType>(type)) {
    if (kind == ReduceOperation::min || kind == ReduceOperation::max)
      return builder.getFloatAttr(
          type, APFloat::getInf(floating.getFloatSemantics(),
                                kind == ReduceOperation::max));
    return builder.getFloatAttr(type,
                                kind == ReduceOperation::prod ? 1.0 : 0.0);
  }
  auto width = type.getIntOrFloatBitWidth();
  APInt value(width, kind == ReduceOperation::prod ? 1 : 0);
  if (kind == ReduceOperation::andi || kind == ReduceOperation::all)
    value = APInt::getAllOnes(width);
  else if (kind == ReduceOperation::min)
    value = op.getUnsignedSrc() ? APInt::getAllOnes(width)
                                : APInt::getSignedMaxValue(width);
  else if (kind == ReduceOperation::max && !op.getUnsignedSrc())
    value = APInt::getSignedMinValue(width);
  return builder.getIntegerAttr(type, value);
}

Value createReductionCombiner(VReduceOp op, OpBuilder &builder, Location loc,
                              Value lhs, Value rhs) {
  auto floating = isa<FloatType>(lhs.getType());
  auto unsignedSource = op.getUnsignedSrc();
  switch (getValueReductionKind(op)) {
  case ReduceOperation::sum:
    if (floating)
      return builder.create<arith::AddFOp>(loc, lhs, rhs);
    return builder.create<arith::AddIOp>(loc, lhs, rhs);
  case ReduceOperation::prod:
    if (floating)
      return builder.create<arith::MulFOp>(loc, lhs, rhs);
    return builder.create<arith::MulIOp>(loc, lhs, rhs);
  case ReduceOperation::min:
    if (floating)
      return builder.create<arith::MinimumFOp>(loc, lhs, rhs);
    if (unsignedSource)
      return builder.create<arith::MinUIOp>(loc, lhs, rhs);
    return builder.create<arith::MinSIOp>(loc, lhs, rhs);
  case ReduceOperation::max:
    if (floating)
      return builder.create<arith::MaximumFOp>(loc, lhs, rhs);
    if (unsignedSource)
      return builder.create<arith::MaxUIOp>(loc, lhs, rhs);
    return builder.create<arith::MaxSIOp>(loc, lhs, rhs);
  case ReduceOperation::all:
  case ReduceOperation::andi:
    return builder.create<arith::AndIOp>(loc, lhs, rhs);
  case ReduceOperation::any:
  case ReduceOperation::ori:
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  case ReduceOperation::xori:
    return builder.create<arith::XOrIOp>(loc, lhs, rhs);
  default:
    llvm_unreachable("unvalidated reduction kind");
  }
}

Value createFilledTensor(OpBuilder &builder, Location loc, Value destination,
                         TypedAttr identity) {
  auto initial = builder.create<arith::ConstantOp>(loc, identity);
  return builder
      .create<linalg::FillOp>(loc, ValueRange{initial}, ValueRange{destination})
      .getResult(0);
}

Value createValueReduction(VReduceOp op, OpBuilder &builder,
                           ArrayRef<int64_t> axes,
                           ArrayRef<ReassociationIndices> reassociation) {
  auto loc = op.getLoc();
  Value destination = builder.create<tensor::CollapseShapeOp>(
      loc, op.getDstValue(), reassociation);
  destination = createFilledTensor(builder, loc, destination,
                                   getReductionIdentity(op, builder));
  auto reduction = builder.create<linalg::ReduceOp>(
      loc, ValueRange{op.getSrc()}, ValueRange{destination}, axes,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange arguments) {
        auto value = createReductionCombiner(op, nested, nestedLoc,
                                             arguments[0], arguments[1]);
        nested.create<linalg::YieldOp>(nestedLoc, value);
      });
  if (!hasNativeVectorReduction(getValueReductionKind(op)))
    reduction->setAttr(kScalarizeReductionAttr, builder.getUnitAttr());
  return reduction.getResult(0);
}

bool isLeftIndexReduction(VReduceOp op) {
  auto kind = op.getArith().getReduceOp();
  return kind == ReduceOperation::min_with_index_left ||
         kind == ReduceOperation::max_with_index_left;
}

Value matchesReducedValue(OpBuilder &builder, Location loc, Value input,
                          Value reduced) {
  if (!isa<FloatType>(input.getType()))
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, input,
                                         reduced);
  auto equal = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                             input, reduced);
  auto inputNan = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO,
                                                input, input);
  auto reducedNan = builder.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::UNO, reduced, reduced);
  auto bothNan = builder.create<arith::AndIOp>(loc, inputNan, reducedNan);
  return builder.create<arith::OrIOp>(loc, equal, bothNan);
}

Value createIndexCandidates(VReduceOp op, OpBuilder &builder, Value reduced) {
  auto loc = op.getLoc();
  auto sourceType = cast<RankedTensorType>(op.getSrc().getType());
  auto rank = sourceType.getRank();
  auto identity = builder.getMultiDimIdentityMap(rank);
  SmallVector<AffineExpr> dimensions;
  for (int64_t axis = 0; axis < rank; ++axis)
    if (axis != op.getReduceDims().front())
      dimensions.push_back(builder.getAffineDimExpr(axis));
  auto projected = AffineMap::get(rank, 0, dimensions, op.getContext());
  SmallVector<Value> dynamicSizes;
  for (int64_t axis = 0; axis < rank; ++axis)
    if (sourceType.isDynamicDim(axis))
      dynamicSizes.push_back(
          builder.create<tensor::DimOp>(loc, op.getSrc(), axis));
  auto empty = builder.create<tensor::EmptyOp>(
      loc, sourceType.getShape(), builder.getI32Type(), dynamicSizes);
  SmallVector<Value> inputs{op.getSrc(), reduced};
  SmallVector<AffineMap> maps{identity, projected};
  maps.push_back(identity);
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);
  auto generic = builder.create<linalg::GenericOp>(
      loc, TypeRange{empty.getType()}, inputs, ValueRange{empty}, maps,
      iterators,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange arguments) {
        auto match =
            matchesReducedValue(nested, nestedLoc, arguments[0], arguments[1]);
        auto position = nested.create<linalg::IndexOp>(
            nestedLoc, op.getReduceDims().front());
        auto index = nested.create<arith::IndexCastOp>(
            nestedLoc, nested.getI32Type(), position);
        auto sentinel = nested.create<arith::ConstantIntOp>(
            nestedLoc, isLeftIndexReduction(op) ? INT32_MAX : INT32_MIN, 32);
        auto selected =
            nested.create<arith::SelectOp>(nestedLoc, match, index, sentinel);
        nested.create<linalg::YieldOp>(nestedLoc, selected.getResult());
      });
  return generic.getResult(0);
}

Value createIndexReduction(VReduceOp op, OpBuilder &builder, Value reduced,
                           ArrayRef<int64_t> axes,
                           ArrayRef<ReassociationIndices> reassociation) {
  auto loc = op.getLoc();
  auto candidates = createIndexCandidates(op, builder, reduced);
  Value destination = builder.create<tensor::CollapseShapeOp>(
      loc, op.getDstIndex(), reassociation);
  destination =
      createFilledTensor(builder, loc, destination,
                         builder.getI32IntegerAttr(
                             isLeftIndexReduction(op) ? INT32_MAX : INT32_MIN));
  auto reduction = builder.create<linalg::ReduceOp>(
      loc, ValueRange{candidates}, ValueRange{destination}, axes,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange arguments) {
        Value value;
        if (isLeftIndexReduction(op))
          value = nested.create<arith::MinSIOp>(nestedLoc, arguments[0],
                                                arguments[1]);
        else
          value = nested.create<arith::MaxSIOp>(nestedLoc, arguments[0],
                                                arguments[1]);
        nested.create<linalg::YieldOp>(nestedLoc, value);
      });
  return reduction.getResult(0);
}

Value mapReductionIndices(VReduceOp op, OpBuilder &builder, Value positions) {
  if (!op.getIndices())
    return positions;
  auto type = cast<RankedTensorType>(positions.getType());
  SmallVector<Value> dynamicSizes;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    if (type.isDynamicDim(axis))
      dynamicSizes.push_back(
          builder.create<tensor::DimOp>(op.getLoc(), positions, axis));
  return builder.create<tensor::GenerateOp>(
      op.getLoc(), type, dynamicSizes,
      [&](OpBuilder &nested, Location loc, ValueRange coordinates) {
        auto position =
            nested.create<tensor::ExtractOp>(loc, positions, coordinates);
        auto index = nested.create<arith::IndexCastOp>(
            loc, nested.getIndexType(), position);
        auto inputCoordinates = llvm::to_vector(coordinates);
        inputCoordinates[op.getReduceDims().front()] = index;
        auto value = nested.create<tensor::ExtractOp>(loc, op.getIndices(),
                                                      inputCoordinates);
        nested.create<tensor::YieldOp>(loc, value.getResult());
      });
}

class ReduceLiftPattern final : public OpRewritePattern<VReduceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(VReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(op, "requires tensor reduction");
    auto kind = getValueReductionKind(op);
    if (!llvm::is_contained(
            {ReduceOperation::sum, ReduceOperation::prod, ReduceOperation::min,
             ReduceOperation::max, ReduceOperation::andi, ReduceOperation::ori,
             ReduceOperation::xori, ReduceOperation::all, ReduceOperation::any},
            kind))
      return rewriter.notifyMatchFailure(op, "unsupported reduction kind");
    auto type = cast<RankedTensorType>(op.getDstValue().getType());
    auto axes = llvm::to_vector(op.getReduceDims());
    llvm::sort(axes);
    auto reassociation = reductionReassociation(type.getRank(), axes);
    auto value = createValueReduction(op, rewriter, axes, reassociation);
    SmallVector<Value> results{rewriter.create<tensor::ExpandShapeOp>(
        op.getLoc(), type, value, reassociation)};
    if (VReduceOp::isWithIndex(op.getArith().getReduceOp())) {
      auto index =
          createIndexReduction(op, rewriter, value, axes, reassociation);
      Value expanded = rewriter.create<tensor::ExpandShapeOp>(
          op.getLoc(), op.getDstIndex().getType(), index, reassociation);
      results.push_back(mapReductionIndices(op, rewriter, expanded));
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

void populateLiftPatterns(RewritePatternSet &patterns, LiftContext &context) {
  patterns.add<ArangeLiftPattern, ReduceLiftPattern>(patterns.getContext());
  patterns.add<CastLiftPattern, BroadcastLiftPattern,
               AttributedElementwiseLiftPattern<VDivOp, DivEmitter>,
               AttributedElementwiseLiftPattern<VMinOp, MinEmitter>,
               AttributedElementwiseLiftPattern<VMaxOp, MaxEmitter>,
               AttributedElementwiseLiftPattern<VCmpOp, CompareEmitter>>(
      patterns.getContext(), context);
  patterns
      .add<ElementwiseLiftPattern<VExpOp, UnaryMathEmitter<math::ExpOp>>,
           ElementwiseLiftPattern<VLnOp, UnaryMathEmitter<math::LogOp>>,
           ElementwiseLiftPattern<VSqrtOp, UnaryMathEmitter<math::SqrtOp>>,
           ElementwiseLiftPattern<VRsqrtOp, UnaryMathEmitter<math::RsqrtOp>>,
           ElementwiseLiftPattern<VRecOp, ReciprocalEmitter>,
           ElementwiseLiftPattern<VNotOp, NotEmitter>,
           ElementwiseLiftPattern<VAbsOp, AbsEmitter>,
           ElementwiseLiftPattern<VReluOp, ReluEmitter>,
           ElementwiseLiftPattern<VShLOp, ShlEmitter>, ShrLiftPattern,
           ElementwiseLiftPattern<VSubOp, SubEmitter>,
           ElementwiseLiftPattern<VSelOp, SelectEmitter>,
           ElementwiseLiftPattern<VOrOp, BitwiseEmitter<arith::OrIOp>>,
           ElementwiseLiftPattern<VAndOp, BitwiseEmitter<arith::AndIOp>>,
           ElementwiseLiftPattern<VXorOp, BitwiseEmitter<arith::XOrIOp>>,
           ElementwiseLiftPattern<VExp2Op, UnaryMathEmitter<math::Exp2Op>>,
           ElementwiseLiftPattern<VLog2Op, UnaryMathEmitter<math::Log2Op>>,
           ElementwiseLiftPattern<VLog10Op, UnaryMathEmitter<math::Log10Op>>,
           ElementwiseLiftPattern<VLog1pOp, UnaryMathEmitter<math::Log1pOp>>,
           ElementwiseLiftPattern<VExpM1Op, UnaryMathEmitter<math::ExpM1Op>>,
           ElementwiseLiftPattern<VIlogbOp, IlogbEmitter>,
           ElementwiseLiftPattern<VTanhOp, UnaryMathEmitter<math::TanhOp>>,
           ElementwiseLiftPattern<VSinOp, UnaryMathEmitter<math::SinOp>>,
           ElementwiseLiftPattern<VCosOp, UnaryMathEmitter<math::CosOp>>,
           ElementwiseLiftPattern<VAtanOp, UnaryMathEmitter<math::AtanOp>>,
           ElementwiseLiftPattern<VTanOp, UnaryMathEmitter<math::TanOp>>,
           ElementwiseLiftPattern<VErfOp, UnaryMathEmitter<math::ErfOp>>,
           ElementwiseLiftPattern<VAddOp, AddEmitter>,
           ElementwiseLiftPattern<VMulOp, MulEmitter>,
           ElementwiseLiftPattern<VMulExtOp, MulSIExtendedEmitter>,
           ElementwiseLiftPattern<VMulExtUiOp, MulUIExtendedEmitter>,
           ElementwiseLiftPattern<VModOp, ModEmitter>,
           ElementwiseLiftPattern<VModUIOp, ModUIEmitter>,
           ElementwiseLiftPattern<VLdexpOp, MulEmitter>,
           ElementwiseLiftPattern<VIsInfOp, IsInfEmitter>,
           ElementwiseLiftPattern<VIsNanOp, IsNanEmitter>>(
          patterns.getContext(), context);
}

void collectVectorizableOps(func::FuncOp function,
                            SmallVectorImpl<Operation *> &targets) {
  function.walk([&](Operation *op) {
    if (isVectorizationCompute(*op))
      targets.push_back(op);
  });
}

LogicalResult verifyNoUnliftedCompute(func::FuncOp function,
                                      std::string &error) {
  WalkResult walk = function.walk([&](Operation *op) {
    if (!isVectorizationCompute(*op))
      return WalkResult::advance();
    error = "no lifting pattern for '" + getOpName(op) + "'";
    return WalkResult::interrupt();
  });
  return walk.wasInterrupted() ? failure() : success();
}

LogicalResult liftFunction(func::FuncOp function, PatternApplicator &applicator,
                           PatternRewriter &rewriter, LiftContext &context) {
  context.error.clear();
  SmallVector<Operation *> targets;
  collectVectorizableOps(function, targets);

  for (Operation *target : targets) {
    if (failed(applicator.matchAndRewrite(target, rewriter))) {
      if (context.error.empty())
        context.error = "no lifting pattern for '" + getOpName(target) + "'";
      return failure();
    }
  }

  if (failed(verifyNoUnliftedCompute(function, context.error)))
    return failure();

  if (failed(verify(function))) {
    context.error = "verification failed after lifting '" +
                    function.getSymName().str() + "'";
    return failure();
  }
  return success();
}

} // namespace

LogicalResult
mlir::hivm::liftFunctionsToLinalg(ArrayRef<func::FuncOp> functions,
                                  std::string &error) {
  error.clear();
  if (functions.empty())
    return success();

  func::FuncOp first = functions.front();
  MLIRContext *context = first.getContext();
  LiftContext liftContext;
  RewritePatternSet patterns(context);
  populateLiftPatterns(patterns, liftContext);

  FrozenRewritePatternSet frozen(std::move(patterns));
  PatternApplicator applicator(frozen);
  applicator.applyDefaultCostModel();
  PatternRewriter rewriter(context);

  for (func::FuncOp function : functions) {
    if (failed(liftFunction(function, applicator, rewriter, liftContext))) {
      error = liftContext.error;
      return failure();
    }
  }
  return success();
}

DiagnosedSilenceableFailure HIVMLiftToLinalgOp::apply(TransformRewriter &,
                                                      TransformResults &results,
                                                      TransformState &state) {
  auto payloadOps = state.getPayloadOps(getVfs());
  SmallVector<func::FuncOp> functions;
  SmallVector<Operation *> lifted;
  for (Operation *payloadOp : payloadOps) {
    auto function = dyn_cast<func::FuncOp>(payloadOp);
    if (!function)
      return emitDefiniteFailure() << "expected only function operations";
    functions.push_back(function);
    lifted.push_back(payloadOp);
  }

  std::string error;
  if (failed(liftFunctionsToLinalg(functions, error)))
    return emitDefiniteFailure() << error;

  results.set(cast<OpResult>(getResult()), lifted);
  return DiagnosedSilenceableFailure::success();
}

void HIVMLiftToLinalgOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getVfsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}
