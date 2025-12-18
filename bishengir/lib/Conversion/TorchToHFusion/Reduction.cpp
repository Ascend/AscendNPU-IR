//===- Reduction.cpp - Conversion impl. for Reduction Ops -----------------===//
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

#include "bishengir/Conversion/TorchToHFusion/PopulatePatterns.h"
#include "bishengir/Conversion/TorchToHFusion/TorchToHFusion.h"
#include "bishengir/Conversion/TorchToHFusion/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"

#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

template <typename OpTy>
static TypedAttr getMaxMinInitValue(Operation *op, Type elementTy,
                                    OpBuilder &rewriter) {
  OpTy atenOp = dyn_cast<OpTy>(op);
  bool isMax = isa<AtenMaxOp, AtenMaxDimOp>(op);

  if (isa<mlir::FloatType>(elementTy))
    return rewriter.getFloatAttr(
        elementTy,
        APFloat::getInf(cast<mlir::FloatType>(elementTy).getFloatSemantics(),
                        /*Negative=*/isMax));

  bool isUnsigned = false;
  auto integerTy = dyn_cast<mlir::IntegerType>(
      cast<BaseTensorType>(atenOp.getSelf().getType()).getDtype());
  isUnsigned = integerTy.isUnsigned();
  TypedAttr fillValue;
  if (!isUnsigned) {
    auto width = cast<mlir::IntegerType>(elementTy).getWidth();
    auto init = isMax ? APSInt::getSignedMinValue(width)
                      : APSInt::getSignedMaxValue(width);
    fillValue = rewriter.getIntegerAttr(elementTy, init);
  } else {
    auto width = cast<mlir::IntegerType>(elementTy).getWidth();
    auto init = isMax ? APInt::getMinValue(width) : APInt::getMaxValue(width);
    fillValue = rewriter.getIntegerAttr(elementTy, init);
  }
  return fillValue;
}

static TypedAttr createInitialValueForReduceOp(Operation *op, Type elementTy,
                                               OpBuilder &rewriter) {
  if (isa<AtenSumOp, AtenSumDimIntListOp>(op))
    return rewriter.getZeroAttr(elementTy);

  if (isa<AtenProdOp, AtenProdDimIntOp>(op)) {
    if (isa<mlir::FloatType>(elementTy))
      return rewriter.getFloatAttr(elementTy, 1.0);
    else if (isa<mlir::IntegerType>(elementTy))
      return rewriter.getIntegerAttr(elementTy, 1);
  }

  if (isa<AtenAnyOp, AtenAnyDimOp, AtenAnyDimsOp>(op))
    return rewriter.getBoolAttr(false);

  if (isa<AtenAllOp, AtenAllDimOp>(op))
    return rewriter.getBoolAttr(true);

  if (isa<AtenMaxOp>(op) && elementTy.isIntOrFloat())
    return getMaxMinInitValue<AtenMaxOp>(op, elementTy, rewriter);

  if (isa<AtenMinOp>(op) && elementTy.isIntOrFloat())
    return getMaxMinInitValue<AtenMinOp>(op, elementTy, rewriter);

  if (isa<AtenMaxDimOp>(op) && elementTy.isIntOrFloat())
    return getMaxMinInitValue<AtenMaxDimOp>(op, elementTy, rewriter);

  if (isa<AtenMinDimOp>(op) && elementTy.isIntOrFloat())
    return getMaxMinInitValue<AtenMinDimOp>(op, elementTy, rewriter);

  op->emitError("unimplemented init value for reduce op");
  return {};
}

namespace {
template <typename OpTy>
class ConvertAtenMinMaxDimOp : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpConversionPattern<OpTy>::getTypeConverter;

  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static_assert(std::is_same<OpTy, AtenMaxDimOp>() ||
                  std::is_same<OpTy, AtenMinDimOp>());
    constexpr bool isMax = std::is_same<OpTy, AtenMaxDimOp>();
    const llvm::StringRef opName = op->getName().getStringRef();

    Location loc = op.getLoc();
    Value input = adaptor.getSelf();
    auto typec = this->getTypeConverter();
    auto valResultType =
        cast<RankedTensorType>(typec->convertType(op.getResult(0).getType()));
    auto idxResultType =
        cast<RankedTensorType>(typec->convertType(op.getResult(1).getType()));
    Type idxElementType =
        getElementTypeOrSelf(typec->convertType(idxResultType));
    if (!isa<IntegerType>(idxElementType))
      return rewriter.notifyMatchFailure(
          op, opName + " to linalg.* requires integer-like result type");

    bool keepDim = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim)))
      return rewriter.notifyMatchFailure(
          op, opName + " requires boolean value for keepdim");

    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(
          op, opName + " to linalg.* requires int value for Dim");

    RankedTensorType inputType = cast<RankedTensorType>(input.getType());
    dim = toPositiveDim(dim, inputType.getRank());
    if (!isValidDim(dim, inputType.getRank()))
      return rewriter.notifyMatchFailure(op, "dim is not a valid dim");

    Type inElementType = inputType.getElementType();
    if (!isa<mlir::FloatType>(inElementType) &&
        !isa<mlir::IntegerType>(inElementType)) {
      return rewriter.notifyMatchFailure(
          op, opName + " to linalg.* requires Float or Integer "
                       "input element type");
    }

    SmallVector<Value> resultShape;
    for (uint32_t i = 0; i < inputType.getRank(); i++) {
      if (i == dim)
        continue;
      resultShape.push_back(getDimOp(rewriter, loc, input, i));
    }

    Value filledTensorIdx =
        createZeroInitTensor(rewriter, loc, resultShape, idxElementType);

    // Second fill the output buffer for the running max or min.
    Value initTensorVal = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(resultShape), inElementType);

    auto fillValueAttr =
        createInitialValueForReduceOp(op, inElementType, rewriter);
    if (!fillValueAttr)
      return rewriter.notifyMatchFailure(op,
                                         opName + "create init value failed");
    Value fillValue = rewriter.create<arith::ConstantOp>(loc, fillValueAttr);
    Value filledTensorVal =
        rewriter.create<linalg::FillOp>(loc, fillValue, initTensorVal).result();

    SmallVector<int64_t> dims = {dim};
    hfusion::ReduceWithIndexKind reduceKind;
    if (isMax) {
      reduceKind = hfusion::ReduceWithIndexKind::MAX;
    } else {
      reduceKind = hfusion::ReduceWithIndexKind::MIN;
    }

    auto reduceKindAttr = mlir::hfusion::ReduceWithIndexKindAttr::get(
        rewriter.getContext(), reduceKind);
    // TODO: get "tile_break_left" value instead of setting it to "true"
    auto hfusionOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        loc, TypeRange{filledTensorVal.getType(), filledTensorIdx.getType()},
        /*input*/ ValueRange{input},
        /*outputValue&Index*/ ValueRange{filledTensorVal, filledTensorIdx},
        reduceKindAttr, BoolAttr::get(rewriter.getContext(), true), dims);

    if (!keepDim) {
      Value rVal = rewriter.create<tensor::CastOp>(loc, valResultType,
                                                   hfusionOp.getResult()[0]);
      Value rIdx = rewriter.create<tensor::CastOp>(loc, idxResultType,
                                                   hfusionOp.getResult()[1]);
      llvm::SmallVector<Value> res{rVal, rIdx};
      rewriter.replaceOp(op, res);
      return success();
    }

    auto rVal = unsqueezeDims(rewriter, loc, hfusionOp.getResult()[0], dims);
    if (failed(rVal))
      return op.emitError("expand dims for val failed");
    auto rIdx = unsqueezeDims(rewriter, loc, hfusionOp.getResult()[0], dims);
    if (failed(rIdx))
      return op.emitError("expand dims for idx failed");

    llvm::SmallVector<Value> unsqueezes = {*rVal, *rIdx};
    rewriter.replaceOp(op, unsqueezes);
    return success();
  }
};

} // namespace

/* LCOV_EXCL_START */
static Value createLinalgPayloadForReduceOp(OpBuilder &b, Location loc,
                                            ValueRange payloadArgs,
                                            Operation *op,
                                            ArrayRef<Value> operands,
                                            Type resultElementType) {
  if (isa<AtenSumOp, AtenSumDimIntListOp>(op)) {
    Value self =
        convertScalarToDtype(b, loc, payloadArgs[0], resultElementType);
    Value result = payloadArgs[1];
    if (isa<mlir::FloatType>(resultElementType))
      return b.create<arith::AddFOp>(loc, self, result);
    else if (isa<mlir::IntegerType>(resultElementType))
      return b.create<arith::AddIOp>(loc, self, result);
  } else if (isa<AtenProdOp, AtenProdDimIntOp>(op)) {
    Value self =
        convertScalarToDtype(b, loc, payloadArgs[0], resultElementType);
    Value result = payloadArgs[1];
    if (isa<mlir::FloatType>(resultElementType))
      return b.create<arith::MulFOp>(loc, self, result);
    else if (isa<mlir::IntegerType>(resultElementType))
      return b.create<arith::MulIOp>(loc, self, result);
  } else if (auto max = dyn_cast<AtenMaxOp>(op)) {
    Value self =
        convertScalarToDtype(b, loc, payloadArgs[0], resultElementType);
    Value result = payloadArgs[1];
    if (isa<mlir::FloatType>(resultElementType))
      return b.create<arith::MaximumFOp>(loc, self, result);
    else if (isa<mlir::IntegerType>(resultElementType)) {
      IntegerType intType = dyn_cast<mlir::IntegerType>(
          cast<BaseTensorType>(max.getSelf().getType()).getDtype());
      if (intType.isUnsigned())
        return b.create<arith::MaxUIOp>(loc, self, result);
      if (intType.isSigned())
        return b.create<arith::MaxSIOp>(loc, self, result);
    }
  } else if (auto min = dyn_cast<AtenMinOp>(op)) {
    Value self =
        convertScalarToDtype(b, loc, payloadArgs[0], resultElementType);
    Value result = payloadArgs[1];
    if (isa<mlir::FloatType>(resultElementType))
      return b.create<arith::MinimumFOp>(loc, self, result);
    else if (isa<mlir::IntegerType>(resultElementType)) {
      IntegerType intType = dyn_cast<mlir::IntegerType>(
          cast<BaseTensorType>(min.getSelf().getType()).getDtype());
      if (intType.isUnsigned())
        return b.create<arith::MinUIOp>(loc, self, result);
      if (intType.isSigned())
        return b.create<arith::MinSIOp>(loc, self, result);
    }
  } else if (isa<AtenAllOp, AtenAllDimOp>(op)) {
    Value elem = payloadArgs[0];
    Value result = payloadArgs[1];
    Value self = convertScalarToDtype(b, loc, elem, resultElementType);
    return b.create<arith::AndIOp>(loc, self, result);
  } else if (isa<AtenAnyOp, AtenAnyDimOp, AtenAnyDimsOp>(op)) {
    Value elem = payloadArgs[0];
    Value result = payloadArgs[1];
    Value self = convertScalarToDtype(b, loc, elem, resultElementType);
    return b.create<arith::OrIOp>(loc, self, result);
  }
  op->emitError("unimplemented lowering in createLinalgPayloadForReduceOp");
  return nullptr;
}
/* LCOV_EXCL_STOP */

namespace {
template <typename... AllowOps>
class ConvertReductionOp : public ConversionPattern {
public:
  ConvertReductionOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}
  virtual ~ConvertReductionOp() = default;
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<AllowOps...>(op))
      return rewriter.notifyMatchFailure(op, "not an allowed reduce op.");

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return op->emitError(
          "invalid operand or result types to use with linalg on tensors");

    FailureOr<torch_to_linalg::ReductionOpInfo> opInfo =
        computeReductionOpInfo(op, operands, rewriter);
    if (failed(opInfo))
      return opInfo;

    Location loc = op->getLoc();
    auto resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    Type elemType = resultType.getElementType();
    LogicalResult elemTypeCheck =
        validateReductionElementType(op, elemType, rewriter);
    if (failed(elemTypeCheck))
      return elemTypeCheck;

    Value reduceOp =
        createReductionOp(loc, elemType, op, operands, *opInfo, rewriter);
    if (!reduceOp)
      return op->emitError(
          "failed to create linalg.reduce operation for reduction");

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, reduceOp);
    return success();
  }

private:
  /// Given a reduction operation that has the `keepdim` attribute and the
  /// (optional) `dim` attribute, return the source tensor operand and the
  /// literal values of the attributes or failure otherwise
  /* LCOV_EXCL_START */
  Value createInitTensorForReduceOp(
      OpBuilder &rewriter, Location loc, Operation *op, Type elementType,
      const torch_to_linalg::ReductionOpInfo &opInfo) const {
    auto inputTy = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getOperand(0).getType()));
    auto resultTy = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    auto elementTy = resultTy.getElementType();
    Value input = opInfo.tensorOperand;

    SmallVector<int64_t> reduceShape;
    SmallVector<Value> dynDims;
    for (unsigned i = 0; i < inputTy.getRank(); i++) {
      if (!opInfo.dimSet.contains(i)) {
        reduceShape.push_back(inputTy.getDimSize(i));
        if (inputTy.isDynamicDim(i))
          dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    // First fill the output buffer with the init value.
    auto emptyTensor =
        rewriter
            .create<tensor::EmptyOp>(loc, reduceShape,
                                     resultTy.getElementType(), dynDims)
            .getResult();

    auto fillValueAttr = createInitialValueForReduceOp(op, elementTy, rewriter);
    if (!fillValueAttr)
      return nullptr;

    auto fillValue = rewriter.create<arith::ConstantOp>(loc, fillValueAttr);
    auto filledTensor = rewriter
                            .create<linalg::FillOp>(loc, ValueRange{fillValue},
                                                    ValueRange{emptyTensor})
                            .result();
    return filledTensor;
  }

  template <typename T>
  FailureOr<torch_to_linalg::ReductionOpInfo>
  computeReductionOpInfoForDimVariantOp(
      T op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const {
    auto opInfo = torch_to_linalg::ReductionOpInfo{false, Value{}, {}};
    typename T::Adaptor adaptor(operands);
    opInfo.tensorOperand = adaptor.getSelf();
    auto inputType = cast<RankedTensorType>(opInfo.tensorOperand.getType());

    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&opInfo.keepDim)))
      return op.emitError("`keepdim` must be a constant bool");

    SmallVector<int64_t> dimList;
    int64_t dim;
    bool isNoneOrEmptyDimList = isa<Torch::NoneType>(op.getDim().getType());
    if (matchPattern(op.getDim(), m_TorchListOfConstantInts(dimList))) {
      // Fix negative dimensions, if any, before adding to the list.
      for (int64_t dim : dimList) {
        dim = toPositiveDim(dim, inputType.getRank());
        // Drop invalid dimensions
        if (isValidDim(dim, inputType.getRank()))
          opInfo.dimSet.insert(dim);
      }
      if (dimList.empty())
        isNoneOrEmptyDimList = true;
    } else if (matchPattern(op.getDim(), m_TorchConstantInt(&dim))) {
      dim = toPositiveDim(dim, inputType.getRank());
      if (!isValidDim(dim, inputType.getRank()))
        return op.emitError("`dim` argument must be valid, invalid received.");
      opInfo.dimSet.insert(dim);
    } else if (!isNoneOrEmptyDimList) {
      return op.emitError("`dim` argument must be a constant int list or None");
    }
    if (isNoneOrEmptyDimList) {
      // If no dimensions were specified, reduce along all dimensions
      for (int64_t i = 0; i < inputType.getRank(); i++)
        opInfo.dimSet.insert(i);
    }

    return opInfo;
  }

  FailureOr<torch_to_linalg::ReductionOpInfo>
  computeReductionOpInfo(Operation *op, ArrayRef<Value> operands,
                         ConversionPatternRewriter &rewriter) const {
    auto opInfo = torch_to_linalg::ReductionOpInfo{false, Value{}, {}};

    if (isa<AtenAnyOp, AtenAllOp, AtenMaxOp, AtenMinOp, AtenSumOp, AtenProdOp,
            AtenNormScalarOp>(op)) {
      opInfo.tensorOperand = operands[0];
      auto inputType = cast<RankedTensorType>(opInfo.tensorOperand.getType());

      // `AtenAny`, `AtenAll`, `AtenSumOp`, `AtenProdOp`, `AtenMaxOp`, and
      // `AtenMinOp` each reduce along all the dimensions of the input tensor.
      for (int64_t i = 0; i < inputType.getRank(); i++)
        opInfo.dimSet.insert(i);

      return opInfo;
    }

    FailureOr<torch_to_linalg::ReductionOpInfo> info;
#define COMPUTE_REDUCETION_INFO(Aten)                                          \
  if (auto atenOp = dyn_cast<Aten>(op)) {                                      \
    info = computeReductionOpInfoForDimVariantOp(atenOp, operands, rewriter);  \
  }

    COMPUTE_REDUCETION_INFO(AtenSumDimIntListOp);
    COMPUTE_REDUCETION_INFO(AtenProdDimIntOp);
    COMPUTE_REDUCETION_INFO(AtenAllDimOp);
    COMPUTE_REDUCETION_INFO(AtenAnyDimOp);
    COMPUTE_REDUCETION_INFO(AtenAnyDimsOp);

#undef COMPUTE_REDUCETION_INFO
    return info;
  }

  LogicalResult
  validateReductionElementType(Operation *op, Type elemType,
                               ConversionPatternRewriter &rewriter) const {
    if ((isa<AtenLinalgVectorNormOp>(op) || isa<AtenFrobeniusNormDimOp>(op) ||
         isa<AtenNormScalarOp>(op)) &&
        !isa<mlir::FloatType>(elemType))
      return op->emitError("only float types are valid for vector norm ops");
    if (isa<AtenAllDimOp>(op) && isa<mlir::IntegerType>(elemType) &&
        elemType.getIntOrFloatBitWidth() == 8)
      return op->emitError("uint8 is not supported");

    // No checks for all other reduction operations
    return success();
  }
  /* LCOV_EXCL_STOP */

  Value createReductionOp(Location loc, Type elemType, Operation *op,
                          ArrayRef<Value> operands,
                          const torch_to_linalg::ReductionOpInfo &opInfo,
                          ConversionPatternRewriter &rewriter) const {
    bool err = false;
    auto reductionBodyBuilder = [&](OpBuilder &builder, Location loc,
                                    ValueRange payloadArgs) {
      Value result = createLinalgPayloadForReduceOp(builder, loc, payloadArgs,
                                                    op, operands, elemType);
      if (result)
        builder.create<linalg::YieldOp>(loc, result);
      err = !result;
    };

    Value initElem =
        createInitTensorForReduceOp(rewriter, loc, op, elemType, opInfo);
    if (!initElem)
      return nullptr;

    SmallVector<int64_t> dims;
    auto inputType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getOperand(0).getType()));
    for (int64_t i = 0; i < inputType.getRank(); i++) {
      if (opInfo.dimSet.contains(i))
        dims.push_back(i);
    }

    auto reduce =
        rewriter
            .create<linalg::ReduceOp>(loc, opInfo.tensorOperand, initElem, dims,
                                      reductionBodyBuilder)
            .getResult(0);

    if (err)
      return nullptr;
    if (!opInfo.keepDim)
      return reduce;

    auto expandResult = unsqueezeDims(rewriter, loc, reduce, dims);
    if (failed(expandResult))
      return nullptr;
    return *expandResult;
  }
};
} // namespace

void mlir::populateReductionPatternsAndLegality(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns,
                                                ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

#define reduceAllowedOps                                                       \
  AtenSumOp, AtenAnyOp, AtenAllOp, AtenSumDimIntListOp, AtenProdOp,            \
      AtenProdDimIntOp, AtenMaxOp, AtenMinOp, AtenAnyDimOp, AtenAllDimOp,      \
      AtenAnyDimsOp

  target.addIllegalOp<reduceAllowedOps>();
  patterns.add<ConvertReductionOp<reduceAllowedOps>>(typeConverter, context);
#undef reduceAllowedOps

  target.addIllegalOp<AtenMaxDimOp>();
  patterns.add<ConvertAtenMinMaxDimOp<AtenMaxDimOp>>(typeConverter, context);
  target.addIllegalOp<AtenMinDimOp>();
  patterns.add<ConvertAtenMinMaxDimOp<AtenMinDimOp>>(typeConverter, context);
}
