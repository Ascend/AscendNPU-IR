//===- NormalizeMath.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/HFusion/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"

namespace mlir::hfusion {

/// normalize logb(x) to ln(x) / ln(b) when log base b is not e
/// eg.
/// y = hfusion elemwise unary {log2} (x)
///  is normalized to
///  y = linalg.elemwise_unary {log}(x) / linalg.elemwise_unary {log}(2)
struct NormalizeLogLikeOp : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    auto hfusionFun = op.getFun();
    if (hfusionFun != hfusion::UnaryFn::log2 &&
        hfusionFun != hfusion::UnaryFn::log10) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");

    Value input = op.getDpsInputs()[0];
    Value output = op.getOutputs()[0];
    if (inType.isF16()) {
      // for precision, cast input to fp32 and compute and then cast it back.
      input = castTo(rewriter, op.getDpsInputs()[0], rewriter.getF32Type());
      output = castTo(rewriter, op.getOutputs()[0], rewriter.getF32Type());
    }

    auto res = logBaseChange(rewriter, op, hfusionFun, input, output);

    if (inType.isF16()) {
      auto roundingAttr =
          rewriter.getAttr<hfusion::RoundModeAttr>(hfusion::RoundMode::RINT);
      auto modeAttr = rewriter.getNamedAttr(
          hfusion::RoundModeAttr::getMnemonic(), roundingAttr);
      auto resF16 = rewriter.create<hfusion::CastOp>(
          op.getLoc(), TypeRange(op.getResults()), ValueRange(res),
          ValueRange(op.getOutputs()[0]), modeAttr);
      rewriter.replaceOp(op, resF16);
    } else {
      rewriter.replaceOp(op, res);
    }

    return success();
  }

private:
  float getBaseNum(hfusion::UnaryFn hfusionFun) const {
    if (hfusionFun == hfusion::UnaryFn::log2) {
      return 2;
    } else if (hfusionFun == hfusion::UnaryFn::log10) {
      return 10;
    }
    llvm_unreachable("unsupport log op");
  }

  Value logBaseChange(PatternRewriter &rewriter, hfusion::ElemwiseUnaryOp op,
                      hfusion::UnaryFn hfusionFun, Value input,
                      Value output) const {
    auto emptyLnCntOp = utils::createEmptyOp(rewriter, op->getLoc(), input);
    auto emptyOutOp = utils::createEmptyOp(rewriter, op->getLoc(), output);
    auto lnOp = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp, linalg::UnaryFn,
                                       linalg::UnaryFnAttr>(
        rewriter, op->getLoc(), linalg::UnaryFn::log, ValueRange{input},
        ValueRange(emptyLnCntOp));

    auto elementType = getElementTypeOrSelf(input.getType());

    float logBase = getBaseNum(hfusionFun);

    auto logBaseValue = rewriter.create<arith::ConstantOp>(
        op->getLoc(), elementType, rewriter.getFloatAttr(elementType, logBase));

    auto fillOp = rewriter.create<linalg::FillOp>(
        op->getLoc(), TypeRange(emptyOutOp), ValueRange{logBaseValue},
        ValueRange{emptyLnCntOp});
    auto ln2Op = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                        linalg::UnaryFn, linalg::UnaryFnAttr>(
        rewriter, op->getLoc(), linalg::UnaryFn::log,
        ValueRange{fillOp.getResults()[0]}, ValueRange(emptyLnCntOp));
    return hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                   linalg::BinaryFnAttr>(
               rewriter, op->getLoc(), linalg::BinaryFn::div,
               ValueRange({lnOp->getResults()[0], ln2Op->getResults()[0]}),
               ValueRange(emptyOutOp))
        ->getResults()[0];
  }
};

/// normalize log1p(x) to ln(x + 1)
/// eg.
/// y = hfusion elemwise unary {log1p} (x)
///  is normalized to
///  y = linalg.elemwise_unary {log}(x + 1)
struct NormalizeLog1pOp : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    auto hfusionFun = op.getFun();
    if (hfusionFun != hfusion::UnaryFn::log1p) {
      return failure();
    }

#ifndef NDEBUG
    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");
#endif

    auto input = op.getDpsInputs()[0];
    auto emptyOp = utils::createEmptyOp(rewriter, op->getLoc(), input);
    auto elementType = getElementTypeOrSelf(input.getType());
    float logOffset;
    if (hfusionFun == hfusion::UnaryFn::log1p) {
      logOffset = 1;
    } else {
      llvm_unreachable("unsupport log op");
    }
    Value plusValue = rewriter.create<arith::ConstantOp>(
        op->getLoc(), elementType,
        rewriter.getFloatAttr(elementType, logOffset));
    auto addOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::add,
            ValueRange({input, plusValue}), ValueRange(emptyOp));

    auto emptyResOp = utils::createEmptyOp(rewriter, op->getLoc(), input);
    auto lnOp = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp, linalg::UnaryFn,
                                       linalg::UnaryFnAttr>(
        rewriter, op->getLoc(), linalg::UnaryFn::log,
        ValueRange{addOp->getResults()}, ValueRange(emptyResOp));

    rewriter.replaceOp(op, lnOp);
    return success();
  }
};

///  normalize mod op to rec op
///   z = x % y
///  is normalized to
///   rem = x - truncate_div(x, y) * y
///  e.g.
///   41 % 20 = 1; 41 % (-20) = -19; (-72) % 8 = 0
///  fp16/bf16 type needs to convert to fp32 to calculate for higher
///  accuracy
struct NormalizeModOp : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
private:
  Value createCmpOpWithType(PatternRewriter &rewriter, Location loc, Value lhs,
                            Value rhs, CompareFn cmpFn, Value typeValue) const {
    Type boolType = rewriter.getIntegerType(1);
    auto cmpInit = utils::createEmptyOpWithTargetElemType(rewriter, loc,
                                                          typeValue, boolType);
    auto cmpPredicateAttr = rewriter.getAttr<hfusion::CompareFnAttr>(cmpFn);
    auto cmpModeAttr = rewriter.getNamedAttr(
        hfusion::CompareFnAttr::getMnemonic(), cmpPredicateAttr);
    return rewriter
        .create<hfusion::CompareOp>(loc, TypeRange(cmpInit),
                                    ValueRange({lhs, rhs}), ValueRange(cmpInit),
                                    ArrayRef{cmpModeAttr})
        ->getResult(0);
  }

  Value createSelectOp(PatternRewriter &rewriter, Location loc, Value predicate,
                       Value x, Value y, Value typeValue) const {
    auto selectOpOut = utils::createEmptyOp(rewriter, loc, typeValue);
    return rewriter
        .create<SelectOp>(loc, TypeRange{selectOpOut.getType()},
                          ValueRange({predicate, x, y}),
                          ValueRange(selectOpOut))
        ->getResults()[0];
  }

  template <typename Op, typename Fn, typename Attr>
  Value createBinaryOpWithEmptyTensor(PatternRewriter &rewriter, Location loc,
                                      Fn op, Value x, Value y,
                                      Value typeValue) const {
    auto emptyTensor = utils::createEmptyOp(rewriter, loc, typeValue);

    return hfusion::createBinaryOp<Op, Fn, Attr>(rewriter, loc, op,
                                                 ValueRange{x, y}, emptyTensor)
        ->getResults()[0];
  }

  Value createHFusionBinaryOp(PatternRewriter &rewriter, Location loc,
                              hfusion::BinaryFn op, Value x, Value y,
                              Value typeValue) const {
    return createBinaryOpWithEmptyTensor<
        hfusion::ElemwiseBinaryOp, hfusion::BinaryFn, hfusion::BinaryFnAttr>(
        rewriter, loc, op, x, y, typeValue);
  }

  Value createLinalgBinaryOp(PatternRewriter &rewriter, Location loc,
                             linalg::BinaryFn op, Value x, Value y,
                             Value typeValue) const {
    return createBinaryOpWithEmptyTensor<
        linalg::ElemwiseBinaryOp, linalg::BinaryFn, linalg::BinaryFnAttr>(
        rewriter, loc, op, x, y, typeValue);
  }

  Value createDiv(PatternRewriter &rewriter, Location loc, Type resType,
                  Value src0, Value src1) const {

    if (resType.isInteger()) {
      return createLinalgBinaryOp(rewriter, loc, linalg::BinaryFn::div, src0,
                                  src1, src0);
    } else {
      // TODO: Use reciprocal here when we fix division to match torch_npu.
      auto divOp = createHFusionBinaryOp(
          rewriter, loc, hfusion::BinaryFn::divfhp, src0, src1, src0);
      // cast directly to resType
      return hfusion::castTo(rewriter, divOp, resType,
                             hfusion::RoundMode::TRUNC);
    }
  }

  Value ensureRankedTensor1F32(OpBuilder &rewriter, Location loc, Value val,
                               Value shapeV) const {
    Type ty = val.getType();
    if (isa<RankedTensorType>(ty)) {
      return val;
    }

    Type refType = shapeV.getType();

    // Must be a ranked tensor
    auto ranked = dyn_cast<RankedTensorType>(refType);
    if (!ranked) {
      llvm::errs() << "Reference tensor is not a ranked tensor\n";
      assert(ranked);
    }

    // result type: same shape as the reference tensor
    RankedTensorType resultTy =
        RankedTensorType::get(ranked.getShape(), val.getType());

    // Use tensor.generate to broadcast the scalar
    return rewriter.create<tensor::GenerateOp>(
        loc, resultTy,
        /*dynamic extents*/ ValueRange{},
        [&](OpBuilder &b, Location genLoc, ValueRange indices) {
          // yield same scalar at every index
          b.create<tensor::YieldOp>(genLoc, val);
        });
  }

  Value handleInfinityModulus(PatternRewriter &rewriter, Location loc, Value x,
                              Value y, Value result) const {
    x = ensureRankedTensor1F32(rewriter, loc, x, result);
    y = ensureRankedTensor1F32(rewriter, loc, y, result);
    auto resultType = dyn_cast<ShapedType>(result.getType());
    Type boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));

    auto nanConstType = getElementTypeOrSelf(resultType);
    auto floatTy = cast<mlir::FloatType>(nanConstType);
    Value constNan =
        rewriter
            .create<arith::ConstantOp>(
                loc, nanConstType,
                rewriter.getFloatAttr(
                    nanConstType, APFloat::getNaN(floatTy.getFloatSemantics())))
            ->getResults()[0];

    auto yIsInf =
        rewriter.create<hfusion::IsInfOp>(loc, boolType, y)->getResults()[0];
    return createSelectOp(rewriter, loc, yIsInf, constNan, result, result);
  }

  Value rewriteModType(PatternRewriter &rewriter, Location loc, Value x,
                       Value y, Value result, Type origType, Type castedType,
                       hfusion::BinaryFn op) const {
    TypeFn castTypeFn;
    if (op == hfusion::BinaryFn::modui) {
      castTypeFn = hfusion::TypeFn::cast_unsigned;
    } else {
      castTypeFn = hfusion::TypeFn::cast_signed;
    }
    auto xCasted = hfusion::castTo(rewriter, x, castedType, castTypeFn);
    auto yCasted = hfusion::castTo(rewriter, y, castedType, castTypeFn);
    auto modOp =
        createHFusionBinaryOp(rewriter, loc, op, xCasted, yCasted, xCasted);

    return hfusion::castTo(rewriter, modOp, origType);
  }

public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::BinaryFn::mod &&
        op.getFun() != hfusion::BinaryFn::modui) {
      return failure();
    }

    auto resTensor = op.getResultTensors()[0];
    auto resTy = dyn_cast<TensorType>(resTensor.getType());
    auto elemType = getElementTypeOrSelf(resTy);
    if (!elemType.isIntOrIndexOrFloat()) {
      return failure();
    }

    if (elemType.isInteger(1)) {
      auto constZero =
          utils::createConstantOp<bool>(rewriter, op.getLoc(), elemType, 0);
      auto zeroTensor = utils::createEmptyOpWithTargetElemType(
          rewriter, op.getLoc(), resTensor, elemType);
      auto zeroOp =
          rewriter.create<linalg::FillOp>(op.getLoc(), constZero, zeroTensor);
      rewriter.replaceOp(op, zeroOp);
      return success();
    }

    // BF16 and F16 are casted to F32
    // All ints use their original encoding throughout.
    // step 1: xCasted = cast(x) => castedtype
    //         yCasted= cast(y) => castedtype
    Value xOrig = op.getInputs()[0];
    Value yOrig = op.getInputs()[1];
    if (elemType.isInteger(8)) {
      auto resultUint8 =
          rewriteModType(rewriter, op.getLoc(), xOrig, yOrig, resTensor,
                         elemType, rewriter.getI16Type(), op.getFun());
      rewriter.replaceOp(op, resultUint8);
      return success();
    }

    if (elemType.isInteger()) {
      // int mod is handled in hivm
      return failure();
    }

    // Reentrant implementation for fp8, cast and run this again with fp32
    if (elemType.isFloat8E4M3FN() || elemType.isFloat8E5M2()) {
      auto resultFp8 = rewriteModType(
          rewriter, op.getLoc(), xOrig, yOrig, resTensor, elemType,
          rewriter.getF32Type(), hfusion::BinaryFn::mod);
      rewriter.replaceOp(op, resultFp8);
      return success();
    }

    Value xCasted = xOrig;
    Value yCasted = yOrig;
    if (elemType.isBF16() || elemType.isF16()) {
      auto castedType = rewriter.getF32Type();
      xCasted = hfusion::castTo(rewriter, xOrig, castedType);
      yCasted = hfusion::castTo(rewriter, yOrig, castedType);
    }

    // step 2: trunc_div = truncate_div(x, y)
    Value truncDiv =
        createDiv(rewriter, op.getLoc(), elemType, xCasted, yCasted);

    // step 3: rem = x - trunc_div * y
    auto mul =
        createLinalgBinaryOp(rewriter, op.getLoc(), linalg::BinaryFn::mul,
                             truncDiv, yOrig, resTensor);

    auto rem = createLinalgBinaryOp(
        rewriter, op.getLoc(), linalg::BinaryFn::sub, xOrig, mul, resTensor);

    // step 7: handle inf (we are done for floats!)
    Value result =
        handleInfinityModulus(rewriter, op.getLoc(), xOrig, yOrig, rem);
    rewriter.replaceOp(op, result);
    return success();
  }
};

///  TODO: hfusion::binaryfn::floormod unsupport right now
///  normalize mod op to rec op
///   z = x % y
///  is normalized to
///   z = x - x // y * y
struct NormalizeFloorModOp
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::BinaryFn::mod) {
      return failure();
    }

    Type elemType = getElementTypeOrSelf(op.getInputs()[0].getType());
    if (!elemType.isIntOrIndexOrFloat()) {
      return failure();
    }
    if (elemType.isInteger(8)) {
      // i8 mod must be converted to f16 mod before
      return failure();
    }

    /// step 1: div = x / y
    auto emptyDivTensor =
        utils::createEmptyOp(rewriter, op->getLoc(), op.getInputs()[0]);
    auto divOP =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::div,
            ValueRange(op.getInputs()), ValueRange(emptyDivTensor));

    Operation *tempOp = divOP;

    /// step 2: floor = floor(res)
    if (isa<FloatType>(elemType)) {
      // insert extra floor for float mod
      auto emptyFloorTensor =
          utils::createEmptyOp(rewriter, op->getLoc(), op.getInputs()[0]);
      auto floorOp =
          hfusion::createUnaryOp<linalg::ElemwiseUnaryOp, linalg::UnaryFn,
                                 linalg::UnaryFnAttr>(
              rewriter, op->getLoc(), linalg::UnaryFn::floor,
              ValueRange{divOP->getResults()[0]}, ValueRange(emptyFloorTensor));
      tempOp = floorOp;
    }

    /// step 3:
    /// for int mod: mul = div * y
    /// for float mod: mul = floor * y
    auto emptyMulTensor =
        utils::createEmptyOp(rewriter, op->getLoc(), op.getInputs()[0]);
    auto mulOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::mul,
            ValueRange({tempOp->getResults()[0], op.getInputs()[1]}),
            ValueRange(emptyMulTensor));
    /// step 4: mod = x - mul
    auto emptySubTensor =
        utils::createEmptyOp(rewriter, op->getLoc(), op.getInputs()[0]);
    auto subOP =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::sub,
            ValueRange({op.getInputs()[0], mulOp->getResults()[0]}),
            ValueRange(emptySubTensor));

    rewriter.replaceOp(op, subOP);
    return success();
  }
};

struct NormalizeCeilandFloorOp
    : public OpRewritePattern<linalg::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != linalg::UnaryFn::ceil &&
        op.getFun() != linalg::UnaryFn::floor) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());

    assert(!inType.isInteger() && "Cast in floor/ceil mode doesn't support "
                                  "integer type input");
    OpBuilder builder(op);
    Value src = op.getInputs()[0];
    hfusion::RoundMode roundMode = op.getFun() == linalg::UnaryFn::ceil
                                       ? hfusion::RoundMode::CEIL
                                       : hfusion::RoundMode::FLOOR;
    if (!archisAscend950) {
      if ((inType.isF16() || inType.isBF16()) && inType == outType) {
        // 910B only support fp32 ceil and floor, so change to fp16->fp32,
        // fp32 ceil/floor and fp32->fp16
        // TODO: add platform info to isHWSupportCeilFLoor(Type)

        // Step1: cast to fp32 to do ceil or floor
        auto intermediate = hfusion::castTo(builder, src, rewriter.getF32Type(),
                                            hfusion::RoundMode::RINT);
        // Step2: enable fp32 cast ability with ceil or floor mode
        // Otherwise, cast fp32 to fp16 type in ceil or floor mode just changes
        // precision loss part.
        src = hfusion::castTo(builder, intermediate, rewriter.getF32Type(),
                              roundMode);
      }
    }

    auto castOp =
        hfusion::castTo(builder, src, outType, roundMode, op.getOutputs()[0]);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

/// normalize 2^x to exp{ln(2)*x}
/// eg.
/// y = hfusion elemwise unary {exp2} (x)
/// is normalized to
///  y = linalg.elemwise_unary{vexp}(ln2 * x)
struct NormalizeExp2Op : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::UnaryFn::exp2) {
      return failure();
    }

    Value src = op.getInputs()[0];
    auto inType = getElementTypeOrSelf(src.getType());
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");

    if (inType.isF16()) {
      // TODO: remove cast after enable automatical high precision computing
      src = hfusion::castTo(rewriter, src, rewriter.getF32Type(),
                            hfusion::RoundMode::ROUND);
    }

    auto elementType = getElementTypeOrSelf(src.getType());
    Value constLnTwo = rewriter.create<arith::ConstantOp>(
        op->getLoc(), elementType,
        rewriter.getFloatAttr(elementType, std::log(2)));

    auto emptyLnCntOp = utils::createEmptyOp(rewriter, op->getLoc(), src);
    auto *mulOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::mul,
            ValueRange({src, constLnTwo}), ValueRange(emptyLnCntOp));

    auto emptyResOp = utils::createEmptyOp(rewriter, op->getLoc(), src);
    auto *expOp = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                         linalg::UnaryFn, linalg::UnaryFnAttr>(
        rewriter, op->getLoc(), linalg::UnaryFn::exp,
        ValueRange{mulOp->getResults()[0]}, ValueRange(emptyResOp));

    Value res = expOp->getResult(0);
    if (inType.isF16()) {
      // TODO: remove cast after enable automatical high precision computing
      res = hfusion::castTo(rewriter, res, rewriter.getF16Type(),
                            hfusion::RoundMode::ROUND);
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct NormalizeMinMax {
  /// Returns a new operand for BinaryFn::maxf (BinaryFn::minf)
  /// that is used when normalizing maxnumf (minnumf) to maxf (minf).
  Value createNewSrcForMinMaxNumFOp(PatternRewriter &rewriter, Location loc,
                                    Value src, double paddingInfValue) const {
    auto elementType = getElementTypeOrSelf(src.getType());
    auto constInfinity = utils::createConstantOp<double>(
        rewriter, loc, elementType, paddingInfValue);

    auto isNanResultTensorType =
        utils::getTensorTypeWithSameShape(src.getType(), rewriter.getI1Type());
    auto isNanOp = rewriter.create<IsNanOp>(loc, isNanResultTensorType, src);

    auto selectOpOut = utils::createEmptyOp(rewriter, loc, src);
    auto selectOp = rewriter.create<SelectOp>(
        loc, TypeRange(selectOpOut),
        ValueRange({isNanOp->getResult(0), constInfinity, src}),
        ValueRange(selectOpOut));

    return selectOp->getResult(0);
  }
};

/// Normalize maxnumf (minnumf) to maxf (minf).
/// eg.
/// dst = hfusion.elemwise_binary {maxnumf} (src0, src1)
/// is normalized to
/// src0_nan_mask = hfusion.isnan(src0)
/// new_src0 = hfusion.select(src0_nan_mask, -inf, src0)
/// src1_nan_mask = hfusion.isnan(src1)
/// new_src1 = hfusion.select(src1_nan_mask, -inf, src1)
/// dst = hfusion.elemwise_binary {maxf} (new_src0, new_src1)
template <BinaryFn funFrom>
struct NormalizeElemwiseMinMaxNumFOp
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp>,
      public NormalizeMinMax {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    static_assert(funFrom == BinaryFn::maxnumf || funFrom == BinaryFn::minnumf,
                  "Argument mismatch. NormaliseMinMaxNumFOp expects "
                  "hfusion::BinaryFn::maxnumf or hfusion::BinaryFn::minnumf");

    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != funFrom) {
      return failure();
    }

    constexpr auto funTo =
        (funFrom == BinaryFn::maxnumf) ? BinaryFn::maxf : BinaryFn::minf;
    constexpr auto paddingInfSign = (funFrom == BinaryFn::maxnumf) ? -1 : 1;

    auto res = rewriteMinMaxNumFOp<funTo>(
        op, rewriter, paddingInfSign * std::numeric_limits<double>::infinity());

    rewriter.replaceOp(op, res);
    return success();
  }

private:
  /// Normalize maxnumf (minnumf) to maxf (minf)
  /// Check comment before struct definition
  template <hfusion::BinaryFn hfusionFn>
  Value rewriteMinMaxNumFOp(hfusion::ElemwiseBinaryOp op,
                            PatternRewriter &rewriter,
                            double paddingInfValue) const {
    static_assert(
        hfusionFn == BinaryFn::maxf || hfusionFn == BinaryFn::minf,
        "Normalization hfusion::BinaryFn::maxnumf (minnumf) allows "
        "only hfusion::BinaryFn::maxf (minf) to be used for replacement");

    Value src0 = op.getInputs()[0];
    Value src1 = op.getInputs()[1];

    auto newSrc0 = createNewSrcForMinMaxNumFOp(rewriter, op->getLoc(), src0,
                                               paddingInfValue);
    auto newSrc1 = createNewSrcForMinMaxNumFOp(rewriter, op->getLoc(), src1,
                                               paddingInfValue);
    auto minMaxFOpOut = utils::createEmptyOp(rewriter, op->getLoc(), src0);
    auto minMaxFOp = createBinaryOp<ElemwiseBinaryOp, BinaryFn, BinaryFnAttr>(
        rewriter, op->getLoc(), hfusionFn, ValueRange({newSrc0, newSrc1}),
        ValueRange(minMaxFOpOut));

    return minMaxFOp->getResult(0);
  }
};

// Normalize reduction op of maxnumf and minnumf
// dst = linalg.reduce { arith.maxnumf } src
// is normalized to
// src_nan_mask = hfusion.isnan(src)
// new_src = hfusion.select(src_nan_mask, -inf, src)
// dst = linalg.reduce { arith.maximumf } src
struct NormalizeReduceMinMaxNumFOp : public OpRewritePattern<linalg::ReduceOp>,
                                     public NormalizeMinMax {
public:
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;
  virtual ~NormalizeReduceMinMaxNumFOp() = default;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    Block &body = op.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    auto *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    if (!isa<arith::MaxNumFOp>(bodyOp) && !isa<arith::MinNumFOp>(bodyOp)) {
      return failure();
    }
    auto paddingInfSign = (isa<arith::MaxNumFOp>(bodyOp)) ? -1 : 1;
    Value src0 = op->getOperands()[0];
    auto newSrc0 = createNewSrcForMinMaxNumFOp(
        rewriter, op->getLoc(), src0,
        paddingInfSign * std::numeric_limits<double>::infinity());
    op->setOperand(0, newSrc0);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(bodyOp);
    Operation *newOp;
    if (isa<arith::MaxNumFOp>(bodyOp)) {
      newOp = rewriter.create<arith::MaximumFOp>(
          bodyOp->getLoc(), bodyOp->getOperand(0), bodyOp->getOperand(1));
    } else {
      newOp = rewriter.create<arith::MinimumFOp>(
          bodyOp->getLoc(), bodyOp->getOperand(0), bodyOp->getOperand(1));
    }
    rewriter.replaceAllUsesWith(bodyOp->getResult(0), newOp->getResult(0));
    rewriter.eraseOp(bodyOp);
    return success();
  }
};

/// normalize expm1(x) to exp(x) - 1
/// eg.
/// y = hfusion elemwise unary {expm1} (x)
/// is normalized to
///  y = linalg.elemwise_unary{exp}(x) -1
struct NormalizeExpM1Op : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    auto hfusionFun = op.getFun();
    if (hfusionFun != hfusion::UnaryFn::expm1) {
      return failure();
    }

    Value src = op.getInputs()[0];
    auto inType = getElementTypeOrSelf(src.getType());
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");

    if (inType.isF16()) {
      // TODO: remove cast after enable automatical high precision computing
      src = hfusion::castTo(rewriter, src, rewriter.getF32Type(),
                            hfusion::RoundMode::ROUND);
    }

    auto elementType = getElementTypeOrSelf(src.getType());
    float downOffset;
    if (hfusionFun == hfusion::UnaryFn::expm1) {
      downOffset = 1;
    } else {
      llvm_unreachable("unsupport exp op");
    }
    Value subValue = rewriter.create<arith::ConstantOp>(
        op->getLoc(), elementType,
        rewriter.getFloatAttr(elementType, downOffset));

    auto emptyExpOp = utils::createEmptyOp(rewriter, op->getLoc(), src);
    auto *expOp = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                         linalg::UnaryFn, linalg::UnaryFnAttr>(
        rewriter, op->getLoc(), linalg::UnaryFn::exp, ValueRange{src},
        ValueRange(emptyExpOp));

    auto emptyResOp = utils::createEmptyOp(rewriter, op->getLoc(), src);
    auto *subOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::sub,
            ValueRange({expOp->getResults()[0], subValue}),
            ValueRange(emptyResOp));
    Value res = subOp->getResult(0);
    if (inType.isF16()) {
      // TODO: remove cast after enable automatical high precision computing
      res = hfusion::castTo(rewriter, res, rewriter.getF16Type(),
                            hfusion::RoundMode::ROUND);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

/// step 1. clip x into [-3.92,3.92]
/// step 2. numer=((((((CST0*y)+T1)*y+T2)*y+T3)*y+T4)*y+T5)*x, y=x^2
/// step 3. demon=((((y+P1)*y+P2)*y+P3)*y+P4)*y+P5, y=x^2
/// step 4: erf(x) = numer / denom
struct NormalizeErfOp : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    auto hfusionFun = op.getFun();
    if (hfusionFun != hfusion::UnaryFn::erf) {
      return failure();
    }

    Value src = op.getInputs()[0];
    auto inType = getElementTypeOrSelf(src);
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");

    if (getElementTypeOrSelf(src).isF16()) {
      // for high precision, cast src to fp32 and compute and then cast it back
      // TODO: remove cast after enable automatical high precision computing
      src = hfusion::castTo(rewriter, src, rewriter.getF32Type(),
                            hfusion::RoundMode::ROUND);
    }

    // 1. clip input into [-3.92, 3.92]
    auto loc = op->getLoc();
    Value clipedInput = ClipInput(rewriter, loc, src, 3.92, -3.92);

    // 2. step 2 numer=((((((CST0*y)+T1)*y+T2)*y+T3)*y+T4)*y+T5)*x,
    auto squareInput = utils::createEmptyOp(rewriter, loc, clipedInput);
    auto *squareOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul,
            ValueRange{clipedInput, clipedInput}, ValueRange(squareInput));

    // 2.1. first z = CST0*y,CST0=0.53443748819e-1,
    double CST0 = 0.53443748819e-1;
    auto numerInit = utils::createEmptyOp(rewriter, loc, clipedInput);
    auto constValInit = rewriter.create<arith::ConstantOp>(
        loc, getElementTypeOrSelf(src),
        rewriter.getFloatAttr(getElementTypeOrSelf(src), CST0));
    auto *numerInitOp = hfusion::createBinaryOp<
        linalg::ElemwiseBinaryOp, linalg::BinaryFn, linalg::BinaryFnAttr>(
        rewriter, loc, linalg::BinaryFn::mul,
        ValueRange{squareOp->getResults()[0], constValInit->getResults()[0]},
        ValueRange(numerInit));

    // 2.2. get polyexpr in the format z = (((((z+T1)*y+T2)*y+T3)*y+T4)*y+T5)
    // {T1, T2, T3, T4, T5}={0.75517016694e1, 0.10162808918e3, 0.13938061484e4,
    // 0.50637915060e4, 0.29639384698e5}
    const llvm::SmallVector<double> numerCoeff{0.75517016694e1, 0.10162808918e3,
                                               0.13938061484e4, 0.50637915060e4,
                                               0.29639384698e5};
    Value numerRes =
        genPolyExpr(rewriter, loc, squareOp->getResults()[0],
                    numerInitOp->getResults()[0], numerCoeff, false);

    // 2.3. mul x , z = z * x
    auto *numerResOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul,
            ValueRange{clipedInput, numerRes}, ValueRange(numerInit));

    // 3. get denom
    // let y=x^2, demon=((((y+P1)*y+P2)*y+P3)*y+P4)*y+P5,
    // P={P1, P2, P3, P4, P5}={0.31212858877e2, 0.39856963806e3,
    // 0.30231248150e4, 0.13243365831e5, 0.26267224157e5}
    const llvm::SmallVector<double> demonCoeff{0.31212858877e2, 0.39856963806e3,
                                               0.30231248150e4, 0.13243365831e5,
                                               0.26267224157e5};
    Value demonRes = genPolyExpr(rewriter, loc, squareOp->getResults()[0],
                                 squareOp->getResults()[0], demonCoeff, false);

    // 4. res = numer / denom
    auto emptyResOp = utils::createEmptyOp(rewriter, op->getLoc(), clipedInput);
    Value res = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                        linalg::BinaryFn, linalg::BinaryFnAttr>(
                    rewriter, loc, linalg::BinaryFn::div,
                    ValueRange{numerResOp->getResults()[0], demonRes},
                    ValueRange(emptyResOp))
                    ->getResult(0);

    if (inType.isF16()) {
      // TODO: remove cast after enable automatical high precision computing
      res = hfusion::castTo(rewriter, res, rewriter.getF16Type(),
                            hfusion::RoundMode::ROUND);
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

/// normalize ilogb(x), which is exponent of frexp(x), to floor(log2(abs(x)))
struct NormalizeIlogbOp : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::UnaryFn::ilogb) {
      return failure();
    }

    Value input = op.getInputs()[0];
#ifndef NDEBUG
    auto inType = getElementTypeOrSelf(input.getType());
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");
#endif
    auto loc = op->getLoc();

    auto absEmptyOp = utils::createEmptyOp(rewriter, loc, input);

    auto xAbs =
        hfusion::createUnaryOp<linalg::ElemwiseUnaryOp, linalg::UnaryFn,
                               linalg::UnaryFnAttr>(
            rewriter, loc, linalg::UnaryFn::abs, input, ValueRange(absEmptyOp))
            ->getResult(0);

    auto log2EmptyOp = utils::createEmptyOp(rewriter, loc, input);

    auto xLog2 = hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp,
                                        hfusion::UnaryFn, hfusion::UnaryFnAttr>(
                     rewriter, loc, hfusion::UnaryFn::log2, xAbs,
                     ValueRange(log2EmptyOp))
                     ->getResult(0);

    auto floorEmptyOp = utils::createEmptyOp(rewriter, loc, input);
    auto xFloor = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                         linalg::UnaryFn, linalg::UnaryFnAttr>(
                      rewriter, loc, linalg::UnaryFn::floor, xLog2,
                      ValueRange(floorEmptyOp))
                      ->getResult(0);

    rewriter.replaceOp(op, xFloor);
    return success();
  }
};

/// nomalize frexp(x), which is mantissa for frexp(x), to x * (ilogb(x) +
/// 1)^(-1)
struct NormalizeLdexpOp : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::BinaryFn::ldexp) {
      return failure();
    }

    Value input = op.getInputs()[0];
#ifndef NDEBUG
    auto inType = getElementTypeOrSelf(input.getType());
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");
#endif
    auto loc = op->getLoc();

    auto mulEmptyOp = utils::createEmptyOp(rewriter, loc, input);

    auto xMul =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul,
            ValueRange{input, op.getInputs()[1]}, ValueRange(mulEmptyOp))
            ->getResult(0);

    rewriter.replaceOp(op, xMul);
    return success();
  }
};

/// normalize powf(baseNum, exponent) as below
/// powf(x, y) = 1, when abs(x) = 1 and abs(y) = inf
///            = nan, when x = -inf and y is not integer value or y is finite
///            = nan, when x < 0 and x is finite. and y is finite and y is not
///            integer
///            = x ^ y = exp(y * ln(|x|)), when x > 0
///            = x ^ y = ((-1) ^ y) * exp(y * ln|x|), when x <  0
///            = 1, when y == 0
/// so
/// partialRes0 = x ^ y = exp(y * ln(|x|)), when x > 0
///             = x ^ y = ((-1) ^ y) * exp(y * ln|x|), when x <  0
/// partialRes1 = select(abs(x)==1 && abs(y)==inf, 1, partialRes0)
/// partialRes2 = select((abs(x) != inf and x < 0 and abs(y) != inf
///               and floor(y) != y), nan, partialRes1), namely when x is
///               negative finite and y is finite and not integer, result is nan
/// pow(x, y) = select(y == 0, 1, partialRes2)
/// TODO : support nan boundary case
/// note: hardware vln will output -inf when x == 0
struct NormalizePowfOp : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;

  /// generate boundary condition when result is one, namely
  /// when abs(x) = 1 and abs(y) = inf, power(x, y) = 1
  Value genBoundaryConditionForOne(PatternRewriter &rewriter, Value baseNum,
                                   Value exponent, Location loc) const {
    /// step1: judge whether abs(x) = 1
    ///   1. absx = abs(x)
    auto absBaseInit = utils::createEmptyOp(rewriter, loc, baseNum);
    auto absBase = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                          linalg::UnaryFn, linalg::UnaryFnAttr>(
                       rewriter, loc, linalg::UnaryFn::abs, ValueRange(baseNum),
                       ValueRange(absBaseInit))
                       ->getResult(0);

    ///   2. mask0 = cmp_eq(absx, 1)
    auto elementType = getElementTypeOrSelf(baseNum.getType());
    arith::ConstantOp constOne = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 1.0));
    auto mask0 =
        createCmpOp(rewriter, loc, absBase, constOne, hfusion::CompareFn::veq)
            ->getResult(0);

    /// step2: judge whether abs(y) = inf
    ///   1. absy = abs(y)
    auto absExpInit = utils::createEmptyOp(rewriter, loc, exponent);
    auto absExp = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                         linalg::UnaryFn, linalg::UnaryFnAttr>(
                      rewriter, loc, linalg::UnaryFn::abs, ValueRange(exponent),
                      ValueRange(absExpInit))
                      ->getResult(0);

    ///   2. mask1 = cmp_eq(absy, inf)
    arith::ConstantOp constInf = nullptr;
    if (elementType.isF16()) {
      constInf = rewriter.create<arith::ConstantOp>(
          loc, elementType, rewriter.getFloatAttr(elementType, 0x7C00));
    } else if (elementType.isF32()) {
      constInf = rewriter.create<arith::ConstantOp>(
          loc, elementType, rewriter.getFloatAttr(elementType, 0x7F800000));
    }
    auto mask1 =
        createCmpOp(rewriter, loc, absExp, constInf, hfusion::CompareFn::veq)
            ->getResult(0);

    /// step3: return boundary condition judgement
    /// 1. res = vand(mask0, mask1)
    return createVandOp(rewriter, loc, mask0, mask1)->getResult(0);
  }

  Value getSignbitOfBaseNum(PatternRewriter &rewriter, Location loc,
                            Value baseNum) const {
    auto elementType = getElementTypeOrSelf(baseNum.getType());
    auto bitWidth = elementType.getIntOrFloatBitWidth();
    Type intType = rewriter.getIntegerType(bitWidth);
    ///    1. x_uint = bitcast(x)
    auto shapedType = dyn_cast_if_present<ShapedType>(baseNum.getType());
    auto bitcastEmptyOp =
        utils::createEmptyOpWithTargetElemType(rewriter, loc, baseNum, intType);
    auto bitcastOp = rewriter.create<hfusion::BitcastOp>(
        loc, TypeRange{shapedType.clone(intType)}, ValueRange{baseNum},
        ValueRange{bitcastEmptyOp});

    ///    2. signbit = shr(x_uint, 31)
    arith::ConstantOp shiftValue = rewriter.create<arith::ConstantOp>(
        loc, intType, rewriter.getIntegerAttr(intType, bitWidth - 1));
    auto shrEmptyOp =
        utils::createEmptyOp(rewriter, loc, bitcastOp.getResults()[0]);
    auto signbit =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, hfusion::BinaryFn::shrsi,
            ValueRange({bitcastOp.getResults()[0], shiftValue}),
            ValueRange{shrEmptyOp})
            ->getResult(0);

    ///    3. mask0 = cmp_eq(signbit, -1)
    arith::ConstantOp constOne = rewriter.create<arith::ConstantOp>(
        loc, intType, rewriter.getIntegerAttr(intType, -1));
    return createCmpOp(rewriter, loc, signbit, constOne, CompareFn::veq)
        ->getResult(0);
  }

  Value judgeIntegerValue(PatternRewriter &rewriter, Location loc,
                          Value baseNum, Value exponent) const {
    ///    1. y_floor = cast_floor(y)
    auto floorEmptyOp = utils::createEmptyOp(rewriter, loc, exponent);
    auto floor = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                        linalg::UnaryFn, linalg::UnaryFnAttr>(
                     rewriter, loc, linalg::UnaryFn::floor,
                     ValueRange({exponent}), ValueRange(floorEmptyOp))
                     ->getResult(0);

    ///    2. mask1 = cmp_eq(y, y_floor)
    return createCmpOp(rewriter, loc, floor, exponent, CompareFn::veq)
        ->getResult(0);
  }

  /// when the signbit of base number x is 1 and exponent y is int value
  ///  step1: judge the signbit of base number x
  ///    1. x_uint = bitcast(x)
  ///    2. signbit = shr(x_uint, 31)
  ///    3. mask0 = cmp_eq(signbit, -1)
  ///  step2: judge whether y is an integer value
  ///    1. y_floor = cast_floor(y)
  ///    2. mask1 = cmp_eq(y, y_floor)
  ///  step3.: return negative condition judgement
  ///    1. res = vand(mask0, mask1)
  Value isNegCondition(PatternRewriter &rewriter, Value baseNum, Value exponent,
                       Location loc) const {
    ///  step1: judge the signbit of base number x
    auto isNeg = getSignbitOfBaseNum(rewriter, loc, baseNum);

    ///  step2: judge whether y is an integer value
    auto isInteger = judgeIntegerValue(rewriter, loc, baseNum, exponent);

    ///  step3.: return negative condition judgement
    ///    1. res = vand(mask0, mask1)
    return createVandOp(rewriter, loc, isNeg, isInteger)->getResult(0);
  }

  /// caculate coef of (-1)^y
  /// (-1)^y = [-2 * (|y| % 2) + 1], when y is integer,
  /// otherwise invalid value calculateCoef
  Value calculateCof(PatternRewriter &rewriter, Location loc,
                     Value input) const {
    auto elementType = getElementTypeOrSelf(input.getType());
    arith::ConstantOp positiveOne = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 1));

    arith::ConstantOp positiveTwo = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 2));

    arith::ConstantOp negativeTwo = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, -2));

    auto absEmptyOp = utils::createEmptyOp(rewriter, loc, input);
    auto absBase = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                          linalg::UnaryFn, linalg::UnaryFnAttr>(
                       rewriter, loc, linalg::UnaryFn::abs, ValueRange(input),
                       ValueRange(absEmptyOp))
                       ->getResult(0);

    auto modEmptyOp = utils::createEmptyOp(rewriter, loc, input);
    auto mod =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, hfusion::BinaryFn::mod,
            ValueRange({absBase, positiveTwo}), ValueRange(modEmptyOp))
            ->getResult(0);

    auto mulEmptyOp = utils::createEmptyOp(rewriter, loc, input);
    auto mul = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                       linalg::BinaryFn, linalg::BinaryFnAttr>(
                   rewriter, loc, linalg::BinaryFn::mul,
                   ValueRange({mod, negativeTwo}), ValueRange(mulEmptyOp))
                   ->getResult(0);

    auto addEmptyOp = utils::createEmptyOp(rewriter, loc, input);
    auto add = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                       linalg::BinaryFn, linalg::BinaryFnAttr>(
                   rewriter, loc, linalg::BinaryFn::add,
                   ValueRange({mul, positiveOne}), ValueRange(addEmptyOp))
                   ->getResult(0);

    return add;
  }

  /// calculate ((-1) ^ y) * exp(y * ln|x|), where x is baseNum and y is
  /// exponent
  Value calculateNegativeCompute(PatternRewriter &rewriter, mlir::Value baseNum,
                                 mlir::Value exponent, Location loc) const {
    auto lnEmptyOp = utils::createEmptyOp(rewriter, loc, baseNum);
    auto mulEmptyOp = utils::createEmptyOp(rewriter, loc, baseNum);
    auto coff = calculateCof(rewriter, loc, exponent);

    ///  step1: compute abs(baseNum)
    auto absEmptyOp = utils::createEmptyOp(rewriter, loc, baseNum);
    auto absBase = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                          linalg::UnaryFn, linalg::UnaryFnAttr>(
                       rewriter, loc, linalg::UnaryFn::abs, baseNum,
                       ValueRange(absEmptyOp))
                       ->getResult(0);

    ///  step2: compute ln(abs(baseNum))
    auto lnBase = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                         linalg::UnaryFn, linalg::UnaryFnAttr>(
                      rewriter, loc, linalg::UnaryFn::log,
                      ValueRange({absBase}), ValueRange(lnEmptyOp))
                      ->getResult(0);

    ///  step3: compute exponent*ln(abs(baseNum))
    auto mul = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                       linalg::BinaryFn, linalg::BinaryFnAttr>(
                   rewriter, loc, linalg::BinaryFn::mul,
                   ValueRange({lnBase, exponent}), ValueRange(mulEmptyOp))
                   ->getResult(0);

    ///  step4: compute exp(exponent*ln(abs(baseNum)))
    auto expEmptyOp = utils::createEmptyOp(rewriter, loc, baseNum);
    auto exp =
        hfusion::createBinaryOp<linalg::ElemwiseUnaryOp, linalg::UnaryFn,
                                linalg::UnaryFnAttr>(
            rewriter, loc, linalg::UnaryFn::exp, mul, ValueRange(expEmptyOp))
            ->getResult(0);

    ///  step5: compute coef*exp(exponent*ln(abs(baseNum)))
    auto mulCoffEmptyOp = utils::createEmptyOp(rewriter, loc, baseNum);
    auto res = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                       linalg::BinaryFn, linalg::BinaryFnAttr>(
                   rewriter, loc, linalg::BinaryFn::mul,
                   ValueRange({exp, coff}), ValueRange(mulCoffEmptyOp))
                   ->getResult(0);
    return res;
  }

  /// calculate exp(y * ln|x|), where x is baseNum and y is exponent
  Value calculatePositiveCompute(PatternRewriter &rewriter, mlir::Value baseNum,
                                 mlir::Value exponent, Location loc) const {
    auto lnEmptyOp = utils::createEmptyOp(rewriter, loc, baseNum);
    auto mulEmptyOp = utils::createEmptyOp(rewriter, loc, baseNum);
    auto resEmptyOp = utils::createEmptyOp(rewriter, loc, baseNum);
    auto absEmptyOp = utils::createEmptyOp(rewriter, loc, baseNum);

    ///  step1: compute abs(baseNum)
    auto absBase = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                          linalg::UnaryFn, linalg::UnaryFnAttr>(
                       rewriter, loc, linalg::UnaryFn::abs, baseNum,
                       ValueRange(absEmptyOp))
                       ->getResult(0);
    ///  step2: compute ln(abs(baseNum))
    auto lnBase = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                         linalg::UnaryFn, linalg::UnaryFnAttr>(
                      rewriter, loc, linalg::UnaryFn::log, ValueRange(absBase),
                      ValueRange(lnEmptyOp))
                      ->getResult(0);

    ///  step3: compute exponent*ln(abs(baseNum))
    auto mul = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                       linalg::BinaryFn, linalg::BinaryFnAttr>(
                   rewriter, loc, linalg::BinaryFn::mul,
                   ValueRange({lnBase, exponent}), ValueRange(mulEmptyOp))
                   ->getResult(0);

    /// step4: compute exp(exponent*ln(abs(baseNum)))
    auto res = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp, linalg::UnaryFn,
                                      linalg::UnaryFnAttr>(
                   rewriter, loc, linalg::UnaryFn::exp, ValueRange(mul),
                   ValueRange(resEmptyOp))
                   ->getResult(0);
    return res;
  }

  Value calculatePower(OpBuilder &rewriter, Location loc, Value baseNum,
                       int exponent) const {
    auto resEmptyOp = utils::createEmptyOp(rewriter, loc, baseNum);
    if (exponent <= 1) {
      return baseNum;
    }
    return hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                   linalg::BinaryFnAttr>(
               rewriter, loc, linalg::BinaryFn::mul,
               ValueRange({baseNum, calculatePower(rewriter, loc, baseNum,
                                                   exponent - 1)}),
               ValueRange(resEmptyOp))
        ->getResult(0);
  }

  /// pow(x, 0.5) converts to sqrt(x)
  void createSqrtOp(hfusion::ElemwiseBinaryOp op, PatternRewriter &rewriter,
                    Value baseNum) const {
    Location loc = op->getLoc();
    auto resEmptyOp = utils::createEmptyOp(rewriter, loc, baseNum);
    auto res = hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp,
                                      hfusion::UnaryFn, hfusion::UnaryFnAttr>(
                   rewriter, loc, hfusion::UnaryFn::sqrt, ValueRange(baseNum),
                   ValueRange(resEmptyOp))
                   ->getResult(0);
    rewriter.replaceOp(op, res);
  }

  float getFillValue(Operation *fillOp) const {
    Value constValue = fillOp->getOperand(0);
    bool isInt = constValue.getType().isIntOrIndex();
    auto constOp =
        dyn_cast_or_null<arith::ConstantOp>(constValue.getDefiningOp());
    if (isInt) {
      auto constFloatAttr = dyn_cast<IntegerAttr>(constOp.getValue());
      return llvm::APIntOps::RoundAPIntToFloat(constFloatAttr.getValue());
    }
    auto constFloatAttr = dyn_cast<FloatAttr>(constOp.getValue());
    return constFloatAttr.getValue().convertToFloat();
  }

  arith::ConstantOp getExponentConstOp(Value exponent,
                                       PatternRewriter &rewriter) const {
    if (auto castOp = exponent.getDefiningOp<hfusion::CastOp>()) {
      if (auto fillOp =
              castOp.getDpsInputs()[0].getDefiningOp<linalg::FillOp>()) {
        auto fillValue = getFillValue(fillOp);
        auto loc = castOp->getLoc();
        auto elementType =
            getElementTypeOrSelf(castOp.getDpsInits()[0].getType());
        auto insertInit = rewriter.create<arith::ConstantOp>(
            loc, elementType, rewriter.getFloatAttr(elementType, fillValue));
        return insertInit;
      }
    }

    if (auto fillOp = exponent.getDefiningOp<linalg::FillOp>()) {
      return dyn_cast_if_present<arith::ConstantOp>(
          fillOp.getInputs()[0].getDefiningOp());
    }
    auto constOp =
        dyn_cast_or_null<arith::ConstantOp>(exponent.getDefiningOp());
    if (constOp == nullptr)
      return constOp;
    auto shapedType = dyn_cast<ShapedType>(constOp.getType());
    if (shapedType) {
      auto scalarElem =
          getScalarFromConstantOp(rewriter, exponent.getLoc(), constOp);
      if (scalarElem.has_value())
        return dyn_cast_or_null<arith::ConstantOp>(scalarElem->getDefiningOp());
    }
    return constOp;
  }

  Value getExponent(PatternRewriter &rewriter, Value baseNum, Value exponent,
                    Location loc) const {
    auto singleElem = singleElemDenseTensorToScalar(exponent, rewriter);
    if (singleElem.has_value()) {
      auto fillEmptyOp = utils::createEmptyOp(rewriter, loc, baseNum);
      return rewriter
          .create<linalg::FillOp>(loc, TypeRange(fillEmptyOp),
                                  ValueRange{singleElem.value()},
                                  ValueRange(fillEmptyOp))
          ->getResult(0);
    }
    return exponent;
  }

  LogicalResult normalizedCstExponentPowf(PatternRewriter &rewriter,
                                          Location loc,
                                          hfusion::ElemwiseBinaryOp op,
                                          Value baseNum, Value exponent) const {
    auto exponentConstOp = getExponentConstOp(exponent, rewriter);
    if (!exponentConstOp)
      return failure();
    auto inType = getElementTypeOrSelf(baseNum.getType());
    auto constFloatAttr = dyn_cast<FloatAttr>(exponentConstOp.getValue());
    auto constFloatValue = constFloatAttr.getValue();
    llvm::APFloat zeroFloat(constFloatValue.getSemantics(), 0);
    if (constFloatValue.isZero()) {
      auto oneConst = rewriter.create<arith::ConstantOp>(
          op->getLoc(), inType, rewriter.getFloatAttr(inType, 1));
      auto fillEmptyOp = utils::createEmptyOp(rewriter, loc, baseNum);
      auto fillOp = rewriter
                        .create<linalg::FillOp>(loc, TypeRange(fillEmptyOp),
                                                ValueRange{oneConst},
                                                ValueRange(fillEmptyOp))
                        ->getResult(0);
      rewriter.replaceOp(op, fillOp);
      return success();
    }

    llvm::APFloat halfFloat(constFloatValue.getSemantics(), "5e-1");
    if (constFloatValue == halfFloat) {
      createSqrtOp(op, rewriter, baseNum);
      return success();
    }

    float constValue = constFloatValue.convertToFloat();
    float intValue = std::round(constValue);
    const int upperLimit = 3;
    if (constFloatValue.isInteger() && intValue <= upperLimit &&
        intValue >= 1) {
      auto resPower =
          calculatePower(rewriter, loc, baseNum, static_cast<int>(intValue));
      rewriter.replaceOp(op, resPower);
      return success();
    }
    return failure();
  }

  /// is_inf = !(abs(input) == inf)
  Value isFinite(PatternRewriter &rewriter, Location loc, Value input) const {
    auto elementType = getElementTypeOrSelf(input.getType());
    // constantOp for inf
    auto constInf = utils::createConstantOp<double>(
        rewriter, loc, elementType, std::numeric_limits<double>::infinity());
    /// abs_input = abs(input)
    auto absInit = utils::createEmptyOp(rewriter, loc, input);
    auto absInput =
        hfusion::createUnaryOp<linalg::ElemwiseUnaryOp, linalg::UnaryFn,
                               linalg::UnaryFnAttr>(
            rewriter, loc, linalg::UnaryFn::abs, ValueRange(input),
            ValueRange(absInit))
            ->getResult(0);

    /// is_infinite = abs_input == inf
    auto isInfinite =
        createCmpOp(rewriter, loc, absInput, constInf, hfusion::CompareFn::veq)
            ->getResult(0);
    auto isFiniteInit = utils::createEmptyOp(rewriter, loc, isInfinite);

    /// is_finite = !is_infinite
    return hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp, hfusion::UnaryFn,
                                  hfusion::UnaryFnAttr>(
               rewriter, loc, hfusion::UnaryFn::vnot, ValueRange(isInfinite),
               ValueRange(isFiniteInit))
        ->getResult(0);
  }

  /// is_nan = x < 0 and x is finite and y is finite and y is not integer
  Value isPowfNanResult(PatternRewriter &rewriter, Location loc, Value baseNum,
                        Value exponent) const {
    /// step1: mask1 = x < 0 and x is finite
    ///   1. is_neg = x < 0
    auto constZero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32Type(),
        rewriter.getFloatAttr(rewriter.getF32Type(), 0.0));
    auto isNeg =
        createCmpOp(rewriter, loc, baseNum, constZero, hfusion::CompareFn::vlt)
            ->getResult(0);
    ///   2. is_x_finite = is_finite(x)
    auto isXFinite = isFinite(rewriter, loc, baseNum);
    auto mask1 = createVandOp(rewriter, loc, isNeg, isXFinite)->getResult(0);

    /// step2: mask2 = y is finite and y is not integer
    ///   1. is_y_finite = is_finite(y)
    auto isYFinite = isFinite(rewriter, loc, exponent);
    ///   2. is_y_float = !isInteger(y)
    auto isInteger = judgeIntegerValue(rewriter, loc, baseNum, exponent);
    auto vnotInit = utils::createEmptyOp(rewriter, loc, isInteger);
    auto isYFloat =
        hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp, hfusion::UnaryFn,
                               hfusion::UnaryFnAttr>(
            rewriter, loc, hfusion::UnaryFn::vnot, ValueRange(isInteger),
            ValueRange(vnotInit))
            ->getResult(0);
    auto mask2 = createVandOp(rewriter, loc, isYFinite, isYFloat)->getResult(0);

    /// step3: is_nan = mask1 and mask2
    return createVandOp(rewriter, loc, mask1, mask2)->getResult(0);
  }

  // is_zero_pow_zero = y == 0
  Value isZeroPowZeroResult(PatternRewriter &rewriter, Location loc,
                            Value exponent) const {
    /// step1: mask = y == 0
    auto constZero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32Type(),
        rewriter.getFloatAttr(rewriter.getF32Type(), 0.0));
    auto mask =
        createCmpOp(rewriter, loc, exponent, constZero, hfusion::CompareFn::veq)
            ->getResult(0);
    return mask;
  }

  LogicalResult normalizePowf(PatternRewriter &rewriter,
                              hfusion::ElemwiseBinaryOp op) const {
    auto inputs = op.getDpsInputs();
    Value baseNum = inputs[0];
    Value exponent = inputs[1];
    Location loc = op->getLoc();
    if (succeeded(
            normalizedCstExponentPowf(rewriter, loc, op, baseNum, exponent)))
      return success();

    // after support scalar value for hfusion op, delete the getExponet func
    // here and directly use the exponent
    auto expTensor = getExponent(rewriter, baseNum, exponent, loc);
    Value isNegativeCond = isNegCondition(rewriter, baseNum, expTensor, loc);
    Value negComRes =
        calculateNegativeCompute(rewriter, baseNum, expTensor, loc);
    Value posComRes =
        calculatePositiveCompute(rewriter, baseNum, exponent, loc);
    auto partialRes0InitOp = utils::createEmptyOp(rewriter, loc, baseNum);
    auto partialRes0 =
        rewriter
            .create<hfusion::SelectOp>(
                loc, TypeRange(partialRes0InitOp),
                ValueRange({isNegativeCond, negComRes, posComRes}),
                ValueRange(partialRes0InitOp))
            ->getResult(0);

    auto inType = getElementTypeOrSelf(baseNum.getType());
    Value constOne = rewriter.create<arith::ConstantOp>(
        loc, inType, rewriter.getFloatAttr(inType, 1.0));
    Value boundaryCondForOne =
        genBoundaryConditionForOne(rewriter, baseNum, expTensor, loc);
    auto partialRes1InitOp = utils::createEmptyOp(rewriter, loc, baseNum);
    auto partialRes1 =
        rewriter
            .create<hfusion::SelectOp>(
                loc, TypeRange(partialRes1InitOp),
                ValueRange({boundaryCondForOne, constOne, partialRes0}),
                ValueRange(partialRes1InitOp))
            ->getResult(0);

    auto floatTy = cast<mlir::FloatType>(inType);
    Value constNan = rewriter.create<arith::ConstantOp>(
        loc, inType,
        rewriter.getFloatAttr(inType,
                              APFloat::getNaN(floatTy.getFloatSemantics())));
    Value isNanCond = isPowfNanResult(rewriter, loc, baseNum, expTensor);
    auto partialRes2InitOp = utils::createEmptyOp(rewriter, loc, baseNum);
    auto partialRes2 = rewriter
                           .create<hfusion::SelectOp>(
                               loc, TypeRange(partialRes2InitOp),
                               ValueRange({isNanCond, constNan, partialRes1}),
                               ValueRange(partialRes2InitOp))
                           ->getResult(0);

    Value isZeroPowZeroCond = isZeroPowZeroResult(rewriter, loc, exponent);
    auto partialRes3InitOp = utils::createEmptyOp(rewriter, loc, baseNum);
    auto partialRes3 =
        rewriter
            .create<hfusion::SelectOp>(
                loc, TypeRange(partialRes3InitOp),
                ValueRange({isZeroPowZeroCond, constOne, partialRes2}),
                ValueRange(partialRes3InitOp))
            ->getResult(0);

    rewriter.replaceOp(op, partialRes3);
    return success();
  }

  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    if (op.getFun() != hfusion::BinaryFn::powf) {
      return failure();
    }

    auto inputs = op.getDpsInputs();
    Value baseNum = inputs[0];
    auto inType = getElementTypeOrSelf(baseNum.getType());
    if (!inType.isF16() && !inType.isF32())
      return failure();

    return normalizePowf(rewriter, op);
  }
};

void populateNormalizeModPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeModOp>(patterns.getContext());
}

void populateNormalizePrimaryMathPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeCeilandFloorOp>(ctx);
  patterns.add<NormalizeLogLikeOp>(ctx);
  patterns.add<NormalizeLog1pOp>(ctx);
  patterns.add<NormalizeReduceMinMaxNumFOp>(ctx);
  patterns.add<NormalizeElemwiseMinMaxNumFOp<BinaryFn::maxnumf>>(ctx);
  patterns.add<NormalizeElemwiseMinMaxNumFOp<BinaryFn::minnumf>>(ctx);
  patterns.add<NormalizeExp2Op>(ctx);
  patterns.add<NormalizeExpM1Op>(ctx);
  patterns.add<NormalizeErfOp>(ctx);
}

void populateNormalizeLateMathPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeIlogbOp>(ctx);
  patterns.add<NormalizeLdexpOp>(ctx);
  patterns.add<NormalizePowfOp>(ctx);
}
} // namespace mlir::hfusion
