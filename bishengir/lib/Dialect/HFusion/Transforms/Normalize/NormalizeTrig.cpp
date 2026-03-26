//===- NormalizeTrig.cpp -----------------------------------------*- C++ -*-===//
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

// normalize sin(x) to sinTayler(norm(x,x_round,0.0))*sign(x_round), where
// round_x=round(input_x*(1/pi))
struct NormalizeSinOp : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::UnaryFn::sin) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");

    // round_x=round(input_x*(1/pi))
    // 1/pi=0.3183098733425140380859375
    Value input = op.getDpsInputs()[0];
    if (inType.isF16()) {
      // for high precision, cast src to fp32 and compute and then cast it back
      // TODO: remove cast after enable automatical high precision computing
      input = hfusion::castTo(rewriter, input, rewriter.getF32Type(),
                              hfusion::RoundMode::ROUND);
    }
    auto loc = op->getLoc();
    auto emptyOp = utils::createEmptyOp(rewriter, loc, input);
    auto elementType = getElementTypeOrSelf(input.getType());
    auto piRecOp = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 1 / (double)M_PI));
    auto inputDivPi =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul, ValueRange{input, piRecOp},
            ValueRange(emptyOp))
            ->getResult(0);

    auto xRound = hfusion::castTo(rewriter, inputDivPi, rewriter.getF32Type(),
                                  hfusion::RoundMode::ROUND);

    // norm_x = x-round(x/pi)*(pi1+pi2+pi3+pi4+pi5)+offset
    // (pi1+pi2+pi3+pi4+pi5) approximates pi
    const llvm::SmallVector<double> piApproParams = {
        3.140625, 0.0009670257568359375, 6.2771141529083251953125e-7,
        1.21644916362129151821136474609375e-10,
        -1.0290623200529979163359041220560e-13};
    auto normInput = norm(rewriter, loc, input, xRound, piApproParams, 0.0);

    // x_res = sinTayler(norm_x)

    auto sinTaylerNorm =
        tayler<hfusion::TaylerMode::SIN>(rewriter, loc, normInput, 5);

    // sign(round_x)=floor(x_round/2)*4- x_round*(2)+1
    auto signX = sign<hfusion::TaylerMode::SIN>(rewriter, loc, xRound);

    Value res = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                        linalg::BinaryFn, linalg::BinaryFnAttr>(
                    rewriter, loc, linalg::BinaryFn::mul,
                    ValueRange{sinTaylerNorm, signX}, ValueRange(emptyOp))
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

/// normalize cos(x)
/// cos(x) = sin(x+pi/2)
///        = sinTayler(norm(x+pi/2,x_round,0.0))*sign(x_round),
/// where
/// round_x = round((x+pi/2)*(1/pi))
///         = sinTayler(norm(x,x_round,pi/2))*sign(x_round),
/// where
/// round_x = round(x*(1/pi)+0.5)
struct NormalizeCosOp : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;

  Value computeRoundX(PatternRewriter &rewriter, Location loc,
                      Value input) const {
    auto emptyOp = utils::createEmptyOp(rewriter, loc, input);
    auto elementType = getElementTypeOrSelf(input.getType());
    auto piRecOp = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 1 / (double)M_PI));
    auto inputDivPi =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul, ValueRange{input, piRecOp},
            ValueRange(emptyOp))
            ->getResult(0);
    auto halfOp = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 0.5));
    auto inputInit =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::add,
            ValueRange{inputDivPi, halfOp}, ValueRange(emptyOp))
            ->getResult(0);

    return hfusion::castTo(rewriter, inputInit, rewriter.getF32Type(),
                           hfusion::RoundMode::ROUND);
  }

  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::UnaryFn::cos) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");

    Value input = op.getDpsInputs()[0];
    if (inType.isF16()) {
      // for high precision, cast src to fp32 and compute and then cast it back
      input = hfusion::castTo(rewriter, input, rewriter.getF32Type(),
                              hfusion::RoundMode::ROUND);
    }

    // step 1: compute round_x
    // round_x = round(input_x*(1/pi)+0.5)
    auto loc = op->getLoc();
    auto xRound = computeRoundX(rewriter, loc, input);

    // step 2: compute norm(x, x_round, pi/2)
    const llvm::SmallVector<double> piApproParams = {
        3.140625, 0.0009670257568359375, 6.2771141529083251953125e-7,
        1.21644916362129151821136474609375e-10,
        -1.0290623200529979163359041220560e-13};
    auto normInput =
        norm(rewriter, loc, input, xRound, piApproParams, (double)M_PI / 2);

    // step 3: sinTayler(norm(x,x_round,pi/2))
    auto cosTayler =
        tayler<hfusion::TaylerMode::SIN>(rewriter, loc, normInput, 5);

    // step 4: compute sign(x_round)
    auto signX = sign<hfusion::TaylerMode::SIN>(rewriter, loc, xRound);

    // step 5: compute cos(x) = sinTayler(norm(x,x_round,pi/2))*sign(x_round)
    auto emptyOp = utils::createEmptyOp(rewriter, loc, input);
    Value res = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                        linalg::BinaryFn, linalg::BinaryFnAttr>(
                    rewriter, loc, linalg::BinaryFn::mul,
                    ValueRange{cosTayler, signX}, ValueRange(emptyOp))
                    ->getResult(0);

    if (inType.isF16()) {
      res = hfusion::castTo(rewriter, res, rewriter.getF16Type(),
                            hfusion::RoundMode::ROUND);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct HighPrecisionNormalizeSinOp
    : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::UnaryFn::sin) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");

    Value input = op.getDpsInputs()[0];
    if (inType.isF16()) {
      // for high precision, cast src to fp32 and compute and then cast it back
      // TODO: remove cast after enable automatical high precision computing
      input = hfusion::castTo(rewriter, input, rewriter.getF32Type(),
                              hfusion::RoundMode::ROUND);
    }
    auto resOr = buildSinOrCos(rewriter, op, input, CalcMode::SIN);
    if (failed(resOr))
      return failure();
    Value res = *resOr;

    if (inType.isF16()) {
      // TODO: remove cast after enable automatical high precision computing
      res = hfusion::castTo(rewriter, res, rewriter.getF16Type(),
                            hfusion::RoundMode::ROUND);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct HighPrecisionNormalizeCosOp
    : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::UnaryFn::cos) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");
    Value input = op.getDpsInputs()[0];
    if (inType.isF16()) {
      // for high precision, cast src to fp32 and compute and then cast it back
      input = hfusion::castTo(rewriter, input, rewriter.getF32Type(),
                              hfusion::RoundMode::ROUND);
    }

    auto resOr = buildSinOrCos(rewriter, op, input, CalcMode::COS);
    if (failed(resOr))
      return failure();
    Value res = *resOr;

    if (inType.isF16()) {
      res = hfusion::castTo(rewriter, res, rewriter.getF16Type(),
                            hfusion::RoundMode::ROUND);
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

/// step 1: normalize x into [-10000, 10000],
/// 1.1 when x's value is too large, the first caculator of _do_taylor will be
/// overflow.
/// 1.2 when epsilon is 0.0001, the approximate value of `tan(pi / 2 - 0.0001)`
/// is 10000, thus normalize data [-10000, 10000]
/// step 2: atan(x) = min(taylor(x), pi / 4 + taylor((x - 1)/(x+1)))
/// 2.1 if abs(x) <= 1,  atan(x) = x - x^3/3 + x^5/5 - x^7/7 ...
/// 2.2 if abs(x) > 1, atan(x) = arctan(1) + arctan((x - 1)/(x + 1)) = pi / 4 +
/// arctan((x - 1)/(x + 1)).
/// step 3: tayor(x) = min(taylor, taylor(y) + atan((x - y)/(1 + xy))).
/// It is with higher precision. where:
/// tan(y) = pi / 8, y = tan(pi / 8) = 0.4142135623730950
struct NormalizeAtanOp : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;

  Value getatanTaylorRes(PatternRewriter &rewriter, Location loc, Value input,
                         int taylerExpansionNum) const {
    /// 1. nomalize x into (x-y)/(1+xy)
    const float M_PI_8 = M_PI / 8;
    const float TAN_M_PI_8 = std::tan(M_PI_8);
    auto elementType = getElementTypeOrSelf(input);
    arith::ConstantOp constOp = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, TAN_M_PI_8));
    Value emptyOne = utils::createEmptyOp(rewriter, loc, input);
    auto fillOp = rewriter.create<linalg::FillOp>(
        loc, TypeRange(emptyOne), ValueRange({constOp->getResults()[0]}),
        ValueRange({emptyOne}));
    /// mulOp = x*y
    auto mulInit = utils::createEmptyOp(rewriter, loc, input);
    auto *mulOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul,
            ValueRange{input, fillOp->getResults()[0]}, mulInit);

    /// addOp = 1 + x*y
    arith::ConstantOp constOne = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 1.0));
    auto *addOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::add,
            ValueRange{mulOp->getResults()[0], constOne->getResults()[0]},
            mulInit);
    /// subOp = x - y
    auto subInit = utils::createEmptyOp(rewriter, loc, input);
    auto *subOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::sub,
            ValueRange{input, fillOp->getResults()[0]}, subInit);
    /// divOp = (x-y)/(1+xy)
    auto *divOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::div,
            ValueRange{subOp->getResults()[0], addOp->getResults()[0]},
            subInit);
    /// absOp = abs((x-y)/(1+xy))
    auto absOP = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                        linalg::UnaryFn, linalg::UnaryFnAttr>(
        rewriter, loc, linalg::UnaryFn::abs, ValueRange{divOp->getResults()[0]},
        ValueRange(subInit));

    /// 2: atan((x-y)/(1+xy))
    auto res1 = tayler<hfusion::TaylerMode::ATAN>(
        rewriter, loc, absOP->getResults()[0], taylerExpansionNum);

    /// 3: atan((x-y)/(1+xy)) + pi /8
    arith::ConstantOp constM_PI_8 = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, M_PI_8));
    auto *res2 =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::add,
            ValueRange{res1, constM_PI_8->getResults()[0]}, subInit);
    return res2->getResults()[0];
  }

  /// if x > 0 and x < tan(pi/8):
  /// atan(x) = x - x^3/3 + x^5/5 - x^7/7 ...
  /// elif x > tan(pi/8) and x < tan(pi/4):
  /// atan(x) = atan(y) + atan((x-y)/(1+xy))
  Value atanTaylor(PatternRewriter &rewriter, Location loc, Value input,
                   int taylerExpansionNum) const {
    // step1: res0 = atan(x)
    auto res0 = tayler<hfusion::TaylerMode::ATAN>(rewriter, loc, input,
                                                  taylerExpansionNum);

    /// step 2: atan(x) = atan(y) + atan((x-y)/(1+xy))
    Value res2 = getatanTaylorRes(rewriter, loc, input, taylerExpansionNum);

    /// 3. atan(x) = min(res0, res2)
    auto atanInit = utils::createEmptyOp(rewriter, loc, input);
    auto *minOp =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, hfusion::BinaryFn::minf, ValueRange{res0, res2},
            atanInit);
    return minOp->getResults()[0];
  }

  // y = (x - 1) / (x + 1)
  Value normalizeInputValue(PatternRewriter &rewriter, Location loc,
                            Value input) const {
    // 1.define one
    auto elementType = getElementTypeOrSelf(input);
    arith::ConstantOp positiveOne = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 1.0));
    arith::ConstantOp negetiveOne = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, -1.0));

    // 2. sub = vadd(input, -one)
    auto subInit = utils::createEmptyOp(rewriter, loc, input);
    auto *subOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::add,
            ValueRange{input, negetiveOne->getResults()[0]}, subInit);

    // 3. add = vadd(input, one)
    auto addInit = utils::createEmptyOp(rewriter, loc, input);
    auto *addOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::add,
            ValueRange{input, positiveOne->getResults()[0]}, addInit);

    // 4. div = vdiv(sub, add)
    auto divInit = utils::createEmptyOp(rewriter, loc, input);
    auto *divOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::div,
            ValueRange{subOp->getResults()[0], addOp->getResults()[0]},
            divInit);
    // 5.vabs(div)
    auto absOP = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                        linalg::UnaryFn, linalg::UnaryFnAttr>(
        rewriter, loc, linalg::UnaryFn::abs, ValueRange{divOp->getResults()[0]},
        ValueRange(divInit));

    return absOP->getResults()[0];
  }

  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    if (op.getFun() != hfusion::UnaryFn::atan) {
      return failure();
    }
    if (!getElementTypeOrSelf(op.getType(0)).isF16() &&
        !getElementTypeOrSelf(op.getType(0)).isF32()) {
      return failure();
    }

    Value input = op.getDpsInputs()[0];
    auto elementType = getElementTypeOrSelf(input);
    if (elementType.isF16()) {
      // for high precision, cast src to fp32 and compute and then cast it back
      // TODO: remove cast after enable automatical high precision computing
      input = hfusion::castTo(rewriter, input, rewriter.getF32Type(),
                              hfusion::RoundMode::ROUND);
    }

    auto loc = op->getLoc();
    /// step 1: normalize x into [-10000, 10000], and abs(x)
    auto clipedInput = ClipInput(rewriter, loc, input, 10000, -10000);
    auto clipedInit = utils::createEmptyOp(rewriter, loc, clipedInput);
    auto absOP = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                        linalg::UnaryFn, linalg::UnaryFnAttr>(
        rewriter, op->getLoc(), linalg::UnaryFn::abs, ValueRange{clipedInput},
        clipedInit);
    Value clipedRangeInput = absOP->getResults()[0];

    /// step 2: atan(x) = min(taylor(x), pi / 4 + taylor((x - 1)/(x+1)))
    /// res0 = taylor(x)
    auto res0 = atanTaylor(rewriter, loc, clipedRangeInput, 7);

    /// res1 = pi / 4 + taylor((x - 1)/(x+1)), where y = (x - 1)/(x+1)
    auto y = normalizeInputValue(rewriter, loc, clipedRangeInput);
    auto taylorY = atanTaylor(rewriter, loc, y, 7);
    arith::ConstantOp constM_PI_4 = rewriter.create<arith::ConstantOp>(
        loc, getElementTypeOrSelf(input),
        rewriter.getFloatAttr(getElementTypeOrSelf(input), M_PI_4));
    Value res1Op = utils::createEmptyOp(rewriter, loc, input);
    auto *res1 =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::add,
            ValueRange{taylorY, constM_PI_4->getResults()[0]}, res1Op);

    /// atan(x) = min(res1, res2)
    Value atanInit = utils::createEmptyOp(rewriter, loc, input);
    auto *atan =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, hfusion::BinaryFn::minf,
            ValueRange{res0, res1->getResults()[0]}, atanInit);

    /// res = sign(x) * atan(x)
    auto signX = sign<hfusion::TaylerMode::ATAN>(rewriter, loc, input);
    Value resInit = utils::createEmptyOp(rewriter, loc, input);
    Value res = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                        linalg::BinaryFn, linalg::BinaryFnAttr>(
                    rewriter, loc, linalg::BinaryFn::mul,
                    ValueRange{atan->getResults()[0], signX}, resInit)
                    ->getResult(0);
    if (elementType.isF16()) {
      // TODO: remove cast after enable automatical high precision computing
      res = hfusion::castTo(rewriter, res, rewriter.getF16Type(),
                            hfusion::RoundMode::ROUND);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

/// y = tan(x)
/// step1: xround = round(x / pi)
/// step2: Calculate res_down1 res_down2
///     p0=3.140625 p1=0.0009670257568359375 p2=6.2771141529083251953125e-7
///     p3=1.21644916362129151821136474609375e-10
///     p4=-1.0290623200529979163359041220560e-13
///     kpi0 = xround * p0; kpi1 = xround * p1...
///     res_down1=x-kpi0-kpi1+1.57079-kpi2+(-0.0000000437)-kpi_3-kpi_4
///     res_down2=x-kpi0-kpi1+(-1.57079)-kpi2+0.00000004371-kpi_3-kpi_4
/// step3: z = x - kpi0 - kpi1 - kpi2 - kpi3 - kpi4 z2 = z * z
/// step4: Calculate res_up res_down
///     CST0 = 0.0698520831551998762793
///     T1 = -6.8711573651634203789 T2 = 61.20362572811089435388
///     res_up = ((((z2*CST0)+T1)*z2)+T2)*z
///     res_down = (z2 - 24.8048928861126769186219) * res_down1 * res_down2
/// step5: y = res_up / res_down
/// note: Changing the order of operations within res_down1/res_down2 may
/// cause small precision errors.
struct NormalizeTanOp : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;

  Value getResDown(PatternRewriter &rewriter, Location loc, Value input,
                   const llvm::SmallVector<double> &offsetCoeff) const {
    Value resInit = utils::createEmptyOp(rewriter, loc, input);
    Value res = input;
    linalg::ElemwiseBinaryOp mulOp;
    auto inType = getElementTypeOrSelf(input.getType());
    for (double coeff : offsetCoeff) {
      arith::ConstantOp constOp = rewriter.create<arith::ConstantOp>(
          loc, inType, rewriter.getFloatAttr(inType, coeff));
      auto curRes =
          hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                  linalg::BinaryFnAttr>(
              rewriter, loc, linalg::BinaryFn::add,
              ValueRange{res, constOp->getResults()[0]}, ValueRange(resInit))
              ->getResult(0);
      res = curRes;
    }
    return res;
  }

  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::UnaryFn::tan) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    assert((inType.isF16() || inType.isF32()) &&
           "only support input Type is f16 or f32");

    Value input = op.getDpsInputs()[0];
    if (inType.isF16()) {
      // for precision, cast input to fp32 and compute and then cast it back.
      // TODO: remove cast after enable automatical high precision computing
      input = hfusion::castTo(rewriter, input, rewriter.getF32Type(),
                              hfusion::RoundMode::ROUND);
    }

    auto loc = op->getLoc();
    auto emptyOp = utils::createEmptyOp(rewriter, loc, input);
    auto elementType = getElementTypeOrSelf(input.getType());
    /// step 1: xround = round(x/pi)
    auto piRecOp = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 1 / (double)M_PI));
    auto inputDivPi =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul, ValueRange{input, piRecOp},
            ValueRange(emptyOp))
            ->getResult(0);
    auto xRound = hfusion::castTo(rewriter, inputDivPi, rewriter.getF32Type(),
                                  hfusion::RoundMode::ROUND);

    /// step2: Calculate res_down1 res_down2
    /// p0=3.140625 p1=0.0009670257568359375 p2=6.2771141529083251953125e-7
    /// p3=1.21644916362129151821136474609375e-10
    /// p4=-1.0290623200529979163359041220560e-13
    /// kpi0 = xround * p0; kpi1 = xround * p1...
    /// res_down1=x-kpi0-kpi1+1.57079-kpi2+(-0.0000000437)-kpi_3-kpi_4
    /// res_down2=x-kpi0-kpi1+(-1.57079)-kpi2+0.00000004371-kpi_3-kpi_4
    const llvm::SmallVector<double> piApproParams = {
        3.140625, 0.0009670257568359375, 6.2771141529083251953125e-7,
        1.21644916362129151821136474609375e-10,
        -1.0290623200529979163359041220560e-13};

    const llvm::SmallVector<double> piApproParamsPart1(
        piApproParams.begin(), piApproParams.begin() + 2);
    Value resDownPart1 = norm(rewriter, loc, input, xRound, piApproParamsPart1);
    Value resDown1 =
        getResDown(rewriter, loc, resDownPart1, {1.57079637050628662109375});
    Value resDown2 =
        getResDown(rewriter, loc, resDownPart1, {-1.57079637050628662109375});

    const llvm::SmallVector<double> piApproParamsPart2 = {piApproParams[2]};
    resDown1 = norm(rewriter, loc, resDown1, xRound, piApproParamsPart2);
    resDown2 = norm(rewriter, loc, resDown2, xRound, piApproParamsPart2);
    resDown1 =
        getResDown(rewriter, loc, resDown1, {-0.00000004371139000189375});
    resDown2 = getResDown(rewriter, loc, resDown2, {0.00000004371139000189375});

    const llvm::SmallVector<double> piApproParamsPart3(piApproParams.end() - 2,
                                                       piApproParams.end());
    resDown1 = norm(rewriter, loc, resDown1, xRound, piApproParamsPart3);
    resDown2 = norm(rewriter, loc, resDown2, xRound, piApproParamsPart3);

    /// step3: z = x - kpi0 - kpi1 - kpi2 - kpi3 - kpi4 z2 = z * z
    const llvm::SmallVector<double> extraPiApproParams(piApproParams.end() - 3,
                                                       piApproParams.end());
    auto normInput =
        norm(rewriter, loc, resDownPart1, xRound, extraPiApproParams);

    auto suareInit = utils::createEmptyOp(rewriter, loc, normInput);
    auto *squareOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul,
            ValueRange{normInput, normInput}, ValueRange(suareInit));

    /// step4: Calculate res_up res_down
    /// CST0 = 0.0698520831551998762793
    /// T1 = -6.8711573651634203789 T2 = 61.20362572811089435388
    /// res_up = ((((z2 * CST0) + T1) * z2) + T2) * z
    /// res_down = (z2 - 24.8048928861126769186219) * res_down1 * res_down2
    double CST0 = 0.0698520831551998762793;
    auto numerInit = utils::createEmptyOp(rewriter, loc, normInput);
    auto constValInit = rewriter.create<arith::ConstantOp>(
        loc, getElementTypeOrSelf(input.getType()),
        rewriter.getFloatAttr(getElementTypeOrSelf(input.getType()), CST0));
    auto *numerInitOp = hfusion::createBinaryOp<
        linalg::ElemwiseBinaryOp, linalg::BinaryFn, linalg::BinaryFnAttr>(
        rewriter, loc, linalg::BinaryFn::mul,
        ValueRange{squareOp->getResults()[0], constValInit->getResults()[0]},
        ValueRange(numerInit));

    Value numerRes = genPolyExpr(
        rewriter, loc, squareOp->getResults()[0], numerInitOp->getResults()[0],
        llvm::SmallVector<double>{-6.8711573651634203789});

    auto constVal = rewriter.create<arith::ConstantOp>(
        loc, getElementTypeOrSelf(input.getType()),
        rewriter.getFloatAttr(getElementTypeOrSelf(input.getType()),
                              61.20362572811089435388));

    auto *numerAddOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::add,
            ValueRange{numerRes, constVal->getResults()[0]},
            ValueRange(numerRes));
    auto *numermulOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul,
            ValueRange{numerAddOp->getResults()[0], normInput},
            ValueRange(numerRes));

    const double const1 = -24.8048928861126769186219;
    auto constValInit1 = rewriter.create<arith::ConstantOp>(
        loc, getElementTypeOrSelf(input.getType()),
        rewriter.getFloatAttr(getElementTypeOrSelf(input.getType()), const1));

    auto resDownInit = utils::createEmptyOp(rewriter, loc, normInput);
    auto *subOp = hfusion::createBinaryOp<
        linalg::ElemwiseBinaryOp, linalg::BinaryFn, linalg::BinaryFnAttr>(
        rewriter, loc, linalg::BinaryFn::add,
        ValueRange{squareOp->getResults()[0], constValInit1->getResults()[0]},
        ValueRange(resDownInit));
    auto *mulOp1 =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul,
            ValueRange{subOp->getResults()[0], resDown1},
            ValueRange(resDownInit));
    auto *mulOp2 =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul,
            ValueRange{mulOp1->getResults()[0], resDown2},
            ValueRange(resDownInit));

    /// step 5: res = res_up/res_down
    auto emptyResOp = utils::createEmptyOp(rewriter, op->getLoc(), normInput);
    Value res =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::div,
            ValueRange{numermulOp->getResults()[0], mulOp2->getResults()[0]},
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

/// Normalize tanh(x)=(exp(x)-exp(-x))/(exp(x)+exp(-x))
///                  =(exp(2x)-1)/(exp(2x)+1)
///                  =(exp(2x')-1)/(exp(2x')+1),
/// where x' = clip(x, [-8.8, 8.8]), so the epison error of tanh(x') <= 1e-8
struct NormalizeTanhOp : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::UnaryFn::tanh) {
      return failure();
    }

    if (!getElementTypeOrSelf(op.getType(0)).isF16() &&
        !getElementTypeOrSelf(op.getType(0)).isF32()) {
      return failure();
    }

    Value input = op.getDpsInputs()[0];
    auto elementType = getElementTypeOrSelf(input);
    if (elementType.isF16()) {
      // for high precision, cast src to fp32 and compute and then cast it back
      // TODO: remove cast after enable automatical high precision computing
      input = hfusion::castTo(rewriter, input, rewriter.getF32Type(),
                              hfusion::RoundMode::ROUND);
    }
    auto loc = op->getLoc();
    // step 1: When x's value is too large, exp(2x) will be overflow.
    // So clip it to [-8.8, 8.8], the epison is ie-8.
    auto clipedInput = ClipInput(rewriter, loc, input, 8.8, -8.8);

    // step 2.1: y = exp(2x)
    auto targetType = getElementTypeOrSelf(input);
    auto constTwo = rewriter.create<arith::ConstantOp>(
        loc, targetType, rewriter.getFloatAttr(rewriter.getF32Type(), 2.0));

    Value mulInit = utils::createEmptyOp(rewriter, loc, input);
    auto mulOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul,
            ValueRange{clipedInput, constTwo->getResults()[0]}, mulInit);

    auto expOp = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                        linalg::UnaryFn, linalg::UnaryFnAttr>(
        rewriter, loc, linalg::UnaryFn::exp, mulOp->getResults()[0], mulInit);

    // step 2.2: numer = exp(2x) - 1
    auto constMinusOne = rewriter.create<arith::ConstantOp>(
        loc, targetType, rewriter.getFloatAttr(rewriter.getF32Type(), -1.0));
    Value numerInit = utils::createEmptyOp(rewriter, loc, input);
    auto numerRes =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::add,
            ValueRange{expOp->getResults()[0], constMinusOne->getResults()[0]},
            numerInit);

    // step 2.3: demon = exp(2x) + 1
    auto constPosOne = rewriter.create<arith::ConstantOp>(
        loc, targetType, rewriter.getFloatAttr(rewriter.getF32Type(), 1.0));
    Value demonInit = utils::createEmptyOp(rewriter, loc, input);
    auto demonRes =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::add,
            ValueRange{expOp->getResults()[0], constPosOne->getResults()[0]},
            demonInit);

    // step 2.4: tanh(x) = numer / demon
    Value res =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::div,
            ValueRange{numerRes->getResults()[0], demonRes->getResults()[0]},
            numerInit)
            ->getResult(0);

    if (elementType.isF16()) {
      // TODO: remove cast after enable automatical high precision computing
      res = hfusion::castTo(rewriter, res, rewriter.getF16Type(),
                            hfusion::RoundMode::ROUND);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

void populateNormalizeTrigPatterns(RewritePatternSet &patterns,
                                   bool enableHighPrecision) {
  MLIRContext *ctx = patterns.getContext();
  if (enableHighPrecision) {
    patterns.add<HighPrecisionNormalizeSinOp>(ctx);
    patterns.add<HighPrecisionNormalizeCosOp>(ctx);
  } else {
    patterns.add<NormalizeSinOp>(ctx);
    patterns.add<NormalizeCosOp>(ctx);
  }
  patterns.add<NormalizeAtanOp>(ctx);
  patterns.add<NormalizeTanOp>(ctx);
  patterns.add<NormalizeTanhOp>(ctx);
}
} // namespace mlir::hfusion
