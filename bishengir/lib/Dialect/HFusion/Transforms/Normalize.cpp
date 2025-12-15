//===- Normalize .cpp -------------------- Normalize HFusion  -------------===//
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
//
// This pass is for normalizing HFusion.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_NORMALIZE
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hfusion-normalize-ops"

using namespace mlir;
using namespace mlir::hfusion;

// norm(x,x_round,offset) = x-x_round*(pi1+pi2+pi3+pi4+pi5)+offset
// (pi1+pi2+pi3+pi4+pi5) approximates pi
static Value norm(PatternRewriter &rewriter, Location loc, Value x,
                  Value xRound, const llvm::SmallVector<double> &piApproParams,
                  std::optional<float> offset = std::nullopt) {
  auto emptyOp = utils::createEmptyOp(rewriter, loc, x);
  auto elementType = getElementTypeOrSelf(x.getType());
  Value resValue = x;
  for (double piApproParam : piApproParams) {
    auto piApproPara = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, piApproParam));
    auto kp = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                      linalg::BinaryFn, linalg::BinaryFnAttr>(
                  rewriter, loc, linalg::BinaryFn::mul,
                  ValueRange{xRound, piApproPara}, ValueRange(emptyOp))
                  ->getResult(0);
    auto x1 = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                      linalg::BinaryFn, linalg::BinaryFnAttr>(
                  rewriter, loc, linalg::BinaryFn::sub,
                  ValueRange{resValue, kp}, ValueRange(emptyOp))
                  ->getResult(0);
    resValue = x1;
  }
  if (offset.has_value()) {
    auto offsetConstant = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, offset.value()));
    return hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                   linalg::BinaryFnAttr>(
               rewriter, loc, linalg::BinaryFn::add,
               ValueRange{resValue, offsetConstant}, ValueRange(emptyOp))
        ->getResult(0);
  }
  return resValue;
}

static SmallVector<double> getTaylerParams(hfusion::TaylerMode taylerMode,
                                           int taylerExpansionNum) {
  SmallVector<double> taylerParams;
  switch (taylerMode) {
  case hfusion::TaylerMode::SIN: {
    taylerParams.push_back(1);
    double taylerAccumulation = 1.0;
    for (int i = 1; i < taylerExpansionNum; i++) {
      taylerAccumulation = taylerAccumulation * (2 * i) * (2 * i + 1) * (-1);
      taylerParams.push_back(1 / taylerAccumulation);
    }
    return taylerParams;
  }
  case hfusion::TaylerMode::ATAN: {
    taylerParams.push_back(1);
    double taylerAccumulation = 1.0;
    for (int i = 1; i < taylerExpansionNum; i++) {
      taylerAccumulation = (i % 2 == 0) ? (2 * i + 1) : (2 * i + 1) * (-1);
      taylerParams.push_back(1 / taylerAccumulation);
    }
    return taylerParams;
  }
  }
  llvm_unreachable("unsupported TaylerMode");
}

static Value getSinSign(PatternRewriter &rewriter, Location loc, Value x) {
  // sign(x)=floor(x/2)*4- x_round*(2)+1
  auto emptyOp = utils::createEmptyOp(rewriter, loc, x);
  auto elementType = getElementTypeOrSelf(x.getType());
  auto half = rewriter.create<arith::ConstantOp>(
      loc, elementType, rewriter.getFloatAttr(elementType, 0.5));
  auto kHalf = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                       linalg::BinaryFn, linalg::BinaryFnAttr>(
                   rewriter, loc, linalg::BinaryFn::mul, ValueRange{x, half},
                   ValueRange(emptyOp))
                   ->getResult(0);
  auto kHalfFloor = hfusion::castTo(rewriter, kHalf, rewriter.getF32Type(),
                                    hfusion::RoundMode::FLOOR);
  auto constFour = rewriter.create<arith::ConstantOp>(
      loc, elementType, rewriter.getFloatAttr(elementType, 4.0));
  auto kHalfFloor4 =
      hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                              linalg::BinaryFnAttr>(
          rewriter, loc, linalg::BinaryFn::mul,
          ValueRange{kHalfFloor, constFour}, ValueRange(emptyOp))
          ->getResult(0);

  auto constMinusTwo = rewriter.create<arith::ConstantOp>(
      loc, elementType, rewriter.getFloatAttr(elementType, -2.0));
  auto k2 = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                    linalg::BinaryFnAttr>(
                rewriter, loc, linalg::BinaryFn::mul,
                ValueRange{x, constMinusTwo}, ValueRange(emptyOp))
                ->getResult(0);

  auto sign = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                      linalg::BinaryFn, linalg::BinaryFnAttr>(
                  rewriter, loc, linalg::BinaryFn::add,
                  ValueRange{kHalfFloor4, k2}, ValueRange(emptyOp))
                  ->getResult(0);

  auto constOne = rewriter.create<arith::ConstantOp>(
      loc, elementType, rewriter.getFloatAttr(elementType, 1.0));
  auto sign1 = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                       linalg::BinaryFn, linalg::BinaryFnAttr>(
                   rewriter, loc, linalg::BinaryFn::add,
                   ValueRange{sign, constOne}, ValueRange(emptyOp))
                   ->getResult(0);
  return sign1;
}

inline double getFPMAX(FloatType fType) {
  if (fType.isF32()) {
    // TODO: make confirmation why TBE process it specially
    return (double)std::pow(2, fType.getWidth() + 30);
  }

  return (double)std::pow(2, fType.getWidth() - 1);
}

inline double getFPMIN(FloatType fType) {
  if (fType.isF32()) {
    // TODO: make confirmation why TBE process it specially
    return (double)std::pow(2, -((int)fType.getWidth() + 30));
  }

  return (double)std::pow(2, -((int)fType.getWidth() - 1));
}

static Value getAtanSign(PatternRewriter &rewriter, Location loc, Value x) {
  // sign(x) = FP_MAX * x /(FP_MIN + FP_MAX *|x|)
  auto elementType = getElementTypeOrSelf(x.getType());
  assert(isa<FloatType>(elementType) && "Only support floatType");
  auto elemFloatType = llvm::dyn_cast<FloatType>(elementType);
  auto FpMaxOp = rewriter.create<arith::ConstantOp>(
      loc, elementType,
      rewriter.getFloatAttr(rewriter.getF32Type(), getFPMAX(elemFloatType)));
  auto FpMinOp = rewriter.create<arith::ConstantOp>(
      loc, elementType,
      rewriter.getFloatAttr(rewriter.getF32Type(), getFPMIN(elemFloatType)));

  auto mulInit = utils::createEmptyOp(rewriter, loc, x);
  auto mulOp = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                       linalg::BinaryFn, linalg::BinaryFnAttr>(
      rewriter, loc, linalg::BinaryFn::mul,
      ValueRange{x, FpMaxOp->getResults()[0]}, ValueRange(mulInit));

  auto addInit = utils::createEmptyOp(rewriter, loc, x);
  auto absOP = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp, linalg::UnaryFn,
                                      linalg::UnaryFnAttr>(
      rewriter, loc, linalg::UnaryFn::abs, ValueRange{mulOp->getResults()[0]},
      ValueRange(addInit));
  auto addOp = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                       linalg::BinaryFn, linalg::BinaryFnAttr>(
      rewriter, loc, linalg::BinaryFn::add,
      ValueRange{absOP->getResults()[0], FpMinOp->getResults()[0]},
      ValueRange(addInit));

  auto divOP = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                       linalg::BinaryFn, linalg::BinaryFnAttr>(
      rewriter, loc, linalg::BinaryFn::div,
      ValueRange({mulOp->getResults()[0], addOp->getResults()[0]}),
      ValueRange(mulInit));
  return divOP->getResults()[0];
}

template <hfusion::TaylerMode taylerMode>
static Value sign(PatternRewriter &rewriter, Location loc, Value x) {
  switch (taylerMode) {
  case hfusion::TaylerMode::SIN: {
    return getSinSign(rewriter, loc, x);
  }
  case hfusion::TaylerMode::ATAN: {
    return getAtanSign(rewriter, loc, x);
  }
  }
  llvm_unreachable("unsupported TaylerMode");
}

Value constructTaylerSeries(PatternRewriter &rewriter, Location loc,
                            Value lastTaylerTerm, Value emptyOp, Value xPow,
                            int taylerExpansionNum,
                            const SmallVector<double> &taylerParams) {
  Value partialRes = lastTaylerTerm;
  auto elementType = getElementTypeOrSelf(xPow.getType());
  for (int i = 0; i < taylerExpansionNum - 2; i++) {
    auto curTaylerParam = rewriter.create<arith::ConstantOp>(
        loc, elementType,
        rewriter.getFloatAttr(elementType,
                              taylerParams[taylerExpansionNum - i - 2]));
    auto curTayerTerm =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::add,
            ValueRange{partialRes, curTaylerParam}, ValueRange(emptyOp))
            ->getResult(0);
    auto curRes =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul,
            ValueRange{curTayerTerm, xPow}, ValueRange(emptyOp))
            ->getResult(0);
    partialRes = curRes;
  }
  return partialRes;
}

// tayler x =
// taylerParams[0]*x+taylerParams[1]*x^3+...+taylerParams[i]*x^(2*i+1)
template <hfusion::TaylerMode taylerMode>
static Value tayler(PatternRewriter &rewriter, Location loc, Value x,
                    int taylerExpansionNum) {
  SmallVector<double> taylerParams =
      getTaylerParams(taylerMode, taylerExpansionNum);

  auto emptyOp = utils::createEmptyOp(rewriter, loc, x);
  auto xPow = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                      linalg::BinaryFn, linalg::BinaryFnAttr>(
                  rewriter, loc, linalg::BinaryFn::mul, ValueRange{x, x},
                  ValueRange(emptyOp))
                  ->getResult(0);

  // Step 1: init the last taylerTerm
  auto elementType = getElementTypeOrSelf(x.getType());
  auto lastTaylerParam = rewriter.create<arith::ConstantOp>(
      loc, elementType,
      rewriter.getFloatAttr(elementType, taylerParams[taylerExpansionNum - 1]));
  auto lastTaylerTerm =
      hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                              linalg::BinaryFnAttr>(
          rewriter, loc, linalg::BinaryFn::mul,
          ValueRange{xPow, lastTaylerParam}, ValueRange(emptyOp))
          ->getResult(0);

  // Step 2: construct the tayler series
  // for i in [0,n-i-2):
  //    partialRes = (partialRes+TaylerParams[n-i-2])*(x^2)
  Value partialRes =
      constructTaylerSeries(rewriter, loc, lastTaylerTerm, emptyOp, xPow,
                            taylerExpansionNum, taylerParams);

  // partialRes1 = (partialRes+1)
  auto constOne = rewriter.create<arith::ConstantOp>(
      loc, elementType, rewriter.getFloatAttr(elementType, 1.0));
  auto partialRes1 =
      hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                              linalg::BinaryFnAttr>(
          rewriter, loc, linalg::BinaryFn::add,
          ValueRange{partialRes, constOne}, ValueRange(emptyOp))
          ->getResult(0);
  // Step 3: multiple common coef
  // tayler(x) = partialRes1*x
  auto res = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                     linalg::BinaryFnAttr>(
                 rewriter, loc, linalg::BinaryFn::mul,
                 ValueRange{partialRes1, x}, ValueRange(emptyOp))
                 ->getResult(0);
  return res;
}

namespace mlir::hfusion {
template <class Derived>
class NormalizeTrigBase : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  explicit NormalizeTrigBase(MLIRContext *ctx, PatternBenefit benefit = 1)
      : OpRewritePattern<hfusion::ElemwiseUnaryOp>(ctx, benefit) {}
  virtual ~NormalizeTrigBase() = default;
  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return failure();
    if (op.getFun() != Derived::kUnaryFn)
      return failure();

    // ---------- FP16 → FP32 ----------
    auto inTy = getElementTypeOrSelf(op.getInputs()[0].getType());
    Value input = op.getDpsInputs()[0];
    if (inTy.isF16())
      input = hfusion::castTo(rewriter, input, rewriter.getF32Type(),
                              hfusion::RoundMode::ROUND);

    auto loc = op->getLoc();
    auto empty = utils::createEmptyOp(rewriter, loc, input);

    // Phase‑1 & Phase‑2
    auto roundInputs = static_cast<const Derived *>(this)->PrepRoundInput(
        rewriter, loc, input, empty);
    auto xFixed = static_cast<const Derived *>(this)->CalculateXFixed(
        rewriter, loc, input, roundInputs, empty);

    // Phase‑3 & Phase‑4
    auto sign = CalculateSign(rewriter, loc, roundInputs[0], empty);
    auto result = static_cast<const Derived *>(this)->CalculateResult(
        rewriter, loc, sign, xFixed, empty);

    // ---------- post process ----------
    Value finalRes =
        static_cast<const Derived *>(this)->postProcess(rewriter, loc, result);

    // ---------- FP32 → FP16 ----------
    if (inTy.isF16())
      finalRes = hfusion::castTo(rewriter, finalRes, rewriter.getF16Type(),
                                 hfusion::RoundMode::ROUND);

    rewriter.replaceOp(op, finalRes);
    return success();
  }
protected:
  template <mlir::linalg::BinaryFn T>
  Value createLinalgBinOp(PatternRewriter &rewriter, Location loc, Value inp1,
                          Value inp2, Value out) const {
    return hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                   linalg::BinaryFnAttr>(
               rewriter, loc, T, ValueRange({inp1, inp2}), out)
        ->getResult(0);
  }

  // a*b + c
  Value mulAdd(PatternRewriter &rewriter, Location loc, Value a, Value b,
               Value c, Value out) const {
    auto mul =
        createLinalgBinOp<linalg::BinaryFn::mul>(rewriter, loc, a, b, out);
    return createLinalgBinOp<linalg::BinaryFn::add>(rewriter, loc, mul, c, out);
  }

  // c - a*b
  Value subMul(PatternRewriter &rewriter, Location loc, Value a, Value b,
               Value c, Value out) const {
    auto mul =
        createLinalgBinOp<linalg::BinaryFn::mul>(rewriter, loc, a, b, out);
    return createLinalgBinOp<linalg::BinaryFn::sub>(rewriter, loc, c, mul, out);
  }

  Value f32Const(PatternRewriter &rewriter, Location loc, float v) const {
    auto f32 = rewriter.getF32Type();
    return rewriter.create<arith::ConstantOp>(loc, f32,
                                              rewriter.getFloatAttr(f32, v));
  }
  // ---------- Phase‑3 ----------
  Value CalculateSign(PatternRewriter &rewriter, Location loc, Value roundInput,
                      Value &empty) const {
    // kover2 = roundInput * HALF
    auto kover2 = createLinalgBinOp<linalg::BinaryFn::mul>(
        rewriter, loc, roundInput, f32Const(rewriter, loc, trig::HALF), empty);
    // floor(kover2)
    auto koverFloor = hfusion::castTo(rewriter, kover2, rewriter.getF32Type(),
                                      hfusion::RoundMode::FLOOR);
    // koverFloor * FOUR
    auto koverFloorM4 = createLinalgBinOp<linalg::BinaryFn::mul>(
        rewriter, loc, koverFloor, f32Const(rewriter, loc, trig::FOUR), empty);
    // k2 = roundInput * NEG_TWO
    auto k2 = createLinalgBinOp<linalg::BinaryFn::mul>(
        rewriter, loc, roundInput, f32Const(rewriter, loc, trig::NEG_TWO),
        empty);
    // sign = koverFloorM4 + k2 + ONE
    auto tmp = createLinalgBinOp<linalg::BinaryFn::add>(
        rewriter, loc, koverFloorM4, k2, empty);
    return createLinalgBinOp<linalg::BinaryFn::add>(
        rewriter, loc, tmp, f32Const(rewriter, loc, trig::ONE), empty);
  }
  // ---------- Phase‑4 ----------
  virtual Value CalculateResult(PatternRewriter &rewriter, Location loc,
                                Value sign, Value xFixed, Value &empty) const {
    // xPow = xFixed * xFixed
    auto xPow = createLinalgBinOp<linalg::BinaryFn::mul>(rewriter, loc, xFixed,
                                                         xFixed, empty);

    // (xPow * RES_MUL_SCA) + RES_ADD_UP
    auto res =
        mulAdd(rewriter, loc, xPow, f32Const(rewriter, loc, trig::RES_MUL_SCA),
               f32Const(rewriter, loc, trig::RES_ADD_UP), empty);

    // (res * xPow) + coeff
    res = mulAdd(rewriter, loc, res, xPow,
                 f32Const(rewriter, loc, trig::COEF_2), empty);
    res = mulAdd(rewriter, loc, res, xPow,
                 f32Const(rewriter, loc, trig::COEF_3), empty);
    res = mulAdd(rewriter, loc, res, xPow, f32Const(rewriter, loc, trig::ONE),
                 empty);

    // res * xFixed * sign
    res = createLinalgBinOp<linalg::BinaryFn::mul>(rewriter, loc, res, xFixed,
                                                   empty);
    return createLinalgBinOp<linalg::BinaryFn::mul>(rewriter, loc, res, sign,
                                                    empty);
  };

  // Phase‑1
  virtual SmallVector<Value> PrepRoundInput(PatternRewriter &rewriter,
                                            Location loc, Value input,
                                            Value &empty) const = 0;

  // Phase‑2
  virtual Value CalculateXFixed(PatternRewriter &rewriter, Location loc,
                                Value input, SmallVector<Value> roundInputs,
                                Value &empty) const = 0;

  virtual Value postProcess(PatternRewriter &rewriter, Location loc,
                            Value input) const {
    return input;
  }
};

/// normalize sin(x) implementation
/// Sin value precision is improved by using various precision levels of PI
/// Using an assortment of arithmetic operations, the Taylor expansion is
/// mimicked to calculate the value of Sin(X) where X is in radians.
struct NormalizeSinOp : public NormalizeTrigBase<NormalizeSinOp> {
  using NormalizeTrigBase<NormalizeSinOp>::NormalizeTrigBase;
  static constexpr hfusion::UnaryFn kUnaryFn = hfusion::UnaryFn::sin;
  // ---------- Phase‑1 ----------
  SmallVector<Value> PrepRoundInput(PatternRewriter &rewriter, Location loc,
                                    Value input, Value &empty) const override {
    // input * PI_FOR_X_DIV
    auto vmul = createLinalgBinOp<linalg::BinaryFn::mul>(
        rewriter, loc, input, f32Const(rewriter, loc, trig::PI_FOR_X_DIV),
        empty);
    // vmul * ONE_OVER_2048
    auto vmul0 = createLinalgBinOp<linalg::BinaryFn::mul>(
        rewriter, loc, vmul, f32Const(rewriter, loc, trig::ONE_OVER_2048),
        empty);
    // rounding
    auto roundPiDiv = hfusion::castTo(rewriter, vmul, rewriter.getF32Type(),
                                      hfusion::RoundMode::ROUND);
    auto roundPiDiv0 = hfusion::castTo(rewriter, vmul0, rewriter.getF32Type(),
                                       hfusion::RoundMode::ROUND);
    // roundPiDiv1 = roundPiDiv - roundPiDiv0 * CONST_2048
    roundPiDiv0 = createLinalgBinOp<linalg::BinaryFn::mul>(
        rewriter, loc, roundPiDiv0, f32Const(rewriter, loc, trig::CONST_2048),
        empty);
    auto roundPiDiv1 = createLinalgBinOp<linalg::BinaryFn::sub>(
        rewriter, loc, roundPiDiv, roundPiDiv0, empty);
    return {roundPiDiv, roundPiDiv0, roundPiDiv1};
  }

  // ---------- Phase‑2 ----------
  Value CalculateXFixed(PatternRewriter &rewriter, Location loc, Value input,
                        SmallVector<Value> roundInputs,
                        Value &empty) const override {
    const float coeffs[4] = {trig::PI_0, trig::PI_1, trig::PI_2, trig::PI_3};
    Value xFixed = input;
    for (int i = 0; i < 4; ++i) {
      // xFixed = xFixed - roundPiDivx * PI_X
      xFixed = subMul(rewriter, loc, f32Const(rewriter, loc, coeffs[i]),
                      roundInputs[1], xFixed, empty);
      xFixed = subMul(rewriter, loc, f32Const(rewriter, loc, coeffs[i]),
                      roundInputs[2], xFixed, empty);
    }
    return xFixed;
  }
};

/// normalize cos(x) implementation
/// Cos value precision is improved by using various precision levels of PI
/// Using an assortment of arithmetic operations, the Taylor expansion is
/// mimicked to calculate the value of Cos(X) where X is in radians.
struct NormalizeCosOp : public NormalizeTrigBase<NormalizeCosOp> {
  using NormalizeTrigBase<NormalizeCosOp>::NormalizeTrigBase;
  static constexpr hfusion::UnaryFn kUnaryFn = hfusion::UnaryFn::cos;

  // ---------- Phase‑1 ----------
  SmallVector<Value> PrepRoundInput(PatternRewriter &rewriter, Location loc,
                                    Value input, Value &empty) const override {
    // input * COS_PI_FOR_X_DIV
    auto vmul = createLinalgBinOp<linalg::BinaryFn::mul>(
        rewriter, loc, input, f32Const(rewriter, loc, trig::PI_FOR_X_DIV),
        empty);
    // vmul + COS_HALF
    auto vmul1 = createLinalgBinOp<linalg::BinaryFn::add>(
        rewriter, loc, vmul, f32Const(rewriter, loc, trig::HALF), empty);
    // vmul * COS_ONE_OVER_2048
    auto vmul0 = createLinalgBinOp<linalg::BinaryFn::mul>(
        rewriter, loc, vmul, f32Const(rewriter, loc, trig::ONE_OVER_2048),
        empty);
    // rounding
    auto roundPiDiv = hfusion::castTo(rewriter, vmul1, rewriter.getF32Type(),
                                      hfusion::RoundMode::ROUND);
    auto roundPiDiv0 = hfusion::castTo(rewriter, vmul0, rewriter.getF32Type(),
                                       hfusion::RoundMode::ROUND);
    // roundPiDiv1 = roundPiDiv - roundPiDiv0 * COS_CONST_2048
    roundPiDiv0 = createLinalgBinOp<linalg::BinaryFn::mul>(
        rewriter, loc, roundPiDiv0, f32Const(rewriter, loc, trig::CONST_2048),
        empty);
    auto roundPiDiv1 = createLinalgBinOp<linalg::BinaryFn::sub>(
        rewriter, loc, roundPiDiv, roundPiDiv0, empty);
    return {roundPiDiv, roundPiDiv0, roundPiDiv1};
  }
  // ---------- Phase‑2 ----------
  Value CalculateXFixed(PatternRewriter &rewriter, Location loc, Value input,
                        SmallVector<Value> roundInputs,
                        Value &empty) const override {
    Value xFixed = input;

    // xFixed = xFixed - roundPiDivx * PI_X
    xFixed = subMul(rewriter, loc, f32Const(rewriter, loc, trig::PI_0),
                    roundInputs[1], xFixed, empty);
    xFixed = subMul(rewriter, loc, f32Const(rewriter, loc, trig::PI_0),
                    roundInputs[2], xFixed, empty);

    xFixed = subMul(rewriter, loc, f32Const(rewriter, loc, trig::PI_1),
                    roundInputs[1], xFixed, empty);
    xFixed = createLinalgBinOp<linalg::BinaryFn::add>(
        rewriter, loc, xFixed, f32Const(rewriter, loc, trig::COS_PI_DOWN),
        empty);
    xFixed = subMul(rewriter, loc, f32Const(rewriter, loc, trig::PI_1),
                    roundInputs[2], xFixed, empty);

    xFixed = subMul(rewriter, loc, f32Const(rewriter, loc, trig::PI_2),
                    roundInputs[1], xFixed, empty);
    xFixed = subMul(rewriter, loc, f32Const(rewriter, loc, trig::PI_2),
                    roundInputs[2], xFixed, empty);

    xFixed = subMul(rewriter, loc, f32Const(rewriter, loc, trig::PI_3),
                    roundInputs[1], xFixed, empty);
    xFixed = subMul(rewriter, loc, f32Const(rewriter, loc, trig::PI_3),
                    roundInputs[2], xFixed, empty);

    xFixed = createLinalgBinOp<linalg::BinaryFn::add>(
        rewriter, loc, xFixed,
        f32Const(rewriter, loc, trig::COS_PI_RESDOWN_ADDS_NEG), empty);
    return xFixed;
  }
  Value postProcess(PatternRewriter &rewriter, Location loc,
                    Value input) const override {
    return ClipInput(rewriter, loc, input, trig::COS_CLIP_HIGH,
                     trig::COS_CLIP_LOW);
  }
};

/// normalize the specific cmp pattern to cast op
/// eg.
///  scalar = const 0
///  src0 = fill(scalar, dst) -> i8
///  y = hfusion.cmpi x, src0 {vne} ->  i1
/// is normalized to
///  y = hfusion.cast x -> i1

struct NormalizeCmpToCastOp : public OpRewritePattern<CompareOp> {
public:
  using OpRewritePattern<CompareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    llvm::SmallVector<Value> inputs = op.getInputs();
    bool isValidPattern = llvm::any_of(inputs, [&](Value &src) {
      if (auto fillOp = src.getDefiningOp<linalg::FillOp>()) {
        if (auto cstOp =
                fillOp.getInputs()[0].getDefiningOp<arith::ConstantIntOp>()) {
          return ((op.getCompareFn() == CompareFn::vne && cstOp.value() == 0));
        }
      }
      return false;
    });
    if (!isValidPattern) {
      return failure();
    }

    hfusion::RoundMode rounding = hfusion::RoundMode::RINT;
    auto roundingAttr = rewriter.getAttr<hfusion::RoundModeAttr>(rounding);
    auto modeAttr = rewriter.getNamedAttr(hfusion::RoundModeAttr::getMnemonic(),
                                          roundingAttr);
    auto castOp = rewriter.create<hfusion::CastOp>(
        op->getLoc(), TypeRange(op.getResults()), ValueRange{inputs[0]},
        ValueRange{op.getOutputs()[0]}, ArrayRef{modeAttr});
    rewriter.replaceOp(op, castOp);

    return success();
  }
};

namespace {
/// Normalize cmp Vne to Not(cmp Veq)
/// Because ne will work incorrectly, if src element value is NAN
/// eg.
///  y = hfusion.compare x, z {vne} ->  i1
/// is normalized to
/// tmp = hfusion.compare x, z {veq} ->  i1
///  y = hfusion.elemwise {unary <vnot>} tmp -> i1
struct NormalizeCmpVne : public OpRewritePattern<CompareOp> {
public:
  using OpRewritePattern<CompareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return failure();
    if (op.getCompareFn() != CompareFn::vne)
      return failure();
    Value lhs = op.getInputs()[0];
    Value rhs = op.getInputs()[1];

    // create eq op
    // replace OG op with not op
    auto veqOp = createCmpOp(rewriter, op->getLoc(), lhs, rhs, CompareFn::veq);
    auto vnotOp =
        hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp, hfusion::UnaryFn,
                               hfusion::UnaryFnAttr>(
            rewriter, op->getLoc(), hfusion::UnaryFn::vnot,
            ValueRange{veqOp->getResults()}, ValueRange(op.getOutputs()));
    rewriter.replaceOp(op, vnotOp);

    return success();
  }
};
} // namespace

/// normalize negf op to mul op
/// eg.
///  y = linalg.elemwise_unary {negf} (x)
///  is normalized to
///  y = linalg.elemwise_binary {mul} (x, -1)
struct NormalizeNegToMul : public OpRewritePattern<linalg::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != linalg::UnaryFn::negf) {
      return failure();
    }

    auto input = op.getDpsInputs()[0];
    auto elementType = getElementTypeOrSelf(input.getType());
    Value one = rewriter.create<arith::ConstantOp>(
        op->getLoc(), elementType, rewriter.getFloatAttr(elementType, -1.0));
    auto mulOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op->getLoc(), linalg::BinaryFn::mul,
            ValueRange{input, one}, ValueRange(op.getDpsInits()[0]));
    rewriter.replaceOp(op, mulOp);
    return success();
  }
};

/// normalize div op to rec op
/// eg.
///  y = linalg.div(1, x)
///  is normalized to
///  y = hfuson.elemwise_unary {rec}(x)
struct NormalizeDivVSToRec : public OpRewritePattern<linalg::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != linalg::BinaryFn::div) {
      return failure();
    }

    auto inputs = op.getDpsInputs();
    auto input0Type = inputs[0].getType();
    if (!input0Type.isIntOrFloat()) {
      return failure();
    }

    auto elemType = getElementTypeOrSelf(input0Type);
    if (elemType.isF32() || elemType.isBF16()) {
      // rec accuracy is not enough for f32, and bf16 will be cast to f32
      // finally
      return failure();
    }

    auto input0ConstOp =
        dyn_cast_or_null<arith::ConstantOp>(inputs[0].getDefiningOp());
    if (!input0ConstOp) {
      return failure();
    }
    auto constFloatAttr = dyn_cast<FloatAttr>(input0ConstOp.getValue());
    if (!constFloatAttr) {
      return failure();
    }
    llvm::APFloat oneFloat(constFloatAttr.getValue().getSemantics(), 1);
    if (!input0ConstOp || constFloatAttr.getValue() != oneFloat) {
      return failure();
    }

    auto recOP = hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp,
                                        hfusion::UnaryFn, hfusion::UnaryFnAttr>(
        rewriter, op->getLoc(), hfusion::UnaryFn::rec, ValueRange{inputs[1]},
        ValueRange(op.getDpsInits()[0]));
    rewriter.replaceOp(op, recOP);
    return success();
  }
};

/// normalize rsqrt op to rec(sqrt) op
/// eg.
///  y = hfusion elemwise unary {rsqrt} (x)
///  is normalized to
///  tmp = hfusion elemwise unary {sqrt} (x)
///  y = hfuson.elemwise_unary {rec}(tmp)
struct NormalizeRSqrtOp : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::UnaryFn::rsqrt) {
      return failure();
    }

    auto input = op.getDpsInputs()[0];
    auto emptyOp = utils::createEmptyOp(rewriter, op->getLoc(), input);

    auto sqrtOP =
        hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp, hfusion::UnaryFn,
                               hfusion::UnaryFnAttr>(
            rewriter, op->getLoc(), hfusion::UnaryFn::sqrt, ValueRange{input},
            ValueRange(emptyOp));

    auto recInput = sqrtOP->getResults();
    auto recOP = hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp,
                                        hfusion::UnaryFn, hfusion::UnaryFnAttr>(
        rewriter, op->getLoc(), hfusion::UnaryFn::rec, ValueRange{recInput},
        ValueRange(op.getDpsInits()[0]));
    rewriter.replaceOp(op, recOP);
    return success();
  }
};

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

///  Normalize mod op to a canonical form that matches Triton dialect:
///    z = x % y
///  is normalized to
///    q = trunc(x / y)                // truncation toward zero
///    z = x - q * y
///  This matches:
///    41 % 20    = 1
///    41 % (-20) = 1
///    (-72) % 8  = 0
///  int/fp16/bf16 types are converted to fp32 for the division to get
///  better numerical behavior.
struct NormalizeModOp : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;

  std::optional<Value> handleInfinityModulus(PatternRewriter &rewriter,
                                             Location loc, Value xF32,
                                             Value yF32, Value result) const {
    auto yType = mlir::dyn_cast<ShapedType>(yF32.getType());
    if (!yType) {
      return std::nullopt;
    }
    Type boolType =
        RankedTensorType::get(yType.getShape(), rewriter.getIntegerType(1));
    auto isInf = rewriter.create<hfusion::IsInfOp>(loc, boolType, yF32);
    auto correctResult = utils::createEmptyOp(rewriter, loc, result);
    return rewriter
        .create<hfusion::SelectOp>(loc, TypeRange{correctResult.getType()},
                                   ValueRange{isInf, xF32, result},
                                   ValueRange{correctResult})
        .getResults()[0];
  }

  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    // Support both signed and unsigned modulo.
    auto fun = op.getFun();
    if (fun != hfusion::BinaryFn::mod && fun != hfusion::BinaryFn::modui) {
      return failure();
    }

    auto resTensor = op.getResultTensors()[0];
    auto resTy = dyn_cast<TensorType>(resTensor.getType());
    if (!resTy) {
      return failure();
    }

    auto elemType = getElementTypeOrSelf(resTy);
    // Only support int/index/float, but not i64.
    if (!elemType.isIntOrIndexOrFloat() || elemType.isInteger(64)) {
      return failure();
    }

    // Original inputs in their original dtype.
    Value xOrig = op.getInputs()[0];
    Value yOrig = op.getInputs()[1];

    // step 1: x_f32 = cast(x) => f32
    //         y_f32 = cast(y) => f32
    Value xF32 = xOrig;
    Value yF32 = yOrig;
    hfusion::TypeFn cast_integer_type = (fun == hfusion::BinaryFn::mod)
                                  ? hfusion::TypeFn::cast_signed
                                  : hfusion::TypeFn::cast_unsigned;
    if (!elemType.isF32()) {
      if (elemType.isIntOrIndex()) {
        // For integer → f32 casts, force TRUNC rounding mode to match
        // C-like modulo semantics and the existing tests.
        xF32 = hfusion::castTo(
            rewriter,
            xOrig,
            rewriter.getF32Type(),
            hfusion::RoundMode::TRUNC,
            cast_integer_type);
        yF32 = hfusion::castTo(
            rewriter,
            yOrig,
            rewriter.getF32Type(),
            hfusion::RoundMode::TRUNC,
            cast_integer_type);
      } else {
        // For non-integer element types (e.g. f16/bf16) just cast to f32.
        xF32 = hfusion::castTo(rewriter, xOrig, rewriter.getF32Type());
        yF32 = hfusion::castTo(rewriter, yOrig, rewriter.getF32Type());
      }
    }

    // step 2: q_f32 = truncate_div(x_f32, y_f32)
    auto emptyDivTensor = utils::createEmptyOpWithTargetElemType(
        rewriter, op->getLoc(), resTensor, rewriter.getF32Type());
    Operation *divOp = nullptr;
    std::optional<Operation **> divF32 = &divOp;
    auto truncDivF32 = hfusion::divWithRoundMode(
        rewriter, op.getLoc(), rewriter.getF32Type(),
        xF32, yF32, emptyDivTensor,
        hfusion::RoundMode::TRUNC, divF32);
    assert((divF32 != std::nullopt) && (*divF32.value()) != nullptr &&
           "div operation cannot be null!");

    // step 3: q = q_f32 castTo elemType
    Value q = hfusion::castTo(rewriter, truncDivF32, elemType);

    // step 4: mul = y * q   (computed in original dtype)
    auto emptyMulTensor =
        utils::createEmptyOpWithTargetElemType(
            rewriter, op->getLoc(), resTensor, elemType);
    auto mul =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op.getLoc(), linalg::BinaryFn::mul,
            ValueRange{yOrig, q}, emptyMulTensor)
            ->getResults()[0];

    // step 5: res = x - mul (still in original dtype)
    auto emptyResTensor =
        utils::createEmptyOpWithTargetElemType(
            rewriter, op->getLoc(), resTensor, elemType);
    auto res =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, op.getLoc(), linalg::BinaryFn::sub,
            ValueRange{xOrig, mul}, emptyResTensor)
            ->getResults()[0];

    // For f32 we keep the "y == inf ? x : res" correction.
    if (elemType.isF32()) {
      auto correctResF32 =
          handleInfinityModulus(rewriter, op.getLoc(), xF32, yF32, res)
              .value_or(res);
      rewriter.replaceOp(op, correctResF32);
      return success();
    }

    // For other types, res is already in elemType, without Torch-specific logic
    // (no rem + y, no z = -1 if y == 0).
    rewriter.replaceOp(op, res);
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

    auto fun = op.getFun();
    if (fun != hfusion::BinaryFn::mod && fun != hfusion::BinaryFn::modui) {
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
    if ((inType.isF16() || inType.isBF16()) && inType == outType) {
      // 910B only support fp32 ceil and floor, so change to fp16->fp32,
      // fp32 ceil/floor and fp32->fp16
      // TODO: add platform info to isHWSupportCeilFLoor(Type)

      // Step1: cast to fp32 to do ceil or floor
      auto intermediate = hfusion::castTo(builder, src, rewriter.getF32Type(),
                                          hfusion::RoundMode::RINT);
      // Step2: enable fp32 cast ability with ceil or floor mode
      // Otherwise, cast fp32 to B16 type in ceil or floor mode just changes
      // precision loss part.
      src = hfusion::castTo(builder, intermediate, rewriter.getF32Type(),
                            roundMode);
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

// get polyexpr in the format [(input + p1) * squareSrc + p2] * squareSrc + ...,
// enableLastMulTerm = false means [(input + p1) * squareSrc + p2] + ... remove
// the last multiplication by squareSrc.
Value genPolyExpr(PatternRewriter &rewriter, Location loc,
                  const Value squareSrc, Value input,
                  const llvm::SmallVector<double> &numerCoeff,
                  bool enableLastMulTerm = true) {
  auto inType = getElementTypeOrSelf(squareSrc.getType());

  Value resInit = utils::createEmptyOp(rewriter, loc, input);
  Value res = input;
  auto numberCoeffSize = numerCoeff.size();
  for (size_t i = 0; i < numberCoeffSize; i++) {
    arith::ConstantOp constOp = rewriter.create<arith::ConstantOp>(
        loc, inType, rewriter.getFloatAttr(inType, numerCoeff[i]));
    auto *addOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::add,
            ValueRange{res, constOp->getResults()[0]}, ValueRange(resInit));
    if (enableLastMulTerm || i != (numberCoeffSize - 1)) {
      auto *mulOp =
          hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                  linalg::BinaryFnAttr>(
              rewriter, loc, linalg::BinaryFn::mul,
              ValueRange{addOp->getResults()[0], squareSrc},
              ValueRange(resInit));
      res = mulOp->getResults()[0];
    } else {
      res = addOp->getResults()[0];
    }
  }
  return res;
}

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

/// normalize integer divsi and divui by float div
/// supports i8/i16/i32/i64 type
/// c = a / b
/// is normalized to
/// fa = castTo<f32>(a)
/// fb = castTo<f32>(b)
/// fc = fa / fb
/// c = castTo<integer>(fc, mode = TRUNC)
struct NormalizeDivSIandDivUIOp
    : public OpRewritePattern<linalg::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if ((op.getFun() != linalg::BinaryFn::div) &&
        (op.getFun() != linalg::BinaryFn::div_unsigned)) {
      return failure();
    }

    auto loc = op->getLoc();
    // linalg::ElemwiseBinaryOp's Outputs and Results must be
    // variadic of ranked tensor of any type values.
    // If the Outputs operand is a scalar, mlir crashes.
    // If the Results operand is a scalar, the verifier reports error.
    auto resTensor = op.getResultTensors()[0];
    auto resTy = dyn_cast<TensorType>(resTensor.getType());
    auto elemTySrc = getElementTypeOrSelf(resTy);
    if (!elemTySrc.isInteger() || elemTySrc.isInteger(64)) {
      return failure();
    }

    // step1. res = divWithRoundMode(x, y, TRUNC)
    rewriter.setInsertionPoint(op);
    auto inputs = op.getDpsInputs();
    auto res = hfusion::divWithRoundMode(rewriter, loc, elemTySrc, inputs[0],
                                         inputs[1], resTensor,
                                         hfusion::RoundMode::TRUNC);
    rewriter.replaceOp(op, res);
    return success();
  }
};

/// Returns whether the input value `v` is rec-like: Rec op or div op
/// with numerator of constant one. Set the denominator in place if true
static bool isRecLike(mlir::Value v, mlir::Value &denominator) {
  Operation *op = v.getDefiningOp();
  if (auto recOp = dyn_cast_or_null<hfusion::ElemwiseUnaryOp>(op)) {
    if (recOp.getFun() != hfusion::UnaryFn::rec) {
      return false;
    }
    denominator = recOp.getDpsInputs()[0];
    return true;
  }
  auto binOp = dyn_cast_or_null<linalg::ElemwiseBinaryOp>(op);
  if (!binOp) {
    return false;
  }
  if (binOp.getFun() != linalg::BinaryFn::div) {
    return false;
  }
  auto inputs = binOp.getDpsInputs();
  mlir::Value divLhs = inputs[0];
  mlir::Value divRhs = inputs[1];
  auto lhsConstOp = dyn_cast_or_null<arith::ConstantOp>(divLhs.getDefiningOp());
  if (!lhsConstOp) {
    return false;
  }

  denominator = divRhs;
  if (auto constFloatAttr = dyn_cast<FloatAttr>(lhsConstOp.getValue())) {
    llvm::APFloat floatOne(constFloatAttr.getValue().getSemantics(), 1);
    return constFloatAttr.getValue() == floatOne;
  }
  if (auto constIntAttr = dyn_cast<IntegerAttr>(lhsConstOp.getValue())) {
    return constIntAttr.getInt() == 1;
  }
  return false;
}

// replace `mulOp` with `newDivLhs/newDivRhs`
static void normalizeMulRecLikeByDiv(linalg::ElemwiseBinaryOp mulOp,
                                     Value newDivLhs, Value newDivRhs,
                                     PatternRewriter &rewriter) {
  assert(mulOp.getFun() == linalg::BinaryFn::mul &&
         "only support div-by-one used by mul bin op");
  auto initTensor = mulOp->getOperand(2);
  auto newDivResult =
      utils::createEmptyOp(rewriter, mulOp.getLoc(), initTensor);
  auto newDivOp =
      hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                              linalg::BinaryFnAttr>(
          rewriter, mulOp.getLoc(), linalg::BinaryFn::div,
          ValueRange{newDivLhs, newDivRhs}, ValueRange(newDivResult));
  rewriter.replaceOp(mulOp, newDivOp);
}

/// normalize mul rec(div-by-one)
/// (1/b) * a -> a/b
/// a * (1/b) -> a/b
struct NormalizeMulRec : public OpRewritePattern<linalg::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    if (op.getFun() != linalg::BinaryFn::mul) {
      return failure();
    }
    auto inputs = op.getDpsInputs();
    mlir::Value mulLhs = inputs[0];
    mlir::Value mulRhs = inputs[1];
    mlir::Value denominator;
    if (isRecLike(mulLhs, denominator)) {
      /// (1/b) * a -> a/b
      normalizeMulRecLikeByDiv(op, mulRhs, denominator, rewriter);
      return success();
    }
    if (isRecLike(mulRhs, denominator)) {
      /// a * (1/b) -> a/b
      normalizeMulRecLikeByDiv(op, mulLhs, denominator, rewriter);
      return success();
    }
    return failure();
  }
};

static Value castInToF32ToOut(hfusion::CastOp &op, PatternRewriter &rewriter) {
  auto dstTy = getElementTypeOrSelf(op.getDpsInitOperand(0)->get());
  auto castSrcToF32 = castTo(rewriter, op.getDpsInputOperand(0)->get(),
                             rewriter.getF32Type(), op.getCast());
  auto castF32ToOut =
      hfusion::castTo(rewriter, castSrcToF32, dstTy, TypeFn::cast_signed);
  return castF32ToOut;
}

// i1/i8/i16 -> f16 -> targetType
static Value castSrcToFp16ToTargetType(hfusion::CastOp &op, Type targetType,
                                       PatternRewriter &rewriter) {
  Type f16Type = rewriter.getF16Type();
  Value dpsInput = op.getDpsInputOperand(0)->get();
  auto castSrcToF16 = castTo(rewriter, dpsInput, f16Type, op.getCast());
  return castTo(rewriter, castSrcToF16, targetType, TypeFn::cast_signed);
}

// i64/i8 -> i1
static Value castSrcTypeToI1ByVCmp(hfusion::CastOp &op, Type srcType,
                                   PatternRewriter &rewriter) {
  // 1. cast src to f16/f32
  Value inValue = op.getInputs()[0];
  Value castF16OrF32Value;
  if (srcType.isInteger(8)) {
    castF16OrF32Value =
        hfusion::castTo(rewriter, inValue, rewriter.getF16Type());
  } else if (srcType.isInteger(16)) {
    castF16OrF32Value = hfusion::castTo(
        rewriter, inValue, rewriter.getF16Type(), hfusion::RoundMode::RINT);
  } else if (srcType.isInteger(32)) {
    castF16OrF32Value = hfusion::castTo(
        rewriter, inValue, rewriter.getF32Type(), hfusion::RoundMode::RINT);
  } else if (srcType.isInteger(64)) {
    castF16OrF32Value = hfusion::castTo(
        rewriter, inValue, rewriter.getF32Type(), hfusion::RoundMode::RINT);
  } else if (srcType.isBF16()) {
    castF16OrF32Value = hfusion::castTo(
        rewriter, inValue, rewriter.getF32Type(), hfusion::RoundMode::RINT);
  } else if (srcType.isF32() || srcType.isF16()) {
    castF16OrF32Value = inValue;
  } else {
    llvm_unreachable("unsupport srcType to i1.");
  }

  // 2. cast: f16/f32 -> i1, dst = vcmpvs_ne(src, 0)
  auto elementType = getElementTypeOrSelf(castF16OrF32Value);
  arith::ConstantOp constZero = rewriter.create<arith::ConstantOp>(
      op->getLoc(), elementType, rewriter.getFloatAttr(elementType, 0.0));

  Value castI1Value = createCmpOp(rewriter, op.getLoc(), castF16OrF32Value,
                                  constZero, CompareFn::vne)
                          ->getResult(0);
  return castI1Value;
}

// i8 -> f16 -> f32 -> i64
static Value castI8ToI64(hfusion::CastOp &op, PatternRewriter &rewriter) {
  // f32->i64
  Value i8ToF32Result =
      castSrcToFp16ToTargetType(op, rewriter.getF32Type(), rewriter);
  Type i64Type = rewriter.getIntegerType(64);
  auto castF32ToDst =
      castTo(rewriter, i8ToF32Result, i64Type, TypeFn::cast_signed);
  return castF32ToDst;
}

hfusion::CastMode getCastMode(hfusion::CastOp op) {
  auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
  auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());

  const bool isF32ToI16 = inType.isF32() && outType.isInteger(16);
  const bool isF32ToI8 = inType.isF32() && outType.isInteger(8);
  const bool isF16ToI8 = inType.isF16() && outType.isInteger(8);
  const bool isI64ToI32 = inType.isInteger(64) && outType.isInteger(32);
  const bool isI64ToI16 = inType.isInteger(64) && outType.isInteger(16);
  const bool isI64ToI8 = inType.isInteger(64) && outType.isInteger(8);
  const bool isI32ToI16 = inType.isInteger(32) && outType.isInteger(16);
  const bool isI32ToI8 = inType.isInteger(32) && outType.isInteger(8);
  const bool isI16ToI8 = inType.isInteger(16) && outType.isInteger(8);

  if (isF32ToI16)
    return hfusion::CastMode::F32TOI16;
  if (isF32ToI8)
    return hfusion::CastMode::F32TOI8;
  if (isF16ToI8)
    return hfusion::CastMode::F16TOI8;
  if (isI64ToI32)
    return hfusion::CastMode::I64TOI32;
  if (isI64ToI16)
    return hfusion::CastMode::I64TOI16;
  if (isI64ToI8)
    return hfusion::CastMode::I64TOI8;
  if (isI32ToI16)
    return hfusion::CastMode::I32TOI16;
  if (isI32ToI8)
    return hfusion::CastMode::I32TOI8;
  if (isI16ToI8)
    return hfusion::CastMode::I16TOI8;

  llvm_unreachable("unsupported cast mode");
}

std::optional<StringRef> getAnnotateOverflowMode(hfusion::CastOp op) {
  std::optional<Operation *> overflowMode =
      utils::getAnnotateOpWithAttr(op.getResult(0), "overflow_mode");
  if (!overflowMode.has_value()) {
    return std::nullopt;
  }
  StringAttr overflowAttrVal =
      overflowMode.value()->getAttrOfType<StringAttr>("overflow_mode");
  return overflowAttrVal.getValue();
}

/// normalize cast from large bit width to small bit width, and dst's data type
/// is integer, when overflow mode is saturate.
/// if data is overflow, it will be saturated to the extreme in this scenario.
/// e.g. Input (float32): tensor([ 128.7000,  127.5000,  100.3000, -129.2000,
/// -128.4000]), Output(int8): tensor([ 127,  127,  100, -128, -128],
/// dtype=torch.int8)
LogicalResult handleSaturateOverFlowMode(hfusion::CastOp op,
                                         PatternRewriter &rewriter) {
  hfusion::CastMode castMode = getCastMode(op);
  Value castValue = op.getInputs()[0];
  auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());
  hfusion::TypeFn castIntegerType = op.getCast();

  switch (castMode) {
  case hfusion::CastMode::F32TOI16:
    castValue = hfusion::castTo(rewriter, castValue, outType,
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::F32TOI8:
    // step 1: cast f32 to f16 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false);
    // step 2: cast f16 to i8 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, outType,
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::F16TOI8:
    castValue = hfusion::castTo(rewriter, castValue, outType,
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::I64TOI32:
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::RINT,
                        std::nullopt, /*enableOverflow=*/false);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::I64TOI16:
    // step 1: cast i32 to f32 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF32Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false);
    // step 2: cast f32 to i16 in TRUNC mode
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow=*/false);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::I64TOI8:
    // step 1: cast i32 to f32 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF32Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false);
    // step 2: cast f32 to f16 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false);
    // step 3: cast f16 to i8 in TRUNC mode
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow=*/false);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::I32TOI16:
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::RINT,
                        std::nullopt, /*enableOverflow=*/false);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::I32TOI8:
    // step 1: cast i32 to f32 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF32Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false);
    // step 2: cast f32 to f16 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false);
    // step 3: cast f16 to i8 in TRUNC mode
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow=*/false);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::I16TOI8:
    // step 1: cast i16 to f16 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false);
    // step 2: cast f16 to i8 in TRUNC mode
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow=*/false);
    rewriter.replaceOp(op, castValue);
    return success();
  }
}

LogicalResult handleTruncOverFlowMode(hfusion::CastOp op,
                                      PatternRewriter &rewriter) {
  auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
  auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());
  auto castIntegerType = op.getCast();

  const bool isF32ToI16 = inType.isF32() && outType.isInteger(16);
  const bool isF32ToI8 = inType.isF32() && outType.isInteger(8);
  const bool isF16ToI8 = inType.isF16() && outType.isInteger(8);
  const bool isI64ToI16 = inType.isInteger(64) && outType.isInteger(16);
  const bool isI64ToI8 = inType.isInteger(64) && outType.isInteger(8);
  const bool isI32ToI8 = inType.isInteger(32) && outType.isInteger(8);
  const bool isI16ToI8 = inType.isInteger(16) && outType.isInteger(8);
  Value castValue = op.getInputs()[0];
  // TODO: The round_mode will be flushed and will be fixed during
  // reconstruction.
  if (isF32ToI16 && op.getEnableOverflow()) {
    // step1: cast f32 to i32 in TRUNC mode
    Value castI32Value = hfusion::castTo(
        rewriter, castValue, rewriter.getI32Type(), hfusion::RoundMode::TRUNC,
        std::nullopt, true, castIntegerType);
    // step2: cast i32 to i16
    castValue = hfusion::castTo(rewriter, castI32Value, rewriter.getI16Type(),
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isF32ToI8) {
    // step 1: cast f32 to i32 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getI32Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt, true,
                                castIntegerType);
    // step 2: cast i32 to i8 in TRUNCWITHOVERFLOW mode
    castValue = hfusion::castTo(rewriter, castValue, outType,
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isF16ToI8 && op.getEnableOverflow()) {
    Value overflowResult = hfusion::OverflowProcess(
        rewriter, castValue, getElementTypeOrSelf(outType));
    castValue = hfusion::castTo(rewriter, overflowResult, outType,
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow=*/false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI64ToI16 || isI64ToI8) {
    // step 1: cast i64 to i32 in TRUNCWITHOVERFLOW mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getI32Type(),
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    // step 2: cast i32 to i16/i8 in TRUNCWITHOVERFLOW mode
    castValue = hfusion::castTo(rewriter, castValue, outType,
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if ((isI32ToI8 || isI16ToI8) &&
             op.getRoundMode() != hfusion::RoundMode::TRUNCWITHOVERFLOW) {
    castValue = hfusion::castTo(rewriter, castValue, outType,
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    rewriter.replaceOp(op, castValue);
    return success();
  }
  return failure();
}

static bool isI1ElemType(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  return elemType.isInteger(1);
}

static bool isI8ElemType(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  return elemType.isInteger(8);
}

static bool isI16ElemType(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  return elemType.isInteger(16);
}

static bool isI64ElemType(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  return elemType.isInteger(64);
}

static bool isF16ElemType(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  return elemType.isF16();
}

template <typename srcType>
static bool isElemType(Type valueType) {
  if constexpr (std::is_same_v<bool, srcType>) {
    return isI1ElemType(valueType);
  }
  if constexpr (std::is_same_v<int8_t, srcType>) {
    return isI8ElemType(valueType);
  }
  if constexpr (std::is_same_v<int16_t, srcType>) {
    return isI16ElemType(valueType);
  }
  if constexpr (std::is_same_v<float, srcType>) {
    return isF16ElemType(valueType);
  }
  return false;
}

static bool hasI1ElemType(const SmallVector<Value> &values) {
  return llvm::any_of(values,
                      [&](Value v) { return isI1ElemType(v.getType()); });
}

static bool hasI8ElemType(const SmallVector<Value> &values) {
  return llvm::any_of(values,
                      [&](Value v) { return isI8ElemType(v.getType()); });
}

[[maybe_unused]] static bool hasI16ElemType(const SmallVector<Value> &values) {
  return llvm::all_of(values,
                      [&](Value v) { return isI16ElemType(v.getType()); });
}

static bool hasF16ElemType(const SmallVector<Value> &values) {
  return llvm::any_of(values,
                      [&](Value v) { return isF16ElemType(v.getType()); });
}

template <typename srcType>
static bool hasElemType(const SmallVector<Value> &values) {
  if constexpr (std::is_same_v<bool, srcType>) {
    return hasI1ElemType(values);
  }
  if constexpr (std::is_same_v<int8_t, srcType>) {
    return hasI8ElemType(values);
  }
  if constexpr (std::is_same_v<int16_t, srcType>) {
    return hasI16ElemType(values);
  }
  if constexpr (std::is_same_v<float, srcType>) {
    return hasF16ElemType(values);
  }
  return false;
}

[[maybe_unused]] static bool allI1ElemType(const SmallVector<Value> &values) {
  return llvm::all_of(values,
                      [&](Value v) { return isI1ElemType(v.getType()); });
}

static bool allI8ElemType(const SmallVector<Value> &values) {
  return llvm::all_of(values,
                      [&](Value v) { return isI8ElemType(v.getType()); });
}

static bool allI16ElemType(const SmallVector<Value> &values) {
  return llvm::all_of(values,
                      [&](Value v) { return isI16ElemType(v.getType()); });
}

/// linalg.(fill/brc) + hfusion.cast
/// is normalized to
/// (arith/hfusion).cast + linalg.(fill/brc)
/// in order to cast quickly
struct NormalizeBrcCast : public OpRewritePattern<hfusion::CastOp> {
  std::optional<Value> getCastedValue(PatternRewriter &rewriter, Location loc,
                                      Value cst, Type srcType, Type dstType,
                                      hfusion::RoundMode roundMode) const {
    auto srcElmTy = getElementTypeOrSelf(srcType);
    auto dstElmTy = getElementTypeOrSelf(dstType);

    hfusion::RoundMode defaultRounding =
        utils::selectRoundMode<hfusion::RoundMode>(srcElmTy, dstElmTy);
    bool scalarSrc = !isa<ShapedType>(cst.getType());
    // only scalar cast has default round mode (e.g arith.sitofp -> <trunc>)
    // do not use scalar castTo when round modes mismatch
    if (!(defaultRounding == roundMode) && scalarSrc)
      return std::nullopt;

    return hfusion::castTo(rewriter, cst, dstElmTy, roundMode);
  }

public:
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    if (!castOp.hasPureTensorSemantics()) {
      return failure();
    }

    Value src = castOp.getDpsInputs()[0];
    if (isa<BlockArgument>(src))
      return failure();

    Operation *defOp = src.getDefiningOp();
    if (!isa<linalg::FillOp>(defOp) && !isa<linalg::BroadcastOp>(defOp))
      return failure();

    auto srcTy = src.getType();
    auto dstTy = dyn_cast<TensorType>(castOp.getOutputs()[0].getType());
    // Disable conversion from brc f16 + cast i8/bool as
    // combined with NormalizeToTargetType pass causes infinite loop
    if (isa<linalg::BroadcastOp>(defOp) && isF16ElemType(srcTy) &&
        (isI1ElemType(dstTy) || isI8ElemType(dstTy))) {
      return failure();
    }

    auto roundMode = castOp.getRoundMode();
    Location loc = castOp.getLoc();

    Value cst = isa<linalg::FillOp>(defOp)
                    ? dyn_cast<linalg::FillOp>(defOp).getInputs()[0]
                    : dyn_cast<linalg::BroadcastOp>(defOp).getInput();

    auto castedVal =
        getCastedValue(rewriter, loc, cst, srcTy, dstTy, roundMode);
    if (!castedVal.has_value())
      return rewriter.notifyMatchFailure(
          castOp, "either round mode or datatype is not supported!");
    Value emptyTensor =
        utils::createEmptyOp(rewriter, loc, castOp.getOutputs()[0]);
    auto *newFillOrBrcOp =
        isa<linalg::FillOp>(defOp)
            ? rewriter.create<linalg::FillOp>(loc, *castedVal, emptyTensor)
            : rewriter.create<linalg::BroadcastOp>(
                  loc, *castedVal, emptyTensor,
                  dyn_cast<linalg::BroadcastOp>(defOp).getDimensionsAttr());

    rewriter.replaceAllUsesWith(castOp.getResults(),
                                newFillOrBrcOp->getResults());
    rewriter.eraseOp(castOp);

    return success();
  }
};

/// convert scalar to point tensor + hfusion.cast + linalg.broadcast
/// on unsupported round modes to optimize linalg.fill + hfusion.cast
struct NormalizefillCastToTensorBrc : public OpRewritePattern<hfusion::CastOp> {
  std::optional<Value>
  getPointTensorCastedValue(PatternRewriter &rewriter, Location loc, Value cst,
                            Type srcType, Type dstType,
                            hfusion::RoundMode roundMode) const {
    auto srcElmTy = getElementTypeOrSelf(srcType);
    auto dstElmTy = getElementTypeOrSelf(dstType);

    hfusion::RoundMode defaultRounding =
        utils::selectRoundMode<hfusion::RoundMode>(srcElmTy, dstElmTy);
    bool scalarSrc = !isa<ShapedType>(cst.getType());
    if ((defaultRounding == roundMode) || !scalarSrc)
      return std::nullopt;

    auto pointSrcTensorType = RankedTensorType::get({}, cst.getType());
    Value pointSrcTensor =
        utils::createStaticShapeEmptyOp(rewriter, loc, pointSrcTensorType);
    auto newFillOp = rewriter.create<linalg::FillOp>(loc, cst, pointSrcTensor);

    return hfusion::castTo(rewriter, newFillOp->getResult(0), dstElmTy,
                           roundMode);
  }

public:
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    if (!castOp.hasPureTensorSemantics()) {
      return failure();
    }

    Value src = castOp.getDpsInputs()[0];
    if (isa<BlockArgument>(src))
      return failure();

    Operation *defOp = src.getDefiningOp();
    if (!isa<linalg::FillOp>(defOp))
      return failure();
    auto fillOp = dyn_cast<linalg::FillOp>(defOp);
    auto srcTy = src.getType();
    auto dstTy = dyn_cast<TensorType>(castOp.getOutputs()[0].getType());
    if (dstTy.getRank() == 0)
      return failure();

    auto roundMode = castOp.getRoundMode();
    Location loc = castOp.getLoc();

    Value cst = fillOp.getInputs()[0];

    auto castedVal =
        getPointTensorCastedValue(rewriter, loc, cst, srcTy, dstTy, roundMode);
    if (!castedVal.has_value())
      return rewriter.notifyMatchFailure(
          castOp, "either round mode or datatype is not supported!");
    Value emptyTensor =
        utils::createEmptyOp(rewriter, loc, castOp.getOutputs()[0]);
    SmallVector<int64_t> dim;
    for (int64_t i = 0; i < dstTy.getRank(); ++i)
      dim.push_back(i);

    auto brcOp =
        rewriter.create<linalg::BroadcastOp>(loc, *castedVal, emptyTensor, dim);

    rewriter.replaceAllUsesWith(castOp.getResults(), brcOp->getResults());
    rewriter.eraseOp(castOp);

    return success();
  }
};

struct NormalizetruncfExtf : public OpRewritePattern<arith::ExtFOp> {
public:
  using OpRewritePattern<arith::ExtFOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtFOp extOp,
                                PatternRewriter &rewriter) const override {
    auto src = extOp.getIn();
    if (isa<BlockArgument>(src))
      return failure();
    auto defOp = src.getDefiningOp<arith::TruncFOp>();
    if (!defOp)
      return failure();
    if (defOp.getIn().getType() != extOp.getOut().getType())
      return failure();
    rewriter.replaceAllUsesWith(extOp.getOut(), defOp.getIn());
    return success();
  }
};

struct NormalizeAnyToF32UnaryRecOp
    : public OpRewritePattern<hfusion::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseUnaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    // currently, only applied to rec unary function
    if (op.getFun() != hfusion::UnaryFn::rec)
      return failure();

    Value inValue = op.getInputs()[0];
    Value outValue = op.getOutputs()[0];

    Type inType = getElementTypeOrSelf(inValue.getType());
    Type outType = getElementTypeOrSelf(outValue.getType());
    // currently, only need handle case where the input type is equal to output
    // type
    if (inType != outType)
      return failure();

    if (inType.isF32())
      return failure();

    Location loc = op->getLoc();

    // TODO: cast to more efficient data type
    auto castedInValue =
        hfusion::castTo(rewriter, inValue, rewriter.getF32Type());

    // create new elemwise_unary op
    auto resEmptyOp = utils::createEmptyOp(rewriter, loc, castedInValue);
    Operation *newOp =
        hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp, hfusion::UnaryFn,
                               hfusion::UnaryFnAttr>(
            rewriter, loc, hfusion::UnaryFn::rec, castedInValue, resEmptyOp);

    // TODO: cast to more efficient data type
    auto castedOutValue =
        hfusion::castTo(rewriter, newOp->getResult(0), outType);
    rewriter.replaceOp(op, castedOutValue);
    return success();
  }
};

struct NormalizeCastLoweringOp : public OpRewritePattern<hfusion::CastOp> {
public:
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::CastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());
    int64_t srcBitWidth = inType.getIntOrFloatBitWidth();
    int64_t dstBitWidth = outType.getIntOrFloatBitWidth();
    if (srcBitWidth > dstBitWidth && outType.isInteger() &&
        !outType.isInteger(1)) {
      auto overflowMode = getAnnotateOverflowMode(op);
      if (overflowMode.has_value() && overflowMode->ends_with("saturate")) {
        // annotation.mark %s {overflow_mode = "saturate"}
        auto overflowModeAttr =
            utils::getAnnotateOpWithAttr(op->getResult(0), "overflow_mode");
        if (!overflowModeAttr.has_value())
          return failure();
        annotation::MarkOp markOp =
            dyn_cast<annotation::MarkOp>(overflowModeAttr.value());
        rewriter.eraseOp(markOp);
        return handleSaturateOverFlowMode(op, rewriter);
      }
      return handleTruncOverFlowMode(op, rewriter);
    }

    const bool isI64ToF16 = inType.isInteger(64) && outType.isF16();
    const bool isIntegerToBF16 =
        (inType.isInteger(64) || inType.isInteger(32) || inType.isInteger(16) ||
         inType.isInteger(8)) &&
        outType.isBF16();
    if (isI64ToF16 || isIntegerToBF16) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << inType << " to "
                 << outType << ", and rewrite to cast (from " << inType
                 << " to f16) "
                 << "\n ");
      Value castResult = castInToF32ToOut(op, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isI8ToI64 = inType.isInteger(8) && outType.isInteger(64);
    if (isI8ToI64) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << inType << " to "
                 << outType << ", and rewrite to cast (from " << inType
                 << " to f16 to f32 to " << outType << ")\n");
      Value castResult = castI8ToI64(op, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isI8ToF32 = inType.isInteger(8) && outType.isF32();
    const bool isI8ToI32 = inType.isInteger(8) && outType.isInteger(32);
    const bool isI8ToI16 = inType.isInteger(8) && outType.isInteger(16);
    if (isI8ToF32 || isI8ToI32 || isI8ToI16) {
      Type targetType = getElementTypeOrSelf(outType);
      Value castResult = castSrcToFp16ToTargetType(op, targetType, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isI1ToI64 = inType.isInteger(1) && outType.isInteger(64);
    if (isI1ToI64) {
      Value inValue = op.getInputs()[0];
      Value castF32Value = hfusion::castTo(
          rewriter, inValue, rewriter.getF32Type(), hfusion::RoundMode::RINT);

      Value castI64Value =
          hfusion::castTo(rewriter, castF32Value, rewriter.getI64Type());
      rewriter.replaceOp(op, castI64Value);
      return success();
    }

    const bool isI32ToF16 = inType.isInteger(32) && outType.isF16();
    if (isI32ToF16) {
      Value inValue = op.getInputs()[0];
      Value castF32Value =
          hfusion::castTo(rewriter, inValue, rewriter.getF32Type());

      Value castF16Value =
          hfusion::castTo(rewriter, castF32Value, rewriter.getF16Type());
      rewriter.replaceOp(op, castF16Value);
      return success();
    }

    const bool isI64ToI1 = inType.isInteger(64) && outType.isInteger(1);
    const bool isI32ToI1 = inType.isInteger(32) && outType.isInteger(1);
    const bool isI16ToI1 = inType.isInteger(16) && outType.isInteger(1);
    const bool isI8ToI1 = inType.isInteger(8) && outType.isInteger(1);
    const bool isBf16ToI1 = inType.isBF16() && outType.isInteger(1);
    const bool isF32ToI1 = inType.isF32() && outType.isInteger(1);
    const bool isF16ToI1 = inType.isF16() && outType.isInteger(1);
    if (isI64ToI1 || isI32ToI1 || isI16ToI1 || isI8ToI1 || isBf16ToI1 ||
        isF32ToI1 || isF16ToI1) {
      Value castResult = castSrcTypeToI1ByVCmp(op, inType, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isI16ToI64 = inType.isInteger(16) && outType.isInteger(64);
    if (isI16ToI64) {
      Value inValue = op.getInputs()[0];
      Value castF32Value = hfusion::castTo(
          rewriter, inValue, rewriter.getF32Type(), hfusion::RoundMode::RINT);

      Value castI64Value =
          hfusion::castTo(rewriter, castF32Value, rewriter.getI64Type());
      rewriter.replaceOp(op, castI64Value);
      return success();
    }

    const bool isI16ToI32 = inType.isInteger(16) && outType.isInteger(32);
    if (isI16ToI32) {
      Value inValue = op.getInputs()[0];
      Value castF32Value = hfusion::castTo(
          rewriter, inValue, rewriter.getF32Type(), hfusion::RoundMode::RINT);

      Value castI32Value =
          hfusion::castTo(rewriter, castF32Value, rewriter.getI32Type());
      rewriter.replaceOp(op, castI32Value);
      return success();
    }

    return failure();
  }
};

/// get the constant integer value which is used mask sign bit
/// e.g. 8 bit mask value is 0b01111111
Value getSignMaskConstValue(PatternRewriter &rewriter, Location loc,
                            int bitwidth) {
  if (bitwidth == 32) {
    arith::ConstantOp maskCstOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(0x7FFFFFFF));
    return maskCstOp->getResults()[0];
  }
  if (bitwidth == 16) {
    arith::ConstantOp maskCstOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI16IntegerAttr(0x7FFF));
    return maskCstOp->getResults()[0];
  }
  llvm_unreachable("unsupported bitwidth");
}

/// get the complement of constant integer value of inf
/// e.g. 16 bit float inf is 0b0111110000000000
///      32 bit float inf is 0b01111111100000000000000000000000
Value getComplementOfInfConstValue(PatternRewriter &rewriter, Location loc,
                                   int bitwidth) {
  if (bitwidth == 32) {
    arith::ConstantOp maskCstOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(-1 * (0x7F800000)));
    return maskCstOp->getResults()[0];
  }
  if (bitwidth == 16) {
    arith::ConstantOp maskCstOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI16IntegerAttr(-1 * (0x7C00)));
    return maskCstOp->getResults()[0];
  }
  llvm_unreachable("unsupported bitwidth");
}

/// mask the sign bit of f32/f16 type input
Value maskSignBit(PatternRewriter &rewriter, Location loc, Value input) {
  Type elemType = getElementTypeOrSelf(input.getType());
  Type castType = rewriter.getIntegerType(elemType.getIntOrFloatBitWidth());
  // 1. init mask constant
  // 2. vdup(7FFF) : (I32/I16)
  auto fillInit =
      utils::createEmptyOpWithTargetElemType(rewriter, loc, input, castType);
  auto fillOp = rewriter.create<linalg::FillOp>(
      loc,
      ValueRange{getSignMaskConstValue(rewriter, loc,
                                       elemType.getIntOrFloatBitWidth())},
      ValueRange{fillInit});
  auto bitcastEmptyOp =
      utils::createEmptyOpWithTargetElemType(rewriter, loc, fillInit, castType);
  auto shapedType = dyn_cast_if_present<ShapedType>(input.getType());
  auto bitcastOp = rewriter.create<hfusion::BitcastOp>(
      loc, TypeRange{shapedType.clone(castType)}, ValueRange{input},
      ValueRange{bitcastEmptyOp});
  auto bitcastInit = bitcastOp->getResults()[0];
  auto vandInit = utils::createEmptyOp(rewriter, loc, bitcastInit);

  // 3. vand(input, input, vdup) : (I32/I16)
  auto vandOP =
      hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                              hfusion::BinaryFnAttr>(
          rewriter, loc, hfusion::BinaryFn::vand,
          ValueRange{bitcastInit, fillOp->getResults()[0]},
          ValueRange{vandInit});
  return vandOP->getResults()[0];
}

/// minus the input with integer value of inf
Value minusInfConstValue(PatternRewriter &rewriter, Location loc, Value input) {
  // namely add complement of integer value of inf
  // e.g. vadd(input, input, -1 * f16_inf).
  auto addInit = utils::createEmptyOp(rewriter, loc, input);
  Type elemType = getElementTypeOrSelf(input.getType());
  auto addOp = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                       linalg::BinaryFn, linalg::BinaryFnAttr>(
      rewriter, loc, linalg::BinaryFn::add,
      ValueRange{input, getComplementOfInfConstValue(
                            rewriter, loc, elemType.getIntOrFloatBitWidth())},
      ValueRange{addInit});
  return addOp->getResults()[0];
}

struct NormalizeIsInfOp : public OpRewritePattern<hfusion::IsInfOp> {
public:
  using OpRewritePattern<hfusion::IsInfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::IsInfOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Type elemType = getElementTypeOrSelf(input.getType());
    if (!elemType.isF16() && !elemType.isBF16() && !elemType.isF32()) {
      return failure();
    }

    // step 1: mask sign bit.
    // 1. vdup(7FFF) : (I32/I16)
    auto loc = op->getLoc();
    auto maskedSignValue = maskSignBit(rewriter, loc, input);

    // step 2: compared with negtive Infinity
    // 3.vadd(input, input, neg_inf_bitcast_as_int).
    auto minusInfValue = minusInfConstValue(rewriter, loc, maskedSignValue);
    // 4.vabs(input, input) : (F16/F32)
    auto rebitcastEmptyOp = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, minusInfValue, elemType);
    auto shapedType = dyn_cast_if_present<ShapedType>(input.getType());
    auto rebitcastOp = rewriter.create<hfusion::BitcastOp>(
        loc, TypeRange{shapedType.clone(elemType)}, ValueRange{minusInfValue},
        ValueRange{rebitcastEmptyOp});
    Value rebitcastInit = rebitcastOp->getResults()[0];
    auto absInit = utils::createEmptyOp(rewriter, loc, rebitcastInit);
    auto absOP = hfusion::createUnaryOp<linalg::ElemwiseUnaryOp,
                                        linalg::UnaryFn, linalg::UnaryFnAttr>(
        rewriter, loc, linalg::UnaryFn::abs, ValueRange{rebitcastInit},
        ValueRange{absInit});

    // 5.vmin(input, input, 1) : (I32/I16)
    Type castType = rewriter.getIntegerType(elemType.getIntOrFloatBitWidth());
    auto bitcastOpForMinEmptyOp = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, absOP->getResults()[0], castType);
    auto bitcastOpForMin = rewriter.create<hfusion::BitcastOp>(
        loc, TypeRange{shapedType.clone(castType)},
        ValueRange{absOP->getResults()[0]}, ValueRange{bitcastOpForMinEmptyOp});
    Value bitcastOpForMinInit = bitcastOpForMin.getResults()[0];
    auto minInit = utils::createEmptyOp(rewriter, loc, bitcastOpForMinInit);
    arith::ConstantOp posOneOp = rewriter.create<arith::ConstantOp>(
        loc, castType, rewriter.getIntegerAttr(castType, 1));
    auto minOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::min_signed,
            ValueRange{bitcastOpForMinInit, posOneOp->getResults()[0]},
            ValueRange{minInit});

    // 6.vmuls(input, input, -1) : (I32/I16)
    arith::ConstantOp negOneOp = rewriter.create<arith::ConstantOp>(
        loc, castType, rewriter.getIntegerAttr(castType, -1));
    auto mulOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul,
            ValueRange({minOp->getResults()[0], negOneOp->getResults()[0]}),
            minOp->getResults()[0]);

    // 7.vadds(input, input, 1) : (I32/I16)
    auto addsOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::add,
            ValueRange({mulOp->getResults()[0], posOneOp->getResults()[0]}),
            mulOp->getResults()[0]);

    // 8.cast(input, int->i1)
    auto roundingAttr =
        rewriter.getAttr<hfusion::RoundModeAttr>(hfusion::RoundMode::RINT);
    auto modeAttr = rewriter.getNamedAttr(hfusion::RoundModeAttr::getMnemonic(),
                                          roundingAttr);
    hfusion::CastOp castToDst = rewriter.create<hfusion::CastOp>(
        loc, TypeRange(op.getOutput()), addsOp->getResults()[0], op.getOutput(),
        modeAttr);
    rewriter.replaceOp(op, castToDst);
    return success();
  }
};

struct NormalizeIsNanOp : public OpRewritePattern<hfusion::IsNanOp> {
public:
  using OpRewritePattern<hfusion::IsNanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::IsNanOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Type elemType = getElementTypeOrSelf(input.getType());
    if (!elemType.isF16() && !elemType.isBF16() && !elemType.isF32()) {
      return failure();
    }

    auto loc = op->getLoc();
    auto *itselfCmp = createCmpOp(rewriter, loc, op.getInput(), op.getInput(),
                                  CompareFn::vne);
    rewriter.replaceOp(op, itselfCmp);

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

/// Convert dense tensor/memref with only 1 element to scalar.
static std::optional<Value>
getScalarFromConstantOp(PatternRewriter &rewriter, Location loc,
                        arith::ConstantOp constant) {
  auto denseAttr = dyn_cast<DenseIntOrFPElementsAttr>(constant.getValue());
  if (!denseAttr) {
    return std::nullopt;
  }

  auto elemType = denseAttr.getElementType();
  if (!elemType.isIntOrIndexOrFloat()) {
    return std::nullopt;
  }

  TypedAttr typedAttr =
      elemType.isIntOrIndex()
          ? (TypedAttr)*denseAttr.getValues<IntegerAttr>().begin()
          : (TypedAttr)*denseAttr.getValues<FloatAttr>().begin();

  return rewriter.create<arith::ConstantOp>(loc, elemType, typedAttr);
}

/// Convert dense tensor/memref with only 1 element to scalar.
static std::optional<Value>
singleElemDenseTensorToScalar(Value operand, PatternRewriter &rewriter) {
  auto constantOp = operand.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return std::nullopt;

  auto shapedType = dyn_cast<ShapedType>(constantOp.getType());
  if (!shapedType)
    return std::nullopt;

  auto shape = shapedType.getShape();
  if (shape.size() > 1 || (!shape.empty() && shape[0] > 1))
    return std::nullopt;

  return getScalarFromConstantOp(rewriter, operand.getLoc(), constantOp);
}

template <typename OpType>
struct NormalizeScalarLikeTensorOp : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    bool isConverted = false;
    SmallVector<Value> inputsNew;
    for (auto inp : op.getInputs()) {
      auto inpNew = singleElemDenseTensorToScalar(inp, rewriter);
      if (inpNew.has_value()) {
        inputsNew.push_back(*inpNew);
        isConverted = true;
      } else {
        inputsNew.push_back(inp);
      }
    }

    SmallVector<Value> outputsNew;
    for (auto out : op.getOutputs()) {
      auto outNew = singleElemDenseTensorToScalar(out, rewriter);
      if (outNew.has_value()) {
        outputsNew.push_back(*outNew);
        isConverted = true;
      } else {
        outputsNew.push_back(out);
      }
    }

    if (!isConverted)
      return failure();

    IRMapping mapper;
    mapper.map(op.getInputs(), ValueRange(inputsNew));
    mapper.map(op.getOutputs(), ValueRange(outputsNew));

    Operation *clonedOp = rewriter.clone(*op, mapper);
    rewriter.replaceOp(op, clonedOp);
    return success();
  }
};

/// Convert linalg.broadcast to linalg.fill if input operand only has one elem.
struct NormalizeScalarLikeTensorLinalgBrcOp
    : public OpRewritePattern<linalg::BroadcastOp> {
public:
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto optInpNew = singleElemDenseTensorToScalar(op.getInput(), rewriter);
    if (!optInpNew.has_value())
      return failure();

    auto fillOp = rewriter.create<linalg::FillOp>(
        op->getLoc(), ValueRange(*optInpNew), op.getInit());
    rewriter.replaceOp(op, fillOp);
    return success();
  }
};

/// normalize i8/i32 CompareOp
///   i8 -> f16
///   i32 -> i64 (except vne and veq)
/// e.g.
///   hfusion.compare ins(%src1, %src2 : tensor<6x6xi32>, tensor<6x6xi32>)
/// is normalized to
///   %cast1 = hfusion.cast %src1 : tensor<6x6xi32> to tensor<6x6xi64>
///   %cast2 = hfusion.cast %src2 : tensor<6x6xi32> to tensor<6x6xi64>
///   hfusion.compare ins(%cast1, %cast2 : tensor<6x6xi64>, tensor<6x6xi64>)
struct NormalizeI8I32CmpOp : public OpRewritePattern<CompareOp> {
public:
  using OpRewritePattern<CompareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    Value lhs = op.getInputs()[0];
    Value rhs = op.getInputs()[1];
    Type lhsElemType = getElementTypeOrSelf(lhs.getType());
#ifndef NDEBUG
    Type rhsElemType = getElementTypeOrSelf(rhs.getType());
    assert(lhsElemType == rhsElemType && "lhs and rhs elemType mismatch");
#endif

    Type targetType = rewriter.getI64Type();
    hfusion::CompareFn cmpFn = op.getCompareFn();
    if (lhsElemType.isInteger(8)) {
      targetType = rewriter.getF16Type();
    } else if (lhsElemType.isInteger(32) && cmpFn != hfusion::CompareFn::vne &&
               cmpFn != hfusion::CompareFn::veq) {
      targetType = rewriter.getI64Type();
    } else {
      return failure();
    }

    hfusion::RoundMode rounding =
        utils::selectRoundMode<hfusion::RoundMode>(lhsElemType, targetType);

    Value castLhs, castRhs;
    static DenseMap<hfusion::CompareFn, hfusion::CompareFn> cmpFnMap = {
        {hfusion::CompareFn::vule, hfusion::CompareFn::vle},
        {hfusion::CompareFn::vult, hfusion::CompareFn::vlt},
        {hfusion::CompareFn::vuge, hfusion::CompareFn::vge},
        {hfusion::CompareFn::vugt, hfusion::CompareFn::vgt},
    };
    
    if (cmpFnMap.contains(cmpFn)) {
      castLhs = hfusion::castTo(rewriter, lhs, targetType, rounding, std::nullopt, true,
                                hfusion::TypeFn::cast_unsigned);
      castRhs = hfusion::castTo(rewriter, rhs, targetType, rounding, std::nullopt, true,
                                hfusion::TypeFn::cast_unsigned);
      cmpFn = cmpFnMap[cmpFn];
    } else {
      castLhs = hfusion::castTo(rewriter, lhs, targetType, rounding);
      castRhs = hfusion::castTo(rewriter, rhs, targetType, rounding);
    }
    
    auto newCmpOp =
        createCmpOp(rewriter, op->getLoc(), castLhs, castRhs, cmpFn);
    rewriter.replaceOp(op, newCmpOp);
    return success();
  }
};

template <typename FuncType, typename FuncAttrType, typename OpType>
static NamedAttribute getOpFunAttr(OpType op, PatternRewriter &rewriter) {
  FuncType func = op.getFunAttr().getValue();
  auto attr = rewriter.getAttr<FuncAttrType>(func);
  auto funAttr = rewriter.getNamedAttr("fun", attr);
  return funAttr;
}

template <typename OpType,
          typename = std::enable_if<
              std::is_same_v<OpType, linalg::ElemwiseBinaryOp> ||
              std::is_same_v<OpType, linalg::ElemwiseUnaryOp> ||
              std::is_same_v<OpType, hfusion::ElemwiseBinaryOp> ||
              std::is_same_v<OpType, hfusion::ElemwiseUnaryOp> ||
              std::is_same_v<OpType, hfusion::SelectOp>>>
static SmallVector<NamedAttribute> getOpAttr(OpType op,
                                             PatternRewriter &rewriter) {
  if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
    return {getOpFunAttr<linalg::BinaryFn, linalg::BinaryFnAttr>(op, rewriter)};
  } else if constexpr (std::is_same_v<OpType, linalg::ElemwiseUnaryOp>) {
    return {getOpFunAttr<linalg::UnaryFn, linalg::UnaryFnAttr>(op, rewriter)};
  } else if constexpr (std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
    return {
        getOpFunAttr<hfusion::BinaryFn, hfusion::BinaryFnAttr>(op, rewriter)};
  } else if constexpr (std::is_same_v<OpType, hfusion::ElemwiseUnaryOp>) {
    return {getOpFunAttr<hfusion::UnaryFn, hfusion::UnaryFnAttr>(op, rewriter)};
  } else if constexpr (std::is_same_v<OpType, hfusion::SelectOp>) {
    // no extra attrs
    return {};
  } else
    llvm_unreachable("Unsupport Normalize OpType.");
}

static void replaceI1ResultsWithTargetType(const SmallVector<Value> &oldResults,
                                           const SmallVector<Value> &newResults,
                                           PatternRewriter &rewriter,
                                           bool enableOverflow = true) {
  assert(oldResults.size() == newResults.size() &&
         "result sizes mismatch when replace op results");
  for (const auto [idx, oldResult] : llvm::enumerate(oldResults)) {
    Value newResult = newResults[idx];
    if (!isI1ElemType(oldResult.getType())) {
      rewriter.replaceAllUsesWith(oldResult, newResult);
      continue;
    }

    Value castResult =
        castTo(rewriter, newResult, rewriter.getI1Type(),
               hfusion::RoundMode::TRUNC, std::nullopt, enableOverflow);
    rewriter.replaceAllUsesWith(oldResult, castResult);
  }
}

struct CastOptions {
  bool isUnsignedOp = false;
  bool enableOverflow = true;
};

static void replaceI8ResultsWithTargetType(const SmallVector<Value> &oldResults,
                                           const SmallVector<Value> &newResults,
                                           PatternRewriter &rewriter,
                                           bool enableOverflow = true,
                                           bool isUnsigned = false) {
  assert(oldResults.size() == newResults.size() &&
         "result sizes mismatch when replace op results");
  for (const auto [idx, oldResult] : llvm::enumerate(oldResults)) {
    Value newResult = newResults[idx];
    if (!isI8ElemType(oldResult.getType())) {
      rewriter.replaceAllUsesWith(oldResult, newResult);
      continue;
    }

    Value castResult = castTo(rewriter, newResult, rewriter.getI8Type(),
                              hfusion::RoundMode::TRUNC,
                              std::nullopt, enableOverflow,
                              isUnsigned ? hfusion::TypeFn::cast_unsigned : hfusion::TypeFn::cast_signed
                              );
    rewriter.replaceAllUsesWith(oldResult, castResult);
  }
}

static void
replaceI16ResultsWithTargetType(const SmallVector<Value> &oldResults,
                                const SmallVector<Value> &newResults,
                                PatternRewriter &rewriter) {
  assert(oldResults.size() == newResults.size() &&
         "result sizes mismatch when replace op results");
  for (const auto [idx, oldResult] : llvm::enumerate(oldResults)) {
    Value newResult = newResults[idx];
    if (!isI16ElemType(oldResult.getType())) {
      rewriter.replaceAllUsesWith(oldResult, newResult);
      continue;
    }

    Value overflowResult =
        hfusion::OverflowProcess(rewriter, newResult, rewriter.getI16Type());
    Value castResult = castTo(rewriter, overflowResult, rewriter.getI16Type());
    rewriter.replaceAllUsesWith(oldResult, castResult);
  }
}

template <typename targetType,
          typename = std::enable_if<(std::is_same_v<bool, targetType> ||
                                     std::is_same_v<int8_t, targetType>)>>
static void replaceResultsWithTargetType(const SmallVector<Value> &oldResults,
                                         const SmallVector<Value> &newResults,
                                         PatternRewriter &rewriter,
                                         std::optional<CastOptions> castOpts) {
  if constexpr (std::is_same_v<bool, targetType>) {
    replaceI1ResultsWithTargetType(oldResults, newResults, rewriter);
  }
  if constexpr (std::is_same_v<int8_t, targetType>) {
    if (castOpts.has_value()) {
      auto opts = castOpts.value();
      replaceI8ResultsWithTargetType(oldResults, newResults, rewriter, opts.enableOverflow, opts.isUnsignedOp);
    } else {
      replaceI8ResultsWithTargetType(oldResults, newResults, rewriter);
    }
  }
}

SmallVector<Value> normalizeF16ToF32(PatternRewriter &rewriter,
                                     const SmallVector<Value> &values) {
  SmallVector<Value> result;
  for (Value v : values) {
    if (!isF16ElemType(v.getType())) {
      result.push_back(v);
      continue;
    }
    Value castResult = castTo(rewriter, v, rewriter.getF32Type());
    result.push_back(castResult);
  }
  return result;
}

template <typename srcType, typename targetType,
          typename = std::enable_if<(std::is_same_v<targetType, Float16Type> ||
                                     std::is_same_v<targetType, Float32Type>)>>
SmallVector<Value> normalizeSrcToTargetType(PatternRewriter &rewriter,
                                            const SmallVector<Value> &values) {
  SmallVector<Value> result;
  for (Value v : values) {
    if (!isElemType<srcType>(v.getType())) {
      result.push_back(v);
      continue;
    }

    Type dstType = rewriter.getType<targetType>();
    Value castResult = castTo(rewriter, v, dstType);
    result.push_back(castResult);
  }
  return result;
}

arith::CmpFPredicate getCmpFloatPredicate(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return arith::CmpFPredicate::OEQ;
  case arith::CmpIPredicate::ne:
    return arith::CmpFPredicate::ONE;
  case arith::CmpIPredicate::slt:
    return arith::CmpFPredicate::OLT;
  case arith::CmpIPredicate::sle:
    return arith::CmpFPredicate::OLE;
  case arith::CmpIPredicate::sgt:
    return arith::CmpFPredicate::OGT;
  case arith::CmpIPredicate::sge:
    return arith::CmpFPredicate::OGE;
  case arith::CmpIPredicate::ult:
    return arith::CmpFPredicate::OLT;
  case arith::CmpIPredicate::ule:
    return arith::CmpFPredicate::OLE;
  case arith::CmpIPredicate::ugt:
    return arith::CmpFPredicate::OGT;
  case arith::CmpIPredicate::uge:
    return arith::CmpFPredicate::OGE;
  }
  llvm_unreachable("unexpected arith::CmpIPredicate");
}

Operation *cloneArithOp(PatternRewriter &rewriter, Location loc,
                        Operation *bodyOp, IRMapping &mapper) {
  const DenseMap<Value, Value> &valueMap = mapper.getValueMap();
  Value oldLhs = bodyOp->getOperand(0);
  Value oldRhs = bodyOp->getOperand(1);
  Value lhs = valueMap.at(oldLhs);
  Value rhs = valueMap.at(oldRhs);
  if (isa<arith::AddFOp>(bodyOp) || isa<arith::AddIOp>(bodyOp)) {
    auto newAddf = rewriter.create<arith::AddFOp>(loc, lhs, rhs);
    return newAddf;
  }
  if (isa<arith::MulFOp>(bodyOp) || isa<arith::MulIOp>(bodyOp)) {
    auto newMulf = rewriter.create<arith::MulFOp>(loc, lhs, rhs);
    return newMulf;
  }
  if (isa<arith::SubFOp>(bodyOp) || isa<arith::SubIOp>(bodyOp)) {
    auto newSubf = rewriter.create<arith::SubFOp>(loc, lhs, rhs);
    return newSubf;
  }
  if (auto cmpi = dyn_cast<arith::CmpIOp>(bodyOp)) {
    auto pred = getCmpFloatPredicate(cmpi.getPredicate());
    auto cmpf = rewriter.create<arith::CmpFOp>(loc, pred, lhs, rhs);
    return cmpf;
  }
  if (auto cmpf = dyn_cast<arith::CmpFOp>(bodyOp)) {
    auto newCmpf =
        rewriter.create<arith::CmpFOp>(loc, cmpf.getPredicate(), lhs, rhs);
    return newCmpf;
  }
  if (isa<arith::DivFOp>(bodyOp) || isa<arith::DivSIOp>(bodyOp) ||
      isa<arith::DivUIOp>(bodyOp)) {
    auto newDivf = rewriter.create<arith::DivFOp>(loc, lhs, rhs);
    return newDivf;
  }
  if (isa<arith::MaximumFOp>(bodyOp) || isa<arith::MaxSIOp>(bodyOp) ||
      isa<arith::MaxUIOp>(bodyOp)) {
    auto newMaxf = rewriter.create<arith::MaximumFOp>(loc, lhs, rhs);
    return newMaxf;
  }
  if (isa<arith::MinimumFOp>(bodyOp) || isa<arith::MinSIOp>(bodyOp) ||
      isa<arith::MinUIOp>(bodyOp)) {
    auto newMinf = rewriter.create<arith::MinimumFOp>(loc, lhs, rhs);
    return newMinf;
  }
  llvm::report_fatal_error("unsupported body op to map");
}

Operation *mapReduceBodyOpToFloat(PatternRewriter &rewriter, Location loc,
                                  Operation *bodyOp, Type srcType,
                                  IRMapping &mapper) {
  if (isa<linalg::YieldOp>(bodyOp)) {
    return rewriter.clone(*bodyOp, mapper);
  }
  if (auto select = dyn_cast<arith::SelectOp>(bodyOp)) {
    Value cond = mapper.lookup(select.getCondition());
    Value trueValue = mapper.lookup(select.getTrueValue());
    Value falseValue = mapper.lookup(select.getFalseValue());
    auto newSelect = rewriter.create<arith::SelectOp>(
        loc, trueValue.getType(), cond, trueValue, falseValue);
    return newSelect;
  }
  // simply clone op with no f16 or i8 operand
  assert(bodyOp->getNumOperands() == 2 && "only support binary arith op");
  Value oldLhs = bodyOp->getOperand(0);
  Value oldRhs = bodyOp->getOperand(1);
  if (srcType == rewriter.getI8Type() && !isI8ElemType(oldLhs.getType()) &&
      !isI8ElemType(oldRhs.getType())) {
    return rewriter.clone(*bodyOp, mapper);
  }
  if (srcType == rewriter.getF16Type() && !isF16ElemType(oldLhs.getType()) &&
      !isF16ElemType(oldRhs.getType())) {
    return rewriter.clone(*bodyOp, mapper);
  }

  // convert arith op from srcType to targetType
  return cloneArithOp(rewriter, loc, bodyOp, mapper);
}

Operation *createNewReduceOp(linalg::ReduceOp op, PatternRewriter &rewriter,
                             Type srcType, Type targetType,
                             SmallVector<Value> &newInputs,
                             SmallVector<Value> &newInits) {
  bool isF16ToF32 = false;
  if (targetType == rewriter.getF32Type() && srcType == rewriter.getF16Type()) {
    isF16ToF32 = true;
  }

  IRMapping mapper;
  for (const auto &[idx, operand] : llvm::enumerate(op.getInputs())) {
    mapper.map(operand, newInputs[idx]);
  }
  for (const auto &[idx, operand] : llvm::enumerate(op.getInits())) {
    mapper.map(operand, newInits[idx]);
  }

  Operation *newOp = rewriter.cloneWithoutRegions(*op, mapper);
  // change f16 result types to targetType
  for (const auto &[idx, res] : llvm::enumerate(op->getResults())) {
    ShapedType shapedType = dyn_cast_or_null<ShapedType>(res.getType());
    bool isTargetType =
        isF16ToF32 ? isF16ElemType(shapedType) : isI8ElemType(shapedType);
    if (!shapedType || !isTargetType) {
      continue;
    }
    auto srcShapedType = shapedType.clone(targetType);
    newOp->getResult(idx).setType(srcShapedType);
  }

  // create reduce op inner region with srcType changed to targetType
  Region &newRegion = newOp->getRegions().front();
  Block *newBlock = rewriter.createBlock(&newRegion);
  rewriter.setInsertionPointToStart(newBlock);

  Block *block = &op.getRegion().front();
  for (BlockArgument bbArg : block->getArguments()) {
    // change op region block srcType arg using targetType
    Type argType = bbArg.getType();
    bool isSrcType = isF16ToF32 ? argType.isF16() : argType.isInteger(8);
    Type newArgType = (isSrcType ? targetType : argType);
    mapper.map(bbArg, newBlock->addArgument(newArgType, bbArg.getLoc()));
  }

  Location loc = newRegion.getLoc();
  for (Operation &bodyOp : *block) {
    // change op within region to targetType.
    Operation *newBodyOp =
        mapReduceBodyOpToFloat(rewriter, loc, &bodyOp, srcType, mapper);
    mapper.map(bodyOp.getResults(), newBodyOp->getResults());
  }
  rewriter.setInsertionPointAfter(newOp);
  return newOp;
}

static DenseSet<linalg::BinaryFn> unsignedLinalgBinFnSet = {
    linalg::BinaryFn::max_unsigned,
    linalg::BinaryFn::min_unsigned,
};
static DenseSet<linalg::BinaryFn> disableOverflowLinalgBinFnSet = {
    linalg::BinaryFn::max_unsigned,
    linalg::BinaryFn::min_unsigned,
};

template <typename OpType>
std::optional<CastOptions> getCastOptsForUint8(OpType op) {
  struct CastOptions castOpts;
  if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
    // some ops' signed info can be determined by func
    auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
    linalg::BinaryFn func = binOp.getFun();
    if (unsignedLinalgBinFnSet.contains(func)) {
      castOpts.isUnsignedOp = true;
    }
    if (disableOverflowLinalgBinFnSet.contains(func)) {
      castOpts.enableOverflow = false;
    }
    return castOpts;
  }

  return std::nullopt;
}

template <typename ElemType>
SmallVector<Value> normalizeToTargetType(PatternRewriter &rewriter,
                                         const SmallVector<Value> &values,
                                         Type targetType,
                                         std::optional<CastOptions> castOpts) {
  SmallVector<Value> result;
  for (Value v : values) {
    if (!isElemType<ElemType>(v.getType())) {
      result.push_back(v);
      continue;
    }
    Value castResult = castTo(rewriter, v, targetType,
        castOpts.has_value() && castOpts.value().isUnsignedOp
            ? hfusion::TypeFn::cast_unsigned : hfusion::TypeFn::cast_signed);
    result.push_back(castResult);
  }
  return result;
}

template <typename ElemType, typename OpType>
struct NormalizeToTargetType : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (!hasElemType<ElemType>(op.getInputs()) &&
        !hasElemType<ElemType>(op.getOutputs())) {
      return failure();
    }

    if (isSupportOperand<ElemType>(op)) {
      return failure();
    }

    bool computeByF16 = shoudComputeByF16(op);
    bool computeByF32 = shoudComputeByF32(op);
    if (!computeByF16 && !computeByF32) {
      return failure();
    }

    auto castOpts = getCastOptsForUint8(op);

    Type targetType;
    if (computeByF16) {
      targetType = rewriter.getF16Type();
    } else if (computeByF32) {
      targetType = rewriter.getF32Type();
    } else {
      llvm_unreachable("Unsupported Op.");
    }
    SmallVector<Value> newInputs =
        normalizeToTargetType<ElemType>(rewriter, op.getInputs(), targetType, castOpts);
    SmallVector<Value> newOutputs =
        normalizeToTargetType<ElemType>(rewriter, op.getOutputs(), targetType, castOpts);
    Operation *newOp = createBodyOp(op, newInputs, newOutputs, rewriter);
    if (std::is_same_v<OpType, hfusion::SelectOp>) {
      replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                     rewriter, false);
    } else {
      // TODO: set argument enableOverflow = false inside for all non-arithmatic
      // op type
      replaceResultsWithTargetType<ElemType>(op->getResults(),
                                             newOp->getResults(), rewriter, castOpts);
    }
    return success();
  }

private:
  template <typename OpElemType, typename std::enable_if_t<!std::is_same_v<OpElemType, int8_t>, int> = 0>
  bool isSupportOperand(OpType op) const {
    return false;
  }

  template <typename OpElemType, typename std::enable_if_t<std::is_same_v<OpElemType, int8_t>, int> = 0>
  bool isSupportOperand(OpType op) const {
    if constexpr (std::is_same_v<OpType, linalg::FillOp> ||
                  std::is_same_v<OpType, linalg::BroadcastOp> ||
                  std::is_same_v<OpType, linalg::CopyOp> ||
                  std::is_same_v<OpType, hfusion::CastOp>) {
      return true;
    }

    if constexpr (std::is_same_v<OpType, hfusion::SelectOp>) {
      return false;
    }

    if constexpr (std::is_same_v<OpType, linalg::ElemwiseUnaryOp> ||
                  std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      // no linalg elemwise unary/binary op support i8
      return false;
    }

    if constexpr (std::is_same_v<OpType, hfusion::ElemwiseUnaryOp>) {
      // only part of hfusion elemwise unary op support i8
      auto unaryOp = cast<hfusion::ElemwiseUnaryOp>(op);
      hfusion::UnaryFn func = unaryOp.getFun();
      static DenseSet<hfusion::UnaryFn> unarySet = {hfusion::UnaryFn::vnot};
      return unarySet.contains(func);
    }

    if constexpr (std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
      // only part of hfusion elemwise binary op support both i8
      auto binOp = cast<hfusion::ElemwiseBinaryOp>(op);
      hfusion::BinaryFn func = binOp.getFun();
      // bit operation can support b8 operand
      static DenseSet<hfusion::BinaryFn> binarySet = {hfusion::BinaryFn::vor,
                                                      hfusion::BinaryFn::vand,
                                                      hfusion::BinaryFn::vxor};
      return binarySet.contains(func);
    }
    return false;
  }

  bool shoudComputeByF16(OpType op) const {
    if constexpr (std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
      auto binOp = cast<hfusion::ElemwiseBinaryOp>(op);
      hfusion::BinaryFn func = binOp.getFun();
      // can compute on i8 directly and no need cast to f16
      static DenseSet<hfusion::BinaryFn> binarySet = {
          // can compute on i8 directly and no need cast to f16
          hfusion::BinaryFn::shli, hfusion::BinaryFn::shrsi,
          hfusion::BinaryFn::shrui,
          // should compute on f32 for high precision and change to use float
          // ops to compute f32 data
          hfusion::BinaryFn::ceildivsi, hfusion::BinaryFn::floordivsi,
          hfusion::BinaryFn::ceildivui, hfusion::BinaryFn::mod,
          hfusion::BinaryFn::modui};
      return !binarySet.contains(func);
    } else if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
      linalg::BinaryFn func = binOp.getFun();
      // should compute on f32 for high precision
      static DenseSet<linalg::BinaryFn> binarySet = {
          linalg::BinaryFn::mul, linalg::BinaryFn::div_unsigned,
          linalg::BinaryFn::div, linalg::BinaryFn::add, linalg::BinaryFn::sub};
      return !binarySet.contains(func);
    }
    return true;
  }

  bool shoudComputeByF32(OpType op) const {
    if constexpr (std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
      auto binOp = cast<hfusion::ElemwiseBinaryOp>(op);
      hfusion::BinaryFn func = binOp.getFun();
      static DenseSet<hfusion::BinaryFn> binarySet = {hfusion::BinaryFn::mod,
                                                      hfusion::BinaryFn::modui};
      return binarySet.contains(func);
    } else if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
      linalg::BinaryFn func = binOp.getFun();
      static DenseSet<linalg::BinaryFn> binarySet = {
          linalg::BinaryFn::mul, linalg::BinaryFn::add, linalg::BinaryFn::sub};
      return binarySet.contains(func);
    }
    return false;
  }

  Operation *createBodyOp(OpType op, SmallVector<Value> &newInputs,
                          SmallVector<Value> &newOutputs,
                          PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    SmallVector<NamedAttribute> attrs = getOpAttr(op, rewriter);
    if constexpr (std::is_same_v<OpType, hfusion::SelectOp> ||
                  std::is_same_v<OpType, linalg::ElemwiseUnaryOp> ||
                  std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
      // no attr needs to be changed
      return rewriter.create<OpType>(loc, ValueRange{newInputs},
                                     ValueRange{newOutputs}, attrs);
    }

    if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      static DenseMap<linalg::BinaryFn, hfusion::BinaryFn> binAttrMap = {
          {linalg::BinaryFn::max_unsigned, hfusion::BinaryFn::maxf},
          {linalg::BinaryFn::max_signed, hfusion::BinaryFn::maxf},
          {linalg::BinaryFn::min_unsigned, hfusion::BinaryFn::minf},
          {linalg::BinaryFn::min_signed, hfusion::BinaryFn::minf},
      };
      auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
      linalg::BinaryFn linalgFn = binOp.getFunAttr().getValue();
      if (binAttrMap.contains(linalgFn)) {
        // convert linalg binary op to hfusion
        hfusion::BinaryFn hfusionFn = binAttrMap[linalgFn];
        return hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp,
                                       hfusion::BinaryFn,
                                       hfusion::BinaryFnAttr>(
            rewriter, loc, hfusionFn, ValueRange{newInputs},
            ValueRange{newOutputs});
      }
      // other linalg elemwise binary op can be created using origin attr
      return rewriter.create<linalg::ElemwiseBinaryOp>(
          loc, ValueRange{newInputs}, ValueRange{newOutputs}, attrs);
    }

    if constexpr (std::is_same_v<OpType, hfusion::ElemwiseUnaryOp>) {
      auto unaryOp = cast<hfusion::ElemwiseUnaryOp>(op);
      hfusion::UnaryFn unaryFn = unaryOp.getFun();
      if (unaryFn == hfusion::UnaryFn::absi) {
        // convert hfusion absi to linalg abs op
        return hfusion::createUnaryOp<linalg::ElemwiseUnaryOp, linalg::UnaryFn,
                                      linalg::UnaryFnAttr>(
            rewriter, loc, linalg::UnaryFn::abs, ValueRange{newInputs},
            ValueRange(newOutputs));
      }
      // other hfusion elemwise binary op can be created using origin attr
      return rewriter.create<hfusion::ElemwiseUnaryOp>(
          loc, ValueRange{newInputs}, ValueRange{newOutputs}, attrs);
    }
    llvm_unreachable("Unsupport OpType to create with F16 operand.");
  }
};

template <typename OpType>
struct NormalizeF16ToF32Type : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    SmallVector<Value> inputs = op.getInputs();
    if (!hasF16ElemType(inputs) || !shouldComputeByF32(op)) {
      return failure();
    }

    normalizeOpF16ToF32(rewriter, op);
    return success();
  }

private:
  void normalizeOpF16ToF32(PatternRewriter &rewriter, OpType op) const {
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> outputs = op.getOutputs();

    SmallVector<Value> newInputs = normalizeF16ToF32(rewriter, inputs);
    SmallVector<Value> newOutputs = normalizeF16ToF32(rewriter, outputs);

    SmallVector<NamedAttribute> attrs = getOpAttr(op, rewriter);
    Operation *newOp = rewriter.create<OpType>(
        op.getLoc(), ValueRange{newInputs}, ValueRange{newOutputs}, attrs);
    Value castResult =
        castTo(rewriter, newOp->getResults()[0], rewriter.getF16Type());
    rewriter.replaceAllUsesWith(op->getResults()[0], castResult);
  }

  bool shouldComputeByF32(OpType op) const {
    // cast f32 to compute for high precision
    // linalg unaryFn op set
    if (std::is_same_v<OpType, linalg::ElemwiseUnaryOp>) {
      static DenseSet<linalg::UnaryFn> linalgUnarySet = {linalg::UnaryFn::log};
      if (auto unaryOp = cast<linalg::ElemwiseUnaryOp>(op)) {
        linalg::UnaryFn unaryFn = unaryOp.getFun();
        if (linalgUnarySet.contains(unaryFn)) {
          return true;
        }
      }
    }

    // hfusion binaryFn op set
    if (std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
      static DenseSet<hfusion::BinaryFn> hfusionBinarySet = {
          hfusion::BinaryFn::powf};
      if (auto binaryOp = cast<hfusion::ElemwiseBinaryOp>(op)) {
        hfusion::BinaryFn binaryFn = binaryOp.getFun();
        if (hfusionBinarySet.contains(binaryFn)) {
          return true;
        }
      }
    }

    // hfusion unaryFn op set
    if (std::is_same_v<OpType, hfusion::ElemwiseUnaryOp>) {
      static DenseSet<hfusion::UnaryFn> hfusionUnarySet = {
          hfusion::UnaryFn::rsqrt};
      if (auto unaryOp = cast<hfusion::ElemwiseUnaryOp>(op)) {
        hfusion::UnaryFn unaryFn = unaryOp.getFun();
        if (hfusionUnarySet.contains(unaryFn)) {
          return true;
        }
      }
    }
    return false;
  }
};

template <typename CumOpType>
struct NormalizeCumOpF16ToF32Type : public OpRewritePattern<CumOpType> {
public:
  using OpRewritePattern<CumOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CumOpType op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = {op.getInput()};
    SmallVector<Value> outputs = {op.getOutput()};
    if ((!hasF16ElemType(inputs) && !hasF16ElemType(outputs)) ||
        !(std::is_same_v<CumOpType, hfusion::CumsumOp> ||
          std::is_same_v<CumOpType, hfusion::CumprodOp>)) {
      return failure();
    }
    auto newInputs = normalizeF16ToF32(rewriter, inputs);
    auto newOutputs = normalizeF16ToF32(rewriter, outputs);
    Operation *newOp = rewriter.create<CumOpType>(
        op.getLoc(), TypeRange{newOutputs}, newInputs[0], op.getCumDims(), op.getReverse());
    Value castResult =
        castTo(rewriter, newOp->getResults()[0], rewriter.getF16Type());
    rewriter.replaceAllUsesWith(op->getResults()[0], castResult);
    return success();
  }
};

template <>
struct NormalizeToTargetType<bool, linalg::ReduceOp>
    : public OpRewritePattern<linalg::ReduceOp> {
public:
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return failure();

    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits))
      return failure();
    Block &body = op.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    Operation *bodyOp = yieldOp.getValues()[0].getDefiningOp();
    if (isa<arith::AddIOp, arith::MaxUIOp, arith::MaxSIOp>(bodyOp)) {
      // As it is a bool, `add` and `max` can be converted into `or`.
      replaceBinary<arith::OrIOp>(bodyOp, rewriter);
      return success();
    }
    if (isa<arith::MulIOp, arith::MinUIOp, arith::MinSIOp>(bodyOp)) {
      // As it is a bool, `mul` and `min` can be converted into `and`.
      replaceBinary<arith::AndIOp>(bodyOp, rewriter);
      return success();
    }
    return failure();
  }

private:
  template <typename targetType>
  void replaceBinary(Operation *op, PatternRewriter &rewriter) const {
    if (op == nullptr) {
      return;
    }
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(op->getBlock());
    auto targetOp = rewriter.create<targetType>(op->getLoc(), op->getOperand(0),
                                                op->getOperand(1));
    rewriter.modifyOpInPlace(op, [&]() { op->replaceAllUsesWith(targetOp); });
  }
};

template <>
struct NormalizeToTargetType<bool, tensor::ConcatOp>
    : public OpRewritePattern<tensor::ConcatOp> {
public:
  using OpRewritePattern<tensor::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op->getResults();
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits))
      return failure();

    auto newInputs =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inputs);
    auto newOp = rewriter.create<tensor::ConcatOp>(op.getLoc(), op.getDim(),
                                                   ValueRange(newInputs));
    replaceI1ResultsWithTargetType({op.getResult()}, {newOp.getResult()},
                                   rewriter,
                                   /*enableOverflow*/ false);

    return success();
  }
};

template <>
struct NormalizeToTargetType<bool, CompareOp>
    : public OpRewritePattern<CompareOp> {
public:
  using OpRewritePattern<CompareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInputs();
    if (!hasI1ElemType(inputs))
      return failure();

    auto newInputs =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inputs);
    Value newLhs = newInputs[0];
    Value newRhs = newInputs[1];
    auto *newOp =
        createCmpOp(rewriter, op->getLoc(), newLhs, newRhs, op.getCompareFn());
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

// If operand1 and operand2 are of type I1 cast them to I16 to avoid unsupported
// I1 for hivm.hir.vsel
template <>
struct NormalizeToTargetType<bool, SelectOp>
    : public OpRewritePattern<SelectOp> {
public:
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value> inputs = op.getInputs();
    Value operand0 = inputs[0];
    Value operand1 = inputs[1];
    Value operand2 = inputs[2];
    if (!isI1ElemType(operand1.getType()) ||
        !isI1ElemType(operand2.getType())) {
      return failure();
    }

    Type i16type = rewriter.getI16Type();
    Value castedLhs = castTo(rewriter, operand1, i16type);
    Value castedRhs = castTo(rewriter, operand2, i16type);

    auto newSelect = rewriter.create<hfusion::SelectOp>(
        op.getLoc(), ValueRange({operand0, castedLhs, castedRhs}),
        castTo(rewriter, op.getOutputs()[0], rewriter.getI16Type()));

    replaceI1ResultsWithTargetType(op->getResults(), newSelect->getResults(),
                                   rewriter, false);

    rewriter.eraseOp(op);

    return success();
  }
};

template <>
struct NormalizeToTargetType<bool, linalg::TransposeOp>
    : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = {op.getInput()};
    SmallVector<Value> inits = op.getDpsInits();
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits))
      return failure();

    auto newInputs =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inputs);
    auto newInits =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inits);
    auto newOp = rewriter.create<linalg::TransposeOp>(
        op.getLoc(), newInputs.front(), newInits.front(), op.getPermutation());
    replaceI1ResultsWithTargetType(op.getResult(), newOp->getResults(),
                                   rewriter,
                                   /*enableOverflow*/ false);

    return success();
  }
};

template <>
struct NormalizeToTargetType<int8_t, linalg::ReduceOp>
    : public OpRewritePattern<linalg::ReduceOp> {
public:
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    if (!shouldComputeByFloat(op)) {
      return failure();
    }
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    if (!hasI8ElemType(inputs) && !hasI8ElemType(inits)) {
      return failure();
    }

    FloatType targetType = nullptr;
    SmallVector<Value> newInputs;
    SmallVector<Value> newInits;
    if (shoudComputeI8ByF32(op)) {
      targetType = rewriter.getF32Type();
      newInputs =
          normalizeSrcToTargetType<int8_t, Float32Type>(rewriter, inputs);
      newInits = normalizeSrcToTargetType<int8_t, Float32Type>(rewriter, inits);
    } else {
      targetType = rewriter.getF16Type();
      newInputs =
          normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inputs);
      newInits = normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inits);
    }

    Operation *newOp = createNewReduceOp(op, rewriter, rewriter.getI8Type(),
                                         targetType, newInputs, newInits);
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter);
    return success();
  }

private:
  bool shouldComputeByFloat(linalg::ReduceOp reduceOp) const {
    Block &body = reduceOp.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    auto bodyOp = yieldOp.getValues()[0].getDefiningOp();
    // can compute on i8 directly and no need cast to float.
    if (isa<arith::XOrIOp>(bodyOp) || isa<arith::OrIOp>(bodyOp) ||
        isa<arith::AndIOp>(bodyOp)) {
      return false;
    }
    return true;
  }

  bool shoudComputeI8ByF32(linalg::ReduceOp op) const {
    Block *block = &op.getRegion().front();
    for (Operation &bodyOp : *block) {
      if (dyn_cast_or_null<arith::AddIOp>(bodyOp)) {
        return true;
      }
    }
    return false;
  }
};

template <typename OpType>
Operation *createInterleaveLikeOp(OpType op, SmallVector<Value> &newInputs,
                                  SmallVector<Value> &newOutputs,
                                  PatternRewriter &rewriter) {
  Location loc = op.getLoc();

  if constexpr (std::is_same_v<OpType, hfusion::InterleaveOp>) {
    return rewriter.create<hfusion::InterleaveOp>(loc, ValueRange(newOutputs),
                                                  ValueRange(newInputs));
  }
  if constexpr (std::is_same_v<OpType, hfusion::DeinterleaveOp>) {
    return rewriter.create<hfusion::DeinterleaveOp>(
        loc, TypeRange(newOutputs), newInputs[0],
        op.getDeInterLeaveChannelIdx());
  }
  llvm_unreachable(
      "Unsupport interleaveLike OpType to create with F16 Operand.");
}

template <>
struct NormalizeToTargetType<int8_t, hfusion::InterleaveOp>
    : public OpRewritePattern<hfusion::InterleaveOp> {
public:
  using OpRewritePattern<hfusion::InterleaveOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::InterleaveOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInput();
    SmallVector<Value> inits = op.getODSResults(0);
    if (!hasI8ElemType(inputs) && !hasI8ElemType(inits)) {
      return failure();
    }

    auto newInputs =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inputs);
    auto newInits =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inits);
    Operation *newOp =
        createInterleaveLikeOp(op, newInputs, newInits, rewriter);
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter, false);

    return success();
  }
};

template <>
struct NormalizeToTargetType<int8_t, hfusion::DeinterleaveOp>
    : public OpRewritePattern<hfusion::DeinterleaveOp> {
public:
  using OpRewritePattern<hfusion::DeinterleaveOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::DeinterleaveOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getODSOperands(0);
    SmallVector<Value> inits = op.getODSResults(0);
    if (!hasI8ElemType(inputs) && !hasI8ElemType(inits)) {
      return failure();
    }

    auto newInputs =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inputs);
    auto newInits =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inits);
    Operation *newOp =
        createInterleaveLikeOp(op, newInputs, newInits, rewriter);
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter, false);
    return success();
  }
};

template <>
struct NormalizeToTargetType<bool, hfusion::ReduceWithIndexOp>
    : public OpRewritePattern<hfusion::ReduceWithIndexOp> {
public:
  using OpRewritePattern<hfusion::ReduceWithIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return failure();

    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits))
      return failure();

    auto newInputs =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inputs);
    auto newInits =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inits);
    Operation *newOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        op.getLoc(), TypeRange{newInits[0].getType(), newInits[1].getType()},
        newInputs, newInits, op.getReduceKindAttr(), op.getTieBreakLeftAttr(),
        op.getDimensionsAttr());
    replaceI1ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter);

    return success();
  }
};

template <>
struct NormalizeToTargetType<int8_t, hfusion::ReduceWithIndexOp>
    : public OpRewritePattern<hfusion::ReduceWithIndexOp> {
public:
  using OpRewritePattern<hfusion::ReduceWithIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    if (!hasI8ElemType(inputs) && !hasI8ElemType(inits)) {
      return failure();
    }

    auto newInputs =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inputs);
    auto newInits =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inits);
    Operation *newOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        op.getLoc(), TypeRange{newInits[0].getType(), newInits[1].getType()},
        newInputs, newInits, op.getReduceKindAttr(), op.getTieBreakLeftAttr(),
        op.getDimensionsAttr());
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter);
    return success();
  }
};

template <>
struct NormalizeToTargetType<int64_t, hfusion::ReduceWithIndexOp>
    : public OpRewritePattern<hfusion::ReduceWithIndexOp> {
public:
  using OpRewritePattern<hfusion::ReduceWithIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    if (((inputs.size() < 2) || !isI64ElemType(inputs[1].getType())) &&
        !isI64ElemType(inits[1].getType())) {
      return failure();
    }
    SmallVector<Value> newInputs;
    SmallVector<Value> newInits;
    newInputs.push_back(inputs[0]);
    if (inputs.size() > 1) {
      Value castIndexInput = castTo(rewriter, inputs[1], rewriter.getI32Type());
      newInputs.push_back(castIndexInput);
    }
    newInits.push_back(inits[0]);
    Value castIndexInit = castTo(rewriter, inits[1], rewriter.getI32Type());
    newInits.push_back(castIndexInit);
    Operation *newOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        op.getLoc(), TypeRange{newInits[0].getType(), newInits[1].getType()},
        newInputs, newInits, op.getReduceKindAttr(), op.getTieBreakLeftAttr(),
        op.getDimensionsAttr());
    Value oldValResult = op->getResult(0);
    Value newValResult = newOp->getResult(0);
    rewriter.replaceAllUsesWith(oldValResult, newValResult);
    Value oldIndexResult = op->getResult(1);
    Value newIndexResult = newOp->getResult(1);
    Value castIndexResult =
        castTo(rewriter, newIndexResult, rewriter.getI64Type());
    rewriter.replaceAllUsesWith(oldIndexResult, castIndexResult);
    return success();
  }
};

template <typename CumOpType>
struct NormalizeCumOpI8ToTargetType : public OpRewritePattern<CumOpType> {
public:
  using OpRewritePattern<CumOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CumOpType op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getODSOperands(0);
    SmallVector<Value> outputs = op.getODSResults(0);
    if (!hasI8ElemType(inputs) && !hasI8ElemType(outputs)) {
      return failure();
    }

    auto newInputs =
        normalizeSrcToTargetType<int8_t, Float32Type>(rewriter, inputs);
    auto newOutputs =
        normalizeSrcToTargetType<int8_t, Float32Type>(rewriter, outputs);
    Operation *newOp = rewriter.create<CumOpType>(
        op.getLoc(), TypeRange{newOutputs}, newInputs[0], op.getCumDims(), op.getReverse());
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter);
    return success();
  }
};

template <>
struct NormalizeToTargetType<int8_t, hfusion::GatherOp>
    : public OpRewritePattern<hfusion::GatherOp> {
public:
  using OpRewritePattern<hfusion::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::GatherOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> source = op.getODSOperands(0);
    SmallVector<Value> indices = op.getODSOperands(1);
    SmallVector<Value> inits = op.getODSOperands(2);
    if (!hasI8ElemType(source) && !hasI8ElemType(inits)) {
      return failure();
    }

    auto newSource =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, source);
    auto newInits =
        normalizeSrcToTargetType<int8_t, Float16Type>(rewriter, inits);
    Operation *newOp = rewriter.create<hfusion::GatherOp>(
        op.getLoc(), newSource[0], indices[0], newInits[0], op.getAxis());
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter, /*enableOverflow*/ false);

    return success();
  }
};

template <>
struct NormalizeToTargetType<int8_t, linalg::BroadcastOp>
    : public OpRewritePattern<linalg::BroadcastOp> {
public:
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    Value input = op.getInput();
    Value init = op.getInit();
    Location loc = op.getLoc();

    if (!isI8ElemType(input.getType()) && !isI8ElemType(init.getType())) {
      return failure();
    }

    Value newInput = hfusion::castTo(rewriter, input, rewriter.getF16Type(),
                                     hfusion::RoundMode::TRUNC);
    Value newInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, init, rewriter.getF16Type());
    Value newBrcOp = rewriter
                         .create<linalg::BroadcastOp>(loc, newInput, newInit,
                                                      op.getDimensionsAttr())
                         ->getResult(0);
    Value newResult = hfusion::castTo(rewriter, newBrcOp, rewriter.getI8Type(),
                                      hfusion::RoundMode::TRUNC, init,
                                      /* enableOverflow = */ false);

    rewriter.replaceAllUsesWith(op->getResult(0), newResult);
    rewriter.eraseOp(op);

    return success();
  }
};

/// normalize x xor y into (!(x&y)) & (x|y)
struct NormalizeXorOp : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != hfusion::BinaryFn::vxor) {
      return failure();
    }

    auto inputs = op.getDpsInputs();
    auto outs = op.getDpsInits();
    assert(!outs.empty() && isa<ShapedType>(outs[0].getType()));

    // x|y
    auto emptyVorOp = utils::createEmptyOp(rewriter, op->getLoc(), outs[0]);
    auto orOp =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, op->getLoc(), hfusion::BinaryFn::vor, inputs,
            ValueRange(emptyVorOp));
    // x&y
    auto emptyVandOp = utils::createEmptyOp(rewriter, op->getLoc(), outs[0]);
    auto vandOp =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, op->getLoc(), hfusion::BinaryFn::vand, inputs,
            ValueRange(emptyVandOp));

    // !(x&y)
    auto vnotOp =
        hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp, hfusion::UnaryFn,
                               hfusion::UnaryFnAttr>(
            rewriter, op->getLoc(), hfusion::UnaryFn::vnot,
            ValueRange{vandOp->getResults()}, ValueRange(vandOp->getResults()));

    // xorop
    auto emptyVxorOp = utils::createEmptyOp(rewriter, op->getLoc(), outs[0]);
    auto vxorOp =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, op->getLoc(), hfusion::BinaryFn::vand,
            ValueRange{vnotOp->getResults()[0], orOp->getResults()[0]},
            ValueRange(emptyVxorOp));
    rewriter.replaceOp(op, vxorOp);
    return success();
  }
};

class NormalizeMuli1i : public OpRewritePattern<linalg::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseBinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return llvm::failure();
    }

    if (op.getFun() != linalg::BinaryFn::mul) {
      return failure();
    }

    auto inputs = op.getDpsInputs();
    Value operand1 = inputs[0];
    Value operand2 = inputs[1];
    if (!isI1ElemType(operand1.getType()) ||
        !isI1ElemType(operand2.getType())) {
      return failure();
    }
    Location loc = op.getLoc();

    auto *andOp = createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                 hfusion::BinaryFnAttr>(
        rewriter, loc, hfusion::BinaryFn::vand, ValueRange{operand1, operand2},
        op.getDpsInits()[0]);

    rewriter.replaceAllUsesWith(op->getResults(), andOp->getResults());
    rewriter.eraseOp(op);

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
    auto signX = sign<hfusion::TaylerMode::ATAN>(rewriter, loc, clipedInput);
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

/// normalize VSUB(s, v) to VADD(s,VMULS(v, -1)).
struct NormalizeSubVSToVMulAndVAdd
    : public OpRewritePattern<linalg::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseBinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return failure();

    if (op.getFun() != linalg::BinaryFn::sub)
      return failure();

    if (!isSVOp(op))
      return failure();

    auto inputs = op.getDpsInputs();
    Value vec = inputs[1];
    Type scalarType = inputs[0].getType();
    Location loc = op.getLoc();

    auto negOne = utils::createConstantOp<float>(rewriter, loc, scalarType, -1);
    Value empty = utils::createEmptyOp(rewriter, loc, vec);
    auto *mulOp = createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                 linalg::BinaryFnAttr>(
        rewriter, loc, linalg::BinaryFn::mul, ValueRange{vec, negOne}, empty);

    auto *addOp = createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                 linalg::BinaryFnAttr>(
        rewriter, loc, linalg::BinaryFn::add,
        ValueRange{inputs[0], mulOp->getResult(0)}, op.getDpsInits()[0]);

    rewriter.replaceAllUsesWith(op->getResults(), addOp->getResults());
    rewriter.eraseOp(op);
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

/// normalize shift i8 as bellow
/// eg.
///   %res = shift %src : i8
/// is normalized to
///   %tmp0 = cast %src i8 to i16
///   %tmp1 = shift %tmp0 : i16
///   %res = cast %tmp1 i16 to i8
struct NormalizeShiftI8ToI16
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    auto fun = op.getFun();
    if (!(fun == hfusion::BinaryFn::shli || fun == hfusion::BinaryFn::shrsi ||
          fun == hfusion::BinaryFn::shrui)) {
      return failure();
    }

    Value input = op.getDpsInputs()[0];
    Type inputElemType = getElementTypeOrSelf(input.getType());
    if (!inputElemType.isInteger(8)) {
      return failure();
    }

    auto loc = op->getLoc();
    auto targetElemType = rewriter.getI16Type();
    auto shift = op.getDpsInputs()[1];
    hfusion::TypeFn cast_integer_type = (fun == hfusion::BinaryFn::shrui)
                                  ? hfusion::TypeFn::cast_unsigned
                                  : hfusion::TypeFn::cast_signed;
    Value inputOfI16 =
        hfusion::castTo(rewriter, input, targetElemType, cast_integer_type);
    Value shiftOfI16 =
        hfusion::castTo(rewriter, shift, targetElemType, cast_integer_type);

    auto shiftInit = utils::createEmptyOp(rewriter, loc, inputOfI16);
    Value resOfI16 =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, fun, ValueRange{inputOfI16, shiftOfI16},
            ValueRange(shiftInit))
            ->getResults()[0];

    auto srcElemType = rewriter.getI8Type();
    auto selectMode =
        utils::selectRoundMode<hfusion::RoundMode>(targetElemType, srcElemType);
    auto roundMode = (fun == hfusion::BinaryFn::shli)
                         ? hfusion::RoundMode::TRUNCWITHOVERFLOW
                         : selectMode;
    auto resOfI8 = hfusion::castTo(rewriter, resOfI16, srcElemType, roundMode,
                                   std::nullopt, true, cast_integer_type);

    rewriter.replaceOp(op, resOfI8);
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

/// normalize ceildivsi or floordivsi i8/i16/i32/i64 as bellow
/// eg.
///   %res = ceildivsi/floordivsi %lhs, %rhs : i8
/// is normalized to
///   %lhsF32 = cast %src i8 to f32
///   %rhsF32 = cast %rhs i8 to f32
///   %divF32 = div %lhsF32, %rhsF32 : f32
///   %castF32 = ceilop/floorop %divF32
///   %res = cast %castF32 f32 to i8
struct NormalizeCDivandFloorDivIntOp
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    auto fun = op.getFun();
    if (!(fun == hfusion::BinaryFn::ceildivsi ||
          fun == hfusion::BinaryFn::ceildivui ||
          fun == hfusion::BinaryFn::floordivsi)) {
      return failure();
    }

    auto resTensor = op.getResultTensors()[0];
    auto resTy = dyn_cast<TensorType>(resTensor.getType());
    auto elemTySrc = getElementTypeOrSelf(resTy);
    if (!elemTySrc.isInteger()) {
      return failure();
    }

    // step1. res = divWithRoundMode(x, y, FLOOR/CEIL)
    rewriter.setInsertionPoint(op);
    auto inputs = op.getDpsInputs();

    auto loc = op->getLoc();
    // TODO: fix to use uint type after support uint op
    hfusion::RoundMode roundMode = (fun == hfusion::BinaryFn::ceildivsi ||
                                    fun == hfusion::BinaryFn::ceildivui)
                                       ? hfusion::RoundMode::CEIL
                                       : hfusion::RoundMode::FLOOR;
    auto res = hfusion::divWithRoundMode(rewriter, loc, elemTySrc, inputs[0],
                                         inputs[1], resTensor, roundMode);
    rewriter.replaceOp(op, res);
    return success();
  }
};

static void replaceF16ResultsWithF32(const SmallVector<Value> &oldResults,
                                     const SmallVector<Value> &newResults,
                                     PatternRewriter &rewriter) {
  assert(oldResults.size() == newResults.size() &&
         "result sizes mismatch when replace op results");
  for (const auto [idx, oldResult] : llvm::enumerate(oldResults)) {
    Value newResult = newResults[idx];
    if (!isF16ElemType(oldResult.getType())) {
      rewriter.replaceAllUsesWith(oldResult, newResult);
      continue;
    }

    Value castResult = castTo(rewriter, newResult, rewriter.getF16Type());
    rewriter.replaceAllUsesWith(oldResult, castResult);
  }
}

/// normalize f16 reduce_sum as bellow for high precision
/// eg.
///    reduce_sum f16
/// is normalized to
///    cast f16 to f32
///    reduce_sum f32
///    cast f32 to f16
struct NormalizeF16ReduceSum : public OpRewritePattern<linalg::ReduceOp> {
public:
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();

    if (!hasF16ElemType(inputs) && !hasF16ElemType(inits)) {
      return failure();
    }

    if (!shouldComputeF16ToF32(op)) {
      return failure();
    }

    SmallVector<Value> newInputs =
        normalizeSrcToTargetType<float, Float32Type>(rewriter, inputs);
    SmallVector<Value> newInits =
        normalizeSrcToTargetType<float, Float32Type>(rewriter, inits);
    Operation *newOp =
        createNewReduceOp(op, rewriter, rewriter.getF16Type(),
                          rewriter.getF32Type(), newInputs, newInits);
    replaceF16ResultsWithF32(op->getResults(), newOp->getResults(), rewriter);

    return success();
  }

private:
  bool shouldComputeF16ToF32(linalg::ReduceOp op) const {
    Block *block = &op.getRegion().front();
    for (Operation &bodyOp : *block) {
      if (dyn_cast_or_null<arith::AddFOp>(bodyOp)) {
        return true;
      }
    }
    return false;
  }
};

// ===----------------------------------------------------------------------===//
// VReduceOp RA [b, r, a]-> transpose [b, a, r] + AR reduce [b, a]
// ===----------------------------------------------------------------------===//

/// Normalize reduceRa_with_index to transpose + reduceAR_with_index +
/// reshape so its performance will be better in some cases
///
/// e.g.
/// %reduced:2 = hfusion.reduce_with_index
///               ins(%0, %1 : tensor<64x32xf32>, tensor<64x32xi32>)
///               outs(%25, %26 : tensor<32xf32>, tensor<32xi32>)
///               dimensions = [0]
///
/// will be normalized to
///
/// %empty_0 = tensor.empty() : tensor<32x64xf32>
/// %transposed_0 = linalg.transpose ins(%0 : tensor<64x32xf32>)
///                   outs(%empty_0 : tensor<32x64xf32>)
///                   permutation = [1, 0]
/// %empty_1 = tensor.empty() : tensor<32x64xi32>
/// %transposed_1 = linalg.transpose ins(%0 : tensor<64x32xi32>)
///                   outs(%empty_1 : tensor<32x64xi32>) permutation = [1,
///                   0]
/// %reduced:2 = hfusion.reduce_with_index
///     ins(%transposed_0, %transposed_1 : tensor<32x64xf32>,
///     tensor<32x64xi32>) outs(%25, %26 : tensor<32xf32>, tensor<32xi32>)
///     dimensions = [1]

struct ReduceWithIndexRAHighPerformance
    : public OpRewritePattern<hfusion::ReduceWithIndexOp> {
  using OpRewritePattern<hfusion::ReduceWithIndexOp>::OpRewritePattern;

  static Value getTransposedValue(Value source, const Location loc,
                                  PatternRewriter &rewriter,
                                  llvm::ArrayRef<int> order) {
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto sourceRank = sourceType.getRank();

    SmallVector<int64_t> perm(order);
    SmallVector<int64_t> originalShape(sourceType.getShape());
    SmallVector<int64_t> transposedShape(sourceRank);
    for (int64_t i = 0; i < sourceRank; i++) {
      transposedShape[i] = originalShape[perm[i]];
    }

    Value transposeInit = rewriter.create<tensor::EmptyOp>(
        loc, transposedShape, sourceType.getElementType());

    Value transpose =
        rewriter.create<linalg::TransposeOp>(loc, source, transposeInit, perm)
            .getResults()[0];

    return transpose;
  }

  // limitation of memref'shape from hivm::transposeOp
  // if we have a tensor like [b, r, a]
  // if eleType is float16
  // The strides of both r, a need to be divisible by 16.
  // if eleType is float32
  // The stride of a or r needs to be divisible by 16,
  // and the other's needs to be divisible by 8.
  // reducedim must be a single one
  static bool
  isSizeCompatibleForTransposeForReduceOp(PatternRewriter &rewriter, Value src,
                                          SmallVector<int64_t> srcShape,
                                          int reduceDim) {
    auto floatEleType =
        dyn_cast<FloatType>(getElementTypeOrSelf(src.getType()));
    // at this level
    // reduce int have been transformed into reduce float for now
    if (!floatEleType) {
      return false;
    }
    const unsigned num_per_block =
        utils::INTR_BYTES_PER_BLOCK /
        (floatEleType.getWidth() / utils::INTR_BITS_PER_BYTE);

    // get total A axis size
    int totalRShape = srcShape[reduceDim];
    int totalAShape = 1;
    for (size_t i = static_cast<size_t>(reduceDim) + 1lu; i < srcShape.size();
         i++) {
      totalAShape *= srcShape[i];
    }

    // refer to the num of the registers
    // used in transpose operation
    const int registerCount = 16;

    if ((totalRShape % num_per_block == 0 &&
         totalAShape % registerCount == 0) ||
        (totalAShape % num_per_block == 0 && totalRShape % registerCount == 0))
      return true;

    return false;
  }

  Value reshapeOpRewriterHelper(Value input, ArrayRef<int64_t> reshape,
                                PatternRewriter &rewriter, Location loc) const {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    // Prepare reshaped tensor type
    auto reshapeType =
        RankedTensorType::get(reshape, inputType.getElementType());
    // Prepare reshape info value
    auto reshapeInfo = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64TensorAttr(reshape));
    return rewriter.create<tensor::ReshapeOp>(loc, reshapeType, input,
                                              reshapeInfo);
  }

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    // reduceOp only handles tensors
    auto loc = op.getLoc();
    auto src = op.getInputs()[0];
    ShapedType srcShapeType = cast<ShapedType>(src.getType());
    ArrayRef<int64_t> srcShape = srcShapeType.getShape();

    auto srcShapeRank = srcShapeType.getRank();

    // only support one axis reduce
    // only handle transpose of ra
    auto reduceDims = op.getDimensions();
    auto reduceDim = reduceDims[0];
    if (reduceDims.size() > 1 || reduceDim == srcShapeRank - 1) {
      return failure();
    }

    SmallVector<Value> newInputs;
    newInputs.insert(newInputs.end(), op.getInputs().begin(),
                     op.getInputs().end());

    if (!isSizeCompatibleForTransposeForReduceOp(
            rewriter, src, SmallVector<int64_t>{srcShape}, reduceDim)) {
      return failure();
    }

    // knowing that we are processing with reduce ra with index
    // then we transpose the tensor
    // create transposeOp
    SmallVector<int32_t> transposePerm;
    for (int i = 0; i < srcShapeRank; i++) {
      if (i != reduceDim)
        transposePerm.push_back(i);
    }
    transposePerm.push_back(reduceDim);

    // create mapper to map the inputs to the new reduce op
    IRMapping mapper;
    for (const auto &[idx, operand] : llvm::enumerate(op.getInputs())) {
      newInputs[idx] = getTransposedValue(newInputs[idx], loc, rewriter,
                                          ArrayRef<int32_t>(transposePerm));
      mapper.map(operand, newInputs[idx]);
    }

    // clone & replace the reduceOp
    SmallVector<int64_t> newReduceDim{srcShapeRank - 1};
    auto newReduceOp = rewriter.clone(*op, mapper);
    dyn_cast<hfusion::ReduceWithIndexOp>(newReduceOp)
        .setDimensions(ArrayRef<int64_t>(newReduceDim));

    rewriter.replaceOp(op, newReduceOp);
    return success();
  }
};

/// normalize mulext(x, y) as bellow
/// inputs: N-bit number x, y
/// step1: perform extension to generate 2N-bit operands from x and y
/// step2: multiply 2N-bit x and y to get mul_res
/// step3: get the high half of the operand by N-bit-right-shifting mul_res
/// step4: get the low half of the operand by N-bit-left-shifting
/// and later N-bit-right-shifting mul_res
/// step5: cast result back to origin type
/// outputs: the N-bit low and the N-bit high halves of the product.
class NormalizeMulExtOp : public OpRewritePattern<hfusion::MulExtOp> {
public:
  using OpRewritePattern<hfusion::MulExtOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::MulExtOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto lhsType = getElementTypeOrSelf(lhs.getType());
    auto rhsType = getElementTypeOrSelf(rhs.getType());
    if (!lhsType.isInteger(8) || !rhsType.isInteger(8)) {
      return failure();
    }

    // step1: perform extension.
    Value lhsI16 = hfusion::castTo(rewriter, lhs, rewriter.getI16Type());
    Value rhsI16 = hfusion::castTo(rewriter, rhs, rewriter.getI16Type());

    // step2: multiply
    auto loc = op.getLoc();
    auto mulInit = utils::createEmptyOp(rewriter, loc, lhsI16);
    auto mulRes =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::mul, ValueRange({lhsI16, rhsI16}),
            ValueRange(mulInit))
            ->getResult(0);

    // step3: get the high half of the operand
    auto bitWidth = lhsType.getIntOrFloatBitWidth();
    arith::ConstantOp shiftValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI16Type(),
        rewriter.getIntegerAttr(rewriter.getI16Type(), bitWidth));
    auto shrHighBitInit = utils::createEmptyOp(rewriter, loc, lhsI16);
    auto shrHighBit =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, hfusion::BinaryFn::shrsi,
            ValueRange{mulRes, shiftValue}, ValueRange(shrHighBitInit))
            ->getResult(0);

    // step4: get the low half of the operand
    auto shlInit = utils::createEmptyOp(rewriter, loc, lhsI16);
    auto shlRes =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, hfusion::BinaryFn::shli,
            ValueRange{mulRes, shiftValue}, ValueRange(shlInit))
            ->getResult(0);
    auto shrLowBitInit = utils::createEmptyOp(rewriter, loc, lhsI16);
    auto shrLowBit =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, loc, hfusion::BinaryFn::shrsi,
            ValueRange{shlRes, shiftValue}, ValueRange(shrLowBitInit))
            ->getResult(0);

    // step5: cast result back to origin type i8
    auto roundMode = hfusion::RoundMode::TRUNCWITHOVERFLOW;
    auto highBitI8 =
        hfusion::castTo(rewriter, shrHighBit, rewriter.getI8Type(), roundMode);
    auto lowBitI8 =
        hfusion::castTo(rewriter, shrLowBit, rewriter.getI8Type(), roundMode);
    rewriter.replaceOp(op, {lowBitI8, highBitI8});
    return success();
  }
};

/// Normalize Powi from I8/I16 to Powf F32
/// Compute with F32, then cast back to I8/I16
/// For example:
/// result = hfusion.powi(i8 x, i8y)
/// is legalized to
/// x_1 = cast x from i8 to f32
/// y_1 = cast y from i8 to f32
/// z_1 = hfusion.powf(f32 x_1, f32 y_1)
/// result = cast z_1 from f32 to i8
struct NormalizeVPowiToPowf
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getFun() != hfusion::BinaryFn::powi) {
      return rewriter.notifyMatchFailure(op, "Doesn't match powi");
    }

    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> outputs = op.getOutputs();
    SmallVector<Value> newInputs;
    SmallVector<Value> newOutputs;
    if (allI8ElemType(inputs) && allI8ElemType(outputs)) {
      newInputs =
          normalizeSrcToTargetType<int8_t, Float32Type>(rewriter, inputs);
      newOutputs =
          normalizeSrcToTargetType<int8_t, Float32Type>(rewriter, outputs);
    } else if (allI16ElemType(inputs) && allI16ElemType(outputs)) {
      newInputs =
          normalizeSrcToTargetType<int16_t, Float32Type>(rewriter, inputs);
      newOutputs =
          normalizeSrcToTargetType<int16_t, Float32Type>(rewriter, outputs);
    } else {
      return rewriter.notifyMatchFailure(op, "powi type is not i8 nor i16");
    }
    Operation *newOp =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(rewriter, op->getLoc(),
                                                       hfusion::BinaryFn::powf,
                                                       newInputs, newOutputs);
    if (allI8ElemType(outputs)) {
      replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                     rewriter);
    } else if (allI16ElemType(outputs)) {
      replaceI16ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                      rewriter);
    }
    return success();
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
struct NormalizeMinMaxNumFOp
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp> {
public:
  using OpRewritePattern<hfusion::ElemwiseBinaryOp>::OpRewritePattern;
  virtual ~NormalizeMinMaxNumFOp() = default;
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
 
    // step 1: new_src0 = hfusion.select(hfusion.isnan(src0), -+inf, src0)
    auto newSrc0 = createNewSrcForMinMaxNumFOp(rewriter, op->getLoc(), src0,
                                               paddingInfValue);
    // step 2: new_src0 = hfusion.select(hfusion.isnan(src0), -+inf, src1)
    auto newSrc1 = createNewSrcForMinMaxNumFOp(rewriter, op->getLoc(), src1,
                                               paddingInfValue);
 
    // step 3: dst = hfusion.elemwise_binary {maxf | minf} (new_src0, new_src1)
    auto minMaxFOpOut = utils::createEmptyOp(rewriter, op->getLoc(), src0);
    auto minMaxFOp = createBinaryOp<ElemwiseBinaryOp, BinaryFn, BinaryFnAttr>(
        rewriter, op->getLoc(), hfusionFn, ValueRange({newSrc0, newSrc1}),
        ValueRange(minMaxFOpOut));
 
    return minMaxFOp->getResult(0);
  }
 
  /// Returns a new operand for BinaryFn::maxf (BinaryFn::minf)
  /// that is used when normalizing maxnumf (minnumf) to maxf (minf).
  /// Check comment before struct definition
  Value createNewSrcForMinMaxNumFOp(PatternRewriter &rewriter, Location loc,
                                    Value src, double paddingInfValue) const {
    auto elementType = getElementTypeOrSelf(src.getType());
    auto constInfinity = utils::createConstantOp<double>(
        rewriter, loc, elementType, paddingInfValue);
 
    // src_nan_mask = hfusion.isnan(src0)
    auto isNanResultTensorType =
        utils::getTensorTypeWithSameShape(src.getType(), rewriter.getI1Type());
    auto isNanOp = rewriter.create<IsNanOp>(loc, isNanResultTensorType, src);
 
    // new_src = hfusion.select(src0_nan_mask, inf, src0)
    auto selectOpOut = utils::createEmptyOp(rewriter, loc, src);
    auto selectOp = rewriter.create<SelectOp>(
        loc, TypeRange(selectOpOut),
        ValueRange({isNanOp->getResult(0), constInfinity, src}),
        ValueRange(selectOpOut));
 
    return selectOp->getResult(0);
  }
};

template <>
struct NormalizeToTargetType<bool, hfusion::InterleaveOp>
    : public OpRewritePattern<hfusion::InterleaveOp> {
public:
  using OpRewritePattern<hfusion::InterleaveOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::InterleaveOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInput();
    SmallVector<Value> inits = op.getODSResults(0);
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits)) {
      return failure();
    }

    auto newInputs =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inputs);
    auto newInits =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inits);
    Operation *newOp =
        createInterleaveLikeOp(op, newInputs, newInits, rewriter);
    replaceI1ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter, false);

    return success();
  }
};

template <>
struct NormalizeToTargetType<bool, linalg::BroadcastOp>
    : public OpRewritePattern<linalg::BroadcastOp> {
public:
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    Value input = op.getInput();
    Value init = op.getInit();
    Location loc = op.getLoc();

    if (!isI1ElemType(input.getType()) && !isI1ElemType(init.getType())) {
      return failure();
    }

    Value newInput = hfusion::castTo(rewriter, input, rewriter.getF16Type(),
                                     hfusion::RoundMode::TRUNC);
    Value newInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, init, rewriter.getF16Type());
    Value newBrcOp = rewriter
                         .create<linalg::BroadcastOp>(loc, newInput, newInit,
                                                      op.getDimensionsAttr())
                         ->getResult(0);
    Value newResult = hfusion::castTo(rewriter, newBrcOp, rewriter.getI1Type(),
                                      hfusion::RoundMode::TRUNC, init,
                                      /* enableOverflow = */ false);

    rewriter.replaceAllUsesWith(op->getResult(0), newResult);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace mlir::hfusion

// Normalize scalar like tensor for linalg and hfusion ops.
void populateNormalizeScalarLikeHFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeScalarLikeTensorOp<hfusion::ElemwiseUnaryOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<hfusion::ElemwiseBinaryOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<hfusion::CompareOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<hfusion::SelectOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<hfusion::CastOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<linalg::ElemwiseUnaryOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<linalg::ElemwiseBinaryOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorLinalgBrcOp>(patterns.getContext());
}

void populateNormalizeI1ToTargetPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeToTargetType<bool, hfusion::InterleaveOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, linalg::BroadcastOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, linalg::ReduceOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, CompareOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, SelectOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, linalg::TransposeOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, tensor::ConcatOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, hfusion::ReduceWithIndexOp>>(ctx);
  patterns.add<NormalizeMuli1i>(ctx);
}

void populateNormalizeI8ToTargetPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeToTargetType<int8_t, hfusion::ElemwiseBinaryOp>>(ctx);
  patterns.add<NormalizeToTargetType<int8_t, hfusion::ElemwiseUnaryOp>>(ctx);
  patterns.add<NormalizeToTargetType<int8_t, linalg::ElemwiseBinaryOp>>(ctx);
  patterns.add<NormalizeToTargetType<int8_t, linalg::ElemwiseUnaryOp>>(ctx);
  patterns.add<NormalizeToTargetType<int8_t, hfusion::SelectOp>>(ctx);
  patterns.add<NormalizeToTargetType<int8_t, linalg::ReduceOp>>(ctx);
  patterns.add<NormalizeToTargetType<int8_t, hfusion::InterleaveOp>>(ctx);
  patterns.add<NormalizeToTargetType<int8_t, hfusion::DeinterleaveOp>>(ctx);
  patterns.add<NormalizeToTargetType<int8_t, hfusion::ReduceWithIndexOp>>(ctx);
  patterns.add<NormalizeToTargetType<int8_t, hfusion::GatherOp>>(ctx);
  patterns.add<NormalizeToTargetType<int8_t, linalg::BroadcastOp>>(ctx);
  patterns.add<NormalizeCumOpI8ToTargetType<hfusion::CumsumOp>>(ctx);
  patterns.add<NormalizeCumOpI8ToTargetType<hfusion::CumprodOp>>(ctx);
}

void populateNormalizeF16ToF32Patterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeF16ToF32Type<linalg::ElemwiseUnaryOp>>(ctx);
  patterns.add<NormalizeF16ToF32Type<hfusion::ElemwiseBinaryOp>>(ctx);
  patterns.add<NormalizeF16ToF32Type<hfusion::ElemwiseUnaryOp>>(ctx);
  patterns.add<NormalizeCumOpF16ToF32Type<hfusion::CumsumOp>>(ctx);
  patterns.add<NormalizeCumOpF16ToF32Type<hfusion::CumprodOp>>(ctx);
}

void populateNormalizeHFusionPatterns(RewritePatternSet &patterns) {
  populateNormalizeF16ToF32Patterns(patterns);
  patterns.add<NormalizeSinOp>(patterns.getContext());
  patterns.add<NormalizeCosOp>(patterns.getContext());
  patterns.add<NormalizeAtanOp>(patterns.getContext());
  patterns.add<NormalizeTanOp>(patterns.getContext());
  patterns.add<NormalizeTanhOp>(patterns.getContext());
  patterns.add<NormalizeI8I32CmpOp>(patterns.getContext());
  patterns.add<NormalizeMulRec>(patterns.getContext());
  patterns.add<NormalizeModOp>(patterns.getContext());
  patterns.add<NormalizeCmpToCastOp>(patterns.getContext());
  patterns.add<NormalizeNegToMul>(patterns.getContext());
  patterns.add<NormalizeDivVSToRec>(patterns.getContext());
  patterns.add<NormalizeVPowiToPowf>(patterns.getContext());
  patterns.add<NormalizeSubVSToVMulAndVAdd>(patterns.getContext());
  patterns.add<NormalizeRSqrtOp>(patterns.getContext());
  patterns.add<NormalizeCeilandFloorOp>(patterns.getContext());
  patterns.add<NormalizeLogLikeOp>(patterns.getContext());
  patterns.add<NormalizeLog1pOp>(patterns.getContext());
  patterns.add<NormalizeExp2Op>(patterns.getContext());
  patterns.add<NormalizeExpM1Op>(patterns.getContext());
  patterns.add<NormalizeErfOp>(patterns.getContext());
  patterns.add<NormalizeBrcCast>(patterns.getContext());
  patterns.add<NormalizefillCastToTensorBrc>(patterns.getContext());
  patterns.add<NormalizeAnyToF32UnaryRecOp>(patterns.getContext());
  patterns.add<NormalizeCastLoweringOp>(patterns.getContext());
  patterns.add<NormalizeIsInfOp>(patterns.getContext());
  patterns.add<NormalizeIsNanOp>(patterns.getContext());
  patterns.add<NormalizeXorOp>(patterns.getContext());
  patterns.add<NormalizeShiftI8ToI16>(patterns.getContext());
  patterns.add<NormalizeIlogbOp>(patterns.getContext());
  patterns.add<NormalizeLdexpOp>(patterns.getContext());
  patterns.add<NormalizePowfOp>(patterns.getContext());
  patterns.add<NormalizeMinMaxNumFOp<BinaryFn::maxnumf>>(patterns.getContext());
  patterns.add<NormalizeMinMaxNumFOp<BinaryFn::minnumf>>(patterns.getContext());
  patterns.add<NormalizeF16ReduceSum>(patterns.getContext());
  patterns.add<ReduceWithIndexRAHighPerformance>(patterns.getContext());
  patterns.add<NormalizetruncfExtf>(patterns.getContext());
  populateNormalizeScalarLikeHFusionPatterns(patterns);
  populateNormalizeI1ToTargetPatterns(patterns);
  populateNormalizeI8ToTargetPatterns(patterns);
  patterns.add<NormalizeCDivandFloorDivIntOp>(patterns.getContext());
  patterns.add<NormalizeMulExtOp>(patterns.getContext());
  patterns.add<NormalizeDivSIandDivUIOp>(patterns.getContext());
  patterns.add<NormalizeCmpVne>(patterns.getContext());
  patterns.add<NormalizeToTargetType<int64_t, hfusion::ReduceWithIndexOp>>(
      patterns.getContext());
}

namespace {
struct NormalizeHFusionPass : public impl::NormalizeBase<NormalizeHFusionPass> {
public:
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    populateNormalizeHFusionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hfusion::createHFusionNormalizeOpsPass() {
  return std::make_unique<NormalizeHFusionPass>();
}
