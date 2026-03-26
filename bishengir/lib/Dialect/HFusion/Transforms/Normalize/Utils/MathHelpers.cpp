//===- MathHelpers.cpp --------------------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"

namespace mlir::hfusion {

// norm(x,x_round,offset) = x-x_round*(pi1+pi2+pi3+pi4+pi5)+offset
// (pi1+pi2+pi3+pi4+pi5) approximates pi
Value norm(PatternRewriter &rewriter, Location loc, Value x,
                  Value xRound, const llvm::SmallVector<double> &piApproParams,
                  std::optional<float> offset) {
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

template <hfusion::TaylerMode taylerMode>
Value sign(PatternRewriter &rewriter, Location loc, Value x) {
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

// tayler x =
// taylerParams[0]*x+taylerParams[1]*x^3+...+taylerParams[i]*x^(2*i+1)
template <hfusion::TaylerMode taylerMode>
Value tayler(OpBuilder &b, Location loc, Value x,
                    int taylerExpansionNum) {
  SmallVector<double> taylerParams =
      getTaylerParams(taylerMode, taylerExpansionNum);

  auto emptyOp = utils::createEmptyOp(b, loc, x);
  auto xPow =
      hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                              linalg::BinaryFnAttr>(
          b, loc, linalg::BinaryFn::mul, ValueRange{x, x}, ValueRange(emptyOp))
          ->getResult(0);

  // Step 1: init the last taylerTerm
  auto elementType = getElementTypeOrSelf(x.getType());
  auto lastTaylerParam = b.create<arith::ConstantOp>(
      loc, elementType,
      b.getFloatAttr(elementType, taylerParams[taylerExpansionNum - 1]));
  auto lastTaylerTerm =
      hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                              linalg::BinaryFnAttr>(
          b, loc, linalg::BinaryFn::mul, ValueRange{xPow, lastTaylerParam},
          ValueRange(emptyOp))
          ->getResult(0);

  // Step 2: construct the tayler series
  // for i in [0,n-i-2):
  //    partialRes = (partialRes+TaylerParams[n-i-2])*(x^2)
  Value partialRes = constructTaylerSeries(
      b, loc, lastTaylerTerm, emptyOp, xPow, taylerExpansionNum, taylerParams);

  // partialRes1 = (partialRes+1)
  auto constOne = b.create<arith::ConstantOp>(loc, elementType,
                                              b.getFloatAttr(elementType, 1.0));
  auto partialRes1 =
      hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                              linalg::BinaryFnAttr>(
          b, loc, linalg::BinaryFn::add, ValueRange{partialRes, constOne},
          ValueRange(emptyOp))
          ->getResult(0);
  // Step 3: multiple common coef
  // tayler(x) = partialRes1*x
  auto res = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                     linalg::BinaryFnAttr>(
                 b, loc, linalg::BinaryFn::mul, ValueRange{partialRes1, x},
                 ValueRange(emptyOp))
                 ->getResult(0);
  return res;
}

SmallVector<double> getTaylerParams(hfusion::TaylerMode taylerMode,
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

Value getSinSign(PatternRewriter &rewriter, Location loc, Value x) {
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

double getFPMAX(FloatType fType) {
  if (fType.isF32()) {
    // TODO: make confirmation why TBE process it specially
    return (double)std::pow(2, fType.getWidth() + 30);
  }

  return (double)std::pow(2, fType.getWidth() - 1);
}

double getFPMIN(FloatType fType) {
  if (fType.isF32()) {
    // TODO: make confirmation why TBE process it specially
    return (double)std::pow(2, -((int)fType.getWidth() + 30));
  }

  return (double)std::pow(2, -((int)fType.getWidth() - 1));
}

Value getAtanSign(PatternRewriter &rewriter, Location loc, Value x) {
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

Value constructTaylerSeries(OpBuilder &b, Location loc, Value lastTaylerTerm,
                            Value emptyOp, Value xPow, int taylerExpansionNum,
                            const SmallVector<double> &taylerParams) {
  Value partialRes = lastTaylerTerm;
  auto elementType = getElementTypeOrSelf(xPow.getType());
  for (int i = 0; i < taylerExpansionNum - 2; i++) {
    auto curTaylerParam = b.create<arith::ConstantOp>(
        loc, elementType,
        b.getFloatAttr(elementType, taylerParams[taylerExpansionNum - i - 2]));
    auto curTayerTerm =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            b, loc, linalg::BinaryFn::add,
            ValueRange{partialRes, curTaylerParam}, ValueRange(emptyOp))
            ->getResult(0);
    auto curRes =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            b, loc, linalg::BinaryFn::mul, ValueRange{curTayerTerm, xPow},
            ValueRange(emptyOp))
            ->getResult(0);
    partialRes = curRes;
  }
  return partialRes;
}

// Lookup table for the Payne-Hanek reduction in sin/cos computation.
// Entry (i - 128) stores a 32-bit word representing bits i through i+31
// of the binary fractional expansion of 2/pi.
// The first 128 zero words are intentional padding to allow uniform
// handling of inputs with different magnitudes.
inline static constexpr std::array<uint32_t, 320> tbl = {
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x0,        0x0,        0x0,
    0x0,        0x0,        0x0,        0x1,        0x2,        0x5,
    0xa,        0x14,       0x28,       0x51,       0xa2,       0x145,
    0x28b,      0x517,      0xa2f,      0x145f,     0x28be,     0x517c,
    0xa2f9,     0x145f3,    0x28be6,    0x517cc,    0xa2f98,    0x145f30,
    0x28be60,   0x517cc1,   0xa2f983,   0x145f306,  0x28be60d,  0x517cc1b,
    0xa2f9836,  0x145f306d, 0x28be60db, 0x517cc1b7, 0xa2f9836e, 0x45f306dc,
    0x8be60db9, 0x17cc1b72, 0x2f9836e4, 0x5f306dc9, 0xbe60db93, 0x7cc1b727,
    0xf9836e4e, 0xf306dc9c, 0xe60db939, 0xcc1b7272, 0x9836e4e4, 0x306dc9c8,
    0x60db9391, 0xc1b72722, 0x836e4e44, 0x6dc9c88,  0xdb93910,  0x1b727220,
    0x36e4e441, 0x6dc9c882, 0xdb939105, 0xb727220a, 0x6e4e4415, 0xdc9c882a,
    0xb9391054, 0x727220a9, 0xe4e44152, 0xc9c882a5, 0x9391054a, 0x27220a94,
    0x4e441529, 0x9c882a53, 0x391054a7, 0x7220a94f, 0xe441529f, 0xc882a53f,
    0x91054a7f, 0x220a94fe, 0x441529fc, 0x882a53f8, 0x1054a7f0, 0x20a94fe1,
    0x41529fc2, 0x82a53f84, 0x54a7f09,  0xa94fe13,  0x1529fc27, 0x2a53f84e,
    0x54a7f09d, 0xa94fe13a, 0x529fc275, 0xa53f84ea, 0x4a7f09d5, 0x94fe13ab,
    0x29fc2757, 0x53f84eaf, 0xa7f09d5f, 0x4fe13abe, 0x9fc2757d, 0x3f84eafa,
    0x7f09d5f4, 0xfe13abe8, 0xfc2757d1, 0xf84eafa3, 0xf09d5f47, 0xe13abe8f,
    0xc2757d1f, 0x84eafa3e, 0x9d5f47d,  0x13abe8fa, 0x2757d1f5, 0x4eafa3ea,
    0x9d5f47d4, 0x3abe8fa9, 0x757d1f53, 0xeafa3ea6, 0xd5f47d4d, 0xabe8fa9a,
    0x57d1f534, 0xafa3ea69, 0x5f47d4d3, 0xbe8fa9a6, 0x7d1f534d, 0xfa3ea69b,
    0xf47d4d37, 0xe8fa9a6e, 0xd1f534dd, 0xa3ea69bb, 0x47d4d377, 0x8fa9a6ee,
    0x1f534ddc, 0x3ea69bb8, 0x7d4d3770, 0xfa9a6ee0, 0xf534ddc0, 0xea69bb81,
    0xd4d37703, 0xa9a6ee06, 0x534ddc0d, 0xa69bb81b, 0x4d377036, 0x9a6ee06d,
    0x34ddc0db, 0x69bb81b6, 0xd377036d, 0xa6ee06db, 0x4ddc0db6, 0x9bb81b6c,
    0x377036d8, 0x6ee06db1, 0xddc0db62, 0xbb81b6c5, 0x77036d8a, 0xee06db14,
    0xdc0db629, 0xb81b6c52, 0x7036d8a5, 0xe06db14a, 0xc0db6295, 0x81b6c52b,
    0x36d8a56,  0x6db14ac,  0xdb62959,  0x1b6c52b3, 0x36d8a566, 0x6db14acc,
    0xdb629599, 0xb6c52b32, 0x6d8a5664, 0xdb14acc9, 0xb6295993, 0x6c52b327,
    0xd8a5664f, 0xb14acc9e, 0x6295993c, 0xc52b3278, 0x8a5664f1, 0x14acc9e2,
    0x295993c4, 0x52b32788, 0xa5664f10, 0x4acc9e21, 0x95993c43, 0x2b327887,
    0x5664f10e, 0xacc9e21c, 0x5993c439, 0xb3278872, 0x664f10e4, 0xcc9e21c8,
    0x993c4390, 0x32788720, 0x64f10e41, 0xc9e21c82, 0x93c43904, 0x27887208,
    0x4f10e410, 0x9e21c820};

static Value cI32(OpBuilder &b, Location loc, int32_t v) {
  auto i32 = b.getI32Type();
  auto attr = b.getI32IntegerAttr(v);
  return b.create<arith::ConstantOp>(loc, i32, attr);
}

static Value cF32(OpBuilder &b, Location loc, float v) {
  auto f32 = b.getF32Type();
  auto attr = b.getF32FloatAttr(v);
  return b.create<arith::ConstantOp>(loc, f32, attr);
}

struct ExtractFloat32Result {
  Value sign;
  Value is_special;
  Value man;
  Value exp;
};

static ExtractFloat32Result extractFloat32(OpBuilder &b, Location loc,
                                           Value input) {
  Value C1 = cI32(b, loc, 1);
  Value C31 = cI32(b, loc, 31);
  Value C23 = cI32(b, loc, 23);
  Value CFF = cI32(b, loc, 255);
  Value C7FFFFF = cI32(b, loc, 0x7FFFFF);

  // ---- 1) bits + exp ----
  auto shapedType = dyn_cast_if_present<ShapedType>(input.getType());
  Type srcElemTy = shapedType.getElementType();
  Type intElemTy = b.getIntegerType(srcElemTy.getIntOrFloatBitWidth());
  Type dstTy = shapedType.clone(intElemTy);

  auto bitcastEmptyOp =
      utils::createEmptyOpWithTargetElemType(b, loc, input, intElemTy);
  auto bits = b.create<hfusion::BitcastOp>(loc, dstTy, input, bitcastEmptyOp)
                  ->getResult(0);

  // sign = (bits>>31)&1
  auto tmp0 =
      createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::shrui, bits, C31)
          ->getResult(0);
  auto sign =
      createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::vand, tmp0, C1)
          ->getResult(0);

  // exp = (bits >> 23) & 0xFF
  auto tmp1 =
      createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::shrui, bits, C23)
          ->getResult(0);
  auto exp = createVandOp(b, loc, CFF, tmp1)->getResult(0);

  // is_special = (exp==0xFF)
  auto is_special = createCmpOp(b, loc, exp, CFF, CompareFn::veq)->getResult(0);
  // man = bits & 0x7FFFFF
  auto man = createVandOp(b, loc, C7FFFFF, bits)->getResult(0);

  return ExtractFloat32Result{sign, is_special, man, exp};
}

struct PayneHanekResult {
  Value k;          // i32
  Value y_float;    // f32
  Value sign;       // i32
  Value is_special; // i1
};

enum class PiReductionMode { Pi, PiOver2 };

static int64_t getStaticNumel(RankedTensorType ty) {
  int64_t prod = 1;
  for (int64_t d : ty.getShape()) {
    if (d == ShapedType::kDynamic)
      return ShapedType::kDynamic;
    prod *= d;
  }
  return prod;
}

static Value CreateGather1DOp(OpBuilder &b, Location loc, Value tbl_t,
                              Value idx) {
  Type tblElemTy = getElementTypeOrSelf(tbl_t);
  Value init = utils::createEmptyOpWithTargetElemType(b, loc, idx, tblElemTy);
  return b.create<hfusion::GatherOp>(loc, tbl_t, idx, init, /*axis=*/0)
      ->getResult(0);
}

static PayneHanekResult
payneHanekScalar_k_yfloat(OpBuilder &b, Location loc, Value inElem, Value tbl_t,
                          PiReductionMode mode = PiReductionMode::Pi) {

  auto f32 = b.getF32Type();

  Value C8 = cI32(b, loc, 8);
  Value C16 = cI32(b, loc, 16);
  Value Crshift, Cand, Cdiv;
  if (mode == PiReductionMode::PiOver2) {
    Crshift = cI32(b, loc, 30);
    Cand = cI32(b, loc, 0x3fffffff);
    Cdiv = cF32(b, loc, 9.313225746154785e-10); // 1/2^30
  } else {
    Crshift = cI32(b, loc, 31);
    Cand = cI32(b, loc, 0x7fffffff);
    Cdiv = cF32(b, loc, 4.656612873077393e-10); // 1/2^31
  }

  Value C32 = cI32(b, loc, 32);
  Value CFFFF = cI32(b, loc, 0xFFFF);
  Value C8388608 = cI32(b, loc, 0x800000); // 1<<23

  auto ph = extractFloat32(b, loc, inElem);
  const auto &[sign, is_special, man, exp] = ph;

  // man_real = man + (1<<23)
  Value man_real =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::add, man, C8388608)
          ->getResult(0);

  // ---- 1) table lookup index ----
  // m = exp + 9 ; idx_high = m - 1 ; idx_low = idx_high + 32

  Value idx_high =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::add, exp, C8)
          ->getResult(0);
  Value idx_low =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::add, idx_high, C32)
          ->getResult(0);

  // ---- 2) gather table limbs ----

  Value two_inv_pi_high = CreateGather1DOp(b, loc, tbl_t, idx_high);
  Value two_inv_pi_low = CreateGather1DOp(b, loc, tbl_t, idx_low);

  // ---- 3) 16-bit limb multiplication ----
  // xh/xl
  Value xh = createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::shrui,
                                           man_real, C16)
                 ->getResult(0);
  Value xl = createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::vand,
                                           man_real, CFFFF)
                 ->getResult(0);

  // phh/phl, plh/pll
  Value phh = createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::shrui,
                                            two_inv_pi_high, C16)
                  ->getResult(0);
  Value phl = createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::vand,
                                            two_inv_pi_high, CFFFF)
                  ->getResult(0);
  Value plh = createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::shrui,
                                            two_inv_pi_low, C16)
                  ->getResult(0);
  Value pll = createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::vand,
                                            two_inv_pi_low, CFFFF)
                  ->getResult(0);

  // ---- 4) t0..t5 ----

  Value mul0 =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::mul, xl, plh)
          ->getResult(0);
  Value t0 =
      createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::shrui, mul0, C16)
          ->getResult(0);
  Value t1 =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::mul, xl, phl)
          ->getResult(0);
  Value mul2 =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::mul, xl, phh)
          ->getResult(0);
  Value and2 = createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::vand,
                                             mul2, CFFFF)
                   ->getResult(0);
  Value t2 =
      createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::shli, and2, C16)
          ->getResult(0);
  Value mul3 =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::mul, xh, pll)
          ->getResult(0);
  Value t3 =
      createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::shrui, mul3, C16)
          ->getResult(0);
  Value t4 =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::mul, xh, plh)
          ->getResult(0);
  Value mul5 =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::mul, xh, phl)
          ->getResult(0);
  Value and5 = createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::vand,
                                             mul5, CFFFF)
                   ->getResult(0);
  Value t5 =
      createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::shli, and5, C16)
          ->getResult(0);

  // r0 = t0+t1+t2+t3+t4+t5

  Value a01 =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::add, t0, t1)
          ->getResult(0);
  Value a23 =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::add, t2, t3)
          ->getResult(0);
  Value a45 =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::add, t4, t5)
          ->getResult(0);
  Value a0123 =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::add, a01, a23)
          ->getResult(0);
  Value r0 =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::add, a0123, a45)
          ->getResult(0);

  // ---- 5) k, y ----
  // pi/2: k = r0 >> 30 pi: k = r0 >> 31
  Value k = createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::shrui, r0,
                                          Crshift)
                ->getResult(0);
  // pi/2: y = r0 & 0x3FFFFFFF pi:y = r0 & 0x7FFFFFFF
  Value y =
      createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::vand, r0, Cand)
          ->getResult(0);

  // ---- 6) y_float ----
  // pi/2: y_float = float(y) * (1/2^30) pi: y_float = float(y) * (1/2^31)
  Value y_f = hfusion::castTo(b, y, f32, hfusion::RoundMode::RINT);
  Value y_float =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::mul, y_f, Cdiv)
          ->getResult(0);

  return PayneHanekResult{k, y_float, sign, is_special};
}

/*
Generated IR once:
module {
  memref.global "private" constant @tbl : memref<?xi32, #hivm.address_space<gm>>
  = dense<"0x..."> {alignment = 32 : i64}
  ......
}
*/
namespace {
llvm::sys::SmartMutex<true> gGlobalTableMutex;
}
static memref::GlobalOp buildGlobalFromInitAttr(
    ModuleOp module, OpBuilder &b, Location loc, llvm::StringRef symName,
    ElementsAttr init, llvm::StringRef visibility = "private",
    int64_t alignmentBytes = 32,
    llvm::StringRef memSpaceStr = "#hivm.address_space<gm>") {
  // Serialize symbol-table mutation to make this helper thread-safe.
  llvm::sys::SmartScopedLock<true> lock(gGlobalTableMutex);
  if (!init)
    llvm::report_fatal_error("initial_value (ElementsAttr) is null.");

  auto tensorTy = dyn_cast<RankedTensorType>(init.getType());
  if (!tensorTy)
    llvm::report_fatal_error("initial_value must be RankedTensorType.");
  if (tensorTy.getRank() != 1)
    llvm::report_fatal_error("This helper expects a 1D table: tensor<NxT>.");

  int64_t N = tensorTy.getDimSize(0);
  if (N <= 0 || ShapedType::isDynamic(N))
    llvm::report_fatal_error("Table size N must be static and > 0.");

  Type elemTy = tensorTy.getElementType();
  MLIRContext *ctx = module.getContext();

  Attribute memSpace = parseAttribute(memSpaceStr, ctx);
  if (!memSpace)
    llvm::report_fatal_error("memSpaceStr parse failed.");

  // Target memref type: memref<N x elemTy, memSpace>
  auto memrefTy =
      MemRefType::get({N}, elemTy, MemRefLayoutAttrInterface{}, memSpace);

  // If a global with the same symbol name already exists, verify it is
  // compatible. We reject mismatches to avoid invalid IR (e.g., existing
  // memref.get_global users would have a different result type).
  if (auto existing = module.lookupSymbol<memref::GlobalOp>(symName)) {
    auto existingTy = dyn_cast<MemRefType>(existing.getType());
    if (!existingTy) {
      existing.emitError("existing symbol is not a MemRefType memref.global");
      llvm::report_fatal_error(
          "refuse to update: existing symbol type is invalid.");
    }

    if (existingTy.getRank() != 1 || existingTy.getDimSize(0) != N ||
        ShapedType::isDynamic(existingTy.getDimSize(0))) {
      existing.emitError("existing memref.global size mismatch: ")
          << existingTy << " vs expected " << memrefTy;
      llvm::report_fatal_error("refuse to update: global size mismatch.");
    }

    if (existingTy.getElementType() != elemTy) {
      existing.emitError("existing memref.global element type mismatch: ")
          << existingTy.getElementType() << " vs expected " << elemTy;
      llvm::report_fatal_error("refuse to update: element type mismatch.");
    }

    if (existingTy.getMemorySpace() != memSpace) {
      existing.emitError("existing memref.global memory space mismatch: ")
          << existingTy.getMemorySpace() << " vs expected " << memSpace;
      llvm::report_fatal_error("refuse to update: memory space mismatch.");
    }

    return existing;
  }

  // Must be created at the top level of the module.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module.getBody());
  OperationState st(loc, memref::GlobalOp::getOperationName());
  st.addAttribute("sym_name", b.getStringAttr(symName));
  st.addAttribute("sym_visibility", b.getStringAttr(visibility));
  st.addAttribute("type", TypeAttr::get(memrefTy));
  st.addAttribute("constant", b.getUnitAttr());
  st.addAttribute("alignment", b.getI64IntegerAttr(alignmentBytes));
  st.addAttribute("initial_value", init);

  return cast<memref::GlobalOp>(b.create(st));
}

// Returns: tbl_t copied to local memory (tensor<NxElemTy>)
// Generated IR:
//%1 = call @__materialize_global_table_tensor_tbl() : () -> tensor<320xi32>
// func.func private @__materialize_global_table_tensor_tbl() -> tensor<320xi32>
// {
//   %gm = memref.get_global @tbl : memref<NxT, memSpace>
//   %gm_s = memref.reinterpret_cast %gm to offset: [0], sizes:[N], strides:[1]
//   : ... -> memref<NxT, strided<[1]>, memSpace> %local = memref.alloc() :
//   memref<NxT> memref.copy %gm_s, %local : ... %tbl_t =
//   bufferization.to_tensor %local restrict writable : memref<NxT> ->
//   tensor<NxT>
//}
static FailureOr<Value>
materializeGlobalTableAsTensor(PatternRewriter &rewriter, Operation *anchorOp,
                               ModuleOp module, Location loc,
                               StringRef globalName) {
  if (!anchorOp)
    return rewriter.notifyMatchFailure(module, "anchorOp is null");

  auto global = module.lookupSymbol<memref::GlobalOp>(globalName);
  if (!global)
    return rewriter.notifyMatchFailure(anchorOp, "memref.global not found");

  auto gmTy = dyn_cast<MemRefType>(global.getType());
  if (!gmTy || gmTy.getRank() != 1)
    return rewriter.notifyMatchFailure(anchorOp,
                                       "global type is not rank-1 MemRefType");

  int64_t N = gmTy.getDimSize(0);
  if (N <= 0 || ShapedType::isDynamic(N))
    return rewriter.notifyMatchFailure(
        anchorOp, "global table size must be static and > 0");

  Type elemTy = gmTy.getElementType();
  auto tensorTy = RankedTensorType::get({N}, elemTy);

  // insert before anchorOp
  PatternRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(anchorOp);

  // Materialize: get_global -> (optional) reinterpret -> alloc -> copy ->
  // to_tensor
  Attribute memSpace = gmTy.getMemorySpace();
  Value tbl_gm = rewriter.create<memref::GetGlobalOp>(loc, gmTy, globalName);

  OpFoldResult off0 = rewriter.getIndexAttr(0);
  OpFoldResult szN = rewriter.getIndexAttr(N);
  OpFoldResult st1 = rewriter.getIndexAttr(1);

  auto stridedLayout = StridedLayoutAttr::get(rewriter.getContext(),
                                              /*offset=*/0, /*strides=*/{1});
  auto gmStridedTy = MemRefType::get({N}, elemTy, stridedLayout, memSpace);

  Value tbl_gm_strided = rewriter.create<memref::ReinterpretCastOp>(
      loc, gmStridedTy, tbl_gm, off0, ArrayRef<OpFoldResult>{szN},
      ArrayRef<OpFoldResult>{st1});
  Value tbl_local =
      rewriter.create<memref::AllocOp>(loc, MemRefType::get({N}, elemTy));
  rewriter.create<memref::CopyOp>(loc, tbl_gm_strided, tbl_local);
  Value tbl_t = rewriter.create<bufferization::ToTensorOp>(
      loc, tensorTy, tbl_local, /*restrict=*/true, /*writable=*/true);

  return tbl_t;
}

template <size_t N>
FailureOr<Value> emitGlobalTableFromUI32ArrayAndLoadAsTensorI32(
    PatternRewriter &rewriter, Operation *anchorOp, ModuleOp module,
    Location loc, llvm::StringRef globalName,
    const std::array<uint32_t, N> &arr) {

  llvm::SmallVector<llvm::APInt, N> vals;
  for (uint32_t v : arr)
    vals.emplace_back(/*numBits=*/32, /*val=*/v, /*isSigned=*/false);

  auto tensorTy = mlir::RankedTensorType::get(
      {static_cast<int64_t>(vals.size())}, rewriter.getI32Type());
  auto init = mlir::DenseElementsAttr::get(tensorTy, vals);

  buildGlobalFromInitAttr(module, rewriter, loc, globalName, init);
  return materializeGlobalTableAsTensor(rewriter, anchorOp, module, loc,
                                        globalName);
}

Value buildSinOrCosCalc(OpBuilder &b, Location loc,
                               Value in,    // tensor<...xf32> any rank
                               Value tbl_t, // tensor<320xi32>
                               CalcMode mode) {
  auto f32 = b.getF32Type();

  auto inTy = dyn_cast<RankedTensorType>(in.getType());
  if (!inTy || !inTy.getElementType().isF32()) {
    llvm::report_fatal_error(
        "buildSinOrCosCalc expects ranked tensor<...xf32> input (any rank).");
  }

  Value C1 = cI32(b, loc, 1);

  // NaN constant f32：0x7FC00000
  Value nan_bits = cI32(b, loc, 0x7FC00000);
  Value nan_f = b.create<arith::BitcastOp>(loc, f32, nan_bits);
  Value pi_over_2 = cF32(b, loc, 1.5707963267948966f);
  Value pi = cF32(b, loc, 3.1415927f);

  auto ph = payneHanekScalar_k_yfloat(b, loc, in, tbl_t, PiReductionMode::Pi);
  const auto &[k, y_float, sign1, is_special] = ph;
  // 2) sign2
  // sign2 = k; if (mode==SIN) sign2 = k xor sign1
  Value sign2 = k;
  if (mode == CalcMode::SIN) {
    sign2 =
        createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::vxor, k, sign1)
            ->getResult(0);
  }

  // 3) tag = sign2 & 1
  Value tag =
      createHFusionElemwiseBinaryOp(b, loc, hfusion::BinaryFn::vand, sign2, C1)
          ->getResult(0);

  // 4) is0 = (tag == 0)
  Value C0 = cI32(b, loc, 0);
  Value is0 = createCmpOp(b, loc, tag, C0, CompareFn::veq)->getResult(0);

  // 5) sign_f = select(is0, +1.0, -1.0)
  Value one_f = cF32(b, loc, 1.0f);
  Value neg_one_f = cF32(b, loc, -1.0f);

  Value signInit =
      utils::createEmptyOpWithTargetElemType(b, loc, in, b.getF32Type());
  Value sign_f = b.create<hfusion::SelectOp>(
                      loc, TypeRange(signInit.getType()),
                      ValueRange({is0, one_f, neg_one_f}), ValueRange(signInit))
                     .getResult(0);

  // 6) y0 = y_float * pi
  Value y0 =
      createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::mul, y_float, pi)
          ->getResult(0);

  // 7) y2:
  // COS: y2 = pi_over_2 - y0
  // SIN: y1 = pi - y0; y2 = min(y1, y0)
  Value y2;
  if (mode == CalcMode::COS) {
    y2 = createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::sub, pi_over_2,
                                      y0)
             ->getResult(0);
  } else {
    Value y1 =
        createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::sub, pi, y0)
            ->getResult(0);
    y2 = createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::min_signed, y1,
                                      y0)
             ->getResult(0);
  }

  // 8) sin_approx = tayler<SIN>(..., y2, 5)
  Value sin_approx = tayler<hfusion::TaylerMode::SIN>(b, loc, y2, 5);

  // 9) result = sin_approx * sign_f
  Value result = createLinalgElemwiseBinaryOp(b, loc, linalg::BinaryFn::mul,
                                              sin_approx, sign_f)
                     ->getResult(0);

  // 10) result_nan = select(is_special, nan_f, result)
  Value outMaskedInit =
      utils::createEmptyOpWithTargetElemType(b, loc, in, b.getF32Type());
  Value result_nan =
      b.create<hfusion::SelectOp>(loc, TypeRange(outMaskedInit.getType()),
                                  ValueRange({is_special, nan_f, result}),
                                  ValueRange(outMaskedInit))
          ->getResult(0);

  return result_nan;
}
/*
 * Goal:
 *   Compute sin(x) and cos(x) using pi-based range reduction:
 *     1) Reduce x to: x ≈ k_pi * pi + r_pi, with r_pi in [0, pi)
 *     2) Use symmetry to map both sin(r_pi) and cos(r_pi) to sin(y) on [0,
 * pi/2] 3) Evaluate sin(y) with a polynomial approximation sin_poly(y)
 *
 * Notes for implementation:
 *   - For large |x|, use Payne–Hanek (or other high-precision reduction) to
 * avoid precision loss in k_pi and r_pi.
 *   - Special values: follow project convention for NaN/Inf propagation.
 *
 * Key identities (pi-based):
 *   A) sin(-x) = -sin(x),   cos(-x) =  cos(x)
 *   B) sin(x + pi) = -sin(x),   cos(x + pi) = -cos(x)
 *      => only parity (k_pi mod 2) matters after reduction by pi.
 *
 *   C) cos(r_pi) = sin(pi/2 - r_pi)
 *      Let t = (pi/2) - r_pi, then t ∈ (-pi/2, pi/2]
 *      Fold to y_cos = |t| ∈ [0, pi/2] and keep sign because sin is odd:
 *        sin(t) = sign(t) * sin(|t|)
 *
 *   D) sin(r_pi) = sin(pi - r_pi)
 *      For r_pi ∈ [0, pi), sin(r_pi) ≥ 0
 *      Fold to y_sin = min(r_pi, pi - r_pi) ∈ [0, pi/2]
 *      No extra sign is needed for this fold (both sides are positive).
 *
 * Step A: Range reduction to [0, pi)
 *   Let ax = abs(x).
 *   Use Payne–Hanek to compute:
 *       ax ≈ k_pi * pi + r_pi
 *   where:
 *       r_pi ∈ [0, pi)
 *       k_pi is an integer
 *
 * Step B: Period sign from k_pi mod 2
 *   Because adding pi flips both sin and cos:
 *       pi_sign = (k_pi & 1) ? -1 : +1
 *
 * Step C: Map to sin(y) on [0, pi/2]
 *   1) For sin(r_pi):
 *        y_sin = min(r_pi, pi - r_pi)      // y_sin ∈ [0, pi/2]
 *        sin_r = sin_poly(y_sin)
 *
 *   2) For cos(r_pi):
 *        t      = (pi/2) - r_pi            // t ∈ (-pi/2, pi/2]
 *        cos_r  = sin_poly(t) // cos(r_pi) = sin(t)
 *
 * Step D: Re-apply signs to get sin(x), cos(x)
 *   - cos(x) is even, so only pi_sign matters:
 *       cos(x) = pi_sign * cos_r
 *
 *   - sin(x) is odd, so re-apply x sign as well:
 *       x_sign = (x < 0) ? -1 : +1
 *       sin(x) = x_sign * pi_sign * sin_r
 *
 * Output:
 *   Approximated sin(x) and cos(x).
 */

FailureOr<Value> buildSinOrCos(PatternRewriter &rewriter,
                                      hfusion::ElemwiseUnaryOp op, Value input,
                                      CalcMode mode) {
  Location loc = op.getLoc();
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module)
    return failure();

  FailureOr<Value> tbl_t_or =
      emitGlobalTableFromUI32ArrayAndLoadAsTensorI32<320>(rewriter, op, module,
                                                          loc, "tbl", tbl);

  if (failed(tbl_t_or))
    return failure();

  auto f32 = rewriter.getF32Type();
  auto inTy = dyn_cast<RankedTensorType>(input.getType());
  int64_t inRank = inTy.getRank();
  Value tbl_t = *tbl_t_or;

  // NOTE: Global cos/sin table is 1D <320> i32.
  // hfusion.gather requires idx/table match on all dims except last.
  // So we collapse input to 1D for the cos/sin path, then expand back.
  if (inRank == 1) {
    return buildSinOrCosCalc(rewriter, loc, input, tbl_t, mode);
  } else {
    int64_t inNumel = getStaticNumel(inTy);
    auto in1DTy = RankedTensorType::get({inNumel}, f32);
    SmallVector<ReassociationIndices, 1> reassoc;
    ReassociationIndices indices(
        llvm::to_vector(llvm::seq<int64_t>(0, inRank)));
    reassoc.push_back(indices);

    // Collapse the N-D input into 1D to satisfy hfusion.gather constraints.
    Value collapsedIn =
        rewriter.create<tensor::CollapseShapeOp>(loc, in1DTy, input, reassoc);
    auto outOrigTy = RankedTensorType::get(inTy.getShape(), f32);

    Value calc_res = buildSinOrCosCalc(rewriter, loc, collapsedIn, tbl_t, mode);

    // recover the reverse reshaped value to preserve reshape correctness
    auto expandBack = rewriter.create<tensor::ExpandShapeOp>(loc, outOrigTy,
                                                             calc_res, reassoc);
    tensor::CollapseShapeOp collapseFromExpand =
        mlir::tensor::reshape_utils::createExpandInverse(rewriter, expandBack);

    Value res = mlir::tensor::reshape_utils::getReverseReshapedValue(
        rewriter, calc_res, {collapseFromExpand.getOperation()});
    return res;
  }
}

// get polyexpr in the format [(input + p1) * squareSrc + p2] * squareSrc + ...,
// enableLastMulTerm = false means [(input + p1) * squareSrc + p2] + ... remove
// the last multiplication by squareSrc.
Value genPolyExpr(PatternRewriter &rewriter, Location loc,
                  const Value squareSrc, Value input,
                  const llvm::SmallVector<double> &numerCoeff,
                  bool enableLastMulTerm) {
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

// Explicit template instantiations
template Value mlir::hfusion::sign<TaylerMode::SIN>(PatternRewriter &, Location, Value);
template Value mlir::hfusion::sign<TaylerMode::ATAN>(PatternRewriter &, Location, Value);
template Value mlir::hfusion::tayler<TaylerMode::SIN>(OpBuilder &, Location, Value, int);
template Value mlir::hfusion::tayler<TaylerMode::ATAN>(OpBuilder &, Location, Value, int);

} // namespace mlir::hfusion
