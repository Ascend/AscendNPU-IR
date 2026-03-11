//===- Normalize .cpp -------------------- Normalize HFusion  -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is for normalizing HFusion.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/MemRef/IR/MemRefImpl.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#include "bishengir/Dialect/Tensor/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Mutex.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_NORMALIZE
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hfusion-normalize-ops"

using namespace mlir;
using namespace mlir::hfusion;

static thread_local bool archIsRegbased{false};
static thread_local bool archisAscend950{false};
static thread_local bool archisAscend310B{false};
static thread_local bool archisMembased{false};

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

// tayler x =
// taylerParams[0]*x+taylerParams[1]*x^3+...+taylerParams[i]*x^(2*i+1)
template <hfusion::TaylerMode taylerMode>
static Value tayler(OpBuilder &b, Location loc, Value x,
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

enum class CalcMode { SIN, COS };

static Value buildSinOrCosCalc(OpBuilder &b, Location loc,
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

static FailureOr<Value> buildSinOrCos(PatternRewriter &rewriter,
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

// Normalize argmax and argmin
// hfusion.reduce_with_index <max>(src)
// is normalized to
// src_nan_mask = hfusion.isnan(src)
// new_src = hfusion.select(src_nan_mask, -inf, src)
// hfusion.reduce_with_index <max> (new_src)
namespace {
struct NormalizeArgMinMaxOp
    : public OpRewritePattern<hfusion::ReduceWithIndexOp> {
  using OpRewritePattern<hfusion::ReduceWithIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    SmallVector<Value> inputs = op.getInputs();
    assert(inputs.size() == 2);
    Value src = inputs[0];
    Value src1 = inputs[1];

    static constexpr llvm::StringLiteral kAlreadyInitalizeInit =
        "already_initialize_init";
    if (op->hasAttr(kAlreadyInitalizeInit)) {
      return failure();
    }

    Type elemType = getElementTypeOrSelf(src);
    if (elemType.isInteger()) {
      return failure();
    }

    rewriter.setInsertionPointAfter(op);
    Location loc = op.getLoc();
    auto kind = op.getReduceKind();
    auto tieBreakLeft = op.getTieBreakLeftAttr();
    auto dims = op.getDimensions();
    auto unsignedSource = op.getUnsignedSrcAttr();
    bool isMin = kind.getReduceWithIndexKind() == ReduceWithIndexKind::MIN;

    auto infSign = isMin ? 1 : -1;
    double signedInf = infSign * std::numeric_limits<double>::infinity();

    auto infValue =
        utils::createConstantOp<double>(rewriter, loc, elemType, signedInf);
    utils::createConstantOp<double>(rewriter, loc, elemType, 0.);

    auto srcMask = utils::createEmptyOpWithTargetElemType(rewriter, loc, src,
                                                          rewriter.getI1Type());
    auto srcNanMask =
        rewriter.create<hfusion::IsNanOp>(loc, srcMask.getType(), src)
            .getResult();

    auto srcNanMasked = utils::createEmptyOp(rewriter, loc, src);
    srcNanMasked =
        rewriter
            .create<hfusion::SelectOp>(loc, TypeRange(srcNanMasked),
                                       ValueRange({srcNanMask, infValue, src}),
                                       ValueRange(srcNanMasked))
            .getResults()[0];

    auto srcNanVals = utils::createEmptyOp(rewriter, loc, op.getResults()[0]);
    auto srcNanIdxs = utils::createEmptyOp(rewriter, loc, op.getResults()[1]);
    auto srcNanReduceOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        loc, TypeRange{srcNanVals.getType(), srcNanIdxs.getType()},
        /*input*/ ValueRange{srcNanMasked, src1},
        /*outputValue&Index*/
        ValueRange{srcNanVals, srcNanIdxs}, kind, unsignedSource, tieBreakLeft,
        dims);
    rewriter.modifyOpInPlace(srcNanReduceOp, [&]() {
      srcNanReduceOp->setAttr(kAlreadyInitalizeInit,
                              UnitAttr::get(op->getContext()));
    });
    rewriter.replaceOp(op, srcNanReduceOp);
    return success();
  }
};
} // namespace
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

/// Normalize divsi with i8 type for regbase:
/// c = a / b
/// is normalized to
/// a16 = castTo<i16>(a)
/// b16 = castTo<i16>(b)
/// c16 = a16 / b16
/// c = castTo<i8>(c16, mode = TRUNC, sat=true)
/// Normalize divsi and divui for membase:
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
    auto resTensor = op.getResultTensors()[0];
    auto resTy = dyn_cast<TensorType>(resTensor.getType());
    auto elemTySrc = getElementTypeOrSelf(resTy);
    if (archisMembased && (!elemTySrc.isInteger())) {
      return failure();
    }
    if (archIsRegbased && (!elemTySrc.isInteger(8))) {
      return failure();
    }
    rewriter.setInsertionPoint(op);
    auto inputs = op.getDpsInputs();
    if (archisMembased) {
      auto res = hfusion::divWithRoundMode(rewriter, loc, elemTySrc, inputs[0],
                                           inputs[1], resTensor,
                                           hfusion::RoundMode::TRUNC);
      rewriter.replaceOp(op, res);
      return success();
    }
    // cast lhs and rhs from u8/i8 to u16/i16
    hfusion::TypeFn castIntegerType =
        (op.getFun() == linalg::BinaryFn::div_unsigned)
            ? hfusion::TypeFn::cast_unsigned
            : hfusion::TypeFn::cast_signed;
    Value castI16X = hfusion::castTo(rewriter, inputs[0], rewriter.getI16Type(),
                                     castIntegerType);
    Value castI16Y = hfusion::castTo(rewriter, inputs[1], rewriter.getI16Type(),
                                     castIntegerType);

    auto divInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, resTensor, rewriter.getI16Type());
    auto divI16 =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, op.getFun(), ValueRange{castI16X, castI16Y},
            ValueRange(divInit))
            ->getResults()[0];
    if (!archisAscend950) {
      Value res = hfusion::castTo(rewriter, divI16, elemTySrc,
                                  hfusion::RoundMode::TRUNC);
      rewriter.replaceOp(op, res);
      return success();
    }
    if (op.getFun() == linalg::BinaryFn::div_unsigned) {
      Value res = hfusion::castTo(rewriter, divI16, elemTySrc,
                                  hfusion::RoundMode::TRUNC);
      rewriter.replaceOp(op, res);
      return success();
    }
    // avoid -128/-1 overflow error in i8 with div.i16
    auto i8ExcdNum =
        utils::createConstantOp<int>(rewriter, loc, rewriter.getI16Type(), 128);
    auto i8MinNum = utils::createConstantOp<int>(rewriter, loc,
                                                 rewriter.getI16Type(), -128);
    Value exceedMask =
        createCmpOp(rewriter, loc, divI16, i8ExcdNum, CompareFn::veq)
            ->getResult(0);
    auto finalResult =
        rewriter
            .create<hfusion::SelectOp>(loc, TypeRange(divInit),
                                       ValueRange{exceedMask, i8MinNum, divI16},
                                       ValueRange(divI16))
            .getResults()[0];

    Value res = hfusion::castTo(rewriter, finalResult, rewriter.getF16Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow*/ false, true);
    res = hfusion::castTo(rewriter, res, elemTySrc, hfusion::RoundMode::TRUNC,
                          std::nullopt,
                          /*enableOverflow*/ false, true, castIntegerType);
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

static Value castU32ToI64ToF32(hfusion::CastOp &op, PatternRewriter &rewriter) {
  auto castU32ToI64 = castTo(rewriter, op.getDpsInputOperand(0)->get(),
                             rewriter.getI64Type(), op.getCast());
  auto castI64ToFp32 = hfusion::castTo(
      rewriter, castU32ToI64, rewriter.getF32Type(), TypeFn::cast_signed);
  return castI64ToFp32;
}

static Value castU32ToI64ToF32ToOut(hfusion::CastOp &op, Type targetType,
                                    PatternRewriter &rewriter) {
  // u32 -> i64 -> fp32
  Value u32ToF32Result = castU32ToI64ToF32(op, rewriter);
  // fp32 -> fp16/bf16
  auto castF32ToOut = hfusion::castTo(rewriter, u32ToF32Result, targetType,
                                      hfusion::TypeFn::cast_signed);
  return castF32ToOut;
}

// i1/i8/i16/u8/u16 -> f16 -> targetType
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

// membase: i8 -> f16 -> f32 -> i64
// regbase: i8 -> i32 -> i64
static Value castI8ToI64(hfusion::CastOp &op, PatternRewriter &rewriter) {
  if (!archIsRegbased) {
    // i8 -> f16 -> f32
    Value i8ToF32Result =
        castSrcToFp16ToTargetType(op, rewriter.getF32Type(), rewriter);
    // f32->i64
    Type i64Type = rewriter.getIntegerType(64);
    auto castF32ToDst =
        castTo(rewriter, i8ToF32Result, i64Type, TypeFn::cast_signed);
    return castF32ToDst;
  } else {
    Value dpsInput = op.getDpsInputOperand(0)->get();
    auto castIntegerType = op.getCast();
    auto castValue =
        castTo(rewriter, dpsInput, rewriter.getI32Type(), castIntegerType);
    return castTo(rewriter, castValue, rewriter.getI64Type(), castIntegerType);
  }
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

namespace {
template <typename OpType>
std::optional<StringRef> getAnnotateOverflowMode(OpType op) {
  std::optional<Operation *> overflowMode =
      utils::getAnnotateOpWithAttr(op.getResult(0), "overflow_mode");
  if (!overflowMode.has_value()) {
    return std::nullopt;
  }
  StringAttr overflowAttrVal =
      overflowMode.value()->getAttrOfType<StringAttr>("overflow_mode");
  return overflowAttrVal.getValue();
}

template <typename OpType>
std::optional<bool> getAnnotateAttrBool(OpType op, StringRef attr) {
  std::optional<Operation *> attrOp =
    utils::getAnnotateOpWithAttr(op.getResult(0), attr);
  if (!attrOp.has_value())
    return std::nullopt;

  if (auto boolAttr =
      attrOp.value()->getAttrOfType<BoolAttr>(attr)) {
    return boolAttr.getValue();
  }

  return std::nullopt;
}

} // namespace

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
    castValue = hfusion::castTo(
        rewriter, castValue, outType, hfusion::RoundMode::TRUNC, std::nullopt,
        /*enableOverflow=*/false, false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::F32TOI8:
    // step 1: cast f32 to f16 in TRUNC mode
    castValue =
        hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                        hfusion::RoundMode::TRUNC, std::nullopt,
                        /*enableOverflow=*/false, false, castIntegerType);
    // step 2: cast f16 to i8 in TRUNC mode
    castValue = hfusion::castTo(
        rewriter, castValue, outType, hfusion::RoundMode::TRUNC, std::nullopt,
        /*enableOverflow=*/false, false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  case hfusion::CastMode::F16TOI8:
    castValue = hfusion::castTo(
        rewriter, castValue, outType, hfusion::RoundMode::TRUNC, std::nullopt,
        /*enableOverflow=*/false, false, castIntegerType);
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
                        std::nullopt, /*enableOverflow*/ false);
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
    castValue =
        hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                        hfusion::RoundMode::TRUNC, std::nullopt,
                        /*enableOverflow=*/false, false, hfusion::TypeFn{});
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
  assert(!archIsRegbased);
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
        std::nullopt, true, false, castIntegerType);
    // step2: cast i32 to i16
    castValue = hfusion::castTo(rewriter, castI32Value, rewriter.getI16Type(),
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isF32ToI8) {
    // step 1: cast f32 to i32 in TRUNC mode
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getI32Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt, true,
                                false, castIntegerType);
    // step 2: cast i32 to i8 in TRUNCWITHOVERFLOW mode
    castValue = hfusion::castTo(rewriter, castValue, outType,
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isF16ToI8 && op.getEnableOverflow()) {
    Value overflowResult = hfusion::OverflowProcess(
        rewriter, castValue, getElementTypeOrSelf(outType));
    castValue =
        hfusion::castTo(rewriter, overflowResult, outType,
                        hfusion::RoundMode::TRUNC, std::nullopt,
                        /*enableOverflow=*/false, false, castIntegerType);
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

namespace {
// Handle overflow_mode = saturate
LogicalResult handleOverflowModeForSaturate(hfusion::CastOp op,
                                            PatternRewriter &rewriter,
                                            bool enableSaturate) {
  auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
  auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());

  const bool isF32ToI8 = inType.isF32() && outType.isInteger(8);
  const bool isI64ToI16 = inType.isInteger(64) && outType.isInteger(16);
  const bool isI64ToI8 = inType.isInteger(64) && outType.isInteger(8);
  const bool isI32ToI8 = inType.isInteger(32) && outType.isInteger(8);
  const bool isF16ToI8 = inType.isF16() && outType.isInteger(8);
  const bool isF32ToI16 = inType.isF32() && outType.isInteger(16);

  const bool isI64ToI32 = inType.isInteger(64) && outType.isInteger(32);
  const bool isI32ToI16 = inType.isInteger(32) && outType.isInteger(16);
  const bool isI16ToI8 = inType.isInteger(16) && outType.isInteger(8);
  hfusion::TypeFn castIntegerType = op.getCast();
  hfusion::UnsignedMode unsignedMode = hfusion::UnsignedMode::SI2SI;

  auto srcUnsigned = getAnnotateAttrBool(op, util::saturateSrcUnsigned);
  const bool srcIsUnsigned = srcUnsigned.value_or(false);
  auto dstUnsigned = getAnnotateAttrBool(op, util::saturateDstUnsigned);
  const bool dstIsUnsigned = dstUnsigned.value_or(false);

  auto srcAttr =
      utils::getAnnotateOpWithAttr(op->getResult(0), util::saturateSrcUnsigned);
  if (srcAttr.has_value()) {
    annotation::MarkOp srcMarkOp =
      dyn_cast<annotation::MarkOp>(srcAttr.value());
    rewriter.eraseOp(srcMarkOp);
  }
  auto dstAttr =
      utils::getAnnotateOpWithAttr(op->getResult(0), util::saturateDstUnsigned);
  if (dstAttr.has_value()) {
    annotation::MarkOp dstMarkOp =
      dyn_cast<annotation::MarkOp>(dstAttr.value());
    rewriter.eraseOp(dstMarkOp);
  }

  const bool isSIToSI = !srcIsUnsigned && !dstIsUnsigned;
  const bool isSIToUI = !srcIsUnsigned && dstIsUnsigned;
  const bool isUIToSI = srcIsUnsigned && !dstIsUnsigned;
  const bool isUIToUI = srcIsUnsigned && dstIsUnsigned;

  if (isSIToUI) {
    unsignedMode = hfusion::UnsignedMode::SI2UI;
  } else if (isUIToSI) {
    unsignedMode = hfusion::UnsignedMode::UI2SI;
  } else if (isUIToUI) {
    unsignedMode = hfusion::UnsignedMode::UI2UI;
  }

  Value castValue = op.getInputs()[0];
  if (isF32ToI16) {
    castValue =
        hfusion::castTo(rewriter, castValue, rewriter.getI32Type(),
                        hfusion::RoundMode::TRUNCWITHOVERFLOW, std::nullopt,
                        /*enableOverflow*/ false, enableSaturate);
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isF32ToI8) {
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt,
                                /*enableOverflow*/ false, enableSaturate);
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isF16ToI8) {
    castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI64ToI32) {
    castValue =
        hfusion::castTo(rewriter, castValue, outType,
                        hfusion::RoundMode::TRUNC, std::nullopt,
                        /*enableOverflow*/ false, enableSaturate,
                        castIntegerType, unsignedMode);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI64ToI16) {
    if (isSIToSI) {
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF32Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue = hfusion::castTo(rewriter, castValue, outType,
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  castIntegerType);
    } else {
      castValue = hfusion::castTo(rewriter, castValue, outType,
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  castIntegerType, unsignedMode);
    }

    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI64ToI8) {
    if (isSIToSI) {
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF32Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue = hfusion::castTo(rewriter, castValue, outType,
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  castIntegerType);
    } else {
      castValue = hfusion::castTo(rewriter, castValue, outType,
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  castIntegerType, unsignedMode);
    }

    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI32ToI16) {
    if (isSIToSI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    } else if (isSIToUI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    } else if (isUIToSI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    } else if (isUIToUI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    }

    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI32ToI8) {
    if (isSIToSI) {
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF32Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue =
          hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                          std::nullopt, /*enableOverflow*/ false,
                          enableSaturate, castIntegerType);
    } else if (isSIToUI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    } else if (isUIToSI) { // u32-s16-f16-s8
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getI16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  castIntegerType, hfusion::UnsignedMode::UI2SI);
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue =
          hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                          std::nullopt, /*enableOverflow*/ false,
                          enableSaturate, castIntegerType);
    } else if (isUIToUI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    }

    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isI16ToI8) {
    if (isSIToSI) {
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate);
      castValue =
          hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                          std::nullopt, /*enableOverflow*/ false,
                          enableSaturate, castIntegerType, unsignedMode);
    } else if (isSIToUI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    } else if (isUIToSI) {
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getI8Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  hfusion::TypeFn::cast_unsigned,
                                  hfusion::UnsignedMode::UI2UI);
      castValue = hfusion::castTo(rewriter, castValue, rewriter.getF16Type(),
                                  hfusion::RoundMode::TRUNC, std::nullopt,
                                  /*enableOverflow*/ false, enableSaturate,
                                  hfusion::TypeFn::cast_unsigned);
      castValue =
          hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                          std::nullopt, /*enableOverflow*/ false,
                          enableSaturate, castIntegerType);
    } else if (isUIToUI) {
      castValue =
        hfusion::castTo(rewriter, castValue, outType, hfusion::RoundMode::TRUNC,
                        std::nullopt, /*enableOverflow*/ false,
                        enableSaturate, castIntegerType, unsignedMode);
    }

    rewriter.replaceOp(op, castValue);
    return success();
  }
  return failure();
}

// Handle overflow_mode = trunc
// For IntToInt, we retain the original hfusion.cast statement
LogicalResult handleOverflowModeForTrunc(hfusion::CastOp op,
                                         PatternRewriter &rewriter) {
  auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
  auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());

  const bool isF32ToI16 = inType.isF32() && outType.isInteger(16);
  const bool isF32ToI8 = inType.isF32() && outType.isInteger(8);
  hfusion::TypeFn castIntegerType = op.getCast();

  Value castValue = op.getInputs()[0];
  if (isF32ToI16) {
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getI32Type(),
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getI16Type(),
                                hfusion::RoundMode::TRUNC, std::nullopt, true,
                                false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  } else if (isF32ToI8) {
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getI32Type(),
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    castValue = hfusion::castTo(rewriter, castValue, rewriter.getIntegerType(8),
                                hfusion::RoundMode::TRUNC, std::nullopt, true,
                                false, castIntegerType);
    rewriter.replaceOp(op, castValue);
    return success();
  }
  return failure();
}

} // namespace

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

// example:
// arith.truncf %arg0 : f32 to bf16
// is normalized to
// %c0 = arith.constant 0 : index
// %from_elements = tensor.from_elements %arg0 : tensor<1xf32>
// %0 = tensor.empty() : tensor<1xbf16>
// %1 = hfusion.cast ins(%from_elements : tensor<1xf32>) outs(%0 :
// tensor<1xbf16>) -> tensor<1xbf16> %extracted = tensor.extract %1[%c0] :
// tensor<1xbf16> for there is no implementation for f32 to bf16 scalar truncf
struct NormalizetruncfBf16 : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const final {
    auto src = op.getIn();
    auto dst = op.getOut();
    auto srcType = src.getType();
    auto dstType = dst.getType();

    if (!srcType.isF32() || !dstType.isBF16()) {
      return failure();
    }

    if (isa<hfusion::CastOp>(op->getParentOp())) {
      return failure();
    }

    auto loc = op.getLoc();
    SmallVector<Value> extentOperands{src};
    auto tensorType = RankedTensorType::get({1}, srcType);
    Value fromElementsOp = rewriter.create<tensor::FromElementsOp>(
        loc, tensorType, extentOperands);
    assert(fromElementsOp.getDefiningOp() != nullptr);
    auto castOp = hfusion::castTo(
        rewriter, fromElementsOp.getDefiningOp()->getResult(0), dstType);

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices = {c0};
    auto extractOp = rewriter.create<tensor::ExtractOp>(
        loc, castOp.getDefiningOp()->getResult(0), indices);
    rewriter.replaceOp(op, extractOp);
    return success();
  }
};

namespace {
// example:
// arith.extf %arg0 : bf16 to f32
// is normalized to
// %c0 = arith.constant 0 : index
// %from_elements = tensor.from_elements %arg0 : tensor<1xbf16>
// %0 = tensor.empty() : tensor<1xf32>
// %1 = hfusion.cast ins(%from_elements : tensor<1xbf16>) outs(%0 :
// tensor<1xf32>) -> tensor<1xf32> %extracted = tensor.extract %1[%c0] :
// tensor<1xf32> for there is no implementation for bf16 to f32 scalar extf
template <typename CastOp>
struct NormalizeScalarExtension : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const final {
    auto src = op.getIn();
    auto dst = op.getOut();
    auto srcType = src.getType();
    auto dstType = dst.getType();
    if (mlir::isa<ShapedType>(srcType) || mlir::isa<ShapedType>(dstType)) {
      return failure();
    }
    if (isa<hfusion::CastOp>(op->getParentOp()) ||
        isa<linalg::MatmulOp>(op->getParentOp()) ||
        isa<linalg::BatchMatmulOp>(op->getParentOp())) {
      return failure();
    }

    auto loc = op.getLoc();
    SmallVector<Value> extentOperands{src};
    auto tensorType = RankedTensorType::get({1}, srcType);
    Value fromElementsOp = rewriter.create<tensor::FromElementsOp>(
        loc, tensorType, extentOperands);
    auto castOp = hfusion::castTo(
        rewriter, fromElementsOp.getDefiningOp()->getResult(0), dstType);

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices = {c0};
    auto extractOp = rewriter.create<tensor::ExtractOp>(
        loc, castOp.getDefiningOp()->getResult(0), indices);
    rewriter.replaceOp(op, extractOp);
    return success();
  }
};
} // namespace

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
    auto castIntegerType = op.getCast();

    // Deal with trunc with overflow for wider-to-narrower-integer-of-non-i1.
    if (srcBitWidth > dstBitWidth && outType.isInteger() &&
        !outType.isInteger(1)) {
      auto overflowMode = getAnnotateOverflowMode(op);
      bool enableSaturate =
          overflowMode.has_value() && overflowMode->ends_with("saturate");
      if (!archIsRegbased) {
        if (enableSaturate) {
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
      } else {
        if (enableSaturate) {
          auto overflowModeAttr =
              utils::getAnnotateOpWithAttr(op->getResult(0), "overflow_mode");
          if (!overflowModeAttr.has_value()) {
            return failure();
          }
          annotation::MarkOp markOp =
              dyn_cast<annotation::MarkOp>(overflowModeAttr.value());
          rewriter.eraseOp(markOp);
          return handleOverflowModeForSaturate(op, rewriter, enableSaturate);
        }
        return handleOverflowModeForTrunc(op, rewriter);
      }
    }

    const bool isI64ToF16 = inType.isInteger(64) && outType.isF16();
    const bool isIntegerToBF16 =
        (inType.isInteger(64) || inType.isInteger(32) || inType.isInteger(16) ||
         inType.isInteger(8)) &&
        outType.isBF16();
    const bool isU16ToF16 = inType.isInteger(16) && outType.isF16() &&
                            hfusion::TypeFn::cast_unsigned == castIntegerType;
    if (isI64ToF16 || isIntegerToBF16 || isU16ToF16) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << inType << " to "
                 << outType << ", and rewrite to cast (from " << inType
                 << " to f16) " << "\n ");
      Value castResult;
      // I8ToBF16: I8ToF16 -> F16ToBF16 (regbase)
      // I8ToBF16: I8ToF16 -> F16ToF32 -> F32ToBF16 (membase)
      if (false && archIsRegbased && isIntegerToBF16 && inType.isInteger(8)) {
        // fixme: arith dialect has no fp-to-bf16 conversion. Need to extend
        // hfusion op first.
        castResult =
            castSrcToFp16ToTargetType(op, rewriter.getBF16Type(), rewriter);
      } else {
        castResult = castInToF32ToOut(op, rewriter);
      }
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isU32ToF32 = inType.isInteger(32) && outType.isF32() &&
                            hfusion::TypeFn::cast_unsigned == castIntegerType;
    if (isU32ToF32) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << inType << " to "
                 << outType << ", and rewrite to cast (from " << inType
                 << " to I64 to " << outType << ")\n");
      Value castResult = castU32ToI64ToF32(op, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isU32ToF16 = inType.isInteger(32) && outType.isF16() &&
                            hfusion::TypeFn::cast_unsigned == castIntegerType;
    const bool isU32ToBF16 = inType.isInteger(32) && outType.isBF16() &&
                             hfusion::TypeFn::cast_unsigned == castIntegerType;
    if (isU32ToF16 || isU32ToBF16) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << inType << " to "
                 << outType << ", and rewrite to cast (from " << inType
                 << " to I64 to F32 to " << outType << ")\n");
      Type targetType = getElementTypeOrSelf(outType);
      Value castResult = castU32ToI64ToF32ToOut(op, targetType, rewriter);
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
    const bool isI1ToI16 = inType.isInteger(1) && outType.isInteger(16);
    const bool isI1ToF32 = inType.isInteger(1) && outType.isF32();

    if (isI8ToF32 || isI1ToI16 ||
        (!archIsRegbased && (isI8ToI32 || isI8ToI16 || isI1ToF32))) {
      Type targetType = getElementTypeOrSelf(outType);
      Value castResult = castSrcToFp16ToTargetType(op, targetType, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isI1ToI32 = inType.isInteger(1) && outType.isInteger(32);
    if (!archIsRegbased && isI1ToI32) {
      Value inValue = op.getInputs()[0];
      Value castF16Value = hfusion::castTo(
          rewriter, inValue, rewriter.getF16Type(), hfusion::RoundMode::RINT);
      Value castI32Value =
          hfusion::castTo(rewriter, castF16Value, rewriter.getI32Type(),
                          hfusion::RoundMode::RINT);
      rewriter.replaceOp(op, castI32Value);
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

    // I16ToI64: I16ToF32 -> F32ToI64 (membase)
    // I16ToI64: I16ToI32 -> I32ToI64 (regbase)
    const bool isI16ToI64 = inType.isInteger(16) && outType.isInteger(64);
    if (isI16ToI64) {
      Value inValue = op.getInputs()[0];
      Value castValue;
      if (archIsRegbased) {
        castValue = hfusion::castTo(rewriter, inValue, rewriter.getI32Type(),
                                    castIntegerType);
      } else {
        castValue = hfusion::castTo(rewriter, inValue, rewriter.getF32Type(),
                                    hfusion::RoundMode::RINT);
      }

      Value castI64Value = hfusion::castTo(
          rewriter, castValue, rewriter.getI64Type(), castIntegerType);
      rewriter.replaceOp(op, castI64Value);
      return success();
    }

    // I16ToI32: I16ToF32 -> F32ToI32 (membase)
    // I16ToI32: OK (regbase)
    const bool isI16ToI32 = inType.isInteger(16) && outType.isInteger(32);
    if (!archIsRegbased && isI16ToI32) {
      Value inValue = op.getInputs()[0];
      Value castF32Value = hfusion::castTo(
          rewriter, inValue, rewriter.getF32Type(), hfusion::RoundMode::RINT);

      Value castI32Value =
          hfusion::castTo(rewriter, castF32Value, rewriter.getI32Type());
      rewriter.replaceOp(op, castI32Value);
      return success();
    }

    // AnyToF8: AnyToF32 -> F32ToF8
    const bool isAnyToF8 = (!inType.isF32()) &&
                           (outType.isFloat8E4M3FN() || outType.isFloat8E5M2());
    if (isAnyToF8) {
      Value castResult = castInToF32ToOut(op, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    // F8ToAny: F8ToF32 -> F32ToAny
    const bool isF8ToAny = (inType.isFloat8E4M3FN() || inType.isFloat8E5M2()) &&
                           (!outType.isF32());
    if (isF8ToAny) {
      Value castResult = castInToF32ToOut(op, rewriter);
      rewriter.replaceOp(op, castResult);
      return success();
    }

    const bool isF32ToU32 = inType.isF32() && outType.isInteger(32) &&
                            TypeFn::cast_unsigned == castIntegerType;
    if (isF32ToU32) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << inType << " to "
                 << outType << ", and rewrite to cast (from " << inType
                 << " to I64 to " << outType << ")\n");
      // F32 -> I64
      Value castF32ToI64 =
          hfusion::castTo(rewriter, op.getDpsInputOperand(0)->get(),
                          rewriter.getI64Type(), TypeFn::cast_signed);
      // I64 -> U32
      Value castI64ToU32 = hfusion::castTo(
          rewriter, castF32ToI64, rewriter.getI32Type(), TypeFn::cast_signed);
      rewriter.replaceOp(op, castI64ToU32);
      return success();
    }

    return failure();
  }
};

namespace {
struct NormalizeCmpOp : public OpRewritePattern<hfusion::CompareOp> {
public:
  using OpRewritePattern<CompareOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::CompareOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    // hfusion::CompareOp is used in cast op when casting to bool(int1).
    // Overflowmode annotation mark is useless in this case,
    // and would cause redundant vf in PreVectorizationFusion Pass.
    // So here we pair and delete overflow mark on CompareOp.
    auto overflowMode = getAnnotateOverflowMode(op);
    if (overflowMode.has_value()) {
      auto overflowModeAttr =
          utils::getAnnotateOpWithAttr(op->getResult(0), "overflow_mode");
      assert(overflowModeAttr.has_value());
      annotation::MarkOp markOp =
          dyn_cast<annotation::MarkOp>(overflowModeAttr.value());
      rewriter.eraseOp(markOp);
      return success();
    }
    return failure();
  }
};
} // namespace

struct NormalizeScalarCastOp : public OpRewritePattern<hfusion::CastOp> {
public:
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    Value originalInput = castOp.getInputs()[0];
    Value originalOutput = castOp.getOutputs()[0];

    auto inputTensorType = dyn_cast<RankedTensorType>(originalInput.getType());

    auto outputTensorType =
        dyn_cast<RankedTensorType>(originalOutput.getType());
    if (!inputTensorType || !outputTensorType ||
        inputTensorType.getRank() != 0 || outputTensorType.getRank() != 0) {
      return failure();
    }

    Type elemType = inputTensorType.getElementType();
    Type outElemType = outputTensorType.getElementType();

    Location loc = castOp.getLoc();
    auto extractInput =
        rewriter.create<tensor::ExtractOp>(loc, originalInput, ValueRange{});
    SmallVector<Value> extentOperands{extractInput};
    RankedTensorType inputDimType = RankedTensorType::get({1}, elemType);
    Value fromElementsOp = rewriter.create<tensor::FromElementsOp>(
        loc, inputDimType, extentOperands);

    auto newCastOp = hfusion::castTo(rewriter, fromElementsOp, outElemType,
                                     castOp.getRoundMode(), std::nullopt,
                                     castOp.getEnableOverflow());
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices = {c0};
    auto extractOp =
        rewriter.create<tensor::ExtractOp>(loc, newCastOp, indices);
    Value newInsert = rewriter.create<tensor::InsertOp>(
        loc, extractOp.getResult(), castOp.getOutputs()[0], ValueRange{});

    rewriter.replaceOp(castOp, newInsert);
    return success();
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

    // step 1: mask sign bit.
    // 1. vdup(7FFF) : (I32/I16)
    auto loc = op->getLoc();
    auto maskedSignValue = maskSignBit(rewriter, loc, input);

    // step 2: compared with negtive Infinity
    // 3.vadd(input, input, neg_inf_bitcast_as_int).
    auto minusInfValue = minusInfConstValue(rewriter, loc, maskedSignValue);

    // step3: change temp result to 1 which is > 1
    // vmin(input, input, 1) : (I32/I16)
    Type castType = rewriter.getIntegerType(elemType.getIntOrFloatBitWidth());
    arith::ConstantOp posOneOp = rewriter.create<arith::ConstantOp>(
        loc, castType, rewriter.getIntegerAttr(castType, 1));
    auto minOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::min_signed,
            ValueRange{minusInfValue, posOneOp->getResults()[0]},
            ValueRange{minusInfValue});

    // step4. change temp result to 0 which is < 0
    // vmax(input, input, 0) : (I32/I16)
    arith::ConstantOp zeroOp = rewriter.create<arith::ConstantOp>(
        loc, castType, rewriter.getIntegerAttr(castType, 0));
    auto maxOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            rewriter, loc, linalg::BinaryFn::max_signed,
            ValueRange({minOp->getResults()[0], zeroOp->getResults()[0]}),
            minOp->getResults()[0]);

    // step5. cast int32 to int1
    // cast(input, i32 -> i1)
    auto roundingAttr =
        rewriter.getAttr<hfusion::RoundModeAttr>(hfusion::RoundMode::RINT);
    auto modeAttr = rewriter.getNamedAttr(hfusion::RoundModeAttr::getMnemonic(),
                                          roundingAttr);
    hfusion::CastOp castToDst = rewriter.create<hfusion::CastOp>(
        loc, TypeRange(op.getOutput()), maxOp->getResults()[0], op.getOutput(),
        modeAttr);
    rewriter.replaceOp(op, castToDst);
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

namespace {
/// Convert linalg.broadcast to linalg.fill if input operand only has one elem.
/// necessary normalization to break cycle on infinite loop of propagate reshape
/// pass.
struct NormalizeScalarLikeTensorLinalgBrcOpNonDense
    : public OpRewritePattern<linalg::BroadcastOp> {
public:
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput().getDefiningOp<arith::ConstantOp>())
      return failure();
    auto inputShape = op.getInput().getType().getShape();
    if (ShapedType::isDynamicShape(inputShape))
      return failure();
    if (llvm::any_of(inputShape, [](auto &val) { return val != 1; }))
      return failure();
    SmallVector<Value> indices;
    indices.resize(
        inputShape.size(),
        rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0).getResult());
    auto extractOp = rewriter.create<tensor::ExtractOp>(op->getLoc(),
                                                        op.getInput(), indices);
    auto fillOp = rewriter.create<linalg::FillOp>(
        op->getLoc(), extractOp.getResult(), op.getInit());
    rewriter.replaceOp(op, fillOp);
    return success();
  }
};
} // namespace

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
    Value castLhs = hfusion::castTo(rewriter, lhs, targetType, rounding);
    Value castRhs = hfusion::castTo(rewriter, rhs, targetType, rounding);
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

    Value castResult =
        castTo(rewriter, newResult, rewriter.getI8Type(),
               hfusion::RoundMode::TRUNC, std::nullopt, enableOverflow, false,
               isUnsigned ? hfusion::TypeFn::cast_unsigned
                          : hfusion::TypeFn::cast_signed);
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
                                         bool isUnsigned) {
  if constexpr (std::is_same_v<bool, targetType>) {
    replaceI1ResultsWithTargetType(oldResults, newResults, rewriter);
  }
  if constexpr (std::is_same_v<int8_t, targetType>) {
    replaceI8ResultsWithTargetType(oldResults, newResults, rewriter, true,
                                   isUnsigned);
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
                                            const SmallVector<Value> &values,
                                            bool unsignedSrc = false) {
  SmallVector<Value> result;
  for (Value v : values) {
    if (!isElemType<srcType>(v.getType())) {
      result.push_back(v);
      continue;
    }

    Type dstType = rewriter.getType<targetType>();
    Value castResult = unsignedSrc ? castTo(rewriter, v, dstType,
                                            hfusion::TypeFn::cast_unsigned)
                                   : castTo(rewriter, v, dstType);
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

bool analyzeUnsignedNeeds(Value value) {
  // Since the `add` and `sub` operations in the Linalg dialect have the same
  // representation for uint and int scenarios, it is not possible to directly
  // determine the sign nature through the Linalg dialect itself. Instead, the
  // sign nature needs to be analyzed through the use-chain of the Value.
  // In cdiv, only a single layer of BFS on the use-chain is required.
  // Other maybe need more levels?
  for (Operation *user : value.getUsers()) {
    // In cdiv, the op type that needs to be checked is hfusion::Cast.
    if (auto castOp = dyn_cast<hfusion::CastOp>(user)) {
      if (castOp.getCast() == hfusion::TypeFn::cast_unsigned) {
        return true;
      }
    }
  }
  return false;
}

template <typename ElemType>
SmallVector<Value> normalizeToTargetType(PatternRewriter &rewriter,
                                         const SmallVector<Value> &values,
                                         Type targetType,
                                         bool isInputUnsigned) {
  SmallVector<Value> result;
  for (Value v : values) {
    if (!isElemType<ElemType>(v.getType())) {
      result.push_back(v);
      continue;
    }
    Value castResult =
        isInputUnsigned ? castTo(rewriter, v, targetType, TypeFn::cast_unsigned)
                        : castTo(rewriter, v, targetType);
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

    Type targetType;
    if (computeByF16) {
      targetType = rewriter.getF16Type();
    } else if (computeByF32) {
      targetType = rewriter.getF32Type();
    } else {
      llvm_unreachable("Unsupported Op.");
    }

    bool isInputUnsigned = false;
    bool UseChainUnsigned = false;
    if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
      linalg::BinaryFn linalgFn = binOp.getFunAttr().getValue();
      if (op->getNumResults() > 0) {
        UseChainUnsigned = analyzeUnsignedNeeds(op->getResult(0));
      }
      isInputUnsigned = linalgFn == linalg::BinaryFn::max_unsigned ||
                        linalgFn == linalg::BinaryFn::min_unsigned ||
                        UseChainUnsigned;
    }

    SmallVector<Value> newInputs = normalizeToTargetType<ElemType>(
        rewriter, op.getInputs(), targetType, isInputUnsigned);
    SmallVector<Value> newOutputs = normalizeToTargetType<ElemType>(
        rewriter, op.getOutputs(), targetType, isInputUnsigned);
    Operation *newOp = createBodyOp(op, newInputs, newOutputs, rewriter);
    if (std::is_same_v<OpType, hfusion::SelectOp>) {
      replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                     rewriter, false);
    } else {
      // TODO: set argument enableOverflow = false inside for all non-arithmatic
      // op type
      replaceResultsWithTargetType<ElemType>(
          op->getResults(), newOp->getResults(), rewriter, isInputUnsigned);
    }
    return success();
  }

private:
  template <typename OpElemType>
  bool isSupportOperand(OpType op) const = delete;

  template <>
  bool isSupportOperand<bool>(OpType op) const {
    return false;
  }

  template <>
  bool isSupportOperand<int8_t>(OpType op) const {
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
      // The following ops support i8 type in c310
      // Could compute on i8 directly and no need cast to f16 nor f32
      if (archisAscend950) {
        if constexpr (std::is_same_v<OpType, linalg::AddOp> ||
                      std::is_same_v<OpType, linalg::SubOp> ||
                      std::is_same_v<OpType, linalg::MaxOp> ||
                      std::is_same_v<OpType, linalg::MinOp> ||
                      std::is_same_v<OpType, linalg::SelectOp>) {
          return true;
        }
        if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>)
          if (
              op.getFun() == linalg::BinaryFn::add ||
              op.getFun() == linalg::BinaryFn::sub ||
              op.getFun() == linalg::BinaryFn::max_signed ||
              op.getFun() == linalg::BinaryFn::min_signed ||
              op.getFun() == linalg::BinaryFn::max_unsigned ||
              op.getFun() == linalg::BinaryFn::min_unsigned)
            return true;
      }
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
          hfusion::BinaryFn::ceildivui, hfusion::BinaryFn::mod};
      return !binarySet.contains(func);
    } else if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
      linalg::BinaryFn func = binOp.getFun();
      // I8 type for add and sub op need cast to f16 and no need cast to f32
      auto firstInputType = binOp.getInputs()[0].getType();
      auto secondInputType = binOp.getInputs()[1].getType();
      const bool isI8Type =
          (isI8ElemType(firstInputType) && isI8ElemType(secondInputType));
      const bool isAddOrSubOp =
          (func == linalg::BinaryFn::add || func == linalg::BinaryFn::sub);
      if (isI8Type && isAddOrSubOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << " I8 type for add and sub op need cast to f16 \n");
        return true;
      }
      // should compute on f32 for high precision
      static DenseSet<linalg::BinaryFn> binarySet = {
          linalg::BinaryFn::mul,
          linalg::BinaryFn::div_unsigned,
          linalg::BinaryFn::div,
          linalg::BinaryFn::add,
          linalg::BinaryFn::sub,
      };
      return !binarySet.contains(func);
    }
    return true;
  }

  bool shoudComputeByF32(OpType op) const {
    if constexpr (std::is_same_v<OpType, hfusion::ElemwiseBinaryOp>) {
      auto binOp = cast<hfusion::ElemwiseBinaryOp>(op);
      hfusion::BinaryFn func = binOp.getFun();
      static DenseSet<hfusion::BinaryFn> binarySet = {hfusion::BinaryFn::mod};
      return binarySet.contains(func);
    } else if constexpr (std::is_same_v<OpType, linalg::ElemwiseBinaryOp>) {
      auto binOp = cast<linalg::ElemwiseBinaryOp>(op);
      linalg::BinaryFn func = binOp.getFun();
      // 910B: should compute on f32
      static DenseSet<linalg::BinaryFn> binarySet = {
          linalg::BinaryFn::mul,
          linalg::BinaryFn::add,
          linalg::BinaryFn::sub,
      };
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
        op.getLoc(), TypeRange{newOutputs}, newInputs[0], op.getCumDims(),
        op.getReverse());
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

struct ReduceNormalize910_95 : public OpRewritePattern<linalg::ReduceOp> {
public:
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    bool archIs950 = hacc::utils::isAscend950(moduleOp);
    if (!archIs950) {
      return failure();
    }
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    // do not cast xor example
    auto &region = op.getRegion();
    if (llvm::any_of(region.getOps(),
                     [](Operation &op) { return isa<arith::XOrIOp>(&op); })) {
      return failure();
    }
    // do not cast reduce_with_index
    if (op->hasAttr("reduce_mode")) {
      return failure();
    }
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> inits = op.getInits();
    if (!hasI8ElemType(inputs) && !hasI8ElemType(inits)) {
      return failure();
    }
    MLIRContext *ctx = rewriter.getContext();
    Type i16Type = IntegerType::get(ctx, 16);

    Block &body = op.getCombiner().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    Operation *bodyOp = yieldOp.getValues()[0].getDefiningOp();

    bool isInputUnsigned = false;
    if (isa<arith::MaxUIOp, arith::MinUIOp>(bodyOp)) {
      isInputUnsigned = true;
    }

    SmallVector<Value> newInputs =
        normalizeSrcToI16Type<int8_t>(rewriter, inputs, isInputUnsigned);
    SmallVector<Value> newInits =
        normalizeSrcToI16Type<int8_t>(rewriter, inits, isInputUnsigned);
    Operation *newOp = createNewReduceI16Op(op, rewriter, rewriter.getI8Type(),
                                            i16Type, newInputs, newInits);
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter);

    return success();
  }

private:
  template <typename srcType>
  SmallVector<Value> normalizeSrcToI16Type(PatternRewriter &rewriter,
                                           const SmallVector<Value> &values,
                                           bool isInputUnsigned) const {
    SmallVector<Value> result;
    result.reserve(values.size());
    for (Value v : values) {
      if (!isElemType<srcType>(v.getType())) {
        result.push_back(v);
        continue;
      }
      Type dstType = rewriter.getI16Type();
      Value castResult =
          isInputUnsigned ? castTo(rewriter, v, dstType, TypeFn::cast_unsigned)
                          : castTo(rewriter, v, dstType);
      result.push_back(castResult);
    }
    return result;
  }

  static Operation *mapReduceBodyOpToI16(PatternRewriter &rewriter,
                                         Location loc, Operation *bodyOp,
                                         Type srcType, IRMapping &mapper) {
    if (isa<linalg::YieldOp>(bodyOp)) {
      return rewriter.clone(*bodyOp, mapper);
    }
    // only support binary arith ops here
    assert(bodyOp->getNumOperands() == 2 && "only support binary arith op");
    Value oldLhs = bodyOp->getOperand(0);
    Value oldRhs = bodyOp->getOperand(1);
    Value newLhs = mapper.lookupOrNull(oldLhs);
    Value newRhs = mapper.lookupOrNull(oldRhs);

    if (!newLhs || !newRhs) {
      return nullptr;
    }

    Type lhsType = newLhs.getType();
    Type rhsType = newRhs.getType();
    if (isa<IntegerType>(lhsType) && isa<IntegerType>(rhsType) &&
        lhsType.getIntOrFloatBitWidth() == 16 &&
        rhsType.getIntOrFloatBitWidth() == 16) {
      if (isa<arith::AddIOp>(bodyOp))
        return rewriter.create<arith::AddIOp>(loc, lhsType, newLhs, newRhs);
      // max/min signed/unsigned
      if (isa<arith::MaxSIOp>(bodyOp))
        return rewriter.create<arith::MaxSIOp>(loc, lhsType, newLhs, newRhs);
      if (isa<arith::MaxUIOp>(bodyOp))
        return rewriter.create<arith::MaxUIOp>(loc, lhsType, newLhs, newRhs);
      if (isa<arith::MinSIOp>(bodyOp))
        return rewriter.create<arith::MinSIOp>(loc, lhsType, newLhs, newRhs);
      if (isa<arith::MinUIOp>(bodyOp))
        return rewriter.create<arith::MinUIOp>(loc, lhsType, newLhs, newRhs);
      // if it is some other integer op, fall back to cloning (with mapping).
      return rewriter.clone(*bodyOp, mapper);
    }

    // for all other cases just clone with the mapper.
    return rewriter.clone(*bodyOp, mapper);
  }
  // create the new reduction op with i16 inputs
  static Operation *createNewReduceI16Op(linalg::ReduceOp op,
                                         PatternRewriter &rewriter,
                                         Type srcType, Type targetType,
                                         SmallVector<Value> &newInputs,
                                         SmallVector<Value> &newInits) {
    IRMapping mapper;
    for (const auto &[idx, operand] : llvm::enumerate(op.getInputs())) {
      mapper.map(operand, newInputs[idx]);
    }
    for (const auto &[idx, operand] : llvm::enumerate(op.getInits())) {
      mapper.map(operand, newInits[idx]);
    }

    Operation *newOp = rewriter.cloneWithoutRegions(*op, mapper);
    for (const auto &[idx, res] : llvm::enumerate(op->getResults())) {
      ShapedType shapedType = dyn_cast_or_null<ShapedType>(res.getType());
      bool isSrcType = shapedType && isI8ElemType(shapedType);
      if (!shapedType || !isSrcType) {
        continue;
      }
      auto newShapedType = shapedType.clone(targetType);
      newOp->getResult(idx).setType(newShapedType);
    }
    Region &newRegion = newOp->getRegions().front();
    Block *newBlock = rewriter.createBlock(&newRegion);
    rewriter.setInsertionPointToStart(newBlock);

    Block *block = &op.getRegion().front();
    for (BlockArgument bbArg : block->getArguments()) {
      Type argType = bbArg.getType();
      bool isSrcType = argType.isInteger(8);
      Type newArgType = (isSrcType ? targetType : argType);
      mapper.map(bbArg, newBlock->addArgument(newArgType, bbArg.getLoc()));
    }

    Location loc = newRegion.getLoc();
    for (Operation &bodyOp : *block) {
      Operation *newBodyOp =
          mapReduceBodyOpToI16(rewriter, loc, &bodyOp, srcType, mapper);
      if (newBodyOp) {
        mapper.map(bodyOp.getResults(), newBodyOp->getResults());
      } else {
        Operation *cloned = rewriter.clone(bodyOp, mapper);
        mapper.map(bodyOp.getResults(), cloned->getResults());
      }
    }
    rewriter.setInsertionPointAfter(newOp);
    return newOp;
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
    assert(inputs.size() == 2);
    if (!hasI1ElemType(inputs) && !hasI1ElemType(inits))
      return failure();

    auto newInputs =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inputs);
    auto newInits =
        normalizeSrcToTargetType<bool, Float16Type>(rewriter, inits);
    Operation *newOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        op.getLoc(), TypeRange{newInits[0].getType(), newInits[1].getType()},
        newInputs, newInits, op.getReduceKindAttr(), op.getUnsignedSrcAttr(),
        op.getTieBreakLeftAttr(), op.getDimensionsAttr());
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

    bool unsignedSrc = op.getUnsignedSrc();
    auto newInputs = normalizeSrcToTargetType<int8_t, Float16Type>(
        rewriter, inputs, unsignedSrc);
    auto newInits = normalizeSrcToTargetType<int8_t, Float16Type>(
        rewriter, inits, unsignedSrc);
    Operation *newOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        op.getLoc(), TypeRange{newInits[0].getType(), newInits[1].getType()},
        newInputs, newInits, op.getReduceKindAttr(), op.getUnsignedSrcAttr(),
        op.getTieBreakLeftAttr(), op.getDimensionsAttr());
    replaceI8ResultsWithTargetType(op->getResults(), newOp->getResults(),
                                   rewriter);
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
        op.getLoc(), TypeRange{newOutputs}, newInputs[0], op.getCumDims(),
        op.getReverse());
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
    auto *resOp = createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                 linalg::BinaryFnAttr>(
        rewriter, loc, linalg::BinaryFn::mul, ValueRange{vec, negOne}, empty);

    // Because vsubs is not a supported hardware instruction,
    // Then vsubs(s, v) = vadds(s, vmuls(-1, v)). This computation can be
    // simplified into vmuls(-1, v), when scalar s equals to zero. Usually we
    // can add SimplifyOps Pass after Normalize at the end of `preProcess`
    // function in HFusionPipelines pass, however this may cause some errors on
    // A2/A3 platform. So this simplify process is taken here to solve this
    // reduntant instruction in simd-vf of A5 platform only.
    // TODO: Pls check errors from pipeline on A2/A3, and checkout detail info
    // by issue #74, on gitcode of A2/A3.
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!(hacc::utils::isAscend950(moduleOp) && isConstantZero(inputs[0]))) {
      resOp = createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                             linalg::BinaryFnAttr>(
          rewriter, loc, linalg::BinaryFn::add,
          ValueRange{inputs[0], resOp->getResult(0)}, op.getDpsInits()[0]);
    }

    rewriter.replaceAllUsesWith(op->getResults(), resOp->getResults());
    rewriter.eraseOp(op);
    return success();
  }
  bool isConstantZero(Value val) const {
    if (auto constOp =
            dyn_cast_or_null<arith::ConstantOp>(val.getDefiningOp())) {
      Attribute attr = constOp.getValue();
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr))
        return intAttr.getValue().isZero();
      if (auto floatAttr = mlir::dyn_cast<FloatAttr>(attr))
        return floatAttr.getValue().isZero();
    }
    return false;
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
    hfusion::TypeFn cast_integer_type =
        (fun == hfusion::BinaryFn::shrui || fun == hfusion::BinaryFn::shli)
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
    auto resOfI8 =
        hfusion::castTo(rewriter, resOfI16, srcElemType, roundMode,
                        std::nullopt, true, false, cast_integer_type);

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

struct NormalizeI8Transpose : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::TransposeOp op,
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
    Value newInput = hfusion::castTo(rewriter, input, rewriter.getI16Type(),
                                     hfusion::RoundMode::TRUNC);
    Value newInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, init, rewriter.getI16Type());
    Value newTransOp = rewriter
                           .create<linalg::TransposeOp>(loc, newInput, newInit,
                                                        op.getPermutation())
                           ->getResult(0);
    Value newResult =
        hfusion::castTo(rewriter, newTransOp, rewriter.getI8Type(),
                        hfusion::RoundMode::TRUNC, init,
                        /* enableOverflow = */ false);
    rewriter.replaceAllUsesWith(op->getResult(0), newResult);
    rewriter.eraseOp(op);
    return success();
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

/// remove linalg.fill Ops that are fed into reduce_with_index as inits
/// becuase everything is handle in template functions for regbase, and
/// every valid reduce_with_index gets lowered to template functions, unlike
/// membase that some lowers to scalar loops
/// For example:
/// %0 = linalg.fill -> tensor<37x3xf32>
/// %1 = linalg.fill -> tensor<37x3xi32>
/// %idx = linalg.fill -> tensor<37x5x3xi32>
/// %2:2 = hfusion.reduce_with_index <max> ins(%data,%idx)
/// outs(%0, %1 :tensor<37x3xf32>, tensor<37x3xi32>) dimensions = [1]
///
/// becomes
///
/// %0 = tensor.empty() -> tensor<37x3xf32>
/// %1 = tensor.empty() -> tensor<37x3xi32>
/// %idx = tensor.empty() -> tensor<37x5x3xi32>
/// %2:2 = hfusion.reduce_with_index <max> ins(%data,%idx)
/// outs(%0, %1 :tensor<37x3xf32>, tensor<37x3xi32>) dimensions = [1]
struct NormalizeReduceWithIndexInitsAndInputs
    : public OpRewritePattern<hfusion::ReduceWithIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool changed = false;

    SmallVector<Value> newInits;
    newInits.reserve(op.getNumDpsInits());
    for (Value init : op.getInits()) {
      if (init.getDefiningOp<tensor::EmptyOp>()) {
        newInits.push_back(init);
        continue;
      }

      auto initTy = dyn_cast<RankedTensorType>(init.getType());
      if (!initTy || !initTy.hasStaticShape())
        return failure();

      // Replace with tensor.empty of the same static shape
      Value empty = rewriter.create<tensor::EmptyOp>(loc, initTy.getShape(),
                                                     initTy.getElementType());
      newInits.push_back(empty);
      changed = true;
    }

    SmallVector<Value> newInputs(op.getInputs().begin(), op.getInputs().end());
    Value indexInput = newInputs[1];

    if (!indexInput.getDefiningOp<tensor::EmptyOp>() &&
        !isa<BlockArgument>(indexInput)) {
      auto ty = dyn_cast<RankedTensorType>(indexInput.getType());
      if (!ty || !ty.hasStaticShape())
        return failure();

      Value empty = rewriter.create<tensor::EmptyOp>(loc, ty.getShape(),
                                                     ty.getElementType());
      newInputs[1] = empty;
      changed = true;
    }

    if (!changed)
      return failure();

    auto newOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        loc, op->getResultTypes(), newInputs, newInits, op.getReduceKindAttr(),
        op.getUnsignedSrcAttr(), op.getTieBreakLeftAttr(),
        op.getDimensionsAttr());

    rewriter.replaceOp(op, newOp->getResults());
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

// helper function to cast i64 index to i32 which we support
struct NormalizeReduceIndexToI32
    : public OpRewritePattern<hfusion::ReduceWithIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    // Basic checks
    if (op->getNumResults() < 2)
      return rewriter.notifyMatchFailure(op, "expects at least two results");

    Location loc = op->getLoc();
    MLIRContext *ctx = op->getContext();
    Value oldIndexVal = op->getResult(1);
    auto oldIndexTT = mlir::dyn_cast<RankedTensorType>(oldIndexVal.getType());
    if (!oldIndexTT)
      return rewriter.notifyMatchFailure(
          op, "index result is not RankedTensorType");
    auto oldElemTy = mlir::dyn_cast<IntegerType>(oldIndexTT.getElementType());
    if (!oldElemTy)
      return rewriter.notifyMatchFailure(op, "index element is not integer");

    // if already i32, nothing to do
    if (oldElemTy.getWidth() == 32)
      return rewriter.notifyMatchFailure(op, "index already i32");

    // target i32 element type and new index RankedTensorType
    IntegerType i32Ty = IntegerType::get(ctx, 32);
    RankedTensorType newIndexTT =
        RankedTensorType::get(oldIndexTT.getShape(), i32Ty);

    // build new inputs with i32 type for index
    SmallVector<Value, 8> newInputs;
    newInputs.reserve(op.getInputs().size());
    for (Value in : op.getInputs()) {
      newInputs.push_back(makeI32ValueFor(rewriter, loc, in, oldElemTy, i32Ty));
    }

    // build new inputs with i32 type for index
    SmallVector<Value, 4> newInits;
    newInits.reserve(op.getInits().size());
    for (Value init : op.getInits()) {
      newInits.push_back(
          makeI32ValueFor(rewriter, loc, init, oldElemTy, i32Ty));
    }

    // new result types: keep first (reduced value) unchanged, second becomes
    // tensor<...xi32>
    Type valueResultTy = op->getResult(0).getType();
    SmallVector<Type, 2> newResultTypes = {valueResultTy, newIndexTT};

    // create the new reduce_with_index op with i32 index results and i32
    // inputs/inits
    auto newOp = rewriter.create<hfusion::ReduceWithIndexOp>(
        loc, ArrayRef<Type>(newResultTypes), ArrayRef<Value>(newInputs),
        ArrayRef<Value>(newInits), op.getReduceKindAttr(),
        op.getUnsignedSrcAttr(), op.getTieBreakLeftAttr(),
        op.getDimensionsAttr());

    Value newIndexVal = newOp->getResult(1);
    Value replacedIndexVal;
    if (oldElemTy.getWidth() != 32) {
      // cast back to original index element type for consumers
      replacedIndexVal = hfusion::castTo(rewriter, newIndexVal, oldElemTy);
    } else {
      replacedIndexVal = newIndexVal;
    }

    op->getResult(0).replaceAllUsesWith(newOp->getResult(0));
    op->getResult(1).replaceAllUsesWith(replacedIndexVal);
    rewriter.eraseOp(op);
    return success();
  }

private:
  // helper function to produce an i32-typed Value for a tensor Value that
  // currently has oldElemTy. If the defining op is tensor.empty, synthesize a
  // new tensor.empty with i32 element type preserving dynamic sizes. Otherwise,
  // use hfusion::castTo to cast the tensor's element type.
  static Value makeI32ValueFor(RewriterBase &rewriter, Location loc, Value val,
                               Type oldElemTy, Type i32Ty) {
    // If not a ranked tensor or element doesn't match oldElemTy, just return
    // val
    auto rt = mlir::dyn_cast<RankedTensorType>(val.getType());
    if (!rt)
      return val;
    auto elem = mlir::dyn_cast<IntegerType>(rt.getElementType());
    if (!elem || elem != oldElemTy)
      return val;

    // create a new tensor.empty with the same
    // shape but i32 element type
    if (Operation *def = val.getDefiningOp()) {
      if (auto emptyOp = dyn_cast<tensor::EmptyOp>(def)) {
        RankedTensorType newRT = RankedTensorType::get(rt.getShape(), i32Ty);
        // collect dynamic sizes (works if emptyOp has them)
        SmallVector<Value, 4> dynSizes;
        for (Value ds : emptyOp.getDynamicSizes())
          dynSizes.push_back(ds);
        return rewriter.create<tensor::EmptyOp>(loc, newRT, dynSizes);
      }
    }

    return hfusion::castTo(rewriter, val, i32Ty);
  }
};
} // namespace mlir::hfusion

namespace NormalizeAtomicOps {

struct SyncBlockLockGuard {
  hivm::CreateSyncBlockLockOp createdLock;
  OpBuilder &builder;
  OpBuilder::InsertionGuard guard;
  SyncBlockLockGuard(OpBuilder &builder, Location loc)
      : builder(builder), guard(builder) {
    Type memrefi64 = MemRefType::get({1}, builder.getI64Type());
    createdLock =
        builder.create<hivm::CreateSyncBlockLockOp>(loc, memrefi64, Value());
    builder.create<hivm::SyncBlockLockOp>(loc, createdLock.getResult());
  }
  ~SyncBlockLockGuard() {
    builder.create<hivm::SyncBlockUnlockOp>(createdLock.getLoc(), createdLock);
  }
};

class StaticSizedBuffer {
  OpBuilder &builder;
  const Location loc;

  const TypedValue<MemRefType> gmMemrefSource;
  memref::SubViewOp definingOp;

public:
  // maybe dynnamic sized Memref
  TypedValue<MemRefType> dBuffer;
  // static sized Memref
  TypedValue<MemRefType> sBuffer;

  StaticSizedBuffer(OpBuilder &_builder, Location _loc,
                    TypedValue<MemRefType> _gmMemrefSource)
      : builder(_builder), loc(_loc), gmMemrefSource(_gmMemrefSource) {
    const Type elemType = gmMemrefSource.getType().getElementType();
    const MemRefType rawMemref =
        MemRefType::get(gmMemrefSource.getType().getShape(), elemType);
    if (gmMemrefSource.getType().hasStaticShape()) {
      dBuffer = sBuffer = builder.create<memref::AllocOp>(loc, rawMemref);
      return;
    }
    definingOp = gmMemrefSource.getDefiningOp<memref::SubViewOp>();
    // dynamically shaped GM should come from a subview of statically shaped GM
    assert(definingOp && definingOp.getSourceType().hasStaticShape() &&
           "dynamically shaped GM should come from a subview of statically "
           "shaped GM");
    sBuffer = builder.create<memref::AllocOp>(
        loc, rawMemref.clone(definingOp.getSourceType().getShape()));
    SmallVector<OpFoldResult> zeroOffsets(gmMemrefSource.getType().getRank(),
                                          builder.getI64IntegerAttr(0));
    dBuffer = builder.create<memref::SubViewOp>(loc, sBuffer, zeroOffsets,
                                                definingOp.getMixedSizes(),
                                                definingOp.getMixedStrides());
  }
  TypedValue<TensorType> toStaticTensor() {
    return builder
        .create<bufferization::ToTensorOp>(
            loc, /*memref:*/ sBuffer, /*restrict:*/ true, /*writable:*/ true)
        .getResult();
  }

  void storeBack(TypedValue<TensorType> tensorSrc) {
    if (dBuffer.getType().hasStaticShape()) {
      builder.create<bufferization::MaterializeInDestinationOp>(
          loc, /*result:*/ Type(), /*source:*/ tensorSrc,
          /*dest:*/ gmMemrefSource, /*restrict:*/ false, /*writable:*/ true);
      return;
    }
    SmallVector<OpFoldResult> zeroOffsets(gmMemrefSource.getType().getRank(),
                                          builder.getI64IntegerAttr(0));
    Value dtensor = builder.create<tensor::ExtractSliceOp>(
        loc, /*source*/ tensorSrc, /*offsets*/ zeroOffsets,
        /*sizes*/ definingOp.getMixedSizes(),
        /*strides*/ definingOp.getMixedStrides());
    builder.create<bufferization::MaterializeInDestinationOp>(
        loc, /*result:*/ Type(), /*source:*/ dtensor,
        /*dest:*/ gmMemrefSource, /*restrict:*/ false, /*writable:*/ true);
  }
};

inline bool isElementFP8(ShapedType st) {
  auto t = dyn_cast<FloatType>(st.getElementType());
  return t && t.getWidth() == 8;
}

struct Elemwise : public OpRewritePattern<hfusion::StoreOp> {
  using OpRewritePattern::OpRewritePattern;
  inline bool isHardwareSupported(Type dtype) const {
    return TypeSwitch<Type, bool>(getElementTypeOrSelf(dtype))
        .Case<Float16Type, Float32Type, BFloat16Type>([](Type) { return true; })
        .Case<IntegerType>([](IntegerType t) {
          return t.getWidth() == 8 || t.getWidth() == 16 || t.getWidth() == 32;
        })
        .Default([](Type) { return false; });
  }

  inline std::optional<std::variant<hfusion::BinaryFn, linalg::BinaryFn>>
  getDecomposedBinFn(hfusion::StoreOp op) const {
    Type elemType = getElementTypeOrSelf(op.getOutputs()[0].getType());
    switch (op.getAtomicKind()) {
    case AtomicKind::AND:
      return hfusion::BinaryFn::vand;
    case AtomicKind::OR:
      return hfusion::BinaryFn::vor;
    case AtomicKind::XOR:
      return hfusion::BinaryFn::vxor;
    case AtomicKind::ADD:
      if (isHardwareSupported(elemType))
        return std::nullopt;
      return linalg::BinaryFn::add;
    case AtomicKind::MAX:
      if (isHardwareSupported(elemType))
        return std::nullopt;
      return linalg::BinaryFn::max_signed;
    case AtomicKind::UMAX:
      return linalg::BinaryFn::max_unsigned;
    case AtomicKind::MIN:
      if (isHardwareSupported(elemType))
        return std::nullopt;
      return linalg::BinaryFn::min_signed;
    case AtomicKind::UMIN:
      return linalg::BinaryFn::min_unsigned;
    default:
      return std::nullopt;
    }
  }

  LogicalResult matchAndRewrite(hfusion::StoreOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto maybeFn = getDecomposedBinFn(op);
    if (!maybeFn.has_value())
      return failure();
    auto getElemwiseBinOpResult =
        [fn /*: variant<BinaryFn, BinaryFn>*/ = maybeFn.value(), &rewriter,
         loc](Value lhsTensor, Value rhsTensor, Value resultTensor) {
          if (std::holds_alternative<hfusion::BinaryFn>(fn)) {
            auto binaryAttr = rewriter.getAttr<hfusion::BinaryFnAttr>(
                std::get<hfusion::BinaryFn>(fn));
            auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
            return rewriter
                .create<hfusion::ElemwiseBinaryOp>(
                    loc, ValueRange{lhsTensor, rhsTensor},
                    ValueRange{resultTensor}, ArrayRef{fnAttr})
                .getResult(0);
          } else {
            auto binaryAttr = rewriter.getAttr<linalg::BinaryFnAttr>(
                std::get<linalg::BinaryFn>(fn));
            auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
            return rewriter
                .create<linalg::ElemwiseBinaryOp>(
                    loc, ValueRange{lhsTensor, rhsTensor},
                    ValueRange{resultTensor}, ArrayRef{fnAttr})
                .getResult(0);
          }
        };

    rewriter.setInsertionPointAfter(op);

    TypedValue<MemRefType> ubMemref = dyn_cast<TypedValue<MemRefType>>(
                               op.getInputs()[0]),
                           gmMemref = dyn_cast<TypedValue<MemRefType>>(
                               op.getOutputs()[0]);
    assert(ubMemref);
    assert(gmMemref);

    // prepare buffer and RHS
    StaticSizedBuffer rhsBuffer(rewriter, loc, gmMemref);
    rewriter.create<memref::CopyOp>(loc, ubMemref, rhsBuffer.dBuffer);

    // Lambda to cast to/from F32 if it's FP8
    // isForward: true => FP8 to FP32; false => FP32 to FP8
    auto castF32IfFP8 = [&rewriter, isFP8 = isElementFP8(gmMemref.getType()),
                         srcType = gmMemref.getType().getElementType(),
                         tarType = rewriter.getF32Type()](
                            Value sourceTensor, bool isForward) -> Value {
      return isFP8
                 ? castTo(rewriter, sourceTensor, isForward ? tarType : srcType)
                 : sourceTensor;
    };

    // compute as f32 if elem type is fp8
    Value rhsTensor = castF32IfFP8(rhsBuffer.toStaticTensor(), true);
    // get LHS in tensor
    StaticSizedBuffer lhsBuffer(rewriter, loc, gmMemref);

    // wrap in lock
    SyncBlockLockGuard _(rewriter, loc);
    // Binary Op
    rewriter.create<memref::CopyOp>(loc, gmMemref, lhsBuffer.dBuffer);
    Value lhsTensor = castF32IfFP8(lhsBuffer.toStaticTensor(), true);
    Value result = castF32IfFP8(
        getElemwiseBinOpResult(lhsTensor, rhsTensor, lhsTensor), false);
    assert(isa<TypedValue<TensorType>>(result));
    // store the result
    lhsBuffer.storeBack(cast<TypedValue<TensorType>>(result));
    rewriter.eraseOp(op);
    return success();
  }
};

struct CAS : public OpRewritePattern<hfusion::AtomicCasOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::AtomicCasOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);
    Location loc = op->getLoc();
    TypedValue<MemRefType> ubComparing = dyn_cast<TypedValue<MemRefType>>(
                               op.getInput()[0]),
                           ubStoring = dyn_cast<TypedValue<MemRefType>>(
                               op.getInput()[1]),
                           gmMemref =
                               dyn_cast<TypedValue<MemRefType>>(op.getDst());
    assert(ubComparing);
    assert(ubStoring);
    assert(gmMemref);
    StaticSizedBuffer comparingBuffer(rewriter, loc, gmMemref);
    StaticSizedBuffer storingBuffer(rewriter, loc, gmMemref);
    StaticSizedBuffer gmValBuffer(rewriter, loc, gmMemref);
    rewriter.create<memref::CopyOp>(loc, ubComparing, comparingBuffer.dBuffer);
    rewriter.create<memref::CopyOp>(loc, ubStoring, storingBuffer.dBuffer);

    SyncBlockLockGuard _(rewriter, loc);
    rewriter.create<memref::CopyOp>(loc, gmMemref, gmValBuffer.dBuffer);
    const TypedValue<TensorType> gmValTensor = gmValBuffer.toStaticTensor();
    // Lambda to cast to/from F32 if it's FP8
    // isForward: true => FP8 to FP32; false => FP32 to FP8
    auto castF32IfFP8 = [&rewriter, isFP8 = isElementFP8(gmValTensor.getType()),
                         srcType = gmValTensor.getType().getElementType(),
                         tarType = rewriter.getF32Type()](
                            Value sourceTensor, bool isForward) -> Value {
      return isFP8
                 ? castTo(rewriter, sourceTensor, isForward ? tarType : srcType)
                 : sourceTensor;
    };

    // compare
    // use f32 instead of fp8 to cmp
    Value cmpResult =
        createCmpOp(rewriter, loc, castF32IfFP8(gmValTensor, true),
                    castF32IfFP8(comparingBuffer.toStaticTensor(), true),
                    CompareFn::veq)
            ->getResult(0);
    // select
    // since intrinsics support selecting fp8, no need for casting
    Value selectedResult =
        rewriter
            .create<hfusion::SelectOp>(
                loc, /*resultTensorTypes*/ gmValTensor.getType(),
                /*inputs*/
                ValueRange{cmpResult, storingBuffer.toStaticTensor(),
                           gmValTensor},
                /*outputs*/ gmValTensor)
            .getResult(0);
    // store the result
    assert(isa<TypedValue<TensorType>>(selectedResult));
    gmValBuffer.storeBack(cast<TypedValue<TensorType>>(selectedResult));
    rewriter.eraseOp(op);
    return success();
  }
};

struct XCHG : public OpRewritePattern<hfusion::AtomicXchgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::AtomicXchgOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);
    Location loc = op->getLoc();
    TypedValue<MemRefType> ubMemref = dyn_cast<TypedValue<MemRefType>>(
                               op.getInput()[0]),
                           gmMemref =
                               dyn_cast<TypedValue<MemRefType>>(op.getDst());
    assert(ubMemref);
    assert(gmMemref);
    StaticSizedBuffer tmpBuffer(rewriter, loc, gmMemref);
    SyncBlockLockGuard _(rewriter, loc);
    rewriter.create<memref::CopyOp>(loc, gmMemref, tmpBuffer.dBuffer);
    rewriter.create<memref::CopyOp>(loc, ubMemref, gmMemref);
    rewriter.create<memref::CopyOp>(loc, tmpBuffer.dBuffer, ubMemref);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace NormalizeAtomicOps

namespace {
// For inputs with non-F16/F32 types, cast to F32 for sorting first, then cast
// back to the original type;
struct NormalizeSortOp : public OpRewritePattern<hfusion::SortOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::SortOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Value input = op.getSrc();
    auto inputType = llvm::dyn_cast<mlir::TensorType>(input.getType());
    mlir::Type elemType = inputType.getElementType();
    auto floatType = llvm::dyn_cast<mlir::FloatType>(elemType);

    // only inputs with non-F16/F32 types will trigger the subsequent rewriting
    if (floatType && (floatType.isF16() || floatType.isF32())) {
      return mlir::failure();
    }

    auto intType = llvm::dyn_cast<mlir::IntegerType>(elemType);

    if (intType && (intType.isInteger(32) || intType.isInteger(64))) {
      return mlir::failure();
    }

    mlir::Type f32Type = mlir::Float32Type::get(rewriter.getContext());

    // convert type of input to f32
    auto castToF32 =
        hfusion::castTo(rewriter, input, f32Type, hfusion::RoundMode::ROUND);

    auto newSortOp = rewriter.create<hfusion::SortOp>(
        op.getLoc(), castToF32.getType(), castToF32, op.getDescending(),
        op.getSortAxis());

    // convert type from f32 to origin type
    auto castBack = hfusion::castTo(rewriter, newSortOp.getResult()[0],
                                    elemType, hfusion::RoundMode::ROUND);

    rewriter.replaceOp(op, castBack);

    return mlir::success();
  }
};
} // namespace

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
  patterns.add<NormalizeToTargetType<bool, linalg::TransposeOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, tensor::ConcatOp>>(ctx);
  patterns.add<NormalizeToTargetType<bool, hfusion::ReduceWithIndexOp>>(ctx);
}

void populateNormalizeI8ToTargetPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  if (archIsRegbased)
    patterns.add<ReduceNormalize910_95>(ctx);
  if (!archIsRegbased) {
    patterns.add<NormalizeToTargetType<int8_t, hfusion::ElemwiseBinaryOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, hfusion::ElemwiseUnaryOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, linalg::ElemwiseBinaryOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, linalg::ElemwiseUnaryOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, hfusion::SelectOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, linalg::ReduceOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, hfusion::InterleaveOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, hfusion::DeinterleaveOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, hfusion::GatherOp>>(ctx);
    patterns.add<NormalizeToTargetType<int8_t, linalg::BroadcastOp>>(ctx);
    patterns.add<NormalizeCumOpI8ToTargetType<hfusion::CumsumOp>>(ctx);
    patterns.add<NormalizeCumOpI8ToTargetType<hfusion::CumprodOp>>(ctx);
  }
  if (archisAscend950) {
    patterns.add<NormalizeToTargetType<int8_t, linalg::ElemwiseBinaryOp>>(ctx);
  }
  // TODO: support regbase i8 template function implementation
  patterns.add<NormalizeToTargetType<int8_t, hfusion::ReduceWithIndexOp>>(ctx);
}

void populateNormalizeF16ToF32Patterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizeF16ToF32Type<linalg::ElemwiseUnaryOp>>(ctx);
  patterns.add<NormalizeF16ToF32Type<hfusion::ElemwiseBinaryOp>>(ctx);
  patterns.add<NormalizeF16ToF32Type<hfusion::ElemwiseUnaryOp>>(ctx);
  patterns.add<NormalizeCumOpF16ToF32Type<hfusion::CumsumOp>>(ctx);
  patterns.add<NormalizeCumOpF16ToF32Type<hfusion::CumprodOp>>(ctx);
}

void populateNormalizeHFusionPatterns(RewritePatternSet &patterns,
                                      bool enableHighPrecision) {
  MLIRContext *ctx = patterns.getContext();
  populateNormalizeF16ToF32Patterns(patterns);
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
  if (!archIsRegbased)
    patterns.add<NormalizeI8I32CmpOp>(ctx);
  patterns.add<NormalizeMulRec>(ctx);
  patterns.add<NormalizeModOp>(ctx);
  if (!archIsRegbased)
    patterns.add<NormalizeCmpToCastOp>(ctx);
  patterns.add<NormalizeNegToMul>(ctx);
  patterns.add<NormalizeDivVSToRec>(ctx);
  patterns.add<NormalizeVPowiToPowf>(ctx);
  patterns.add<NormalizeSubVSToVMulAndVAdd>(ctx);
  patterns.add<NormalizeRSqrtOp>(ctx);
  patterns.add<NormalizeCeilandFloorOp>(ctx);
  patterns.add<NormalizeLogLikeOp>(ctx);
  patterns.add<NormalizeLog1pOp>(ctx);
  patterns.add<NormalizeReduceMinMaxNumFOp>(ctx);
  patterns.add<NormalizeElemwiseMinMaxNumFOp<BinaryFn::maxnumf>>(ctx);
  patterns.add<NormalizeElemwiseMinMaxNumFOp<BinaryFn::minnumf>>(ctx);
  patterns.add<NormalizeExp2Op>(ctx);
  patterns.add<NormalizeExpM1Op>(ctx);
  patterns.add<NormalizeErfOp>(ctx);
  patterns.add<NormalizeBrcCast>(ctx);
  patterns.add<NormalizefillCastToTensorBrc>(ctx);
  patterns.add<NormalizeAnyToF32UnaryRecOp>(ctx);
  patterns.add<NormalizeCastLoweringOp>(ctx);
  patterns.add<NormalizeCmpOp>(ctx);
  patterns.add<NormalizeIsInfOp>(ctx);
  patterns.add<NormalizeIsNanOp>(ctx);
  // For RegBased, there patterns are not needed.
  if (!archIsRegbased) {
    patterns.add<NormalizeXorOp>(ctx);
    // For V300, the instructions of vshl and vshr support
    // u8/s8/u16/s16/u32/s32, so it need not do this job.
    patterns.add<NormalizeShiftI8ToI16>(ctx);
    // tranpose(vgather) is done in template function
    patterns.add<ReduceWithIndexRAHighPerformance>(ctx);
  }
  if (archIsRegbased) {
    patterns.add<NormalizeReduceWithIndexInitsAndInputs>(ctx);
    patterns.add<NormalizeReduceIndexToI32>(ctx);
    patterns.add<NormalizeShiftI8ToI16>(ctx);
  }
  populateNormalizeI8ToTargetPatterns(patterns);
  patterns.add<NormalizeIlogbOp>(ctx);
  patterns.add<NormalizeLdexpOp>(ctx);
  patterns.add<NormalizePowfOp>(ctx);
  patterns.add<NormalizeF16ReduceSum>(ctx);
  patterns.add<ReduceWithIndexRAHighPerformance>(ctx);
  patterns.add<NormalizetruncfExtf>(ctx);
  populateNormalizeScalarLikeHFusionPatterns(patterns);
  populateNormalizeI1ToTargetPatterns(patterns);
  patterns.add<NormalizeCDivandFloorDivIntOp>(ctx);
  patterns.add<NormalizeMulExtOp>(ctx);
  patterns.add<NormalizetruncfBf16>(ctx);
  patterns.add<NormalizeScalarExtension<arith::ExtFOp>>(ctx);
  if (archIsRegbased) {
    patterns.add<NormalizeScalarCastOp>(ctx);
    patterns.add<NormalizeArgMinMaxOp>(ctx);
    patterns.add<NormalizeScalarLikeTensorLinalgBrcOpNonDense>(
        patterns.getContext());
  }
  patterns.add<NormalizeDivSIandDivUIOp>(ctx);
  if (!archIsRegbased)
    patterns.add<NormalizeCmpVne>(ctx);
  if (archisAscend950) {
    patterns.add<NormalizeAtomicOps::Elemwise>(ctx);
    patterns.add<NormalizeAtomicOps::CAS>(ctx);
    patterns.add<NormalizeAtomicOps::XCHG>(ctx);
  }
  patterns.add<NormalizeSortOp>(ctx);
}

namespace {
struct NormalizeHFusionPass : public impl::NormalizeBase<NormalizeHFusionPass> {
  explicit NormalizeHFusionPass(const NormalizeOptions &options)
      : NormalizeBase(options) {};

public:
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation()->getParentOfType<ModuleOp>();
    archIsRegbased = hacc::utils::isRegBasedArch(moduleOp);
    archisAscend950 = hacc::utils::isAscend950(moduleOp);
    archisAscend310B = hacc::utils::isAscend310B(moduleOp);
    archisMembased = hacc::utils::isMemBasedArch(moduleOp);
    RewritePatternSet patterns(&getContext());
    populateNormalizeHFusionPatterns(patterns, enableHighPrecision);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::hfusion::createHFusionNormalizeOpsPass(const NormalizeOptions &options) {
  return std::make_unique<NormalizeHFusionPass>(options);
}
