//===-------- TrigTemplateHelpers.h ---------------------------------------===//
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

#ifndef BISHENGIR_TRANSFORMS_REGBASE_NORMALIZE_UTILS_TRIGTEMPLATEHELPERS_H
#define BISHENGIR_TRANSFORMS_REGBASE_NORMALIZE_UTILS_TRIGTEMPLATEHELPERS_H

#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/regbase/Normalize/Utils/Kinds.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Mutex.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/Threading.h"

#include <array>
#include <cmath>
#include <limits>
#include <optional>

namespace mlir {

// Sin: sin(x) = sign(x) * (-1)^q * sin(y)
// Cos: cos(x) =          (-1)^q * sin(pi/2 - r)
// where q = floor(|x|/pi), r = |x| - q*pi in [0, pi), y = min(r, pi-r).
enum class HighPrecisionTrigMode { Sin, Cos };

// pi split into several exactly representable pieces. During range reduction we
// subtract k * pi one piece at a time, i.e.
//   x - k * pi = x - k * (pi_0 + pi_1 + ...)
// which keeps more precision than multiplying by one rounded pi constant.
inline constexpr std::array<double, 5> kPiApproximations = {
    3.140625,
    0.0009670257568359375,
    6.2771141529083251953125e-7,
    1.21644916362129151821136474609375e-10,
    -1.0290623200529979163359041220560e-13,
};

// atan(pi / 8) is the hand-picked pivot used by the historical HFusion
// normalization. Around this point the transformed argument
//   (x - tan(pi / 8)) / (1 + x * tan(pi / 8))
// stays much smaller than x for inputs near the middle of [0, 1], so the same
// short Taylor series becomes noticeably more accurate.
inline constexpr double kTanPiOver8 = 0.4142135623730950;

// The pi / 2 terms are split into high and low parts:
//   pi / 2 ~= kTanPiOver2High + kTanPiOver2Low.
// They are added in the same staged order as the old rewrite so that small
// rounding differences near the tan poles at z = +/- pi / 2 are preserved.
inline constexpr double kTanPiOver2High = 1.57079637050628662109375;
inline constexpr double kTanPiOver2Low = -0.00000004371139000189375;

// Coefficients for the tan numerator polynomial:
//   numerator(z) = ((C0 * z^2 + C1) * z^2 + C2) * z.
// The final multiplication by z makes the numerator odd, matching tan(-z) =
// -tan(z).
inline constexpr double kTanNumeratorCoeff0 = 0.0698520831551998762793;
inline constexpr double kTanNumeratorCoeff1 = -6.8711573651634203789;
inline constexpr double kTanNumeratorCoeff2 = 61.20362572811089435388;

// Coefficient for the first tan denominator factor:
//   denominator(z) = (z^2 + D0) * (z + pi / 2) * (z - pi / 2).
// The last two factors make the rational approximation grow near the
// mathematical tan poles where cos(z) is zero.
inline constexpr double kTanDenominatorCoeff0 = -24.8048928861126769186219;

// The atan polynomial is evaluated only after clipping to a finite interval.
// This avoids overflow in intermediate products and is still numerically safe
// because atan(x) is already within 1e-4 of pi / 2 when |x| is 10000.
inline constexpr double kAtanClipLowerBound = -10000.0;
inline constexpr double kAtanClipUpperBound = 10000.0;
inline constexpr int kAtanTaylorTermCount = 7;

// The tanh rewrite evaluates exp(2x) after clipping. The bounds keep the
// exponential finite while preserving tanh saturation for larger inputs.
inline constexpr double kTanhClipUpperBound = 8.8;
inline constexpr double kTanhClipLowerBound = -8.8;

// Lookup table for the Payne-Hanek reduction in sin/cos computation.
// Entry (i - 128) stores a 32-bit word representing bits i through i+31
// of the binary fractional expansion of 2/pi.
inline constexpr std::array<uint32_t, 320> kPayneHanekTable = {
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

inline Value createFloatConstant(PatternRewriter &rewriter, Location loc,
                                 Type elementType, double value) {
  return rewriter
      .create<arith::ConstantOp>(loc, elementType,
                                 rewriter.getFloatAttr(elementType, value))
      .getResult();
}

// Creates a scalar i32 constant of the given bit-width.
// E.g. createIntegerConstant(rewriter, loc, 32, 255) => arith.constant 255 : i32
inline Value createIntegerConstant(PatternRewriter &rewriter, Location loc,
                                   unsigned width, int64_t value) {
  return rewriter
      .create<arith::ConstantOp>(loc, rewriter.getIntegerType(width),
                                 rewriter.getIntegerAttr(
                                     rewriter.getIntegerType(width), value))
      .getResult();
}

// Broadcasts an i32 scalar constant into a tensor with the same shape as `like`,
// using dialect-specific fill via `Traits::createFillOp`.
// E.g. createIntegerTensorConstantLike<Traits>(rewriter, loc, t<4x8xf32>, 0xff)
//   => fill<4x8xi32>(scalar 0xff : i32)
template <typename Traits>
inline Value createIntegerTensorConstantLike(PatternRewriter &rewriter,
                                             Location loc, Value like,
                                             int64_t value) {
  Value scalar = createIntegerConstant(rewriter, loc, 32, value);
  Value init =
      utils::createEmptyOpWithTargetElemType(rewriter, loc, like,
                                             rewriter.getI32Type());
  return Traits::createFillOp(rewriter, loc, scalar, init);
}

// Broadcasts an f32 scalar constant into a tensor with the same shape as `like`.
// E.g. createFloatTensorConstantLike<Traits>(rewriter, loc, t<3x4xf32>, 1.0)
//   => fill<3x4xf32>(scalar 1.0 : f32)
template <typename Traits>
inline Value createFloatTensorConstantLike(PatternRewriter &rewriter,
                                           Location loc, Value like,
                                           double value) {
  Value scalar =
      createFloatConstant(rewriter, loc, rewriter.getF32Type(), value);
  Value init =
      utils::createEmptyOpWithTargetElemType(rewriter, loc, like,
                                             rewriter.getF32Type());
  return Traits::createFillOp(rewriter, loc, scalar, init);
}

template <typename Traits>
inline Value createConstantShiftOp(PatternRewriter &rewriter, Location loc,
                                   Value input, int32_t shiftAmount,
                                   ShiftKind kind) {
  Value shift = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(shiftAmount));
  Value init = utils::createEmptyOp(rewriter, loc, input);
  return Traits::createShiftOp(rewriter, loc, input, shift, init, kind);
}

// IEEE 754 float32 bit-field extraction from a tensor of f32 values.
// fields:  sign = bits >> 31,  exponent = (bits >> 23) & 0xff,
//          mantissa = bits & 0x7fffff,  isSpecial = (exponent == 0xff).
// E.g. x = -6.5 = 0xc0d00000 => sign=1, exponent=129, mantissa=0x500000.
struct ExtractFloat32Result {
  Value sign;
  Value isSpecial;
  Value mantissa;
  Value exponent;
};

// Payne-Hanek range reduction result for a batch of f32 inputs.
//   |x| / pi = q + f,  q = floor(|x|/pi),  f in [0, 1)
//   parity = q mod 2,  reducedFraction = f,  r = f * pi in [0, pi)
//   signedResult = sign * (-1)^parity * sin(min(r, pi-r))  (for sin),
//                  (-1)^parity * sin(pi/2 - r)              (for cos).
// E.g. x = -7.3: |x|/pi ≈ 2.32, parity=0, f≈0.32, r≈1.017, sign=1.
struct PayneHanekResult {
  Value parity;
  Value reducedFraction;
  Value sign;
  Value isSpecial;
};

// Returns the total element count in a static-shaped ranked tensor.
// Returns ShapedType::kDynamic if any dimension is dynamic.
// E.g. tensor<3x4x5xf32> => 60, tensor<?x4xf32> => kDynamic.
inline int64_t computeTensorStaticNumel(RankedTensorType tensorType) {
  int64_t numel = 1;
  for (int64_t dim : tensorType.getShape()) {
    if (ShapedType::isDynamic(dim))
      return ShapedType::kDynamic;
    numel *= dim;
  }
  return numel;
}

// Creates or returns an existing memref::GlobalOp from a 1-D tensor constant.
// Thread-safe: only one global with the same name is created.
// E.g. buildTrigTableGlobalFromInitAttr(module, builder, loc, "tbl",
//       dense<[0,1,2,5]> : tensor<4xi32>)
//   => memref.global "private" @tbl : memref<4xi32, #hivm.address_space<gm>>
inline memref::GlobalOp
buildTrigTableGlobalFromInitAttr(ModuleOp module, OpBuilder &builder, Location loc,
                        llvm::StringRef symName, ElementsAttr init,
                        llvm::StringRef visibility = "private",
                        int64_t alignmentBytes = 32,
                        llvm::StringRef memSpaceStr =
                            "#hivm.address_space<gm>") {
  static llvm::sys::SmartMutex<true> globalTableMutex;
  llvm::sys::SmartScopedLock<true> lock(globalTableMutex);

  auto tensorType = dyn_cast<RankedTensorType>(init.getType());
  if (!tensorType || tensorType.getRank() != 1)
    llvm::report_fatal_error("expected rank-1 tensor init attr");

  int64_t size = tensorType.getDimSize(0);
  if (size <= 0 || ShapedType::isDynamic(size))
    llvm::report_fatal_error("table size must be static and positive");

  Type elementType = tensorType.getElementType();
  MLIRContext *context = module.getContext();
  Attribute memSpace = mlir::parseAttribute(memSpaceStr, context);
  if (!memSpace)
    llvm::report_fatal_error("failed to parse memref memory space");

  auto memrefType =
      MemRefType::get({size}, elementType, MemRefLayoutAttrInterface{},
                      memSpace);
  if (auto existing = module.lookupSymbol<memref::GlobalOp>(symName))
    return existing;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  OperationState state(loc, memref::GlobalOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(symName));
  state.addAttribute("sym_visibility", builder.getStringAttr(visibility));
  state.addAttribute("type", TypeAttr::get(memrefType));
  state.addAttribute("constant", builder.getUnitAttr());
  state.addAttribute("alignment", builder.getI64IntegerAttr(alignmentBytes));
  state.addAttribute("initial_value", init);
  return cast<memref::GlobalOp>(builder.create(state));
}

// Loads a pre-existing memref::GlobalOp as a 1-D tensor via
// get_global -> reinterpret_cast -> alloc -> copy -> to_tensor.
// Dialect-agnostic (uses only MLIR core ops).
// E.g. materializeTrigTableAsTensor(rewriter, anchorOp, module, loc, "tbl")
//   => bufferization.to_tensor %local : memref<320xi32> -> tensor<320xi32>
inline FailureOr<Value>
materializeTrigTableAsTensor(PatternRewriter &rewriter, Operation *anchorOp,
                               ModuleOp module, Location loc,
                               StringRef globalName) {
  if (!anchorOp)
    return rewriter.notifyMatchFailure(module, "anchor op is null");

  auto global = module.lookupSymbol<memref::GlobalOp>(globalName);
  if (!global)
    return rewriter.notifyMatchFailure(anchorOp, "memref.global not found");

  auto globalType = dyn_cast<MemRefType>(global.getType());
  if (!globalType || globalType.getRank() != 1)
    return rewriter.notifyMatchFailure(anchorOp, "global is not rank-1 memref");

  int64_t size = globalType.getDimSize(0);
  if (size <= 0 || ShapedType::isDynamic(size))
    return rewriter.notifyMatchFailure(anchorOp, "invalid global table size");

  Type elementType = globalType.getElementType();
  auto tensorType = RankedTensorType::get({size}, elementType);

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(anchorOp);
  Value globalValue =
      rewriter.create<memref::GetGlobalOp>(loc, globalType, globalName);
  OpFoldResult offset = rewriter.getIndexAttr(0);
  OpFoldResult extent = rewriter.getIndexAttr(size);
  OpFoldResult stride = rewriter.getIndexAttr(1);
  auto stridedLayout = StridedLayoutAttr::get(rewriter.getContext(), 0, {1});
  auto stridedType =
      MemRefType::get({size}, elementType, stridedLayout,
                      globalType.getMemorySpace());
  Value reinterpreted = rewriter.create<memref::ReinterpretCastOp>(
      loc, stridedType, globalValue, offset, ArrayRef<OpFoldResult>{extent},
      ArrayRef<OpFoldResult>{stride});
  Value local = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get({size}, elementType));
  rewriter.create<memref::CopyOp>(loc, reinterpreted, local);
  return rewriter
      .create<bufferization::ToTensorOp>(loc, tensorType, local,
                                         /*restrict=*/true, /*writable=*/true)
      .getResult();
}

// Builds a memref::GlobalOp from a compile-time uint32_t array and loads it
// as a 1-D i32 tensor. Global is created once (thread-safe) and reused.
// E.g. emitTrigTableFromUI32ArrayAndLoadAsTensorI32(rewriter, op, mod, loc,
//       "tbl", kPayneHanekTable) => tensor<320xi32>
template <size_t N>
inline FailureOr<Value> emitTrigTableFromUI32ArrayAndLoadAsTensorI32(
    PatternRewriter &rewriter, Operation *anchorOp, ModuleOp module,
    Location loc, llvm::StringRef globalName,
    const std::array<uint32_t, N> &array) {
  llvm::SmallVector<llvm::APInt, N> values;
  for (uint32_t value : array)
    values.emplace_back(32, value, /*isSigned=*/false);
  auto tensorType = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, rewriter.getI32Type());
  auto init = DenseElementsAttr::get(tensorType, values);
  buildTrigTableGlobalFromInitAttr(module, rewriter, loc, globalName, init);
  return materializeTrigTableAsTensor(rewriter, anchorOp, module, loc,
                                      globalName);
}

// Decomposes a tensor of f32 values into their IEEE 754 bit fields,
// extracting sign, exponent, mantissa, and isSpecial as i32 tensors.
// Uses dialect-specific bitcast/shift via `Traits` to stay dialect-agnostic.
// E.g. x = 6.5 = 0x40d00000 => sign=0, exponent=129, mantissa=0x500000.
//      x = -inf = 0xff800000 => sign=1, exponent=255, mantissa=0, isSpecial=t.
template <typename Traits>
inline ExtractFloat32Result extractFloat32BitFields(PatternRewriter &rewriter,
                                           Location loc, Value input) {
  Value one = createIntegerTensorConstantLike<Traits>(rewriter, loc, input, 1);
  Value exponentMask =
      createIntegerTensorConstantLike<Traits>(rewriter, loc, input, 255);
  Value mantissaMask = createIntegerTensorConstantLike<Traits>(
      rewriter, loc, input, 0x7fffff);
  Value hiddenBit = createIntegerTensorConstantLike<Traits>(
      rewriter, loc, input, 0x800000);
  (void)hiddenBit;

  Type intElementType = rewriter.getI32Type();
  auto inputType = cast<ShapedType>(input.getType());
  Type bitcastType = RankedTensorType::get(inputType.getShape(), intElementType);
  Value bits = Traits::createBitcastOp(rewriter, loc, bitcastType, input);

  Value signShifted =
      createConstantShiftOp<Traits>(rewriter, loc, bits, 31,
                                    ShiftKind::RightUnsigned);
  Value sign = Traits::createBinaryOp(rewriter, loc, signShifted, one,
                                      utils::createEmptyOp(rewriter, loc,
                                                           signShifted),
                                      BinaryKind::And);
  Value exponentShifted =
      createConstantShiftOp<Traits>(rewriter, loc, bits, 23,
                                    ShiftKind::RightUnsigned);
  Value exponent =
      Traits::createBinaryOp(rewriter, loc, exponentShifted, exponentMask,
                             utils::createEmptyOp(rewriter, loc,
                                                  exponentShifted),
                             BinaryKind::And);
  Value isSpecial =
      Traits::createCmpOp(rewriter, loc, exponent, exponentMask,
                          CompareKind::EQ);
  Value mantissa =
      Traits::createBinaryOp(rewriter, loc, bits, mantissaMask,
                             utils::createEmptyOp(rewriter, loc, bits),
                             BinaryKind::And);
  return {sign, isSpecial, mantissa, exponent};
}

// Payne-Hanek range reduction for f32 inputs.
//   x = m * 2^(e-23),  m = mantissa with hidden bit,  e = biased exponent.
// Multiplies m by cached limbs of 2/pi, keeping only the top 32 fractional
// bits which encode parity and reducedBits such that:
//   accumulated = parity * 2^31 + reducedBits
//   parity = accumulated >> 31 = floor(x/pi) mod 2
//   reducedFraction = (accumulated & 0x7fffffff) / 2^31 in [0, 1)
// The final reduced angle is r = reducedFraction * pi.
// E.g. x = 7.3: x/pi ≈ 2.32, parity=0, reducedFraction≈0.32, r≈1.017.
template <typename Traits>
inline PayneHanekResult buildPayneHanekReduction(PatternRewriter &rewriter,
                                                 Location loc, Value input,
                                                 Value tableTensor) {
  Value hiddenBit = createIntegerTensorConstantLike<Traits>(
      rewriter, loc, input, 0x800000);
  Value low16Mask = createIntegerTensorConstantLike<Traits>(
      rewriter, loc, input, 0xffff);
  Value exponentBias =
      createIntegerTensorConstantLike<Traits>(rewriter, loc, input, 8);
  Value gatherOffset =
      createIntegerTensorConstantLike<Traits>(rewriter, loc, input, 32);
  Value keep31Bits = createIntegerTensorConstantLike<Traits>(
      rewriter, loc, input, 0x7fffffff);
  Value one = createIntegerTensorConstantLike<Traits>(rewriter, loc, input, 1);

  auto extracted = extractFloat32BitFields<Traits>(rewriter, loc, input);
  Value mantissaWithHiddenBit =
      Traits::createBinaryOp(rewriter, loc, extracted.mantissa, hiddenBit,
                             utils::createEmptyOp(rewriter, loc,
                                                  extracted.mantissa),
                             BinaryKind::Add);

  Value indexHigh =
      Traits::createBinaryOp(rewriter, loc, extracted.exponent, exponentBias,
                             utils::createEmptyOp(rewriter, loc,
                                                  extracted.exponent),
                             BinaryKind::Add);
  Value indexLow =
      Traits::createBinaryOp(rewriter, loc, indexHigh, gatherOffset,
                             utils::createEmptyOp(rewriter, loc, indexHigh),
                             BinaryKind::Add);

  Value twoInvPiHigh = Traits::createGather1DOp(rewriter, loc, tableTensor,
                                                indexHigh);
  Value twoInvPiLow = Traits::createGather1DOp(rewriter, loc, tableTensor,
                                               indexLow);

  Value xHigh =
      createConstantShiftOp<Traits>(rewriter, loc, mantissaWithHiddenBit, 16,
                                    ShiftKind::RightUnsigned);
  Value xLow = Traits::createBinaryOp(
      rewriter, loc, mantissaWithHiddenBit, low16Mask,
      utils::createEmptyOp(rewriter, loc, mantissaWithHiddenBit),
      BinaryKind::And);
  Value highHigh =
      createConstantShiftOp<Traits>(rewriter, loc, twoInvPiHigh, 16,
                                    ShiftKind::RightUnsigned);
  Value highLow = Traits::createBinaryOp(
      rewriter, loc, twoInvPiHigh, low16Mask,
      utils::createEmptyOp(rewriter, loc, twoInvPiHigh), BinaryKind::And);
  Value lowHigh =
      createConstantShiftOp<Traits>(rewriter, loc, twoInvPiLow, 16,
                                    ShiftKind::RightUnsigned);
  Value lowLow = Traits::createBinaryOp(
      rewriter, loc, twoInvPiLow, low16Mask,
      utils::createEmptyOp(rewriter, loc, twoInvPiLow), BinaryKind::And);
  Value product0 = Traits::createBinaryOp(
      rewriter, loc, xLow, lowHigh, utils::createEmptyOp(rewriter, loc, xLow),
      BinaryKind::Mul);
  Value term0 =
      createConstantShiftOp<Traits>(rewriter, loc, product0, 16,
                                    ShiftKind::RightUnsigned);
  Value term1 = Traits::createBinaryOp(
      rewriter, loc, xLow, highLow,
      utils::createEmptyOp(rewriter, loc, xLow), BinaryKind::Mul);
  Value product2 = Traits::createBinaryOp(
      rewriter, loc, xLow, highHigh,
      utils::createEmptyOp(rewriter, loc, xLow), BinaryKind::Mul);
  Value product2Low = Traits::createBinaryOp(
      rewriter, loc, product2, low16Mask,
      utils::createEmptyOp(rewriter, loc, product2), BinaryKind::And);
  Value term2 =
      createConstantShiftOp<Traits>(rewriter, loc, product2Low, 16,
                                    ShiftKind::Left);
  Value product3 = Traits::createBinaryOp(
      rewriter, loc, xHigh, lowLow,
      utils::createEmptyOp(rewriter, loc, xHigh),
      BinaryKind::Mul);
  Value term3 =
      createConstantShiftOp<Traits>(rewriter, loc, product3, 16,
                                    ShiftKind::RightUnsigned);
  Value term4 = Traits::createBinaryOp(
      rewriter, loc, xHigh, lowHigh,
      utils::createEmptyOp(rewriter, loc, xHigh), BinaryKind::Mul);
  Value product5 = Traits::createBinaryOp(
      rewriter, loc, xHigh, highLow,
      utils::createEmptyOp(rewriter, loc, xHigh), BinaryKind::Mul);
  Value product5Low = Traits::createBinaryOp(
      rewriter, loc, product5, low16Mask,
      utils::createEmptyOp(rewriter, loc, product5), BinaryKind::And);
  Value term5 =
      createConstantShiftOp<Traits>(rewriter, loc, product5Low, 16,
                                    ShiftKind::Left);

  Value partial01 = Traits::createBinaryOp(
      rewriter, loc, term0, term1,
      utils::createEmptyOp(rewriter, loc, term0),
      BinaryKind::Add);
  Value partial23 = Traits::createBinaryOp(
      rewriter, loc, term2, term3,
      utils::createEmptyOp(rewriter, loc, term2),
      BinaryKind::Add);
  Value partial45 = Traits::createBinaryOp(
      rewriter, loc, term4, term5,
      utils::createEmptyOp(rewriter, loc, term4),
      BinaryKind::Add);
  Value partial0123 = Traits::createBinaryOp(
      rewriter, loc, partial01, partial23,
      utils::createEmptyOp(rewriter, loc, partial01), BinaryKind::Add);
  Value accumulated = Traits::createBinaryOp(
      rewriter, loc, partial0123, partial45,
      utils::createEmptyOp(rewriter, loc, partial0123), BinaryKind::Add);

  // `accumulated` now contains the leading 32 fractional bits of
  // mantissa * (2 / pi). Interpreting those bits as
  //
  //   accumulated = parity * 2^31 + reducedBits
  //
  // gives:
  //   parity       = floor((x / pi)) mod 2
  //   reducedBits  = fractional remainder in [0, 2^31)
  //
  // The floating remainder is therefore:
  //   reducedFraction = reducedBits / 2^31 in [0, 1),
  // and the actual reduced angle used later is `reducedFraction * pi`.
  Value parity =
      createConstantShiftOp<Traits>(rewriter, loc, accumulated, 31,
                                    ShiftKind::RightUnsigned);
  Value reducedBits = Traits::createBinaryOp(
      rewriter, loc, accumulated, keep31Bits,
      utils::createEmptyOp(rewriter, loc, accumulated), BinaryKind::And);
  Value reducedFloat = Traits::createCastOp(rewriter, loc, reducedBits,
                                            rewriter.getF32Type(),
                                            CastRoundKind::Round);
  Value scale = createFloatConstant(rewriter, loc, rewriter.getF32Type(),
                                    4.656612873077393e-10);
  Value reducedFraction = Traits::createBinaryOp(
      rewriter, loc, reducedFloat, scale,
      utils::createEmptyOp(rewriter, loc, reducedFloat), BinaryKind::Mul);

  return {parity, reducedFraction, extracted.sign, extracted.isSpecial};
}
inline double getFloatingPointMaxForAtan(FloatType floatType) {
  if (floatType.isF32()) {
    return std::pow(2.0, floatType.getWidth() + 30);
  }
  return std::pow(2.0, floatType.getWidth() - 1);
}

inline double getFloatingPointMinForAtan(FloatType floatType) {
  if (floatType.isF32()) {
    return std::pow(2.0, -(static_cast<int>(floatType.getWidth()) + 30));
  }
  return std::pow(2.0, -(static_cast<int>(floatType.getWidth()) - 1));
}

inline SmallVector<double>
getTaylorSeriesCoefficients(TaylerMode taylerMode, int termCount) {
  if (termCount <= 0)
    llvm_unreachable("Taylor expansion term count must be positive");

  SmallVector<double> coefficients;
  coefficients.reserve(termCount);
  switch (taylerMode) {
  case TaylerMode::SIN: {
    // Coefficients for
    //   sin(x) = x - x^3 / 3! + x^5 / 5! - ...
    // returned as [1, -1/3!, 1/5!, ...] so callers can build
    //   x * (c0 + c1 * x^2 + c2 * x^4 + ...).
    coefficients.push_back(1.0);
    double coefficientDenominator = 1.0;
    for (int i = 1; i < termCount; ++i) {
      coefficientDenominator *= -1.0 * (2 * i) * (2 * i + 1);
      coefficients.push_back(1.0 / coefficientDenominator);
    }
    return coefficients;
  }
  case TaylerMode::ATAN: {
    // Coefficients for
    //   atan(x) = x - x^3 / 3 + x^5 / 5 - ...
    // returned in the same odd-polynomial form.
    for (int i = 0; i < termCount; ++i) {
      double sign = i % 2 == 0 ? 1.0 : -1.0;
      coefficients.push_back(sign / static_cast<double>(2 * i + 1));
    }
    return coefficients;
  }
  }
  llvm_unreachable("unsupported TaylerMode");
}

// Computes k = round(x / pi + offset). For cosine, offset = 0.5 because
//   cos(x) = sin(x + pi / 2)
// and (x + pi / 2) / pi = x / pi + 0.5.
template <typename Traits>
Value buildRoundedPiMultiple(PatternRewriter &rewriter, Location loc,
                             Value input, Type roundedElementType,
                             std::optional<double> offset = std::nullopt) {
  Type elementType = getElementTypeOrSelf(input.getType());
  Value empty = utils::createEmptyOp(rewriter, loc, input);

  Value piReciprocal = createFloatConstant(
      rewriter, loc, elementType, 1.0 / static_cast<double>(M_PI));
  Value quotientByPi = Traits::createBinaryOp(
      rewriter, loc, input, piReciprocal, empty, BinaryKind::Mul);

  if (offset.has_value()) {
    Value offsetValue =
        createFloatConstant(rewriter, loc, elementType, *offset);
    quotientByPi = Traits::createBinaryOp(rewriter, loc, quotientByPi,
                                          offsetValue, empty, BinaryKind::Add);
  }

  return Traits::createCastOp(rewriter, loc, quotientByPi, roundedElementType,
                              CastRoundKind::Round);
}

// Builds r = x - k * pi
// by repeatedly subtracting k * pi_i for each split piece of pi.
template <typename Traits>
Value buildRangeReducedTrigInput(
    PatternRewriter &rewriter, Location loc, Value input,
    Value roundedPiMultiple, llvm::ArrayRef<double> piApproximations,
    std::optional<double> offset = std::nullopt) {
  Value empty = utils::createEmptyOp(rewriter, loc, input);
  Type elementType = getElementTypeOrSelf(input.getType());
  Value rangeReducedInput = input;

  for (double piApproximation : piApproximations) {
    Value piApproximationValue =
        createFloatConstant(rewriter, loc, elementType, piApproximation);
    Value scaledPiMultiple = Traits::createBinaryOp(
        rewriter, loc, roundedPiMultiple, piApproximationValue, empty,
        BinaryKind::Mul);
    rangeReducedInput = Traits::createBinaryOp(
        rewriter, loc, rangeReducedInput, scaledPiMultiple, empty,
        BinaryKind::Sub);
  }

  if (!offset.has_value())
    return rangeReducedInput;

  // Optional final shift for formulas such as
  //   cos(x) = sin(x - k * pi + pi / 2).
  Value offsetValue =
      createFloatConstant(rewriter, loc, elementType, *offset);
  return Traits::createBinaryOp(rewriter, loc, rangeReducedInput, offsetValue,
                                empty, BinaryKind::Add);
}

// Evaluates
//   x * (c0 + c1 * x^2 + ... + cn * x^(2n))
// with Horner's method. For sine this becomes
//   x * (1 - x^2 / 3! + x^4 / 5! - ...),
// and for atan it becomes
//   x * (1 - x^2 / 3 + x^4 / 5 - ...).
// Keeping the polynomial in this odd-function form lets callers share the same
// evaluator across sin and atan.
template <typename Traits>
Value buildTaylorApproximation(PatternRewriter &rewriter, Location loc,
                               Value input,
                               llvm::ArrayRef<double> coefficients) {
  if (coefficients.empty())
    llvm_unreachable("Taylor coefficients must not be empty");

  Type elementType = getElementTypeOrSelf(input.getType());
  Value empty = utils::createEmptyOp(rewriter, loc, input);
  Value squaredInput = Traits::createBinaryOp(rewriter, loc, input, input,
                                              empty, BinaryKind::Mul);
  Value approximation = createFloatConstant(rewriter, loc, elementType,
                                            coefficients.back());

  for (int index = static_cast<int>(coefficients.size()) - 2; index >= 0;
       --index) {
    approximation = Traits::createBinaryOp(rewriter, loc, squaredInput,
                                           approximation, empty,
                                           BinaryKind::Mul);
    Value coefficientValue = createFloatConstant(rewriter, loc, elementType,
                                                 coefficients[index]);
    approximation = Traits::createBinaryOp(rewriter, loc, approximation,
                                           coefficientValue, empty,
                                           BinaryKind::Add);
  }

  return Traits::createBinaryOp(rewriter, loc, approximation, input, empty,
                                BinaryKind::Mul);
}

// Elementwise helper for the common polynomial step
//   y = input + constant.
// It materializes `constant` with the same element type as `input`, then uses
// the dialect-specific binary-add operation supplied by `Traits`.
template <typename Traits>
Value addConstant(PatternRewriter &rewriter, Location loc, Value input,
                  double constant) {
  Type elementType = getElementTypeOrSelf(input.getType());
  Value empty = utils::createEmptyOp(rewriter, loc, input);
  Value constantValue =
      createFloatConstant(rewriter, loc, elementType, constant);
  return Traits::createBinaryOp(rewriter, loc, input, constantValue, empty,
                                BinaryKind::Add);
}

// Adds constants to a polynomial and multiplies by the already-computed
// squared input after each addition:
//   result = (((input + c0) * x2 + c1) * x2 + c2) * x2 ...
//
// `multiplyAfterLastAdd = false` leaves off the final multiply, which is useful
// when the caller needs
//   ((input + c0) * x2 + c1)
// and will apply a different final operation afterwards.
template <typename Traits>
Value buildPolynomialFromSquaredInput(PatternRewriter &rewriter, Location loc,
                                      Value squaredInput, Value input,
                                      llvm::ArrayRef<double> coefficients,
                                      bool multiplyAfterLastAdd = true) {
  if (coefficients.empty())
    llvm_unreachable("polynomial coefficients must not be empty");

  Value empty = utils::createEmptyOp(rewriter, loc, input);
  Value result = input;
  for (int index = 0, end = static_cast<int>(coefficients.size());
       index < end; ++index) {
    result = addConstant<Traits>(rewriter, loc, result, coefficients[index]);
    if (multiplyAfterLastAdd || index != end - 1) {
      result = Traits::createBinaryOp(rewriter, loc, result, squaredInput,
                                      empty, BinaryKind::Mul);
    }
  }
  return result;
}

template <typename Traits>
Value buildSinParitySign(PatternRewriter &rewriter, Location loc,
                         Value roundedPiMultiple) {
  Type elementType = getElementTypeOrSelf(roundedPiMultiple.getType());
  Value empty = utils::createEmptyOp(rewriter, loc, roundedPiMultiple);

  // `roundedPiMultiple` is the integer k in x = r + k * pi. The sequence below computes
  //   1 - 2 * (k - 2 * floor(k / 2))
  // which is +1 when k is even and -1 when k is odd, i.e. (-1)^k.
  Value half = createFloatConstant(rewriter, loc, elementType, 0.5);
  Value halfPiMultiple = Traits::createBinaryOp(
      rewriter, loc, roundedPiMultiple, half, empty, BinaryKind::Mul);
  Value flooredHalfPiMultiple =
      Traits::createCastOp(rewriter, loc, halfPiMultiple,
                           rewriter.getF32Type(), CastRoundKind::Floor);

  Value four = createFloatConstant(rewriter, loc, elementType, 4.0);
  Value scaledFlooredHalfPiMultiple = Traits::createBinaryOp(
      rewriter, loc, flooredHalfPiMultiple, four, empty, BinaryKind::Mul);

  Value minusTwo = createFloatConstant(rewriter, loc, elementType, -2.0);
  Value minusTwicePiMultiple = Traits::createBinaryOp(
      rewriter, loc, roundedPiMultiple, minusTwo, empty, BinaryKind::Mul);

  Value parityTerm = Traits::createBinaryOp(
      rewriter, loc, scaledFlooredHalfPiMultiple, minusTwicePiMultiple, empty,
      BinaryKind::Add);
  Value one = createFloatConstant(rewriter, loc, elementType, 1.0);
  return Traits::createBinaryOp(rewriter, loc, parityTerm, one, empty,
                                BinaryKind::Add);
}

// Branch-free clamp to [lowerBound, upperBound]. Prevents overflow in later
// arithmetic while preserving sign (restored separately if needed).
template <typename Traits>
Value buildClampedInput(PatternRewriter &rewriter, Location loc, Value input,
                        double lowerBound, double upperBound) {
  Type elementType = getElementTypeOrSelf(input.getType());
  Value empty = utils::createEmptyOp(rewriter, loc, input);
  Value upperBoundValue =
      createFloatConstant(rewriter, loc, elementType, upperBound);
  Value clampedToUpper = Traits::createBinaryOp(
      rewriter, loc, input, upperBoundValue, empty, BinaryKind::Min);
  Value lowerBoundValue =
      createFloatConstant(rewriter, loc, elementType, lowerBound);
  return Traits::createBinaryOp(rewriter, loc, clampedToUpper, lowerBoundValue,
                                empty, BinaryKind::Max);
}

template <typename Traits>
Value buildAbsValue(PatternRewriter &rewriter, Location loc, Value input) {
  Value empty = utils::createEmptyOp(rewriter, loc, input);
  return Traits::createUnaryOp(rewriter, loc, input, empty, UnaryKind::Abs);
}

// Returns
//   |(x - a) / (1 + a * x)|
// which is the reduced argument from the angle-addition identity
//   atan(x) = atan(a) + atan((x - a) / (1 + a * x))   for x >= a.
// Using abs makes the transformation branch-free: when x < a, the numerator
// flips sign, but the caller later takes the smaller of the direct polynomial
// and the shifted polynomial, which selects the correct branch.
template <typename Traits>
Value buildAbsAtanAngleReducedInput(PatternRewriter &rewriter, Location loc,
                                    Value input, double pivot) {
  Type elementType = getElementTypeOrSelf(input.getType());
  Value empty = utils::createEmptyOp(rewriter, loc, input);
  Value pivotValue = createFloatConstant(rewriter, loc, elementType, pivot);
  Value one = createFloatConstant(rewriter, loc, elementType, 1.0);

  Value scaledInput = Traits::createBinaryOp(rewriter, loc, input, pivotValue,
                                             empty, BinaryKind::Mul);
  Value denominator = Traits::createBinaryOp(rewriter, loc, scaledInput, one,
                                             empty, BinaryKind::Add);
  Value numerator = Traits::createBinaryOp(rewriter, loc, input, pivotValue,
                                           empty, BinaryKind::Sub);
  Value reducedInput = Traits::createBinaryOp(rewriter, loc, numerator,
                                              denominator, empty,
                                              BinaryKind::Div);
  return buildAbsValue<Traits>(rewriter, loc, reducedInput);
}

// Evaluates atan(x) for x in [0, 1] using two branch-free candidates:
//   1. direct Taylor series around 0
//   2. pi / 8 + atan((x - tan(pi / 8)) / (1 + x * tan(pi / 8)))
//
// The second formula is exact on [tan(pi / 8), 1]. For smaller x it produces a
// larger angle because the reduced argument becomes negative and we take abs().
// Taking `min(direct, shifted)` therefore acts like a branch-free selector:
// it keeps the direct series on [0, tan(pi / 8)] and the shifted series on
// [tan(pi / 8), 1].
template <typename Traits>
Value buildAtanUnitIntervalApproximation(PatternRewriter &rewriter,
                                         Location loc,
                                         Value nonNegativeInput) {
  SmallVector<double> atanTaylorCoefficients =
      getTaylorSeriesCoefficients(TaylerMode::ATAN,
                                  kAtanTaylorTermCount);
  Value empty = utils::createEmptyOp(rewriter, loc, nonNegativeInput);
  Type elementType = getElementTypeOrSelf(nonNegativeInput.getType());

  Value directApproximation = buildTaylorApproximation<Traits>(
      rewriter, loc, nonNegativeInput, atanTaylorCoefficients);

  Value piOver8ReducedInput = buildAbsAtanAngleReducedInput<Traits>(
      rewriter, loc, nonNegativeInput, kTanPiOver8);
  Value shiftedApproximation = buildTaylorApproximation<Traits>(
      rewriter, loc, piOver8ReducedInput, atanTaylorCoefficients);
  Value piOver8 =
      createFloatConstant(rewriter, loc, elementType, M_PI / 8.0);
  shiftedApproximation = Traits::createBinaryOp(
      rewriter, loc, shiftedApproximation, piOver8, empty, BinaryKind::Add);

  return Traits::createBinaryOp(rewriter, loc, directApproximation,
                                shiftedApproximation, empty, BinaryKind::Min);
}

// Builds a sign tensor without using comparisons:
//   sign(x) = FP_MAX * x / (FP_MIN + |FP_MAX * x|)
//
// For finite x this evaluates to a value very close to +1 or -1, and for NaN
// it naturally propagates NaN through the final multiplication. The constants
// intentionally match the previous HFusion-only implementation.
template <typename Traits>
Value buildAtanSign(PatternRewriter &rewriter, Location loc, Value input) {
  Type elementType = getElementTypeOrSelf(input.getType());
  auto floatType = cast<FloatType>(elementType);
  Value empty = utils::createEmptyOp(rewriter, loc, input);

  Value fpMax = createFloatConstant(rewriter, loc, elementType,
                                    getFloatingPointMaxForAtan(floatType));
  Value fpMin = createFloatConstant(rewriter, loc, elementType,
                                    getFloatingPointMinForAtan(floatType));
  Value scaledInput =
      Traits::createBinaryOp(rewriter, loc, input, fpMax, empty,
                             BinaryKind::Mul);
  Value denominatorAbs = buildAbsValue<Traits>(rewriter, loc, scaledInput);
  Value denominator = Traits::createBinaryOp(rewriter, loc, denominatorAbs,
                                             fpMin, empty, BinaryKind::Add);
  return Traits::createBinaryOp(rewriter, loc, scaledInput, denominator, empty,
                                BinaryKind::Div);
}

// Full atan normalization for floating-point tensors.
//
// The computation is performed on |x| and the sign is restored afterwards:
//   atan(x) = sign(x) * atan(|x|)
//
// The positive branch uses two exact angle-addition identities:
//   1. atan(x) = pi / 8 + atan((x - tan(pi / 8)) / (1 + x * tan(pi / 8)))
//      which improves accuracy on the upper half of [0, 1].
//   2. atan(x) = pi / 4 + atan((x - 1) / (x + 1))
//      which maps x >= 1 back into [0, 1].
//
// The implementation keeps the rewrite branch-free by computing every
// candidate and taking the smaller angle at each stage:
//   direct(x)      = Taylor on x
//   pi_over_8(x)   = pi / 8 + Taylor(reduced around tan(pi / 8))
//   unit(x)        = min(direct(x), pi_over_8(x))
//   reciprocal(x)  = pi / 4 + unit(|(x - 1) / (x + 1)|)
//   atan(|x|)      = min(unit(x), reciprocal(x))
template <typename Traits>
Value buildAtanApproximation(PatternRewriter &rewriter, Location loc,
                             Value input) {
  Type elementType = getElementTypeOrSelf(input.getType());
  Value empty = utils::createEmptyOp(rewriter, loc, input);

  Value clippedInput =
      buildClampedInput<Traits>(rewriter, loc, input, kAtanClipLowerBound,
                                kAtanClipUpperBound);
  Value absoluteInput = buildAbsValue<Traits>(rewriter, loc, clippedInput);

  Value unitIntervalApproximation = buildAtanUnitIntervalApproximation<Traits>(
      rewriter, loc, absoluteInput);

  Value reciprocalReducedInput = buildAbsAtanAngleReducedInput<Traits>(
      rewriter, loc, absoluteInput, 1.0);
  Value reciprocalApproximation = buildAtanUnitIntervalApproximation<Traits>(
      rewriter, loc, reciprocalReducedInput);
  Value piOver4 =
      createFloatConstant(rewriter, loc, elementType, M_PI / 4.0);
  reciprocalApproximation = Traits::createBinaryOp(
      rewriter, loc, reciprocalApproximation, piOver4, empty,
      BinaryKind::Add);

  Value magnitudeApproximation = Traits::createBinaryOp(
      rewriter, loc, unitIntervalApproximation, reciprocalApproximation, empty,
      BinaryKind::Min);
  Value sign = buildAtanSign<Traits>(rewriter, loc, input);
  return Traits::createBinaryOp(rewriter, loc, magnitudeApproximation, sign,
                                empty, BinaryKind::Mul);
}

template <typename Traits>
Value buildTanApproximation(PatternRewriter &rewriter, Location loc,
                            Value input) {
  Type elementType = getElementTypeOrSelf(input.getType());
  Value empty = utils::createEmptyOp(rewriter, loc, input);

  // Because tan has period pi, first choose the nearest multiple of pi:
  //   k = round(x / pi).
  // The later reduced argument z = x - k * pi is close to 0, where the
  // rational approximation is accurate.
  Value roundedPiMultiple =
      buildRoundedPiMultiple<Traits>(rewriter, loc, input,
                                     rewriter.getF32Type());

  // Start the range reduction with the two largest split pieces of pi:
  //   z12 = x - k * (pi_0 + pi_1).
  // The staged form keeps more low bits than subtracting k * pi in one rounded
  // multiply.
  llvm::ArrayRef<double> piPart1(kPiApproximations.data(), 2);
  Value reducedAfterPart1 = buildRangeReducedTrigInput<Traits>(
      rewriter, loc, input, roundedPiMultiple, piPart1);

  // Build the two pole-distance factors. Conceptually these are
  //   z + pi / 2
  //   z - pi / 2
  // and they appear in the denominator because tan(z) has poles at
  // z = +/- pi / 2.
  //
  // The high part of pi / 2 is applied before subtracting the smaller pi
  // pieces, matching the old operation order to preserve precision behavior.
  Value denominatorFactor1 = addConstant<Traits>(
      rewriter, loc, reducedAfterPart1, kTanPiOver2High);
  Value denominatorFactor2 = addConstant<Traits>(
      rewriter, loc, reducedAfterPart1, -kTanPiOver2High);

  // Finish the middle piece of both pole factors, then add the low pi / 2 split
  // so that
  //   factor1 ~= x - k * pi + pi / 2
  //   factor2 ~= x - k * pi - pi / 2.
  llvm::ArrayRef<double> piPart2(kPiApproximations.data() + 2, 1);
  denominatorFactor1 = buildRangeReducedTrigInput<Traits>(
      rewriter, loc, denominatorFactor1, roundedPiMultiple, piPart2);
  denominatorFactor2 = buildRangeReducedTrigInput<Traits>(
      rewriter, loc, denominatorFactor2, roundedPiMultiple, piPart2);
  denominatorFactor1 = addConstant<Traits>(
      rewriter, loc, denominatorFactor1, kTanPiOver2Low);
  denominatorFactor2 = addConstant<Traits>(
      rewriter, loc, denominatorFactor2, -kTanPiOver2Low);

  // Subtract the final tiny pi pieces from both pole factors. Changing this
  // order can move the last few bits near the poles, so keep it explicit.
  llvm::ArrayRef<double> piPart3(kPiApproximations.data() + 3, 2);
  denominatorFactor1 = buildRangeReducedTrigInput<Traits>(
      rewriter, loc, denominatorFactor1, roundedPiMultiple, piPart3);
  denominatorFactor2 = buildRangeReducedTrigInput<Traits>(
      rewriter, loc, denominatorFactor2, roundedPiMultiple, piPart3);

  // Compute the fully reduced argument and its square:
  //   z  = x - k * (pi_0 + pi_1 + pi_2 + pi_3 + pi_4)
  //   z2 = z * z.
  Value reducedInput = buildRangeReducedTrigInput<Traits>(
      rewriter, loc, reducedAfterPart1, roundedPiMultiple,
      llvm::ArrayRef<double>(kPiApproximations.data() + 2, 3));
  Value squaredInput = Traits::createBinaryOp(
      rewriter, loc, reducedInput, reducedInput, empty, BinaryKind::Mul);

  // Numerator polynomial in Horner form:
  //   numerator = ((C0 * z2 + C1) * z2 + C2) * z.
  Value numeratorSeed = Traits::createBinaryOp(
      rewriter, loc, squaredInput,
      createFloatConstant(rewriter, loc, elementType, kTanNumeratorCoeff0),
      empty, BinaryKind::Mul);
  Value numerator = buildPolynomialFromSquaredInput<Traits>(
      rewriter, loc, squaredInput, numeratorSeed, {kTanNumeratorCoeff1});
  numerator = addConstant<Traits>(rewriter, loc, numerator,
                                  kTanNumeratorCoeff2);
  numerator = Traits::createBinaryOp(rewriter, loc, numerator, reducedInput,
                                     empty, BinaryKind::Mul);

  // Denominator polynomial/factors:
  //   denominator = (z2 + D0) * (z + pi / 2) * (z - pi / 2).
  // Dividing the odd numerator by this even denominator gives the final tan
  // approximation.
  Value denominator = addConstant<Traits>(rewriter, loc, squaredInput,
                                          kTanDenominatorCoeff0);
  denominator = Traits::createBinaryOp(rewriter, loc, denominator,
                                       denominatorFactor1, empty,
                                       BinaryKind::Mul);
  denominator = Traits::createBinaryOp(rewriter, loc, denominator,
                                       denominatorFactor2, empty,
                                       BinaryKind::Mul);
  return Traits::createBinaryOp(rewriter, loc, numerator, denominator, empty,
                                BinaryKind::Div);
}

/// Normalize tanh(x) as:
///   tanh(x) = (exp(2x') - 1) / (exp(2x') + 1)
/// where x' = clamp(x, [-8.8, 8.8]). Clipping avoids overflow in exp(2x) while
/// keeping the tanh error below the tolerance.
template <typename Traits>
Value buildTanhApproximation(PatternRewriter &rewriter, Location loc,
                             Value input) {
  Type elementType = getElementTypeOrSelf(input.getType());

  // Keep the same operation order as the original HFusion-only rewrite.
  Value clippedInput =
      buildClampedInput<Traits>(rewriter, loc, input, kTanhClipLowerBound,
                                kTanhClipUpperBound);

  Value two = createFloatConstant(rewriter, loc, elementType, 2.0);
  Value expInit = utils::createEmptyOp(rewriter, loc, input);
  Value doubledInput = Traits::createBinaryOp(
      rewriter, loc, clippedInput, two, expInit, BinaryKind::Mul);
  Value exponential = Traits::createUnaryOp(rewriter, loc, doubledInput,
                                            expInit, UnaryKind::Exp);

  Value minusOne = createFloatConstant(rewriter, loc, elementType, -1.0);
  Value numeratorInit = utils::createEmptyOp(rewriter, loc, input);
  Value numerator = Traits::createBinaryOp(
      rewriter, loc, exponential, minusOne, numeratorInit, BinaryKind::Add);

  Value one = createFloatConstant(rewriter, loc, elementType, 1.0);
  Value denominatorInit = utils::createEmptyOp(rewriter, loc, input);
  Value denominator = Traits::createBinaryOp(
      rewriter, loc, exponential, one, denominatorInit, BinaryKind::Add);

  return Traits::createBinaryOp(rewriter, loc, numerator, denominator,
                                numeratorInit, BinaryKind::Div);
}

/// Shared high-precision sin/cos normalization (HFusion & HIVM).
///
/// Reduces |x| modulo pi via Payne-Hanek:
///   |x|/pi = q + f, q = floor(|x|/pi), f in [0,1), r = f*pi in [0,pi).
/// Reconstructs:
///   sin(x) = sign(x) * (-1)^q * sin(min(r, pi-r))
///   cos(x) =          (-1)^q * sin(pi/2 - r)
/// Special inputs are masked to quiet NaN.
///
/// E.g. x = -5.2: |x|/pi = 1+f, q=1, r=f*pi, sign(x)=-1, (-1)^q=-1 => +sin(r).
template <typename Traits, typename OpType>
FailureOr<Value>
buildHighPrecisionSinOrCos(PatternRewriter &rewriter, OpType op, Value input,
                           HighPrecisionTrigMode mode) {
  Location loc = op.getLoc();
  ModuleOp module = op->template getParentOfType<ModuleOp>();
  if (!module)
    return failure();

  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || !inputType.getElementType().isF32())
    return failure();

  auto tableOr = emitTrigTableFromUI32ArrayAndLoadAsTensorI32<320>(
      rewriter, op, module, loc, "tbl", kPayneHanekTable);
  if (failed(tableOr))
    return failure();
  Value tableTensor = *tableOr;

  Value flattenedInput = input;
  RankedTensorType flattenedType = inputType;
  SmallVector<ReassociationIndices, 1> reassociation;
  bool reshaped = false;
  if (inputType.getRank() > 1) {
    int64_t numel = computeTensorStaticNumel(inputType);
    if (numel == ShapedType::kDynamic)
      return failure();
    flattenedType = RankedTensorType::get({numel}, rewriter.getF32Type());
    reassociation.push_back(
        llvm::to_vector(llvm::seq<int64_t>(0, inputType.getRank())));
    flattenedInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, flattenedType, input, reassociation);
    reshaped = true;
  }

  auto payneHanek = buildPayneHanekReduction<Traits>(
      rewriter, loc, flattenedInput, tableTensor);
  Value parity = payneHanek.parity;
  Value reducedFraction = payneHanek.reducedFraction;
  Value inputSign = payneHanek.sign;
  Value isSpecial = payneHanek.isSpecial;

  Value pi = createFloatConstant(rewriter, loc, rewriter.getF32Type(),
                                 3.1415927f);
  Value piOverTwo = createFloatConstant(rewriter, loc, rewriter.getF32Type(),
                                        1.5707963267948966f);
  Value signBits = parity;
  if (mode == HighPrecisionTrigMode::Sin) {
    // sin(-x) = -sin(x), while the Payne-Hanek parity already accounts for the
    // `+/- pi` half-period flips. XORing the input sign with the parity bit
    // reconstructs the final sign of the sine result.
    signBits = Traits::createBinaryOp(rewriter, loc, parity, inputSign,
                                      utils::createEmptyOp(rewriter, loc, parity),
                                      BinaryKind::Xor);
  }
  Value signMask = createIntegerTensorConstantLike<Traits>(rewriter, loc,
                                                           signBits, 1);
  Value signTag = Traits::createBinaryOp(rewriter, loc, signBits, signMask,
                                         utils::createEmptyOp(rewriter, loc, signBits),
                                         BinaryKind::And);
  Value zeroTag =
      createIntegerTensorConstantLike<Traits>(rewriter, loc, signBits, 0);
  Value signIsPositive =
      Traits::createCmpOp(rewriter, loc, signTag, zeroTag, CompareKind::EQ);
  Value posOne =
      createFloatTensorConstantLike<Traits>(rewriter, loc, reducedFraction, 1.0);
  Value negOne = createFloatTensorConstantLike<Traits>(rewriter, loc,
                                                       reducedFraction, -1.0);
  Value signValue = Traits::createSelectOp(
      rewriter, loc, signIsPositive, posOne, negOne,
      utils::createEmptyOpWithTargetElemType(rewriter, loc, reducedFraction,
                                             rewriter.getF32Type()));

  Value reducedAngle = Traits::createBinaryOp(
      rewriter, loc, reducedFraction, pi,
      utils::createEmptyOp(rewriter, loc, reducedFraction), BinaryKind::Mul);
  Value approximationInput;
  if (mode == HighPrecisionTrigMode::Cos) {
    // cos(r) = sin(pi / 2 - r), with r in [0, pi).
    approximationInput = Traits::createBinaryOp(
        rewriter, loc, piOverTwo, reducedAngle,
        utils::createEmptyOp(rewriter, loc, reducedAngle), BinaryKind::Sub);
  } else {
    // sin(r) is non-negative on [0, pi], so fold the reduced angle into the
    // first quadrant:
    //   y = min(r, pi - r) in [0, pi / 2].
    Value piMinusAngle = Traits::createBinaryOp(
        rewriter, loc, pi, reducedAngle,
        utils::createEmptyOp(rewriter, loc, reducedAngle), BinaryKind::Sub);
    approximationInput = Traits::createBinaryOp(
        rewriter, loc, piMinusAngle, reducedAngle,
        utils::createEmptyOp(rewriter, loc, reducedAngle),
        BinaryKind::MinSigned);
  }

  Value approximation = buildTaylorApproximation<Traits>(
      rewriter, loc, approximationInput,
      getTaylorSeriesCoefficients(TaylerMode::SIN, 5));
  Value signedResult = Traits::createBinaryOp(
      rewriter, loc, approximation, signValue,
      utils::createEmptyOp(rewriter, loc, approximation), BinaryKind::Mul);
  Value qnanTensor =
      createFloatTensorConstantLike<Traits>(rewriter, loc, signedResult,
                                            std::numeric_limits<float>::quiet_NaN());
  Value maskedResult = Traits::createSelectOp(
      rewriter, loc, isSpecial, qnanTensor, signedResult,
      utils::createEmptyOpWithTargetElemType(rewriter, loc, signedResult,
                                             rewriter.getF32Type()));

  if (!reshaped)
    return maskedResult;

  auto restoredType =
      RankedTensorType::get(inputType.getShape(), rewriter.getF32Type());
  return rewriter
      .create<tensor::ExpandShapeOp>(loc, restoredType, maskedResult,
                                     reassociation)
      .getResult();
}

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_REGBASE_NORMALIZE_UTILS_TRIGTEMPLATEHELPERS_H
