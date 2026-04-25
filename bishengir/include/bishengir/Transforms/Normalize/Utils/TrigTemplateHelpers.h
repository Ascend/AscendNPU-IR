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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_TRIGTEMPLATEHELPERS_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_TRIGTEMPLATEHELPERS_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Normalize/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include <array>
#include <cmath>
#include <optional>

namespace mlir {

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

// The atan polynomial is evaluated only after clipping to a finite interval.
// This avoids overflow in intermediate products and is still numerically safe
// because atan(x) is already within 1e-4 of pi / 2 when |x| is 10000.
inline constexpr double kAtanClipLowerBound = -10000.0;
inline constexpr double kAtanClipUpperBound = 10000.0;
inline constexpr int kAtanTaylorTermCount = 7;

inline Value createFloatConstant(PatternRewriter &rewriter, Location loc,
                                 Type elementType, double value) {
  return rewriter
      .create<arith::ConstantOp>(loc, elementType,
                                 rewriter.getFloatAttr(elementType, value))
      .getResult();
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
getTaylorSeriesCoefficients(hfusion::TaylerMode taylerMode, int termCount) {
  if (termCount <= 0)
    llvm_unreachable("Taylor expansion term count must be positive");

  SmallVector<double> coefficients;
  coefficients.reserve(termCount);
  switch (taylerMode) {
  case hfusion::TaylerMode::SIN: {
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
  case hfusion::TaylerMode::ATAN: {
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

  return Traits::castTo(rewriter, loc, quotientByPi, roundedElementType,
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
      Traits::castTo(rewriter, loc, halfPiMultiple, rewriter.getF32Type(),
                     CastRoundKind::Floor);

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
      getTaylorSeriesCoefficients(hfusion::TaylerMode::ATAN,
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

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_TRIGTEMPLATEHELPERS_H
