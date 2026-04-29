//===-------- NormalizeTrigTemplate.h -------------------------------------===//
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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZETRIGTEMPLATE_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZETRIGTEMPLATE_H

#include "bishengir/Transforms/Normalize/Utils/TrigTemplateHelpers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {

/// Rewrites `sin(x)` as:
///   1. k = round(x / pi)
///   2. r = x - k * pi
///   3. sin(x) = sin(r) * sign(k)
///
/// Then evaluate `sin(r)` with the odd Taylor polynomial
///   sin(r) ~= r - r^3 / 3! + r^5 / 5! - r^7 / 7! + r^9 / 9!
/// written in Horner form. Keeping `r` near zero makes this short series
/// accurate, and the parity of `k` restores the sign removed by shifting the
/// input by multiples of pi.
template <typename SinOpType, typename Traits>
struct NormalizeSinOpTemplate : public OpRewritePattern<SinOpType> {
public:
  using OpRewritePattern<SinOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(SinOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeSin(op))
      return failure();

    Value originalInput = op.getDpsInputs()[0];
    Type inputType = getElementTypeOrSelf(originalInput.getType());
    if (!inputType.isF16() && !inputType.isF32())
      return failure();

    Location loc = op.getLoc();
    Value input = originalInput;
    if (inputType.isF16()) {
      input = Traits::castTo(rewriter, loc, input, rewriter.getF32Type(),
                             CastRoundKind::Round);
    }

    // Reuse the same empty tensor result shape for the elementwise ops created
    // during normalization.
    Value empty = utils::createEmptyOp(rewriter, loc, input);
    Value roundedPiMultiple =
        buildRoundedPiMultiple<Traits>(rewriter, loc, input,
                                       rewriter.getF32Type());
    // r = x - k * pi, where k is chosen so that r stays close to 0.
    Value rangeReducedInput = buildRangeReducedTrigInput<Traits>(
        rewriter, loc, input, roundedPiMultiple, kPiApproximations, 0.0);

    // Evaluate
    //   sin(r) ~= r * (1 - r^2 / 3! + r^4 / 5! - r^6 / 7! + r^8 / 9!)
    // on the reduced input.
    Value sinApproximation = buildTaylorApproximation<Traits>(
        rewriter, loc, rangeReducedInput,
        getTaylorSeriesCoefficients(hfusion::TaylerMode::SIN, 5));
    // sin(x + k * pi) = (-1)^k * sin(x).
    Value sinSign =
        buildSinParitySign<Traits>(rewriter, loc, roundedPiMultiple);
    Value result = Traits::createBinaryOp(rewriter, loc, sinApproximation,
                                          sinSign, empty, BinaryKind::Mul);

    if (inputType.isF16()) {
      result = Traits::castTo(rewriter, loc, result, rewriter.getF16Type(),
                              CastRoundKind::Round);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Rewrites `cos(x)` as:
///   1. k = round(x / pi + 0.5)
///   2. r = x - k * pi + pi / 2
///   3. cos(x) = sin(r) * sign(k)
///
/// This uses the identity `cos(x) = sin(x + pi / 2)`, so cosine can reuse the
/// same sine Taylor polynomial after range reduction. The rounded multiple `k`
/// tracks which half-period the input came from so the final parity sign can be
/// restored afterwards.
template <typename CosOpType, typename Traits>
struct NormalizeCosOpTemplate : public OpRewritePattern<CosOpType> {
public:
  using OpRewritePattern<CosOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CosOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeCos(op))
      return failure();

    Value originalInput = op.getDpsInputs()[0];
    Type inputType = getElementTypeOrSelf(originalInput.getType());
    if (!inputType.isF16() && !inputType.isF32())
      return failure();

    Location loc = op.getLoc();
    Value input = originalInput;
    if (inputType.isF16()) {
      input = Traits::castTo(rewriter, loc, input, rewriter.getF32Type(),
                             CastRoundKind::Round);
    }

    Value empty = utils::createEmptyOp(rewriter, loc, input);
    Value roundedPiMultiple = buildRoundedPiMultiple<Traits>(
        rewriter, loc, input, rewriter.getF32Type(), 0.5);
    // r = x - k * pi + pi / 2, which is equivalent to reducing x + pi / 2.
    Value rangeReducedInput = buildRangeReducedTrigInput<Traits>(
        rewriter, loc, input, roundedPiMultiple, kPiApproximations, M_PI / 2);

    // After the shift, cosine uses the same approximation as sine:
    //   sin(r) ~= r - r^3 / 3! + r^5 / 5! - r^7 / 7! + r^9 / 9!.
    Value cosApproximation = buildTaylorApproximation<Traits>(
        rewriter, loc, rangeReducedInput,
        getTaylorSeriesCoefficients(hfusion::TaylerMode::SIN, 5));
    // cos(x) = (-1)^k * sin(r) for the reduced form above.
    Value cosSign =
        buildSinParitySign<Traits>(rewriter, loc, roundedPiMultiple);
    Value result = Traits::createBinaryOp(rewriter, loc, cosApproximation,
                                          cosSign, empty, BinaryKind::Mul);

    if (inputType.isF16()) {
      result = Traits::castTo(rewriter, loc, result, rewriter.getF16Type(),
                              CastRoundKind::Round);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Rewrites `atan(x)` with a branch-free sequence built from the same shared
/// helpers used by the templated sin/cos normalization.
///
/// The rewrite first computes on the magnitude only:
///   atan(x) = sign(x) * atan(|x|)
///
/// Then it reduces `|x|` in two stages so the odd Taylor series is evaluated
/// only on small arguments:
///   1. On `[0, 1]`, compare the direct series with the shifted identity
///        atan(x) = pi / 8 + atan((x - tan(pi / 8)) / (1 + x * tan(pi / 8)))
///      and keep the smaller branch.
///   2. On `[1, +inf)`, use
///        atan(x) = pi / 4 + atan((x - 1) / (x + 1))
///      to map the problem back into `[0, 1]`, then reuse step 1.
///
/// As in the historical HFusion implementation, the input is clipped to
/// `[-10000, 10000]` before the polynomial work starts. This keeps the
/// intermediates finite while changing the final angle by at most a negligible
/// amount because atan is already extremely close to `pi / 2` there.
template <typename AtanOpType, typename Traits>
struct NormalizeAtanOpTemplate : public OpRewritePattern<AtanOpType> {
public:
  using OpRewritePattern<AtanOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(AtanOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeAtan(op))
      return failure();

    Value originalInput = op.getDpsInputs()[0];
    Type inputType = getElementTypeOrSelf(originalInput.getType());
    if (!inputType.isF16() && !inputType.isF32())
      return failure();

    Location loc = op.getLoc();
    Value input = originalInput;
    if (inputType.isF16()) {
      input = Traits::castTo(rewriter, loc, input, rewriter.getF32Type(),
                             CastRoundKind::Round);
    }

    Value result = buildAtanApproximation<Traits>(rewriter, loc, input);
    if (inputType.isF16()) {
      result = Traits::castTo(rewriter, loc, result, rewriter.getF16Type(),
                              CastRoundKind::Round);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Rewrites `tan(x)` by first removing whole multiples of `pi`:
///   1. k = round(x / pi)
///   2. z = x - k * pi
///
/// Because `tan(x + k * pi) = tan(x)`, the reduced argument `z` stays near the
/// origin and the approximation only needs to model one principal interval.
/// The rewrite then evaluates:
///   tan(x) = tan(z)
///          ~= (((C0 * z^2 + C1) * z^2 + C2) * z)
///             / ((z^2 + D0) * (z + pi / 2) * (z - pi / 2)).
///
/// The odd numerator matches the symmetry `tan(-z) = -tan(z)`, while the
/// denominator factors at `z = +/- pi / 2` capture the poles where `cos(z)=0`.
/// As in the original rewrite, the `pi / 2` terms are built from split high and
/// low pieces in a fixed order so small rounding behavior near those poles is
/// preserved.
template <typename TanOpType, typename Traits>
struct NormalizeTanOpTemplate : public OpRewritePattern<TanOpType> {
public:
  using OpRewritePattern<TanOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(TanOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeTan(op))
      return failure();

    Value originalInput = op.getDpsInputs()[0];
    Type inputType = getElementTypeOrSelf(originalInput.getType());
    if (!inputType.isF16() && !inputType.isF32())
      return failure();

    Location loc = op.getLoc();
    Value input = originalInput;
    if (inputType.isF16()) {
      input = Traits::castTo(rewriter, loc, input, rewriter.getF32Type(),
                             CastRoundKind::Round);
    }

    Value result = buildTanApproximation<Traits>(rewriter, loc, input);
    if (inputType.isF16()) {
      result = Traits::castTo(rewriter, loc, result, rewriter.getF16Type(),
                              CastRoundKind::Round);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Rewrites `tanh(x)` using the exponential identity:
///   1. x' = clamp(x, -8.8, 8.8)
///   2. e = exp(2 * x')
///   3. tanh(x) ~= (e - 1) / (e + 1)
///
/// The clamp keeps `exp(2 * x')` finite and bounded while preserving the
/// saturated value of tanh within the tolerance.
template <typename TanhOpType, typename Traits>
struct NormalizeTanhOpTemplate : public OpRewritePattern<TanhOpType> {
public:
  using OpRewritePattern<TanhOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(TanhOpType op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldNormalizeTanh(op))
      return failure();

    Value originalInput = op.getDpsInputs()[0];
    Type inputType = getElementTypeOrSelf(originalInput.getType());
    if (!inputType.isF16() && !inputType.isF32())
      return failure();

    Location loc = op.getLoc();
    Value input = originalInput;
    if (inputType.isF16()) {
      input = Traits::castTo(rewriter, loc, input, rewriter.getF32Type(),
                             CastRoundKind::Round);
    }

    Value result = buildTanhApproximation<Traits>(rewriter, loc, input);
    if (inputType.isF16()) {
      result = Traits::castTo(rewriter, loc, result, rewriter.getF16Type(),
                              CastRoundKind::Round);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZETRIGTEMPLATE_H
