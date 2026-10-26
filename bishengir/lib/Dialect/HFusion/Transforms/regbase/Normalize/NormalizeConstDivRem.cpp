//===- NormalizeConstDivRem.cpp
//--------------------------------------------===//
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
//
// Strength-reduce i32 div/rem by a non-power-of-two splat constant on A5.
// LLVM computes full-width signed/unsigned magic factors and corrections.
// Keep the high multiply as HFusion mulext/mulextui, so vectorization and the
// ordinary Arith-to-AVE conversion can select native VMULL without i64 vectors.
// Power-of-two divisors are deliberately excluded: NormalizePow2 owns them.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeTraitsBase.h"
#include "bishengir/Transforms/regbase/Normalize/NormalizeArithmeticTemplate.h"
#include "llvm/Support/DivisionByConstantInfo.h"

using namespace mlir;

namespace mlir::hfusion {
namespace {

/// Accept scalar/splat constants and their fill/broadcast wrappers. Dynamic
/// and non-splat divisors must retain the original implementation.
static std::optional<APInt> getConstantDivisor(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (NormalizeTraitsBase::matchFillOp(def)) {
      value = NormalizeTraitsBase::getFillInput(def);
      continue;
    }
    if (NormalizeTraitsBase::matchBroadcastOp(def)) {
      value = NormalizeTraitsBase::getBroadcastInput(def);
      continue;
    }
    auto constant = dyn_cast<arith::ConstantOp>(def);
    if (!constant)
      return std::nullopt;
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getValue();
    if (auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue()))
      if (dense.isSplat())
        return dense.getSplatValue<APInt>();
    return std::nullopt;
  }
  return std::nullopt;
}

template <typename SourceOp>
struct NormalizeConstDivRemPattern : public OpRewritePattern<SourceOp>,
                                     private NormalizeTraitsBase {
  NormalizeConstDivRemPattern(MLIRContext *context)
      : OpRewritePattern<SourceOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    bool isUnsigned;
    bool isRemainder;
    if constexpr (std::is_same_v<SourceOp, linalg::ElemwiseBinaryOp>) {
      if (op.getFun() != linalg::BinaryFn::div &&
          op.getFun() != linalg::BinaryFn::div_unsigned)
        return failure();
      isUnsigned = op.getFun() == linalg::BinaryFn::div_unsigned;
      isRemainder = false;
    } else {
      if (op.getFun() != hfusion::BinaryFn::mod &&
          op.getFun() != hfusion::BinaryFn::modui)
        return failure();
      isUnsigned = op.getFun() == hfusion::BinaryFn::modui;
      isRemainder = true;
    }
    if (!archIsAscend950() || !op.hasPureTensorSemantics())
      return failure();

    Value x = op.getDpsInputs()[0];
    Value rhs = op.getDpsInputs()[1];
    Value init = op.getDpsInits()[0];
    auto type = dyn_cast<RankedTensorType>(x.getType());
    if (!type || !type.getElementType().isInteger(32) ||
        init.getType() != type ||
        getElementTypeOrSelf(rhs.getType()) != type.getElementType())
      return failure();
    if (isa<RankedTensorType>(rhs.getType()) && rhs.getType() != type)
      return failure();

    auto divisor = getConstantDivisor(rhs);
    if (!divisor || divisor->getBitWidth() != 32 || divisor->isZero())
      return failure();
    APInt magnitude = isUnsigned ? *divisor : divisor->abs();
    // The registration order is documentary: disjoint matches preserve the
    // Pow2 fastpath even though all patterns run in one greedy rewrite set.
    // Also exclude +/-1 and INT_MIN from the magic generator.
    if (magnitude.isPowerOf2())
      return failure();

    Location loc = op.getLoc();
    auto scalar = [&](const APInt &value) -> Value {
      return rewriter.create<arith::ConstantOp>(
          loc, type.getElementType(),
          rewriter.getIntegerAttr(type.getElementType(), value));
    };
    auto binary = [&](Value lhs, Value right, BinaryKind kind) -> Value {
      return createBinaryOp(rewriter, loc, lhs, right,
                            utils::createEmptyOp(rewriter, loc, init), kind);
    };
    auto shiftRight = [&](Value value, unsigned amount,
                          bool isSigned) -> Value {
      if (amount == 0)
        return value;
      return createConstantShift<NormalizeTraitsBase>(
          rewriter, loc, value, amount,
          isSigned ? ShiftKind::RightSigned : ShiftKind::RightUnsigned);
    };
    auto mulHigh = [&](Value value, const APInt &magic) -> Value {
      Value factor = createFillOp(rewriter, loc, scalar(magic),
                                  utils::createEmptyOp(rewriter, loc, value));
      if (isUnsigned)
        return rewriter
            .create<hfusion::MulExtUiOp>(loc, type, type, value, factor)
            .getHigh();
      return rewriter.create<hfusion::MulExtOp>(loc, type, type, value, factor)
          .getHigh();
    };

    // Principle: replace division by a certified integer fixed-point
    // reciprocal. For unsigned n, divisor d0 and a candidate p, let
    //   P = 2^p, M_full = ceil(P / d0), e = M_full * d0 - P.
    // Writing n = q * d0 + r, with 0 <= r < d0, gives
    //   n * M_full / P = q + r / d0 + n * e / (d0 * P).
    // Thus floor(n * M_full / P) equals q exactly when
    //   n * e < (d0 - r) * P.
    // LLVM chooses the multiplier and shifts so this bound holds over the
    // entire allowed input range; rounding a reciprocal alone is insufficient.
    // Here / denotes exact rational division; floor/ceil are explicit.
    // Products and M_full are unbounded mathematical integers in the proof.
    //
    // PreShift removes factors of two before forming the reciprocal:
    //   d = 2^k * d0, n = x >>u k => floor(x / d) = floor(n / d0).
    // For a 32-bit multiplier, umulhi(n, M) = floor(n * M / 2^32).
    // Consequently, with p = 32 + s, the quotient is umulhi(n, M) >>u s.
    // For example, unsigned /768 uses p=41, M=0xAAAAAAAB and e=256.
    // Since n < 2^32, n*e < 2^40 < P; even the smallest gap to the next
    // integer (r=d0-1) exceeds the reciprocal error, so s=41-32=9 is exact.
    //
    // IsAdd handles a reciprocal needing 33 bits: M_full = 2^32 + M.
    // With h = umulhi(n, M), its quotient is floor((n + h) / 2^s).
    // The sum n+h can overflow, so compute floor((n+h)/2) instead as
    //   h + ((n - h) >>u 1).
    // This is safe because h <= n and the result is <= n. LLVM reduces
    // PostShift by one to account for this halving; the omitted high magic
    // bit is therefore recovered without requiring 33-bit runtime arithmetic.
    //
    // Signed division uses a signed scaled reciprocal. If its effective
    // coefficient is M+2^32 (d>0, M<0) or M-2^32 (d<0, M>0), encoding M
    // as i32 loses a term +/-x*2^32 in the full product. Adding/subtracting
    // x after smulhi restores that term in the high half. Arithmetic right
    // shift then rounds down. LLVM's signed magic bounds ensure that the
    // resulting t satisfies trunc(x/d) = t + (t < 0), so t >>u 31 supplies
    // the required 0/1 correction toward zero, including reciprocal bias.
    // E.g. signed x=-768, d=768 uses M=0x2AAAAAAB and ShiftAmount=7:
    // smulhi=-129, t=-2, and adding the sign bit gives the exact quotient -1.
    //
    // Lane-wise rewrites (>>u is logical; >>s is arithmetic right shift):
    //   divui(x, d): n = x >>u PreShift; h = umulhi(n, M);
    //                if (IsAdd) h += (n - h) >>u 1;
    //                q = h >>u PostShift.
    //   divsi(x, d): h = smulhi(x, M);
    //                if (d > 0 && M < 0) h += x;
    //                if (d < 0 && M > 0) h -= x;
    //                t = h >>s ShiftAmount; q = t + (t >>u 31).
    // Once q is exact, x = q*d + r gives rem(x,d) = x - q*d, with the
    // original signed/unsigned remainder semantics and no second division.
    Value quotient;
    if (isUnsigned) {
      auto magic = llvm::UnsignedDivisionByConstantInfo::get(*divisor);
      Value input = shiftRight(x, magic.PreShift, /*isSigned=*/false);
      quotient = mulHigh(input, magic.Magic);
      if (magic.IsAdd) {
        // Compute (input+h)/2 without overflowing a 32-bit intermediate.
        Value delta = binary(input, quotient, BinaryKind::Sub);
        delta = shiftRight(delta, 1, /*isSigned=*/false);
        quotient = binary(quotient, delta, BinaryKind::Add);
      }
      quotient = shiftRight(quotient, magic.PostShift, /*isSigned=*/false);
    } else {
      auto magic = llvm::SignedDivisionByConstantInfo::get(*divisor);
      quotient = mulHigh(x, magic.Magic);
      if (divisor->isStrictlyPositive() && magic.Magic.isNegative())
        quotient = binary(quotient, x, BinaryKind::Add);
      else if (divisor->isNegative() && magic.Magic.isStrictlyPositive())
        quotient = binary(quotient, x, BinaryKind::Sub);
      quotient = shiftRight(quotient, magic.ShiftAmount, /*isSigned=*/true);
      // Convert negative arithmetic-shift rounding to truncation toward zero.
      Value correction = shiftRight(quotient, 31, /*isSigned=*/false);
      quotient = binary(quotient, correction, BinaryKind::Add);
    }
    Value result = quotient;
    if (isRemainder)
      result =
          binary(x, binary(quotient, rhs, BinaryKind::Mul), BinaryKind::Sub);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateNormalizeConstDivRemPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeConstDivRemPattern<linalg::ElemwiseBinaryOp>,
               NormalizeConstDivRemPattern<hfusion::ElemwiseBinaryOp>>(
      patterns.getContext());
}

} // namespace mlir::hfusion
