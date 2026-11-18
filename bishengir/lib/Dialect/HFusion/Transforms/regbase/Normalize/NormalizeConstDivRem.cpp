//===- NormalizeConstDivRem.cpp ---------------------------------*- C++ -*-===//
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
// Strength-reduce integer div/rem by a splat constant on A5.
//
// Power-of-two divisors use shift/and fast paths for i8/i16/i32:
//   divui(x, 2^k) -> x >>u k
//   div(x, 2^k)   -> (x + corr) >>s k, corr = (x >>s (w-1)) & (2^k-1)
//   modui(x, 2^k) -> x & (2^k-1)
//   mod(x, 2^k)   -> ((x + corr) & (2^k-1)) - corr
//
// Other i32 constant divisors use LLVM's signed/unsigned magic factors and
// corrections. Keep the high multiply as HFusion mulext/mulextui, so
// vectorization and Arith-to-AVE conversion select native VMULL without i64
// vectors. Disjoint matches preserve the power-of-two fast paths, controlled
// independently by enable-div-rem-pow2.
//
// The divisor is recognized through scalar/splat constants and fill/broadcast
// wrappers. Only Ascend950 is rewritten: integer div/rem otherwise lower to
// CCEC's software vdiv/vmod implementations, losing the constant divisor.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeTraitsBase.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/regbase/Normalize/NormalizeArithmeticTemplate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/DivisionByConstantInfo.h"

using namespace mlir;

namespace mlir::hfusion {
namespace {

/// Benefit above the default 1 so these patterns win over NormalizeModOpRegBase
/// (i8 mod) and the final-stage NormalizeDivSIandDivUIOp (i8 div widening)
/// inside the single greedy pattern set of the regbase normalize pass.
constexpr unsigned kPow2Benefit = 2;

/// Matches a power-of-two constant divisor and returns k = log2(y).
/// Recognized sources: bare scalar arith.constant, splat tensor constant,
/// and constants wrapped by linalg.fill / broadcast. Returns std::nullopt for
/// non-constants (dynamic divisor), zero, non-power-of-two bit patterns and,
/// for signed ops, non-positive values (this also excludes y = 2^(w-1), whose
/// bit pattern is INT_MIN; positive powers of two are at most 2^(w-2), so the
/// signed k bound is implied).
std::optional<unsigned> getPowerOfTwoLog2(Value rhs, IntegerType elemTy,
                                          bool isUnsignedOp) {
  Value v = rhs;
  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (NormalizeTraitsBase::matchFillOp(def))
    v = NormalizeTraitsBase::getFillInput(def);
  else if (NormalizeTraitsBase::matchBroadcastOp(def))
    v = NormalizeTraitsBase::getBroadcastInput(def);

  if (getElementTypeOrSelf(v.getType()) != elemTy)
    return std::nullopt;

  auto cst = v.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return std::nullopt;

  APInt val;
  if (auto attr = dyn_cast<IntegerAttr>(cst.getValue())) {
    val = attr.getValue();
  } else if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue())) {
    if (!dense.isSplat())
      return std::nullopt;
    val = dense.getSplatValue<APInt>();
  } else {
    return std::nullopt;
  }

  if (!isUnsignedOp && val.isNegative())
    return std::nullopt;
  int32_t k = val.exactLogBase2();
  if (k < 0)
    return std::nullopt;
  return static_cast<unsigned>(k);
}

/// Scalar constant 2^k-1 of `elemTy`.
Value createLowBitsMask(PatternRewriter &rewriter, Location loc,
                        IntegerType elemTy, unsigned k) {
  return rewriter.create<arith::ConstantOp>(
      loc, elemTy,
      rewriter.getIntegerAttr(elemTy,
                              APInt::getLowBitsSet(elemTy.getWidth(), k)));
}

/// corr = (x >>s (w-1)) & (2^k-1): negative-input correction that turns the
/// floor semantics of the arithmetic shift into C-style truncation.
/// `initLike` supplies the shape of the products (the op's outs init).
Value buildSignedCorrection(PatternRewriter &rewriter, Location loc, Value x,
                            Value initLike, IntegerType elemTy, unsigned k) {
  unsigned w = elemTy.getWidth();
  Value shifted = createConstantShift<NormalizeTraitsBase>(
      rewriter, loc, x, w - 1, ShiftKind::RightSigned);
  Value mask = createLowBitsMask(rewriter, loc, elemTy, k);
  return NormalizeTraitsBase::createBinaryOp(
      rewriter, loc, shifted, mask,
      utils::createEmptyOp(rewriter, loc, initLike), BinaryKind::And);
}

/// All-zero tensor with the shape of `initLike` (the op's outs init), valid
/// for both static and dynamic shapes.
Value createZeroTensor(PatternRewriter &rewriter, Location loc,
                       Value initLike) {
  Value zero = utils::createConstantOp<int64_t>(
      rewriter, loc, getElementTypeOrSelf(initLike.getType()), 0);
  Value dst = utils::createEmptyOp(rewriter, loc, initLike);
  return NormalizeTraitsBase::createFillOp(rewriter, loc, zero, dst);
}

/// Shared match logic: A5 gate, shaped-dividend gate, element-type gate
/// (i8/i16/i32; i64 products lower to scalar loops/libcalls) and constant
/// divisor match. Returns (x, elemTy, k) on success.
std::optional<std::tuple<Value, IntegerType, unsigned>>
matchPow2DivRem(Operation *op, Value init, Value x, Value rhs,
                bool isUnsignedOp) {
  if (!NormalizeTraitsBase::archIsAscend950())
    return std::nullopt;

  // The two input type variables of elemwise_binary are independent, so a
  // scalar dividend against a splat tensor divisor is valid IR. Such an op is
  // a degenerate, foldable form; skip it rather than build shaped products
  // from a non-shaped operand.
  if (!isa<ShapedType>(x.getType()))
    return std::nullopt;

  auto elemTy = dyn_cast<IntegerType>(getElementTypeOrSelf(x.getType()));
  if (!elemTy || (elemTy.getWidth() != 8 && elemTy.getWidth() != 16 &&
                  elemTy.getWidth() != 32))
    return std::nullopt;
  if (getElementTypeOrSelf(rhs.getType()) != elemTy)
    return std::nullopt;

  // The original op numerically casts its inputs to the result element
  // type before computing ("promoting it to the same data type as the
  // accumulator/output"). A narrowing/widening result would truncate x
  // first (e.g. i32 65535 into an i16 result becomes -1), which the
  // correction computed at the input width cannot model -- keep the
  // original op for those forms.
  if (getElementTypeOrSelf(init.getType()) != elemTy)
    return std::nullopt;

  // Preserve the source operation's integer promotion semantics. Both
  // linalg and HFusion elemwise binary ops use a dialect-local TypeFnAttr.
  if (auto castAttr = op->getAttrOfType<linalg::TypeFnAttr>("cast"))
    if (castAttr.getValue() != linalg::TypeFn::cast_signed)
      return std::nullopt;
  if (auto castAttr = op->getAttrOfType<hfusion::TypeFnAttr>("cast"))
    if (castAttr.getValue() != hfusion::TypeFn::cast_signed)
      return std::nullopt;

  auto k = getPowerOfTwoLog2(rhs, elemTy, isUnsignedOp);
  if (!k)
    return std::nullopt;
  return std::make_tuple(x, elemTy, *k);
}

/// modui(x, 2^k) -> x & (2^k-1); k = 0 -> 0.
struct NormalizePow2RemUIToAnd
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp>,
      private NormalizeTraitsBase {
  NormalizePow2RemUIToAnd(MLIRContext *ctx)
      : OpRewritePattern<hfusion::ElemwiseBinaryOp>(ctx, kPow2Benefit) {}

  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics() || op.getFun() != hfusion::BinaryFn::modui)
      return failure();

    auto m = matchPow2DivRem(op.getOperation(), op.getDpsInits()[0],
                             op.getDpsInputs()[0], op.getDpsInputs()[1],
                             /*isUnsignedOp=*/true);
    if (!m)
      return failure();
    auto [x, elemTy, k] = *m;

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value init = op.getDpsInits()[0];
    if (k == 0) {
      rewriter.replaceOp(op, createZeroTensor(rewriter, loc, init));
      return success();
    }
    Value res = createBinaryOp(
        rewriter, loc, x, createLowBitsMask(rewriter, loc, elemTy, k),
        utils::createEmptyOp(rewriter, loc, init), BinaryKind::And);
    rewriter.replaceOp(op, res);
    return success();
  }
};

/// mod(x, 2^k) -> ((x + corr) & (2^k-1)) - corr; k = 0 -> 0.
struct NormalizePow2RemSIToShift
    : public OpRewritePattern<hfusion::ElemwiseBinaryOp>,
      private NormalizeTraitsBase {
  NormalizePow2RemSIToShift(MLIRContext *ctx)
      : OpRewritePattern<hfusion::ElemwiseBinaryOp>(ctx, kPow2Benefit) {}

  LogicalResult matchAndRewrite(hfusion::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics() || op.getFun() != hfusion::BinaryFn::mod)
      return failure();

    auto m = matchPow2DivRem(op.getOperation(), op.getDpsInits()[0],
                             op.getDpsInputs()[0], op.getDpsInputs()[1],
                             /*isUnsignedOp=*/false);
    if (!m)
      return failure();
    auto [x, elemTy, k] = *m;

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value init = op.getDpsInits()[0];
    if (k == 0) {
      rewriter.replaceOp(op, createZeroTensor(rewriter, loc, init));
      return success();
    }
    Value corr = buildSignedCorrection(rewriter, loc, x, init, elemTy, k);
    Value mask = createLowBitsMask(rewriter, loc, elemTy, k);
    Value sum = createBinaryOp(rewriter, loc, x, corr,
                               utils::createEmptyOp(rewriter, loc, init),
                               BinaryKind::Add);
    Value rem = createBinaryOp(rewriter, loc, sum, mask,
                               utils::createEmptyOp(rewriter, loc, init),
                               BinaryKind::And);
    Value res = createBinaryOp(rewriter, loc, rem, corr,
                               utils::createEmptyOp(rewriter, loc, init),
                               BinaryKind::Sub);
    rewriter.replaceOp(op, res);
    return success();
  }
};

/// div_unsigned(x, 2^k) -> x >>u k; k = 0 -> x.
struct NormalizePow2DivUIToShift
    : public OpRewritePattern<linalg::ElemwiseBinaryOp>,
      private NormalizeTraitsBase {
  NormalizePow2DivUIToShift(MLIRContext *ctx)
      : OpRewritePattern<linalg::ElemwiseBinaryOp>(ctx, kPow2Benefit) {}

  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics() ||
        op.getFun() != linalg::BinaryFn::div_unsigned)
      return failure();

    auto m = matchPow2DivRem(op.getOperation(), op.getDpsInits()[0],
                             op.getDpsInputs()[0], op.getDpsInputs()[1],
                             /*isUnsignedOp=*/true);
    if (!m)
      return failure();
    auto [x, elemTy, k] = *m;

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    if (k == 0) {
      rewriter.replaceOp(op, x);
      return success();
    }
    Value shift = utils::createConstantOp<int64_t>(rewriter, loc, elemTy, k);
    Value init = op.getDpsInits()[0];
    Value res = createShiftOp(rewriter, loc, x, shift,
                              utils::createEmptyOp(rewriter, loc, init),
                              ShiftKind::RightUnsigned);
    rewriter.replaceOp(op, res);
    return success();
  }
};

/// div(x, 2^k) -> (x + corr) >>s k; k = 0 -> x.
struct NormalizePow2DivSIToShift
    : public OpRewritePattern<linalg::ElemwiseBinaryOp>,
      private NormalizeTraitsBase {
  NormalizePow2DivSIToShift(MLIRContext *ctx)
      : OpRewritePattern<linalg::ElemwiseBinaryOp>(ctx, kPow2Benefit) {}

  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics() || op.getFun() != linalg::BinaryFn::div)
      return failure();

    auto m = matchPow2DivRem(op.getOperation(), op.getDpsInits()[0],
                             op.getDpsInputs()[0], op.getDpsInputs()[1],
                             /*isUnsignedOp=*/false);
    if (!m)
      return failure();
    auto [x, elemTy, k] = *m;

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    if (k == 0) {
      rewriter.replaceOp(op, x);
      return success();
    }
    Value init = op.getDpsInits()[0];
    Value corr = buildSignedCorrection(rewriter, loc, x, init, elemTy, k);
    Value sum = createBinaryOp(rewriter, loc, x, corr,
                               utils::createEmptyOp(rewriter, loc, init),
                               BinaryKind::Add);
    Value shift = utils::createConstantOp<int64_t>(rewriter, loc, elemTy, k);
    Value res = createShiftOp(rewriter, loc, sum, shift,
                              utils::createEmptyOp(rewriter, loc, init),
                              ShiftKind::RightSigned);
    rewriter.replaceOp(op, res);
    return success();
  }
};

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

void populateNormalizeConstDivRemPatterns(RewritePatternSet &patterns,
                                          bool enableDivRemPow2) {
  if (enableDivRemPow2)
    patterns.add<NormalizePow2DivUIToShift, NormalizePow2DivSIToShift,
                 NormalizePow2RemUIToAnd, NormalizePow2RemSIToShift>(
        patterns.getContext());
  patterns.add<NormalizeConstDivRemPattern<linalg::ElemwiseBinaryOp>,
               NormalizeConstDivRemPattern<hfusion::ElemwiseBinaryOp>>(
      patterns.getContext());
}

} // namespace mlir::hfusion
