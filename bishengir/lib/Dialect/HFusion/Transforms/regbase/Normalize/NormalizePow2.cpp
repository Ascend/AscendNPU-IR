//===- NormalizePow2.cpp ----------------------------------------*- C++ -*-===//
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
// Strength reduction of integer div/div_unsigned/mod/modui with a
// power-of-two constant divisor into shift/and ops.
//
//   divui(x, 2^k)  ->  x >>u k
//   div(x, 2^k)    ->  (x + corr) >>s k,  corr = (x >>s (w-1)) & (2^k-1)
//   modui(x, 2^k)  ->  x & (2^k-1)
//   mod(x, 2^k)    ->  ((x + corr) & (2^k-1)) - corr
//
// The divisor constant is recognized through scalar constants, splat tensor
// constants and linalg.fill/broadcast wrappers. Only Ascend950 is rewritten:
// there, integer div/mod lower to a libcall into CCEC's software vdiv/vmod
// implementations (Template/lib/RegBase/Vector/Vdiv.cpp -- the constant
// divisor information is lost once the libcall is formed), while shifts/ands
// are native vector instructions.
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

} // namespace

void populateNormalizePow2DivRemPatterns(RewritePatternSet &patterns,
                                         bool enable) {
  if (!enable)
    return;
  MLIRContext *ctx = patterns.getContext();
  patterns.add<NormalizePow2DivUIToShift, NormalizePow2DivSIToShift,
               NormalizePow2RemUIToAnd, NormalizePow2RemSIToShift>(ctx);
}

} // namespace mlir::hfusion
