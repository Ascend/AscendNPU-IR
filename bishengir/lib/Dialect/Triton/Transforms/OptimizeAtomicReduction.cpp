//===- OptimizeAtomicReduction.cpp - Skip no-op atomics ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A `tt.atomic_rmw` reduction whose `$val` is the reduction identity is a
// no-op: the memory is left untouched. Identities are `0` for
// `add`/`fadd`/`or`/`xor`/`umax`, all-ones for `and`/`umin`, signed minimum for
// `max` (or -inf for floats), and signed maximum for `min` (or +inf for
// floats). When `$val` is produced by a chain of `arith.select` ops that
// selects between the identity and another value `X`, the atomic is only
// meaningful on the lanes where the chain does not evaluate to the identity.
// This pass folds the select conditions into the atomic's mask
// (`mask &= (select_chain != identity)`) so the no-op lanes are skipped, and
// forwards `X` as the new value operand.
//
// The chain is scanned top-down: each `arith.select` must have a trivial
// (constant) identity on one arm; the other arm is scanned recursively and, if
// it is not itself such a chain, becomes `X`. A select between two non-trivial
// arms ends the scan without a result.
//
// The result of `tt.atomic_rmw` is undefined for masked-off lanes, so the
// rewrite is only applied when the atomic result is unused.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "optimize-atomic-reduction"
#define DBGS() (llvm::dbgs() << "[OptimizeAtomicReduction] ")

namespace bishengir {
namespace triton {
#define GEN_PASS_DEF_OPTIMIZEATOMICREDUCTION
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;

/// Return true if `value` is the integer reduction identity for `op`, i.e.
/// `op(identity, mem)` leaves `mem` untouched.
static bool isIdentityInteger(const APInt &value, RMWOp op) {
  switch (op) {
  case RMWOp::ADD:
  case RMWOp::OR:
  case RMWOp::XOR:
  case RMWOp::UMAX:
    return value.isZero();
  case RMWOp::AND:
  case RMWOp::UMIN:
    return value.isAllOnes();
  case RMWOp::MAX:
    return value.isMinSignedValue();
  case RMWOp::MIN:
    return value.isMaxSignedValue();
  default:
    return false;
  }
}

/// Return true if `value` is the floating-point reduction identity for `op`.
static bool isIdentityFloat(const APFloat &value, RMWOp op) {
  switch (op) {
  case RMWOp::FADD:
    return value.isZero();
  case RMWOp::MAX:
    return value.isInfinity() && value.isNegative();
  case RMWOp::MIN:
    return value.isInfinity() && !value.isNegative();
  default:
    return false;
  }
}

/// Return true if `attr` is a scalar or tensor constant equal to the reduction
/// identity of `op`.
static bool isIdentityAttr(Attribute attr, RMWOp op) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return isIdentityInteger(intAttr.getValue(), op);
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return isIdentityFloat(floatAttr.getValue(), op);

  auto denseAttr = dyn_cast<DenseElementsAttr>(attr);
  if (!denseAttr || denseAttr.getNumElements() == 0)
    return false;

  Type elemTy = denseAttr.getElementType();
  if (isa<IntegerType>(elemTy)) {
    if (denseAttr.isSplat())
      return isIdentityInteger(denseAttr.getSplatValue<APInt>(), op);
    return llvm::all_of(denseAttr.getValues<APInt>(), [op](APInt value) {
      return isIdentityInteger(value, op);
    });
  }
  if (isa<FloatType>(elemTy)) {
    if (denseAttr.isSplat())
      return isIdentityFloat(denseAttr.getSplatValue<APFloat>(), op);
    return llvm::all_of(denseAttr.getValues<APFloat>(), [op](APFloat value) {
      return isIdentityFloat(value, op);
    });
  }
  return false;
}

/// Return true if `v` is a reduction identity constant, possibly hidden behind
/// `tt.splat`.
static bool isIdentityConstant(Value v, RMWOp op) {
  if (auto splatOp = v.getDefiningOp<SplatOp>())
    return isIdentityConstant(splatOp.getSrc(), op);
  auto constOp = v.getDefiningOp<arith::ConstantOp>();
  return constOp && isIdentityAttr(constOp.getValue(), op);
}

/// Return true if `v` is a reduction identity for `op`, i.e. `op(identity,
/// mem)` leaves `mem` untouched:
///   add/fadd/or/xor/umax: zero
///   and/umin: all-ones
///   max: signed minimum, or -inf for floats
///   min: signed maximum, or +inf for floats
static bool isNoOpIdentity(Value v, RMWOp op) {
  return isIdentityConstant(v, op);
}

/// Scan a select chain to find the value selected on the non-identity lanes.
///
/// Returns that value, or `std::nullopt` when the chain does not select between
/// the reduction identity and some other value.  The scan is top-down:
///   * a select between two arms that are both non-identity yields no result;
///   * a select with a trivial (constant) identity on one arm recurses into the
///     other arm.  If that recursive scan fails, the other arm itself is the
///     non-identity value; otherwise the inner result is propagated.
std::optional<Value> scanSelectChain(Value v, RMWOp op) {
  auto selectOp = v.getDefiningOp<arith::SelectOp>();
  if (!selectOp)
    return v;

  Value rhs = selectOp.getTrueValue();
  Value lhs = selectOp.getFalseValue();

  // Canonicalize their order by swapping the non no-op identity operand to
  // RHS.
  if (!isNoOpIdentity(lhs, op))
    std::swap(rhs, lhs);

  // Bail out if none of operands is no-op identity.
  if (!isNoOpIdentity(lhs, op))
    return std::nullopt;

  // Scan RHS further.
  auto ret = scanSelectChain(rhs, op);
  return ret ? *ret : rhs;
}

/// Build an i1 value (scalar or tensor) matching the shape of `valueTy`.
Value buildBoolValue(OpBuilder &builder, Location loc, Type valueTy,
                     bool value) {
  Type i1Ty = builder.getI1Type();
  IntegerAttr attr = builder.getIntegerAttr(i1Ty, value);
  if (auto tensorTy = dyn_cast<RankedTensorType>(valueTy)) {
    auto boolTy = RankedTensorType::get(tensorTy.getShape(), i1Ty);
    return builder.create<arith::ConstantOp>(
        loc, boolTy, DenseElementsAttr::get(boolTy, attr));
  }
  return builder.create<arith::ConstantOp>(loc, attr);
}

/// Rebuild the select chain as an i1 predicate that is true on the lanes where
/// the chain does not take the reduction identity.
Value buildNonIdentityPredicate(Value v, RMWOp op, OpBuilder &builder,
                                DenseMap<Value, Value> &cache) {
  auto it = cache.find(v);
  if (it != cache.end())
    return it->second;

  Value result;
  if (auto selectOp = v.getDefiningOp<arith::SelectOp>()) {
    Value truePred =
        buildNonIdentityPredicate(selectOp.getTrueValue(), op, builder, cache);
    Value falsePred =
        buildNonIdentityPredicate(selectOp.getFalseValue(), op, builder, cache);
    result = builder.create<arith::SelectOp>(
        selectOp.getLoc(), selectOp.getCondition(), truePred, falsePred);
  } else {
    result = buildBoolValue(builder, v.getLoc(), v.getType(),
                            !isNoOpIdentity(v, op));
  }
  cache[v] = result;
  return result;
}

/// Fold `predicate` into the atomic's mask and forward `newVal` as its value.
void foldReduction(AtomicRMWOp op, Value newVal, Value predicate,
                   PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value newMask = predicate;
  if (Value mask = op.getMask())
    newMask = rewriter.create<arith::AndIOp>(loc, mask, predicate);
  LLVM_DEBUG(DBGS() << "folding no-op lanes into mask of " << op << '\n');
  rewriter.modifyOpInPlace(op, [&]() {
    op.getValMutable().assign(newVal);
    op.getMaskMutable().assign(newMask);
  });
}

struct OptimizeAtomicReductionPattern : public OpRewritePattern<AtomicRMWOp> {
  using OpRewritePattern<AtomicRMWOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AtomicRMWOp op,
                                PatternRewriter &rewriter) const override {
    RMWOp rmwOp = op.getAtomicRmwOp();

    // This transformation is only valid for atomic reductions.
    if (!op.getResult().use_empty())
      return failure();

    Value val = op.getVal();

    // Scan the select chain for the value it takes on the non-identity lanes.
    auto nonIdentity = scanSelectChain(val, rmwOp);
    if (!nonIdentity)
      return failure();

    // Every arm of the chain is the identity.
    if (isNoOpIdentity(*nonIdentity, rmwOp)) {
      rewriter.eraseOp(op);
      return success();
    }

    // Bail out if the val is not defined by a select chain.
    if (*nonIdentity == val)
      return failure();

    DenseMap<Value, Value> cache;
    Value predicate = buildNonIdentityPredicate(val, rmwOp, rewriter, cache);
    foldReduction(op, *nonIdentity, predicate, rewriter);
    return success();
  }
};

struct OptimizeAtomicReductionPass
    : public impl::OptimizeAtomicReductionBase<OptimizeAtomicReductionPass> {
  void runOnOperation() final {
    auto func = getOperation();
    LLVM_DEBUG(DBGS() << "running on " << func.getName() << '\n');

    RewritePatternSet patterns(&getContext());
    patterns.add<OptimizeAtomicReductionPattern>(&getContext());
    (void)applyPatternsGreedily(func, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createOptimizeAtomicReductionPass() {
  return std::make_unique<OptimizeAtomicReductionPass>();
}

} // namespace triton
} // namespace bishengir
