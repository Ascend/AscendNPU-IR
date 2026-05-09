//===- AffineMinMaxValueBounds.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Transforms/AffineMinMaxValueBounds.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/MathExtras.h"

#include <limits>
#include <vector>

namespace bishengir {

using namespace mlir;

namespace {

static std::optional<int64_t> getArithConstantIndexValue(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  return std::nullopt;
}

/// Exact constant when `arith.constant` or ValueBounds collapses LB==UB.
static std::optional<int64_t> getCollapsedConstantIndexValue(Value v) {
  if (auto cst = getArithConstantIndexValue(v))
    return cst;

  auto lb = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::LB, v);
  if (failed(lb))
    return std::nullopt;
  auto ub = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB, v,
      /*stopCondition=*/nullptr,
      /*closedUB=*/true);
  if (failed(ub) || *lb != *ub)
    return std::nullopt;
  return *lb;
}

static std::optional<int64_t> safeDivToI64(__int128 numer, int64_t denomPos) {
  if (denomPos <= 0)
    return std::nullopt;
  if (numer < 0)
    return std::nullopt;
  __int128 q = numer / static_cast<__int128>(denomPos);
  if (q < 0 || q > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(q);
}

static std::optional<int64_t> safeAddI64(int64_t a, int64_t b) {
  __int128 s = static_cast<__int128>(a) + static_cast<__int128>(b);
  if (s < std::numeric_limits<int64_t>::min() ||
      s > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(s);
}

static std::optional<int64_t>
computeTripCountFromHeader(int64_t lb, int64_t ub, int64_t step) {
  if (step == 0)
    return std::nullopt;
  if (step > 0) {
    if (ub <= lb)
      return 0;
    __int128 diff = static_cast<__int128>(ub) - static_cast<__int128>(lb);
    __int128 numer = diff + static_cast<__int128>(step) - 1;
    return safeDivToI64(numer, step);
  }

  int64_t stepAbs = -step;
  if (ub >= lb)
    return 0;
  __int128 diff = static_cast<__int128>(lb) - static_cast<__int128>(ub);
  __int128 numer = diff + static_cast<__int128>(stepAbs) - 1;
  return safeDivToI64(numer, stepAbs);
}

/// Same reasoning as the former `ScfForEliminateSingleIteration` pass header /
/// IV analysis (trip count or constant induction variable).
static FailureOr<bool> getSingleIterationTripCount(scf::ForOp forOp,
                                                   int64_t &tripCountOut) {
  Value iv = forOp.getInductionVar();

  auto lbV = getCollapsedConstantIndexValue(forOp.getLowerBound());
  auto ubV = getCollapsedConstantIndexValue(forOp.getUpperBound());
  auto stepV = getCollapsedConstantIndexValue(forOp.getStep());
  if (lbV && ubV && stepV) {
    auto tc = computeTripCountFromHeader(*lbV, *ubV, *stepV);
    if (!tc)
      return failure();
    tripCountOut = *tc;
    return true;
  }

  auto ivLB = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::LB, iv);
  auto ivUB = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB, iv,
      /*stopCondition=*/nullptr,
      /*closedUB=*/true);
  if (failed(ivLB) || failed(ivUB))
    return false;
  if (*ivLB != *ivUB)
    return false;

  tripCountOut = 1;
  return true;
}

static std::optional<int64_t>
evalAffineExprWithDims(AffineExpr expr, ArrayRef<int64_t> dimValues) {
  switch (expr.getKind()) {
  case AffineExprKind::Constant:
    return cast<AffineConstantExpr>(expr).getValue();
  case AffineExprKind::DimId: {
    auto pos = cast<AffineDimExpr>(expr).getPosition();
    if (pos >= dimValues.size())
      return std::nullopt;
    return dimValues[pos];
  }
  case AffineExprKind::SymbolId:
    return std::nullopt;
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    auto bin = cast<AffineBinaryOpExpr>(expr);
    auto lhsV = evalAffineExprWithDims(bin.getLHS(), dimValues);
    auto rhsV = evalAffineExprWithDims(bin.getRHS(), dimValues);
    if (!lhsV || !rhsV)
      return std::nullopt;
    switch (expr.getKind()) {
    case AffineExprKind::Add:
      return *lhsV + *rhsV;
    case AffineExprKind::Mul:
      return (*lhsV) * (*rhsV);
    case AffineExprKind::FloorDiv:
      if (*rhsV == 0)
        return std::nullopt;
      return llvm::divideFloorSigned(*lhsV, *rhsV);
    case AffineExprKind::CeilDiv:
      if (*rhsV == 0)
        return std::nullopt;
      return llvm::divideCeilSigned(*lhsV, *rhsV);
    case AffineExprKind::Mod:
      if (*rhsV <= 0)
        return std::nullopt;
      return llvm::mod(*lhsV, *rhsV);
    default:
      llvm_unreachable("unexpected affine binary op kind");
    }
  }
  }
  llvm_unreachable("unexpected affine expr kind");
}

static std::optional<std::pair<int64_t, int64_t>>
computeDiscreteScfForBoundsForAffineMinMax(AffineMap map, Value dimValue,
                                           bool isMin) {
  if (map.getNumDims() != 1 || map.getNumResults() == 0)
    return std::nullopt;

  auto blockArg = dyn_cast<BlockArgument>(dimValue);
  if (!blockArg)
    return std::nullopt;

  auto *parentBlock = blockArg.getOwner();
  auto parentOp = parentBlock->getParentOp();
  auto forOp = dyn_cast<scf::ForOp>(parentOp);
  if (!forOp)
    return std::nullopt;

  if (forOp.getInductionVar() != dimValue)
    return std::nullopt;

  auto lbOpt = getArithConstantIndexValue(forOp.getLowerBound());
  auto ubOpt = getArithConstantIndexValue(forOp.getUpperBound());
  auto stepOpt = getArithConstantIndexValue(forOp.getStep());
  if (!lbOpt || !ubOpt || !stepOpt)
    return std::nullopt;

  int64_t lb = *lbOpt;
  int64_t ub = *ubOpt;
  int64_t step = *stepOpt;
  if (step == 0)
    return std::nullopt;

  constexpr int64_t kMaxSamples = 4096;
  int64_t samples = 0;
  std::vector<int64_t> ivs;
  ivs.reserve(std::min<int64_t>(kMaxSamples, 16));

  if (step > 0) {
    for (int64_t iv = lb; iv < ub; iv += step) {
      ivs.push_back(iv);
      if (++samples >= kMaxSamples)
        return std::nullopt;
    }
  } else {
    for (int64_t iv = lb; iv > ub; iv += step) {
      ivs.push_back(iv);
      if (++samples >= kMaxSamples)
        return std::nullopt;
    }
  }
  if (ivs.empty())
    return std::make_pair<int64_t, int64_t>(/*lb*/ 0, /*ub*/ 0);

  std::optional<int64_t> bestMin, bestMax;
  for (int64_t iv : ivs) {
    int64_t d0 = iv;
    ArrayRef<int64_t> dims(&d0, 1);
    bool first = true;
    int64_t reduced = 0;
    for (auto expr : map.getResults()) {
      auto v = evalAffineExprWithDims(expr, dims);
      if (!v)
        return std::nullopt;
      if (first) {
        reduced = *v;
        first = false;
      } else if (isMin) {
        reduced = std::min<int64_t>(reduced, *v);
      } else {
        reduced = std::max<int64_t>(reduced, *v);
      }
    }

    if (!bestMin || !bestMax) {
      bestMin = reduced;
      bestMax = reduced;
      continue;
    }
    bestMin = std::min(*bestMin, reduced);
    bestMax = std::max(*bestMax, reduced);
  }

  if (!bestMin || !bestMax)
    return std::nullopt;
  return std::make_pair(*bestMin, *bestMax);
}

struct IndexValueConstantBounds {
  FailureOr<int64_t> lb;
  FailureOr<int64_t> ub;
};

static IndexValueConstantBounds computeIndexValueConstantBounds(Value value) {
  return {
      ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::LB, value,
          /*stopCondition=*/nullptr,
          /*closedUB=*/true),
      ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::UB, value,
          /*stopCondition=*/nullptr,
          /*closedUB=*/true),
  };
}

static void tryRecordConstantBounds(
    Value value, llvm::DenseMap<Value, std::pair<int64_t, int64_t>> *value2bounds) {
  if (!value2bounds)
    return;

  auto b = computeIndexValueConstantBounds(value);
  if (succeeded(b.lb) && succeeded(b.ub))
    (*value2bounds)[value] = {*b.lb, *b.ub};
}

static void recordAffineMinMaxOpBounds(
    Value result, AffineMap map, Value dimValue, bool isMin,
    llvm::DenseMap<Value, std::pair<int64_t, int64_t>> *value2bounds) {
  auto b = computeIndexValueConstantBounds(result);

  std::optional<std::pair<int64_t, int64_t>> discrete;
  if (failed(b.lb) || failed(b.ub))
    discrete = computeDiscreteScfForBoundsForAffineMinMax(map, dimValue, isMin);

  if (value2bounds) {
    std::optional<int64_t> finalLb;
    std::optional<int64_t> finalUb;
    if (succeeded(b.lb))
      finalLb = *b.lb;
    else if (discrete)
      finalLb = discrete->first;

    if (succeeded(b.ub))
      finalUb = *b.ub;
    else if (discrete)
      finalUb = discrete->second;

    if (finalLb && finalUb)
      (*value2bounds)[result] = {*finalLb, *finalUb};
  }
}

static void handleAffineMinMaxForValueBounds(
    Value result, AffineMap map, ValueRange dimOperands, bool isMin,
    llvm::DenseMap<Value, std::pair<int64_t, int64_t>> &value2bounds) {
  if (map.getNumDims() == 1 && dimOperands.size() == 1) {
    Value dimValue = *dimOperands.begin();
    recordAffineMinMaxOpBounds(result, map, dimValue, isMin, &value2bounds);
    return;
  }
  tryRecordConstantBounds(result, &value2bounds);
}

} // namespace

void AffineMinMaxValueBoundsCollector::populate(Operation *root) {
  value2bounds_.clear();
  root->walk([&](Operation *op) {
    if (auto minOp = dyn_cast<affine::AffineMinOp>(op)) {
      handleAffineMinMaxForValueBounds(minOp.getResult(), minOp.getAffineMap(),
                                       minOp.getDimOperands(),
                                       /*isMin=*/true, value2bounds_);
    } else if (auto maxOp = dyn_cast<affine::AffineMaxOp>(op)) {
      handleAffineMinMaxForValueBounds(maxOp.getResult(), maxOp.getAffineMap(),
                                       maxOp.getDimOperands(),
                                       /*isMin=*/false, value2bounds_);
    }
  });
}

static bool provesSingleIterationViaAffineMinMaxHeader(
    scf::ForOp forOp,
    const llvm::DenseMap<Value, std::pair<int64_t, int64_t>> &value2bounds) {
  auto stepOpt = getCollapsedConstantIndexValue(forOp.getStep());
  if (!stepOpt || *stepOpt == 0)
    return false;

  std::optional<int64_t> lowerConst;
  if (auto lbCst = getArithConstantIndexValue(forOp.getLowerBound())) {
    lowerConst = *lbCst;
  } else {
    auto it = value2bounds.find(forOp.getLowerBound());
    if (it != value2bounds.end() && it->second.first == it->second.second)
      lowerConst = it->second.first;
    else if (auto c = getCollapsedConstantIndexValue(forOp.getLowerBound()))
      lowerConst = *c;
  }

  std::optional<int64_t> upperLB, upperUB;
  if (auto ubCst = getArithConstantIndexValue(forOp.getUpperBound())) {
    upperLB = *ubCst;
    upperUB = *ubCst;
  } else {
    auto it = value2bounds.find(forOp.getUpperBound());
    if (it != value2bounds.end()) {
      upperLB = it->second.first;
      upperUB = it->second.second;
    } else if (auto c = getCollapsedConstantIndexValue(forOp.getUpperBound())) {
      upperLB = *c;
      upperUB = *c;
    }
  }

  if (!lowerConst || !upperLB || !upperUB)
    return false;

  const int64_t lb = *lowerConst;
  const int64_t step = *stepOpt;
  auto lbPlusStep = safeAddI64(lb, step);
  if (!lbPlusStep)
    return false;

  if (step > 0) {
    return (*upperLB > lb) && (*upperUB <= *lbPlusStep);
  }

  return (*upperUB < lb) && (*upperLB >= *lbPlusStep);
}

bool AffineMinMaxValueBoundsCollector::provesSingleIterationScfFor(
    scf::ForOp forOp) const {
  int64_t tripCount = 0;
  auto tripOk = getSingleIterationTripCount(forOp, tripCount);
  if (succeeded(tripOk) && tripOk.value() && tripCount == 1)
    return true;
  return provesSingleIterationViaAffineMinMaxHeader(forOp, value2bounds_);
}

} // namespace bishengir
