//===- IndexBoundAnalyzer.cpp - Small index value bound analysis ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Utils/IndexBoundAnalyzer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::utils;

static constexpr unsigned kMaxLocalAnalysisDepth = 8;

static void printBound(llvm::raw_ostream &os, std::optional<int64_t> bound) {
  if (bound) {
    os << *bound;
  } else {
    os << '?';
  }
}

static std::optional<int64_t> computeConstantBound(OpFoldResult value,
                                                   presburger::BoundType type) {
  auto bound = ValueBoundsConstraintSet::computeConstantBound(
      type, ValueBoundsConstraintSet::Variable(value),
      /*stopCondition=*/nullptr,
      /*closedUB=*/type == presburger::BoundType::UB);
  if (succeeded(bound)) {
    return *bound;
  }
  return std::nullopt;
}

static std::optional<int64_t> maxKnown(std::optional<int64_t> lhs,
                                       std::optional<int64_t> rhs) {
  if (lhs && rhs)
    return std::max(*lhs, *rhs);
  return lhs ? lhs : rhs;
}

static std::optional<int64_t>
maxKnownIfBoth(std::optional<int64_t> lhs, std::optional<int64_t> rhs) {
  if (lhs && rhs)
    return std::max(*lhs, *rhs);
  return std::nullopt;
}

static std::optional<int64_t> minKnown(std::optional<int64_t> lhs,
                                       std::optional<int64_t> rhs) {
  if (lhs && rhs)
    return std::min(*lhs, *rhs);
  return lhs ? lhs : rhs;
}

static std::optional<int64_t>
minKnownIfBoth(std::optional<int64_t> lhs, std::optional<int64_t> rhs) {
  if (lhs && rhs)
    return std::min(*lhs, *rhs);
  return std::nullopt;
}

static bool isValidIndexBound(OpFoldResult value) {
  if (dyn_cast_if_present<Attribute>(value)) {
    return getConstantIntValue(value).has_value();
  }
  Value dynamicValue = dyn_cast_if_present<Value>(value);
  if (!dynamicValue) {
    return false;
  }
  return dynamicValue.getType().isIndex();
}

static IndexBounds getLocalBounds(OpFoldResult value, unsigned depth);

static IndexBounds getLocalBounds(Value value, unsigned depth) {
  if (depth >= kMaxLocalAnalysisDepth || !value.getType().isIndex())
    return {};

  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return {constant.value(), constant.value(), getAsOpFoldResult(value)};

  if (auto minOp = value.getDefiningOp<arith::MinSIOp>()) {
    IndexBounds lhs = getLocalBounds(minOp.getLhs(), depth + 1);
    IndexBounds rhs = getLocalBounds(minOp.getRhs(), depth + 1);
    return {minKnownIfBoth(lhs.lower, rhs.lower), minKnown(lhs.upper, rhs.upper),
            getAsOpFoldResult(value)};
  }

  if (auto maxOp = value.getDefiningOp<arith::MaxSIOp>()) {
    IndexBounds lhs = getLocalBounds(maxOp.getLhs(), depth + 1);
    IndexBounds rhs = getLocalBounds(maxOp.getRhs(), depth + 1);
    return {maxKnown(lhs.lower, rhs.lower), maxKnownIfBoth(lhs.upper, rhs.upper),
            getAsOpFoldResult(value)};
  }

  return {};
}

static IndexBounds getLocalBounds(OpFoldResult value, unsigned depth) {
  if (auto constant = getConstantIntValue(value))
    return {*constant, *constant, value};
  if (auto dynamicValue = dyn_cast_if_present<Value>(value))
    return getLocalBounds(dynamicValue, depth);
  return {};
}

static ValueBoundsConstraintSet::ComparisonOperator
toValueBoundsPredicate(BoundComparisonPredicate predicate) {
  using Cmp = ValueBoundsConstraintSet::ComparisonOperator;
  switch (predicate) {
  case BoundComparisonPredicate::LT:
    return Cmp::LT;
  case BoundComparisonPredicate::LE:
    return Cmp::LE;
  case BoundComparisonPredicate::EQ:
    return Cmp::EQ;
  case BoundComparisonPredicate::GT:
    return Cmp::GT;
  case BoundComparisonPredicate::GE:
    return Cmp::GE;
  }
  llvm_unreachable("unknown bound comparison predicate");
}

static BoundComparisonPredicate
getNegatedPredicate(BoundComparisonPredicate predicate) {
  switch (predicate) {
  case BoundComparisonPredicate::LT:
    return BoundComparisonPredicate::GE;
  case BoundComparisonPredicate::LE:
    return BoundComparisonPredicate::GT;
  case BoundComparisonPredicate::EQ:
    llvm_unreachable("equality negation is not representable");
  case BoundComparisonPredicate::GT:
    return BoundComparisonPredicate::LE;
  case BoundComparisonPredicate::GE:
    return BoundComparisonPredicate::LT;
  }
  llvm_unreachable("unknown bound comparison predicate");
}

static FailureOr<bool> compareSources(OpFoldResult lhs,
                                      BoundComparisonPredicate predicate,
                                      OpFoldResult rhs) {
  ValueBoundsConstraintSet::Variable lhsVar(lhs);
  ValueBoundsConstraintSet::Variable rhsVar(rhs);

  if (predicate == BoundComparisonPredicate::EQ) {
    return ValueBoundsConstraintSet::areEqual(lhsVar, rhsVar);
  }

  if (ValueBoundsConstraintSet::compare(
          lhsVar, toValueBoundsPredicate(predicate), rhsVar)) {
    return true;
  }

  if (ValueBoundsConstraintSet::compare(
          lhsVar, toValueBoundsPredicate(getNegatedPredicate(predicate)),
          rhsVar)) {
    return false;
  }

  return failure();
}

static std::optional<bool>
compareConstantBounds(const IndexBounds &lhs,
                      BoundComparisonPredicate predicate,
                      const IndexBounds &rhs) {
  switch (predicate) {
  case BoundComparisonPredicate::LT:
    if (lhs.upper && rhs.lower && *lhs.upper < *rhs.lower)
      return true;
    if (lhs.lower && rhs.upper && *lhs.lower >= *rhs.upper)
      return false;
    return std::nullopt;
  case BoundComparisonPredicate::LE:
    if (lhs.upper && rhs.lower && *lhs.upper <= *rhs.lower)
      return true;
    if (lhs.lower && rhs.upper && *lhs.lower > *rhs.upper)
      return false;
    return std::nullopt;
  case BoundComparisonPredicate::EQ:
    if (lhs.lower && lhs.upper && rhs.lower && rhs.upper &&
        *lhs.lower == *lhs.upper && *rhs.lower == *rhs.upper &&
        *lhs.lower == *rhs.lower)
      return true;
    if (lhs.lower && rhs.upper && *lhs.lower > *rhs.upper)
      return false;
    if (rhs.lower && lhs.upper && *rhs.lower > *lhs.upper)
      return false;
    return std::nullopt;
  case BoundComparisonPredicate::GT:
    if (lhs.lower && rhs.upper && *lhs.lower > *rhs.upper)
      return true;
    if (lhs.upper && rhs.lower && *lhs.upper <= *rhs.lower)
      return false;
    return std::nullopt;
  case BoundComparisonPredicate::GE:
    if (lhs.lower && rhs.upper && *lhs.lower >= *rhs.upper)
      return true;
    if (lhs.upper && rhs.lower && *lhs.upper < *rhs.lower)
      return false;
    return std::nullopt;
  }
  llvm_unreachable("unknown bound comparison predicate");
}

void IndexBounds::print(llvm::raw_ostream &os) const {
  os << '[';
  printBound(os, lower);
  os << ", ";
  printBound(os, upper);
  os << ']';
}

llvm::raw_ostream &mlir::utils::operator<<(llvm::raw_ostream &os,
                                           const IndexBounds &bounds) {
  bounds.print(os);
  return os;
}

llvm::raw_ostream &mlir::utils::operator<<(llvm::raw_ostream &os,
                                           BoundCompareResult result) {
  switch (result.kind) {
  case BoundCompareResult::Sat:
    return os << "sat";
  case BoundCompareResult::Unsat:
    return os << "unsat";
  case BoundCompareResult::Unknown:
    return os << "unknown";
  }
  llvm_unreachable("unknown bound comparison result");
}

static BoundCompareResult compareBounds(const IndexBounds &lhs,
                                        BoundComparisonPredicate predicate,
                                        const IndexBounds &rhs) {
  if (!lhs.source || !rhs.source) {
    return BoundCompareResult::Unknown;
  }

  FailureOr<bool> result = compareSources(*lhs.source, predicate, *rhs.source);
  if (succeeded(result)) {
    return *result ? BoundCompareResult::Sat : BoundCompareResult::Unsat;
  }
  if (std::optional<bool> result =
          compareConstantBounds(lhs, predicate, rhs)) {
    return *result ? BoundCompareResult::Sat : BoundCompareResult::Unsat;
  }
  return BoundCompareResult::Unknown;
}

BoundCompareResult mlir::utils::operator<(const IndexBounds &lhs,
                                          const IndexBounds &rhs) {
  return compareBounds(lhs, BoundComparisonPredicate::LT, rhs);
}

BoundCompareResult mlir::utils::operator<=(const IndexBounds &lhs,
                                           const IndexBounds &rhs) {
  return compareBounds(lhs, BoundComparisonPredicate::LE, rhs);
}

BoundCompareResult mlir::utils::operator==(const IndexBounds &lhs,
                                           const IndexBounds &rhs) {
  return compareBounds(lhs, BoundComparisonPredicate::EQ, rhs);
}

BoundCompareResult mlir::utils::operator>(const IndexBounds &lhs,
                                          const IndexBounds &rhs) {
  return compareBounds(lhs, BoundComparisonPredicate::GT, rhs);
}

BoundCompareResult mlir::utils::operator>=(const IndexBounds &lhs,
                                           const IndexBounds &rhs) {
  return compareBounds(lhs, BoundComparisonPredicate::GE, rhs);
}

IndexBounds IndexBoundAnalyzer::get(OpFoldResult value) const {
  if (!isValidIndexBound(value)) {
    return {};
  }

  IndexBounds bounds;
  bounds.lower = computeConstantBound(value, presburger::BoundType::LB);
  bounds.upper = computeConstantBound(value, presburger::BoundType::UB);
  bounds.source = value;
  IndexBounds localBounds = getLocalBounds(value, /*depth=*/0);
  if (!bounds.lower)
    bounds.lower = localBounds.lower;
  if (!bounds.upper)
    bounds.upper = localBounds.upper;
  return bounds;
}

IndexBounds IndexBoundAnalyzer::get(Value value) const {
  return get(getAsOpFoldResult(value));
}

BoundCompareResult
IndexBoundAnalyzer::compare(OpFoldResult lhs,
                            BoundComparisonPredicate predicate,
                            OpFoldResult rhs) const {
  return compareBounds(get(lhs), predicate, get(rhs));
}

BoundCompareResult IndexBoundAnalyzer::compare(Value lhs,
                                               BoundComparisonPredicate predicate,
                                               Value rhs) const {
  return compareBounds(get(lhs), predicate, get(rhs));
}
