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
#include "llvm/Support/Casting.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::utils;

static void printBound(llvm::raw_ostream &os, std::optional<int64_t> bound) {
  if (bound)
    os << *bound;
  else
    os << '?';
}

static std::optional<int64_t> maxKnown(std::optional<int64_t> lhs,
                                       std::optional<int64_t> rhs) {
  if (lhs && rhs)
    return std::max(*lhs, *rhs);
  return lhs ? lhs : rhs;
}

static std::optional<int64_t> minKnown(std::optional<int64_t> lhs,
                                       std::optional<int64_t> rhs) {
  if (lhs && rhs)
    return std::min(*lhs, *rhs);
  return lhs ? lhs : rhs;
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

IndexBounds IndexBoundAnalyzer::get(OpFoldResult value) const {
  if (auto constant = getConstantIntValue(value))
    return {*constant, *constant};

  if (auto dynamicValue = dyn_cast_if_present<Value>(value))
    return get(dynamicValue);

  return {};
}

IndexBounds IndexBoundAnalyzer::get(Value value) const {
  return get(value, /*depth=*/0);
}

bool IndexBoundAnalyzer::hasUpperBoundAtMost(OpFoldResult value,
                                             int64_t bound) const {
  IndexBounds bounds = get(value);
  return bounds.upper && *bounds.upper <= bound;
}

IndexBounds IndexBoundAnalyzer::get(Value value, unsigned depth) const {
  if (!value.getType().isIndex())
    return {};
  if (auto constant = getConstantIntValue(getAsOpFoldResult(value)))
    return {*constant, *constant};

  if (depth >= maxAnalysisDepth)
    return {};

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return {};

  if (auto maxOp = dyn_cast<arith::MaxSIOp>(defOp)) {
    IndexBounds lhs = get(maxOp.getLhs(), depth + 1);
    IndexBounds rhs = get(maxOp.getRhs(), depth + 1);
    return {maxKnown(lhs.lower, rhs.lower),
            lhs.upper && rhs.upper
                ? std::optional<int64_t>(std::max(*lhs.upper, *rhs.upper))
                : std::nullopt};
  }

  if (auto minOp = dyn_cast<arith::MinSIOp>(defOp)) {
    IndexBounds lhs = get(minOp.getLhs(), depth + 1);
    IndexBounds rhs = get(minOp.getRhs(), depth + 1);
    return {lhs.lower && rhs.lower
                ? std::optional<int64_t>(std::min(*lhs.lower, *rhs.lower))
                : std::nullopt,
            minKnown(lhs.upper, rhs.upper)};
  }

  return {};
}
