//===- IndexBoundAnalyzer.h - Small index value bound analysis --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_UTILS_INDEXBOUNDANALYZER_H
#define BISHENGIR_DIALECT_UTILS_INDEXBOUNDANALYZER_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace utils {

/// Three-state result for static bound comparisons.
///
/// In boolean contexts, only `Sat` evaluates to true. `Unsat` and `Unknown`
/// both evaluate to false so callers can use `if (lhsBound <= rhsBound)` for
/// conservative "proven true" checks.
struct BoundCompareResult {
  enum Kind { Sat, Unsat, Unknown };

  constexpr BoundCompareResult(Kind kind) : kind(kind) {}

  constexpr bool isSat() const { return kind == Sat; }
  constexpr bool isUnsat() const { return kind == Unsat; }
  constexpr bool isUnknown() const { return kind == Unknown; }

  /// Boolean conversion is intentionally conservative: only `Sat` is true.
  /// Use `isUnsat()`/`isUnknown()` when the difference matters; `!result` means
  /// "not proven true", not "proven false".
  explicit constexpr operator bool() const { return isSat(); }

  Kind kind;
};

enum class BoundComparisonPredicate { LT, LE, EQ, GT, GE };

struct IndexBounds {
  std::optional<int64_t> lower;
  std::optional<int64_t> upper;
  std::optional<OpFoldResult> source;

  void print(llvm::raw_ostream &os) const;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const IndexBounds &bounds);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              BoundCompareResult result);
BoundCompareResult operator<(const IndexBounds &lhs, const IndexBounds &rhs);
BoundCompareResult operator<=(const IndexBounds &lhs, const IndexBounds &rhs);
BoundCompareResult operator==(const IndexBounds &lhs, const IndexBounds &rhs);
BoundCompareResult operator>(const IndexBounds &lhs, const IndexBounds &rhs);
BoundCompareResult operator>=(const IndexBounds &lhs, const IndexBounds &rhs);

/// Computes conservative bounds for index expressions.
///
/// The analyzer is a thin wrapper around MLIR's ValueBoundsConstraintSet. It
/// keeps constant lower/upper bounds for cheap range queries and preserves the
/// original value/attribute so comparisons can query the constraint set again.
class IndexBoundAnalyzer {
public:
  IndexBounds get(OpFoldResult value) const;
  IndexBounds get(Value value) const;

  BoundCompareResult compare(OpFoldResult lhs,
                             BoundComparisonPredicate predicate,
                             OpFoldResult rhs) const;
  BoundCompareResult compare(Value lhs, BoundComparisonPredicate predicate,
                             Value rhs) const;
};

} // namespace utils
} // namespace mlir

#endif // BISHENGIR_DIALECT_UTILS_INDEXBOUNDANALYZER_H
