//===- AffineMinMaxValueBounds.h - Affine min/max value bounds -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TRANSFORMS_AFFINEMINMAXVALUEBOUNDS_H
#define BISHENGIR_TRANSFORMS_AFFINEMINMAXVALUEBOUNDS_H

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace mlir {
class Operation;
} // namespace mlir

namespace bishengir {

/// Collects proven constant [LB, UB] intervals for index-typed SSA values that
/// are results of `affine.min` / `affine.max` (ValueBounds when possible, plus a
/// discrete fallback over `scf.for` induction variables). The map is intended
/// for analyses such as proving a `scf.for` executes exactly once.
class AffineMinMaxValueBoundsCollector {
public:
  using BoundsMap = llvm::DenseMap<mlir::Value, std::pair<int64_t, int64_t>>;

  /// Walk nested ops under `root` and record bounds for each affine min/max
  /// result.
  void populate(mlir::Operation *root);

  void clear() { value2bounds_.clear(); }

  const BoundsMap &getBoundsMap() const { return value2bounds_; }
  BoundsMap &getBoundsMap() { return value2bounds_; }

  std::optional<std::pair<int64_t, int64_t>>
  lookup(mlir::Value v) const {
    auto it = value2bounds_.find(v);
    if (it == value2bounds_.end())
      return std::nullopt;
    return it->second;
  }

  /// True when the loop is provably exactly one iteration: trip count / constant
  /// IV (ValueBounds), or header reasoning using `arith.constant`,
  /// ValueBounds-collapsed indices, and intervals from `lookup`.
  bool provesSingleIterationScfFor(mlir::scf::ForOp forOp) const;

private:
  BoundsMap value2bounds_;
};

} // namespace bishengir

#endif // BISHENGIR_TRANSFORMS_AFFINEMINMAXVALUEBOUNDS_H
