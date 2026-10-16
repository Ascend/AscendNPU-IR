//===- OrderingConstraint.h - Per-block ordering constraints ------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ORDERING_ORDERINGCONSTRAINT_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ORDERING_ORDERINGCONSTRAINT_H

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/FusionUnit.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Ordering/OrderingMap.h"

namespace mlir {
class Block;

namespace hfusion {

class PlanContext;

/// A per-block ordering constraint, expressed as two facets of one module:
///
///   - plan time: judge whether fusing is admissible   -> canAccept()
///   - add time:  adjust physical order                -> adjust()
///   - add time:  update the constraint data           -> update()
///
/// The constraint is built from a fixed set of ordering rules (the knowledge
/// of what creates an order), consumed once per block by compute(); every other
/// method only touches the transitive closure.
class OrderingConstraint {
public:
  /// Compute the closure for `block` from the ordering rules.
  void compute(Block *block, const PlanContext &ctx);

  /// Whether fusing `a` into `b` is admissible under the constraint.
  bool canAccept(FusionUnit a, FusionUnit b) const;

  /// Make `lhs` and `rhs` adjacent while preserving the order of constrained
  /// entities between them. `lhs` and `rhs` must be distinct units in a block.
  void adjust(FusionUnit lhs, FusionUnit rhs) const;

  /// Contract `from` into `to`, creating `to` on first use and updating the
  /// constraint data.
  void update(FusionUnit from, FusionUnit to);

private:
  OrderingMap<FusionUnit> order;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ORDERING_ORDERINGCONSTRAINT_H
