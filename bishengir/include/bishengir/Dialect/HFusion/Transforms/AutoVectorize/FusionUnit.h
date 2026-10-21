//===- FusionUnit.h - Fusable unit for AutoVectorizeV2 planning -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_FUSIONUNIT_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_FUSIONUNIT_H

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/FusedNode.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/PointerUnion.h"

namespace mlir {
namespace hfusion {

class PlanContext;

/// A fusable unit in the block: either a single op that has not been fused yet,
/// or a whole fused node. This is the node type of OrderingConstraint.
class FusionUnit : public llvm::PointerUnion<Operation *, const FusedNode *> {
  using Base = llvm::PointerUnion<Operation *, const FusedNode *>;

public:
  /// Default-constructs the invalid sentinel (no entity).
  FusionUnit() : Base(nullptr) {}

  /// A standalone (not-yet-fused) op.
  FusionUnit(Operation *op) : Base(op) {}

  /// A whole fused node.
  FusionUnit(const FusedNode *node) : Base(node) {}

  /// Whether this denotes a real entity (op or node), not the invalid sentinel.
  bool isValid() const { return getOpaqueValue() != nullptr; }
};

} // namespace hfusion
} // namespace mlir

namespace llvm {
/// Teach DenseMap/DenseSet how to key on FusionUnit (mirrors the PointerUnion
/// specialisation, forwarding the empty/tombstone keys of the first type).
template <> struct DenseMapInfo<mlir::hfusion::FusionUnit> {
  using Unit = mlir::hfusion::FusionUnit;
  using FirstInfo = DenseMapInfo<mlir::Operation *>;

  static Unit getEmptyKey() { return Unit(FirstInfo::getEmptyKey()); }
  static Unit getTombstoneKey() { return Unit(FirstInfo::getTombstoneKey()); }
  static unsigned getHashValue(const Unit &v) {
    return DenseMapInfo<intptr_t>::getHashValue((intptr_t)v.getOpaqueValue());
  }
  static bool isEqual(const Unit &lhs, const Unit &rhs) {
    return lhs.getOpaqueValue() == rhs.getOpaqueValue();
  }
};
} // namespace llvm

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_FUSIONUNIT_H
