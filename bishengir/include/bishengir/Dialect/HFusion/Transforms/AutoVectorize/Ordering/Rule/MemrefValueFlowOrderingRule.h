//===- MemrefValueFlowOrderingRule.h - memref value-flow order -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ORDERING_RULE_MEMREFVALUEFLOWORDERINGRULE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ORDERING_RULE_MEMREFVALUEFLOWORDERINGRULE_H

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Ordering/OrderingMap.h"

namespace mlir {
class Block;
class Operation;

namespace hfusion {

/// Memref value-flow ordering rule. Some side-effecting ops write a memref
/// without producing it as an SSA result (memref::CopyOp writes its target,
/// memref::StoreOp writes its memref); the consumers of that memref value
/// therefore have no SSA use-def edge back to the writer. This rule recovers
/// those edges so the writer orders the written value's users after it.
class MemrefValueFlowOrderingRule {
public:
  using map = OrderingMap<Operation *>;
  void apply(Block *block, map &edges) const;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ORDERING_RULE_MEMREFVALUEFLOWORDERINGRULE_H
