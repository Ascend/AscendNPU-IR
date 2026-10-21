//===- SsaOrderingRule.h - SSA use-def ordering rule ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ORDERING_RULE_SSAORDERINGRULE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ORDERING_RULE_SSAORDERINGRULE_H

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Ordering/OrderingMap.h"

namespace mlir {
class Block;
class Operation;

namespace hfusion {

/// SSA use-def ordering rule: a use-def edge (op defines a value its user
/// consumes) imposes an order between op and the user's ancestor in the block.
class SsaOrderingRule {
public:
  using map = OrderingMap<Operation *>;
  void apply(Block *block, map &edges) const;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_ORDERING_RULE_SSAORDERINGRULE_H
