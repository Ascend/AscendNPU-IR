//===- SsaOrderingRule.cpp - SSA use-def ordering rule --*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Ordering/Rule/SsaOrderingRule.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace mlir {
namespace hfusion {

void SsaOrderingRule::apply(Block *block, map &edges) const {
  for (auto &op : block->getOperations())
    for (Operation *user : op.getUsers())
      if (Operation *ancestor = block->findAncestorOpInBlock(*user)) {
        edges[&op].down.insert(ancestor);
        edges[ancestor].up.insert(&op);
      }
}

} // namespace hfusion
} // namespace mlir
