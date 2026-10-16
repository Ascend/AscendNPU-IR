//===- MemrefValueFlowOrderingRule.cpp - memref value-flow order -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Ordering/Rule/MemrefValueFlowOrderingRule.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace mlir {
namespace hfusion {

void MemrefValueFlowOrderingRule::apply(Block *block, map &edges) const {
  for (auto &op : block->getOperations()) {
    // The memref value written by the op, whose users flow after it.
    Value written;
    if (auto copyOp = dyn_cast<memref::CopyOp>(op))
      written = copyOp.getTarget();
    else if (auto storeOp = dyn_cast<memref::StoreOp>(op))
      written = storeOp.getMemref();
    else
      continue;

    for (Operation *user : written.getUsers())
      if (Operation *ancestor = block->findAncestorOpInBlock(*user)) {
        edges[&op].down.insert(ancestor);
        edges[ancestor].up.insert(&op);
      }
  }
}

} // namespace hfusion
} // namespace mlir
