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
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace hfusion {

namespace {

static void addForwardEdge(Operation *lhs, Operation *rhs,
                           MemrefValueFlowOrderingRule::map &edges) {
  if (!lhs || !rhs || lhs == rhs)
    return;
  Operation *before = lhs->isBeforeInBlock(rhs) ? lhs : rhs;
  Operation *after = before == lhs ? rhs : lhs;
  edges[before].down.insert(after);
  edges[after].up.insert(before);
}

} // namespace

void MemrefValueFlowOrderingRule::apply(Block *block, map &edges) const {
  for (auto &op : block->getOperations()) {
    llvm::SmallVector<Value> copySrc;
    llvm::SmallVector<Value> targets;

    op.walk([&copySrc, &targets](Operation *inner) {
      if (auto copy = dyn_cast<memref::CopyOp>(inner)) {
        copySrc.push_back(copy.getSource());
        targets.push_back(copy.getTarget());
      } else if (auto store = dyn_cast<memref::StoreOp>(inner)) {
        targets.push_back(store.getMemref());
      }
    });

    for (Value src : copySrc) {
      for (Operation *user : src.getUsers())
        addForwardEdge(block->findAncestorOpInBlock(*user), &op, edges);
    }

    for (Value written : targets)
      for (Operation *user : written.getUsers())
        addForwardEdge(&op, block->findAncestorOpInBlock(*user), edges);
  }
}

} // namespace hfusion
} // namespace mlir
