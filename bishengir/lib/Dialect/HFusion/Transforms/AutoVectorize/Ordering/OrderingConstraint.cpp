//===- OrderingConstraint.cpp - Per-block ordering constraints ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "hfusion-auto-vectorize-v2"

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Ordering/OrderingConstraint.h"

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Ordering/Rule/MemrefValueFlowOrderingRule.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Ordering/Rule/SsaOrderingRule.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/PlanContext.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <utility>

using namespace mlir;

namespace mlir {
namespace hfusion {

namespace {
// Convert the op-keyed direct edges to FusionUnit keys.
void transformMap(const OrderingMap<Operation *> &from,
                  OrderingMap<FusionUnit> &to, const PlanContext &ctx) {
  for (auto &[key, sets] : from) {
    auto &out = to[ctx.getUnit(key)];
    for (Operation *d : sets.down)
      out.down.insert(ctx.getUnit(d));
    for (Operation *u : sets.up)
      out.up.insert(ctx.getUnit(u));
  }
}
} // namespace

void OrderingConstraint::compute(Block *block, const PlanContext &ctx) {
  // Build Operation*-level direct edges from every rule. Every op in the block
  // gets an entry, even isolated ops that carry no edge, so the closure step
  // below can always find a key to re-point.
  OrderingMap<Operation *> direct;
  for (auto &op : block->getOperations())
    direct[&op];
  SsaOrderingRule().apply(block, direct);
  MemrefValueFlowOrderingRule().apply(block, direct);

  // Convert the op-keyed direct edges to FusionUnit keys, then tighten them
  // into transitive constraints.
  order.clear();
  OrderingMap<FusionUnit> nodeMap;
  transformMap(direct, nodeMap, ctx);

  for (auto &[node, sets] : nodeMap) {
    DenseSet<FusionUnit> up, down;
    SmallVector<FusionUnit> worklist;
    for (FusionUnit d : sets.down)
      if (down.insert(d).second)
        worklist.push_back(d);
    while (!worklist.empty()) {
      FusionUnit cur = worklist.pop_back_val();
      auto it = nodeMap.find(cur);
      if (it == nodeMap.end())
        continue;
      for (FusionUnit d : it->second.down)
        if (down.insert(d).second)
          worklist.push_back(d);
    }
    worklist.clear();
    for (FusionUnit u : sets.up)
      if (up.insert(u).second)
        worklist.push_back(u);
    while (!worklist.empty()) {
      FusionUnit cur = worklist.pop_back_val();
      auto it = nodeMap.find(cur);
      if (it == nodeMap.end())
        continue;
      for (FusionUnit u : it->second.up)
        if (up.insert(u).second)
          worklist.push_back(u);
    }
    auto &out = order[node];
    out.up.insert(up.begin(), up.end());
    out.down.insert(down.begin(), down.end());
  }
}

bool OrderingConstraint::canAccept(FusionUnit a, FusionUnit b) const {
  auto intersects = [](const DenseSet<FusionUnit> &lhs,
                       const DenseSet<FusionUnit> &rhs) {
    const auto &small = lhs.size() <= rhs.size() ? lhs : rhs;
    const auto &large = lhs.size() <= rhs.size() ? rhs : lhs;
    return llvm::any_of(small, [&](FusionUnit h) { return large.contains(h); });
  };
  auto aIt = order.find(a);
  auto bIt = order.find(b);
  // A unit without an entry has no ordering, hence no conflict.
  if (aIt == order.end() || bIt == order.end())
    return true;
  const auto &aSets = aIt->second;
  const auto &bSets = bIt->second;
  return !(intersects(aSets.down, bSets.up) ||
           intersects(aSets.up, bSets.down));
}

namespace {

Operation *getFront(FusionUnit unit) {
  if (auto *node = unit.dyn_cast<const FusedNode *>())
    return node->front();
  return unit.get<Operation *>();
}

Operation *getBack(FusionUnit unit) {
  if (auto *node = unit.dyn_cast<const FusedNode *>())
    return node->back();
  return unit.get<Operation *>();
}

void moveAfter(FusionUnit unit, Operation *anchor) {
  if (auto *node = unit.dyn_cast<const FusedNode *>()) {
    for (Operation *op : node->rops())
      op->moveAfter(anchor);
    return;
  }
  unit.get<Operation *>()->moveAfter(anchor);
}

void moveBefore(FusionUnit unit, Operation *anchor) {
  if (auto *node = unit.dyn_cast<const FusedNode *>()) {
    for (Operation *op : node->ops())
      op->moveBefore(anchor);
    return;
  }
  unit.get<Operation *>()->moveBefore(anchor);
}

} // namespace

void OrderingConstraint::adjust(FusionUnit lhs, FusionUnit rhs) const {
  assert(lhs.isValid() && rhs.isValid() && lhs != rhs &&
         "adjust expects two distinct ordering units");
  if (!getFront(lhs)->isBeforeInBlock(getFront(rhs)))
    std::swap(lhs, rhs);

  auto lhsIt = order.find(lhs);
  if (lhsIt == order.end())
    llvm::report_fatal_error(
        "adjust: lhs node missing ordering-constraint entry");

  Operation *lhsBack = getBack(lhs);
  Operation *rhsFront = getFront(rhs);
  SmallVector<FusionUnit> toMove;
  for (FusionUnit successor : lhsIt->second.down) {
    if (successor == lhs || successor == rhs)
      continue;
    Operation *successorFront = getFront(successor);
    Operation *successorBack = getBack(successor);
    if (lhsBack->isBeforeInBlock(successorFront) &&
        successorBack->isBeforeInBlock(rhsFront))
      toMove.push_back(successor);
  }

  llvm::sort(toMove, [](FusionUnit a, FusionUnit b) {
    return getFront(a)->isBeforeInBlock(getFront(b));
  });
  for (FusionUnit successor : llvm::reverse(toMove))
    moveAfter(successor, getBack(rhs));
  moveBefore(lhs, getFront(rhs));
}

void OrderingConstraint::update(FusionUnit from, FusionUnit to) {
  // Insert `to` first: on the first addLeaf it is absent and operator[]
  // creates it (and may rehash `order`), so this must happen before taking any
  // reference into `order`.
  auto &node = order[to];

  auto fromIt = order.find(from);
  if (fromIt == order.end())
    llvm::report_fatal_error(
        "update: from node missing ordering-constraint entry");
  auto &op = fromIt->second;

  // 1. `to` absorbs `from`'s constraint, dropping self/internal references.
  node.up.insert(op.up.begin(), op.up.end());
  node.down.insert(op.down.begin(), op.down.end());
  node.up.erase(to);
  node.down.erase(to);
  node.up.erase(from);
  node.down.erase(from);

  // 2. Re-point `from`'s neighbours at `to`. Entities referencing `from` are
  //    exactly its reverse edges (up ∪ down). Use find() — never operator[] —
  //    so no key can be created here.
  for (FusionUnit x : llvm::concat<const FusionUnit>(op.up, op.down)) {
    if (x == to || x == from)
      continue;
    auto it = order.find(x);
    if (it == order.end())
      continue;
    if (it->second.up.erase(from))
      it->second.up.insert(to);
    if (it->second.down.erase(from))
      it->second.down.insert(to);
  }

  // 3. Retire op.
  order.erase(from);
}

} // namespace hfusion
} // namespace mlir
