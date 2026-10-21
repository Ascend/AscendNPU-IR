//===- OrderingConstraint.cpp - Per-block ordering constraints ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Ordering/OrderingConstraint.h"

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Ordering/Rule/MemrefValueFlowOrderingRule.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Ordering/Rule/SsaOrderingRule.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/PlanContext.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
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
  OrderingMap<FusionUnit> unitMap;
  transformMap(direct, unitMap, ctx);

  order.clear();
  auto collect = [&](const auto &direct, auto getSet,
                     DenseSet<FusionUnit> &result) {
    SmallVector<FusionUnit> worklist;
    for (FusionUnit fu : getSet(direct))
      if (result.insert(fu).second)
        worklist.push_back(fu);
    while (!worklist.empty()) {
      FusionUnit cur = worklist.pop_back_val();
      auto it = unitMap.find(cur);
      if (it == unitMap.end())
        continue;
      for (FusionUnit fu : getSet(it->second))
        if (result.insert(fu).second)
          worklist.push_back(fu);
    }
  };
  for (auto &[unit, direct] : unitMap) {
    DenseSet<FusionUnit> up, down;
    collect(
        direct, [](const auto &sets) -> const auto & { return sets.up; }, up);
    collect(
        direct, [](const auto &sets) -> const auto & { return sets.down; },
        down);
    auto &out = order[unit];
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
    // Walk the node's ops in reverse: moveAfter splices each op immediately
    // after anchor, so forward traversal would reverse their relative order.
    for (Operation *op : node->rops())
      op->moveAfter(anchor);
    return;
  }
  unit.get<Operation *>()->moveAfter(anchor);
}

void moveBefore(FusionUnit unit, Operation *anchor) {
  if (auto *node = unit.dyn_cast<const FusedNode *>()) {
    // Forward traversal preserves order: moveBefore splices each op immediately
    // before anchor, so each subsequent op lands right before the previous one,
    // keeping their relative order (anchor ends up after all of them).
    for (Operation *op : node->ops())
      op->moveBefore(anchor);
    return;
  }
  unit.get<Operation *>()->moveBefore(anchor);
}

} // namespace

void OrderingConstraint::reorder(FusionUnit lhs, FusionUnit rhs) {
  assert(lhs.isValid() && rhs.isValid() && lhs != rhs &&
         "reorder expects two distinct ordering units");
  if (!getFront(lhs)->isBeforeInBlock(getFront(rhs)))
    std::swap(lhs, rhs);

  auto lhsIt = order.find(lhs);
  if (lhsIt == order.end())
    llvm::report_fatal_error(
        "reorder: lhs unit missing ordering-constraint entry");

  // The only caller (FusedNode::consolidateToTail) reaches here after
  // canAccept() admitted the pair, so no constrained successor of lhs is also
  // required to precede rhs. Relocating that intervening successor past rhs
  // therefore never violates a constraint the fusion depends on. Guard it: when
  // this fires, a caller bypassed the canAccept() gate.
  assert(canAccept(lhs, rhs) &&
         "reorder expects an admissible (canAccept) lhs/rhs pair");

  Operation *lhsBack = getBack(lhs);
  Operation *rhsFront = getFront(rhs);
  SmallVector<FusionUnit> toMove;
  for (FusionUnit successor : lhsIt->second.down)
    if (lhsBack->isBeforeInBlock(getFront(successor)) &&
        getBack(successor)->isBeforeInBlock(rhsFront))
      toMove.push_back(successor);

  llvm::sort(toMove, [](FusionUnit a, FusionUnit b) {
    return getFront(a)->isBeforeInBlock(getFront(b));
  });
  for (FusionUnit successor : llvm::reverse(toMove))
    moveAfter(successor, getBack(rhs));
  moveBefore(lhs, getFront(rhs));
}

void OrderingConstraint::merge(FusionUnit from, FusionUnit to) {
  // Insert `to` first: on the first addLeaf it is absent and operator[]
  // creates it (and may rehash `order`), so this must happen before taking any
  // reference into `order`.
  auto &toSets = order[to];

  auto fromIt = order.find(from);
  if (fromIt == order.end())
    llvm::report_fatal_error(
        "merge: from unit missing ordering-constraint entry");
  auto &fromSets = fromIt->second;

  // 1. `to` absorbs `from`'s constraint, dropping self/internal references.
  toSets.up.insert(fromSets.up.begin(), fromSets.up.end());
  toSets.down.insert(fromSets.down.begin(), fromSets.down.end());
  toSets.up.erase(to);
  toSets.down.erase(to);
  toSets.up.erase(from);
  toSets.down.erase(from);

  // 2. Re-point `from`'s neighbours at `to`. Entities referencing `from` are
  //    exactly its reverse edges (up ∪ down). Use find() — never operator[] —
  //    so no key can be created here.
  for (FusionUnit x :
       llvm::concat<const FusionUnit>(fromSets.up, fromSets.down)) {
    // `to` can reference `from` directly (a `to -> from` / `from -> to`
    // edge); re-pointing that reference would insert `to` into its own set,
    // a meaningless self-edge. Skip.
    if (x == to)
      continue;
    // A transitively closed closure never references a unit's own key, and
    // every unit named in the closure still has an entry: merge() erases a
    // unit only after re-pointing its neighbours away, so a dangling
    // reference cannot occur.
    assert(x != from && "closure must not reference the source's own unit");
    auto it = order.find(x);
    assert(it != order.end() &&
           "every unit in the closure must still have an entry");
    if (it->second.up.erase(from))
      it->second.up.insert(to);
    if (it->second.down.erase(from))
      it->second.down.insert(to);
  }

  // 3. Every new path passes through the contracted unit. Connect all merged
  //    predecessors to all merged successors in both directions to keep the
  //    transitive closure. These loops keep reading `toSets` (a reference into
  //    `order`) while writing into other entries, so they must use find() —
  //    never operator[] — because inserting a missing key would rehash `order`
  //    and invalidate `toSets`. Neither set contains `from` or `to` (erased in
  //    step 1), so only already-present keys are looked up.
  for (FusionUnit predecessor : toSets.up) {
    auto it = order.find(predecessor);
    assert(it != order.end() &&
           "every predecessor in the closure must still have an entry");
    it->second.down.insert(toSets.down.begin(), toSets.down.end());
  }
  for (FusionUnit successor : toSets.down) {
    auto it = order.find(successor);
    assert(it != order.end() &&
           "every successor in the closure must still have an entry");
    it->second.up.insert(toSets.up.begin(), toSets.up.end());
  }

  // 4. Retire the source unit.
  order.erase(from);
}

} // namespace hfusion
} // namespace mlir
