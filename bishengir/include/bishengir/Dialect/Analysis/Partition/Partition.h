//===- Partition.h --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_PARTITION_PARTITION_H
#define BISHENGIR_DIALECT_ANALYSIS_PARTITION_PARTITION_H

#include "mlir/IR/Operation.h"
#include "llvm/Support/ErrorHandling.h"
#include <functional>

using namespace mlir;

namespace mlir::analysis {

using Group = SmallVector<Operation *>;
using GroupRef = ArrayRef<Operation *>;
using Groups = SmallVector<Group>;

using Relation = std::function<Groups(GroupRef)>;
using RelationChain = SmallVector<Relation>;

namespace {

// \forall x, y \in groups x \cap y = \emptyset
inline bool validateGroups(const Groups &groups) {
  DenseSet<Operation *> seen;

  for (auto &group : groups) {
    for (auto &op : group) {
      if (seen.contains(op)) {
        return false;
      }
      seen.insert(op);
    }
  }
  return true;
}

inline Groups applyRelationsImpl(RelationChain &relations, Groups &&groups) {
  if (relations.empty()) {
    return groups;
  }

  auto relation = relations.pop_back_val();
  Groups newGroups;
  for (auto &group : groups) {
    auto newGroupsStep = relation(group);
    for (auto &newGroup : newGroupsStep)
      newGroups.push_back(std::move(newGroup));
  }

  assert(validateGroups(newGroups));
  return applyRelationsImpl(relations, std::move(newGroups));
}

// R_n(R_{n-1}(...(R_1(root))...))
inline Groups applyRelations(Operation *root, const RelationChain &relations) {
  RelationChain relationsReversed(relations.rbegin(), relations.rend());
  Groups groups{Group{root}};

  groups = applyRelationsImpl(relationsReversed, std::move(groups));
  assert(validateGroups(groups));
  return groups;
}

} // namespace

template <typename Impl> struct Partition {
  static LogicalResult run(Operation *root, const RelationChain &relations) {
    auto groups = applyRelations(root, relations);
    for (auto &group : groups) {
      if (failed(rewrite(std::move(group)))) {
        return failure();
      }
    }
    return success();
  }

private:
  static LogicalResult rewrite(Group &&group) {
    if (group.empty())
      llvm_unreachable("invariant: group cannot be empty");
    return Impl::rewriteImpl(std::move(group));
  }
};

} // namespace mlir::analysis

#endif
