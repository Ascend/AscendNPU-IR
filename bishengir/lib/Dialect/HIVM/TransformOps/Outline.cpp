//===- Outline.cpp - HIVM group outlining engine and transform op ---------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/TransformOps/Outline.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFFusionBlock.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFFusionOutliner.h"
#include "bishengir/Dialect/HIVM/TransformOps/Groups.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <string>

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::transform;

namespace mlir {
namespace hivm {
namespace {
LogicalResult validateOutlineGroup(ModuleOp root, const Group &group,
                                   llvm::DenseSet<Operation *> &seen,
                                   std::string &error) {
  if (group.empty()) {
    error = "outline group must not be empty";
    return failure();
  }
  Block *parent = nullptr;
  Operation *previous = nullptr;
  for (Operation *member : group) {
    if (!member || !root->isProperAncestor(member) ||
        !member->getParentOfType<func::FuncOp>()) {
      error = "outline group member must be inside a payload function";
      return failure();
    }
    if (!seen.insert(member).second) {
      error = "outline groups must be disjoint";
      return failure();
    }
    if (parent && parent != member->getBlock()) {
      error = "outline group members must share a parent block";
      return failure();
    }
    if (previous && !previous->isBeforeInBlock(member)) {
      error = "outline group must be in topological order";
      return failure();
    }
    if (member->hasTrait<OpTrait::IsTerminator>()) {
      error = "outline group must not contain a terminator";
      return failure();
    }
    parent = member->getBlock();
    previous = member;
  }
  return success();
}

LogicalResult validateOutlineUses(const Group &group, std::string &error) {
  analysis::VFFusionBlock block;
  for (Operation *member : group)
    block.fuseOp(member);
  auto *last = group.back();
  for (Value output : block.recomputeOutputs()) {
    for (Operation *user : output.getUsers()) {
      auto *ancestor = last->getBlock()->findAncestorOpInBlock(*user);
      if (ancestor && ancestor != last && ancestor->isBeforeInBlock(last) &&
          !llvm::is_contained(group, ancestor)) {
        error =
            "outline group result has an external use before the last member";
        return failure();
      }
    }
  }
  return success();
}

LogicalResult validateOutlineGroups(ModuleOp root, ArrayRef<Group> groups,
                                    std::string &error) {
  llvm::DenseSet<Operation *> seen;
  for (const Group &group : groups)
    if (failed(validateOutlineGroup(root, group, seen, error)))
      return failure();
  for (const Group &group : groups)
    if (failed(validateOutlineUses(group, error)))
      return failure();
  for (Operation *member : seen)
    for (Operation *ancestor = member->getParentOp(); ancestor;
         ancestor = ancestor->getParentOp())
      if (seen.contains(ancestor)) {
        error = "outline groups must not be nested";
        return failure();
      }
  return success();
}

void assertGroupContract(const Group &group) {
#ifndef NDEBUG
  assert(!group.empty() && "outline group must not be empty");
  Operation *leader = group.front();
  assert(leader && "outline group contains a null operation");
  Block *parentBlock = leader->getBlock();
  assert(parentBlock && "outline group must be inside a block");
  func::FuncOp parentFunction = leader->getParentOfType<func::FuncOp>();
  assert(parentFunction && "outline group must be inside a function");

  for (Operation *member : group) {
    assert(member && "outline group contains a null operation");
    assert(member->getParentOfType<func::FuncOp>() == parentFunction &&
           "outline group members must share a parent function");
    assert(member->getBlock() == parentBlock &&
           "outline group members must share a parent block");
  }
  for (size_t index = 1; index < group.size(); ++index)
    assert(group[index - 1]->isBeforeInBlock(group[index]) &&
           "outline group must be in topological order");
#else
  (void)group;
#endif
}

analysis::VFFusionBlock buildFusionBlock(const Group &group) {
  analysis::VFFusionBlock block;
  for (Operation *member : group)
    block.fuseOp(member);
  return block;
}

FailureOr<func::FuncOp> outlineGroup(const Group &group, std::string &error) {
  assertGroupContract(group);
  func::FuncOp parentFunction = group.front()->getParentOfType<func::FuncOp>();
  analysis::VFFusionBlock block = buildFusionBlock(group);

  OpBuilder builder(group.front()->getContext());
  builder.setInsertionPoint(group.front());

  analysis::VFFusionOutliner outliner;
  FailureOr<func::FuncOp> outlined =
      outliner.outline(parentFunction, block, builder);
  if (failed(outlined)) {
    error = "failed to outline a group into a function";
    return failure();
  }

  builder.setInsertionPointAfter(group.back());
  if (failed(outliner.createInvoke(*outlined, block, builder))) {
    error = "failed to create a call to an outlined function";
    return failure();
  }
  return *outlined;
}

} // namespace

FailureOr<SmallVector<func::FuncOp>>
outlineGroups(ModuleOp root, ArrayRef<Group> groups, std::string &error) {
  error.clear();
  if (!root) {
    error = "outline root must be a module";
    return failure();
  }
  if (failed(validateOutlineGroups(root, groups, error)))
    return failure();

  SmallVector<func::FuncOp> functions;
  for (const Group &group : llvm::reverse(groups)) {
    auto outlined = outlineGroup(group, error);
    if (failed(outlined))
      return failure();
    functions.push_back(*outlined);
  }
  return functions;
}

} // namespace hivm
} // namespace mlir

namespace {
FailureOr<ModuleOp> resolveOutlineRoot(Operation *user,
                                       ArrayRef<Operation *> roots) {
  if (roots.size() != 1)
    return user->emitOpError() << "expected exactly one payload root";
  auto root = dyn_cast<ModuleOp>(roots.front());
  if (!root)
    return user->emitOpError() << "expected a payload module root";
  return root;
}

FailureOr<GroupsAttr> resolveOutlineGroups(Operation *user,
                                           ArrayRef<Attribute> params) {
  if (params.size() != 1)
    return user->emitOpError() << "expected exactly one groups attribute";
  auto groups = dyn_cast<GroupsAttr>(params.front());
  if (!groups)
    return user->emitOpError() << "expected a #hivm_vectorize.groups attribute";
  return groups;
}

} // namespace

DiagnosedSilenceableFailure HIVMOutlineOp::apply(TransformRewriter &,
                                                 TransformResults &results,
                                                 TransformState &state) {
  auto payloadOps = state.getPayloadOps(getRoot());
  SmallVector<Operation *> roots(payloadOps.begin(), payloadOps.end());
  auto root = resolveOutlineRoot(getOperation(), roots);
  if (failed(root))
    return DiagnosedSilenceableFailure::definiteFailure();

  auto encodedGroups =
      resolveOutlineGroups(getOperation(), state.getParams(getGroups()));
  if (failed(encodedGroups))
    return DiagnosedSilenceableFailure::definiteFailure();

  std::string error;
  auto groups = decodeGroups(*root, *encodedGroups, error);
  if (failed(groups))
    return emitDefiniteFailure() << error;

  auto outlined = outlineGroups(*root, *groups, error);
  if (failed(outlined))
    return emitDefiniteFailure() << error;

  SmallVector<Operation *> functions;
  for (func::FuncOp function : *outlined)
    functions.push_back(function.getOperation());

  results.set(cast<OpResult>(getNewRoot()), {root->getOperation()});
  results.set(cast<OpResult>(getFuncs()), functions);
  return DiagnosedSilenceableFailure::success();
}

void HIVMOutlineOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getRootMutable(), effects);
  onlyReadsHandle(getGroupsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}
