//===- Groups.cpp - HIVM group ordinal transport --------------------------===//
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

#include "bishengir/Dialect/HIVM/TransformOps/Groups.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hivm;

namespace mlir {
namespace hivm {
namespace {
SmallVector<Operation *> collectPreorder(ModuleOp root) {
  SmallVector<Operation *> operations;
  root->walk<WalkOrder::PreOrder>(
      [&](Operation *op) { operations.push_back(op); });
  return operations;
}

LogicalResult validateGroupOperation(Operation *op, std::string &error) {
  if (isa<ModuleOp>(op)) {
    error = "group member must not be a module";
    return failure();
  }
  if (isa<func::FuncOp>(op)) {
    error = "group member must not be a function";
    return failure();
  }
  return success();
}

FailureOr<Group> decodeGroup(ArrayRef<Operation *> operations,
                             DenseI64ArrayAttr encoded, std::string &error) {
  if (encoded.empty()) {
    error = "group must not be empty";
    return failure();
  }

  Group members;
  llvm::DenseSet<Operation *> seen;
  for (int64_t ordinal : encoded.asArrayRef()) {
    if (ordinal < 0 || static_cast<size_t>(ordinal) >= operations.size()) {
      error = "group ordinal " + std::to_string(ordinal) +
              " is outside the payload root";
      return failure();
    }
    Operation *member = operations[ordinal];
    if (failed(validateGroupOperation(member, error)))
      return failure();
    if (!seen.insert(member).second) {
      error = "group contains a duplicate operation";
      return failure();
    }
    members.push_back(member);
  }
  return members;
}

} // namespace

FailureOr<GroupOrdinals> buildGroupOrdinals(ModuleOp root, std::string &error) {
  error.clear();
  if (!root) {
    error = "payload root must be a module";
    return failure();
  }

  GroupOrdinals ordinals;
  int64_t nextOrdinal = 0;
  root->walk<WalkOrder::PreOrder>(
      [&](Operation *op) { ordinals[op] = nextOrdinal++; });
  return ordinals;
}

FailureOr<Groups> decodeGroups(ModuleOp root, GroupsAttr groups,
                               std::string &error) {
  error.clear();
  if (!root) {
    error = "payload root must be a module";
    return failure();
  }
  if (!groups) {
    error = "groups must be a #hivm_vectorize.groups attribute";
    return failure();
  }

  auto operations = collectPreorder(root);
  Groups decoded;
  for (DenseI64ArrayAttr encoded : groups.getGroups()) {
    auto group = decodeGroup(operations, encoded, error);
    if (failed(group))
      return failure();
    decoded.push_back(std::move(*group));
  }
  return decoded;
}

FailureOr<GroupsAttr> encodeGroups(ModuleOp root, ArrayRef<Group> groups,
                                   std::string &error) {
  error.clear();
  auto ordinals = buildGroupOrdinals(root, error);
  if (failed(ordinals))
    return failure();

  SmallVector<DenseI64ArrayAttr> encoded;
  for (const Group &group : groups) {
    if (group.empty()) {
      error = "group must not be empty";
      return failure();
    }
    SmallVector<int64_t> members;
    for (Operation *member : group) {
      if (!member) {
        error = "group contains a null operation";
        return failure();
      }
      if (failed(validateGroupOperation(member, error)))
        return failure();
      auto it = ordinals->find(member);
      if (it == ordinals->end()) {
        error = "group member is outside the payload root";
        return failure();
      }
      members.push_back(it->second);
    }
    encoded.push_back(DenseI64ArrayAttr::get(root.getContext(), members));
  }
  return GroupsAttr::get(root.getContext(), encoded);
}

} // namespace hivm
} // namespace mlir
