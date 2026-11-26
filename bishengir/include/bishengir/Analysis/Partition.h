//===- Partition.h - Generic partition relation algebra --------*- C++ -*-===//
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

#ifndef BISHENGIR_ANALYSIS_PARTITION_H
#define BISHENGIR_ANALYSIS_PARTITION_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <functional>
#include <string>
#include <utility>

namespace mlir {
namespace hivm {
using Group = SmallVector<Operation *>;

using Groups = SmallVector<Group>;

using Relation = std::function<FailureOr<Groups>(ArrayRef<Operation *>)>;

namespace detail {
inline LogicalResult verifyInputGroups(ArrayRef<Group> input,
                                       std::string &error) {
  llvm::DenseSet<Operation *> seen;
  for (const auto &group : input) {
    if (group.empty()) {
      error = "partition input group must not be empty";
      return failure();
    }
    for (Operation *member : group) {
      if (!member) {
        error = "partition input group contains a null operation";
        return failure();
      }
      if (!seen.insert(member).second) {
        error = "partition input group contains a duplicate operation";
        return failure();
      }
    }
  }
  return success();
}

inline bool isMemberOrDescendant(ArrayRef<Operation *> input,
                                 Operation *member) {
  return llvm::any_of(input, [member](Operation *candidate) {
    return candidate->isAncestor(member);
  });
}

inline LogicalResult verifyOutputGroup(ArrayRef<Operation *> inputGroup,
                                       const Group &group,
                                       llvm::DenseSet<Operation *> &seen,
                                       std::string &error) {
  if (group.empty()) {
    error = "partition relation returned an empty group";
    return failure();
  }
  for (Operation *member : group) {
    if (!member) {
      error = "partition relation returned a null operation";
      return failure();
    }
    if (!isMemberOrDescendant(inputGroup, member)) {
      error =
          "partition relation returned an operation outside the input group";
      return failure();
    }
    if (!seen.insert(member).second) {
      error = "partition relation returned a duplicate operation";
      return failure();
    }
  }
  return success();
}

} // namespace detail

inline FailureOr<Groups> applyRelation(ArrayRef<Group> input,
                                       const Relation &relation,
                                       std::string &error) {
  error.clear();
  if (failed(detail::verifyInputGroups(input, error)))
    return failure();

  Groups result;
  llvm::DenseSet<Operation *> seen;
  for (const auto &group : input) {
    auto produced = relation(group);
    if (failed(produced)) {
      if (error.empty())
        error = "partition relation failed";
      return failure();
    }
    for (const auto &outputGroup : *produced) {
      if (failed(detail::verifyOutputGroup(group, outputGroup, seen, error)))
        return failure();
      result.push_back(outputGroup);
    }
  }
  return result;
}

inline FailureOr<Groups> applyRelations(Operation *root,
                                        ArrayRef<Relation> relations,
                                        std::string &error) {
  error.clear();
  if (!root) {
    error = "partition root must not be null";
    return failure();
  }

  Groups groups{Group{root}};
  for (const auto &relation : relations) {
    auto next = applyRelation(groups, relation, error);
    if (failed(next))
      return failure();
    groups = std::move(*next);
  }
  return groups;
}

} // namespace hivm
} // namespace mlir

#endif
