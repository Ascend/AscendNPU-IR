//===------------- CorePipeInfo.h ---- Graph Sync Solver ------------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_COREPIPEINFO_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_COREPIPEINFO_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

namespace mlir::hivm::syncsolver {

struct Occurrence;
struct EventIdNode;
struct MemInfoTree;

struct CorePipeInfo {
  hivm::TCoreType coreType{hivm::TCoreType::CUBE_OR_VECTOR};
  hivm::PIPE pipe{hivm::PIPE::PIPE_UNASSIGNED};

  CorePipeInfo() = default;

  CorePipeInfo(hivm::TCoreType coreType, hivm::PIPE pipe)
      : coreType(coreType), pipe(pipe) {}

  CorePipeInfo(std::pair<hivm::TCoreType, hivm::PIPE> corePipePair)
      : mlir::hivm::syncsolver::CorePipeInfo(corePipePair.first,
                                             corePipePair.second) {}

  bool operator==(const CorePipeInfo &other) const {
    return std::tie(coreType, pipe) == std::tie(other.coreType, other.pipe);
  }

  bool operator!=(const CorePipeInfo &other) const { return !(*this == other); }

  bool operator<(const CorePipeInfo &other) const {
    return std::tie(coreType, pipe) < std::tie(other.coreType, other.pipe);
  }
};
} // namespace mlir::hivm::syncsolver

namespace llvm {
template <> struct DenseMapInfo<mlir::hivm::syncsolver::CorePipeInfo> {
  using CorePipePairTy = std::pair<mlir::hivm::TCoreType, mlir::hivm::PIPE>;
  static inline mlir::hivm::syncsolver::CorePipeInfo getEmptyKey() {
    // Use sentinel values that are guaranteed never to appear as valid keys
    return DenseMapInfo<CorePipePairTy>::getEmptyKey();
  }
  static inline mlir::hivm::syncsolver::CorePipeInfo getTombstoneKey() {
    // Use a different set of sentinel values
    return DenseMapInfo<CorePipePairTy>::getTombstoneKey();
  }
  static unsigned
  getHashValue(const mlir::hivm::syncsolver::CorePipeInfo &val) {
    // Combine hashes of members
    return DenseMapInfo<CorePipePairTy>::getHashValue({val.coreType, val.pipe});
  }
  static bool isEqual(const mlir::hivm::syncsolver::CorePipeInfo &lhs,
                      const mlir::hivm::syncsolver::CorePipeInfo &rhs) {
    // Use the defined operator==
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_COREPIPEINFO_H
