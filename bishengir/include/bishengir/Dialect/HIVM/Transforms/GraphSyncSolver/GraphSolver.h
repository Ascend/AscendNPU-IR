//===------------- GraphSolver.h ---- Graph Sync Solver -------------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_GRAPHSOLVER_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_GRAPHSOLVER_H

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include <set>

namespace mlir::hivm::syncsolver {

class GraphSolver {
public:
  struct Edge {
    ConflictPair *const conflictPair;
    const CorePipeInfo corePipeSrc;
    const CorePipeInfo corePipeDst;
    const int startIndex;
    const int endIndex;
    const bool isUnitFlag;
    Edge() = delete;
    Edge(ConflictPair *conflictPair, CorePipeInfo corePipeSrc,
         CorePipeInfo corePipeDst, int startIndex, int endIndex,
         bool isUnitFlag)
        : conflictPair(conflictPair), corePipeSrc(corePipeSrc),
          corePipeDst(corePipeDst), startIndex(startIndex), endIndex(endIndex),
          isUnitFlag(isUnitFlag) {}
    Edge(CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst, int startIndex,
         int endIndex)
        : Edge(nullptr, corePipeSrc, corePipeDst, startIndex, endIndex, false) {
    }
    bool operator<(const Edge &other) const;
  };

  // adjacencyList[pipeSrc][pipeDst] stores a set of Edge objects representing
  // directed transitions from pipeSrc to pipeDst that are valid for a given
  // (startIndex,endIndex) lifetime. Used by runDijkstra to compute minimum
  // distance paths between two pipe ids taking ordering constraints into
  // account.
  llvm::DenseMap<CorePipeInfo, llvm::DenseMap<CorePipeInfo, std::set<Edge>>>
      adjacencyList;

  // Add a pipe-pair edge annotated with its active index interval.
  void addPair(ConflictPair *conflictPair, CorePipeInfo corePipeSrc,
               CorePipeInfo corePipeDst, int startIndex, int endIndex,
               bool isUnitFlag = false);

  // Build adjacency list from a ConflictPair by decomposing it into edges.
  void addConflictPair(syncsolver::ConflictPair *conflictPair);

  // Compact or merge overlapping edges to speed up Dijkstra queries.
  void optimizeAdjacencyList();

  // Run shortest-path search (Dijkstra-like) with ordering constraints to find
  // the minimal reachable index for a path from startPipe to endPipe.
  std::optional<int> runDijkstra(CorePipeInfo corePipeSrc,
                                 CorePipeInfo corePipeDst, int startIndex,
                                 int endIndex);

  std::optional<int> runDijkstraUnitFlagEnabled(Occurrence *occ1,
                                                Occurrence *occ2,
                                                CorePipeInfo corePipeSrc,
                                                CorePipeInfo corePipeDst,
                                                int startIndex, int endIndex);
};
} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_GRAPHSOLVER_H
