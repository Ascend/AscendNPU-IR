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
#include "llvm/ADT/SmallVector.h"

namespace mlir::hivm::syncsolver {

class GraphSolver {
public:
  // Configuration options.
  const SyncSolverOptions options;

  struct Edge {
    int startIndex{-1};
    int endIndex{-1};

    Edge() = delete;
    Edge(int startIndex, int endIndex)
        : startIndex(startIndex), endIndex(endIndex) {}
  };

  // adjacencyList[pipeSrc][pipeDst] stores a set of Edge objects representing
  // directed transitions from pipeSrc to pipeDst that are valid for a given
  // (startIndex,endIndex) lifetime. Used by runDijkstra to compute minimum
  // distance paths between two pipe ids taking ordering constraints into
  // account.
  llvm::DenseMap<CorePipeInfo,
                 llvm::DenseMap<CorePipeInfo, llvm::SmallVector<Edge>>>
      adjacencyList;

  virtual ~GraphSolver() = default;
  GraphSolver(const SyncSolverOptions &options) : options(options) {}

  // Build adjacency list from a ConflictPair by decomposing it into edges.
  void addConflictPair(syncsolver::ConflictPair *conflictPair);

  // Add a pipe-pair edge annotated with its active index interval.
  virtual void addPair(CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                       ConflictPair *conflictPair);

  // Run shortest-path search (Dijkstra-like) with ordering constraints to find
  // the minimal reachable index for a path from startPipe to endPipe.
  virtual std::optional<int> runDijkstra(CorePipeInfo corePipeSrc,
                                         CorePipeInfo corePipeDst,
                                         int startIndex, int endIndex,
                                         Occurrence *occ1 = nullptr,
                                         Occurrence *occ2 = nullptr);
};

class GraphSolverUnitFlag : public GraphSolver {
public:
  struct Edge {
    int startIndex{-1};
    int endIndex{-1};
    bool isUnitFlag{false};
    ConflictPair *conflictPair{nullptr};

    Edge() = delete;
    Edge(int startIndex, int endIndex, bool isUnitFlag,
         ConflictPair *conflictPair = nullptr)
        : startIndex(startIndex), endIndex(endIndex), isUnitFlag(isUnitFlag),
          conflictPair(conflictPair) {}
  };

  llvm::DenseMap<CorePipeInfo,
                 llvm::DenseMap<CorePipeInfo, llvm::SmallVector<Edge>>>
      adjacencyList;

  GraphSolverUnitFlag(const SyncSolverOptions &options)
      : GraphSolver(options) {}

  void addPair(CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
               ConflictPair *conflictPair) override;

  std::optional<int> runDijkstra(CorePipeInfo corePipeSrc,
                                 CorePipeInfo corePipeDst, int startIndex,
                                 int endIndex, Occurrence *occ1 = nullptr,
                                 Occurrence *occ2 = nullptr) override;
};
} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_GRAPHSOLVER_H
