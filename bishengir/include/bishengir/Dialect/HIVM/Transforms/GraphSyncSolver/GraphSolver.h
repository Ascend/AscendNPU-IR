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

class GraphSolverBase {
public:
  // Configuration options.
  const SyncSolverOptions options;

  llvm::SmallVector<int> barrierAllIndexes, tempBarrierAllIndexes;
  llvm::DenseMap<CorePipeInfo, llvm::SmallVector<int>> barrierIndexes,
      tempBarrierIndexes;

  virtual ~GraphSolverBase() = default;
  GraphSolverBase(const SyncSolverOptions &options) : options(options) {}

  void clearBarrierIndexes();

  // Build adjacency list from a ConflictPair by decomposing it into edges.
  void insertConflictPair(syncsolver::ConflictPair *conflictPair,
                          bool isTemp = false);
  void eraseConflictPair(syncsolver::ConflictPair *conflictPair,
                         bool isTemp = false);

  bool checkAnyBarrierAllBetween(int startIndex, int endIndex);
  bool checkAnyBarrierBetween(CorePipeInfo corePipe, int startIndex,
                              int endIndex);

  virtual void clearAdjList(bool isTemp) = 0;

  // Add a pipe-pair edge annotated with its active index interval.
  virtual void insertEdge(CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                          ConflictPair *conflictPair, bool isTemp = false) = 0;
  virtual void eraseEdge(CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                         ConflictPair *conflictPair, bool isTemp = false) = 0;

  // Run shortest-path search (Dijkstra-like) with ordering constraints to find
  // the minimal reachable index for a path from startPipe to endPipe.
  virtual std::optional<int> runDijkstra(CorePipeInfo corePipeSrc,
                                         CorePipeInfo corePipeDst,
                                         int startIndex, int endIndex,
                                         Occurrence *occ1 = nullptr,
                                         Occurrence *occ2 = nullptr) = 0;
};

class GraphSolver : public GraphSolverBase {
public:
  struct Edge {
    int startIndex{-1};
    int endIndex{-1};

    Edge() = delete;
    Edge(int startIndex, int endIndex)
        : startIndex(startIndex), endIndex(endIndex) {}

    bool operator==(const Edge &other) const {
      return std::tie(startIndex, endIndex) ==
             std::tie(other.startIndex, other.endIndex);
    }

    bool operator!=(const Edge &other) const { return !(*this == other); }
  };

  llvm::DenseMap<CorePipeInfo,
                 llvm::DenseMap<CorePipeInfo, llvm::SmallVector<Edge>>>
      adjacencyList, tempAdjacencyList;

  GraphSolver(const SyncSolverOptions &options) : GraphSolverBase(options) {}

  void clearAdjList(bool isTemp = false) override;

  void insertEdge(CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                  ConflictPair *conflictPair, bool isTemp = false) override;
  void eraseEdge(CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                 ConflictPair *conflictPair, bool isTemp = false) override;

  std::optional<int> runDijkstra(CorePipeInfo corePipeSrc,
                                 CorePipeInfo corePipeDst, int startIndex,
                                 int endIndex, Occurrence *occ1 = nullptr,
                                 Occurrence *occ2 = nullptr) override;
};

class GraphSolverUnitFlag : public GraphSolverBase {
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

    bool operator==(const Edge &other) const {
      return std::tie(startIndex, endIndex, isUnitFlag, conflictPair) ==
             std::tie(other.startIndex, other.endIndex, other.isUnitFlag,
                      other.conflictPair);
    }

    bool operator!=(const Edge &other) const { return !(*this == other); }
  };

  llvm::DenseMap<CorePipeInfo,
                 llvm::DenseMap<CorePipeInfo, llvm::SmallVector<Edge>>>
      adjacencyList, tempAdjacencyList;

  GraphSolverUnitFlag(const SyncSolverOptions &options)
      : GraphSolverBase(options) {}

  void clearAdjList(bool isTemp = false) override;

  void insertEdge(CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                  ConflictPair *conflictPair, bool isTemp = false) override;
  void eraseEdge(CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                 ConflictPair *conflictPair, bool isTemp = false) override;

  std::optional<int> runDijkstra(CorePipeInfo corePipeSrc,
                                 CorePipeInfo corePipeDst, int startIndex,
                                 int endIndex, Occurrence *occ1 = nullptr,
                                 Occurrence *occ2 = nullptr) override;
};
} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_GRAPHSOLVER_H
