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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include <set>

namespace mlir::hivm::syncsolver {

class GraphSolver {
public:
  struct Edge {
    mlir::hivm::PIPE pipeFromId{mlir::hivm::PIPE::PIPE_UNASSIGNED};
    mlir::hivm::PIPE pipeToId{mlir::hivm::PIPE::PIPE_UNASSIGNED};
    int startIndex{-1};
    int endIndex{-1};
    Edge() = delete;
    Edge(mlir::hivm::PIPE pipeFromId, mlir::hivm::PIPE pipeToId, int startIndex,
         int endIndex)
        : pipeFromId(pipeFromId), pipeToId(pipeToId), startIndex(startIndex),
          endIndex(endIndex) {}
    bool operator<(const Edge &other) const;
  };

  // adjacencyList[pipeFrom][pipeTo] stores a set of Edge objects representing
  // directed transitions from pipeFrom to pipeTo that are valid for a given
  // (startIndex,endIndex) lifetime. Used by runDijkstra to compute minimum
  // distance paths between two pipe ids taking ordering constraints into
  // account.
  llvm::DenseMap<mlir::hivm::PIPE,
                 llvm::DenseMap<mlir::hivm::PIPE, std::set<Edge>>>
      adjacencyList;

  // Add a pipe-pair edge annotated with its active index interval.
  void addPair(mlir::hivm::PIPE startPipeId, mlir::hivm::PIPE endPipeId,
               int startIndex, int endIndex);

  // Build adjacency list from a ConflictPair by decomposing it into edges.
  void addConflictPair(syncsolver::ConflictPair *conflictPair);

  // Compact or merge overlapping edges to speed up Dijkstra queries.
  void optimizeAdjacencyList();

  // Run shortest-path search (Dijkstra-like) with ordering constraints to find
  // the minimal reachable index for a path from startPipe to endPipe.
  std::optional<int> runDijkstra(mlir::hivm::PIPE startPipe,
                                 mlir::hivm::PIPE endPipe, int startIndex,
                                 int maxDistance);
};
} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_GRAPHSOLVER_H
