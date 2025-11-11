//===----------- GraphSolver.cpp ---- Graph Sync Solver -------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/GraphSolver.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include <optional>
#include <queue>

#define DEBUG_TYPE "hivm-graph-sync-solver-dij"

using namespace mlir;
using namespace hivm::syncsolver;

// Compare edges (used for ordered sets). Edges must share endpoints when
// compared.
bool GraphSolver::Edge::operator<(const Edge &other) const {
  assert(pipeFromId == other.pipeFromId && pipeToId == other.pipeToId);
  if (startIndex != other.startIndex) {
    return startIndex < other.startIndex;
  }
  return endIndex < other.endIndex;
}

// Add an adjacency edge annotated with an active index interval.
void GraphSolver::addPair(hivm::PIPE startPipeId, hivm::PIPE endPipeId,
                          int startIndex, int endIndex) {
  Edge edge(startPipeId, endPipeId, startIndex, endIndex);
  adjacencyList[startPipeId][endPipeId].insert(edge);
}

// Convert a ConflictPair into adjacency edges (handles PIPE_ALL
// special-casing).
void GraphSolver::addConflictPair(ConflictPair *conflictPair) {
  assert(conflictPair != nullptr);
  if (conflictPair->isBarrier() &&
      conflictPair->setPipe == hivm::PIPE::PIPE_ALL) {
    for (int i = 0; i < static_cast<int>(hivm::PIPE::PIPE_NUM); i++) {
      int startIndex = conflictPair->startIndex;
      int endIndex = conflictPair->endIndex;
      assert(startIndex == endIndex);
      auto setPipe = static_cast<hivm::PIPE>(i);
      auto waitPipe = hivm::PIPE::PIPE_ALL;
      addPair(setPipe, waitPipe, startIndex, endIndex);
    }
  } else {
    int startIndex = conflictPair->startIndex;
    int endIndex = conflictPair->endIndex;
    auto setPipe = conflictPair->setPipe;
    auto waitPipe = conflictPair->waitPipe;
    addPair(setPipe, waitPipe, startIndex, endIndex);
  }
}

// Compact adjacency lists by removing dominated edges to accelerate queries.
void GraphSolver::optimizeAdjacencyList() {
  for (auto &[startPipeId, toMap] : adjacencyList) {
    for (auto &[endPipeId, edges] : toMap) {
      std::set<Edge> optimizedEdges;
      for (auto &edge : edges) {
        while (!optimizedEdges.empty() &&
               optimizedEdges.rbegin()->endIndex >= edge.endIndex) {
          optimizedEdges.erase(*optimizedEdges.rbegin());
        }
        optimizedEdges.insert(edge);
      }
      edges = std::move(optimizedEdges);
    }
  }
}

// Run a Dijkstra-like search over pipes using index intervals as
// weights/constraints. Returns minimal reachable index for endPipe or empty
// optional if unreachable.
std::optional<int> GraphSolver::runDijkstra(hivm::PIPE startPipe,
                                            hivm::PIPE endPipe, int startIndex,
                                            int maxDistance) {
  llvm::DenseMap<hivm::PIPE, int> distance;
  std::priority_queue<std::pair<int, hivm::PIPE>,
                      std::vector<std::pair<int, hivm::PIPE>>,
                      std::greater<std::pair<int, hivm::PIPE>>>
      que;
  que.emplace(startIndex, startPipe);

  while (!que.empty()) {
    auto [endIndex, curPipe] = que.top();
    que.pop();

    if (curPipe == endPipe && distance.count(curPipe)) {
      break;
    }

    if (distance.count(curPipe) && distance[curPipe] < endIndex) {
      continue;
    }

    if (distance.count(curPipe) && distance[curPipe] > maxDistance) {
      break;
    }

    if (curPipe == hivm::PIPE::PIPE_ALL) {
      distance[endPipe] = endIndex;
      break;
    }

    for (auto &[endPipe, edges] : adjacencyList[curPipe]) {
      auto it = edges.lower_bound(Edge(curPipe, endPipe, endIndex, -1));
      for (; it != edges.end(); it++) {
        if (!distance.count(endPipe) || distance[endPipe] > (it->endIndex)) {
          distance[endPipe] = it->endIndex;
          que.emplace(it->endIndex, endPipe);
        } else {
          break;
        }
      }
    }
  }

  return distance.count(endPipe) ? distance[endPipe] : std::optional<int>();
}
