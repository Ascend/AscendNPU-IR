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
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"
#include "llvm/Support/Debug.h"
#include <optional>
#include <queue>
#include <utility>

#define DEBUG_TYPE "hivm-gss-graph-solver"

using namespace mlir;
using namespace hivm::syncsolver;

// Compare edges (used for ordered sets). Edges must share endpoints when
// compared.
bool GraphSolver::Edge::operator<(const Edge &other) const {
  assert(corePipeSrc == other.corePipeSrc);
  assert(corePipeDst == other.corePipeDst);
  if (startIndex != other.startIndex) {
    return startIndex < other.startIndex;
  }
  return endIndex < other.endIndex;
}

// Add an adjacency edge annotated with an active index interval.
void GraphSolver::addPair(CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                          int startIndex, int endIndex,
                          int conflictPairDebugId) {
  Edge edge(corePipeSrc, corePipeDst, startIndex, endIndex,
            conflictPairDebugId);
  adjacencyList[edge.corePipeSrc][edge.corePipeDst].insert(edge);
}

// Convert a ConflictPair into adjacency edges (handles PIPE_ALL
// special-casing).
void GraphSolver::addConflictPair(ConflictPair *conflictPair) {
  assert(conflictPair != nullptr);
  DEBUG_WITH_TYPE("gss-graph-solver-add-conflict-pair", {
    llvm::dbgs() << "add-conflict-pair:\n";
    llvm::dbgs() << conflictPair->str() << '\n';
  });
  if (conflictPair->isBarrier() &&
      conflictPair->setCorePipeInfo.pipe == hivm::PIPE::PIPE_ALL) {
    for (int i = 0; i < static_cast<int>(hivm::PIPE::PIPE_NUM); i++) {
      auto coreSrc = conflictPair->setCorePipeInfo.coreType;
      auto setPipe = static_cast<hivm::PIPE>(i);
      auto coreDst = conflictPair->waitCorePipeInfo.coreType;
      auto waitPipe = hivm::PIPE::PIPE_ALL;
      assert(coreSrc == coreDst);
      int startIndex = conflictPair->startIndex;
      int endIndex = conflictPair->endIndex;
      assert(startIndex == endIndex);
      addPair(CorePipeInfo(coreSrc, setPipe), CorePipeInfo(coreDst, waitPipe),
              startIndex, endIndex, conflictPair->id);
    }
  } else if (!conflictPair->isBarrier() &&
             conflictPair->waitCorePipeInfo.pipe == hivm::PIPE::PIPE_S) {
    for (int i = 0; i < static_cast<int>(hivm::PIPE::PIPE_NUM); i++) {
      auto coreDst = conflictPair->waitCorePipeInfo.coreType;
      auto waitPipe = static_cast<hivm::PIPE>(i);
      addPair(conflictPair->setCorePipeInfo, CorePipeInfo(coreDst, waitPipe),
              conflictPair->startIndex, conflictPair->endIndex,
              conflictPair->id);
    }
  } else {
    auto corePipeSrc = conflictPair->setCorePipeInfo;
    auto corePipeDst = conflictPair->waitCorePipeInfo;
    int startIndex = conflictPair->startIndex;
    int endIndex = conflictPair->endIndex;
    addPair(corePipeSrc, corePipeDst, startIndex, endIndex, conflictPair->id);
  }
}

// Compact adjacency lists by removing dominated edges to accelerate queries.
void GraphSolver::optimizeAdjacencyList() {
  for (auto &[corePipeSrc, dstMap] : adjacencyList) {
    for (auto &[corePipeDst, edges] : dstMap) {
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
std::optional<int> GraphSolver::runDijkstra(CorePipeInfo corePipeSrc,
                                            CorePipeInfo corePipeDst,
                                            int startIndex, int endIndex) {
  llvm::DenseMap<CorePipeInfo, int> distance;
  std::priority_queue<std::pair<int, CorePipeInfo>,
                      std::vector<std::pair<int, CorePipeInfo>>,
                      std::greater<std::pair<int, CorePipeInfo>>>
      que;
  que.emplace(startIndex, corePipeSrc);
  auto [coreDst, pipeDst] = corePipeDst;

  LLVM_DEBUG(llvm::dbgs() << "dij-start-end-indices: " << startIndex << ' '
                          << endIndex << '\n');

  while (!que.empty()) {
    auto [curIndex, curCorePipe] = que.top();
    auto [curCore, curPipe] = curCorePipe;
    que.pop();

    LLVM_DEBUG(llvm::dbgs() << "dij-step: " << curCore << ' ' << curPipe << ' '
                            << curIndex << '\n');

    if (curCorePipe == corePipeDst && distance.count(corePipeDst)) {
      break;
    }

    if (distance.count(curCorePipe) && distance[curCorePipe] < curIndex) {
      continue;
    }

    if (distance.count(curCorePipe) && distance[curCorePipe] > endIndex) {
      break;
    }

    if (curCore == coreDst &&
        ((curIndex != startIndex && curPipe == hivm::PIPE::PIPE_S) ||
         curPipe == hivm::PIPE::PIPE_ALL)) {
      distance[corePipeDst] = curIndex;
      break;
    }

    for (auto &[endCorePipe, edges] : adjacencyList[curCorePipe]) {
      auto it =
          edges.lower_bound(Edge(curCorePipe, endCorePipe, curIndex, -1, -1));
      for (; it != edges.end(); it++) {
        if (!distance.count(endCorePipe) ||
            (distance[endCorePipe] > (it->endIndex))) {
          distance[endCorePipe] = it->endIndex;
          que.emplace(it->endIndex, endCorePipe);
        }
      }
    }
  }

  return distance.count(corePipeDst) ? distance[corePipeDst]
                                     : std::optional<int>();
}
