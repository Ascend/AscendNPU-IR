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

void GraphSolver::clearAdjList(bool isTemp) {
  if (isTemp) {
    tempAdjacencyList.clear();
  } else {
    adjacencyList.clear();
  }
}

void GraphSolver::addPair(CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                          ConflictPair *conflictPair, bool isTemp) {
  Edge edge(conflictPair->startIndex, conflictPair->endIndex);
  if (isTemp) {
    tempAdjacencyList[corePipeSrc][corePipeDst].emplace_back(std::move(edge));
  } else {
    adjacencyList[corePipeSrc][corePipeDst].emplace_back(std::move(edge));
  }
}

std::optional<int> GraphSolver::runDijkstra(CorePipeInfo corePipeSrc,
                                            CorePipeInfo corePipeDst,
                                            int startIndex, int endIndex,
                                            Occurrence *occ1,
                                            Occurrence *occ2) {
  (void)occ1;
  (void)occ2;
  llvm::DenseMap<CorePipeInfo, int> distance;
  struct QueElement {
    int index{-1};
    CorePipeInfo corePipe;
    QueElement(int index, CorePipeInfo corePipe)
        : index(index), corePipe(corePipe) {}
    bool operator>(const QueElement &other) const {
      return index > other.index;
    }
  };
  std::priority_queue<QueElement, std::vector<QueElement>,
                      std::greater<QueElement>>
      que;
  que.emplace(QueElement(startIndex, corePipeSrc));

  LLVM_DEBUG(llvm::dbgs() << "dij-start-end-indices: " << startIndex << ' '
                          << endIndex << '\n');

  while (!que.empty()) {
    auto curElement = que.top();
    auto curIndex = curElement.index;
    auto curCorePipe = curElement.corePipe;
    que.pop();

    LLVM_DEBUG(llvm::dbgs() << "dij-step: " << curCorePipe.coreType << ' '
                            << curCorePipe.pipe << ' ' << curIndex << '\n');

    auto curDistIt = distance.find(curCorePipe);
    if (curDistIt != distance.end()) {
      if (curDistIt->second < curIndex) {
        continue;
      }
      if (curCorePipe == corePipeDst) {
        return curIndex;
      }
    }

    if (curCorePipe.coreType == corePipeDst.coreType) {
      if (curDistIt != distance.end() &&
          curCorePipe.pipe == hivm::PIPE::PIPE_S) {
        return curIndex;
      }
      if (curCorePipe.pipe == hivm::PIPE::PIPE_ALL) {
        return curIndex;
      }
    }

    llvm::SmallVector<CorePipeInfo> startCorePipeInfos = {curCorePipe};
    if (curDistIt != distance.end() && curCorePipe.pipe == hivm::PIPE::PIPE_S) {
      for (auto &[startCorePipe, map] : adjacencyList) {
        if (startCorePipe.coreType == curCorePipe.coreType) {
          startCorePipeInfos.push_back(startCorePipe);
        }
      }
    }

    auto processAdjList = [&](auto &adjList, CorePipeInfo startCorePipe) {
      for (auto &[endCorePipe, edges] : adjList[startCorePipe]) {
        for (auto &edge : edges) {
          if (edge.startIndex < curIndex || edge.endIndex > endIndex) {
            continue;
          }
          auto [nextIt, isInserted] =
              distance.insert({endCorePipe, edge.endIndex});
          if (isInserted || (nextIt->second > edge.endIndex)) {
            nextIt->second = edge.endIndex;
            que.emplace(QueElement(edge.endIndex, endCorePipe));
          }
        }
      }
    };
    for (auto startCorePipe : startCorePipeInfos) {
      processAdjList(adjacencyList, startCorePipe);
      processAdjList(tempAdjacencyList, startCorePipe);
    }
  }

  return {};
}
