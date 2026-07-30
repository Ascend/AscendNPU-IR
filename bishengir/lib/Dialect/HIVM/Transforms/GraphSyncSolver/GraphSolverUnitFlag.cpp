//===------ GraphSolverUnitFlag.cpp ---- Graph Sync Solver ----------------===//
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

void GraphSolverUnitFlag::addPair(CorePipeInfo corePipeSrc,
                                  CorePipeInfo corePipeDst,
                                  ConflictPair *conflictPair) {
  Edge edge(conflictPair->startIndex, conflictPair->endIndex,
            conflictPair->replacedWithUnitFlag, conflictPair);
  adjacencyList[corePipeSrc][corePipeDst].emplace_back(std::move(edge));
}

std::optional<int> GraphSolverUnitFlag::runDijkstra(
    CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst, int startIndex,
    int endIndex, Occurrence *occ1, Occurrence *occ2) {
  using DistKey = std::tuple<int, int, CorePipeInfo>;
  struct QueElement {
    int index{-1};
    DistKey distKey;
    QueElement(int index, const DistKey &distKey)
        : index(index), distKey(distKey) {}
    bool operator>(const QueElement &other) const {
      return index > other.index;
    }
  };

  llvm::DenseMap<DistKey, int> distance;
  std::priority_queue<QueElement, std::vector<QueElement>,
                      std::greater<QueElement>>
      que;
  que.emplace(QueElement(startIndex, DistKey(false, false, corePipeSrc)));
  LLVM_DEBUG(llvm::dbgs() << "dij-start-end-indices: " << startIndex << ' '
                          << endIndex << '\n');

  while (!que.empty()) {
    auto curElement = que.top();
    auto curIndex = curElement.index;
    auto curDistKey = curElement.distKey;
    auto [curIsUnitFlag, curIsOccDst, curCorePipe] = curDistKey;
    que.pop();

    LLVM_DEBUG(llvm::dbgs() << "dij-step: " << curCorePipe.coreType << ' '
                            << curCorePipe.pipe << ' ' << curIsUnitFlag << ' '
                            << curIsOccDst << ' ' << curIndex << '\n');

    auto curIt = distance.find(curDistKey);
    if (curIt != distance.end()) {
      if (curIt->second < curIndex) {
        continue;
      }
      if (curCorePipe == corePipeDst && !(curIsUnitFlag && !curIsOccDst)) {
        return curIndex;
      }
    }

    if (curCorePipe.coreType == corePipeDst.coreType) {
      if (curIndex != startIndex && curCorePipe.pipe == hivm::PIPE::PIPE_S) {
        return curIndex;
      }
      if (curCorePipe.pipe == hivm::PIPE::PIPE_ALL) {
        return curIndex;
      }
    }

    for (auto &[endCorePipe, edges] : adjacencyList[curCorePipe]) {
      for (auto &edge : edges) {
        if (edge.startIndex < curIndex || edge.endIndex > endIndex) {
          continue;
        }
        if (edge.isUnitFlag) {
          if (curIndex == startIndex && edge.startIndex != startIndex) {
            continue;
          }
        }
        assert(edge.conflictPair != nullptr);
        DistKey nextKey(edge.isUnitFlag, (edge.conflictPair->waitOcc == occ2),
                        endCorePipe);
        auto [nextIt, isInserted] = distance.insert({nextKey, edge.endIndex});
        if (isInserted || nextIt->second > edge.endIndex) {
          nextIt->second = edge.endIndex;
          que.emplace(QueElement(edge.endIndex, nextKey));
        }
      }
    }
  }

  std::optional<int> retDist;
  if (auto it = distance.find(
          DistKey(/*isUnitFlag=*/false, /*isOccDst=*/false, corePipeDst));
      it != distance.end()) {
    retDist = retDist.has_value() ? std::min(retDist.value(), it->second)
                                  : it->second;
  }
  if (auto it = distance.find(
          DistKey(/*isUnitFlag=*/false, /*isOccDst=*/true, corePipeDst));
      it != distance.end()) {
    retDist = retDist.has_value() ? std::min(retDist.value(), it->second)
                                  : it->second;
  }
  if (auto it = distance.find(
          DistKey(/*isUnitFlag=*/true, /*isOccDst=*/true, corePipeDst));
      it != distance.end()) {
    retDist = retDist.has_value() ? std::min(retDist.value(), it->second)
                                  : it->second;
  }
  return retDist;
}
