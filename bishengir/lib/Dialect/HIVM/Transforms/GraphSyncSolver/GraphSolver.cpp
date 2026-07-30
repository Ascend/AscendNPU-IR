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

void GraphSolver::addPair(CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                          ConflictPair *conflictPair) {
  Edge edge(conflictPair->startIndex, conflictPair->endIndex);
  adjacencyList[corePipeSrc][corePipeDst].emplace_back(std::move(edge));
}

void GraphSolver::addConflictPair(ConflictPair *conflictPair) {
  assert(conflictPair != nullptr);
  DEBUG_WITH_TYPE("gss-graph-solver-add-conflict-pair", {
    llvm::dbgs() << "add-conflict-pair:\n";
    llvm::dbgs() << conflictPair->str() << '\n';
  });
  if (conflictPair->isBarrier()) {
    if (conflictPair->setCorePipeInfo.pipe == hivm::PIPE::PIPE_ALL) {
      llvm::SmallVector<std::tuple<hivm::TCoreType, hivm::TCoreType>>
          srcDstCores;
      if (options.isCrossCoreMode()) {
        srcDstCores = {
            {hivm::TCoreType::CUBE, hivm::TCoreType::VECTOR},
            {hivm::TCoreType::VECTOR, hivm::TCoreType::CUBE},
        };
      } else {
        srcDstCores = {
            {hivm::TCoreType::CUBE_OR_VECTOR, hivm::TCoreType::CUBE_OR_VECTOR},
        };
      }
      for (auto &[srcCore, dstCore] : srcDstCores) {
        for (int i = 0; i < static_cast<int>(hivm::PIPE::PIPE_NUM); i++) {
          auto setPipe = static_cast<hivm::PIPE>(i);
          auto waitPipe = hivm::PIPE::PIPE_ALL;
          addPair(CorePipeInfo(srcCore, setPipe),
                  CorePipeInfo(dstCore, waitPipe), conflictPair);
        }
      }
    } else {
      addPair(conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo,
              conflictPair);
    }
  } else {
    if (conflictPair->waitCorePipeInfo.pipe == hivm::PIPE::PIPE_S) {
      for (int i = 0; i < static_cast<int>(hivm::PIPE::PIPE_NUM); i++) {
        auto coreDst = conflictPair->waitCorePipeInfo.coreType;
        auto waitPipe = static_cast<hivm::PIPE>(i);
        addPair(conflictPair->setCorePipeInfo, CorePipeInfo(coreDst, waitPipe),
                conflictPair);
      }
    } else {
      addPair(conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo,
              conflictPair);
    }
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

    auto curIt = distance.find(curCorePipe);
    if (curIt != distance.end()) {
      if (curIt->second < curIndex) {
        continue;
      }
      if (curCorePipe == corePipeDst) {
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
        auto [nextIt, isInserted] =
            distance.insert({endCorePipe, edge.endIndex});
        if (isInserted || nextIt->second > edge.endIndex) {
          nextIt->second = edge.endIndex;
          que.emplace(QueElement(edge.endIndex, endCorePipe));
        }
      }
    }
  }

  return {};
}
