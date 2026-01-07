//===- DimensionAnalyzer.cpp ----------------------------------------------===//
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

#include "bishengir/Dialect/HIVM/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/Utils/Util.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::utils::debugger;

#define DEBUG_TYPE "dimension-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hivm {
namespace detail {

bool DimensionAnalyzer::isParallelDim(Dimension dim) {
  auto solverIndex = solverCollapserElem_->find(
      getArgumentRefOrCreateDummy(dim.first)[dim.second]);
  LDBG("Checking parallelDim of " << solverIndex);
  auto tilingDimKindVal = tilingDimKindMap.find(solverIndex);
  if (tilingDimKindVal != tilingDimKindMap.end()) {
    return tilingDimKindVal->getSecond() == TilingDimensionKind::Parallel;
  }
  // By default, assume it's parallel
  return true;
}

/// Get the optimal tiling dimension for each value in the operation.
/// Analyzes parallel dimensions across all storeOp and selects
/// the dimension that appears most frequently as a parallel dimension.
/// Uses a heuristic where if the majority of stores have a higher dimension
/// available, that dimension is chosen for tiling.
void DimensionAnalyzer::computeTilingDim(bool isVectorOp) {
  DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>> parallelDimMaps;
  DenseMap<int64_t, int> numStoreOps;
  for (auto [value, _] : argumentsRefPointer_)
    tilingDim_[value] = -1;

  isVectorOp ? computeTilingDimImpl<hivm::StoreOp>(parallelDimMaps, numStoreOps)
             : computeTilingDimImpl<hivm::FixpipeOp>(parallelDimMaps, numStoreOps);

  DenseMap<int64_t, int> selectedTilingParIdxMap;
  for (const auto &[groupIndex, parallelDimMap] : parallelDimMaps) {
    auto numStoreOp = numStoreOps.at(groupIndex);
    for (const auto &[parentIndex, candidate] : parallelDimMap) {
      if (static_cast<int64_t>(candidate.size()) == numStoreOp) {
        int64_t higherDimCnt = 0;
        for (auto [store, dim] : candidate) {
          int64_t &curDim = tilingDim_[store];
          if (curDim == -1 || curDim > dim)
            higherDimCnt++;
        }
        // try to find majority of dimension is higher
        if (2 * higherDimCnt >= numStoreOp) {
          selectedTilingParIdxMap[groupIndex] = parentIndex;
          for (auto [store, dim] : candidate)
            tilingDim_[store] = dim;
        }
      }
    }
  }
  for (auto[_, parIdx] : selectedTilingParIdxMap)
    selectedTilingParIdx.insert(parIdx);
  LDBG("Number of independent set: " << selectedTilingParIdx.size());
}

int64_t DimensionAnalyzer::getTilingDim(Value v) {
  if (!argumentsRefPointer_.contains(v))
    return -1;
  auto argRef = argumentsRefPointer_.at(v);
  auto rank = utils::getShapeRank(v.getType()).value_or(0);
  for (size_t i = 0; i < rank; i++) {
    auto parentIndex = solverCollapserElem_->find(getArgumentRef(v)[i]);
    if (selectedTilingParIdx.contains(parentIndex))
      return i;
  }
  return -1;
}

template <typename StoreOpTy>
void DimensionAnalyzer::computeTilingDimImpl(
    DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>> &parallelDimMap,
    DenseMap<int64_t, int> &numStoreOps) {
  op_->walk<WalkOrder::PreOrder>([&](StoreOpTy op) {
    auto src = op.getSrc();
    auto rank = utils::getShapeRank(src.getType()).value_or(0);
    auto args = getArgumentRefOrCreateDummy(src);
    auto srcRef = solverGroup_->find(argumentsRefPointer_.at(src));
    numStoreOps[srcRef]++;
    for (size_t i = 0; i < rank; i++) {
      Dimension dim(src, i);
      if (isParallelDim(dim)) {
        auto parentIndex = solverCollapserElem_->find(args[i]);
        parallelDimMap[srcRef][parentIndex].push_back(dim);
      }
    }
  });
}

} // namespace detail
} // namespace hivm
} // namespace mlir