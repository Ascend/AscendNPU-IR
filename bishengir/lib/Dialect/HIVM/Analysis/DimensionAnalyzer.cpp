//===- DimensionAnalyzer.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  auto args = getArgumentRefOrCreateDummy(dim.first);
  auto solverIndex = solverCollapserElem_->find(args[dim.second]);
  LDBG("Checking parallelDim of " << solverIndex << "("
                                  << solverShapeElem_->find(args[dim.second])
                                  << ")");
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
bool DimensionAnalyzer::computeTilingDim(bool isVectorOp) {
  DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>> parallelDimMaps;
  DenseMap<int64_t, int> numStoreOps;
  DenseMap<int64_t, SmallVector<Dimension>> parallelDimMap;
  bool isBroadcastAxisCase = false;
  for (auto [value, _] : argumentsRefPointer_)
    tilingDim_[value] = -1;

  if (isVectorOp) {
    computeTilingDimImpl<hivm::StoreOp>(parallelDimMaps, numStoreOps);
    computeTilingDimImpl<hivm::CopyOp>(parallelDimMaps, numStoreOps);
  } else {
    computeTilingDimImpl<hivm::FixpipeOp>(parallelDimMaps, numStoreOps);
  }

  DenseMap<int64_t, int> selectedTilingParIdxMap;
  for (const auto &[groupIndex, parallelDimMap] : parallelDimMaps) {
    auto numStoreOp = numStoreOps.at(groupIndex);
    LDBG("Group " << groupIndex << " has " << numStoreOp << " operations");
    for (const auto &[parentIndex, candidate] : parallelDimMap) {
      if (static_cast<int64_t>(candidate.size()) == numStoreOp) {
        int64_t higherDimCnt = 0;
        SmallVector<int64_t> candidateDims;
        for (auto [store, cDim] : candidate) {
          auto storeRef = getArgumentRef(store);
          int64_t curDim = tilingDim_[store];
          auto dim = cDim;
          auto solverIndex = solverShapeElem_->find(storeRef[dim]);
          LDBG("Checking if " << solverIndex << " is transposed dim");
          if (transposedDimMap.contains(solverIndex)) {
            LDBG("Found transposed mapping("
                 << solverIndex << "): " << dim << " to "
                 << transposedDimMap.at(solverIndex));
            dim = transposedDimMap.at(solverIndex);
          }
          if (curDim != -1) {
            solverIndex = solverShapeElem_->find(storeRef[curDim]);
            if (transposedDimMap.contains(solverIndex))
              curDim = transposedDimMap.at(solverIndex);
          }
          candidateDims.push_back(dim);
          if (curDim == -1 || curDim > dim)
            higherDimCnt++;
        }
        LDBG("Candidate of " << parentIndex << " in group " << groupIndex << " is " << utils::debugger::to_string(candidateDims));
        // try to find majority of dimension is higher
        if (2 * higherDimCnt >= numStoreOp) {
          selectedTilingParIdxMap[groupIndex] = parentIndex;
          for (auto [store, dim] : candidate)
            tilingDim_[store] = dim;
        }
      }
    }
  }
  LDBG("Selected independent tiling dims: " << selectedTilingParIdxMap.size());
  for (auto [_, parIdx] : selectedTilingParIdxMap) {
    selectedTilingParIdx.insert(parIdx);
    isBroadcastAxisCase |= broadcastAxisCaseCandidate.contains(parIdx);
  }
  LDBG(utils::debugger::to_string(selectedTilingParIdx));
  return isBroadcastAxisCase;
}

int64_t DimensionAnalyzer::getTilingDim(Value v) {
  if (!argumentsRefPointer_.contains(v))
    return -1;
  auto rank = utils::getShapeRank(v.getType()).value_or(0);
  int64_t tilingDim = -1;
  int order = -1;
  auto args = getArgumentRef(v);
  for (size_t i = 0; i < rank; i++) {
    auto parentIndex = solverCollapserElem_->find(args[i]);
    if (selectedTilingParIdx.contains(parentIndex)) {
      auto solverIndex = solverShapeElem_->find(args[i]);
      int candOrder = (int)i;
      if (auto it = transposedDimMap.find(solverIndex);
          it != transposedDimMap.end()) {
        candOrder = it->second;
      }
      if (tilingDim == -1 || order > candOrder) {
        tilingDim = (int64_t)i;
        order = candOrder;
      }
    }
  }
  return tilingDim;
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
    LDBG("Checking operation: " << op << " in group " << srcRef);
    if (rank == 0)
      return;
    auto shape = utils::getShape(src.getType());
    DenseSet<int> usedParentIdx;
    for (size_t i = 0; i < rank; i++) {
      Dimension dim(src, i);
      if (isParallelDim(dim) && shape[i] != 1) {
        LDBG("Dim " << i << " is selected in group " << srcRef);
        if (ShapedType::isDynamic(shape[i])) {
          if (auto extractSliceOp = src.template getDefiningOp<tensor::ExtractSliceOp>();
              extractSliceOp && extractSliceOp.getSourceType().getDimSize(i) == 1)
              continue;
          if (auto subviewOp = src.template getDefiningOp<memref::SubViewOp>();
              subviewOp && subviewOp.getSourceType().getDimSize(i) == 1)
              continue;
        }
        auto parentIndex = solverCollapserElem_->find(args[i]);
        if (usedParentIdx.insert(parentIndex).second) {
          parallelDimMap[srcRef][parentIndex].push_back(dim);
        } else {
          op->emitWarning() << "Detected dimensions are in the same group in one "
                               "storeOp. It is recommended to try with "
                               "strict-mode=false if TileAndBindSubBlock fails";
          broadcastAxisCaseCandidate.insert(parentIndex);
        }
      }
    }
  });
}

} // namespace detail
} // namespace hivm
} // namespace mlir