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
  auto args = getArgumentRefOrCreateDummy(dim.first);
  auto solverIndex = solverCollapserElem_->find(args[dim.second]);
  LDBG("Checking parallelDim of " << solverIndex << "("
                                  << solverShapeElem_->find(args[dim.second])
                                  << ")");
  auto tilingDimKindVal = tilingDimKindMap.find(solverIndex);
  if (tilingDimKindVal != tilingDimKindMap.end())
    return tilingDimKindVal->getSecond() == TilingDimensionKind::Parallel;
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
        LDBG("Candidate of " << parentIndex << " in group " << groupIndex
                             << " is "
                             << utils::debugger::to_string(candidateDims));
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

/// Tells us if we can still treat axis \p i as a tiling candidate for every
/// \c StoreOpTy, even when the *view* on that axis has unknown size or
/// size 1. This will try to recover the size of the parent buffer.
///
/// Example: axis \p i is 0. The store operands are \c memref.subview results;
/// axis 0 of each view may be `?` (or a length-1 row), but the parent buffers
/// are still \c 16x16, so the helper can recover size 16 for both sides.
/// \code
/// %subSrc = memref.subview %ub[0, 0] [1, 16] [1, 1]
///     : memref<16x16xf16, #hivm.address_space<ub>>
///     to memref<?x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
/// %subDst = memref.subview %gm[0, 0] [1, 16] [1, 1]
///     : memref<16x16xf16, #hivm.address_space<gm>>
///     to memref<?x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
/// hivm.store
///     ins(%subSrc : memref<?x16xf16, strided<[16, 1]>,
///     #hivm.address_space<ub>>) outs(%subDst : memref<?x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>)
/// \endcode
template <typename StoreOpTy>
static bool checkTileableMaskedStore(StoreOpTy storeOp, size_t i) {
  auto src = storeOp.getSrc();
  auto dst = storeOp.getDst();
  int64_t srcOrigDim = ShapedType::kDynamic;
  int64_t dstOrigDim = ShapedType::kDynamic;
  if (auto extractSliceOp = src.template getDefiningOp<tensor::ExtractSliceOp>()) {
    srcOrigDim = extractSliceOp.getSourceType().getDimSize(i);
  } else if (auto subviewOp = src.template getDefiningOp<memref::SubViewOp>()) {
    srcOrigDim = subviewOp.getSourceType().getDimSize(i);
  }
  if (auto extractSliceOp = dst.template getDefiningOp<tensor::ExtractSliceOp>()) {
    dstOrigDim = extractSliceOp.getSourceType().getDimSize(i);
  } else if (auto subviewOp = dst.template getDefiningOp<memref::SubViewOp>()) {
    dstOrigDim = subviewOp.getSourceType().getDimSize(i);
  }
  return srcOrigDim != ShapedType::kDynamic && srcOrigDim != 1 &&
         srcOrigDim == dstOrigDim;
}

/// Walks every \c StoreOpTy under the analyzed op. For each source axis that
/// is parallel, appends that \c Dimension to \p parallelDimMap (keyed by solver
/// group and parent index) unless the axis is dynamic or size 1; for
/// \c hivm::StoreOp, \c checkTileableMaskedStore may still allow the latter.
///
/// A \b solver group is a union-find bucket: during analysis, \c solverGroup_
/// (a \c SimpleUnionFind over \c argumentsRefPointer_ indices)
/// (See DimensionAnalyzer::processBFS)
///
/// The implementation runs \c op_->walk<WalkOrder::PreOrder> on a root op
/// with a callback that only fires for operations of type \c StoreOpTy.
/// Each time a store is hit, \c numStoreOps for that source's solver group
/// (\c srcRef) is bumped, then we scan every rank index \c i of
/// the source: if the dimension is parallel and passes
/// the dynamic/broadcast (and optional masked-store) rules, we record
/// (Value, DimensionIndex) pair under parallelDimMap for that group and
/// collapsed parent index. The same op kind is used for \c hivm::CopyOp, \c
/// hivm::FixpipeOp depending on how \c computeTilingDim was invoked.
template <typename StoreOpTy>
void DimensionAnalyzer::computeTilingDimImpl(
    DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>>
        &parallelDimMap,
    DenseMap<int64_t, int> &numStoreOps) {
  op_->walk<WalkOrder::PreOrder>([&](StoreOpTy op) {
    auto src = op.getSrc();
    auto rank = utils::getShapeRank(src.getType()).value_or(0);
    auto args = getArgumentRefOrCreateDummy(src);
    // See that each tiling dimension differs for each \c groupIndex.
    // Each operation in a group is independent, horizontal, and totally
    // separated to other operations in a different group. In common kernels,
    // there will only be 1 group.
    auto groupIndex = solverGroup_->find(argumentsRefPointer_.at(src));
    numStoreOps[groupIndex]++;
    LDBG("Checking operation: " << op << " in group " << groupIndex);
    if (rank == 0)
      return;
    auto shape = utils::getShape(src.getType());
    DenseSet<int> usedParentIdx;
    for (size_t i = 0; i < rank; i++) {
      Dimension dim(src, i);
      if (isParallelDim(dim)) {
        if (ShapedType::isDynamic(shape[i]) || shape[i] == 1) {
          if (!checkTileableMaskedStore(op, i))
            continue;
        }
        LDBG("Dim " << i << " is selected in group " << groupIndex);
        auto parentIndex = solverCollapserElem_->find(args[i]);
        if (usedParentIdx.insert(parentIndex).second) {
          parallelDimMap[groupIndex][parentIndex].push_back(dim);
        } else {
          op->emitWarning()
              << "Detected dimensions are in the same group in one "
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