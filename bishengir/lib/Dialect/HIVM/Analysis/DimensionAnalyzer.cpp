//===- DimensionAnalyzer.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/STLExtras.h"
#include <type_traits>

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
  createDummyRefIfNotExist({dim.first});
  auto args = getValueDimIndices(dim.first);
  auto solverCollapserIndex = structuralDsu_->find(args[dim.second]);
  auto solverShapeIndex = equivalentDsu_->find(args[dim.second]);
  LDBG("Checking parallelDim of " << solverCollapserIndex << "("
                                  << solverShapeIndex << ")");
  auto tilingDimKindVal =
      tilingDimKindMapForCollapser.find(solverCollapserIndex);
  if (tilingDimKindVal != tilingDimKindMapForCollapser.end()) {
    if (tilingDimKindVal->getSecond() != TilingDimensionKind::Parallel &&
        broadcastAxisCaseCandidate.find(solverCollapserIndex) !=
            broadcastAxisCaseCandidate.end()) {
      if (auto it = tilingDimKindMapForShape.find(solverShapeIndex);
          it != tilingDimKindMapForShape.end()) {
        LDBG("Checking parallelDim for broadcast two dims case: "
             << static_cast<int>(it->getSecond()));
        return it->getSecond() == TilingDimensionKind::Parallel ||
               it->getSecond() == TilingDimensionKind::Reduce;
      }
      return true;
    }
    return tilingDimKindVal->getSecond() == TilingDimensionKind::Parallel ||
           tilingDimKindVal->getSecond() == TilingDimensionKind::Reduce;
  }

  // By default, assume it's parallel
  return true;
}

bool DimensionAnalyzer::isReduceDim(Dimension dim) {
  createDummyRefIfNotExist({dim.first});
  auto args = getValueDimIndices(dim.first);
  auto solverCollapserIndex = structuralDsu_->find(args[dim.second]);
  auto solverShapeIndex = equivalentDsu_->find(args[dim.second]);
  LDBG("Checking reduceDim of " << solverCollapserIndex << "("
                                << solverShapeIndex << ")");
  auto tilingDimKindVal =
      tilingDimKindMapForCollapser.find(solverCollapserIndex);
  if (tilingDimKindVal != tilingDimKindMapForCollapser.end()) {
    if (tilingDimKindVal->getSecond() != TilingDimensionKind::Parallel &&
        broadcastAxisCaseCandidate.find(solverCollapserIndex) !=
            broadcastAxisCaseCandidate.end()) {
      if (auto it = tilingDimKindMapForShape.find(solverShapeIndex);
          it != tilingDimKindMapForShape.end()) {
        LDBG("Checking parallelDim for broadcast two dims case: "
             << static_cast<int>(it->getSecond()));
        return it->getSecond() == TilingDimensionKind::Reduce;
      }
      return false;
    }
    return tilingDimKindVal->getSecond() == TilingDimensionKind::Reduce;
  }

  return false;
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
  bool isBroadcastAxisCase = !invalidUpdates.empty();

  for (auto [value, _] : valueToDimIndicesIndex_)
    tilingDim_[value] = -1;

  if (isVectorOp) {
    computeTilingDimImpl<hivm::StoreOp>(parallelDimMaps, numStoreOps);
    computeTilingDimImpl<hivm::CopyOp>(parallelDimMaps, numStoreOps);
    computeTilingDimImpl<hivm::StrideStoreOp>(parallelDimMaps, numStoreOps);
    computeTilingDimImpl<hivm::IndirectStoreOp>(parallelDimMaps, numStoreOps);
    computeTilingDimImpl<hivm::VReduceOp>(parallelDimMaps, numStoreOps);
  } else {
    computeTilingDimImpl<hivm::FixpipeOp>(parallelDimMaps, numStoreOps);
  }
  computeTilingDimImpl<hivm::DebugOp>(parallelDimMaps, numStoreOps);

  mlir::detail::SimpleUnionFind candGroupDSU(argumentTotalLength_);
  SmallVector<int64_t> candidateGroupSize(argumentTotalLength_);
  for (const auto &[groupIndex, parallelDimMap] : parallelDimMaps) {
    for (const auto &[parentIndex, candidate] : parallelDimMap)
      candidateGroupSize[parentIndex] = static_cast<int64_t>(candidate.size());
  }
  LDBG("Connecting invalid updates: ");
  DenseSet<std::pair<int, int>> invalidConnection;
  for (auto &[repIdx, indices] : invalidUpdates) {
    if (llvm::any_of(indices,
                     [repIdx = repIdx](auto idx) { return idx == repIdx; })) {
      for (auto idx : indices) {
        if (idx != repIdx) {
          invalidConnection.insert({idx, repIdx});
          invalidConnection.insert({repIdx, idx});
        }
      }
    }
  }
  for (auto &[repIdx, indices] : invalidUpdates) {
    auto idxToMerge = indices[0];
    auto groupIdx = valueGroupDSU_->find(dimIdxToArgIdx_[idxToMerge]);
    for (auto [store, dim] : parallelDimMaps[groupIdx][idxToMerge])
      tilingDim_[store] = dim;
    for (auto &idx : llvm::drop_begin(indices)) {
      if (invalidConnection.contains({repIdx, idx}) ||
          exclusiveDimIdx[repIdx].contains(idx))
        continue;
      auto curSize = candidateGroupSize[candGroupDSU.find(idxToMerge)];
      auto size = candidateGroupSize[candGroupDSU.find(idx)];
      if (curSize < size || invalidConnection.contains({repIdx, idxToMerge})) {
        idxToMerge = idx;
        for (auto [store, dim] : parallelDimMaps[groupIdx][idxToMerge])
          tilingDim_[store] = dim;
      } else if (curSize == size) {
        if (2 * getHigherDimCounts(parallelDimMaps[groupIdx][idx]) >= size) {
          idxToMerge = idx;
          for (auto [store, dim] : parallelDimMaps[groupIdx][idxToMerge])
            tilingDim_[store] = dim;
        }
      }
    }
    if (candGroupDSU.find(repIdx) == candGroupDSU.find(idxToMerge) ||
        invalidConnection.contains({repIdx, idxToMerge}) ||
        exclusiveDimIdx[repIdx].contains(idxToMerge))
      continue;
    LDBG("repIdx " << repIdx << "("
                   << candidateGroupSize[candGroupDSU.find(repIdx)]
                   << "): " << to_string(indices));
    auto totalSize = candidateGroupSize[candGroupDSU.find(repIdx)] +
                     candidateGroupSize[candGroupDSU.find(idxToMerge)];
    LDBG("Merging " << repIdx << " with " << idxToMerge << " to be "
                    << totalSize);
    candGroupDSU.join(repIdx, idxToMerge);
    candidateGroupSize[candGroupDSU.find(repIdx)] = totalSize;
  }

  for (auto [value, _] : valueToDimIndicesIndex_)
    tilingDim_[value] = -1;

  DenseMap<int64_t, int> selectedTilingParIdxMap;
  for (const auto &[groupIndex, parallelDimMap] : parallelDimMaps) {
    auto numStoreOp = 0;
    if (auto it = numStoreOps.find(groupIndex); it != numStoreOps.end())
      numStoreOp = it->second;
    LDBG("Group " << groupIndex << " has " << numStoreOp << " operations");
    for (const auto &[parentIndex, candidate] : parallelDimMap) {
      if (candidateGroupSize[candGroupDSU.find(parentIndex)] == numStoreOp) {
        SmallVector<int64_t> candidateDims;
        int64_t higherDimCnt = getHigherDimCounts(candidate, &candidateDims);

        LDBG("Candidate of "
             << parentIndex << " in group " << groupIndex << " is "
             << utils::debugger::to_string(candidateDims) << " with "
             << higherDimCnt << " priority dimensions");
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
  if (!valueToDimIndicesIndex_.contains(v))
    return -1;
  auto rank = utils::getShapeRank(v.getType()).value_or(0);
  int64_t tilingDim = -1;
  int order = -1;
  auto args = getValueDimIndices(v);
  for (size_t i = 0; i < rank; i++) {
    auto parentIndex = structuralDsu_->find(args[i]);
    if (selectedTilingParIdx.contains(parentIndex) &&
        isParallelDim(Dimension(v, i))) {
      auto solverIndex = equivalentDsu_->find(args[i]);
      int candOrder = static_cast<int>(i);
      if (auto it = transposedDimMap.find(solverIndex);
          it != transposedDimMap.end()) {
        candOrder = it->second;
      }
      if (isReduceDim(Dimension(v, i)))
        candOrder += static_cast<int>(rank);
      if (tilingDim == -1 || order > candOrder) {
        tilingDim = (int64_t)i;
        order = candOrder;
      }
    }
  }
  return tilingDim;
}

int64_t
DimensionAnalyzer::getHigherDimCounts(ArrayRef<Dimension> candidate,
                                      SmallVectorImpl<int64_t> *candidateDims) {
  int64_t higherDimCnt = 0;
  for (auto [store, cDim] : candidate) {
    auto storeRef = getValueDimIndices(store);
    int64_t curDim = tilingDim_[store];
    int64_t curOrigDim = curDim;
    int64_t origDim = cDim;
    auto dim = cDim;
    auto solverIndex = equivalentDsu_->find(storeRef[dim]);
    LDBG("Checking if " << solverIndex << " is transposed dim");
    if (transposedDimMap.contains(solverIndex)) {
      LDBG("Found transposed mapping(" << solverIndex << "): " << dim << " to "
                                       << transposedDimMap.at(solverIndex));
      dim = transposedDimMap.at(solverIndex);
    }
    if (curDim != -1) {
      solverIndex = equivalentDsu_->find(storeRef[curOrigDim]);
      if (transposedDimMap.contains(solverIndex))
        curDim = transposedDimMap.at(solverIndex);
      // Reduced dim should be selected with lowest priority
      if (isReduceDim(Dimension(store, curOrigDim))) {
        curDim += static_cast<int64_t>(storeRef.size());
        LDBG("Reduced dim detected for curDim: lowering priority");
      }
    }
    if (candidateDims)
      candidateDims->push_back(dim);
    if (isReduceDim(Dimension(store, origDim))) {
      dim += static_cast<int64_t>(storeRef.size());
      LDBG("Reduced dim detected for dim: lowering priority");
    }
    if (curDim == -1 || curDim > dim)
      higherDimCnt++;
  }
  return higherDimCnt;
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
///     #hivm.address_space<ub>>) outs(%subDst : memref<?x16xf16, strided<[16,
///     1]>, #hivm.address_space<gm>>)
/// \endcode
template <typename StoreOpTy>
static bool checkTileableMaskedStore(StoreOpTy storeOp, size_t i) {
  auto src = storeOp.getSrc();
  Value dst;
  if constexpr (std::is_same_v<StoreOpTy, hivm::VReduceOp>) {
    dst = storeOp.getDstValue();
  } else {
    dst = storeOp.getDst();
  }

  int64_t srcOrigDim = ShapedType::kDynamic;
  int64_t dstOrigDim = ShapedType::kDynamic;
  if (auto extractSliceOp =
          src.template getDefiningOp<tensor::ExtractSliceOp>()) {
    srcOrigDim = extractSliceOp.getSourceType().getDimSize(i);
  } else if (auto subviewOp = src.template getDefiningOp<memref::SubViewOp>()) {
    srcOrigDim = subviewOp.getSourceType().getDimSize(i);
  }
  if (auto extractSliceOp =
          dst.template getDefiningOp<tensor::ExtractSliceOp>()) {
    dstOrigDim = extractSliceOp.getSourceType().getDimSize(i);
  } else if (auto subviewOp = dst.template getDefiningOp<memref::SubViewOp>()) {
    dstOrigDim = subviewOp.getSourceType().getDimSize(i);
  }
  return srcOrigDim != ShapedType::kDynamic && srcOrigDim != 1 &&
         srcOrigDim == dstOrigDim;
}

template <>
bool checkTileableMaskedStore<hivm::DebugOp>(hivm::DebugOp storeOp, size_t i) {
  return false;
}

/// Walks every \c StoreOpTy under the analyzed op. For each source axis that
/// is parallel, appends that \c Dimension to \p parallelDimMap (keyed by solver
/// group and parent index) unless the axis is dynamic or size 1; for
/// \c hivm::StoreOp, \c checkTileableMaskedStore may still allow the latter.
///
/// A \b value group is a union-find bucket: during analysis, \c valueGroupDSU_
/// in base analyzer groups \c valueToDimIndicesIndex_ ids.
/// (See DimensionAnalyzer::processBFS / processPreOrderWalk)
///
/// The implementation runs \c op_->walk<WalkOrder::PreOrder> on a root op
/// with a callback that only fires for operations of type \c StoreOpTy.
/// Each time a store is hit, \c numStoreOps for that source's solver group
/// (\c srcRef) is bumped, then we scan every rank index \c i of
/// the source: if the dimension is parallel and passes
/// the dynamic/broadcast (and optional masked-store) rules, we record
/// (Value, DimensionIndex) pair under parallelDimMap for that group and
/// collapsed parent index. The same op kind is used for \c hivm::CopyOp, \c
/// hivm::IndirectStoreOp, or \c hivm::FixpipeOp depending on how \c
/// computeTilingDim was invoked.
template <typename StoreOpTy>
void DimensionAnalyzer::computeTilingDimImpl(
    DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>>
        &parallelDimMap,
    DenseMap<int64_t, int> &numStoreOps) {
  op_->walk<WalkOrder::PreOrder>([&](StoreOpTy op) {
    Value src;
    if constexpr (std::is_same_v<StoreOpTy, hivm::DebugOp>) {
      src = op.getArg();
    } else {
      src = op.getSrc();
    }
    auto rank = utils::getShapeRank(src.getType()).value_or(0);
    createDummyRefIfNotExist({src});
    auto args = getValueDimIndices(src);
    // See that each tiling dimension differs for each \c groupIndex.
    // Each operation in a group is independent, horizontal, and totally
    // separated to other operations in a different group. In common kernels,
    // there will only be 1 group.
    if (rank == 0)
      return;
    auto groupIndex = getValueGroupIndex(src);
    numStoreOps[groupIndex]++;
    LDBG("Checking operation: " << op << " in group " << groupIndex);
    auto shape = utils::getShape(src.getType());
    DenseSet<int> usedParentIdx;
    for (size_t i = 0; i < rank; i++) {
      auto parentIndex = structuralDsu_->find(args[i]);
      if (!usedParentIdx.insert(parentIndex).second) {
        op->emitWarning() << "Detected dimensions are in the same group in one "
                             "storeOp. It is recommended to try with "
                             "strict-mode=false if TileAndBindSubBlock fails";
        broadcastAxisCaseCandidate.insert(parentIndex);
      }
    }
    usedParentIdx.clear();
    for (size_t i = 0; i < rank; i++) {
      Dimension dim(src, i);
      if (isParallelDim(dim)) {
        if (ShapedType::isDynamic(shape[i]) || shape[i] == 1) {
          if (!checkTileableMaskedStore(op, i))
            continue;
        }
        if constexpr (std::is_same_v<StoreOpTy, hivm::VReduceOp>) {
          if (shape[i] <= 4)
            continue;
        }
        LDBG("Dim " << i << " is selected in group " << groupIndex);
        auto parentIndex = structuralDsu_->find(args[i]);
        if (usedParentIdx.insert(parentIndex).second) {
          parallelDimMap[groupIndex][parentIndex].push_back(dim);
        } else {
          auto &otherDim = parallelDimMap[groupIndex][parentIndex].back();
          if (isReduceDim(otherDim) && !isReduceDim(dim))
            otherDim = dim;
        }
      }
    }
  });
}

} // namespace detail
} // namespace hivm
} // namespace mlir
