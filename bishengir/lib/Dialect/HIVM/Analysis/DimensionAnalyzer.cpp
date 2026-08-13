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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
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

template <typename StoreOpTy> static Value getTilingTarget(StoreOpTy storeOp) {
  if constexpr (std::is_same_v<StoreOpTy,
                               bufferization::MaterializeInDestinationOp>)
    return storeOp.getSource();
  else if constexpr (std::is_same_v<StoreOpTy, hivm::LocalStoreOp>)
    return storeOp.getData();
  else if constexpr (std::is_same_v<StoreOpTy, hivm::DebugOp>)
    return storeOp.getArg();
  else if constexpr (std::is_same_v<StoreOpTy, hivm::FixpipeOp>)
    return storeOp.getDst();
  else
    return storeOp.getSrc();
}

template <typename StoreOpTy> static Value getStoreLikeSrc(StoreOpTy storeOp) {
  if constexpr (std::is_same_v<StoreOpTy,
                               bufferization::MaterializeInDestinationOp>)
    return storeOp.getSource();
  else if constexpr (std::is_same_v<StoreOpTy, hivm::LocalStoreOp>)
    return storeOp.getData();
  else if constexpr (std::is_same_v<StoreOpTy, hivm::DebugOp>)
    return storeOp.getArg();
  else
    return storeOp.getSrc();
}

template <typename StoreOpTy> static Value getStoreLikeDst(StoreOpTy storeOp) {
  if constexpr (std::is_same_v<StoreOpTy,
                               bufferization::MaterializeInDestinationOp>)
    return storeOp.getDest();
  else if constexpr (std::is_same_v<StoreOpTy, hivm::VReduceOp>)
    return storeOp.getDstValue();
  else if constexpr (std::is_same_v<StoreOpTy, hivm::LocalStoreOp>)
    return storeOp.getAddr();
  else
    return storeOp.getDst();
}

static bool isUbToL1Copy(hivm::CopyOp op) {
  auto dstOrig = utils::tracebackMemRef(op.getDst());
  if (!dstOrig.getDefiningOp<memref::AllocOp>())
    return false;
  auto maybeAddressSpace = GetBufferSpaceAttr(dstOrig);
  return maybeAddressSpace.has_value() &&
         maybeAddressSpace->getAddressSpace() == hivm::AddressSpace::L1;
}

bool DimensionAnalyzer::isParallelDim(Dimension dim) {
  createDummyRefIfNotExist({dim.first});
  auto args = getValueDimIndices(dim.first);
  auto solverCollapserIndex = structuralDsu_->find(args[dim.second]);
  [[maybe_unused]] auto solverShapeIndex =
      equivalentDsu_->find(args[dim.second]);
  LDBG("Checking parallelDim of " << solverCollapserIndex << "("
                                  << solverShapeIndex << ")");
  auto tilingDimKindVal =
      tilingDimKindMapForCollapser.find(solverCollapserIndex);
  if (tilingDimKindVal != tilingDimKindMapForCollapser.end())
    return tilingDimKindVal->getSecond() == TilingDimensionKind::Parallel ||
           tilingDimKindVal->getSecond() == TilingDimensionKind::Reduce;

  // By default, assume it's parallel
  return true;
}

bool DimensionAnalyzer::isReduceDim(int64_t index) {
  auto solverCollapserIndex = structuralDsu_->find(index);
  auto tilingDimKindVal =
      tilingDimKindMapForCollapser.find(solverCollapserIndex);
  if (tilingDimKindVal != tilingDimKindMapForCollapser.end())
    return tilingDimKindVal->getSecond() == TilingDimensionKind::Reduce;

  return false;
}

bool DimensionAnalyzer::isReduceDim(Dimension dim) {
  createDummyRefIfNotExist({dim.first});
  auto args = getValueDimIndices(dim.first);
  auto solverCollapserIndex = structuralDsu_->find(args[dim.second]);
  [[maybe_unused]] auto solverShapeIndex =
      equivalentDsu_->find(args[dim.second]);
  LDBG("Checking reduceDim of " << solverCollapserIndex << "("
                                << solverShapeIndex << ")");
  return isReduceDim(args[dim.second]);
}

bool DimensionAnalyzer::hasCandidateGroupConnection(
    ArrayRef<DenseSet<int64_t>> connections, int64_t lhs, int64_t rhs) const {
  lhs = candGroupDSU_->find(lhs);
  rhs = candGroupDSU_->find(rhs);
  return connections[lhs].contains(rhs);
}

void DimensionAnalyzer::addCandidateGroupConnection(
    MutableArrayRef<DenseSet<int64_t>> connections, int64_t lhs, int64_t rhs) {
  lhs = candGroupDSU_->find(lhs);
  rhs = candGroupDSU_->find(rhs);
  if (lhs == rhs)
    return;
  connections[lhs].insert(rhs);
  connections[rhs].insert(lhs);
}

void DimensionAnalyzer::mergeCandidateGroupConnections(
    MutableArrayRef<DenseSet<int64_t>> connections, int64_t lhs, int64_t rhs) {
  auto merged = candGroupDSU_->find(lhs);
  DenseSet<int64_t> connected;
  for (auto idx : connections[lhs])
    connected.insert(candGroupDSU_->find(idx));
  for (auto idx : connections[rhs])
    connected.insert(candGroupDSU_->find(idx));

  connections[lhs].clear();
  connections[rhs].clear();
  for (auto idx : connected) {
    connections[idx].erase(lhs);
    connections[idx].erase(rhs);
    if (idx == merged)
      continue;
    connections[merged].insert(idx);
    connections[idx].insert(merged);
  }
}

void DimensionAnalyzer::setTilingDimFromMergeIndex(
    int64_t idxToMerge, int64_t groupIdx,
    DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>>
        &parallelDimMaps) {
  for (auto [store, dim] : parallelDimMaps[groupIdx][idxToMerge])
    tilingDim_[store] = dim;
}

int64_t DimensionAnalyzer::selectInvalidUpdateMergeIndex(
    int64_t repIdx, ArrayRef<int64_t> indices,
    ArrayRef<DenseSet<int64_t>> invalidConnection,
    ArrayRef<DenseSet<int64_t>> candidateExclusiveDimIdx,
    DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>>
        &parallelDimMaps,
    ArrayRef<int64_t> candidateGroupSize) {
  int64_t idxToMerge = indices.front();
  int64_t groupIdx = valueGroupDSU_->find(dimIdxToArgIdx_[idxToMerge]);
  setTilingDimFromMergeIndex(idxToMerge, groupIdx, parallelDimMaps);

  for (int64_t idx : llvm::drop_begin(indices)) {
    if (hasCandidateGroupConnection(invalidConnection, repIdx, idx) ||
        hasCandidateGroupConnection(candidateExclusiveDimIdx, repIdx, idx))
      continue;

    int64_t curSize = candidateGroupSize[candGroupDSU_->find(idxToMerge)];
    int64_t size = candidateGroupSize[candGroupDSU_->find(idx)];
    if (curSize < size ||
        hasCandidateGroupConnection(invalidConnection, repIdx, idxToMerge) ||
        hasCandidateGroupConnection(candidateExclusiveDimIdx, repIdx,
                                    idxToMerge)) {
      idxToMerge = idx;
      setTilingDimFromMergeIndex(idxToMerge, groupIdx, parallelDimMaps);
      continue;
    }

    if (curSize != size)
      continue;

    int64_t curHigherDimScore =
        getHigherDimScore(parallelDimMaps[groupIdx][idxToMerge]);
    int64_t higherDimScore = getHigherDimScore(parallelDimMaps[groupIdx][idx]);
    if (2 * higherDimScore >= size && higherDimScore > curHigherDimScore) {
      idxToMerge = idx;
      setTilingDimFromMergeIndex(idxToMerge, groupIdx, parallelDimMaps);
    }
  }

  return idxToMerge;
}

bool DimensionAnalyzer::shouldSkipInvalidUpdateMerge(
    int64_t repIdx, int64_t idxToMerge,
    ArrayRef<DenseSet<int64_t>> invalidConnection,
    ArrayRef<DenseSet<int64_t>> candidateExclusiveDimIdx) const {
  return candGroupDSU_->find(repIdx) == candGroupDSU_->find(idxToMerge) ||
         hasCandidateGroupConnection(invalidConnection, repIdx, idxToMerge) ||
         hasCandidateGroupConnection(candidateExclusiveDimIdx, repIdx,
                                     idxToMerge);
}

void DimensionAnalyzer::applyInvalidUpdateMerge(
    int64_t repIdx, int64_t idxToMerge,
    MutableArrayRef<int64_t> candidateGroupSize,
    MutableArrayRef<DenseSet<int64_t>> invalidConnection,
    MutableArrayRef<DenseSet<int64_t>> candidateExclusiveDimIdx) {
  int64_t totalSize = candidateGroupSize[candGroupDSU_->find(repIdx)] +
                      candidateGroupSize[candGroupDSU_->find(idxToMerge)];
  LDBG("Merging " << repIdx << " with " << idxToMerge << " to be "
                  << totalSize);
  int64_t repRoot = candGroupDSU_->find(repIdx);
  int64_t idxRoot = candGroupDSU_->find(idxToMerge);
  candGroupDSU_->join(repIdx, idxToMerge);
  mergeCandidateGroupConnections(invalidConnection, repRoot, idxRoot);
  mergeCandidateGroupConnections(candidateExclusiveDimIdx, repRoot, idxRoot);
  candidateGroupSize[candGroupDSU_->find(repIdx)] = totalSize;
}

void DimensionAnalyzer::processInvalidUpdates(
    DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>>
        &parallelDimMaps,
    MutableArrayRef<int64_t> candidateGroupSize,
    SmallVector<DenseSet<int64_t>> &candidateExclusiveDimIdx) {
  if (invalidUpdates.empty())
    return;

  LDBG("Connecting invalid updates: ");
  SmallVector<DenseSet<int64_t>> invalidConnection(argumentTotalLength_);

  // Greedy invalid-update ordering: merge groups with fewer reduction dims
  // first so non-reduction (parallel) tiling candidates take priority during
  // later candidate-group selection.
  auto reduceDimSum = [&](int64_t repIdx, ArrayRef<int64_t> indices) {
    int64_t sum = isReduceDim(repIdx);
    for (int64_t idx : indices)
      sum += isReduceDim(idx);
    return sum;
  };
  llvm::stable_sort(invalidUpdates, [&](const auto &lhs, const auto &rhs) {
    return reduceDimSum(lhs.first, lhs.second) <
           reduceDimSum(rhs.first, rhs.second);
  });

  for (auto &[repIdx, indices] : invalidUpdates) {
    int64_t idxToMerge = selectInvalidUpdateMergeIndex(
        repIdx, indices, invalidConnection, candidateExclusiveDimIdx,
        parallelDimMaps, candidateGroupSize);
    if (shouldSkipInvalidUpdateMerge(repIdx, idxToMerge, invalidConnection,
                                     candidateExclusiveDimIdx))
      continue;

    LDBG("repIdx " << repIdx << "("
                   << candidateGroupSize[candGroupDSU_->find(repIdx)]
                   << "): " << to_string(indices));
    applyInvalidUpdateMerge(repIdx, idxToMerge, candidateGroupSize,
                            invalidConnection, candidateExclusiveDimIdx);
  }
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

  loopTilingDims_.clear();

  SmallVector<std::pair<Operation *, Value>> collectedTilingOps;
  if (isVectorOp) {
    collectTilingDim<
        hivm::StoreOp, hivm::CopyOp, bufferization::MaterializeInDestinationOp,
        hivm::StrideStoreOp, hivm::IndirectStoreOp, hivm::LocalStoreOp>(
        parallelDimMaps, numStoreOps, collectedTilingOps);
    // FIXME: Support reduction dim slicing
    if (isRegbased)
      collectTilingDim<hivm::VReduceOp>(parallelDimMaps, numStoreOps,
                                        collectedTilingOps);
    if (isa<scf::ForOp>(op_))
      collectTilingDim<scf::YieldOp>(parallelDimMaps, numStoreOps,
                                     collectedTilingOps);
  } else {
    collectTilingDim<hivm::FixpipeOp>(parallelDimMaps, numStoreOps,
                                      collectedTilingOps);
  }
  collectTilingDim<hivm::DebugOp>(parallelDimMaps, numStoreOps,
                                  collectedTilingOps);

  candGroupDSU_ =
      std::make_unique<mlir::detail::SimpleUnionFind>(argumentTotalLength_);
  SmallVector<int64_t> candidateGroupSize(argumentTotalLength_);
  for (const auto &[groupIndex, parallelDimMap] : parallelDimMaps) {
    for (const auto &[parentIndex, candidate] : parallelDimMap)
      candidateGroupSize[parentIndex] = static_cast<int64_t>(candidate.size());
  }
  auto candidateExclusiveDimIdx = exclusiveDimIdx;
  processInvalidUpdates(parallelDimMaps, candidateGroupSize,
                        candidateExclusiveDimIdx);

  for (auto [value, _] : valueToDimIndicesIndex_)
    tilingDim_[value] = -1;

  DenseMap<int64_t, int> selectedTilingCandGroupMap;
  for (const auto &[groupIndex, parallelDimMap] : parallelDimMaps) {
    auto numStoreOp = 0;
    if (auto it = numStoreOps.find(groupIndex); it != numStoreOps.end())
      numStoreOp = it->second;
    LDBG("Group " << groupIndex << " has " << numStoreOp << " operations");
    DenseMap<int64_t, SmallVector<Dimension>> candidatesByCandGroup;
    for (const auto &[parentIndex, candidate] : parallelDimMap) {
      auto candGroup = candGroupDSU_->find(parentIndex);
      llvm::append_range(candidatesByCandGroup[candGroup], candidate);
    }
    for (const auto &[candGroup, candidate] : candidatesByCandGroup) {
      if (candidateGroupSize[candGroup] == numStoreOp) {
        SmallVector<int64_t> candidateDims;
        int64_t higherDimScore = getHigherDimScore(candidate, &candidateDims);

        LDBG("Candidate of "
             << candGroup << " in group " << groupIndex << " is "
             << utils::debugger::to_string(candidateDims) << " with "
             << higherDimScore << " priority score");
        // try to find majority of dimension is higher
        if (2 * higherDimScore >= numStoreOp) {
          selectedTilingCandGroupMap[groupIndex] = candGroup;
          for (auto [store, dim] : candidate)
            tilingDim_[store] = dim;
        }
      }
    }
  }
  LDBG("Selected independent tiling dims: "
       << selectedTilingCandGroupMap.size());
  for (auto [groupIndex, candGroup] : selectedTilingCandGroupMap) {
    for (const auto &[parentIndex, _] : parallelDimMaps[groupIndex]) {
      if (candGroupDSU_->find(parentIndex) == candGroup)
        selectedTilingParIdx.insert(parentIndex);
    }
  }
  LDBG(utils::debugger::to_string(selectedTilingParIdx));
  for (auto [op, source] : collectedTilingOps)
    recordLoopTilingDims(op, source);
  return isBroadcastAxisCase;
}

void DimensionAnalyzer::recordLoopTilingDims(Operation *op, Value source) {
  int64_t tilingDim = getTilingDimImpl(source);
  if (tilingDim == -1 || !candGroupDSU_)
    return;

  auto args = getValueDimIndices(source);
  int64_t parIdx = structuralDsu_->find(args[tilingDim]);
  int64_t candGroup = candGroupDSU_->find(parIdx);
  for (auto loopOp = op->getParentOfType<LoopLikeOpInterface>();
       loopOp && op_->isAncestor(loopOp.getOperation());
       loopOp = loopOp->getParentOfType<LoopLikeOpInterface>()) {
    loopTilingDims_[loopOp][candGroup].insert(parIdx);
  }
}

const DenseMap<int64_t, DenseSet<int64_t>> &
DimensionAnalyzer::getLoopTilingDims(LoopLikeOpInterface loop) const {
  static const DenseMap<int64_t, DenseSet<int64_t>> empty;
  auto it = loopTilingDims_.find(loop);
  return it == loopTilingDims_.end() ? empty : it->second;
}

int64_t DimensionAnalyzer::getTilingDimImpl(Value v) {
  if (!valueToDimIndicesIndex_.contains(v))
    return -1;
  auto rank = utils::getShapeRank(v.getType()).value_or(0);
  int64_t tilingDim = -1;
  int order = -1;
  auto args = getValueDimIndices(v);
  LDBG("getTilingDim: " << utils::debugger::to_string(llvm::map_to_vector(
           args, [&](auto arg) { return structuralDsu_->find(arg); })));
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
        tilingDim = static_cast<int64_t>(i);
        order = candOrder;
      }
    }
  }
  return tilingDim;
}

int64_t DimensionAnalyzer::getTilingDim(Value v) {
  int64_t tilingDim = getTilingDimImpl(v);
  if (tilingDim == -1)
    return -1;

  auto loopOp = v.getDefiningOp<LoopLikeOpInterface>();
  if (!loopOp || !candGroupDSU_ || !op_->isAncestor(loopOp.getOperation()))
    return tilingDim;

  auto loopTiledDims = getLoopTilingDims(loopOp);
  if (loopTiledDims.empty())
    return tilingDim;

  auto args = getValueDimIndices(v);
  int64_t chosenParIdx = structuralDsu_->find(args[tilingDim]);
  int64_t candGroup = candGroupDSU_->find(chosenParIdx);
  if (auto tiledParIdxsIt = loopTiledDims.find(candGroup);
      tiledParIdxsIt != loopTiledDims.end() &&
      !tiledParIdxsIt->second.contains(chosenParIdx))
    return -1;
  return tilingDim;
}

int64_t
DimensionAnalyzer::getHigherDimScore(ArrayRef<Dimension> candidate,
                                     SmallVectorImpl<int64_t> *candidateDims) {
  int64_t higherDimScore = 0;
  const int64_t candidateSize = static_cast<int64_t>(candidate.size());
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
    const bool candIsReduce = isReduceDim(Dimension(store, origDim));
    if (candIsReduce) {
      dim += static_cast<int64_t>(storeRef.size());
      LDBG("Reduced dim detected for dim: lowering priority");
    }
    const bool curIsReduce =
        curDim != -1 && isReduceDim(Dimension(store, curOrigDim));
    const bool candWins = curDim == -1 || curDim > dim;
    if (candIsReduce != curIsReduce) {
      if (candWins)
        higherDimScore += candidateSize;
      else
        higherDimScore -= candidateSize;
    } else if (candWins) {
      higherDimScore++;
    }
  }
  return higherDimScore;
}

bool DimensionAnalyzer::isValidTilingSize(int64_t dim) const {
  if (isRegbased)
    return !ShapedType::isDynamic(dim) && dim != 1;
  return !ShapedType::isDynamic(dim) && dim % 2 == 0;
}

template <typename StoreOpTy>
bool DimensionAnalyzer::checkTileableMaskedStore(StoreOpTy storeOp,
                                                 size_t i) const {
  auto src = getStoreLikeSrc(storeOp);
  auto dst = getStoreLikeDst(storeOp);

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
  return isValidTilingSize(srcOrigDim) && srcOrigDim == dstOrigDim;
}

template <>
bool DimensionAnalyzer::checkTileableMaskedStore(scf::YieldOp storeOp,
                                                 size_t i) const {
  return false;
}

template <>
bool DimensionAnalyzer::checkTileableMaskedStore(hivm::DebugOp storeOp,
                                                 size_t i) const {
  return false;
}

template <typename... StoreOpTys>
void DimensionAnalyzer::collectTilingDim(
    DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>>
        &parallelDimMaps,
    DenseMap<int64_t, int> &numStoreOps,
    SmallVectorImpl<std::pair<Operation *, Value>> &collectedTilingOps) {
  (collectTilingDimImpl<StoreOpTys>(parallelDimMaps, numStoreOps,
                                    collectedTilingOps),
   ...);
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
void DimensionAnalyzer::collectTilingDimImpl(
    DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>>
        &parallelDimMap,
    DenseMap<int64_t, int> &numStoreOps,
    SmallVectorImpl<std::pair<Operation *, Value>> &collectedTilingOps) {
  op_->walk<WalkOrder::PreOrder>([&](StoreOpTy op) {
    SmallVector<Value> sources;
    if constexpr (std::is_same_v<StoreOpTy, scf::YieldOp>) {
      sources = llvm::to_vector(op.getResults());
    } else if constexpr (std::is_same_v<StoreOpTy, hivm::CopyOp>) {
      if (isUbToL1Copy(op))
        sources.push_back(getTilingTarget(op));
    } else {
      sources.push_back(getTilingTarget(op));
    }
    for (auto src : sources) {
      collectedTilingOps.emplace_back(op.getOperation(), src);
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
        Dimension dim(src, i);
        if (isParallelDim(dim)) {
          if (!isValidTilingSize(shape[i])) {
            if (!checkTileableMaskedStore(op, i)) {
              // temporarily disable tiling for 1xN VReduce
              if constexpr (std::is_same_v<StoreOpTy, hivm::VReduceOp>)
                break;
              continue;
            }
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
            llvm_unreachable("Invalid case.");
          }
        }
      }
    }
  });
}

int64_t DimensionAnalyzer::getGlobalTilingAxisId(Value v) {
  int64_t localDimIdx = getTilingDim(v);
  if (localDimIdx == -1)
    return -1;
  auto args = DimensionAnalyzerBase::getValueDimIndices(v);
  return structuralDsu_->find(args[localDimIdx]);
}

} // namespace detail
} // namespace hivm
} // namespace mlir
