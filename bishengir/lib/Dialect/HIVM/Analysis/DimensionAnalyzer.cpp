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
#include "bishengir/Dialect/Utils/Util.h"
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

bool DimensionAnalyzer::isParallelDim(Dimension dim) {
  createDummyRefIfNotExist({dim.first});
  auto args = getValueDimIndices(dim.first);
  auto solverCollapserIndex = structuralDsu_->find(args[dim.second]);
  auto solverShapeIndex = equivalentDsu_->find(args[dim.second]);
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

bool DimensionAnalyzer::isReduceDim(Dimension dim) {
  createDummyRefIfNotExist({dim.first});
  auto args = getValueDimIndices(dim.first);
  auto solverCollapserIndex = structuralDsu_->find(args[dim.second]);
  auto solverShapeIndex = equivalentDsu_->find(args[dim.second]);
  LDBG("Checking reduceDim of " << solverCollapserIndex << "("
                                << solverShapeIndex << ")");
  auto tilingDimKindVal =
      tilingDimKindMapForCollapser.find(solverCollapserIndex);
  if (tilingDimKindVal != tilingDimKindMapForCollapser.end())
    return tilingDimKindVal->getSecond() == TilingDimensionKind::Reduce;

  return false;
}

template <typename StoreOpTy> Value getStoreLikeSrc(StoreOpTy storeOp) {
  if constexpr (std::is_same_v<StoreOpTy, hivm::LocalStoreOp>)
    return storeOp.getData();
  else if constexpr (std::is_same_v<StoreOpTy, hivm::DebugOp>)
    return storeOp.getArg();
  else
    return storeOp.getSrc();
}

template <typename StoreOpTy> Value getStoreLikeDst(StoreOpTy storeOp) {
  if constexpr (std::is_same_v<StoreOpTy, hivm::VReduceOp>)
    return storeOp.getDstValue();
  else if constexpr (std::is_same_v<StoreOpTy, hivm::LocalStoreOp>)
    return storeOp.getAddr();
  else
    return storeOp.getDst();
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
    computeTilingDimImpl<hivm::LocalStoreOp>(parallelDimMaps, numStoreOps);
    // FIXME: Support reduction dim slicing
    if (isRegbased)
      computeTilingDimImpl<hivm::VReduceOp>(parallelDimMaps, numStoreOps);
    if (isa<scf::ForOp>(op_))
      computeTilingDimImpl<scf::YieldOp>(parallelDimMaps, numStoreOps);
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
  SmallVector<DenseSet<int64_t>> invalidConnection(argumentTotalLength_);
  auto candidateExclusiveDimIdx = exclusiveDimIdx;
  auto hasConnection = [&](auto &connections, int lhs, int rhs) {
    lhs = candGroupDSU.find(lhs);
    rhs = candGroupDSU.find(rhs);
    return connections[lhs].contains(rhs);
  };
  auto addConnection = [&](auto &connections, int lhs, int rhs) {
    lhs = candGroupDSU.find(lhs);
    rhs = candGroupDSU.find(rhs);
    if (lhs == rhs)
      return;
    connections[lhs].insert(rhs);
    connections[rhs].insert(lhs);
  };
  auto mergeConnections = [&](auto &connections, int lhs, int rhs) {
    auto merged = candGroupDSU.find(lhs);
    DenseSet<int64_t> connected;
    for (auto idx : connections[lhs])
      connected.insert(candGroupDSU.find(idx));
    for (auto idx : connections[rhs])
      connected.insert(candGroupDSU.find(idx));

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
  };
  for (auto &[repIdx, indices] : invalidUpdates) {
    if (llvm::any_of(indices,
                     [repIdx = repIdx](auto idx) { return idx == repIdx; })) {
      for (auto idx : indices) {
        if (idx != repIdx)
          addConnection(invalidConnection, repIdx, idx);
      }
    }
  }
  for (auto &[repIdx, indices] : invalidUpdates) {
    auto idxToMerge = indices[0];
    auto groupIdx = valueGroupDSU_->find(dimIdxToArgIdx_[idxToMerge]);
    for (auto [store, dim] : parallelDimMaps[groupIdx][idxToMerge])
      tilingDim_[store] = dim;
    for (auto &idx : llvm::drop_begin(indices)) {
      if (hasConnection(invalidConnection, repIdx, idx) ||
          hasConnection(candidateExclusiveDimIdx, repIdx, idx))
        continue;
      auto curSize = candidateGroupSize[candGroupDSU.find(idxToMerge)];
      auto size = candidateGroupSize[candGroupDSU.find(idx)];
      if (curSize < size ||
          hasConnection(invalidConnection, repIdx, idxToMerge) ||
          hasConnection(candidateExclusiveDimIdx, repIdx, idxToMerge)) {
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
        hasConnection(invalidConnection, repIdx, idxToMerge) ||
        hasConnection(candidateExclusiveDimIdx, repIdx, idxToMerge))
      continue;
    LDBG("repIdx " << repIdx << "("
                   << candidateGroupSize[candGroupDSU.find(repIdx)]
                   << "): " << to_string(indices));
    auto totalSize = candidateGroupSize[candGroupDSU.find(repIdx)] +
                     candidateGroupSize[candGroupDSU.find(idxToMerge)];
    LDBG("Merging " << repIdx << " with " << idxToMerge << " to be "
                    << totalSize);
    auto repRoot = candGroupDSU.find(repIdx);
    auto idxRoot = candGroupDSU.find(idxToMerge);
    candGroupDSU.join(repIdx, idxToMerge);
    mergeConnections(invalidConnection, repRoot, idxRoot);
    mergeConnections(candidateExclusiveDimIdx, repRoot, idxRoot);
    candidateGroupSize[candGroupDSU.find(repIdx)] = totalSize;
    for (auto &idx : indices) {
      if (candGroupDSU.find(idx) != candGroupDSU.find(idxToMerge))
        addConnection(invalidConnection, repIdx, idx);
    }
  }

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
      auto candGroup = candGroupDSU.find(parentIndex);
      llvm::append_range(candidatesByCandGroup[candGroup], candidate);
    }
    for (const auto &[candGroup, candidate] : candidatesByCandGroup) {
      if (candidateGroupSize[candGroup] == numStoreOp) {
        SmallVector<int64_t> candidateDims;
        int64_t higherDimCnt = getHigherDimCounts(candidate, &candidateDims);

        LDBG("Candidate of "
             << candGroup << " in group " << groupIndex << " is "
             << utils::debugger::to_string(candidateDims) << " with "
             << higherDimCnt << " priority dimensions");
        // try to find majority of dimension is higher
        if (2 * higherDimCnt >= numStoreOp) {
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
      if (candGroupDSU.find(parentIndex) == candGroup)
        selectedTilingParIdx.insert(parentIndex);
    }
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
    SmallVector<Value> sources;
    if constexpr (std::is_same_v<StoreOpTy, scf::YieldOp>) {
      sources = llvm::to_vector(op.getResults());
    } else {
      sources.push_back(getStoreLikeSrc(op));
    }
    for (auto src : sources) {
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
      std::optional<size_t> forcedDim = inferForcedTilingDim<StoreOpTy>(op);
      for (size_t i = 0; i < rank; i++) {
        if (forcedDim.has_value() && i != forcedDim.value()) {
          continue;
        }
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

/// Infer whether it is necessary to enforce a specific tiling dimension.
///
/// For Matrix Multiplication (C = A * B):
/// - If A is loaded inside the loop and B is outside, the loop marches along
/// the M-axis.
/// - If B is loaded inside the loop and A is outside, the loop marches along
/// the N-axis.
template <typename StoreOpTy>
std::optional<size_t> DimensionAnalyzer::inferForcedTilingDim(StoreOpTy op) {
  auto hasLoadProducerInsideScope = [this](Value rootVal) -> bool {
    SmallVector<Value, 4> worklist = {rootVal};
    llvm::SmallPtrSet<Operation *, 8> visited;
    while (!worklist.empty()) {
      Value currVal = worklist.pop_back_val();
      Operation *defOp = currVal.getDefiningOp();

      if (!defOp || !this->op_->isProperAncestor(defOp))
        continue;
      if (!visited.insert(defOp).second)
        continue;

      if (isa<hivm::LoadOp>(defOp))
        return true;

      for (Value opnd : defOp->getOperands())
        worklist.push_back(opnd);
    }
    return false;
  };

  if constexpr (std::is_same_v<StoreOpTy, hivm::FixpipeOp>) {
    Operation *curr = op.getSrc().getDefiningOp();

    while (curr && !isa<hivm::MmadL1Op>(curr) && curr->getNumOperands() > 0) {
      curr = curr->getOperand(0).getDefiningOp();
    }

    if (auto mmadOp = dyn_cast_or_null<hivm::MmadL1Op>(curr)) {
      bool isAInside = hasLoadProducerInsideScope(mmadOp.getA());
      bool isBInside = hasLoadProducerInsideScope(mmadOp.getB());

      if (isAInside && !isBInside) {
        LDBG("Forced tiling dim 0 (M-axis) based on internal Load A");
        return 0;
      }

      if (!isAInside && isBInside) {
        LDBG("Forced tiling dim 1 (N-axis) based on internal Load B");
        return 1;
      }
    }
  }

  return std::nullopt;
}

} // namespace detail
} // namespace hivm
} // namespace mlir
