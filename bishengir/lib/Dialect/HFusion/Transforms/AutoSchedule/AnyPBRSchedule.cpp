//===- AnyPBRSchedule.cpp -- Any Pointwise/Broadcast/Reduce Schedule ------===//
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
//
// This file implements auto schedule policy for any axis pointwise,
// broadcast, and reduction kernels.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AnyPBRSchedule.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/ValueHandle.h"
#include "bishengir/Dialect/Utils/ReachabilityAnalyzer.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "AutoScheduleAttrDefs.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include <algorithm>

#define DEBUG_TYPE "hfusion-auto-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [AnyPBR] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::utils;

static std::optional<hfusion::AtomicKind>
tryMapReduceToAtomicKind(linalg::ReduceOp reduceOp) {
  // We only support these kind of atomic kind now
  Block &body = reduceOp.getCombiner().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto *bodyOp = yieldOp.getValues()[0].getDefiningOp();
  if (isa<arith::AddFOp>(bodyOp) || isa<arith::AddIOp>(bodyOp)) {
    return hfusion::AtomicKind::ADD;
  }
  if (isa<arith::MaximumFOp>(bodyOp) || isa<arith::MaxSIOp>(bodyOp) ||
      isa<arith::MaxNumFOp>(bodyOp)) {
    return hfusion::AtomicKind::MAX;
  }
  if (isa<arith::MinimumFOp>(bodyOp) || isa<arith::MinSIOp>(bodyOp) ||
      isa<arith::MinNumFOp>(bodyOp)) {
    return hfusion::AtomicKind::MIN;
  }
  return {};
}

namespace {

using ConsumerInfoPair =
    std::pair<Operation *, hfusion::detail::ConsumerWithReduction>;

BitVector generateAxisMaskForOutput(const hfusion::detail::StoreOpInfo &info) {
  assert(!info.inputsAnchorDimension.empty());
  return info.inputsAnchorDimension.front();
}

BitVector generateAxisMaskForReduce(const hfusion::detail::ReduceInfo &info) {
  assert(!info.inputsAnchorDimension.empty());
  return info.inputsAnchorDimension.front();
}

/// Axis mask represents the relationship between the consumer and the
/// anchor.
/// If the axis mask is 0, it means that the consumer doesn't contain
/// the anchor axis.
/// This mask is unrelated to the tiling key.
BitVector generateAxisMaskForConsumer(const AnyPBRKernelInfo &kernelInfo,
                                      const ConsumerInfoPair &consumerAndInfo) {
  auto [consumer, info] = consumerAndInfo;
  assert(info.getType() != hfusion::detail::ConsumerType::kUnknown);
  if (info.getType() == hfusion::detail::ConsumerType::kReduction) {
    auto reductionInfo = kernelInfo.reduceOp2Info.at(consumer);
    return generateAxisMaskForReduce(reductionInfo);
  }
  auto storeOpInfo = kernelInfo.storeOp2Info.at(consumer);
  return generateAxisMaskForOutput(storeOpInfo);
}

using ReduceInfoPair = std::pair<Operation *, hfusion::detail::ReduceInfo>;

BitVector generateTilingMaskForReduce(int64_t tilingKey,
                                      const ReduceInfoPair &reduceInfo) {
  auto [_, info] = reduceInfo;
  // Size of the tiling mask is same as the anchor rank.
  assert(!info.inputsAnchorDimension.empty());
  BitVector tilingMask(info.inputsAnchorDimension.front().size(), false);
  // We should set all the reduction axis prior to tiling key to be true.
  // This is because the parallel axes should have already been tiled in
  // `tileParallelAxesAndFuseProducers`.
  // Specifically, we need to find the reduce axis in the anchor, and find
  // it's relationship with the tiling key in the anchor.
  // For example:
  //   reduce: [a, b, d, c], reduce_dim = [0, 3]
  //   anchor: [e, b, a, d, c]
  //
  // For Reduce dim a, it's position within the anchor is 2.
  // If the tiling key is greater than or equal to 2, we need to tile a.
  // For Reduce dim c, it's position within the anchor is 4.
  // If the tiling key is greater than or equal to 4, we need to tile c.
  assert(!info.inputsInterchange.empty());
  auto currentInterchange = info.inputsInterchange.front();
  for (int64_t reduceDimInOp : info.reductionDims) {
    // Find the reduce dims in anchor
    int64_t reduceDimInAnchor = currentInterchange[reduceDimInOp];
    if (tilingKey >= reduceDimInAnchor)
      tilingMask[reduceDimInAnchor] = true;
  }
  return tilingMask;
}

BitVector generateTilingMaskForOutput(const hfusion::detail::StoreOpInfo &info,
                                      bool isStrictlyParallel) {
  // Size of the tiling mask is same as the anchor rank.
  assert(!info.inputsAnchorDimension.empty());
  BitVector tilingMask(info.inputsAnchorDimension.front().size(), false);
  assert(!info.inputsInterchange.empty());
  auto currentInterchange = info.inputsInterchange.front();
  if (isStrictlyParallel) {
    // Only tile the dims that are strictly parallel (i.e., not related to
    // reduce axis).
    // This is because we only tile each axis once. For cases like:
    //   Reduce    [A, R] -> [A]
    //   Broadcast [A] -> [A, R]
    //   Return    [A, R]
    // Even though R is indeed a parallel axis in `Return`, we should tile it.
    for (auto parallelDimsInOutput : info.strictlyParallelDims) {
      assert(parallelDimsInOutput <
             static_cast<int64_t>(currentInterchange.size()));
      auto dimInAnchor = currentInterchange[parallelDimsInOutput];
      tilingMask[dimInAnchor] = true;
    }
    return tilingMask;
  }
  // Tile reduction axis.
  for (auto reduceDimsInOutput : info.looselyReductionDims) {
    assert(reduceDimsInOutput <
           static_cast<int64_t>(currentInterchange.size()));
    auto dimInAnchor = currentInterchange[reduceDimsInOutput];
    tilingMask[dimInAnchor] = true;
  }
  return tilingMask;
}

/// Tiling mask represents whether we need to tile the consumer given
/// the current tiling key.
BitVector
generateTilingMaskForConsumer(int64_t tilingKey,
                              const AnyPBRKernelInfo &kernelInfo,
                              const ConsumerInfoPair &consumerAndInfo) {
  auto [consumer, info] = consumerAndInfo;
  assert(info.getType() != hfusion::detail::ConsumerType::kUnknown);
  if (info.getType() == hfusion::detail::ConsumerType::kReduction) {
    auto reductionInfo = kernelInfo.reduceOp2Info.at(consumer);
    return generateTilingMaskForReduce(tilingKey, {consumer, reductionInfo});
  }
  auto storeOpInfo = kernelInfo.storeOp2Info.at(consumer);
  return generateTilingMaskForOutput(storeOpInfo, /*isStrictlyParallel=*/false);
}

bool needToSplitReductionForOp(int64_t tilingKey,
                               const ReduceInfoPair &reduceInfo) {
  return generateTilingMaskForReduce(tilingKey, reduceInfo).any();
}

void analyzeProducersForOutputsWithReductionAxes(AnyPBRKernelInfo *kernelInfo) {
  assert(kernelInfo != nullptr);
  if (kernelInfo->reduceOp2Info.empty())
    return;

  for (auto &[storeOp, storeOpInfo] : kernelInfo->storeOp2Info) {
    auto analysisResult = hfusion::detail::analyzeProducersForStoreOp(
        cast<hfusion::StoreOp>(storeOp), storeOpInfo,
        kernelInfo->reduceDimsInAnchor, kernelInfo->getAnalyzer());
    if (failed(analysisResult))
      continue;

    kernelInfo->recordFusibleProducerAnalysisResult(
        std::move(analysisResult.value()));
  }
}

void analyzeMultiCoreReduceInfo(AnyPBRKernelInfo *kernelInfo) {
  // TODO: Support multiple reducesOp bind to multi cores.
  if (kernelInfo->reduceOp2Info.size() > 1 ||
      // TODO: Support dynamic shape.
      !kernelInfo->isPureStaticKernel()) {
    return;
  }

  bool hasValidReduceToBindMultiCore = false;
  for (auto outputValue : kernelInfo->outputValues) {
    auto storeOp = outputValue.getDefiningOp<hfusion::StoreOp>();
    for (const auto &[reduceOp, _] : kernelInfo->reduceOp2Info) {
      // We can only bind multi-core to reduce if it is
      // reduce + store and there isn't any op in between.
      bool reduceImmFollowedByStore = llvm::all_of(
          reduceOp->getResult(0).getUsers(),
          [storeOp](const Operation *op) -> bool { return op == storeOp; });
      // TODO: Support reduce with multiple results too
      if (!reduceImmFollowedByStore || reduceOp->getResults().size() > 1) {
        continue;
      }

      auto maybeAtomicKind =
          tryMapReduceToAtomicKind(cast<linalg::ReduceOp>(reduceOp));
      if (maybeAtomicKind.has_value()) {
        storeOp.setAtomicKind(maybeAtomicKind.value());
        hasValidReduceToBindMultiCore = true;
      }
    }
  }
  kernelInfo->enableMultiCoreReduce =
      (kernelInfo->enableMultiCoreReduce && hasValidReduceToBindMultiCore);
}

/// Distributes tiled loops into per-dimension collections based on tiling mask.
///
/// \param tiledLoops Tiled loops of all dimensions
/// \param tilingMask Bit vector indicating which dimensions are tiled
/// \param tiledLoopsForEachDim Tiled loops for dimensions with tiling mask
void collectTiledLoopsForEachDim(
    ValueHandles tiledLoops, const BitVector &tilingMask,
    SmallVector<ValueHandles> &tiledLoopsForEachDim) {
  const size_t totalRank = tilingMask.size();
  size_t tiledLoopIndex = 0;
  for (size_t dimension = 0; dimension < totalRank; ++dimension) {
    if (tilingMask.test(dimension)) {
      assert(tiledLoopIndex < tiledLoops.size() &&
             "Insufficient tiled loops for the given tiling mask");
      tiledLoopsForEachDim[dimension].push_back(tiledLoops[tiledLoopIndex++]);
    } else {
      tiledLoopsForEachDim[dimension].push_back(nullptr);
    }
  }
  assert(tiledLoopIndex == tiledLoops.size() &&
         "All tiled loops should be consumed");
}

LogicalResult checkBoolElementType(func::FuncOp func) {
  // TODO: disable op with unsupported element type using a black list
  auto result = func->walk([](linalg::ReduceOp op) {
    for (Type type : op->getOperandTypes()) {
      auto elemType = getElementTypeOrSelf(type);
      if (elemType.isIntOrFloat() && elemType.getIntOrFloatBitWidth() == 1) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

} // namespace

//===----------------------------------------------------------------------===//
// AnyPBRKernelInfoCollector
//===----------------------------------------------------------------------===//

LogicalResult AnyPBRKernelInfoCollector::visitLinalgOpImpl(Operation *op) {
  auto *kernelInfo = dyn_cast_or_null<AnyPBRKernelInfo>(getInfo());
  if (!kernelInfo)
    return failure();

  if (!isa<linalg::ReduceOp>(op))
    return success();

  auto reduceOp = cast<linalg::ReduceOp>(op);
  auto reduceInfo = kernelInfo->reduceOp2Info.at(reduceOp);
  // Collect reduce op's producer info
  auto *dimensionAnalyzer = kernelInfo->getAnalyzer();
  auto analysisResult = detail::analyzeProducersForReductionOp(
      reduceOp, reduceInfo, dimensionAnalyzer);
  kernelInfo->recordFusibleProducerAnalysisResult(std::move(analysisResult));
  return success();
}

LogicalResult AnyPBRKernelInfoCollector::postVisitFuncImpl(func::FuncOp f) {
  if (failed(KernelInfoCollector::postVisitFuncImpl(f)))
    return failure();

  auto *kernelInfo = dyn_cast_or_null<AnyPBRKernelInfo>(getInfo());
  assert(kernelInfo != nullptr);
  if (kernelInfo == nullptr) {
    return failure();
  }
  for (const auto &[_, info] : kernelInfo->reduceOp2Info) {
    kernelInfo->reduceDimsInAnchor.clear();
    assert(!info.inputsInterchange.empty());
    auto interchange = info.inputsInterchange.front();
    for (auto reduceDim : info.reductionDims)
      kernelInfo->reduceDimsInAnchor.insert(interchange[reduceDim]);
  }
  LDBG("Reduce Dims are " << debugger::to_string(
           kernelInfo->reduceDimsInAnchor));

  // Collect producer information for hfusion.store op
  // We collect the information here because it depends on reduction op's
  // information.
  analyzeProducersForOutputsWithReductionAxes(kernelInfo);

  // This part of code is for binding reduce to multi core
  if (kernelInfo->enableMultiCoreReduce)
    analyzeMultiCoreReduceInfo(kernelInfo);

  return success();
}

//===----------------------------------------------------------------------===//
// AnyPBRKernelInfo
//===----------------------------------------------------------------------===//

void AnyPBRKernelInfo::recordFusibleProducerAnalysisResult(
    detail::FusibleProducerAnalysisResult &&result) {
  Operation *consumer = nullptr;
  for (auto [key, value] : result.consumer2ProducerMap) {
    if (!consumer)
      consumer = key.first;

    auto [_, isInserted] = consumer2Producer_.try_emplace(key, value);
    if (!isInserted)
      llvm::report_fatal_error("duplicate consumer + axis pair");
  }
  assert(consumer != nullptr);
  auto [_, isInserted] =
      consumer2Info_.try_emplace(consumer, result.consumerInfo);
  if (!isInserted)
    llvm::report_fatal_error("duplicate consumer");
}

SmallVector<NamedAttribute>
AnyPBRKernelInfo::getReductionProducers(Operation *consumer, int64_t key) {
  // If the pair {consumer, anchorDim} is recorded in the
  // `consumer2ProducerMap`, it means that:
  //    1) the anchorDim is a reduce axis
  //    2) the consumer has some producers
  //
  // So we can construct pairs from {consumer, 0}, ..., {consumer, anchorDim}
  // And for all the dim-idx less than or equal to the key, we can get the
  // producers tags.P
  SmallVector<NamedAttribute> producerTags;
  for (auto dimIdx = 0; dimIdx <= key; ++dimIdx) {
    auto iter = consumer2Producer_.find({consumer, dimIdx});
    if (iter == consumer2Producer_.cend())
      continue;

    producerTags.push_back(iter->second.getIdentifier());
  }
  return producerTags;
}

const hfusion::detail::Consumer2InfoMap &
AnyPBRKernelInfo::getConsumer2Info() const {
  return consumer2Info_;
}

//===----------------------------------------------------------------------===//
// AnyPBRScheduler
//===----------------------------------------------------------------------===//

LogicalResult AnyPBRScheduler::analyzeAndVerifyKernelImpl() {
  // Collect base information first.
  auto *kernelInfo = dyn_cast_or_null<AnyPBRKernelInfo>(getKernelInfo());
  if (failed(AnyPBRKernelInfoCollector(kernelInfo, getAutoScheduleOptions())
                 .run()))
    return getOriginalKernel()->emitError()
           << "Failed to collect AnyPBR kernel info";

  if (failed(checkBoolElementType(getOriginalKernel())))
    return getOriginalKernel()->emitError() << "Unsupported i1 element type";

  return success();
}

TilingComputeFn AnyPBRScheduler::calculateTilingImpl() {
  return [this](KernelInfo *kernelInfo,
                StmtExprBuilder *opBuilder) -> TilingFnResultTy {
    OpBuilder::InsertionGuard g(*opBuilder);
    auto *anyPBRInfo = dyn_cast_or_null<AnyPBRKernelInfo>(kernelInfo);
    assert(anyPBRInfo != nullptr);

    // The number of tiling cases is equal to the number of tileable axes.
    int64_t numTilingCases =
        static_cast<int64_t>(anyPBRInfo->getAnalyzer()->getAnchorRank());
    assert(numTilingCases > 0 &&
           "The number of tileable axes should be greater than 0");
    auto maxBufferCnt = kernelInfo->maxBufferCnt;
    assert(maxBufferCnt > 0 && "buffer count should be greater than zero!");

    // Calculate tiling data.
    MLIRContext *ctx = opBuilder->getContext();

    // Get all dimension size
    SmallVector<Expr> dims;
    LDBG("--- using dimension analyzer to get the max dims");
    auto maxRankDims = kernelInfo->getAnalyzer()->getAnchorShape();
    for (const auto &[maxRankDimsIndex, maxRankDimCandidates] :
         llvm::enumerate(maxRankDims)) {
      auto dim = opBuilder->createConstExpr(maxRankDimCandidates[0].second);

      for (auto &[value, index] : llvm::drop_begin(maxRankDimCandidates)) {
        dim = max(dim,
                  opBuilder->createExpr(value, index, *getKernelTilingMap()));
        LDBG(value << " of index " << index);
      }
      dims.push_back(std::move(dim));

      LDBG("Anchor index " << maxRankDimsIndex);
      LDBG(dim.getMaterializedValue());
    }

    // Align strides
    for (const auto &alignInfo : kernelInfo->getStrideAlignments()) {
      auto [idx, alignment] = alignInfo;
      assert(idx < static_cast<int>(dims.size()));
      LDBG("[Stride Alignment Info] dim: " << idx << " stride is aligned to: "
                                           << alignment);
      dims[idx] = dims[idx].alignTo(alignment);
    }

    Expr smallestTypeBits =
        opBuilder->createConstExpr(kernelInfo->getSmallestElementTypeBits());
    Expr ubMaxSizeInBits = opBuilder->createConstExpr(kUBMaxSizeInBits);
    Expr ubAvailableNumInSmallestTypeBits =
        ubMaxSizeInBits.floorDiv(smallestTypeBits).floorDiv(maxBufferCnt);
    // ub available number align down to block size
    Expr alignedBufferSizeInBits =
        (ubAvailableNumInSmallestTypeBits * smallestTypeBits)
            .alignDown(kUBAlignSizeInBytes * kNumBitsInByte);
    ubAvailableNumInSmallestTypeBits =
        alignedBufferSizeInBits.floorDiv(smallestTypeBits);

    // Reserve extra ub space in case we need to align sizes
    for (const auto &alignInfo : kernelInfo->getSizeAlignments()) {
      auto [idx, alignment] = alignInfo;
      assert(idx < static_cast<int>(dims.size()));
      LDBG("[Size Alignment Info] dim: " << idx << " size is aligned to: "
                                         << alignment);
      // Reserve extra ub space by floorDiv origin ub space by size alignment
      ubAvailableNumInSmallestTypeBits =
          ubAvailableNumInSmallestTypeBits.floorDiv(alignment);
    }

    size_t numTilingData = static_cast<size_t>(numTilingCases) +
                           /*numTilingKey=*/1u + /*numBufferSize=*/1u +
                           /*numMultiCoreForParallelAxis*/ 1u +
                           /*numMultiCoreForReduceAxis*/ 1u;
    TilingStruct s(numTilingData);
    TilingCases c;

    // The constructed array holds the accumulated number of elements up to a
    // certain dimension.
    auto accumulatedDims =
        tiling::getAccumulatedDims(llvm::to_vector(llvm::reverse(dims)));

    /// For the i-th tiling case, assume that we can load all the data
    /// from the i-th to {N-1}-th axes to UB. In other words,
    /// For Tiling Case N-1:
    /// Tiling Sizes = [1, ..., 1, ubAvailableNum]
    ///
    /// For Tiling Case N-2:
    /// Tiling Sizes = [1, ..., ubAvailableNum / dim_{N-2}, dim_{N-1}]
    ///
    /// For Tiling Case 0:
    /// Tiling Sizes = [ubAvailableNum / (dim_{1} *
    ///                                   dim_{2} * ... * dim_{N-1}),
    ///                 dim1, ..., dim_{N-1}]
    ///
    /// Therefore, the tiling key value selection logic is:
    ///   if (ubAvailableNum <= dim_{N-1})
    ///      tilingKey = N - 1
    ///   else if (ubAvailableNum <= dim_{N-1} * dim_{N-2}:
    ///      tilingKey = N - 2
    ///   ...
    ///   else:
    ///      tilingKey = 0
    /// This can also be constructed using a nested selection statement.
    Expr tilingKey = opBuilder->createConstExpr(0);
    for (const auto &[idx, accumulatedValue] :
         llvm::enumerate(llvm::drop_begin(llvm::reverse(accumulatedDims)))) {
      Expr tilingCase = opBuilder->createConstExpr(idx + 1);
      tilingKey = select(ubAvailableNumInSmallestTypeBits <= accumulatedValue,
                         tilingCase, tilingKey);
    }

    auto tilingDataType = IntegerType::get(ctx, 64);
    int64_t dimUpperBound = static_cast<int64_t>(numTilingCases) - 1;
    // use `max` in case ub available num is floor div to 0
    Expr ubRemainingNum = max(ubAvailableNumInSmallestTypeBits, 1);
    for (int64_t dimIdx = dimUpperBound; dimIdx >= 0; --dimIdx) {
      if (failed(c.addKey(dimIdx)))
        return {};

      LDBG("Added tiling case: " << dimIdx);
      /// Consider the following three cases:
      ///
      ///                        (b) dimIdx
      ///                         tilingKey
      ///                            |      (c) dimIdx
      ///        (a) dimIdx          |           |
      ///             |              |           |
      /// [dim_{0}, dim_{1}, ..., dim_{N-2}, dim_{N-1}]
      ///
      /// For case (a), since the selected tiling key is larger than the dim
      /// index, we cannot load more than one line of dim_{1}. Thus the tile
      /// size is 1.
      /// For case (b), since the tiling key is equal to the dim index, we
      /// can partially load dim_{N-2} to the UB, and the tiling size is:
      /// ubAvailableNum / dim_{N-2}.
      /// For case (c), since the tiling key is less than the dim index, we
      /// can fully load dim_{N-1}, and the tiling size is dim_{N-1}.
      Expr tilingKeyGreaterThanDim = tilingKey > dimIdx;
      Expr tilingKeyEqualToDim = tilingKey == dimIdx;
      // FIXME: Don't need to cut full-load dimensions.
      Expr dimSize = dims[dimIdx];
      Expr tileSize = select(
          tilingKeyGreaterThanDim, opBuilder->createConstExpr(1),
          select(tilingKeyEqualToDim, min(ubRemainingNum, dimSize), dimSize));

      for (const auto &[idx, alignment] : kernelInfo->getTileAlignments()) {
        // make sure tiled sizes are aligned
        if (dimIdx != idx) {
          continue;
        }
        LDBG("[Tile Alignment] dim: " << idx << " aligned down: " << alignment);
        tileSize = max(tileSize.alignDown(alignment), 1);
      }

      auto tilingDataForDim = TilingData(std::move(tileSize), tilingDataType);
      // Add heuristic values for case (a).
      for (int64_t tilingKey = dimUpperBound; tilingKey > dimIdx; --tilingKey) {
        int64_t constOne = 1;
        LDBG("Setting tiling data heuristic value: tilingKey="
             << tilingKey << " heuristic=" << constOne);
        tilingDataForDim.setHeuristicValueForKey(tilingKey, constOne);
      }
      s[dimIdx + 1] = std::make_unique<TilingData>(std::move(tilingDataForDim));
      ubRemainingNum = ubRemainingNum.floorDiv(tileSize);
    }

    // Add back the reserved ub space for size alignment
    for (const auto &[idx, alignment] : kernelInfo->getSizeAlignments()) {
      auto &tilingDataPtr = s[idx + 1];
      Expr reservedTileSize = *tilingDataPtr->getExpr();
      // Refine the tileSize by multiple the alignment with reserved tileSize
      // NOTE: tileSize can be greater than dimSize
      Expr refinedTileSize = reservedTileSize * alignment;
      tilingDataPtr->setData(std::move(refinedTileSize));
      // If tiling needs size align, don't apply heuristics.
      // TODO: Better handle this.
      tilingDataPtr->resetHeuristics();
    }

    // Allocate multi cores for Parallel and Reduce axes.
    Expr totalCores = opBuilder->createConstExpr(kernelInfo->blockDim);
    const auto &reduceDims = anyPBRInfo->reduceDimsInAnchor;
    SmallVector<Expr> tileSizes;
    for (size_t i = 0; i < dims.size(); ++i) {
      tileSizes.push_back(*(s[i + 1]->getExpr()));
    }
    auto [coresForParallel, coresForReduce] =
        getMultiCoreNum(totalCores, reduceDims, tileSizes, dims, opBuilder);
    s[anyPBRInfo->getParallelBlockDimTilingDataIdx()] =
        std::make_unique<TilingData>(
            TilingData(std::move(coresForParallel), tilingDataType));
    s[anyPBRInfo->getReduceBlockDimTilingDataIdx()] =
        std::make_unique<TilingData>(
            TilingData(std::move(coresForReduce), tilingDataType));

    // tiling key
    s[0] = std::make_unique<TilingData>(
        TilingData(std::move(tilingKey), tilingDataType));

    // TODO: Move buffer size out of tiling data as it's always a compile-time
    // constant.
    s[numTilingData - 1] = std::make_unique<TilingData>(
        alignedBufferSizeInBits.floorDiv(kNumBitsInByte),
        tilingDataType); // buffer size

    Expr totalSizeInBits = smallestTypeBits;
    for (int64_t dimIdx = numTilingCases - 1; dimIdx >= 0; --dimIdx) {
      totalSizeInBits = totalSizeInBits * (*s[dimIdx + 1]->getExpr());
    }
    opBuilder->createConstraintVerification(
        alignedBufferSizeInBits >= totalSizeInBits,
        "Buffer size is not enough for the given tiling!");
    return TilingFnResultTy(std::make_pair(std::move(c), std::move(s)));
  };
}

bool AnyPBRScheduler::needToSplitReduction(TilingKey key) const {
  if (!hasReduceOp()) {
    LDBG("don't need to split reduction because there is no reduce op");
    return false;
  }
  auto testFn =
      std::bind(needToSplitReductionForOp, key, std::placeholders::_1);
  return llvm::any_of(getKernelInfo()->reduceOp2Info, testFn);
}

bool AnyPBRScheduler::hasReduceOp() const {
  const auto *anyPBRInfo = dyn_cast_or_null<AnyPBRKernelInfo>(getKernelInfo());
  assert(anyPBRInfo != nullptr);
  if (anyPBRInfo == nullptr) {
    return false;
  }
  return !anyPBRInfo->reduceDimsInAnchor.empty();
}

ValueHandleFoldResults AnyPBRScheduler::getTilingFactors(
    TilingKey tilingKey, const SmallVector<TilingData *> &tilingData,
    const BitVector &axisMask, const BitVector &tilingMask,
    ArrayRef<int64_t> tileSizeInterchange) const {
  LDBG("Getting tiling factors");
  assert(!tilingData.empty() && "no tiling data when get tiling factors");
  // Drop the first tiling data, which is tiling key
  auto tileSizeTilingData = llvm::to_vector(llvm::drop_begin(tilingData));
  assert(tileSizeTilingData.size() >= axisMask.size());
  assert(axisMask.size() == tilingMask.size());

  auto getTilingFactor = [&tilingKey, &tilingMask, &tileSizeTilingData,
                          this](size_t idx) {
    if (!tilingMask.test(idx))
      return ValueHandleFoldResult(0, getContext());

    TilingData *td = tileSizeTilingData[idx];
    LDBG("Tile size for " << idx << ": ");
    if (td->isConst()) {
      LDBG("Constant: " << td->getConst());
    } else {
      LDBG("Dynamic: " << td->getHandle());
    }
    // If the tiling factor is a const 1, tile it with constant value.
    // Otherwise the constantize might fail as IR gets complicated.
    if (td->isConst() && td->getConst() == 1)
      return ValueHandleFoldResult(1, getContext());

    auto maybeHeuristicVal = td->getHeuristicValueForKey(tilingKey);
    if (maybeHeuristicVal.has_value()) {
      const int64_t heuristicVal = maybeHeuristicVal.value();
      LDBG("Getting tiling data heuristic value: tilingKey="
           << tilingKey << " heuristic=" << heuristicVal);
      return ValueHandleFoldResult(heuristicVal, getContext());
    }

    return ValueHandleFoldResult{td->getHandle()};
  };

  ValueHandleFoldResults results;
  for (auto interchange : tileSizeInterchange) {
    results.push_back(getTilingFactor(interchange));
  }
  return results;
}

void AnyPBRScheduler::applyCanonicalization(OpBuilder &opBuilder) {
  applyPatterns(
      getFuncHandle(opBuilder),
      /*patterns=*/
      SmallVector<TransformPatternKind>{
          TransformPatternKind::CSE, TransformPatternKind::CANONICALIZATION,
          TransformPatternKind::MERGE_CONSECUTIVE_INSERT_EXTRACT_SLICE,
          TransformPatternKind::RESOLVE_RANKED_SHAPED_TYPE_RESULT_DIMS},
      opBuilder,
      /*disablePatterns=*/
      SmallVector<CanonicalizationPatternKind>{
          CanonicalizationPatternKind::kSimplifyTrivialLoops});
}

SmallVector<int64_t> AnyPBRScheduler::getOpInterchangeAxes(
    SmallVector<int64_t> normalizedInterchange) const {
  // Reduce to order
  SmallVector<int64_t> uniqued = normalizedInterchange;
  std::sort(uniqued.begin(), uniqued.end());
  uniqued.erase(std::unique(uniqued.begin(), uniqued.end()), uniqued.end());
  if (uniqued.size() != normalizedInterchange.size()) {
    LDBG("Maybe result is not a proper order");
    return {};
  }
  SmallVector<int64_t> inversed(normalizedInterchange.size());
  // Assign as a permutation
  for (size_t idx = 0; idx < normalizedInterchange.size(); ++idx) {
    int64_t &compressed = normalizedInterchange[idx];
    compressed = llvm::lower_bound(uniqued, compressed) - uniqued.begin();
    inversed[compressed] = static_cast<int64_t>(idx);
  }

  LDBG("Compressed to " << utils::debugger::to_string(normalizedInterchange));
  LDBG("Inversed to " << utils::debugger::to_string(inversed));
  // inverse as well
  return inversed;
}

ValueHandle *AnyPBRScheduler::tileParallelAxesAndFuseProducers(
    TilingKey tilingKey, TilingInfo &tilingInfo,
    const AnyPBRKernelInfo &kernelInfo, OpBuilder &opBuilder) {
  LDBG("Begin to tile parallel axes for outputs");
  auto totalRank = kernelInfo.getAnalyzer()->getAnchorRank();
  // This is a 2D array. The row corresponds to the number of tilable axes of
  // the anchor, and the column corresponds to the outputs. The stored value is
  // either a loop handle or a nullptr, depending on whether this axis is tiled.
  SmallVector<ValueHandles> outputTiledLoops(totalRank);
  for (size_t outputIdx = 0; outputIdx < kernelInfo.numOutputs; outputIdx++) {
    auto *outputOp = kernelInfo.outputValues[outputIdx].getDefiningOp();
    const auto outputInfo = kernelInfo.storeOp2Info.at(outputOp);
    // Initialize axis mask and tiling mask.
    BitVector axisMask = generateAxisMaskForOutput(outputInfo);
    BitVector tilingMask =
        generateTilingMaskForOutput(outputInfo, /*isStrictlyParallel=*/true);
    // If the output doesn't share any axis with the anchor, bail out.
    // This could happen for cases like:
    //   reduce [A] -> []
    //   return []
    if (axisMask.none()) {
      LDBG("Skipping output because it doesn't share any axis with the "
           "anchor");
      continue;
    }
    assert(!outputInfo.inputsInterchange.empty());
    auto currentInterchange = outputInfo.inputsInterchange.front();
    LDBG("output: " << *outputOp);
    LDBG("axis mask: " << utils::debugger::to_string(axisMask));
    LDBG("tiling mask: " << utils::debugger::to_string(tilingMask));
    LDBG("interchange: " << utils::debugger::to_string(currentInterchange));

    auto tileSizes = getTilingFactors(tilingKey, tilingInfo.getTilingStruct(),
                                      axisMask, tilingMask, currentInterchange);

    // Tile parallel axes using `scf.for` op.
    ValueHandles opsToTile = {
        getOpsWithAttr(hfusion::ReturnOperandNumAttr::name, opBuilder,
                       opBuilder.getI64IntegerAttr(outputIdx))};
    ForTilingResult tileUsingForResult =
        tileUsingFor(opsToTile, tileSizes, opBuilder,
                     getOpInterchangeAxes(currentInterchange));

    // We can take the front because there is only one hfusion.store tiled
    // each time.
    auto tiledLoops = tileUsingForResult.loops.front();
    collectTiledLoopsForEachDim(tiledLoops, tilingMask, outputTiledLoops);
  }

  applyCanonicalization(opBuilder);

  // Fuse independent `scf.for` ops for every dimension.
  ValueHandles fusedLoops = fuseLoopsForEachDim(outputTiledLoops, opBuilder);
  if (fusedLoops.empty())
    return nullptr;

  // Coalesce loops starting from outermost loop and normalize it.
  auto *coalescedLoop = coalesceLoops(fusedLoops.front(), opBuilder);
  normalizeLoop(coalescedLoop, opBuilder);

  // Fuse producers into `scf.for` op.
  ValueHandle *producerOps = getIntermediateProducers(opBuilder);
  ValueHandles targetsToFuseInto = {producerOps};
  ValueHandles fusedLoopList = {coalescedLoop};
  fuseIntoContaining(targetsToFuseInto, fusedLoopList, opBuilder);
  // Handle to outermost loop is invalidated, needs rematching.
  coalescedLoop->setStatus(HandleStatus::kNeedsRematch);
  LDBG("Finished tiling parallel axes for outputs");
  return coalescedLoop;
}

LogicalResult AnyPBRScheduler::createScheduleImpl(TilingKey key,
                                                  OpBuilder &opBuilder) {
  TilingInfo *tilingInfo = getTilingInfo();
  assert(tilingInfo != nullptr);

  auto *anyPBRInfo = dyn_cast_or_null<AnyPBRKernelInfo>(getKernelInfo());
  assert(anyPBRInfo != nullptr);
  if (anyPBRInfo == nullptr) {
    return failure();
  }

  // Get handles to tiling data.
  ValueHandles tilingDataHandles =
      getTilingStructHandles(tilingInfo->getTilingStruct(), opBuilder);

  // Tile cache writes' parallel axes.
  ValueHandle *tiledParallelLoop = tileParallelAxesAndFuseProducers(
      key, *tilingInfo, *anyPBRInfo, opBuilder);

  // Bind parallel axis to multicore.
  if (anyPBRInfo->enableMultiCoreReduce) {
    bindLoopToMulticore(tiledParallelLoop, *anyPBRInfo, opBuilder,
                        tilingInfo->getTilingData(
                            anyPBRInfo->getParallelBlockDimTilingDataIdx()));
  } else {
    bindLoopToMulticore(tiledParallelLoop, *anyPBRInfo, opBuilder);
  }

  // Set buffer size.
  ValueHandle *producerOps = getIntermediateProducers(opBuilder);
  ValueHandles targetsToSetBufferSize = {producerOps};

  if (!needToSplitReduction(key))
    return setBufferSize(tilingInfo, targetsToSetBufferSize, opBuilder);

  LDBG("Need to split condition is true");
  // Apply canonicalization before tiling again.
  applyCanonicalization(opBuilder);

  auto totalRank = anyPBRInfo->getAnalyzer()->getAnchorRank();
  SmallVector<ValueHandles> tiledReductionLoopsForOutputConsumer(totalRank);
  SmallVector<NamedAttribute> jointProducerIdentifier;
  ValueHandle *tiledReduceLoop = nullptr;
  // Tile the reduction axis in the reduce op and the store op.
  for (const auto &consumerAndInfo : anyPBRInfo->getConsumer2Info()) {
    auto [consumer, consumerInfo] = consumerAndInfo;
    LDBG("Consumer: " << *consumer);
    // Initialize axis mask and tiling mask.
    BitVector axisMask =
        generateAxisMaskForConsumer(*anyPBRInfo, consumerAndInfo);
    BitVector tilingMask =
        generateTilingMaskForConsumer(key, *anyPBRInfo, consumerAndInfo);
    LDBG("axis mask: " << utils::debugger::to_string(axisMask));
    LDBG("tiling mask: " << utils::debugger::to_string(tilingMask));
    // Bail out conditions:
    if (axisMask.none()) {
      LDBG("Skipping consumer because it doesn't share any axis with the "
           "consumer");
      continue;
    }
    if (tilingMask.none()) {
      LDBG("Skipping consumer because there is nothing to tile.");
      continue;
    }

    NamedAttribute consumerIdentifier = consumerInfo.getIdentifier();
    SmallVector<NamedAttribute> producerIdentifiers =
        anyPBRInfo->getReductionProducers(consumer, key);
    if (consumerInfo.getType() == detail::ConsumerType::kReduction) {
      auto reductionInfo = anyPBRInfo->reduceOp2Info.at(consumer);
      auto currentInterchange = reductionInfo.inputsInterchange.front();
      LDBG("interchange: " << utils::debugger::to_string(currentInterchange));
      ValueHandleFoldResults tileSizes =
          getTilingFactors(key, tilingInfo->getTilingStruct(), axisMask,
                           tilingMask, currentInterchange);
      auto reduceHandles =
          ValueHandles{getOpsWithAttr(consumerIdentifier.getName(), opBuilder,
                                      consumerIdentifier.getValue())};
      ForReductionTilingResult reductionTileResult =
          tileReductionUsingFor(reduceHandles, tileSizes, opBuilder,
                                /*multiReduceNum=*/reductionInfo.numResults);
      auto *tiledLoopHandle = reductionTileResult.loops.front();
      MatchOptions options = {/*needsReverse=*/true,
                              /*childHandleOrValue=*/tiledLoopHandle};
      ValueHandles producerHandles =
          mergeProducerHandles(producerIdentifiers, options, opBuilder);
      ValueHandles handles = {tiledLoopHandle};
      fuseIntoContaining(producerHandles, handles, opBuilder,
                         /*duplicateProducers=*/true,
                         /*applyCanonicalizeAfterEachFusion=*/true);
      tiledReduceLoop = tiledLoopHandle;

      for (const auto &inits : reductionTileResult.reductionInitOp)
        targetsToSetBufferSize.append(inits);
      targetsToSetBufferSize.append(reductionTileResult.partialReductionOp);
      targetsToSetBufferSize.append(reductionTileResult.finalReductionOp);
    } else {
      auto opInfo = anyPBRInfo->storeOp2Info.at(consumer);
      assert(!opInfo.inputsInterchange.empty());
      auto currentInterchange = opInfo.inputsInterchange.front();
      LDBG("interchange: " << utils::debugger::to_string(currentInterchange));
      ValueHandleFoldResults tileSizes =
          getTilingFactors(key, tilingInfo->getTilingStruct(), axisMask,
                           tilingMask, currentInterchange);
      auto cacheWriteHandles =
          ValueHandles{getOpsWithAttr(consumerIdentifier.getName(), opBuilder,
                                      consumerIdentifier.getValue())};
      ForTilingResult tileResult =
          tileUsingFor(cacheWriteHandles, tileSizes, opBuilder,
                       getOpInterchangeAxes(currentInterchange));
      assert(!tileResult.loops.front().empty());

      // We can take the front because there is only one consumer op tiled
      // each time.
      auto tiledLoops = tileResult.loops.front();
      collectTiledLoopsForEachDim(tiledLoops, tilingMask,
                                  tiledReductionLoopsForOutputConsumer);

      jointProducerIdentifier.append(producerIdentifiers.begin(),
                                     producerIdentifiers.end());
    }
  }

  LDBG("Tiling consumer done");

  // Fuse the tiled reduction axis for output consumer.
  ValueHandles fusedReductionLoop = {};
  if (!tiledReductionLoopsForOutputConsumer.empty()) {
    applyCanonicalization(opBuilder);
    fusedReductionLoop =
        fuseLoopsForEachDim(tiledReductionLoopsForOutputConsumer, opBuilder);
  }

  ValueHandle *loopToFuseInto = nullptr;
  if (!fusedReductionLoop.empty()) {
    // can be empty if has no output consumer with tilable reduction axis
    setStatusTo(fusedReductionLoop, HandleStatus::kNeedsRematch);
    auto *coalescedLoop = coalesceLoops(fusedReductionLoop.front(), opBuilder);
    normalizeLoop(coalescedLoop, opBuilder);
    loopToFuseInto = coalescedLoop;
  }

  if (loopToFuseInto) {
    MatchOptions options;
    options.needsReverse = true;
    options.childHandleOrValue = loopToFuseInto;
    loopToFuseInto->setStatus(HandleStatus::kNeedsRematch);
    ValueHandles targetProducerHandle =
        mergeProducerHandles(jointProducerIdentifier, options, opBuilder);
    ValueHandles targetLoopHandle = {loopToFuseInto};
    fuseIntoContaining(targetProducerHandle, targetLoopHandle, opBuilder,
                       /*duplicateProducers=*/false,
                       /*applyCanonicalizeAfterEachFusion=*/true);
  }

  if (anyPBRInfo->enableMultiCoreReduce && tiledReduceLoop) {
    // Warning:: if this is bigger than 3, there might be precision lost
    TilingData *coreNumForReduceAxis =
        tilingInfo->getTilingData(anyPBRInfo->getReduceBlockDimTilingDataIdx());
    tiledReduceLoop->setStatus(HandleStatus::kNeedsRematch);
    normalizeLoop(tiledReduceLoop, opBuilder);
    bindLoopToMulticore(tiledReduceLoop, *anyPBRInfo, opBuilder,
                        coreNumForReduceAxis);
    LDBG("bind reduction loop to multicore");
  }

  return setBufferSize(tilingInfo, targetsToSetBufferSize, opBuilder);
}

ValueHandles AnyPBRScheduler::mergeProducerHandles(
    const SmallVector<NamedAttribute> &producerIdentifiers,
    const MatchOptions &options, OpBuilder &opBuilder) {
  return {getOpsWithAttrs(/*requiredAttrs=*/{}, opBuilder, producerIdentifiers,
                          options)};
}

void AnyPBRScheduler::bindLoopToMulticore(ValueHandle *loop,
                                          AnyPBRKernelInfo &kernelInfo,
                                          OpBuilder &opBuilder,
                                          const TilingData *numOfCores) {
  if (!loop)
    return;

  LoopTileResult tileResult;
  // if numOfCores is not specified, we tile and bind it to all available cores
  if (!numOfCores) {
    tileResult =
        tileLoop(loop, ValueHandleFoldResult(kernelInfo.blockDim, getContext()),
                 opBuilder,
                 LoopTileOptions{/*mode=*/LoopTileMode::kFactorMode,
                                 /*isReorderMode=*/true});
  } else if (numOfCores->isConst()) {
    tileResult = tileLoop(
        loop, ValueHandleFoldResult(numOfCores->getConst(), getContext()),
        opBuilder,
        LoopTileOptions{/*mode=*/LoopTileMode::kFactorMode,
                        /*isReorderMode=*/true});
  } else {
    tileResult = tileLoop(loop, ValueHandleFoldResult{numOfCores->getHandle()},
                          opBuilder,
                          LoopTileOptions{/*mode=*/LoopTileMode::kFactorMode,
                                          /*isReorderMode=*/true});
  }
  normalizeLoop(tileResult.outerLoop, opBuilder);
  auto mapping = hivm::HIVMBlockMappingAttr::get(getContext());
  mapForToForall(tileResult.outerLoop, opBuilder,
                 MapForToForallOptions{mapping, /*annotate_only=*/true});
  loop->setStatus(HandleStatus::kNeedsRematch);
}

std::pair<Expr, Expr> AnyPBRScheduler::getMultiCoreNum(
    Expr totalCores, const SetVector<int64_t> &reduceDims,
    const SmallVector<Expr> &tileSizes, const SmallVector<Expr> &dimSizes,
    StmtExprBuilder *opBuilder) const {
  // Get available cores from `totalCores` for loop iteration of `dimIdx`
  auto getAvailableCoresForDim = [&tileSizes, &dimSizes,
                                  &totalCores](int64_t dimIdx) {
    // Get loop iteration num for dim by `dimSize / tileSize`
    Expr dimSize = dimSizes[dimIdx];
    const Expr &tileSize = tileSizes[dimIdx];
    // NOTE: `dimSize < tileSize` can happen when reserve and restore ub space,
    // use `max` with `1` to avoid div-by-zero error
    Expr loopIterationNum = max(dimSize.floorDiv(tileSize), 1);
    // If there exist remaining available cores, allocate one core for one
    // loop iteration
    Expr coresForDim =
        select(totalCores > 0, min(totalCores, loopIterationNum), 1);
    totalCores = totalCores.floorDiv(coresForDim);
    return coresForDim;
  };

  // Allocate the remaining cores to the ParallelAxis.
  // From high dimension to low dimension.
  Expr totalCoresForParallelAxis = opBuilder->createConstExpr(1);
  for (size_t dimIdx = 0; dimIdx < dimSizes.size(); ++dimIdx) {
    if (reduceDims.contains(dimIdx))
      continue;
    Expr coresForDim = getAvailableCoresForDim(dimIdx);
    totalCoresForParallelAxis = totalCoresForParallelAxis * coresForDim;
  }

  // Allocate the remaining cores to the ReduceAxis.
  // From low dimension to high dimension.
  Expr totalCoresForReduceAxis = opBuilder->createConstExpr(1);
  for (int64_t dimIdx = static_cast<int64_t>(dimSizes.size()); dimIdx >= 0;
       --dimIdx) {
    if (!reduceDims.contains(dimIdx))
      continue;
    Expr coresForDim = getAvailableCoresForDim(dimIdx);
    totalCoresForReduceAxis = totalCoresForReduceAxis * coresForDim;
  }
  return {totalCoresForParallelAxis, totalCoresForReduceAxis};
}

LogicalResult
AnyPBRScheduler::setBufferSize(const TilingInfo *tilingInfo,
                               ValueHandles &targetsToSetBufferSize,
                               OpBuilder &opBuilder) {
  // The buffer size is the last tiling data
  TilingData *bufferSize = tilingInfo->getTilingData(tilingInfo->size() - 1);
  assert(bufferSize->isConst() && "buffer size should be const");
  uint64_t bufferSizeConst = static_cast<uint64_t>(bufferSize->getConst());
  if (bufferSizeConst == 0u)
    return getToBeScheduledKernel().emitError(
        "Buffer size is less than or equal to zero. Possibly because there is "
        "not enough space on local memory!");

  // Apply canonicalize before setting buffer size to make sure that dead
  // operations are erased.
  applyCanonicalization(opBuilder);
  // Rematch handles to make sure they are valid.
  setStatusTo(targetsToSetBufferSize, HandleStatus::kNeedsRematch);
  SetBufferSizeOptions bufferSizeOptions{transform::SetBufferSizeMode::kPerByte,
                                         getKernelInfo()->smallestElementType};
  SchedulerBase::setBufferSize(targetsToSetBufferSize, bufferSizeConst,
                               opBuilder, bufferSizeOptions);
  return success();
}
