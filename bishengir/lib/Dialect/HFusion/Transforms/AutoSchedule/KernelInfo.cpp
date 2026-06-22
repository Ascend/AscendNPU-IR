//===- KernelInfo.cpp -- Definition for Kernel Info -------------*- C++ -*-===//
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
// This file implements kernel info definition.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/KernelInfo.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <utility>

#include "AutoSchedule/AutoScheduleAttrDefs.h"

#define DEBUG_TYPE "hfusion-auto-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Kernel Info] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

hfusion::detail::StoreOpInfo::StoreOpInfo(size_t numLoops) {
  auto arange = llvm::seq<int64_t>(0, numLoops);
  strictlyParallelDims = {arange.begin(), arange.end()};
}

//===----------------------------------------------------------------------===//
// KernelInfo
//===----------------------------------------------------------------------===//

LogicalResult KernelInfo::initializeDimensionAnalyzer() {
  analyzer_ =
      std::make_unique<hfusion::detail::DimensionAnalyzer>(originalKernel);
  if (failed(analyzer_->initialize()))
    return originalKernel->emitError() << "Failed to analyze dimension";

  analyzer_->computeAnchor();
  return success();
}

int64_t KernelInfo::getSmallestElementTypeBits() {
  assert(smallestElementType != Type());
  LDBG("Smallest tensor element type is " << smallestElementType);
  return smallestElementType.getIntOrFloatBitWidth();
}

bool KernelInfo::isPureStaticKernel() {
  return !getAnalyzer()->getMaxRankDimShape().empty() &&
         !ShapedType::isDynamicShape(getAnalyzer()->getMaxRankDimShape());
}

using AlignmentMaybe = std::optional<KernelInfo::DimAndAlignment>;

SmallVector<KernelInfo::DimAndAlignment> KernelInfo::getStrideAlignments() {
  SmallVector<AlignmentMaybe> alignmentsMaybe{
      getStrideAlignmentsForBroadcastOp(),    getStrideAlignmentsForReduceOp(),
      getStrideAlignmentsForExtractSliceOp(), getStrideAlignmentsForConcatOp(),
      getStrideAlignmentsForTransposeOp(),
  };
  SmallVector<AlignmentMaybe> alignments(llvm::make_filter_range(
      alignmentsMaybe, [](const AlignmentMaybe &alignMaybe) {
        // alignment value should be larger than 1 to take effect
        return alignMaybe.has_value() && alignMaybe->second > 1;
      }));

  // only need to align the lowest dimension
  auto *lowestDimAlignment = std::max_element(
      alignments.begin(), alignments.end(),
      [](const AlignmentMaybe &lhs, const AlignmentMaybe &rhs) {
        return lhs->first < rhs->first;
      });

  SmallVector<KernelInfo::DimAndAlignment> results;
  if (lowestDimAlignment != alignments.end())
    results.push_back(lowestDimAlignment->value());
  return results;
}

int64_t findNthMaskedAxis(const BitVector &axisMask, int64_t n) {
  assert(n >= 0 && "not allow negative n");
  for (int64_t idx = 0; idx < axisMask.size(); ++idx) {
    if (!axisMask.test(idx)) {
      continue;
    }
    if (n == 0) {
      return idx;
    }
    n--;
  }
  llvm::report_fatal_error("cannot find n-th masked axis");
}

static void dumpSizeAlignInfo(int64_t alignDim, int64_t alignUnit) {
  LDBG("[Size Align]: "
       << "dim: " << alignDim << " size is aligned to: " << alignUnit);
}

/// Although pre-schedule passes will ensure last-dim concat will be 32-bytes
/// aligned, but the alignment cannot be ensured after tiling, so we will need
/// extra size alignment for last-dim concat.
///
/// for example:
///   %concat = tensor.concat dim(1): tensor<2x1024xi8>, tensor<2x2048xi8>
///   %broadcasted = linalg.broadcast
///                  ins(%concat : tensor<2x3072xi8>)
///                  outs(%0 : tensor<2x3072x5x1024xi8>) dimensions = [2, 3]
/// with tile size: [1, 1, 5, 1024], where the tile size for concat axis
/// unaligned. Therefore, we should align dim(1) size `1` to `16`, so that the
/// ub2ub copy after decomposing concat will not trigger runtime error.
///
SmallVector<KernelInfo::DimAndAlignment>
KernelInfo::getSizeAlignmentsForConcatOp() {
  SmallVector<KernelInfo::DimAndAlignment> alignments;
  for (const auto &[op, info] : concatOp2Info) {
    if (info.concatDim != info.rank - 1) {
      continue;
    }
    LDBG("[Size Align] op: " << *op);
    assert(!info.resultsInterchange.empty());
    auto interchange = info.resultsInterchange.front();
    int64_t alignDim = interchange[info.concatDim];
    int64_t alignUnit =
        kUBAlignSizeInBytes * kNumBitsInByte / info.elemBitwidth;
    alignments.push_back({alignDim, alignUnit});
    dumpSizeAlignInfo(alignDim, alignUnit);
  }
  return alignments;
}

SmallVector<KernelInfo::DimAndAlignment>
KernelInfo::getSizeAlignmentsForTransposeOp() {
  SmallVector<KernelInfo::DimAndAlignment> alignments;
  auto update = [&alignments](int alignDim, int alignBytes) {
    dumpSizeAlignInfo(alignDim, alignBytes);
    alignments.push_back({alignDim, alignBytes});
  };
  for (const auto &[op, info] : transposeOp2Info) {
    if (!info.transposeLastDim) {
      // size alignment is only necessary for last dim transpose.
      // e.g. [a, b, c] -> [b, a, c], c should be 32 byte strided aligned, no
      // need to do size align on a and b
      continue;
    }

    LDBG("[Size Align] op: " << *op);
    assert(!info.inputsInterchange.empty());
    auto interchange = info.inputsInterchange.front();
    int64_t dim0 = interchange[info.permuteDims.first];
    int64_t dim1 = interchange[info.permuteDims.second];
    switch (smallestElementType.getIntOrFloatBitWidth()) {
    case 8: {
      // b8 : align to 32
      update(dim0, 32);
      update(dim1, 32);
      break;
    }
    case 16:
    case 32: {
      // everything aligns to 16 in b16 and b32
      // - b16: align to 16
      // - b32: [a, b] -> [b, a], a align to 16(8), b align to 8(16)
      update(dim0, 16);
      update(dim1, 16);
      break;
    }
    default:
      llvm::report_fatal_error("Unexpected type for transpose op");
    }
  }
  return alignments;
}

// Hardware does not provide the instruction to cast i32/i16 to i8. To achieve
// the same functionality, we use transpose and ub_to_ub copy instruction, which
// will require extra size alignments.
//
// TODO: extract common utils to get size alignments for both HIVM
// AlignAllocSize pass and KernelInfo
//
// rank >= 2 && isI32ToI8:
// - src[dim0] align to: 32 (i32)
// - src[dim1] align to: 8  (i8)
// - dst[dim0] align to: 32 (i32)
// - dst[dim1] align to: 32 (i8)
//
// rank >= 2 && isI16ToI8:
// - src[dim0] align to: 32 (i16)
// - src[dim1] align to: 16 (i8)
// - dst[dim0] align to: 32 (i16)
// - dst[dim1] align to: 32 (i8)
//
// rank == 1 && isI32ToI8:
// - size < 32:
// -- src[dim] align to: 256  (i32)
// -- dst[dim] align to: 1024 (i8)
// - size >= 32:
// -- src[dim] align to: 1024 (i32)
// -- dst[dim] align to: 1024 (i8)
//
// rank == 1 && isI16ToI8:
// - size < 32:
// -- src[dim] align to: 512  (i16)
// -- dst[dim] align to: 1024 (i8)
// - size >= 32:
// -- src[dim] align to: 1024 (i16)
// -- dst[dim] align to: 1024 (i8)
SmallVector<KernelInfo::DimAndAlignment>
KernelInfo::getSizeAlignmentsForCastOp() {
  SmallVector<KernelInfo::DimAndAlignment> alignments;
  auto update = [&alignments](int alignDim, int alignBytes) {
    dumpSizeAlignInfo(alignDim, alignBytes);
    alignments.push_back({alignDim, alignBytes});
  };
  for (const auto &[op, info] : castOp2Info) {
    Type srcType = info.srcElemType;
    Type dstType = info.dstElemType;
    const bool isI32ToI8 = srcType.isInteger(32) && dstType.isInteger(8);
    const bool isI16ToI8 = srcType.isInteger(16) && dstType.isInteger(8);
    if (!isI32ToI8 && !isI16ToI8) {
      continue;
    }

    LDBG("[Size Align] op: " << *op);
    assert(!info.inputsInterchange.empty());
    assert(!info.resultsAnchorDimension.empty());
    auto srcInterchange = info.inputsInterchange.front();
    auto dstInterchange = info.resultsInterchange.front();

    if (info.rank >= 2) {
      int64_t srcDim0 = srcInterchange[info.rank - 2];
      int64_t srcDim1 = srcInterchange[info.rank - 1];
      int64_t dstDim0 = dstInterchange[info.rank - 2];
      int64_t dstDim1 = dstInterchange[info.rank - 1];
      if (isI32ToI8) {
        update(srcDim0, 32);
        update(srcDim1, 8);
        update(dstDim0, 32);
        update(dstDim1, 32);
      }
      if (isI16ToI8) {
        update(srcDim0, 32);
        update(srcDim1, 16);
        update(dstDim0, 32);
        update(dstDim1, 32);
      }
      continue;
    }
    if (info.rank != 1) {
      // align alloc size not handle zero-rank case
      continue;
    }
    int64_t srcDim = srcInterchange[info.rank - 1];
    int64_t dstDim = dstInterchange[info.rank - 1];
    int64_t dimSize = info.shape.back();
    if (isI32ToI8) {
      if (dimSize < 32) {
        update(srcDim, 256);
        update(dstDim, 1024);
      } else {
        update(srcDim, 1024);
        update(dstDim, 1024);
      }
    }
    if (isI16ToI8) {
      if (dimSize < 32) {
        update(srcDim, 512);
        update(dstDim, 1024);
      } else {
        update(srcDim, 1024);
        update(dstDim, 1024);
      }
    }
  }
  return alignments;
}

SmallVector<KernelInfo::DimAndAlignment> KernelInfo::getSizeAlignments() {
  DenseMap<int, int> alignMap;
  SmallVector<KernelInfo::DimAndAlignment> alignments;
  alignments.append(getSizeAlignmentsForCastOp());
  alignments.append(getSizeAlignmentsForTransposeOp());
  alignments.append(getSizeAlignmentsForConcatOp());

  for (auto [alignDim, alignBytes] : alignments) {
    if (!alignMap.contains(alignDim)) {
      alignMap[alignDim] = alignBytes;
      continue;
    }
    // use least common multiple if have diff align bytes for same dim
    alignMap[alignDim] = std::lcm(alignMap[alignDim], alignBytes);
  }

  return SmallVector<KernelInfo::DimAndAlignment>{alignMap.begin(),
                                                  alignMap.end()};
}

SmallVector<KernelInfo::DimAndAlignment> KernelInfo::getTileAlignments() {
  SmallVector<KernelInfo::DimAndAlignment> alignments;
  alignments.append(getSizeAlignments());
  alignments.append(getStrideAlignments());
  return alignments;
}

std::optional<KernelInfo::DimAndAlignment>
KernelInfo::getStrideAlignmentsForReduceOp() {
  if (reduceOp2Info.empty())
    return std::nullopt;

  // Assume that all reduction ops have the same rank and reduction dims.
  const auto pair = *(reduceOp2Info.cbegin());
  auto reductionDims = llvm::to_vector(pair.second.reductionDims);
  llvm::sort(reductionDims);
  if (reductionDims.empty())
    return std::nullopt;

  auto lastReductionDim = reductionDims.back();
  auto totalDims = pair.second.numLoops;
  if (totalDims == 1) {
    // Special Case:
    // If there is only one loop, and it's a reduction loop, no need to align.
    // For example:
    // tensor<15xf32> reduced to tensor<1xf32>, reduction dims is 1.
    return std::nullopt;
  }

  int32_t alignDim;
  if (lastReductionDim == totalDims - 1) {
    // For last axis reduce, need to align the penultimate axis.
    // For example:
    // tensor<?x15xf32> reduced to tensor<?x1xf32>, reduction dims is 1.
    alignDim = lastReductionDim - 1;
  } else {
    // For n-last axis reduce, need to align the last reduce axis.
    // For example:
    // tensor<15x15xf32> reduced to tensor<1x15xf32>, reduction dims is 0.
    alignDim = lastReductionDim;
  }

  assert(alignDim < static_cast<int32_t>(totalDims));
  LDBG("[Alignment Info] dimension to align: " << alignDim);
  TensorType typeToAlign =
      cast<TensorType>(pair.first->getOperand(0).getType());
  typeToAlign = typeToAlign.clone(typeToAlign.getShape(), smallestElementType);
  LDBG("[Alignment Info] type before alignment: " << typeToAlign);

  // Note: the input to `collectAlignUnits` is the axis to which the stride
  // should be aligned. The stride is aligned by aligning the **next**
  // dimension. So in terms of the shape, we need to align the next dimension.
  SmallVector<int32_t> alignDims{static_cast<int32_t>(alignDim)};
  SmallVector<int32_t> alignBytes{static_cast<int32_t>(kUBAlignSizeInBytes)};
  auto alignment = utils::collectAlignUnits(alignDims, alignBytes, typeToAlign);
  LDBG("[Alignment Info] alignment unit: " << alignment[alignDim + 1]);
  return std::make_pair(static_cast<int>(alignDim + 1),
                        alignment[alignDim + 1]);
}

std::optional<KernelInfo::DimAndAlignment>
KernelInfo::getStrideAlignmentsForBroadcastOp() {
  if (broadcastOp2Info.empty())
    return std::nullopt;

  TensorType typeWithMaxRankAfterBroadcast;
  SetVector<int64_t> broadcastDims;
  for (auto [op, brcInfo] : broadcastOp2Info) {
    auto currentType = cast<TensorType>(op->getResult(0).getType());
    if (typeWithMaxRankAfterBroadcast == TensorType() ||
        typeWithMaxRankAfterBroadcast.getRank() < currentType.getRank()) {
      typeWithMaxRankAfterBroadcast = currentType;
    }
    broadcastDims.insert(brcInfo.broadcastDims.begin(),
                         brcInfo.broadcastDims.end());
  }
  auto broadcastDimsVec = llvm::to_vector(broadcastDims);
  llvm::sort(broadcastDimsVec);
  if (broadcastDimsVec.empty())
    return std::nullopt;

  int32_t alignDim;
  // The last broadcast dimension needs to be aligned.
  auto lastBroadcastDim = broadcastDimsVec.back();
  auto totalDims = typeWithMaxRankAfterBroadcast.getRank();
  if (lastBroadcastDim == totalDims - 1) {
    // Special Case:
    // If there is only a single broadcast dimension, and it's also the final
    // dimension of the tensor, there is no need to do alignment.
    // For example:
    // tensor<1xf32> to tensor<15xf32>, broadcast dim is 0.
    if (llvm::hasSingleElement(broadcastDimsVec)) {
      return std::nullopt;
    }
    // Otherwise, sill need to align the penultimate broadcast axis.
    alignDim = *(broadcastDimsVec.rbegin() + 1);
  } else {
    alignDim = lastBroadcastDim;
  }

  assert(lastBroadcastDim < typeWithMaxRankAfterBroadcast.getRank());
  LDBG("[Alignment Info] dimension to align: " << alignDim);
  TensorType typeToAlign = typeWithMaxRankAfterBroadcast.clone(
      typeWithMaxRankAfterBroadcast.getShape(), smallestElementType);
  LDBG("[Alignment Info] type before alignment: " << typeToAlign);

  // Note: the input to `collectAlignUnits` is the axis to which the stride
  // should be aligned. The stride is aligned by aligning the **next**
  // dimension. So in terms of the shape, we need to align the next dimension.
  SmallVector<int32_t> alignDims{static_cast<int32_t>(alignDim)};
  SmallVector<int32_t> alignBytes{static_cast<int32_t>(kUBAlignSizeInBytes)};
  auto alignment = utils::collectAlignUnits(alignDims, alignBytes, typeToAlign);
  LDBG("[Alignment Info] alignment unit: " << alignment[alignDim + 1]);
  return std::make_pair(static_cast<int>(alignDim + 1),
                        alignment[alignDim + 1]);
}

std::optional<KernelInfo::DimAndAlignment>
KernelInfo::getStrideAlignmentsForExtractSliceOp() {
  if (extractSliceOp2Info.empty()) {
    return std::nullopt;
  }

  // find partial slice requiring alignment
  int64_t alignDim = -1;
  LDBG("Get stride alignments for extract slice op");
  for (auto [op, sliceInfo] : extractSliceOp2Info) {
    if (sliceInfo.partialSlicedDims.empty()) {
      continue;
    }
    auto sliceOp = cast<tensor::ExtractSliceOp>(op);
    RankedTensorType resultType = sliceOp.getResultType();
    assert(!sliceInfo.resultsAnchorDimension.empty());
    BitVector axisMask = sliceInfo.resultsAnchorDimension.front();
    LDBG("Result type is " << resultType);
    if (resultType.getRank() > axisMask.size()) {
      // slice rank may be bigger than anchor rank, we should ignore such
      // slice op since they will not affect alignment. for example:
      //   %slice = extract_slice %arg    <- slice
      //   %collapse = collapse_shape %slice
      //   elemwise_unary %collapse       <- anchor
      // TODO: make scheduler fail for this case
      continue;
    }

    int64_t lowestPartialSliceDim = sliceInfo.partialSlicedDims.back();
    int64_t lowestMaskedDim =
        findNthMaskedAxis(axisMask, lowestPartialSliceDim);
    alignDim = std::max(alignDim, lowestMaskedDim);
  }

  if (alignDim == -1) {
    return std::nullopt;
  }

  int64_t totalDims = static_cast<int64_t>(analyzer_->getAnchorRank());
  if (alignDim == totalDims - 1) {
    // Special Case:
    // If there is only a single extract slice dimension, and it's also the
    // final dimension of the tensor, there is no need to do alignment.
    // For example: tensor<16xf32> to tensor<15xf32>, slice dim is 0.
    if (totalDims == 1) {
      return std::nullopt;
    }
    // Otherwise, sill need to align the penultimate slice axis.
    alignDim -= 1;
  }

  assert(alignDim < totalDims);
  LDBG("[Alignment Info] dimension to align: " << alignDim);
  TensorType typeToAlign = RankedTensorType::get(
      analyzer_->getMaxRankDimShape(), smallestElementType);
  LDBG("[Alignment Info] type before alignment: " << typeToAlign);

  SmallVector<int32_t> alignDims{static_cast<int32_t>(alignDim)};
  SmallVector<int32_t> alignBytes{static_cast<int32_t>(kUBAlignSizeInBytes)};
  auto alignment = utils::collectAlignUnits(alignDims, alignBytes, typeToAlign);
  LDBG("[Alignment Info] alignment unit: " << alignment[alignDim + 1]);
  return std::make_pair(static_cast<int>(alignDim + 1),
                        alignment[alignDim + 1]);
}

std::optional<KernelInfo::DimAndAlignment>
KernelInfo::getStrideAlignmentsForTransposeOp() {
  if (transposeOp2Info.empty())
    return std::nullopt;

  // set the last dim of transpose op on anchor as stride align dim
  int64_t alignDim = -1;
  for (const auto &[op, info] : transposeOp2Info) {
    int64_t lastDimToAlign = info.numLoops - 1;
    auto [dim0, dim1] = info.permuteDims;
    if (lastDimToAlign == dim0 || lastDimToAlign == dim1) {
      // perm dims including last dim will be size aligned,
      // no need to extra stride alignment
      continue;
    }
    assert(!info.inputsAnchorDimension.empty());
    assert(!info.resultsAnchorDimension.empty());
    int64_t lowestMaskedDimOnAnchor = std::max(
        findNthMaskedAxis(info.inputsAnchorDimension.front(), lastDimToAlign),
        findNthMaskedAxis(info.resultsAnchorDimension.front(), lastDimToAlign));
    alignDim = std::max(alignDim, lowestMaskedDimOnAnchor);
  }

  if (alignDim == -1) {
    return std::nullopt;
  }

  int64_t totalDims = static_cast<int64_t>(analyzer_->getAnchorRank());
  if (alignDim == totalDims - 1) {
    // if the transpose dim to align is last dim, align the penultimate axis
    alignDim -= 1;
  }

  assert(alignDim < totalDims);
  TensorType typeToAlign = RankedTensorType::get(
      analyzer_->getMaxRankDimShape(), smallestElementType);
  LDBG("[Transpose Stride Align] type before alignment: " << typeToAlign);

  // `alignDim` specify the dim in `typeToAlign` that requires stride alignment
  SmallVector<int32_t> alignDims{static_cast<int32_t>(alignDim)};
  SmallVector<int32_t> alignBytes{static_cast<int32_t>(kUBAlignSizeInBytes)};
  auto alignment = utils::collectAlignUnits(alignDims, alignBytes, typeToAlign);
  // `alignDim` is aligned by increasing `alignDim + 1` size to `alignment` unit
  LDBG("[Transpose Stride Align] alignment dim: " << (alignDim + 1));
  LDBG("[Transpose Stride Align] alignment unit: " << alignment[alignDim + 1]);
  return std::make_pair(static_cast<int>(alignDim + 1),
                        alignment[alignDim + 1]);
}

std::optional<KernelInfo::DimAndAlignment>
KernelInfo::getStrideAlignmentsForConcatOp() {
  if (concatOp2Info.empty())
    return std::nullopt;

  int64_t alignDim = -1;
  for (const auto &[op, info] : concatOp2Info) {
    if (info.rank <= 1) {
      continue;
    }
    int64_t lastDimToAlign = info.rank - 1;
    int64_t maskedDimOnAnchor =
        findNthMaskedAxis(info.resultsAnchorDimension.front(), lastDimToAlign);
    alignDim = std::max(alignDim, maskedDimOnAnchor);
  }
  if (alignDim == -1) {
    return std::nullopt;
  }

  int64_t totalDims = static_cast<int64_t>(analyzer_->getAnchorRank());
  if (alignDim == totalDims - 1) {
    // if the dim to align is last dim on anchor, align the penultimate axis
    alignDim -= 1;
  }

  TensorType typeToAlign = RankedTensorType::get(
      analyzer_->getMaxRankDimShape(), smallestElementType);
  LDBG("[Concat Stride Align] type before alignment: " << typeToAlign);

  // `alignDim` specify the dim in `typeToAlign` that requires stride alignment
  SmallVector<int32_t> alignDims{static_cast<int32_t>(alignDim)};
  SmallVector<int32_t> alignBytes{static_cast<int32_t>(kUBAlignSizeInBytes)};
  auto alignment = utils::collectAlignUnits(alignDims, alignBytes, typeToAlign);
  // `alignDim` is aligned by increasing `alignDim + 1` size to `alignment` unit
  LDBG("[Concat Stride Align] alignment dim: " << (alignDim));
  LDBG("[Concat Stride Align] alignment unit: " << alignment[alignDim + 1]);
  return std::make_pair(static_cast<int>(alignDim + 1),
                        alignment[alignDim + 1]);
}

BlockArgument KernelInfo::getKernelFuncArg(size_t idx) {
  return originalKernel.getArgument(idx);
}

uint32_t KernelInfo::getParallelBlockDimTilingDataIdx() {
  /* Tiling Key + Tiling Cases*/
  return 1 + getAnalyzer()->getAnchorRank();
}

uint32_t KernelInfo::getReduceBlockDimTilingDataIdx() {
  /* Tiling Key + Tiling Cases + Parallel Block Dim*/
  return 1 + getAnalyzer()->getAnchorRank() + 1;
}
