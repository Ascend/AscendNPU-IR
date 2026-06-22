//===- Utils.cpp ----------------------------------------------------------===//
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

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "hivm-extra-buffer"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define DBGSNL() (llvm::dbgs() << "\n")
namespace mlir {
namespace hivm {
namespace util {
const static int brcFirstFactorUnalign = 1;
const static int brcLastFactorAlign = 2;
const static int brcLastFactorUnalign = 8;
const static int halfBits = 16;

namespace {
int64_t getNumPerBlock(Type t) {
  unsigned bitWidth = getElementTypeOrSelf(t).getIntOrFloatBitWidth();

  unsigned bytesPerElem = CEIL_DIV(bitWidth, utils::INTR_BITS_PER_BYTE);

  return CEIL_DIV(INTR_BYTES_PER_BLOCK, bytesPerElem);
}

int64_t getNumPerRepeat(Type t) {
  unsigned bitWidth = getElementTypeOrSelf(t).getIntOrFloatBitWidth();

  unsigned bytesPerElem = CEIL_DIV(bitWidth, utils::INTR_BITS_PER_BYTE);

  return CEIL_DIV(INTR_BYTES_PER_REPEAT, bytesPerElem);
}

int64_t largestPow2Le(int64_t n) {
  if (n <= 0)
    return 0;
  int64_t r = 1;
  while (r * 2 <= n)
    r *= 2;
  return r;
}

/// Stride descriptors extracted from a MemRef-shaped type. Returns std::nullopt
/// when either the type is not a MemRefType or the layout is not a static
/// StridedLayoutAttr.
struct StrideInfo {
  SmallVector<int64_t> srcStrides;
  SmallVector<int64_t> dstStrides;
};

std::optional<StrideInfo>
getStaticStrides(ShapedType srcType, ShapedType dstType) {
  auto srcMemref = dyn_cast<MemRefType>(srcType);
  auto dstMemref = dyn_cast<MemRefType>(dstType);
  if (!srcMemref || !dstMemref)
    return std::nullopt;
  auto srcLayout = dyn_cast<StridedLayoutAttr>(srcMemref.getLayout());
  auto dstLayout = dyn_cast<StridedLayoutAttr>(dstMemref.getLayout());
  if (!srcLayout || !dstLayout)
    return std::nullopt;
  auto srcStrides = srcLayout.getStrides();
  auto dstStrides = dstLayout.getStrides();
  for (auto s : srcStrides)
    if (s == ShapedType::kDynamic)
      return std::nullopt;
  for (auto s : dstStrides)
    if (s == ShapedType::kDynamic)
      return std::nullopt;
  return StrideInfo{SmallVector<int64_t>(srcStrides.begin(), srcStrides.end()),
                    SmallVector<int64_t>(dstStrides.begin(), dstStrides.end())};
}

/// Extract a single static stride from a ShapedType at the given axis.
/// Returns fallback if the type is not a MemRefType, has no StridedLayoutAttr,
/// or the stride is dynamic.
int64_t getStaticStride(ShapedType type, int64_t axis, int64_t fallback) {
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref)
    return fallback;
  auto layout = dyn_cast<StridedLayoutAttr>(memref.getLayout());
  if (!layout)
    return fallback;
  int64_t stride = layout.getStrides()[axis];
  return stride != ShapedType::kDynamic ? stride : fallback;
}

/// ARA dichotomy parameters derived from the reduction axis size s1.
struct ARADichotomyParams {
  int64_t mainSize;     // largest power of 2 <= s1
  int64_t tailSize;     // s1 - mainSize
  int64_t dichotomyNum; // log2(mainSize)
  int64_t repeatTimes;  // dichotomyNum + tailSize
};

ARADichotomyParams computeARADichotomy(int64_t s1) {
  int64_t mainSize = largestPow2Le(s1);
  int64_t tailSize = s1 - mainSize;
  int64_t dichotomyNum = 0;
  for (int64_t m = mainSize; m > 1; m >>= 1)
    dichotomyNum++;
  return {mainSize, tailSize, dichotomyNum, dichotomyNum + tailSize};
}

/// Compute the extra buffer size for a 3D middle-axis reduce (ARA).
/// The reduce_ara template has two runtime paths:
///   1. Dichotomy: repeatTimes <= s0, uses tmp of (mainSize/2)*s0*alignedS2
///   2. Peeled:    repeatTimes >  s0, loops over s0 calling reduce_ra
/// For XOR, both paths need additional buffer due to the XOR front-end.
///
/// \param baseline  The default tmp_buf (e.g., srcAllocTotalSize) to return
///                  when ARA-specific computation doesn't apply.
/// \param isXor     Whether the operation is XOR (needs extra XOR front buffer).
std::optional<int64_t>
computeARAExtraBufferSize(ShapedType srcType, int64_t baseline, bool isXor) {
  if (srcType.getRank() != 3 || !srcType.hasStaticShape())
    return baseline;

  int64_t s0 = srcType.getShape()[0];
  int64_t s1 = srcType.getShape()[1];
  int64_t s2 = srcType.getShape()[2];
  int64_t numPerBlk = getNumPerBlock(srcType.getElementType());
  int64_t stride1 = getStaticStride(srcType, srcType.getRank() - 2, s2);

  auto dp = computeARADichotomy(s1);
  int64_t alignedS2 = ceilFactor(s2, numPerBlk);

  if (dp.repeatTimes <= s0) {
    // Dichotomy path: (mainSize/2) * s0 * alignedS2
    int64_t dichotomyHalf = (dp.mainSize / 2) * s0 * alignedS2;
    if (isXor) {
      return std::max(baseline,
                      ceilFactor(dichotomyHalf, numPerBlk) + dichotomyHalf);
    }
    return std::max(baseline, dichotomyHalf);
  }

  // Peeled path: loop over s0, each iteration calls reduce_ra([s1, s2]).
  if (isXor) {
    int64_t xorFront = ceilFactor(s1 * alignedS2 / 2, numPerBlk);
    int64_t dichotomyPart = (dp.mainSize / 2) * stride1;
    int64_t xorReduceRaTmp = xorFront + dichotomyPart;
    return std::max(s1 * s2 + numPerBlk, xorReduceRaTmp);
  }
  int64_t reduceRaTmp = (dp.mainSize / 2) * stride1;
  return std::max(s1 * s2, reduceRaTmp);
}

/// Determine the effective row count (aDim) for the 3D+ last-axis reduce
/// (AAR pattern). This mirrors the path-selection logic in reduce_aar:
///   1. Fusion path  — src/dst fusable with matching rows_per_plane
///                      → fused 2D view [a0*rows_per_plane, r]
///   2. Fallback     — loop direction chosen by dst stride; aDim equals the
///                      inner-loop row count of a single reduce_ar call
///   3. No stride    — contiguous, flattened to [a0*a1, r]
int64_t getAARAdim(ShapedType srcType, ShapedType dstType) {
  int64_t rank = srcType.getRank();
  int64_t a0 = srcType.getShape()[rank - 3];
  int64_t a1 = srcType.getShape()[rank - 2];

  auto info = getStaticStrides(srcType, dstType);
  if (!info)
    return a0 * a1;

  int64_t srcStride0 = info->srcStrides[rank - 3];
  int64_t srcStride1 = info->srcStrides[rank - 2];
  int64_t srcStride2 = info->srcStrides[rank - 1];
  int64_t dstStride0 = info->dstStrides[rank - 3];
  int64_t dstStride1 = info->dstStrides[rank - 2];
  int64_t dstStride2 = info->dstStrides[rank - 1];

  bool srcFusable =
      srcStride1 != 0 && srcStride0 % srcStride1 == 0 && srcStride2 == 1;
  bool dstFusable =
      dstStride1 != 0 && dstStride0 % dstStride1 == 0 && dstStride2 == 1;

  if (srcFusable && dstFusable) {
    int64_t srcRowsPerPlane = srcStride0 / srcStride1;
    int64_t dstRowsPerPlane = dstStride0 / dstStride1;
    if (srcRowsPerPlane == dstRowsPerPlane)
      return a0 * srcRowsPerPlane;
  }

  // Fallback: mirror reduce_aar's loop-direction heuristic.
  // Loop-over-a0 → reduce_ar([a1, r]) → aDim = a1
  // Loop-over-a1 → reduce_ar([a0, r]) → aDim = a0
  bool loopOverA0 =
      (dstStride1 == 1) || (dstStride0 != 1 && a0 <= a1);
  return loopOverA0 ? a1 : a0;
}
} // namespace
static std::optional<int64_t>
refineBroadcastExtraBufferSize(ShapedType dstType, int64_t srcMaxSizeMaybe,
                               int64_t dstMaxSizeMaybe, AxisKind axisKind,
                               AlignKind alignKind) {
  if (dstType.getRank() == 1) {
    return std::nullopt;
  }

  auto dstShape = dstType.getShape();
  int64_t elementPerBlock =
      vectorBlockSizeBit / dstType.getElementTypeBitWidth();
  if (axisKind == AxisKind::FIRST) {
    if (alignKind == AlignKind::ALIGN) {
      return std::nullopt;
    } else {
      // Unknown broadcast temp buffer is same to unaligned broadcast.
      if (!dstType.hasStaticShape()) {
        return dstMaxSizeMaybe * brcFirstFactorUnalign;
      }
      // Calc first brc unalign/unknown_align temp: (1, ..., c) -> (b, ..., c)
      int64_t b = dstShape[0];
      int64_t c = dstShape[dstType.getRank() - 1];
      if (dstType.getRank() > 2) { // max first axis broadcast is 2
        // Calc first brc unalign/unknown_align temp: (1, ..., a, c) -> (b, ...,
        // a, c) BRC_FIRST_LIB_MAX_RANK = 3, a is the penultimate  axis.
        int64_t a = dstShape[dstType.getRank() - 2]; // reduce rank by 2

        // Convert Nd to (N-1)d: (b, ..., a, c) -> (b, ..., a*c)
        c = a * c;
      }

      // Calc first brc 2d unalign/unknown_align temp: (1, c) -> (b, c), other
      // axises will be throwed as loop.
      c = static_cast<int>(llvm::alignTo(c, elementPerBlock));
      return b * c;
    }
  }
  if (axisKind == AxisKind::MIDDLE) {
    if (alignKind == AlignKind::ALIGN) {
      return std::nullopt;
    } else {
      // TODO : support unalign
      llvm::report_fatal_error(
          "unsupport unalign and unknown align middle-axis broadcast");
    }
  }

  if (axisKind == AxisKind::LAST) {
    // Calc last brc (..., a, 1) -> (..., a, b) temp buffer
    int64_t a =
        dstShape[dstType.getRank() - 2]; // get the 2nd last shape of dest
    int64_t b = dstShape[dstType.getRank() - 1];
    if (alignKind == AlignKind::ALIGN) {
      bool needTempBuffer =
          ((a % srcNumPerRepeatOfVBRCBIntrin != 0) || (b != elementPerBlock)) &&
          (dstType.getElementTypeBitWidth() != 64);
      if (!needTempBuffer) {
        // When broadcast (a, 1) to (a, b), a is multiple of
        // NumPerRepeatOfVBRCBIntrin and b is elementPerBlock, temp buffer is
        // 0(not std::nullopt, because brc Op lib fun has temp buffer param).
        return 0;
      }

      if (!dstType.hasStaticShape()) {
        int64_t extra_buffer = std::max<int64_t>(
            dstMaxSizeMaybe * brcLastFactorAlign, 8 * elementPerBlock);
        // return the number of elements.
        return dstType.getElementTypeBitWidth() == 1
                   ? extra_buffer + elementPerBlock * 2 +
                         dstMaxSizeMaybe * halfBits
                   : extra_buffer;
      }

      a = static_cast<int>(llvm::alignTo(a, srcNumPerRepeatOfVBRCBIntrin));
      // return the number of elements.
      // need to calculate as 16-bit type
      return dstType.getElementTypeBitWidth() == 1
                 ? (a + 2) * elementPerBlock + a * halfBits
                 : a * elementPerBlock;
    } else {
      // Unknown broadcast temp buffer is same to unaligned broadcast.
      if (!dstType.hasStaticShape()) {
        if (srcMaxSizeMaybe == 0)
          return std::nullopt;
        auto alignedSrc =
            llvm::alignTo(srcMaxSizeMaybe, srcNumPerRepeatOfVBRCBIntrin);
        b = dstMaxSizeMaybe / srcMaxSizeMaybe;
        auto alignedB = llvm::alignTo(b, elementPerBlock);
        return alignedSrc * alignedB;
      }
      auto alignedB = llvm::alignTo(b, elementPerBlock);
      if (dstType.getElementTypeBitWidth() == 64) {
        return a * static_cast<int>(alignedB);
      }
      auto alignedA = llvm::alignTo(a, srcNumPerRepeatOfVBRCBIntrin);
      return alignedA * alignedB;
    }
  }

  return std::nullopt;
}

static std::optional<int64_t>
getExtraBufferSizeForBroadcastOpSingleDim(Operation *op, BufferSizeUnit unit,
                                          int64_t broadcastDim) {
  auto dpsOp = cast<DestinationStyleOpInterface>(op);
  // Extra buffer size is inferred from dst operand.
  auto *srcVec = dpsOp.getDpsInputOperand(0);
  auto *dstVec = dpsOp.getDpsInitOperand(0);
  ShapedType srcVecType = cast<ShapedType>(srcVec->get().getType());
  ShapedType dstVecType = cast<ShapedType>(dstVec->get().getType());
  AlignKind alignKind = deduceAlignmentForDPSInitOperand(*dstVec);
  AxisKind axisKind =
      utils::getOutlinedAxisKind(broadcastDim, dstVecType.getRank());
  if (axisKind == AxisKind::MIDDLE)
    // Mid axis does not need extra buffer.
    return std::nullopt;

  if (axisKind == AxisKind::FIRST) {
    if (alignKind == AlignKind::ALIGN)
      return std::nullopt;
    alignKind = AlignKind::UNALIGNED;

    if (unit == BufferSizeUnit::FACTOR)
      // Unknown broadcast temp buffer is same to unaligned broadcast.
      return brcFirstFactorUnalign;
  }

  if (axisKind == AxisKind::LAST) {
    if (llvm::all_of(srcVecType.getShape(),
                     [](int size) -> bool { return size == 1; }))
      // broadcast (1, ..., 1) to (1, ..., b) will be collapsed, which is equal
      // to broadcast 1d, and broadcast 1d do not need temp buffer.
      return std::nullopt;

    if (unit == BufferSizeUnit::FACTOR)
      // The exact value for temp buffer can only be calculated for
      // BufferSizeUnit::ELEMENT mode. This is just an upper bound value.
      return brcLastFactorUnalign;
  }

  // BufferSizeUnit::ELEMENT
  std::optional<int64_t> srcMaxSizeMaybe =
      utils::traceToAllocMaxSize(srcVec->get());
  std::optional<int64_t> dstMaxSizeMaybe =
      utils::traceToAllocMaxSize(dstVec->get());
  assert(srcMaxSizeMaybe && dstMaxSizeMaybe && "Alloc size is null.");
  return refineBroadcastExtraBufferSize(dstVecType, srcMaxSizeMaybe.value(),
                                        dstMaxSizeMaybe.value(), axisKind,
                                        alignKind);
}

std::optional<int64_t> getExtraBufferSizeForBroadcastOp(Operation *op,
                                                        BufferSizeUnit unit) {
  assert(op && isa<hivm::VBrcOp>(op) && "Operation should be a brc op!");
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  assert(dpsOp);
  if (dpsOp.hasPureBufferSemantics()) {
    if (unit != BufferSizeUnit::ELEMENT) {
      op->emitWarning("Currently only support inferring extra buffer size in "
                      "unit of element for bufferized op!");
      return 0;
    }
  }
  std::optional<int64_t> result;
  std::vector<int64_t> broadcastDims;
  if (auto vBrcOp = dyn_cast<hivm::VBrcOp>(op)) {
    broadcastDims = vBrcOp.getBroadcastDims();
  } else {
    llvm::report_fatal_error("Not implemented!");
  }
  for (auto broadcastDim : broadcastDims) {
    std::optional<int64_t> bufSizeMaybe =
        getExtraBufferSizeForBroadcastOpSingleDim(op, unit, broadcastDim);
    result = std::max(result, bufSizeMaybe);
  }
  return result;
}

std::optional<int64_t> getVcgReduceExtraBufferSize(ShapedType srcType,
                                                   int64_t srcAllocTotalSize,
                                                   int64_t reductionDim) {
  int64_t rDim = srcType.getShape()[reductionDim];
  auto eleType = srcType.getElementType();
  const int numPerBlock = getNumPerBlock(eleType);
  const int numPerRepeat = getNumPerRepeat(eleType);
  if (rDim <= numPerRepeat)
    return std::nullopt;
  if (reductionDim == 0) { // Reduce R
    int64_t buffSize = ceilFactor(srcAllocTotalSize, numPerBlock) / numPerBlock;
    return buffSize;
  }
  // Reduce AR, for now we assume it is a power of numPerBlock
  //  if not then we need align each row post reduction
  int64_t buffSize = ceilFactor(srcAllocTotalSize, numPerBlock) / numPerBlock;
  return buffSize;
}

std::optional<int64_t>
refineReduceExtraBufferSize(ShapedType srcType, int64_t srcAllocTotalSize,
                            int64_t reductionDim, hivm::ReduceOperation arithOp,
                            bool saveUbUf,
                            ShapedType dstType = ShapedType()) {
  auto eleType = srcType.getElementType();
  if (!srcType.hasStaticShape()) {
    if (eleType.isInteger() && (reductionDim == srcType.getRank() - 1)) {
      if (arithOp == hivm::ReduceOperation::xori) {
        return 3 * srcAllocTotalSize;
      }
      return 2 * srcAllocTotalSize;
    }
    return srcAllocTotalSize;
  }
  const int numPerBlock = getNumPerBlock(eleType);
  const int numPerRepeat = getNumPerRepeat(eleType);

  int64_t rDim = srcType.getShape()[reductionDim];
  int64_t aDim;
  if (reductionDim == 0) {
    aDim = 1;
  } else if (srcType.getRank() >= 3 &&
             reductionDim == srcType.getRank() - 1) {
    aDim = getAARAdim(srcType, dstType);
  } else {
    aDim = srcType.getShape()[reductionDim - 1];
  }
  int64_t extraBufferSize = 0;
  if (eleType.isInteger() || arithOp == hivm::ReduceOperation::prod ||
      arithOp == hivm::ReduceOperation::ori ||
      arithOp == hivm::ReduceOperation::xori) {
    if (rDim > numPerRepeat) {
      if (arithOp == hivm::ReduceOperation::xori) {
        extraBufferSize = aDim * numPerRepeat * 2 + aDim * numPerBlock;
      } else {
        extraBufferSize = aDim * numPerRepeat + aDim * numPerBlock;
      }
    } else {
      if (arithOp == hivm::ReduceOperation::xori) {
        extraBufferSize = 3 * srcAllocTotalSize;
      } else {
        extraBufferSize = 2 * srcAllocTotalSize;
      }
    }
    return extraBufferSize;
  }
  if ((eleType.isF32() || eleType.isF16())) {
    if ((arithOp == hivm::ReduceOperation::max ||
         arithOp == hivm::ReduceOperation::min ||
         arithOp == hivm::ReduceOperation::sum) &&
        saveUbUf) {
      return getVcgReduceExtraBufferSize(srcType, srcAllocTotalSize,
                                         reductionDim);
    }
    if ((arithOp == hivm::ReduceOperation::max ||
         arithOp == hivm::ReduceOperation::min) &&
        reductionDim == 0 && srcType.getRank() == 1) {
      if (rDim <= numPerRepeat) {
        return std::nullopt;
      }
      return numPerRepeat;
    }
    if (rDim < numPerBlock) {
      if (rDim % 2 == 0) {
        extraBufferSize = srcAllocTotalSize / 2;
      } else {
        return std::nullopt;
      }
    } else if (rDim >= numPerBlock && rDim <= numPerRepeat) {
      return std::nullopt;
    } else if (rDim > numPerRepeat && rDim <= numPerRepeat * 2) {
      extraBufferSize = aDim * numPerRepeat;
    } else if (rDim > numPerRepeat * 2) {
      extraBufferSize = srcAllocTotalSize / 2;
    }
    return extraBufferSize;
  }
  return srcAllocTotalSize;
}

std::optional<int64_t>
getExtraBufferSizeForReduceOpSingleDim(Operation *op, BufferSizeUnit unit,
                                       int64_t reductionDim, bool saveUbUf) {
  ShapedType srcType = cast<ShapedType>(op->getOpOperand(0).get().getType());
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  ShapedType dstType =
      dpsOp ? cast<ShapedType>(dpsOp.getDpsInitOperand(0)->get().getType())
            : ShapedType();
  auto vReduceOp = dyn_cast<hivm::VReduceOp>(op);
  hivm::ReduceOperation arithOp = vReduceOp.getArith().getReduceOp();
  auto eleType = srcType.getElementType();
  if (unit == BufferSizeUnit::FACTOR) {
    if (eleType.isInteger() && (reductionDim == srcType.getRank() - 1)) {
      if (arithOp == hivm::ReduceOperation::xori) {
        return 3 * REDUCE_DEFAULT_FACTOR;
      }
      return 2 * REDUCE_DEFAULT_FACTOR;
    }
    return REDUCE_DEFAULT_FACTOR;
  }

  std::optional<int64_t> srcAllocTotalSize =
      utils::traceToAllocMaxSize(op->getOpOperand(0).get());
  assert(srcAllocTotalSize);
  if (VReduceOp::isArgminOrArgmax(arithOp)) {
    // * R/AR: 1 ub_block_unit
    // * RA: r * sizeof(Index) aligned to ub_block_unit + 1 extra ub_block_unit
    int64_t rank = srcType.getRank();
    int64_t elementBitWidth = srcType.getElementTypeBitWidth();
    assert(vectorBlockSizeBit % elementBitWidth == 0);
    int64_t numElemPerBlock = vectorBlockSizeBit / elementBitWidth;
    if (reductionDim == rank - 1) {
      // R/AR
      return numElemPerBlock;
    }
    if (srcType.hasStaticShape()) {
      int64_t aDimension = 1; // This is A dimension in RA term
      for (auto dim : srcType.getShape()) {
        if (dim == reductionDim) {
          continue;
        }
        aDimension *= dim;
      }
      constexpr int64_t bitsPerByte = 8;
      // FIXME indices hardcodes as int32
      const int64_t eltByteSize = CEIL_DIV(elementBitWidth, bitsPerByte);
      constexpr int64_t idxByteSize = sizeof(int32_t);
      int64_t numElemPerRepeat = getNumPerRepeat(eleType);
      int64_t elementsSize =
          ceilDiv(aDimension * idxByteSize, numElemPerRepeat) * numElemPerRepeat /
          eltByteSize;
      int64_t maskSize =
          ceilDiv(aDimension, bitsPerByte * numElemPerBlock) * numElemPerBlock;
      return elementsSize + maskSize + numElemPerBlock;
    }
    // RA, dynamic shape: 1.5× alloc_size (int64_t, no float), align to block.
    const int64_t alloc = srcAllocTotalSize.value();
    const int64_t scaledAlloc = static_cast<int64_t>(1.5 * alloc);
    return ceilFactor(scaledAlloc, numElemPerBlock);
  }
  if (arithOp == hivm::ReduceOperation::sum ||
      arithOp == hivm::ReduceOperation::max ||
      arithOp == hivm::ReduceOperation::min ||
      arithOp == hivm::ReduceOperation::prod ||
      arithOp == hivm::ReduceOperation::ori ||
      arithOp == hivm::ReduceOperation::andi) {
    if (reductionDim != srcType.getRank() - 1) {
      // reduce(RA/RA0A1) — not last axis.
      int64_t baseline = srcAllocTotalSize.value();
      return computeARAExtraBufferSize(srcType, baseline, /*isXor=*/false);
    }
    // reduce_sum/reduce_max/reduce_min/reduce_prod
    // reduce_or/reduce_and last axis
    // reduce(R/AR).
    return refineReduceExtraBufferSize(srcType, srcAllocTotalSize.value(),
                                       reductionDim, arithOp, saveUbUf, dstType);
  }
  if (arithOp == hivm::ReduceOperation::xori) {
    if (reductionDim != srcType.getRank() - 1) {
      // reduce_xor not last axis reduce(RA/RA0A1)
      int64_t elementPerBlock =
          vectorBlockSizeBit / srcType.getElementTypeBitWidth();
      int64_t baseline = srcAllocTotalSize.value() + elementPerBlock;
      return computeARAExtraBufferSize(srcType, baseline, /*isXor=*/true);
    }
    // reduce_xor last axis reduce(R/AR)
    return refineReduceExtraBufferSize(srcType, srcAllocTotalSize.value(),
                                       reductionDim, arithOp, saveUbUf, dstType);
  }
  llvm::report_fatal_error("unsupported reduce case");
}

std::optional<int64_t> getExtraBufferSizeForReduceOp(Operation *op,
                                                     BufferSizeUnit unit) {
  bool saveUbUf = false;
  if (op->getParentOfType<func::FuncOp>()->hasAttr(
          hivm::EnableSavingUbAttr::name)) {
    saveUbUf = true;
  }
  assert(op && isa<hivm::VReduceOp>(op) && "Operation should be a reduce op!");
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  assert(dpsOp);
  if (dpsOp.hasPureBufferSemantics()) {
    if (unit != BufferSizeUnit::ELEMENT) {
      op->emitWarning("Currently only support inferring extra buffer size in "
                      "unit of element for bufferized op!");
      return 0;
    }
  }

  auto vReduceOp = dyn_cast<hivm::VReduceOp>(op);
  std::vector<int64_t> reductionDims = vReduceOp.getReduceDims();
  std::optional<int64_t> bufSize = 0;
  bool needTempBuffer = false;
  for (auto reductionDim : reductionDims) {
    std::optional<int64_t> tmpBufSize = getExtraBufferSizeForReduceOpSingleDim(
        op, unit, reductionDim, saveUbUf);
    if (tmpBufSize) {
      bufSize = std::max(bufSize, tmpBufSize);
      needTempBuffer = true;
    }
  }
  return needTempBuffer ? bufSize : std::nullopt;
}

} // namespace util
} // namespace hivm
} // namespace mlir
