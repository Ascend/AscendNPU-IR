//===- Utils.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

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
      llvm_unreachable(
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
    llvm_unreachable("Not implemented!");
  }
  for (auto broadcastDim : broadcastDims) {
    std::optional<int64_t> bufSizeMaybe =
        getExtraBufferSizeForBroadcastOpSingleDim(op, unit, broadcastDim);
    result = std::max(result, bufSizeMaybe);
  }
  return result;
}

std::optional<int64_t>
refineReduceExtraBufferSize(ShapedType srcType, int64_t srcAllocTotalSize,
                            int64_t reductionDim,
                            hivm::ReduceOperation arithOp) {
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
  const int numPerBlock = mlir::utils::getNumPerBlock(eleType);
  const int numPerRepeat = mlir::utils::getNumPerRepeat(eleType);

  int64_t rDim = srcType.getShape()[reductionDim];
  int64_t aDim = srcType.getShape()[0];
  if (reductionDim == 0) {
    aDim = 1;
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
                                       int64_t reductionDim) {
  ShapedType srcType = cast<ShapedType>(op->getOpOperand(0).get().getType());
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
  if (utils::isReduceWithIndex(arithOp)) {
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
      // RA, static shape
      // use r * sizeof(Index) aligned to ub_block_unit + 1 extra ub_block_unit
      int64_t reductionDimLength = srcType.getShape()[reductionDim];
      // TODO: library only supports 32 bit index; add verifier for
      // ReduceWithIndexOp to check this
      ShapedType indexType =
          cast<ShapedType>(vReduceOp.getDpsInits()[1].getType());
      int64_t indexBitWidth = indexType.getElementTypeBitWidth();
      int64_t totalBitLength =
          ceilFactor(reductionDimLength * indexBitWidth, vectorBlockSizeBit) +
          vectorBlockSizeBit;
      return totalBitLength / elementBitWidth;
    }
    // RA, dynamic shape
    // use 1.5 * alloc_size aligned to ub_block_unit
    return ceilFactor(1.5 * srcAllocTotalSize.value(), numElemPerBlock);
  }
  if (arithOp == hivm::ReduceOperation::sum ||
      arithOp == hivm::ReduceOperation::max ||
      arithOp == hivm::ReduceOperation::min ||
      arithOp == hivm::ReduceOperation::prod ||
      arithOp == hivm::ReduceOperation::ori ||
      arithOp == hivm::ReduceOperation::andi) {
    if (reductionDim != srcType.getRank() - 1) {
      // reduce_sum/reduce_max/reduce_min/reduce_prod
      // reduce_or/reduce_and not last axis
      // reduce(RA/RA0A1).
      return srcAllocTotalSize.value();
    }
    // reduce_sum/reduce_max/reduce_min/reduce_prod
    // reduce_or/reduce_and last axis
    // reduce(R/AR).
    return refineReduceExtraBufferSize(srcType, srcAllocTotalSize.value(),
                                       reductionDim, arithOp);
  }
  if (arithOp == hivm::ReduceOperation::xori) {
    if (reductionDim != srcType.getRank() - 1) {
      // reduce_xor not last axis reduce(RA/RA0A1), requires additional tmp_buf
      // space of src/2 to process the xor operation. Since src/2 will cause the
      // instruction starting address to be misaligned by 32 bytes, an
      // additional block is required.
      int64_t elementPerBlock =
          vectorBlockSizeBit / srcType.getElementTypeBitWidth();
      return srcAllocTotalSize.value() + elementPerBlock;
    }
    // reduce_xor last axis reduce(R/AR)
    return refineReduceExtraBufferSize(srcType, srcAllocTotalSize.value(),
                                       reductionDim, arithOp);
  }
  llvm_unreachable("unsupported reduce case");
}

std::optional<int64_t> getExtraBufferSizeForReduceOp(Operation *op,
                                                     BufferSizeUnit unit) {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (hacc::utils::isRegBasedArch(moduleOp))
    return std::nullopt;
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
    std::optional<int64_t> tmpBufSize =
        getExtraBufferSizeForReduceOpSingleDim(op, unit, reductionDim);
    if (tmpBufSize) {
      bufSize = std::max(bufSize, tmpBufSize);
      needTempBuffer = true;
    }
  }
  return needTempBuffer ? bufSize : std::nullopt;
}

AlignKind deduceAlignmentForDPSInitOperand(OpOperand &operand) {
  Value operandValue = operand.get();
  MemRefType maybeMemRefType = dyn_cast<MemRefType>(operandValue.getType());
  if (maybeMemRefType)
    return deduceAlignmentForMemRefType(maybeMemRefType);

  // Try deduce alignment kind for tensor.
  AlignKind alignKind{AlignKind::UNKNOWN};
  auto owner = dyn_cast<DestinationStyleOpInterface>(operand.getOwner());
  if (!owner)
    return alignKind;

  // If tied result is tagged with alignment info, return it as it is.
  Value tiedResult = owner.getTiedOpResult(&operand);
  auto markOpsWithAlignmentInfo =
      llvm::make_filter_range(tiedResult.getUsers(), [](Operation *user) {
        return isa<annotation::MarkOp>(user) &&
               user->hasAttrOfType<AlignKindAttr>(AlignKindAttr::name);
      });
  if (markOpsWithAlignmentInfo.empty())
    return alignKind;

  auto alignmentInfo =
      llvm::map_to_vector<1>(markOpsWithAlignmentInfo, [](Operation *markOp) {
        return markOp->getAttrOfType<AlignKindAttr>(AlignKindAttr::name)
            .getValue();
      });
  if (!llvm::all_equal(alignmentInfo)) {
    LDBG("WARNING: Conflicting alignment annotation for operand #"
         << operand.getOperandNumber() << " in " << *owner);
    return AlignKind::UNKNOWN;
  }
  return alignmentInfo.front();
}

AlignKind deduceAlignmentForMemRefType(MemRefType vecType) {
  Type eleType = vecType.getElementType();
  int eleSize = static_cast<int>(eleType.getIntOrFloatBitWidth() / 8);

  AlignKind alignKind{AlignKind::UNKNOWN};
  int64_t toCheck{0};

  StridedLayoutAttr dstLayout =
      dyn_cast<StridedLayoutAttr>(vecType.getLayout());
  if (dstLayout) {
    ArrayRef<int64_t> strides = dstLayout.getStrides();
    if (strides.size() <
        2) { // if strides is less than 2, alignment is impossible
      return AlignKind::UNKNOWN;
    }

    toCheck = strides[strides.size() - 2]; // get the 2nd last strides
  } else {
    int rank = vecType.getRank();
    if (rank == 0) {
      return AlignKind::UNKNOWN;
    }

    toCheck = vecType.getDimSize(rank - 1);
  }

  if (toCheck != ShapedType::kDynamic) {
    auto isAlignedToBlock = [](int eleNum, int eleSize) {
      return eleNum * eleSize % BL == 0;
    };
    if (isAlignedToBlock(toCheck, eleSize)) {
      alignKind = AlignKind::ALIGN;
    } else {
      alignKind = AlignKind::UNALIGNED;
    }
  } else {
    alignKind = AlignKind::UNKNOWN;
  }

  return alignKind;
}

} // namespace util
} // namespace hivm
} // namespace mlir
