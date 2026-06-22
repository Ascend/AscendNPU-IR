//===- ExtraBuffer.cpp ----------------------------------------------------===//
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

#include "bishengir/Dialect/HFusion/Utils/ExtraBuffer.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

namespace mlir {
namespace hfusion {
namespace util {
const static int brcFirstFactorUnalign = 1;
const static int brcLastFactorAlign = 2;
const static int brcLastFactorUnalign = 8;
const static int halfBits = 16;

std::optional<int64_t>
refineBroadcastExtraBufferSize(ShapedType dstType, int64_t srcMaxSizeMaybe,
                               int64_t dstMaxSizeMaybe, hivm::AxisKind axisKind,
                               hivm::AlignKind alignKind) {
  if (dstType.getRank() == 1) {
    return std::nullopt;
  }

  auto dstShape = dstType.getShape();
  int64_t elementPerBlock =
      vectorBlockSizeBit / dstType.getElementTypeBitWidth();
  if (axisKind == hivm::AxisKind::FIRST) {
    if (alignKind == hivm::AlignKind::ALIGN) {
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
  if (axisKind == hivm::AxisKind::MIDDLE) {
    if (alignKind == hivm::AlignKind::ALIGN) {
      return std::nullopt;
    } else {
      // TODO : support unalign
      llvm::report_fatal_error(
          "unsupport unalign and unknown align middle-axis broadcast");
    }
  }

  if (axisKind == hivm::AxisKind::LAST) {
    // Calc last brc (..., a, 1) -> (..., a, b) temp buffer
    int64_t a =
        dstShape[dstType.getRank() - 2]; // get the 2nd last shape of dest
    int64_t b = dstShape[dstType.getRank() - 1];
    if (alignKind == hivm::AlignKind::ALIGN) {
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
  hivm::AlignKind alignKind = deduceAlignmentForDPSInitOperand(*dstVec);
  hivm::AxisKind axisKind =
      utils::getOutlinedAxisKind(broadcastDim, dstVecType.getRank());
  if (axisKind == hivm::AxisKind::MIDDLE)
    // Mid axis does not need extra buffer.
    return std::nullopt;

  if (axisKind == hivm::AxisKind::FIRST) {
    if (alignKind == hivm::AlignKind::ALIGN)
      return std::nullopt;
    alignKind = hivm::AlignKind::UNALIGNED;

    if (unit == BufferSizeUnit::FACTOR)
      // Unknown broadcast temp buffer is same to unaligned broadcast.
      return brcFirstFactorUnalign;
  }

  if (axisKind == hivm::AxisKind::LAST) {
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
  assert(op && isa<linalg::BroadcastOp>(op) && "Operation should be a brc op!");
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
  if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(op)) {
    broadcastDims = broadcastOp.getDimensions();
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

std::optional<int64_t> getExtraBufferSizeForReduceOp(Operation *op,
                                                     BufferSizeUnit unit) {
  assert(op &&
         (isa<linalg::ReduceOp>(op) || isa<hfusion::ReduceWithIndexOp>(op)) &&
         "Operation should be a reduce op!");
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  assert(dpsOp);
  if (dpsOp.hasPureBufferSemantics()) {
    if (unit != BufferSizeUnit::ELEMENT) {
      op->emitWarning("Currently only support inferring extra buffer size in "
                      "unit of element for bufferized op!");
      return 0;
    }
  }

  if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op)) {
    // Extra buffer size is inferred from source operand.
    std::optional<int64_t> bufSize =
        unit == BufferSizeUnit::ELEMENT
            ? utils::traceToAllocMaxSize(dpsOp.getDpsInputOperand(0)->get())
            : 1;
    return bufSize;
  }

  if (auto reduceOp = dyn_cast<hfusion::ReduceWithIndexOp>(op)) {
    assert(unit == BufferSizeUnit::FACTOR);
    // Cannot use 1.5 here because the return type is std::optional<int64_t>
    std::optional<int64_t> bufSize = 2;
    return bufSize;
  }

  return std::nullopt;
}
} // namespace util
} // namespace hfusion
} // namespace mlir