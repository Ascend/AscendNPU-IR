//===- GetExtraBuffers.cpp - HIVM get extra buffer implementation ---------===//
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// Macros to help generate `getExtraBuffer`
//===----------------------------------------------------------------------===//

#define ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(OP_NAME)     \
  OperandRange OP_NAME::getExtraBuffers() { return getTempBufferMutable(); }

#define ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(OP_NAME)     \
  bool OP_NAME::shouldAllocExtraBufferForScalarOrOTFBrc() {                    \
    return !this->getBroadcastAttr().empty() ||                                \
           llvm::any_of(getOperandTypes(),                                     \
                        [&](Type type) { return type.isIntOrFloat(); });       \
  }

#define ENABLE_OP_SHOULD_NOT_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(OP_NAME) \
  bool OP_NAME::shouldAllocExtraBufferForScalarOrOTFBrc() { return false; }

ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VReduceOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VSelOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VBrcOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VCastOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VXorOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VPowOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VInterleaveOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VMulextendedOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VGatherOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VTransposeOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VSortOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VSubOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VDivOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VMulOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VAddOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VMaxOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VMinOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VAndOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VOrOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VShLOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VShROp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VNotOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VAbsOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VLnOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VReluOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VExpOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VRsqrtOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VSqrtOp)
ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION(VRecOp)
#undef ENABLE_DEFAULT_OP_GET_OPTIONAL_TEMP_BUFFER_IMPLEMENTATION

// Vector Binary Op
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VAddOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VMulOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VMaxOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VMinOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VSubOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VDivOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VAndOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VOrOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VShLOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VShROp)
// Vector Unary Op
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VNotOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VAbsOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VLnOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VReluOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VExpOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VRsqrtOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VSqrtOp)
ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VRecOp)
#undef ENABLE_OP_SHOULD_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC

ENABLE_OP_SHOULD_NOT_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VSelOp)
ENABLE_OP_SHOULD_NOT_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VBrcOp)
ENABLE_OP_SHOULD_NOT_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VCastOp)
ENABLE_OP_SHOULD_NOT_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VReduceOp)
ENABLE_OP_SHOULD_NOT_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VTransposeOp)
ENABLE_OP_SHOULD_NOT_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VInterleaveOp)
ENABLE_OP_SHOULD_NOT_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VXorOp)
ENABLE_OP_SHOULD_NOT_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VMulextendedOp)
ENABLE_OP_SHOULD_NOT_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VPowOp)
ENABLE_OP_SHOULD_NOT_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VGatherOp)
ENABLE_OP_SHOULD_NOT_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC(VSortOp)
#undef ENABLE_OP_SHOULD_NOT_ALLOC_EXTRA_BUFFER_FOR_SCALAR_OR_OTF_BRC

//===----------------------------------------------------------------------===//
// Macros to help generate `getExtraBufferSize`
//===----------------------------------------------------------------------===//

namespace mlir::hivm {
int64_t getLastAxisInlineBrcBuffSize(MemRefType srcVecType,
                                     int64_t upperLimit) {
  unsigned int srcWidth = srcVecType.getElementType().getIntOrFloatBitWidth();
  int64_t numPerBlock =
      (util::INTR_BYTES_PER_BLOCK * utils::INTR_BITS_PER_BYTE) / srcWidth;
  auto srcSizes = srcVecType.getShape();
  int rank = srcVecType.getRank();
  assert(rank >= 1 && "rank must be >= 1.");
  if (rank == 1) {
    return numPerBlock;
  }

  if (srcSizes[rank - 2] == ShapedType::kDynamic) {
    return upperLimit;
  }

  if (srcSizes[rank - 2] == 1) {
    return numPerBlock;
  }
  return numPerBlock * util::ceilFactor(srcSizes[rank - 2],
                                        util::srcNumPerRepeatOfVBRCBIntrin);
}

template <typename HIVMOP>
bool isHardwareNotSupportedVS() {
  return std::is_same<hivm::VDivOp, HIVMOP>::value ||
         std::is_same<hivm::VSubOp, HIVMOP>::value;
}

bool isSrcBrcInline(MemRefType srcVecType, MemRefType dstVecType) {
  auto srcSizes = srcVecType.getShape();
  auto dstSizes = dstVecType.getShape();
  int rank = dstVecType.getRank();
  bool isSrcBrcInline = false;
  if (srcSizes[rank - 1] == ShapedType::kDynamic) {
    return isSrcBrcInline;
  }
  if (dstSizes[rank - 1] != ShapedType::kDynamic) {
    isSrcBrcInline =
        srcSizes[rank - 1] != dstSizes[rank - 1] && srcSizes[rank - 1] == 1;
  } else {
    isSrcBrcInline = srcSizes[rank - 1] == 1;
  }
  return isSrcBrcInline;
}

template <typename HIVMOP>
std::optional<int64_t> getExtraBufferSizeForBinaryOp(HIVMOP op) {
  MemRefType dstVecType = cast<MemRefType>(op.getDst()[0].getType());
  unsigned int dstWidth = dstVecType.getElementType().getIntOrFloatBitWidth();
  int64_t numPerBlock =
      (util::INTR_BYTES_PER_BLOCK * utils::INTR_BITS_PER_BYTE) / dstWidth;

  int64_t baseBuffSize = 0;
  bool src0ScalarType = op.getSrc()[0].getType().isIntOrFloat();
  bool src1ScalarType = op.getSrc()[1].getType().isIntOrFloat();
  assert((!src0ScalarType || !src1ScalarType) &&
         "Binary vector elementwise does not support both source is scalar.");
  if (isHardwareNotSupportedVS<HIVMOP>()) {
    // The hardware does not support the vs op and requires an additional
    // tmp_buf for scalar
    if (src0ScalarType || src1ScalarType) {
      baseBuffSize = numPerBlock;
    }
  }

  // The last axis brc inline scenario requires an additional tmp_buf
  int64_t lastAxisInlineBrcBuffSize = 0;
  bool isSrc0BrcInline = false;
  bool isSrc1BrcInline = false;
  int64_t upperLimit = utils::traceToAllocMaxSize(op.getDst()[0]).value();
  if (!src0ScalarType) {
    MemRefType src0VecType = cast<MemRefType>(op.getSrc()[0].getType());
    isSrc0BrcInline = isSrcBrcInline(src0VecType, dstVecType);
    if (isSrc0BrcInline) {
      lastAxisInlineBrcBuffSize +=
          getLastAxisInlineBrcBuffSize(src0VecType, upperLimit);
    }
  }
  if (!src1ScalarType) {
    MemRefType src1VecType = cast<MemRefType>(op.getSrc()[1].getType());
    isSrc1BrcInline = isSrcBrcInline(src1VecType, dstVecType);
    if (isSrc1BrcInline) {
      lastAxisInlineBrcBuffSize +=
          getLastAxisInlineBrcBuffSize(src1VecType, upperLimit);
    }
  }
  // In the vv scenario, the inline brc is not used and in the vs scenario,
  // hardware support for vs instructions does not require tmp_buf.
  if ((!src0ScalarType && !src1ScalarType && !isSrc0BrcInline &&
       !isSrc1BrcInline) ||
      (!src0ScalarType && src1ScalarType &&
       !isHardwareNotSupportedVS<HIVMOP>() && !isSrc0BrcInline)) {
    return std::nullopt;
  }

  return baseBuffSize + lastAxisInlineBrcBuffSize;
}

template <typename HIVMOP>
std::optional<int64_t> getExtraBufferSizeForUnaryOp(HIVMOP op) {
  MemRefType dstVecType = cast<MemRefType>(op.getDst()[0].getType());

  assert((!(op.getSrc()[0].getType().isIntOrFloat())) &&
         "Unary vector elementwise does not support source is scalar.");

  // The last axis brc inline scenario requires an additional tmp_buf
  MemRefType src0VecType = cast<MemRefType>(op.getSrc()[0].getType());
  auto src0Sizes = src0VecType.getShape();
  int rank = dstVecType.getRank();
  if (src0Sizes[rank - 1] == ShapedType::kDynamic) {
    return std::nullopt;
  }

  bool isSrc0BrcInline = isSrcBrcInline(src0VecType, dstVecType);
  if (!isSrc0BrcInline) {
    return std::nullopt;
  }

  int64_t upperLimit = utils::traceToAllocMaxSize(op.getDst()[0]).value();
  int64_t lastAxisInlineBrcBuffSize =
      getLastAxisInlineBrcBuffSize(src0VecType, upperLimit);
  return lastAxisInlineBrcBuffSize;
}
} // namespace mlir::hivm

//===----------------------------------------------------------------------===//
// Macros for getExtraBufferSize
//===----------------------------------------------------------------------===//

#define ENABLE_BINARY_OP_GET_EXTRA_BUFFER_SIZE(OP_NAME)                        \
  std::optional<int64_t> OP_NAME::getExtraBufferSize() {                       \
    return getExtraBufferSizeForBinaryOp(*this);                               \
  }

#define ENABLE_UNARY_OP_GET_EXTRA_BUFFER_SIZE(OP_NAME)                         \
  std::optional<int64_t> OP_NAME::getExtraBufferSize() {                       \
    return getExtraBufferSizeForUnaryOp(*this);                                \
  }

// Vector Binary Op
ENABLE_BINARY_OP_GET_EXTRA_BUFFER_SIZE(VMulOp)
ENABLE_BINARY_OP_GET_EXTRA_BUFFER_SIZE(VAddOp)
ENABLE_BINARY_OP_GET_EXTRA_BUFFER_SIZE(VMaxOp)
ENABLE_BINARY_OP_GET_EXTRA_BUFFER_SIZE(VMinOp)
ENABLE_BINARY_OP_GET_EXTRA_BUFFER_SIZE(VAndOp)
ENABLE_BINARY_OP_GET_EXTRA_BUFFER_SIZE(VOrOp)
ENABLE_BINARY_OP_GET_EXTRA_BUFFER_SIZE(VSubOp)
ENABLE_BINARY_OP_GET_EXTRA_BUFFER_SIZE(VDivOp)
ENABLE_BINARY_OP_GET_EXTRA_BUFFER_SIZE(VShLOp)
ENABLE_BINARY_OP_GET_EXTRA_BUFFER_SIZE(VShROp)
#undef ENABLE_BINARY_OP_GET_EXTRA_BUFFER_SIZE
// Vector Unary Op
ENABLE_UNARY_OP_GET_EXTRA_BUFFER_SIZE(VNotOp)
ENABLE_UNARY_OP_GET_EXTRA_BUFFER_SIZE(VAbsOp)
ENABLE_UNARY_OP_GET_EXTRA_BUFFER_SIZE(VLnOp)
ENABLE_UNARY_OP_GET_EXTRA_BUFFER_SIZE(VReluOp)
ENABLE_UNARY_OP_GET_EXTRA_BUFFER_SIZE(VExpOp)
ENABLE_UNARY_OP_GET_EXTRA_BUFFER_SIZE(VRsqrtOp)
ENABLE_UNARY_OP_GET_EXTRA_BUFFER_SIZE(VSqrtOp)
ENABLE_UNARY_OP_GET_EXTRA_BUFFER_SIZE(VRecOp)
#undef ENABLE_UNARY_OP_GET_EXTRA_BUFFER_SIZE

//===----------------------------------------------------------------------===//
// VSelOp
//===----------------------------------------------------------------------===//

std::optional<int64_t> VSelOp::getExtraBufferSize() {
  MemRefType dstVecType = cast<MemRefType>(this->getDst()[0].getType());

  // vsel int64 vv case with i8 condition
  int rank = dstVecType.getRank();
  int libMaxRank = 1u;
  int64_t dstSize = 1;
  if (dstVecType.hasStaticShape()) {
    for (int i = 1; i <= libMaxRank; i++) {
      dstSize *= dstVecType.getShape()[rank - i];
    }
  } else {
    dstSize = utils::traceToAllocMaxSize(this->getDst()[0]).value();
  }
  uint64_t numI32PerBlock = util::INTR_BYTES_PER_BLOCK / sizeof(int32_t);
  uint64_t numHalfPerBlock = util::INTR_BYTES_PER_BLOCK / sizeof(int16_t);
  const int i32Factor = sizeof(int64_t) / sizeof(int32_t);
  const int halfFactor = sizeof(int64_t) / sizeof(int16_t);
  int64_t i32AlignSize =
      util::ceilFactor(dstSize, static_cast<int64_t>(numI32PerBlock)) /
      i32Factor;
  int64_t halfAlignSize =
      util::ceilFactor(dstSize, static_cast<int64_t>(numHalfPerBlock)) /
      halfFactor;
  // For cast i8->half->i32: need 1 size i32 type buffer + 1 size half type
  // buffer. Reuse memory for the rest of the implementation
  int64_t extraSize = i32AlignSize + halfAlignSize;
  return extraSize;
}

//===----------------------------------------------------------------------===//
// VSortOp
//===----------------------------------------------------------------------===//

std::optional<int64_t> VSortOp::getExtraBufferSize() {
  ShapedType srcVecType = cast<ShapedType>(getSrc().getType());
  bool needSortIndex = getDst().size() == 2;
  auto rank = srcVecType.getRank();
  int64_t sortDimSize = 0;
  if (srcVecType.hasStaticShape()) {
    int64_t lastRankDim = srcVecType.getShape()[rank - 1];
    sortDimSize = lastRankDim == 1 ? 0 : lastRankDim;
  } else {
    std::optional<int64_t> srcAllocTotalSize =
        utils::traceToAllocMaxSize(this->getSrc());
    assert(srcAllocTotalSize);
    sortDimSize = srcAllocTotalSize.value();
  }

  auto sortAxisAlign = util::ceilFactor(sortDimSize, VBITSORT_NUM_PER_REPEAT);
  bool isDescending = this->getDescending();
  Type elemType = getElementTypeOrSelf(this->getSrc().getType());

  if (needSortIndex) {
    if (isDescending) {
      if (elemType.isF32())
        return sortAxisAlign * 4;
      if (elemType.isF16())
        return sortAxisAlign * 8;
    } else {
      if (elemType.isF32())
        return sortAxisAlign * 5;
      if (elemType.isF16())
        return sortAxisAlign * 9;
    }
  } else {
    if (isDescending) {
      if (elemType.isF32())
        return sortAxisAlign * 5;
      if (elemType.isF16())
        return sortAxisAlign * 10;
    } else {
      if (elemType.isF32())
        return sortAxisAlign * 6;
      if (elemType.isF16())
        return sortAxisAlign * 11;
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// VXor
//===----------------------------------------------------------------------===//

std::optional<int64_t> VXorOp::getExtraBufferSize() {
  std::optional<int64_t> srcAllocTotalSize =
      utils::traceToAllocMaxSize(this->getSrc()[0]);
  return srcAllocTotalSize;
}