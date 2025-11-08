//===- AllocBuffers.cpp - HIVM extra buffer alloc implementation ----------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::hivm;

namespace {
Value allocExtraBuffer(Operation *op, const SmallVector<int64_t> &bufSize,
                       Type elemType) {
  mlir::IRRewriter rewriter(op->getContext());
  rewriter.setInsertionPoint(op);
  return rewriter.create<memref::AllocOp>(op->getLoc(),
                                          MemRefType::get(bufSize, elemType));
}
} // namespace

//===----------------------------------------------------------------------===//
// Macros for AllocExtraBufferIfPossible
//===----------------------------------------------------------------------===//

#define ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(OP_NAME)                         \
  LogicalResult OP_NAME::allocExtraBuffersIfPossible() {                       \
    if (this->getTempBuffer()) {                                               \
      this->emitWarning("already has extra temp buffer");                      \
      return success();                                                        \
    }                                                                          \
    auto bufSizeMaybe = getExtraBufferSize();                                  \
    if (bufSizeMaybe) {                                                        \
      MemRefType dstVecType = cast<MemRefType>(getDst()[0].getType());         \
      Value extraBuf = allocExtraBuffer(getOperation(), {*bufSizeMaybe},       \
                                        dstVecType.getElementType());          \
      this->getTempBufferMutable().assign(extraBuf);                           \
      return success();                                                        \
    }                                                                          \
    return success();                                                          \
  }

// Vector Binary Op
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VMulOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VAddOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VMaxOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VMinOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VAndOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VOrOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VSubOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VDivOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VShLOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VShROp)
// Vector Unary Op
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VNotOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VAbsOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VLnOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VReluOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VExpOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VRsqrtOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VSqrtOp)
ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE(VRecOp)
#undef ENABLE_ALLOC_EXTRA_BUFFER_IF_POSSIBLE

//===----------------------------------------------------------------------===//
// VBrcOp
//===----------------------------------------------------------------------===//

LogicalResult VBrcOp::allocExtraBuffersIfPossible() {
  if (this->getTempBuffer()) {
    this->emitWarning("already has extra temp buffer");
    return success();
  }
  llvm::ArrayRef<int64_t> broadcastDims = this->getBroadcastDims();
  if (broadcastDims.size() > 1) {
    return this->emitError("broadcast dimensions array is not decomposed yet");
  }

  auto bufSizeMaybe = util::getExtraBufferSizeForBroadcastOp(
      this->getOperation(), util::BufferSizeUnit::ELEMENT);
  if (bufSizeMaybe) {
    MemRefType dstVecType = cast<MemRefType>(this->getDst().getType());
    Value extraBuf = allocExtraBuffer(this->getOperation(), {*bufSizeMaybe},
                                      dstVecType.getElementType());
    this->getTempBufferMutable().assign(extraBuf);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// VCastOp
//===----------------------------------------------------------------------===//

LogicalResult VCastOp::allocExtraBuffersIfPossible() {
  if (this->getTempBuffer()) {
    this->emitWarning("already has extra temp buffer");
    return success();
  }

  auto srcType = getElementTypeOrSelf(this->getSrc()[0]);
  auto dstType = getElementTypeOrSelf(this->getDst()[0]);
  const bool isI32ToI8 = srcType.isInteger(32) && dstType.isInteger(8);
  const bool isI16ToI8 = srcType.isInteger(16) && dstType.isInteger(8);
  const bool isI64ToI32 = srcType.isInteger(64) && dstType.isInteger(32);
  const bool isI32ToI16 = srcType.isInteger(32) && dstType.isInteger(16);
  if (!srcType.isInteger(1) && !isI32ToI8 && !isI16ToI8 && !isI32ToI16 &&
      !isI64ToI32) {
    return success();
  }

  MemRefType dstVecType = cast<MemRefType>(this->getDst()[0].getType());
  Value extraBuf;
  if (isI32ToI8 || isI16ToI8) {
    // The implementation of high-bit truncation through transposition
    // requires additional tmp_buf to store the transposed result.
    std::optional<int64_t> srcAllocTotalSize =
        utils::traceToAllocMaxSize(this->getSrc()[0]);
    SmallVector<int64_t> extraBufSizes{srcAllocTotalSize.value() * 2};
    ShapedType srcVecType = cast<ShapedType>(this->getSrc()[0].getType());
    auto eleType = srcVecType.getElementType();
    extraBuf = allocExtraBuffer(this->getOperation(), extraBufSizes, eleType);
  } else if (isI32ToI16 || isI64ToI32) {
    if (this->getRoundMode() != hivm::RoundMode::TRUNCWITHOVERFLOW) {
      return success();
    }
    // Need to generate a dummy memref.alloc with size 0, because the op
    // template has a unified interface.
    SmallVector<int64_t> extraBufSizes{0};
    ShapedType srcVecType = cast<ShapedType>(this->getSrc()[0].getType());
    extraBuf = allocExtraBuffer(this->getOperation(), extraBufSizes,
                                srcVecType.getElementType());
  } else {
    // alloc temp buffer which size is three blocks
    SmallVector<int64_t> extraBufSizes{
        util::INTR_BYTES_PER_BLOCK * 3 * 8 /
        dstVecType.getElementType().getIntOrFloatBitWidth()};
    extraBuf = allocExtraBuffer(this->getOperation(), extraBufSizes,
                                dstVecType.getElementType());
  }

  this->getTempBufferMutable().assign(extraBuf);
  return success();
}

//===----------------------------------------------------------------------===//
// VGatherOp
//===----------------------------------------------------------------------===//

LogicalResult VGatherOp::allocExtraBuffersIfPossible() {
  if (this->getTempBuffer()) {
    this->emitWarning("already has extra temp buffer");
    return success();
  }

  // the num of indices determines the size of temp buffer
  ShapedType srcVecType = cast<ShapedType>(this->getIndices().getType());

  std::optional<int64_t> srcAllocTotalSize =
      utils::traceToAllocMaxSize(this->getIndices());
  assert(srcAllocTotalSize);
  SmallVector<int64_t> extraBufSizes{srcAllocTotalSize.value()};
  Value extraBuf = allocExtraBuffer(this->getOperation(), extraBufSizes,
                                    srcVecType.getElementType());
  this->getTempBufferMutable().assign(extraBuf);
  return success();
}

//===----------------------------------------------------------------------===//
// VInterleaveOp
//===----------------------------------------------------------------------===//

LogicalResult VInterleaveOp::allocExtraBuffersIfPossible() {
  if (this->getTempBuffer()) {
    this->emitWarning("already has extra temp buffer");
    return success();
  }
  ShapedType srcVecType = cast<ShapedType>(this->getSrc()[0].getType());
  int64_t numPerBlock = util::INTR_BYTES_PER_BLOCK * 8 /
                        srcVecType.getElementType().getIntOrFloatBitWidth();
  auto shape = srcVecType.getShape();
  int64_t interleaveDimLen = shape[shape.size() - 1];
  int64_t srclastDimAlign =
      util::ceilFactor(interleaveDimLen, util::srcNumPerRepeatOfVBRCBIntrin);
  int64_t dstlastDimAlign = util::ceilFactor(interleaveDimLen * 2, numPerBlock);
  int64_t totalSize = srclastDimAlign * numPerBlock + dstlastDimAlign;
  SmallVector<int64_t> extraBufSizes{totalSize};
  Value extraBuf = allocExtraBuffer(this->getOperation(), extraBufSizes,
                                    srcVecType.getElementType());
  this->getTempBufferMutable().assign(extraBuf);
  return success();
}

//===----------------------------------------------------------------------===//
// VMulExtendedOp
//===----------------------------------------------------------------------===//

LogicalResult VMulextendedOp::allocExtraBuffersIfPossible() {
  if (this->getTempBuffer()) {
    this->emitWarning("already has extra temp buffer");
    return success();
  }

  std::optional<int64_t> srcAllocTotalSize =
      utils::traceToAllocMaxSize(this->getSrc()[0]);
  assert(srcAllocTotalSize);
  int64_t int32NumPerBlock =
      static_cast<int64_t>(util::INTR_BYTES_PER_BLOCK / sizeof(int32_t));
  int64_t dstAlignSize =
      util::ceilFactor(srcAllocTotalSize.value(), int32NumPerBlock);
  int64_t extraSize = 3 * dstAlignSize;
  SmallVector<int64_t> extraBufSizes{extraSize};
  Value extraBuf = allocExtraBuffer(this->getOperation(), extraBufSizes,
                                    IntegerType::get(this->getContext(), 32));
  this->getTempBufferMutable().assign(extraBuf);
  return success();
}

//===----------------------------------------------------------------------===//
// VPowOp
//===----------------------------------------------------------------------===//

LogicalResult VPowOp::allocExtraBuffersIfPossible() {
  if (this->getTempBuffer()) {
    this->emitWarning("already has extra temp buffer");
    return success();
  }

  // only need extra temp buffer for int32
  Type srcElemType = getElementTypeOrSelf(getDpsInputs().front().getType());
  if (!srcElemType.isSignlessInteger(32)) {
    this->emitWarning("only int32 type need extra temp buffer");
    return success();
  }

  ShapedType srcVecType = cast<ShapedType>(this->getSrc()[0].getType());

  int64_t boolBitWidth = 1;
  int64_t int32NumPerBlock =
      static_cast<int64_t>(util::INTR_BYTES_PER_BLOCK / sizeof(int32_t));
  int64_t boolNumPerBlock = util::vectorBlockSizeBit / boolBitWidth;
  int64_t int32BoolDiv = boolNumPerBlock / int32NumPerBlock;
  std::optional<int64_t> srcAllocTotalSize =
      utils::traceToAllocMaxSize(this->getSrc()[0]);
  assert(srcAllocTotalSize);
  int64_t shapeSize = srcAllocTotalSize.value();
  int64_t boolAlignSize =
      util::ceilFactor(shapeSize, boolNumPerBlock) / int32BoolDiv;
  // buf size for reduce_sum
  int64_t int32NumPerRepeat =
      static_cast<int64_t>(util::INTR_BYTES_PER_REPEAT / sizeof(int32_t));
  int64_t reduceAlignSize = int32NumPerRepeat + int32NumPerBlock;
  if (shapeSize <= int32NumPerRepeat) {
    reduceAlignSize = util::ceilFactor(3 * shapeSize / 2, int32NumPerBlock);
  }
  // need 2 f32/i32 type of src shape size buffer, a 1-block size condition_addr
  // buf for vsel , 2 bool type of src shape size buffer view_as f32 and 1
  // buffer for reduce_sum
  int64_t extraSize =
      2 * shapeSize + int32NumPerBlock + 2 * boolAlignSize + reduceAlignSize;
  SmallVector<int64_t> extraBufSizes{extraSize};
  Value extraBuf = allocExtraBuffer(this->getOperation(), extraBufSizes,
                                    srcVecType.getElementType());
  this->getTempBufferMutable().assign(extraBuf);
  return success();
}

//===----------------------------------------------------------------------===//
// VReduceOp
//===----------------------------------------------------------------------===//

LogicalResult VReduceOp::allocExtraBuffersIfPossible() {
  if (this->getTempBuffer()) {
    this->emitWarning("already has extra temp buffer");
    return success();
  }

  MemRefType srcVecType = cast<MemRefType>(this->getSrc().getType());
  auto bufSizeMaybe = hivm::util::getExtraBufferSizeForReduceOp(
      this->getOperation(), util::BufferSizeUnit::ELEMENT);
  if (bufSizeMaybe) {
    // the temporary buffer must have the same rank with src and dst vectors, to
    // satisfy 'HIVMOpSameOperandsAndResultRank' trait
    Value extraBuf = allocExtraBuffer(this->getOperation(), {*bufSizeMaybe},
                                      srcVecType.getElementType());
    this->getTempBufferMutable().assign(extraBuf);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// VSelOp
//===----------------------------------------------------------------------===//

LogicalResult VSelOp::allocExtraBuffersIfPossible() {
  if (this->getTempBuffer()) {
    this->emitWarning("already has extra temp buffer");
    return success();
  }

  MemRefType dstVecType = cast<MemRefType>(this->getDst()[0].getType());
  bool src0ScalarType = getSrc()[1].getType().isIntOrFloat();
  bool src1ScalarType = getSrc()[2].getType().isIntOrFloat();

  auto condType = getElementTypeOrSelf(getSrc()[0].getType());
  auto srcType = getElementTypeOrSelf(getSrc()[1].getType());
  if (srcType.isInteger(64) &&
      (condType.isInteger(1) || src0ScalarType || src1ScalarType))
    return success(); // vsel int64 scalar or int1 condition no buffer needed

  if (!srcType.isInteger(64)) {
    // The number of elements for each stride inside the buffer
    // should be equal to the number of buffers * elements per buffer.
    // So, for generic cases where a single stride i.e., blocks per repeat
    // should be used which is calculated below. If input is 16bit then we
    // can fit 16 values in a stride. Similarly, for 32bit input there can
    // only be 8 values in a stride.
    int resWidth =
        static_cast<int>(dstVecType.getElementType().getIntOrFloatBitWidth());
    int numElemsPerStride = util::INTR_BYTES_PER_REPEAT / resWidth;

    // For base-case just for storing boolean cmp values
    // we can make do with 8 values i.e., 8 values repeat per block
    int buffSize = numElemsPerStride;

    // If first input is scalar then we can do the same thing as above
    // i.e., 8 more values.
    if (src0ScalarType) {
      buffSize += numElemsPerStride;
    }

    // If second input is scalar then we can do the same thing as above
    // i.e., 8 more values.
    if (src1ScalarType) {
      buffSize += numElemsPerStride;
    }
    SmallVector<int64_t> extraBufSizes{buffSize};
    Value extraBuf = allocExtraBuffer(this->getOperation(), extraBufSizes,
                                      dstVecType.getElementType());
    this->getTempBufferMutable().assign(extraBuf);
    return success();
  }

  auto extraSize = getExtraBufferSize();
  assert(extraSize.has_value() && "impossible case for null VSelOp buffsize");

  SmallVector<int64_t> extraBufSizes{extraSize.value()};
  Value extraBuf = allocExtraBuffer(this->getOperation(), extraBufSizes,
                                    dstVecType.getElementType());
  this->getTempBufferMutable().assign(extraBuf);
  return success();
}

//===----------------------------------------------------------------------===//
// VSortOp
//===----------------------------------------------------------------------===//

LogicalResult VSortOp::allocExtraBuffersIfPossible() {
  if (this->getTempBuffer()) {
    this->emitWarning("already has extra temp buffer");
    return success();
  }

  ShapedType srcVecType = cast<ShapedType>(getSrc().getType());

  std::optional<int64_t> buffSize = getExtraBufferSize();
  assert(buffSize.has_value() && "impossible case for null VSortOp buffsize");
  SmallVector<int64_t> extraBufSizes{buffSize.value()};
  Value extraBuf = allocExtraBuffer(this->getOperation(), extraBufSizes,
                                    srcVecType.getElementType());
  this->getTempBufferMutable().assign(extraBuf);
  return success();
}

//===----------------------------------------------------------------------===//
// VTransposeOp
//===----------------------------------------------------------------------===//

LogicalResult VTransposeOp::allocExtraBuffersIfPossible() {
  if (this->getTempBuffer()) {
    this->emitWarning("already has extra temp buffer");
    return success();
  }

  auto srcType = getElementTypeOrSelf(this->getSrc());
  const bool isTransposeWithLastAxis =
      hivm::util::isTransposeWithLastAxis(this->getPermutation());
  if (srcType.isInteger(64) && isTransposeWithLastAxis) {
    // vtranspose op needs extra buffer when data type is int64/uint64,
    // and the number of extra buffer is 2, the size of single extra buffer is
    // 256 elements, so extra buffer needs 512 elements in total.
    constexpr int extraBufNum = 2;
    constexpr int singleExtraBufSize =
        static_cast<int>(util::VNCHWCONV_INTR_BYTES_PER_REPEAT) /
        sizeof(int64_t) * 4;
    SmallVector<int64_t> extraBufSizes{extraBufNum * singleExtraBufSize};
    Value extraBuf =
        allocExtraBuffer(this->getOperation(), extraBufSizes, srcType);
    this->getTempBufferMutable().assign(extraBuf);
    return success();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VXor
//===----------------------------------------------------------------------===//

LogicalResult VXorOp::allocExtraBuffersIfPossible() {
  if (this->getTempBuffer()) {
    this->emitWarning("already has extra temp buffer");
    return success();
  }
  ShapedType srcVecType = cast<ShapedType>(this->getSrc()[0].getType());

  std::optional<int64_t> srcAllocTotalSize = getExtraBufferSize();
  assert(srcAllocTotalSize);

  SmallVector<int64_t> extraBufSizes{srcAllocTotalSize.value()};
  Value extraBuf = allocExtraBuffer(this->getOperation(), extraBufSizes,
                                    srcVecType.getElementType());
  this->getTempBufferMutable().assign(extraBuf);
  return success();
}
