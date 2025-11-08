//===- HIVMAlignAllocSize.cpp ---- Align Alloc Size Pass ------------------===//
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
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/AlignBuffer/Util.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "hivm-align-alloc-size"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_ALIGNALLOCSIZE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

void collectOpAlignInfo(
    Operation *op, SmallVector<int64_t> checkDims,
    llvm::SmallDenseMap<Value, uint32_t> *alignBytes,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  assert(alignBytes != nullptr);
  for (auto oper : op->getOperands()) {
    auto elemTypeBytes = getElementTypeOrSelf(oper).getIntOrFloatBitWidth() / 8;
    auto shape = cast<ShapedType>(oper.getType()).getShape();
    for (auto checkDim : checkDims) {
      assert(checkDim >= 0 && checkDim < static_cast<int64_t>(shape.size()));
      assert((*alignBytes)[oper] != 0);
      if (ShapedType::isDynamic(shape[checkDim]) ||
          (shape[checkDim] * elemTypeBytes) % (*alignBytes)[oper] != 0) {
        auto operAlignInfo = std::make_unique<OperAlignInfo>(
            oper, checkDim, (*alignBytes)[oper]);
        operAlignInfoList->push_back(std::move(operAlignInfo));
      }
    }
  }
}

LogicalResult getUnAlignSizeInfo(
    VTransposeOp op,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  // get alignment bytes
  auto srcType = op.getSrc().getType();
  auto maybeHwAlignBytes = getHWAlignBytes(srcType);
  if (!maybeHwAlignBytes.has_value()) {
    return failure();
  }

  // get transpose loop dims
  SmallVector<int64_t> transposeLoopDims;
  op.getTransposeLoopDims(transposeLoopDims);

  // collect unalign info of all operands if transpose dims are not aligned
  auto hwAlignBytes = maybeHwAlignBytes.value();
  llvm::SmallDenseMap<Value, uint32_t> operHwAlignBytes;
  for (auto oper : op->getOperands()) {
    operHwAlignBytes[oper] = hwAlignBytes;
  }
  collectOpAlignInfo(op.getOperation(), transposeLoopDims, &operHwAlignBytes,
                     operAlignInfoList);

  auto elemTypeBytes = getElementTypeOrSelf(srcType).getIntOrFloatBitWidth() /
                       mlir::utils::INTR_BITS_PER_BYTE;
  const int b32InByte = 4;
  if (elemTypeBytes != b32InByte) {
    return success();
  }

  // when it is B32 type, judge if there is one dim that is already double
  // aligned. if not, should choose one dim to do double alignment, e.g.
  // 8x16xf32 or 16x8xf32.
  auto srcShape = cast<ShapedType>(op.getSrc().getType()).getShape();
  bool isAlreadyDoubleAlign = false;
  for (auto transDim : transposeLoopDims) {
    auto alignedSrcDimBytes =
        CEIL_FACTOR(static_cast<uint64_t>(srcShape[transDim]) * elemTypeBytes,
                    hwAlignBytes);
    if (alignedSrcDimBytes % (hwAlignBytes * 2) == 0) {
      isAlreadyDoubleAlign = true;
    }
  }
  if (isAlreadyDoubleAlign) {
    return success();
  }

  // must choose double align dim from two transpose dims
  if (transposeLoopDims.size() != 2) {
    // For B32, do transpose decompose first
    return failure();
  }

  operAlignInfoList->clear();
  // choose transdim 0 as double align dim
  auto srcTrans0AlignInfo = std::make_unique<OperAlignInfo>(
      op.getSrc(), transposeLoopDims[0], hwAlignBytes);
  operAlignInfoList->push_back(std::move(srcTrans0AlignInfo));
  auto srcTrans1AlignInfo = std::make_unique<OperAlignInfo>(
      op.getSrc(), transposeLoopDims[1], hwAlignBytes * 2);
  operAlignInfoList->push_back(std::move(srcTrans1AlignInfo));

  auto dstTrans0AlignInfo = std::make_unique<OperAlignInfo>(
      op.getDst(), transposeLoopDims[0], hwAlignBytes * 2);
  operAlignInfoList->push_back(std::move(dstTrans0AlignInfo));
  auto dstTrans1AlignInfo = std::make_unique<OperAlignInfo>(
      op.getDst(), transposeLoopDims[1], hwAlignBytes);
  operAlignInfoList->push_back(std::move(dstTrans1AlignInfo));
  return success();
}

LogicalResult getCastSrcUnAlignSizeInfo(
    Value src, SmallVector<int64_t> castAlignDims, int64_t bytesFactor,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  // get alignment bytes
  ShapedType srcType = cast<ShapedType>(src.getType());
  auto maybeHwAlignBytes = getHWAlignBytes(srcType);
  if (!maybeHwAlignBytes.has_value()) {
    return failure();
  }

  // collect unalign info of src if cast src dims are not aligned.
  auto hwAlignBytes = maybeHwAlignBytes.value();
  auto elemTypeBytes = getElementTypeOrSelf(srcType).getIntOrFloatBitWidth() /
                       mlir::utils::INTR_BITS_PER_BYTE;
#ifndef NDEBUG
  const int b16InByte = 2;
  const int b32InByte = 4;
  assert((elemTypeBytes == b16InByte || elemTypeBytes == b32InByte) &&
         "Src only supports b32/b16 in cast overflow.");
#endif
  auto shape = cast<ShapedType>(src.getType()).getShape();
  int64_t numElemPerBlock = mlir::utils::INTR_BYTES_PER_BLOCK / elemTypeBytes;
  int64_t numElemPerBlockForDst = numElemPerBlock * bytesFactor;
  int64_t rank = srcType.getRank();
  // For example (a, b)strides<n1, 1>*i32 cast to (a, b)strides<n2, 1>*i8:
  // 1. (a, b)strides<n1, 1>*i32 view as (a, b*4)strides<n1*4, 1>*i8
  // 2. i8 transpose: Used to separate the high and low bits of int32, make sure
  //    the shape of tranpose is 32*32 aligned.
  // 3. i8 copyubtoub: Take out the lower 8 bits.
  // 4. i8 transpose: Transpose back to get the final cast result, make sure the
  //    shape of tranpose is 32*32 aligned.
  // 5. When n2 is aligned with multiple blocks, need to add another copyubtoub
  //    to adjust it to the target stride.
  if (rank == 1) {
    // The 1D scene is quite special and needs to be converted into a
    // corresponding 2D scene to implement.
    if (!ShapedType::isDynamic(shape[0]) && shape[0] <= numElemPerBlockForDst) {
      hwAlignBytes = static_cast<unsigned>(
          CEIL_FACTOR(shape[0] * bytesFactor, numElemPerBlockForDst) *
          numElemPerBlockForDst);
    } else {
      hwAlignBytes = static_cast<unsigned>(numElemPerBlockForDst *
                                           numElemPerBlockForDst * bytesFactor);
    }
    if (ShapedType::isDynamic(shape[0]) ||
        (shape[0] * elemTypeBytes) % hwAlignBytes != 0) {
      auto srcAlignInfo = std::make_unique<OperAlignInfo>(src, 0, hwAlignBytes);
      operAlignInfoList->push_back(std::move(srcAlignInfo));
    }
  } else {
#ifndef NDEBUG
    const int supportedCastAlignDimSize = 2;
    assert(castAlignDims.size() == supportedCastAlignDimSize &&
           "When cast rank >= 2, castAlignDims size must be equal to 2");
#endif
    // Align the second axis in castAlignDims.
    if (ShapedType::isDynamic(shape[castAlignDims[1]]) ||
        (shape[castAlignDims[1]] * elemTypeBytes) % hwAlignBytes != 0) {
      auto srcAlignInfo =
          std::make_unique<OperAlignInfo>(src, castAlignDims[1], hwAlignBytes);
      operAlignInfoList->push_back(std::move(srcAlignInfo));
    }
    // Align the first axis in castAlignDims.
    hwAlignBytes = static_cast<unsigned>(numElemPerBlockForDst * bytesFactor);
    if (ShapedType::isDynamic(shape[castAlignDims[0]]) ||
        (static_cast<uint64_t>(shape[castAlignDims[0]]) * elemTypeBytes) %
                hwAlignBytes !=
            0) {
      auto srcAlignInfo =
          std::make_unique<OperAlignInfo>(src, castAlignDims[0], hwAlignBytes);
      operAlignInfoList->push_back(std::move(srcAlignInfo));
    }
  }
  return success();
}

LogicalResult getCastDstUnAlignSizeInfo(
    Value dst, SmallVector<int64_t> castAlignDims,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  // get alignment bytes
  ShapedType dstType = cast<ShapedType>(dst.getType());
  auto maybeHwAlignBytes = getHWAlignBytes(dstType);
  if (!maybeHwAlignBytes.has_value()) {
    return failure();
  }

  // collect unalign info of dst if cast dst dims are not aligned
  auto hwAlignBytes = maybeHwAlignBytes.value();
  auto elemTypeBytes = getElementTypeOrSelf(dstType).getIntOrFloatBitWidth() /
                       mlir::utils::INTR_BITS_PER_BYTE;
  assert(elemTypeBytes == 1 && "Dst only supports b8 in cast overflow.");
  auto shape = cast<ShapedType>(dst.getType()).getShape();
  uint64_t numElemPerBlock = mlir::utils::INTR_BYTES_PER_BLOCK / elemTypeBytes;
  int64_t rank = dstType.getRank();
  // For example (a, b)strides<n1, 1>*i32 cast to (a, b)strides<n2, 1>*i8:
  // 1. (a, b)strides<n1, 1>*i32 view as (a, b*4)strides<n1*4, 1>*i8
  // 2. i8 transpose: Used to separate the high and low bits of int32, make sure
  //    the shape of tranpose is 32*32 aligned.
  // 3. i8 copyubtoub: Take out the lower 8 bits.
  // 4. i8 transpose: Transpose back to get the final cast result, make sure the
  //    shape of tranpose is 32*32 aligned.
  // 5. When n2 is aligned with multiple blocks, need to add another copyubtoub
  //    to adjust it to the target stride.
  if (rank == 1) {
    // The 1D scene is quite special and needs to be converted into a
    // corresponding 2D scene to implement.
    hwAlignBytes = numElemPerBlock * numElemPerBlock;
  }
  for (auto checkDim : castAlignDims) {
    if (ShapedType::isDynamic(shape[checkDim]) ||
        (static_cast<uint64_t>(shape[checkDim]) * elemTypeBytes) %
                hwAlignBytes !=
            0) {
      auto dstAlignInfo =
          std::make_unique<OperAlignInfo>(dst, checkDim, hwAlignBytes);
      operAlignInfoList->push_back(std::move(dstAlignInfo));
    }
  }
  return success();
}

LogicalResult getUnAlignSizeInfo(
    VCastOp op,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  auto srcType = cast<ShapedType>(op.getSrc()[0].getType());
  auto dstType = cast<ShapedType>(op.getDst()[0].getType());
  auto srcElemTypeBytes =
      getElementTypeOrSelf(srcType).getIntOrFloatBitWidth() /
      mlir::utils::INTR_BITS_PER_BYTE;
  auto dstElemTypeBytes =
      getElementTypeOrSelf(dstType).getIntOrFloatBitWidth() /
      mlir::utils::INTR_BITS_PER_BYTE;
  auto bytesFactor = srcElemTypeBytes / dstElemTypeBytes;

  // Get the cast axis that needs to be aligned.
  SmallVector<int64_t> castAlignDims;
  int64_t rank = srcType.getRank();
  if (rank == 1) {
    castAlignDims.push_back(0);
  } else if (rank >= 2) {
    castAlignDims.push_back(rank - 2);
    castAlignDims.push_back(rank - 1);
  } else {
    llvm_unreachable("cast op rank need lager than 0.");
  }

  // Get the unalign information of the axis corresponding to cast src.
  if (failed(getCastSrcUnAlignSizeInfo(op.getSrc()[0], castAlignDims,
                                       bytesFactor, operAlignInfoList))) {
    return failure();
  }
  // Get the unalign information of the axis corresponding to cast dst.
  if (failed(getCastDstUnAlignSizeInfo(op.getDst()[0], castAlignDims,
                                       operAlignInfoList))) {
    return failure();
  }
  return success();
}

LogicalResult getUnAlignSizeInfo(
    VSortOp op,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  // get alignment bytes
  ShapedType srcType = cast<ShapedType>(op.getSrc().getType());
  auto maybeHwAlignBytes = getHWAlignBytes(srcType);
  if (!maybeHwAlignBytes.has_value()) {
    return failure();
  }

  // Get the sort axis that needs to be aligned.
  SmallVector<int64_t> sortAlignDims;
  int64_t rank = srcType.getRank();
  sortAlignDims.push_back(rank - 1);

  llvm::SmallDenseMap<Value, uint32_t> operHwAlignBytes;
  for (auto oper : op->getOperands()) {
    ShapedType operType = cast<ShapedType>(oper.getType());
    auto elemTypeBytes =
        getElementTypeOrSelf(operType).getIntOrFloatBitWidth() / 8;
    unsigned int numElemPerBlock =
        mlir::utils::INTR_BYTES_PER_BLOCK / elemTypeBytes;
    operHwAlignBytes[oper] =
        maybeHwAlignBytes.value() * (VBITSORT_NUM_PER_REPEAT / numElemPerBlock);
  }
  collectOpAlignInfo(op.getOperation(), sortAlignDims, &operHwAlignBytes,
                     operAlignInfoList);
  return success();
}

Value createAlignedValue(
    PatternRewriter &rewriter, memref::AllocOp op,
    const SmallVectorImpl<OpFoldResult> &origShape,
    const llvm::SmallVectorImpl<OpFoldResult> &alignedShape) {
  SmallVector<Value> dynSizes;
  SmallVector<int64_t> staticSizes;
  dispatchIndexOpFoldResults(alignedShape, dynSizes, staticSizes);
  auto unAlignType = op.getType();
  MemRefType alignedTy = MemRefType::Builder(unAlignType)
                             .setShape(staticSizes)
                             .setLayout(MemRefLayoutAttrInterface());
  auto alignedAlloc =
      rewriter.create<memref::AllocOp>(op.getLoc(), alignedTy, dynSizes);

  SmallVector<OpFoldResult> offsets(alignedTy.getRank(),
                                    rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(alignedTy.getRank(),
                                    rewriter.getIndexAttr(1));
  MemRefType subviewResTy =
      cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          unAlignType.getShape(), alignedTy, offsets, origShape, strides));

  auto alignedMemRef = rewriter.create<memref::SubViewOp>(
      op.getLoc(), subviewResTy, alignedAlloc, offsets, origShape, strides);
  return alignedMemRef;
}

struct AlignAllocSizePattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    auto alignDimsAttr = op->getAttr(hivm::AllocAlignDimsAttr::name);
    auto alignBytesAttr = op->getAttr(hivm::AllocAlignValueInByteAttr::name);
    if (alignDimsAttr == nullptr || alignBytesAttr == nullptr)
      return failure();

    SmallVector<OpFoldResult> shape = op.getMixedSizes();

    auto alignBytes = cast<DenseI32ArrayAttr>(alignBytesAttr).asArrayRef();
    auto alignDims = cast<DenseI32ArrayAttr>(alignDimsAttr).asArrayRef();
    llvm::SmallVector<uint32_t> alignUnits(shape.size(), 1);
    auto elmentTypeBytes =
        getElementTypeOrSelf(op->getResultTypes()[0]).getIntOrFloatBitWidth() /
        mlir::utils::INTR_BITS_PER_BYTE;
    for (size_t i = 0; i < alignDims.size(); i++) {
      alignUnits[alignDims[i]] =
          (static_cast<uint64_t>(alignBytes[i]) / elmentTypeBytes);
    }
    rewriter.setInsertionPointAfter(op);
    llvm::SmallVector<OpFoldResult> alignedShape(shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
      alignedShape[i] =
          AlignUpOFR(rewriter, op->getLoc(), shape[i], alignUnits[i]);
    }

    if (!isEqualConstantIntOrValueArray(alignedShape, shape)) {
      auto alignedMemRef =
          createAlignedValue(rewriter, op, shape, alignedShape);
      if (failed(replaceAndPropagateMemRefType(rewriter, op.getLoc(), op,
                                               alignedMemRef))) {
        LDBG("Cannot replace with aligned memref " << (Value)alignedMemRef);
        return failure();
      }
      rewriter.eraseOp(op);
      return success();
    } else {
      LDBG("no need to do alignment");
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op->removeAttr(hivm::AllocAlignDimsAttr::name);
      op->removeAttr(hivm::AllocAlignValueInByteAttr::name);
    });
    return success();
  }
};

struct AlignAllocSizePass
    : public impl::AlignAllocSizeBase<AlignAllocSizePass> {
public:
  void runOnOperation() override;
};
} // namespace

void populateAlignAllocAlignPattern(RewritePatternSet &patterns) {
  patterns.add<AlignAllocSizePattern>(patterns.getContext());
}

template <typename HIVMOP>
LogicalResult alignAllocSize(HIVMOP op, OpBuilder &builder) {
  if (!op.hasPureBufferSemantics()) {
    return failure();
  }

  std::vector<std::unique_ptr<OperAlignInfo>> operAlignInfoList;
  if (failed(getUnAlignSizeInfo(op, &operAlignInfoList))) {
    return failure();
  }

  for (auto &it : operAlignInfoList) {
    createAlignMarkOp(builder, op->getLoc(), it->operand, it->alignDims,
                      it->alignBytes, hivm::AllocAlignDimsAttr::name.str(),
                      hivm::AllocAlignValueInByteAttr::name.str());
  }
  return success();
}

LogicalResult markAllocAlign(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  WalkResult result = funcOp->walk([&builder](Operation *op) {
    if (auto transposeOp = dyn_cast<hivm::VTransposeOp>(op)) {
      if (!isLastDimTranspose(transposeOp)) {
        // un-last transpose, no need to do alloc size alignment, just do stride
        // alignment
        return WalkResult::skip();
      }

      if (failed(alignAllocSize(transposeOp, builder))) {
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    } else if (auto castOp = dyn_cast<hivm::VCastOp>(op)) {
      auto srcType = getElementTypeOrSelf(castOp.getSrc()[0]);
      auto dstType = getElementTypeOrSelf(castOp.getDst()[0]);
      const bool isI32ToI8 = srcType.isInteger(32) && dstType.isInteger(8);
      const bool isI16ToI8 = srcType.isInteger(16) && dstType.isInteger(8);
      if (!isI32ToI8 && !isI16ToI8) {
        return WalkResult::skip();
      }

      if (failed(alignAllocSize(castOp, builder))) {
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    } else if (auto sortOp = dyn_cast<hivm::VSortOp>(op)) {
      if (failed(alignAllocSize(sortOp, builder))) {
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  if (result == WalkResult::interrupt()) {
    return failure();
  }
  return success();
}

void AlignAllocSizePass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  // step 1: mark size align info
  if (failed(markAllocAlign(funcOp))) {
    return signalPassFailure();
  }

  LDBG("IR after marking alloc align");
  LDBG(funcOp);

  // step 2: propagate up align info to root memref.alloc
  RewritePatternSet patterns(&getContext());
  populatePropagateAlignUpToRootAllocationPattern(
      patterns, hivm::AllocAlignDimsAttr::name.str(),
      hivm::AllocAlignValueInByteAttr::name.str());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }

  LDBG("IR after propagating up alloc size to root memref.alloc");
  LDBG(funcOp);

  // step 3: modify the alloc and do size alignment
  patterns.clear();
  populateAlignAllocAlignPattern(patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }

  LDBG("IR after modifying the alloc size");
  LDBG(funcOp);
}

std::unique_ptr<Pass> mlir::hivm::createAlignAllocSizePass() {
  return std::make_unique<AlignAllocSizePass>();
}
