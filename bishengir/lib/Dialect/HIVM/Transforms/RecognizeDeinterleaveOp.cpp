//===- RecognizeDeinterleaveOp.cpp -----------------------------*- C++ -*-===//
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
//===---------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMRECOGNIZEDEINTERLEAVEOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-recognize-deinterleave-op"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

Value traceStaticAlloc(Value value, SmallVector<OpFoldResult> &offsets,
                       SmallVector<OpFoldResult> &sizes,
                       SmallVector<OpFoldResult> &strides) {
  // Currently, only support single subview and static alloc
  Operation *op = value.getDefiningOp();
  if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
    offsets = subviewOp.getMixedOffsets();
    sizes = subviewOp.getMixedSizes();
    strides = subviewOp.getMixedStrides();

    Value viewSrc = subviewOp.getViewSource();
    Operation *viewSrcOp = viewSrc.getDefiningOp();
    if (isa<memref::SubViewOp>(viewSrcOp) || isa<memref::ViewOp>(viewSrcOp)) {
      // TODO: can there be multiple subviews/view from alloc?
      return Value();
    }
    return traceStaticAlloc(subviewOp.getViewSource(), offsets, sizes, strides);
  }
  if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
    auto allocMemRef = cast<MemRefType>(allocOp.getType());
    return allocMemRef.hasStaticShape() ? allocOp.getMemref() : nullptr;
  }
  return Value();
}

int64_t computeChannelNum(Type type, int64_t alignBytes) {
  Type elemType = getElementTypeOrSelf(type);
  int64_t byteWidth = elemType.getIntOrFloatBitWidth() / 8;
  return alignBytes / byteWidth;
}

// check if last dim is effectively marked to do stride align
bool isLastStrideMarkedAlign(Value value) {
  auto markMaybe =
      utils::getAnnotateOpWithAttr(value, StrideAlignDimsAttr::name);
  if (!markMaybe.has_value()) {
    // no stride align annotation mark
    return false;
  }
  auto markOp = cast<annotation::MarkOp>(markMaybe.value());

  auto alignDims =
      markOp->getAttrOfType<DenseI32ArrayAttr>(hivm::StrideAlignDimsAttr::name);
  auto alignBytes = markOp->getAttrOfType<DenseI32ArrayAttr>(
      hivm::StrideAlignValueInByteAttr::name);

  if (alignDims == nullptr || alignBytes == nullptr || alignDims.empty() ||
      alignBytes.empty()) {
    // no stride align if no effective align dims and bytes
    return false;
  }

  if (alignDims.size() != alignBytes.size()) {
    // not valid dims to align
    return false;
  }

  // find align bytes for last dim if exists
  ShapedType shapeType = cast<ShapedType>(value.getType());
  bool alignLastDim = false;
  int32_t lastDimAlignBytes = -1;
  for (auto alignPair :
       llvm::zip(alignDims.asArrayRef(), alignBytes.asArrayRef())) {
    if (std::get<0>(alignPair) == shapeType.getRank() - 1) {
      alignLastDim = true;
      lastDimAlignBytes = std::get<1>(alignPair);
    }
  }

  if (!alignLastDim) {
    // last dim is not marked to do align
    return false;
  }
  // last dim is effectively marked aligned only if align bytes is not one
  return lastDimAlignBytes != 1;
}

bool isLastDimContinuous(Value value) {
  MemRefType memref = cast<MemRefType>(value.getType());
  return isLastMemrefDimUnitStride(memref) && !isLastStrideMarkedAlign(value);
}

bool isLastDimUnContinuous(Value value) {
  MemRefType memref = cast<MemRefType>(value.getType());
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(getStridesAndOffset(memref, strides, offset))) {
    // not sure about uncontinuous if failed to get strides
    return false;
  }
  int64_t rank = memref.getRank();
  if (rank == 0) {
    // no stride info for zero-rank memref
    return false;
  }
  if (ShapedType::isDynamic(strides[rank - 1])) {
    // if last stride is dynamic, not sure if uncontinuous
    return false;
  } else {
    // if last stride is static, infer if uncontinuous
    return !isLastDimContinuous(value);
  }
}

bool isDeinterleavePattern(Value src, Value dst) {
  MemRefType srcMemRef = cast<MemRefType>(src.getType());
  auto srcSpace = dyn_cast<hivm::AddressSpaceAttr>(srcMemRef.getMemorySpace());
  if (srcSpace && (srcSpace.getAddressSpace() != hivm::AddressSpace::GM)) {
    // only support deinterleave for gm src
    return false;
  }

  MemRefType dstMemRef = cast<MemRefType>(dst.getType());
  auto dstSpace = dyn_cast<hivm::AddressSpaceAttr>(dstMemRef.getMemorySpace());
  if (dstSpace && (dstSpace.getAddressSpace() != hivm::AddressSpace::UB)) {
    // only support deinterleave for ub dst
    return false;
  }

  Type elemType = getElementTypeOrSelf(dstMemRef);
  int64_t rank = dstMemRef.getRank();
  if (elemType.isInteger(64) || rank >= 3) {
    // TODO: unsupport i64 type deinterleave and 3d deinterleave
    return false;
  }

  // ensure: src must be uncontinuous and dst must be continuous
  return isLastDimUnContinuous(src) && isLastDimContinuous(dst);
}

void markEnableStrideAlign(Value value, int32_t alignDim, int32_t alignBytes,
                           Location loc, PatternRewriter &rewriter) {
  rewriter.setInsertionPointAfterValue(value);
  auto markOp = rewriter.create<annotation::MarkOp>(loc, value);
  rewriter.modifyOpInPlace(markOp, [&]() {
    markOp->setAttr(hivm::StrideAlignDimsAttr::name,
                    DenseI32ArrayAttr::get(markOp.getContext(), {alignDim}));
    markOp->setAttr(hivm::StrideAlignValueInByteAttr::name,
                    DenseI32ArrayAttr::get(markOp.getContext(), {alignBytes}));
  });
}

namespace {
bool isAllZero(const SmallVector<OpFoldResult> &values) {
  return llvm::all_of(values, [](const OpFoldResult &ofr) {
    return isConstantIntValue(ofr, 0);
  });
}

void adaptToDeinterleaveOp(PatternRewriter &rewriter, Value deinterleaveSrc,
                           Value deinterleaveDst, Operation *op) {
  // use deinterleave op to adapt new subview to old subview
  rewriter.setInsertionPointAfter(op);
  hivm::DeinterleaveMode hivmDeinterleaveMode =
      hivm::symbolizeDeinterleaveMode(0).value();
  int64_t channelNum = computeChannelNum(deinterleaveDst.getType(), 32);
  rewriter.create<hivm::VDeinterleaveOp>(
      op->getLoc(), op->getResultTypes(), /*src=*/deinterleaveSrc,
      /*dst=*/deinterleaveDst, channelNum, hivmDeinterleaveMode);

  // adjust dst subview of old load op
  rewriter.modifyOpInPlace(op, [&] { op->setOperand(1, deinterleaveSrc); });
}

struct RecognizeDeinterleaveOpForLoad : public OpRewritePattern<hivm::LoadOp> {
  using OpRewritePattern<hivm::LoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::LoadOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics.");
    }

    Value src = op.getSrc();
    Value dst = op.getDst();
    if (!isDeinterleavePattern(src, dst)) {
      return rewriter.notifyMatchFailure(
          op, " only recognize deinterleave op for load op if src last dim not "
              "unit stride and dst last dim unit stride.");
    }

    // trace dst subview to static alloc site
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    Value allocForDst = traceStaticAlloc(dst, offsets, sizes, strides);
    if (!allocForDst) {
      return rewriter.notifyMatchFailure(
          op, " only recognize deinterleave op for load op with dst traced "
              "from static alloc size.");
    }

    MemRefType allocMemRef = cast<MemRefType>(allocForDst.getType());
    int64_t rank = allocMemRef.getRank();
    auto dstMemRef = cast<MemRefType>(dst.getType());
    if (allocMemRef.getRank() != dstMemRef.getRank()) {
      return rewriter.notifyMatchFailure(
          op, " cannot recognize deinterleave op for alloc with rank-reducing "
              "subview.");
    }

    if (!isAllZero(offsets)) {
      return rewriter.notifyMatchFailure(
          op, " only recognize deinterleave op with zero offset alloc.");
    }

    Location loc = op->getLoc();
    // reuse old alloc and subview from it as deinterleave dst
    Value deinterleaveDst = allocForDst;
    bool isLoadToSubview = isa<memref::SubViewOp>(dst.getDefiningOp());
    if (isLoadToSubview) {
      deinterleaveDst = rewriter.create<memref::SubViewOp>(
          loc, allocForDst, offsets, sizes, strides);
    }

    // create new alloc and subview from it as load dst and deinterleave src
    Value newAlloc = rewriter.create<memref::AllocOp>(loc, allocMemRef);
    Value deinterleaveSrc = newAlloc;
    if (isLoadToSubview) {
      deinterleaveSrc = rewriter.create<memref::SubViewOp>(
          loc, newAlloc, offsets, sizes, strides);
    }

    adaptToDeinterleaveOp(rewriter, deinterleaveSrc, deinterleaveDst, op);

    // adjust strides to hwAlignBytes for deinterleave to work
    auto hwAlignBytes = util::getHWAlignBytes(dstMemRef.getMemorySpace());
    markEnableStrideAlign(deinterleaveSrc, rank - 1, hwAlignBytes, loc,
                          rewriter);
    return success();
  }
};

struct RecognizeDeinterleaveOpForCopy : public OpRewritePattern<hivm::CopyOp> {
  using OpRewritePattern<hivm::CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::CopyOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics.");
    }

    Value src = op.getSrc();
    Value dst = op.getDst();
    if (!isDeinterleavePattern(src, dst)) {
      return rewriter.notifyMatchFailure(
          op, " only recognize deinterleave op for copy op if src last dim not "
              "unit stride and dst lowest dim is of size one");
    }

    // trace dst subview to static alloc site
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    Value allocForDst = traceStaticAlloc(dst, offsets, sizes, strides);
    if (!allocForDst) {
      return rewriter.notifyMatchFailure(
          op, " only recognize deinterleave op for copy op with dst traced "
              "from static alloc size.");
    }

    MemRefType allocMemRef = cast<MemRefType>(allocForDst.getType());
    int64_t rank = allocMemRef.getRank();
    auto dstMemRef = cast<MemRefType>(dst.getType());
    if (allocMemRef.getRank() != dstMemRef.getRank()) {
      return rewriter.notifyMatchFailure(
          op, " cannot recognize deinterleave op for alloc with rank-reducing "
              "subview.");
    }

    if (!isAllZero(offsets)) {
      return rewriter.notifyMatchFailure(
          op, " only recognize deinterleave op with zero offset alloc.");
    }

    Location loc = op->getLoc();
    // reuse old alloc and subview from it as deinterleave dst
    Value deinterleaveDst = allocForDst;
    bool isCopyToSubview = isa<memref::SubViewOp>(dst.getDefiningOp());
    if (isCopyToSubview) {
      deinterleaveDst = rewriter.create<memref::SubViewOp>(
          loc, allocForDst, offsets, sizes, strides);
    }

    // create new alloc and subview from it as copy dst and deinterleave src
    Value newAlloc = rewriter.create<memref::AllocOp>(loc, allocMemRef);
    Value deinterleaveSrc = newAlloc;
    if (isCopyToSubview) {
      deinterleaveSrc = rewriter.create<memref::SubViewOp>(
          loc, newAlloc, offsets, sizes, strides);
    }

    adaptToDeinterleaveOp(rewriter, deinterleaveSrc, deinterleaveDst, op);

    // adjust strides to hwAlignBytes for deinterleave to work
    auto hwAlignBytes = util::getHWAlignBytes(dstMemRef.getMemorySpace());
    markEnableStrideAlign(deinterleaveSrc, rank - 1, hwAlignBytes, op->getLoc(),
                          rewriter);
    return success();
  }
};

struct RecognizeDeinterleaveOpPass
    : public impl::HIVMRecognizeDeinterleaveOpBase<
          RecognizeDeinterleaveOpPass> {
  void runOnOperation() override;
};
} // namespace

void RecognizeDeinterleaveOpPass::runOnOperation() {
  auto funcOp = getOperation();
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  patterns.add<RecognizeDeinterleaveOpForLoad>(ctx);
  patterns.add<RecognizeDeinterleaveOpForCopy>(ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createHIVMRecognizeDeinterleaveOpPass() {
  return std::make_unique<RecognizeDeinterleaveOpPass>();
}
