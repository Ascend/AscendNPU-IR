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

FailureOr<Value> backtraceStaticAlloc(Value loadDst,
                                      SmallVector<Operation *> &traceOps) {
  SmallVector<Value> worklist = {loadDst};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    LDBG("trace static alloc: " << value);
    MemRefType memrefType = cast<MemRefType>(value.getType());
    auto [strides, offset] = getStridesAndOffset(memrefType);
    if (memrefType.hasStaticShape() &&
        std::is_sorted(strides.rbegin(), strides.rend())) {
      LDBG("trace static alloc succeed");
      return value;
    }
    auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(value.getDefiningOp());
    if (!viewLikeOp)
      break;
    traceOps.push_back(viewLikeOp);
    worklist.push_back(viewLikeOp.getViewSource());
  }
  LDBG("trace static alloc fail");
  return failure();
}

LogicalResult forwardRecoverAlloc(PatternRewriter &rewriter, Location loc,
                                  Value &initAlloc,
                                  SmallVector<Operation *> &traceOps) {
  for (auto op : llvm::reverse(traceOps)) {
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
      SmallVector<OpFoldResult> offsets = subviewOp.getMixedOffsets();
      if (llvm::any_of(offsets, [](const OpFoldResult &ofr) -> bool {
            return !isConstantIntValue(ofr, 0);
          }))
        return failure();
      initAlloc = rewriter.create<memref::SubViewOp>(
          loc, initAlloc, offsets, subviewOp.getMixedSizes(),
          subviewOp.getMixedStrides());
    } else if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(op)) {
      initAlloc = rewriter.create<memref::CollapseShapeOp>(
          loc, collapseOp.getResult().getType(), initAlloc,
          collapseOp.getReassociation());
    } else if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(op)) {
      initAlloc = rewriter.create<memref::ExpandShapeOp>(
          loc, expandOp.getResult().getType(), initAlloc,
          expandOp.getReassociation(), expandOp.getOutputShape(),
          expandOp.getStaticOutputShape());
    } else {
      return rewriter.notifyMatchFailure(
          op, " cannot recognize deinterleave op from the backtrace ops");
    }
    LDBG("recover traced operation: " << initAlloc);
  }
  return success();
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

void generateDeinterleaveOp(PatternRewriter &rewriter, int32_t alignDim,
                            int32_t alignBytes, Value deinterleaveSrc,
                            Value deinterleaveDst, Operation *op) {
  OpBuilder::InsertionGuard guard(rewriter);
  // Generate DeinterleaveOp
  rewriter.setInsertionPointAfter(op);
  hivm::DeinterleaveMode hivmDeinterleaveMode =
      hivm::symbolizeDeinterleaveMode(0).value();
  Type elemType = getElementTypeOrSelf(deinterleaveDst.getType());
  int64_t byteWidth = elemType.getIntOrFloatBitWidth() / 8;
  int64_t channelNum = alignBytes / byteWidth;
  auto deinterleaveOp = rewriter.create<hivm::VDeinterleaveOp>(
      op->getLoc(), op->getResultTypes(), /*src=*/deinterleaveSrc,
      /*dst=*/deinterleaveDst, channelNum, hivmDeinterleaveMode);
  LDBG("generate deinterleaveOp: " << deinterleaveOp);
  rewriter.modifyOpInPlace(op, [&] { op->setOperand(1, deinterleaveSrc); });
  // Generate MarkOp with alignDim and alignBytes attributes
  rewriter.setInsertionPointAfterValue(deinterleaveSrc);
  auto markOp =
      rewriter.create<annotation::MarkOp>(op->getLoc(), deinterleaveSrc);
  rewriter.modifyOpInPlace(markOp, [&]() {
    markOp->setAttr(hivm::StrideAlignDimsAttr::name,
                    DenseI32ArrayAttr::get(markOp.getContext(), {alignDim}));
    markOp->setAttr(hivm::StrideAlignValueInByteAttr::name,
                    DenseI32ArrayAttr::get(markOp.getContext(), {alignBytes}));
  });
  LDBG("generate markOp: " << markOp);
}

struct RecognizeDeinterleaveOpForLoad : public OpRewritePattern<hivm::LoadOp> {
  using OpRewritePattern<hivm::LoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::LoadOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics.");
    }
    // Only recognize deinterleave op
    // when src last dim != 1 and dst last dim == 1
    Value loadSrc = op.getSrc();
    Value loadDst = op.getDst();
    if (!isDeinterleavePattern(loadSrc, loadDst)) {
      return rewriter.notifyMatchFailure(
          op, " only recognize deinterleave op for load op if src last dim not "
              "unit stride and dst last dim unit stride.");
    }

    // Back trace dst to get the closest static memrefType
    SmallVector<Operation *> traceOps;
    auto closestStaticTypedValue = backtraceStaticAlloc(loadDst, traceOps);
    if (failed(closestStaticTypedValue))
      return failure();

    // Configure deinterleaveOp src and dst
    // Deinterleave src is a newly created allocOp
    // Deinterleave dst has the same address as the load dst
    auto loc = op->getLoc();
    Value deinterleaveDst = *closestStaticTypedValue;
    Value deinterleaveSrc = rewriter.create<memref::AllocOp>(
        loc, cast<MemRefType>(closestStaticTypedValue->getType()));

    // Forward recover the deinterleaveOp shape from the op shape
    if (failed(forwardRecoverAlloc(rewriter, loc, deinterleaveDst, traceOps)))
      return failure();
    if (failed(forwardRecoverAlloc(rewriter, loc, deinterleaveSrc, traceOps)))
      return failure();

    // Generate deinterleaveOp and adjust its src strides
    auto dstMemRef = cast<MemRefType>(loadDst.getType());
    auto hwAlignBytes = util::getHWAlignBytes(dstMemRef.getMemorySpace());
    generateDeinterleaveOp(rewriter, dstMemRef.getRank() - 1, hwAlignBytes,
                           deinterleaveSrc, deinterleaveDst, op);
    return success();
  }
};

struct RecognizeDeinterleaveOpPass
    : public impl::HIVMRecognizeDeinterleaveOpBase<
          RecognizeDeinterleaveOpPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<RecognizeDeinterleaveOpForLoad>(ctx);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::hivm::createHIVMRecognizeDeinterleaveOpPass() {
  return std::make_unique<RecognizeDeinterleaveOpPass>();
}
