//===- RecognizeDiscontiguousStore.cpp ---------------------*- C++ -*-===//
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMRECOGNIZEDISCONTINUOUSSTORE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-recognize-discontinuous-store"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
// Element channel count fitting into alignBytes (32B block).
static int64_t computeChannelNum(MemRefType type, int64_t alignBytes) {
  return alignBytes / type.getElementType().getIntOrFloatBitWidth() * 8;
}

// Reassociation splitting/merging the last dim: [{0},...,{R-2},{R-1,R}].
static SmallVector<ReassociationIndices>
buildSplitLastDimReassociation(int64_t rank) {
  SmallVector<ReassociationIndices> reassoc;
  for (int64_t i = 0; i < rank - 1; ++i)
    reassoc.push_back({static_cast<int64_t>(i)});
  reassoc.push_back({rank - 1, rank});
  return reassoc;
}

// Static per-dim upper bounds of src traced back to its defining alloc.
// Returns alloc shape for static allocs of the same rank; for rank-1 src
// falls back to total element count (preserved across rank-changing reshapes).
static std::optional<SmallVector<int64_t>>
getStaticAllocUpperBounds(Value src, int64_t rank) {
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy)
    return std::nullopt;

  // Fast path: view-like chain to memref::AllocOp. collapse/expand traversed
  // but rejected by the rank check below.
  if (auto maybeAlloc = traceDefOp<memref::AllocOp>(src)) {
    auto allocTy = dyn_cast<MemRefType>((*maybeAlloc)->getResult(0).getType());
    if (allocTy && allocTy.hasStaticShape() && allocTy.getRank() == rank &&
        allocTy.getElementType() == srcTy.getElementType()) {
      return SmallVector<int64_t>(allocTy.getShape().begin(),
                                  allocTy.getShape().end());
    }
  }

  // Fallback for rank-1: total element count as the single dim bound.
  if (rank == 1) {
    auto totalMaybe = utils::traceToAllocMaxSize(src);
    if (totalMaybe && *totalMaybe > 0)
      return SmallVector<int64_t>{*totalMaybe};
  }
  return std::nullopt;
}

// Build a MemRefType that uses the default contiguous layout when `strides`
// match the row-major layout derived from `shape`, and otherwise attaches an
// explicit StridedLayoutAttr. This keeps alloc types canonical (and avoids
// redundant layout attributes) when the computed strides are contiguous.
static MemRefType buildContiguousOrStridedType(ArrayRef<int64_t> shape,
                                               ArrayRef<int64_t> strides,
                                               Type elemType,
                                               Attribute memSpace,
                                               MLIRContext *ctx) {
  assert(shape.size() == strides.size() && "shape/stride rank mismatch");
  auto isDefaultRowMajor = [&] {
    int64_t expected = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      if (strides[i] != expected)
        return false;
      expected *= shape[i];
    }
    return true;
  }();
  if (isDefaultRowMajor)
    return MemRefType::get(shape, elemType, MemRefLayoutAttrInterface{}, memSpace);
  return MemRefType::get(shape, elemType,
                         StridedLayoutAttr::get(ctx, /*offset=*/0, strides),
                         memSpace);
}

// Rewrites UB->GM store (last-dim continuous -> discontinuous) by building a
// 32B-aligned view of src: expand_shape + vbrc + subview + collapse, so the
// store reads src with last-dim stride = channelNum.
//   %src(..,stride=1) -> %dst(..,stride>1)
//   expand_shape %src -> (..,lastDim,1)
//   vbrc %expand -> %alloc(..,lastDim,channelNum) brc_dims=[R]
//   subview[..,0][..,1] -> (..,lastDim,1) stride=(..,channelNum,1)
//   collapse[[..],[R-1,R]] -> (..,lastDim) stride=(..,channelNum)
//   store %collapsed -> %dst
struct RecognizeDisContinuousStore : public OpRewritePattern<hivm::StoreOp> {
  using OpRewritePattern<hivm::StoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::StoreOp op,
                                PatternRewriter &rewriter) const override {
    Value src;
    MemRefType srcTy;
    int64_t rank = 0;
    int64_t srcOffset = 0;
    SmallVector<int64_t> srcStrides;
    if (failed(checkPreconditions(op, rewriter, src, srcTy, rank, srcOffset,
                                  srcStrides)))
      return failure();
    Location loc = op.getLoc();
    int64_t channelNum = computeChannelNum(srcTy, 32);
    Type elemType = srcTy.getElementType();
    Attribute memSpace = srcTy.getMemorySpace();
    MLIRContext *ctx = rewriter.getContext();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto getDimSize = [&](int64_t i) -> OpFoldResult {
      int64_t s = srcTy.getDimSize(i);
      if (!ShapedType::isDynamic(s))
        return rewriter.getIndexAttr(s);
      return rewriter.create<memref::DimOp>(loc, src, i).getResult();
    };

    // expand_shape: src(R) -> R+1, append a size-1 dim at the end as brc axis.
    auto expandReassociation = buildSplitLastDimReassociation(rank);
    SmallVector<int64_t> expandShape(srcTy.getShape().begin(),
                                     srcTy.getShape().end());
    expandShape.push_back(1);
    SmallVector<int64_t> expandStrides(srcStrides.begin(), srcStrides.end());
    expandStrides.push_back(1);
    auto expandTy = memref::ExpandShapeOp::computeExpandedType(
        srcTy, expandShape, expandReassociation);
    if (failed(expandTy))
      return rewriter.notifyMatchFailure(op, "failed to compute expand type");
    auto expandOp = rewriter.create<memref::ExpandShapeOp>(loc, *expandTy, src,
                                                           expandReassociation);

    // Static sizing decision:
    //   - src static: use src dims, no narrowing subview.
    //   - src dynamic: must trace to static alloc (upper bounds + narrow), else
    //   fail.
    bool srcIsStatic = srcTy.hasStaticShape();
    std::optional<SmallVector<int64_t>> staticUbs;
    if (!srcIsStatic) {
      staticUbs = getStaticAllocUpperBounds(src, rank);
      if (!staticUbs)
        return rewriter.notifyMatchFailure(
            op, "dynamic src must trace to a static alloc");
    }
    bool needNarrowSubview = !srcIsStatic;

    // alloc: aligned buffer (.., lastDim, channelNum).
    auto dimUb = [&](int64_t i) {
      return staticUbs ? (*staticUbs)[i] : srcTy.getDimSize(i);
    };
    SmallVector<int64_t> allocShape, allocStrides;
    for (int64_t i = 0; i < rank; ++i) {
      allocShape.push_back(dimUb(i));
      allocStrides.push_back(srcStrides[i] * channelNum);
    }
    allocShape.push_back(channelNum);
    allocStrides.push_back(1);
    auto allocTy = buildContiguousOrStridedType(allocShape, allocStrides,
                                                elemType, memSpace, ctx);
    auto alignedAlloc = rewriter.create<memref::AllocOp>(loc, allocTy);

    // Narrow alloc to actual (dynamic) sizes so vbrc's non-broadcast dims match
    // the expand shape.
    Value vbrcDst = alignedAlloc.getResult();
    if (needNarrowSubview) {
      SmallVector<OpFoldResult> narrowOffsets(rank + 1,
                                              rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> narrowStrides(rank + 1,
                                              rewriter.getIndexAttr(1));
      SmallVector<OpFoldResult> narrowSizes;
      for (int64_t i = 0; i < rank; ++i)
        narrowSizes.push_back(getDimSize(i));
      narrowSizes.push_back(rewriter.getIndexAttr(channelNum));
      SmallVector<int64_t> narrowShape(srcTy.getShape().begin(),
                                       srcTy.getShape().end());
      narrowShape.push_back(channelNum);
      SmallVector<int64_t> zeros(rank + 1, 0), ones(rank + 1, 1);
      auto narrowSubviewTy =
          cast<MemRefType>(memref::SubViewOp::inferResultType(
              allocTy, zeros, narrowShape, ones));
      vbrcDst = rewriter
                    .create<memref::SubViewOp>(
                        loc, narrowSubviewTy, alignedAlloc.getResult(),
                        narrowOffsets, narrowSizes, narrowStrides)
                    .getResult();
    }

    // vbrc: broadcast expand's last dim(1) to channelNum.
    auto brcDimsAttr = rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{rank});
    rewriter.create<hivm::VBrcOp>(loc, TypeRange(), expandOp.getResult(),
                                  vbrcDst, brcDimsAttr);

    // subview: slice the broadcast dim to size 1.
    SmallVector<OpFoldResult> subviewOffsets(rank + 1,
                                             rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> subviewStrides(rank + 1,
                                             rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> subviewSizes;
    for (int64_t i = 0; i < rank; ++i)
      subviewSizes.push_back(getDimSize(i));
    subviewSizes.push_back(rewriter.getIndexAttr(1));
    auto subviewTy = cast<MemRefType>(memref::SubViewOp::inferResultType(
        allocTy, subviewOffsets, subviewSizes, subviewStrides));
    auto subviewOp = rewriter.create<memref::SubViewOp>(
        loc, subviewTy, vbrcDst, subviewOffsets, subviewSizes, subviewStrides);

    // collapse: merge last two dims, stride = channelNum * 1 = channelNum.
    auto reassociation = buildSplitLastDimReassociation(rank);
    auto collapseTy = memref::CollapseShapeOp::computeCollapsedType(
        subviewOp.getType(), reassociation);
    auto collapseOp = rewriter.create<memref::CollapseShapeOp>(
        loc, collapseTy, subviewOp.getResult(), reassociation);
    rewriter.modifyOpInPlace(
        op, [&] { op->setOperand(0, collapseOp.getResult()); });
    return success();
  }

private:
  // Validates semantics, address spaces, rank, and stride constraints.
  // On success fills the output parameters with src/type/stride invariants.
  static LogicalResult checkPreconditions(hivm::StoreOp op,
                                          PatternRewriter &rewriter, Value &src,
                                          MemRefType &srcTy, int64_t &rank,
                                          int64_t &srcOffset,
                                          SmallVector<int64_t> &srcStrides);
};

// Validates buffer semantics, address spaces (src=UB, dst=GM), rank equality,
// and stride constraints (src last stride=1, dst last stride>1 static).
// On success fills the output parameters with invariants derived from src.
LogicalResult RecognizeDisContinuousStore::checkPreconditions(
    hivm::StoreOp op, PatternRewriter &rewriter, Value &src, MemRefType &srcTy,
    int64_t &rank, int64_t &srcOffset, SmallVector<int64_t> &srcStrides) {
  if (!op.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(op, " op should have buffer semantics.");

  src = op.getSrc();
  Value dst = op.getDst();

  srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !dstTy)
    return rewriter.notifyMatchFailure(op, "only support memref type.");

  auto srcSpace = dyn_cast<hivm::AddressSpaceAttr>(srcTy.getMemorySpace());
  if (!srcSpace || srcSpace.getAddressSpace() != hivm::AddressSpace::UB)
    return rewriter.notifyMatchFailure(op, "src should be in UB.");
  auto dstSpace = dyn_cast<hivm::AddressSpaceAttr>(dstTy.getMemorySpace());
  if (!dstSpace || dstSpace.getAddressSpace() != hivm::AddressSpace::GM)
    return rewriter.notifyMatchFailure(op, "dst should be in GM.");

  rank = srcTy.getRank();
  if (rank == 0 || dstTy.getRank() != rank)
    return rewriter.notifyMatchFailure(op,
                                       "src/dst rank should be equal and > 0.");

  if (failed(getStridesAndOffset(srcTy, srcStrides, srcOffset)) ||
      llvm::any_of(srcStrides,
                   [](int64_t s) { return ShapedType::isDynamic(s); }) ||
      srcStrides[rank - 1] != 1)
    return rewriter.notifyMatchFailure(
        op, "src strides should be static and last stride = 1.");

  int64_t dstOffset;
  SmallVector<int64_t> dstStrides;
  if (failed(getStridesAndOffset(dstTy, dstStrides, dstOffset)) ||
      ShapedType::isDynamic(dstStrides[rank - 1]) || dstStrides[rank - 1] <= 1)
    return rewriter.notifyMatchFailure(
        op, "dst last stride should be static and > 1.");

  return success();
}

struct RecognizeDisContinuousStorePass
    : public impl::HIVMRecognizeDisContinuousStoreBase<
          RecognizeDisContinuousStorePass> {
  void runOnOperation() override;
};
} // namespace

void RecognizeDisContinuousStorePass::runOnOperation() {
  auto funcOp = getOperation();
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  patterns.add<RecognizeDisContinuousStore>(ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createHIVMRecognizeDisContinuousStorePass() {
  return std::make_unique<RecognizeDisContinuousStorePass>();
}