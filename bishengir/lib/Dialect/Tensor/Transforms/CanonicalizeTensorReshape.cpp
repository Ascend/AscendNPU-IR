//===--- CanonicalizeTensorReshape.cpp -  canonicalize tensor reshape------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025~2026. All rights reserved.
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
// This file implements a pass to canonicalize tensor reshape operations.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
#define GEN_PASS_DEF_CANONICALIZETENSORRESHAPE
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "canonicalize-tensor-reshape"

using namespace mlir;
using namespace mlir::tensor;

namespace mlir::tensor {
namespace {
struct CanonicalizeTensorReshape
    : public impl::CanonicalizeTensorReshapeBase<CanonicalizeTensorReshape> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace
static LogicalResult
shapeOpToCollapseShapeOpRewriteHelper(tensor::ReshapeOp &op,
                                      PatternRewriter &rewriter) {
  RankedTensorType srcType =
      llvm::dyn_cast<RankedTensorType>(op.getSource().getType());
  std::optional<int64_t> resultTotalSize =
      mlir::utils::getStaticTotalSize(srcType.getShape());
  if (!resultTotalSize) {
    return failure();
  }
  // Generate new RankedTensorType resultType.
  ShapedType srcShapedType =
      llvm::dyn_cast<ShapedType>(op.getSource().getType());
  RankedTensorType resultType = RankedTensorType::get(
      ArrayRef({resultTotalSize.value()}), srcShapedType.getElementType());

  // Generate new CollapseShapeOp collapsedResultIndices.
  SmallVector<ReassociationIndices> collapseIndices = {
      (llvm::to_vector<2>(llvm::seq<int64_t>(0, srcShapedType.getRank())))};

  auto collapseShapeOp = rewriter.create<tensor::CollapseShapeOp>(
      op.getLoc(), resultType, op.getSource(), collapseIndices);
  rewriter.replaceOp(op, collapseShapeOp);
  return success();
}

static LogicalResult shapeOpToExpandOpRewriteHelper(tensor::ReshapeOp &op,
                                                    PatternRewriter &rewriter) {
  ShapedType dstShapedType =
      llvm::dyn_cast<ShapedType>(op.getResult().getType());
  SmallVector<ReassociationIndices> expandIndices = {
      (llvm::to_vector<2>(llvm::seq<int64_t>(0, dstShapedType.getRank())))};

  Value expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
      op.getLoc(), llvm::dyn_cast<RankedTensorType>(op.getResult().getType()),
      op.getSource(), expandIndices);
  rewriter.replaceOp(op, expandShapeOp);

  return success();
}

struct CanonicalizeTensorReshapeOpPattern
    : public OpRewritePattern<tensor::ReshapeOp> {
public:
  using OpRewritePattern<tensor::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType shapeType =
        llvm::dyn_cast<RankedTensorType>(op.getShape().getType());
    RankedTensorType srcType =
        llvm::dyn_cast<RankedTensorType>(op.getSource().getType());
    RankedTensorType dstType =
        llvm::dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!srcType || !dstType || !shapeType) {
      return failure();
    }
    if (!shapeType.hasStaticShape()) {
      return failure();
    }
    if (srcType.getShape().size() == 1) {
      return shapeOpToExpandOpRewriteHelper(op, rewriter);
    } else if (shapeType.getShape().size() == 1 &&
               shapeType.getShape()[0] == 1) {
      return shapeOpToCollapseShapeOpRewriteHelper(op, rewriter);
    }
    return failure();
  }
};

class FoldReshape : public OpRewritePattern<tensor::ReshapeOp> {
public:
  using OpRewritePattern<tensor::ReshapeOp>::OpRewritePattern;

  class RankedTensorWithInfo : public RankedTensorType {
  public:
    explicit RankedTensorWithInfo(RankedTensorType rankedTensor)
        : RankedTensorType(rankedTensor) {}

    int64_t getStaticTotalMult() const {
      int64_t staticTotalMult = 1;
      for (auto el : this->getShape()) {
        if (!ShapedType::isDynamic(el)) {
          staticTotalMult *= el;
        }
      }
      return staticTotalMult;
    }
  };

  // Asserts if dynamic are the same for now
  bool isInferrable(RankedTensorWithInfo typeA,
                    RankedTensorWithInfo typeB) const {
    int dynCountA = typeA.getNumDynamicDims();
    int dynCountB = typeB.getNumDynamicDims();
    if (dynCountA > 1 || dynCountB > 1)
      return false;
    if (dynCountA != dynCountB)
      return false;
    // Only support 1 to 1 and/or 0 to 0 for now, other case should be derivable
    // by logic!
    // Without divisibility API, parting can only assume the value is the same
    return (typeA.getStaticTotalMult() == typeB.getStaticTotalMult());
  }

  LogicalResult matchAndRewrite(tensor::ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    Value src = reshapeOp.getSource();
    Value dst = reshapeOp.getResult();

    RankedTensorWithInfo srcType(dyn_cast<RankedTensorType>(src.getType()));
    RankedTensorWithInfo dstType(dyn_cast<RankedTensorType>(dst.getType()));
    if (!srcType || !dstType) {
      return failure();
    }
    // Check if the dynamic has the same value
    if (!isInferrable(srcType, dstType)) {
      return rewriter.notifyMatchFailure(
          reshapeOp, "Dynamic requirement is not satisfied");
    }
    SmallVector<ReassociationIndices> newReassociationExpand,
        newReassociationCollapse;
    SmallVector<int64_t> srcShape(srcType.getShape());
    SmallVector<int64_t> dstShape(dstType.getShape());
    SmallVector<int64_t> newExpandShape;
    bool compatibleReassociation = areLooseReassociationsCompatible(
        newReassociationExpand, newReassociationCollapse, srcShape, dstShape,
        newExpandShape);
    if (!compatibleReassociation)
      return failure();

    utils::renumberReassociation(newReassociationExpand);
    utils::renumberReassociation(newReassociationCollapse);
    rewriter.setInsertionPointAfter(reshapeOp);
    auto newExpandType =
        RankedTensorType::get(newExpandShape, getElementTypeOrSelf(dstType));
    auto newExpandOp = rewriter.create<tensor::ExpandShapeOp>(
        reshapeOp.getLoc(), newExpandType, src, newReassociationExpand);
    auto newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
        reshapeOp.getLoc(), dstType, newExpandOp.getResult(),
        newReassociationCollapse);
    rewriter.replaceAllUsesWith(reshapeOp, newCollapseOp.getResult());
    return success();
  }
};

/// Fold an nD memref load followed by a pure collapse to rank-3 into a direct
/// 3D load:
///   reinterpret_cast(nD) -> alloc(nD) <- copy <- to_tensor -> reshape/collapse
/// becomes:
///   reinterpret_cast(3D) -> alloc(3D) <- copy <- to_tensor[+cast]
///
/// Motivation: Triton leading-unit shapes (e.g. 1x1xMxK -> 1xMxK) otherwise
/// keep a rank>=4 buffer through InferHIVM / TileBMM. Folding early reuses the
/// existing 3D batch-matmul layout path without changing those passes.
///
/// Equivalence (when guards below hold):
///   copy(strided nD -> contiguous nD) ; tensor-collapse to 3D
/// == reinterpret nD as contiguous-group 3D view ; copy -> contiguous 3D
///
/// Safety guards (bugs caught in review — do not relax casually):
/// - Sole use of to_tensor: memref.copy has no results, so use_empty(copy) is
///   always true; erasing the chain while other to_tensor users remain would
///   leave them reading an uninitialized alloc.
/// - Insert each replacement at the *corresponding* original op (not all at
///   alloc): preserves dominance if alloc precedes reinterpret_cast, and keeps
///   the copy's program point relative to intervening side effects.
/// - Contiguous stride groups only (overflow-safe); respect collapse_shape's
///   own reassociation (reshape infers via getReassociationIndicesForCollapse).
/// - Preserve element type, memref memory spaces, alloc alignment; to_tensor
///   result must be the memref's tensor equivalent (no encoding) — reapply
///   dst encoding with tensor.cast when needed.
///
/// Scope limits: only memref.copy loads (not hivm.load); only static dst rank
/// 3; store-side 3D->nD expand is intentionally unchanged.
static FailureOr<SmallVector<int64_t>>
collapseContiguousStrides(ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides,
                          ArrayRef<ReassociationIndices> reassoc) {
  SmallVector<int64_t> newStrides;
  newStrides.reserve(reassoc.size());
  for (const ReassociationIndices &group : reassoc) {
    if (group.empty())
      return failure();
    for (size_t i = 0, e = group.size(); i + 1 < e; ++i) {
      int64_t d = group[i];
      int64_t next = group[i + 1];
      int64_t expected;
      if (llvm::MulOverflow(sizes[next], strides[next], expected) ||
          strides[d] != expected)
        return failure();
    }
    newStrides.push_back(strides[group.back()]);
  }
  return newStrides;
}

static FailureOr<SmallVector<int64_t>>
collapseSizes(ArrayRef<int64_t> sizes, ArrayRef<ReassociationIndices> reassoc) {
  SmallVector<int64_t> newSizes;
  newSizes.reserve(reassoc.size());
  for (const ReassociationIndices &group : reassoc) {
    // Match collapseContiguousStrides: reject empty reassociation groups.
    if (group.empty())
      return failure();
    int64_t size = 1;
    for (int64_t d : group) {
      if (llvm::MulOverflow(size, sizes[d], size))
        return failure();
    }
    newSizes.push_back(size);
  }
  return newSizes;
}

static LogicalResult
foldNdMemrefLoadToRank3(Value tensorSrc, RankedTensorType srcType,
                        RankedTensorType dstType,
                        ArrayRef<ReassociationIndices> reassoc,
                        PatternRewriter &rewriter, Operation *reshapeLikeOp) {
  if (!srcType.hasStaticShape() || !dstType.hasStaticShape())
    return failure();
  if (dstType.getRank() != 3 || srcType.getRank() <= 3)
    return failure();
  if (static_cast<int64_t>(reassoc.size()) != dstType.getRank())
    return failure();

  // Sole consumer: otherwise erasing/rewriting the load chain would drop or
  // corrupt data for other users of to_tensor.
  if (!tensorSrc.hasOneUse())
    return rewriter.notifyMatchFailure(
        reshapeLikeOp, "to_tensor / reshape source has multiple consumers");

  auto toTensor = tensorSrc.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensor || toTensor.getResult() != tensorSrc)
    return failure();

  Value allocVal = toTensor.getMemref();
  auto alloc = allocVal.getDefiningOp<memref::AllocOp>();
  if (!alloc || !alloc.getType().hasStaticShape())
    return failure();
  if (!llvm::equal(alloc.getType().getShape(), srcType.getShape()))
    return failure();

  // Alloc users must be exactly {copy as target, to_tensor}. Any other user
  // (e.g. a second view) means we cannot safely replace/erase this chain.
  memref::CopyOp copyOp = nullptr;
  for (Operation *user : allocVal.getUsers()) {
    if (user == toTensor)
      continue;
    auto copy = dyn_cast<memref::CopyOp>(user);
    if (!copy || copy.getTarget() != allocVal || copyOp)
      return failure();
    copyOp = copy;
  }
  if (!copyOp)
    return failure();

  auto reinterp = copyOp.getSource().getDefiningOp<memref::ReinterpretCastOp>();
  if (!reinterp)
    return failure();

  ArrayRef<int64_t> srcShape = srcType.getShape();
  ArrayRef<int64_t> dstShape = dstType.getShape();
  ArrayRef<int64_t> staticSizes = reinterp.getStaticSizes();
  ArrayRef<int64_t> staticStrides = reinterp.getStaticStrides();
  ArrayRef<int64_t> staticOffsets = reinterp.getStaticOffsets();
  if (staticSizes.size() != static_cast<size_t>(srcType.getRank()) ||
      staticStrides.size() != static_cast<size_t>(srcType.getRank()) ||
      staticOffsets.size() != 1)
    return failure();
  if (llvm::any_of(staticSizes, ShapedType::isDynamic) ||
      llvm::any_of(staticStrides, ShapedType::isDynamic) ||
      ShapedType::isDynamic(staticOffsets[0]))
    return failure();
  if (!llvm::equal(staticSizes, srcShape))
    return failure();

  FailureOr<SmallVector<int64_t>> newSizesOr =
      collapseSizes(staticSizes, reassoc);
  if (failed(newSizesOr) || !llvm::equal(*newSizesOr, dstShape))
    return failure();
  ArrayRef<int64_t> newSizes = *newSizesOr;

  FailureOr<SmallVector<int64_t>> newStridesOr =
      collapseContiguousStrides(staticSizes, staticStrides, reassoc);
  if (failed(newStridesOr))
    return rewriter.notifyMatchFailure(
        reshapeLikeOp, "cannot collapse non-contiguous strides into 3D load");
  ArrayRef<int64_t> newStrides = *newStridesOr;

  if (srcType.getElementType() != dstType.getElementType())
    return failure();

  int64_t offset = staticOffsets[0];
  Type elemType = dstType.getElementType();
  auto reinterpType = cast<MemRefType>(reinterp.getType());
  Attribute srcMemSpace = reinterpType.getMemorySpace();
  Attribute allocMemSpace = alloc.getType().getMemorySpace();

  auto layout =
      StridedLayoutAttr::get(rewriter.getContext(), offset, newStrides);
  auto newSrcMemrefType =
      MemRefType::get(newSizes, elemType, layout, srcMemSpace);
  // Contiguous destination buffer: identity layout, preserve memory space.
  auto newAllocType =
      MemRefType::get(newSizes, elemType,
                      /*layout=*/MemRefLayoutAttrInterface(), allocMemSpace);

  // Insert each replacement at the corresponding original op so SSA dominance
  // and memory side-effect ordering (relative to intervening ops) are kept.
  rewriter.setInsertionPoint(reinterp);
  Value newReinterp = rewriter.create<memref::ReinterpretCastOp>(
      reinterp.getLoc(), newSrcMemrefType, reinterp.getSource(), offset,
      newSizes, newStrides);

  rewriter.setInsertionPoint(alloc);
  Value newAlloc = rewriter.create<memref::AllocOp>(
      alloc.getLoc(), newAllocType, alloc.getAlignmentAttr());

  rewriter.setInsertionPoint(copyOp);
  rewriter.create<memref::CopyOp>(copyOp.getLoc(), newReinterp, newAlloc);

  rewriter.setInsertionPoint(toTensor);
  // to_tensor result must be the tensor-equivalent of the memref (no encoding).
  // Preserve dst encoding via an explicit cast when needed.
  RankedTensorType plainTensorType =
      RankedTensorType::get(dstType.getShape(), dstType.getElementType());
  Value newToTensor = rewriter.create<bufferization::ToTensorOp>(
      toTensor.getLoc(), plainTensorType, newAlloc, toTensor.getRestrict(),
      toTensor.getWritable());
  if (plainTensorType != dstType) {
    newToTensor = rewriter.create<tensor::CastOp>(toTensor.getLoc(), dstType,
                                                  newToTensor);
  }

  rewriter.replaceOp(reshapeLikeOp, newToTensor);

  // Sole-use was required above, so the old chain is now dead.
  rewriter.eraseOp(toTensor);
  rewriter.eraseOp(copyOp);
  rewriter.eraseOp(alloc);
  if (reinterp->use_empty())
    rewriter.eraseOp(reinterp);
  return success();
}

/// tensor.reshape of an nD memref load into a static 3D tensor.
class FoldNdLoadReshapeTo3D : public OpRewritePattern<tensor::ReshapeOp> {
public:
  explicit FoldNdLoadReshapeTo3D(MLIRContext *context)
      : OpRewritePattern<tensor::ReshapeOp>(context, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(tensor::ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<RankedTensorType>(reshapeOp.getSource().getType());
    auto dstType = dyn_cast<RankedTensorType>(reshapeOp.getResult().getType());
    if (!srcType || !dstType)
      return failure();
    std::optional<SmallVector<ReassociationIndices>> reassoc =
        getReassociationIndicesForCollapse(srcType.getShape(),
                                           dstType.getShape());
    if (!reassoc)
      return failure();
    return foldNdMemrefLoadToRank3(reshapeOp.getSource(), srcType, dstType,
                                   *reassoc, rewriter, reshapeOp);
  }
};

/// tensor.collapse_shape of an nD memref load into a static 3D tensor.
class FoldNdLoadCollapseTo3D
    : public OpRewritePattern<tensor::CollapseShapeOp> {
public:
  explicit FoldNdLoadCollapseTo3D(MLIRContext *context)
      : OpRewritePattern<tensor::CollapseShapeOp>(context, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<RankedTensorType>(collapseOp.getSrcType());
    auto dstType = dyn_cast<RankedTensorType>(collapseOp.getResultType());
    if (!srcType || !dstType)
      return failure();
    // Respect the op's own reassociation (do not re-infer).
    return foldNdMemrefLoadToRank3(collapseOp.getSrc(), srcType, dstType,
                                   collapseOp.getReassociationIndices(),
                                   rewriter, collapseOp);
  }
};


class FoldStaticReshape : public OpRewritePattern<tensor::ReshapeOp> {
public:
  using OpRewritePattern<tensor::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    Value src = reshapeOp.getSource();
    Value dst = reshapeOp.getResult();

    RankedTensorType srcType = dyn_cast<RankedTensorType>(src.getType());
    RankedTensorType dstType = dyn_cast<RankedTensorType>(dst.getType());
    if (!srcType || !dstType || !srcType.hasStaticShape() || !dstType.hasStaticShape()) {
      return failure();
    }
    
    std::optional<int64_t> srcTotalSize = mlir::utils::getStaticTotalSize(srcType.getShape());
    std::optional<int64_t> dstTotalSize = mlir::utils::getStaticTotalSize(dstType.getShape());
    if (!srcTotalSize || !dstTotalSize || srcTotalSize.value() != dstTotalSize.value()) {
      return failure();
    }

    // Prevent invalid collapse/expand if source or destination rank is 0 (scalar)
    if (srcType.getRank() == 0 || dstType.getRank() == 0) {
      return failure();
    }

    SmallVector<ReassociationIndices> collapseReassociation(1);
    for (int64_t i = 0; i < srcType.getRank(); ++i) {
      collapseReassociation[0].push_back(i);
    }
    SmallVector<ReassociationIndices> expandReassociation(1);
    for (int64_t i = 0; i < dstType.getRank(); ++i) {
      expandReassociation[0].push_back(i);
    }

    rewriter.setInsertionPointAfter(reshapeOp);
    auto flat1DType = RankedTensorType::get({srcTotalSize.value()}, srcType.getElementType());
    // collapse_shape: srcShape -> 1D
    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        reshapeOp.getLoc(), flat1DType, src, collapseReassociation);
    // expand_shape: 1D -> dstShape
    auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
        reshapeOp.getLoc(), dstType, collapseOp.getResult(),
        expandReassociation);
    rewriter.replaceAllUsesWith(reshapeOp, expandOp.getResult());
    return success();
  }
};

void CanonicalizeTensorReshape::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  // Prefer folding nD load+reshape/collapse into a 3D load before decomposing
  // reshape into collapse/expand pairs.
  patterns.insert<FoldNdLoadReshapeTo3D, FoldNdLoadCollapseTo3D>(context);
  patterns.insert<CanonicalizeTensorReshapeOpPattern>(patterns.getContext());
  patterns.insert<FoldReshape>(patterns.getContext());
  // This pattern decomposes `tensor.reshape` into a one-dimensional tensor using collapse and then expands it.
  patterns.insert<FoldStaticReshape>(patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}
} // namespace mlir::tensor

std::unique_ptr<Pass> mlir::tensor::createCanonicalizeTensorReshapePass() {
  return std::make_unique<CanonicalizeTensorReshape>();
}
