//===--- CanonicalizeTensorReshape.cpp -  canonicalize tensor reshape------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to canonicalize tensor reshape operations.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

void CanonicalizeTensorReshape::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<CanonicalizeTensorReshapeOpPattern>(patterns.getContext());
  patterns.insert<FoldReshape>(patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}
} // namespace mlir::tensor

std::unique_ptr<Pass> mlir::tensor::createCanonicalizeTensorReshapePass() {
  return std::make_unique<CanonicalizeTensorReshape>();
}
