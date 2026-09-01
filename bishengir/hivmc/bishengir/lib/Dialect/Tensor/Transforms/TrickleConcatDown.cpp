//===- TrickleConcatDown.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"

#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "trickle-concat-down"
namespace mlir {
#define GEN_PASS_DEF_TRICKLECONCATDOWN
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace mlir {

namespace tensor {
using namespace mlir::hfusion;
using namespace mlir::tensor::reshape_utils;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;
namespace {
class TrickleConcatElemwise : public mlir::OpRewritePattern<tensor::ConcatOp> {
public:
  explicit TrickleConcatElemwise(MLIRContext *context)
      : OpRewritePattern<tensor::ConcatOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(tensor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
TrickleConcatElemwise::matchAndRewrite(tensor::ConcatOp concatOp,
                                       PatternRewriter &rewriter) const {
  auto dim = concatOp.getDim();
  auto concatResultShape = utils::getShape(concatOp.getResult().getType());
  auto rank = concatResultShape.size();
  // check for this usage
  if (!concatOp->hasOneUse())
    return failure();
  Operation *user = *concatOp->getUsers().begin();
  // Check if its unary like
  if (!isMarkedAsElementwiseUnaryOp(user))
    return failure();
  if (dim != rank - 1)
    return failure();
  // Gather lastDimensions static shapes
  SmallVector<int64_t> lastDimensions;
  for (auto opr : concatOp.getOperands()) {
    if (isa<RankedTensorType>(opr.getType())) {
      auto shape = utils::getShape(opr.getType()).back();
      lastDimensions.push_back(shape);
    }
  }
  if (utils::areShapesAligned(lastDimensions, 32))
    return failure();
  auto dstStyOp = dyn_cast<DestinationStyleOpInterface>(user);
  if (!dstStyOp)
    return failure();
  SmallVector<Value> newOperands;

  for (auto opr : concatOp->getOperands()) {
    // Manipulate this parameter
    if (isa<RankedTensorType>(opr.getType()) &&
        utils::getShapeRank(opr) == rank) {
      Operation *dstClone = rewriter.clone(*user);
      auto newSrc = cast<DestinationStyleOpInterface>(dstClone);
      rewriter.setInsertionPoint(dstClone);
      auto outEmpty = utils::createEmptyOpWithTargetElemType(
          rewriter, newSrc.getLoc(), opr,
          getElementTypeOrSelf(dstClone->getResultTypes().front()));
      // Change it and clone it into empty
      // Iterate new Src
      SmallVector<Value> newSrcOperands;
      for (auto &newSrcOpr : newSrc->getOpOperands()) {
        if (newSrc.isDpsInput(&newSrcOpr)) {
          newSrcOperands.push_back(opr);
        } else if (newSrc.isDpsInit(&newSrcOpr)) {
          newSrcOperands.push_back(outEmpty);
        }
      }
      updateDefiningOp(newSrc, rewriter, newSrcOperands);
      newOperands.push_back(newSrc->getResult(0));
    } else {
      newOperands.push_back(opr);
    }
  }
  rewriter.modifyOpInPlace(concatOp, [&]() {
    concatOp->setOperands(newOperands);
    auto concatInputType =
        getElementTypeOrSelf(concatOp->getOperandTypes().front());
    concatOp.getResult().setType(
        concatOp.getResultType().clone(concatInputType));
  });
  rewriter.replaceAllUsesWith(user->getResults(), concatOp.getResult());
  rewriter.eraseOp(user);
  return success();
}

class TrickleConcatSliceModifying
    : public mlir::OpRewritePattern<tensor::ConcatOp> {
public:
  explicit TrickleConcatSliceModifying(MLIRContext *context)
      : OpRewritePattern<tensor::ConcatOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(tensor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    auto dim = concatOp.getDim();

    if (!concatOp->hasOneUse())
      return failure();
    Operation *user = *concatOp->getUsers().begin();
    auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    if (!extractSliceOp)
      return failure();

    // Ensure all operands have static sizes along the concat dimension.
    for (auto operand : concatOp.getOperands()) {
      auto type = utils::getShape(operand.getType());
      if (ShapedType::isDynamic(type[dim]))
        return failure();
    }

    auto sliceStaticOffsets = extractSliceOp.getStaticOffsets();
    auto sliceStaticSizes = extractSliceOp.getStaticSizes();
    auto sliceStaticStrides = extractSliceOp.getStaticStrides();

    int64_t concatStride = sliceStaticStrides[dim];
    int64_t startOffset = sliceStaticOffsets[dim];
    int64_t remainingSize = sliceStaticSizes[dim];

    if (ShapedType::isDynamic(concatStride) ||
        ShapedType::isDynamic(startOffset) ||
        ShapedType::isDynamic(remainingSize))
      return failure();

    SmallVector<int64_t> newSizes(concatOp.getOperands().size(), 0);
    SmallVector<int64_t> newOffsets(concatOp.getOperands().size(), -1);

    // Calculate per-operand slice parameters.
    for (auto [idx, operand] : llvm::enumerate(concatOp.getOperands())) {
      if (remainingSize == 0)
        break;
      auto opType = cast<RankedTensorType>(operand.getType());
      int64_t opDimSize = opType.getDimSize(dim);
      if (startOffset >= opDimSize) {
        startOffset -= opDimSize;
        continue;
      }

      int64_t available = opDimSize - startOffset;
      int64_t maxPossible =
          (available / concatStride) + !!(available % concatStride);
      int64_t sizeToTake = std::min(maxPossible, remainingSize);
      assert(sizeToTake > 0);

      newOffsets[idx] = startOffset;
      newSizes[idx] = sizeToTake;
      remainingSize -= sizeToTake;

      int64_t currentStop = startOffset + (sizeToTake - 1) * concatStride;
      startOffset = currentStop + concatStride - opDimSize;
    }
    if (remainingSize > 0) {
      return failure();
    }
    // Create new slices and collect operands.
    SmallVector<Value> newConcatOperands;
    auto loc = concatOp.getLoc();
    SmallVector<OpFoldResult> offsets = extractSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = extractSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = extractSliceOp.getMixedStrides();

    for (auto [idx, operand] : llvm::enumerate(concatOp.getOperands())) {
      if (newSizes[idx] <= 0)
        continue;

      offsets[dim] = rewriter.getIndexAttr(newOffsets[idx]);
      sizes[dim] = rewriter.getIndexAttr(newSizes[idx]);

      auto newSlice = rewriter.create<tensor::ExtractSliceOp>(
          loc, operand, offsets, sizes, strides);
      newConcatOperands.push_back(newSlice);
    }

    // Create new concat and replace original ops.
    auto newConcat = rewriter.create<tensor::ConcatOp>(
        loc, extractSliceOp.getType(), dim, newConcatOperands);
    rewriter.replaceOp(extractSliceOp, newConcat.getResult());
    rewriter.eraseOp(concatOp);

    return success();
  }
};

class TrickleConcatDownPass
    : public impl::TrickleConcatDownBase<TrickleConcatDownPass> {
  using Base::Base;
  void runOnOperation() final;
};

void TrickleConcatDownPass::runOnOperation() {
  func::FuncOp f = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  patterns.add<TrickleConcatElemwise>(context);
  patterns.add<TrickleConcatSliceModifying>(context);
  if (failed(applyPatternsGreedily(f, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<Pass> createTrickleConcatDownPass() {
  return std::make_unique<TrickleConcatDownPass>();
}

} // namespace tensor
} // namespace mlir
