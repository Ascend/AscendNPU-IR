//===- FoldExtractInsertPair.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir {
#define GEN_PASS_DEF_FOLDEXTRACTINSERTPAIR
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
#define DEBUG_TYPE "fold-extract-insert-pair"

namespace {

static bool isStaticOneElementTensor(Type type) {
  auto rankedTy = dyn_cast<RankedTensorType>(type);
  if (!rankedTy)
    return false;

  if (!rankedTy.hasStaticShape())
    return false;

  return rankedTy.getNumElements() == 1;
}

/// Fold:
///
///   %e = tensor.extract %src[%idx] : tensor<1xf32>
///   %r = tensor.insert %e into %dst[%idx] : tensor<1xf32>
///
/// into:
///
///   %r = %src
///
/// This is only legal when %src and %r have the same statically one-element
/// tensor type. In that case, inserting the only element reconstructs the
/// whole tensor.
///
/// This removes chains like:
///
///   %result = linalg.elemwise_binary ...
///   %extracted = tensor.extract %result[%c0] : tensor<1xf32>
///   %inserted = tensor.insert %extracted into %empty[%c0]
///   %consumer = linalg.elemwise_unary ins(%inserted) ...
///
/// and rewrites the consumer to use %result directly.
class FoldExtractInsertPair : public OpRewritePattern<tensor::InsertOp> {
public:
  using OpRewritePattern<tensor::InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertOp insertOp,
                                PatternRewriter &rewriter) const override {
    auto extractOp = insertOp.getScalar().getDefiningOp<tensor::ExtractOp>();
    if (!extractOp)
      return rewriter.notifyMatchFailure(
          insertOp, "inserted scalar is not produced by tensor.extract");

    Value srcTensor = extractOp.getTensor();
    Type srcType = srcTensor.getType();
    Type resultType = insertOp.getResult().getType();

    if (srcType != resultType)
      return rewriter.notifyMatchFailure(
          insertOp, "source tensor type and insert result type differ");

    if (!isStaticOneElementTensor(resultType))
      return rewriter.notifyMatchFailure(insertOp,
                                         "not a statically one-element tensor");

    // Be conservative: require extracting and inserting at the same indices.
    if (!llvm::equal(extractOp.getIndices(), insertOp.getIndices()))
      return rewriter.notifyMatchFailure(insertOp,
                                         "extract and insert indices differ");

    // Conservatively require the destination to be tensor.empty. For a
    // statically one-element tensor the rewrite would also be valid on other
    // destinations, since the only element is replaced, but restricting it to
    // tensor.empty keeps the transformation narrowly targeted.
    if (!insertOp.getDest().getDefiningOp<tensor::EmptyOp>())
      return rewriter.notifyMatchFailure(
          insertOp, "insert destination is not tensor.empty");

    rewriter.replaceOp(insertOp, srcTensor);

    if (extractOp->use_empty())
      rewriter.eraseOp(extractOp);

    return success();
  }
};

/// Fold redundant extract_slice -> transfer_write -> insert_slice pattern:
///
///   %slice = tensor.extract_slice %src[offsets] [sizes] [strides]
///            : tensor<...> to tensor<?x...>
///   %written = vector.transfer_write %vec, %slice[offsets] {mask}
///              : vector<...>, tensor<?x...>
///   %result = tensor.insert_slice %written into %slice[offsets] [sizes]
///   [strides]
///             : tensor<?x...> into tensor<?x...>
///
/// into:
///
///   %result = %written
///
/// This pattern matches when:
/// 1. %slice is defined by tensor.extract_slice
/// 2. %written is defined by vector.transfer_write to %slice at the same
/// offsets
/// 3. %result is tensor.insert_slice of %written into %slice at the same
/// offsets
/// 4. The sizes and strides of extract_slice and insert_slice match
///
/// This is redundant because %written already has the same shape as %slice,
/// and inserting it back at the same offset with the same size is a no-op.
class FoldExtractSliceInsertSlice
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    Value source = insertOp.getSource();
    Value dest = insertOp.getDest();
    auto transferWriteOp = source.getDefiningOp<vector::TransferWriteOp>();
    if (!transferWriteOp)
      return rewriter.notifyMatchFailure(
          insertOp, "source is not defined by vector.transfer_write");
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    auto tensorValue = transferWriteOp.getSource();
#else
    auto tensorValue = transferWriteOp.getBase();
#endif
    auto extractSliceOp = tensorValue.getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractSliceOp)
      return rewriter.notifyMatchFailure(
          insertOp,
          "transfer_write dest is not defined by tensor.extract_slice");

    if (extractSliceOp.getResult() != dest) {
      return rewriter.notifyMatchFailure(
          insertOp,
          "extract_slice and insert_slice operate on different tensors");
    }

    // Check if transfer_write indices match insert_slice offsets
    auto writeIndices = transferWriteOp.getIndices();
    auto insertOffsets = insertOp.getMixedOffsets();
    if (writeIndices.size() != insertOffsets.size()) {
      return rewriter.notifyMatchFailure(insertOp, "offset sizes differ");
    }

    for (auto [writeIdx, insertOff] : llvm::zip(writeIndices, insertOffsets)) {
      auto writeInt = mlir::getConstantIntValue(writeIdx);
      auto insertInt = mlir::getConstantIntValue(insertOff);

      if (writeInt && insertInt) {
        if (writeInt.value() == insertInt.value())
          continue;
        return rewriter.notifyMatchFailure(insertOp, "offsets do not match");
      }

      // For dynamic values, require exact match
      if (auto insertVal = mlir::dyn_cast_if_present<mlir::Value>(insertOff)) {
        if (writeIdx != insertVal) {
          return rewriter.notifyMatchFailure(insertOp, "offsets do not match");
        }
      } else {
        return rewriter.notifyMatchFailure(insertOp, "offsets do not match");
      }
    }

    // Check if extract_slice sizes match insert_slice sizes
    auto extractSizes = extractSliceOp.getMixedSizes();
    auto insertSizes = insertOp.getMixedSizes();
    if (extractSizes.size() != insertSizes.size()) {
      return rewriter.notifyMatchFailure(insertOp, "size ranks differ");
    }

    for (auto [extractSz, insertSz] : llvm::zip(extractSizes, insertSizes)) {
      auto extractInt = mlir::getConstantIntValue(extractSz);
      auto insertInt = mlir::getConstantIntValue(insertSz);

      if (extractInt && insertInt) {
        if (extractInt.value() != insertInt.value()) {
          return rewriter.notifyMatchFailure(insertOp, "sizes do not match");
        }
      } else if (extractSz != insertSz) {
        // For dynamic sizes, require exact value match
        auto extractVal = mlir::dyn_cast_if_present<mlir::Value>(extractSz);
        auto insertVal = mlir::dyn_cast_if_present<mlir::Value>(insertSz);
        if (!extractVal || !insertVal || extractVal != insertVal) {
          return rewriter.notifyMatchFailure(insertOp, "sizes do not match");
        }
      }
    }

    // Check if strides match (all should be 1 for this pattern)
    auto insertStrides = insertOp.getMixedStrides();
    for (auto insertStride : insertStrides) {
      auto strideInt = mlir::getConstantIntValue(insertStride);
      if (!strideInt || strideInt.value() != 1) {
        return rewriter.notifyMatchFailure(insertOp, "strides must be 1");
      }
    }

    rewriter.replaceOp(insertOp, source);
    return success();
  }
};

/// Folds an insert_slice into an extract_slice that is inserted back into the
/// same tensor:
///
///   %slice = tensor.extract_slice %full[po] [ps] [pst]
///   %inner = tensor.insert_slice %src into %slice[co] [cs] [cst]
///   %res = tensor.insert_slice %inner into %full[po] [ps] [pst]
///
/// into a single insert_slice that writes %src directly into %full at the
/// merged position, provided %inner has no other uses. This cleans up the
/// slice chains that FoldSliceIntoScfForPattern creates inside the loop body.
struct FoldInsertSliceIntoExtractedSlice
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp backOp,
                                PatternRewriter &rewriter) const override {
    auto innerOp = backOp.getSource().getDefiningOp<tensor::InsertSliceOp>();
    if (!innerOp || !innerOp->hasOneUse())
      return failure();
    auto extractOp = innerOp.getDest().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractOp)
      return failure();
    // The write-back must target the same tensor the slice was taken from,
    // at the same position.
    if (extractOp.getSource() != backOp.getDest())
      return failure();
    if (backOp.getMixedOffsets() != extractOp.getMixedOffsets() ||
        backOp.getMixedSizes() != extractOp.getMixedSizes() ||
        backOp.getMixedStrides() != extractOp.getMixedStrides())
      return failure();

    SmallVector<OpFoldResult> offsets, sizes, strides;
    if (failed(affine::mergeOffsetsSizesAndStrides(
            rewriter, backOp.getLoc(), extractOp, innerOp,
            extractOp.getDroppedDims(), offsets, sizes, strides)))
      return failure();
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        backOp, innerOp.getSource(), extractOp.getSource(), offsets, sizes,
        strides);
    return success();
  }
};

struct FoldExtractInsertPairPass
    : public impl::FoldExtractInsertPairBase<FoldExtractInsertPairPass> {
  void runOnOperation() override;
};
} // namespace

void FoldExtractInsertPairPass::runOnOperation() {
  auto funcOp = getOperation();
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<FoldExtractInsertPair>(ctx);
  patterns.add<FoldExtractSliceInsertSlice>(ctx);
  patterns.add<FoldInsertSliceIntoExtractedSlice>(ctx);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::tensor::createFoldExtractInsertPairPass() {
  return std::make_unique<FoldExtractInsertPairPass>();
}
