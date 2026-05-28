//===- FoldExtractInsertPair.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir {
#define GEN_PASS_DEF_FOLDEXTRACTINSERTPAIR
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;
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
///   %93 = linalg.elemwise_binary ...
///   %extracted_14 = tensor.extract %93[%c0] : tensor<1xf32>
///   %inserted_15 = tensor.insert %extracted_14 into %empty[%c0]
///   %94 = linalg.elemwise_unary ins(%inserted_15) ...
///
/// and rewrites the unary op to consume %93 directly.
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
      return rewriter.notifyMatchFailure(
          insertOp, "not a statically one-element tensor");

    // Be conservative: require extracting and inserting at the same indices.
    if (!llvm::equal(extractOp.getIndices(), insertOp.getIndices()))
      return rewriter.notifyMatchFailure(
          insertOp, "extract and insert indices differ");

    // Optional conservative check:
    // Require the destination to be tensor.empty. This matches your generated IR
    // and avoids surprising rewrites on non-empty destinations.
    //
    // For a statically one-element tensor, this rewrite is also valid even if
    // the destination is not tensor.empty, because the only element is replaced.
    // Keep this check if you want the transformation to be narrowly targeted.
    if (!insertOp.getDest().getDefiningOp<tensor::EmptyOp>())
      return rewriter.notifyMatchFailure(
          insertOp, "insert destination is not tensor.empty");

    rewriter.replaceOp(insertOp, srcTensor);

    if (extractOp->use_empty())
      rewriter.eraseOp(extractOp);

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
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hfusion::createFoldExtractInsertPairPass() {
  return std::make_unique<FoldExtractInsertPairPass>();
}
