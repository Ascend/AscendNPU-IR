//===-- PostVectorizationNormalization.cpp - Post-vectorize normalization -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replace vector.transfer_read with broadcast i1 with
// non-broadcast read + sitofp to f16 + vector.broadcast + arith.cmpf une 0 chain,
// because the hardware forbids i1 broadcast.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_POSTVECTORIZATIONNORMALIZATION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "post-vectorization-normalization"

using namespace mlir;
using namespace mlir::hfusion;

namespace {
struct FixBitTypeBroadcastTransferReadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  // Rewrites:
  //   %r = vector.transfer_read %src[...], %pad
  //       {permutation_map = <with broadcast>}
  //       : tensor<...xi1>, vector<...xi1>
  // into:
  //   %read  = vector.transfer_read %src[...], %pad
  //            {in_bounds = [true,...]}
  //            : tensor<...xi1>, vector<...xi1>  // broadcast dims -> 1
  //   %cast  = arith.sitofp %read : vector<...xi1> to vector<...xf16>
  //   %bcast = vector.broadcast %cast : vector<...xf16> to vector<...xf16>
  //   %zero  = arith.constant dense<0.0> : vector<...xf16>
  //   %r     = arith.cmpf une %bcast, %zero : vector<...xf16> -> vector<...xi1>
  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    // Skip transfer_reads inside vector.mask — replacing a single op with
    // a chain would break the mask (which requires exactly one body op).
    if (isa<vector::MaskOp>(op->getParentOp()))
      return failure();
    auto resultVecType = op.getVectorType();
    auto elemType = resultVecType.getElementType();
    auto intTy = dyn_cast<IntegerType>(elemType);
    if (!intTy || intTy.getIntOrFloatBitWidth() > 1)
      return failure();

    auto map = op.getPermutationMap();
    if (map.isEmpty())
      return failure();

    auto rank = resultVecType.getRank();

    SmallVector<bool> isNonBroadcastDim(rank, false);
    for (auto expr : map.getResults())
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
        isNonBroadcastDim[dimExpr.getPosition()] = true;

    // Only match if a broadcast dim has actual size > 1.
    auto hasActualBroadcast = llvm::any_of(
        llvm::seq<int>(0, rank), [&isNonBroadcastDim, &resultVecType](auto i) {
          return !isNonBroadcastDim[i] && resultVecType.getDimSize(i) > 1;
        });
    if (!hasActualBroadcast)
      return failure();

    // Don't transform if the target f16 vector exceeds the hardware
    // broadcast limit (VL_BITS / 16 bits per element).
    if (!ShapedType::isDynamic(resultVecType.getNumElements()) &&
        resultVecType.getNumElements() * 16 > hivm::util::VL_BITS)
      return failure();

    SmallVector<int64_t> intermediateShape(rank, 1);
    for (int64_t i = 0; i < rank; i++)
      if (isNonBroadcastDim[i])
        intermediateShape[i] = resultVecType.getDimSize(i);

    auto intermediateVecType =
        VectorType::get(intermediateShape, elemType);
    auto intermediateVecTypeF16 =
        VectorType::get(intermediateShape, rewriter.getF16Type());
    auto targetVecTypeF16 =
        VectorType::get(resultVecType.getShape(), rewriter.getF16Type());

    SmallVector<bool> inBounds(rank, true);

    auto read = rewriter.create<vector::TransferReadOp>(
        op.getLoc(), intermediateVecType, op.getSource(),
        op.getIndices(), map, op.getPadding(),
        /*mask=*/Value(), rewriter.getBoolArrayAttr(inBounds));

    auto cast = rewriter.create<arith::SIToFPOp>(
        op.getLoc(), intermediateVecTypeF16, read);

    auto bcast = rewriter.create<vector::BroadcastOp>(
        op.getLoc(), targetVecTypeF16, cast);

    auto zero = rewriter.create<arith::ConstantOp>(
        op.getLoc(), targetVecTypeF16,
        rewriter.getZeroAttr(targetVecTypeF16));

    auto cmp = rewriter.create<arith::CmpFOp>(
        op.getLoc(), arith::CmpFPredicate::UNE, bcast, zero);

    rewriter.replaceOp(op, cmp);
    return success();
  }
};

struct PostVectorizationNormalizationPass
    : public impl::PostVectorizationNormalizationBase<
          PostVectorizationNormalizationPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FixBitTypeBroadcastTransferReadPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hfusion::createPostVectorizationNormalizationPass() {
  return std::make_unique<PostVectorizationNormalizationPass>();
}