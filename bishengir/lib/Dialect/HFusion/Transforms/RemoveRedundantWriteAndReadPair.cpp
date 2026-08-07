//===--------------- RemoveRedundantWriteAndReadPair.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_REMOVEREDUNDANTWRITEANDREADPAIR
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "remove-redundant-write-and-read-pair"

using namespace mlir;
using namespace mlir::hfusion;

namespace {
/// Replace the transfer_read by source vector of transfer_write.
/// Example:
/// ```
/// %7 = arith.cmpi ne, %6, %cst_2
/// %8 = vector.transfer_write %7, %extract_slice_4[%c0, %c0]
/// %inserted_slice = tensor.insert_slice %8 into %extract_slice_4
/// %13 = vector.transfer_read %inserted_slice[%c0, %c0]
/// %20 = arith.select %13, %19, %17
/// ```
/// To:
/// ```
/// %7 = arith.cmpi ne, %6, %cst_2
/// %20 = arith.select %7, %19, %17
/// ```
struct FoldTransferReadAfterWriteAndInsertSlice
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    if (readOp.hasOutOfBoundsDim() ||
        !llvm::isa<RankedTensorType>(readOp.getShapedType()))
      return failure();
    auto defInsertSlice = readOp.getSource().getDefiningOp<tensor::InsertSliceOp>();
    if (!defInsertSlice)
      return failure();
    auto defWrite = defInsertSlice.getSource().getDefiningOp<vector::TransferWriteOp>();
    if (!defWrite)
      return failure();
    if (readOp.getIndices() != defWrite.getIndices() ||
        readOp.getVectorType().getNumElements() !=
            defWrite.getVectorType().getNumElements())
      return failure();
    // add shape cast, if the read shape different with write shape
    if (readOp.getVectorType() != defWrite.getVectorType()) {
      Location loc = readOp->getLoc();
      vector::ShapeCastOp shapeCast = rewriter.create<vector::ShapeCastOp>(
          loc, readOp.getVectorType(), defWrite.getVector());
      rewriter.replaceOp(readOp, shapeCast);
    } else {
      rewriter.replaceOp(readOp, defWrite.getVector());
    }
    return success();
  }
};

struct RemoveRedundantWriteAndReadPairPass
    : public impl::RemoveRedundantWriteAndReadPairBase<
        RemoveRedundantWriteAndReadPairPass> {
public:
  void runOnOperation() override;
};
} // namespace

void RemoveRedundantWriteAndReadPairPass::runOnOperation() {
  func::FuncOp func = getOperation();
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<FoldTransferReadAfterWriteAndInsertSlice>(ctx);

  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}


std::unique_ptr<Pass>
mlir::hfusion::createRemoveRedundantWriteAndReadPairPass() {
  return std::make_unique<RemoveRedundantWriteAndReadPairPass>();
}