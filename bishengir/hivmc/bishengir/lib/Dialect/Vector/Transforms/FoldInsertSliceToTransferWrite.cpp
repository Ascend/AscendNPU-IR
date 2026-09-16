//===---------------- FoldInsertSliceToTransferWrite.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FOLDINSERTSLICETOTRANSFERWRITE
#include "bishengir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "fold-insert-slice-to-transfer-write"

using namespace mlir;
using namespace mlir::tensor;

namespace {

// Fold insert_slice target to src transfer_write's dest to avoid write to empty
// tensor
//
// before:
// - %0 = arith.add ... : vector<64xf32>
// - %1 = tensor.empty() : tensor<1xf32>
// - %2 = vector.transfer_write %0, %1 : vector<64xf32>, tensor<1xf32>
// - tensor.insert_slice %2 into %arg0[%offset] [1] [1]
//
// after:
// - %0 = arith.add ... : vector<64xf32>
// - %slice0 = tensor.extract_slice %arg0[%offset] [1] [1]
// - %3 = vector.transfer_write %0, %slice0 : vector<64xf32>, tensor<1xf32>
// - tensor.insert_slice %2 into %arg0[%offset] [1] [1]
//
struct FoldInsertSliceToTransferWrite
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    // 1. Check if source comes from vector.transfer_write
    auto transferWriteOp =
        insertOp.getSource().getDefiningOp<vector::TransferWriteOp>();
    if (!transferWriteOp)
      return failure();

    // 2. transfer_write dst should not be tensor.extract_slice, which means
    // this pattern already applied
    if (transferWriteOp.getSource().getDefiningOp<tensor::ExtractSliceOp>())
      return failure();

    rewriter.setInsertionPoint(transferWriteOp);

    // 3. move transfer_write close to insert_slice to avoid dominate issues if
    // the new extract_slice before transfer_write depends on the operands of
    // insert_slice op
    if (transferWriteOp->hasOneUse()) {
      rewriter.modifyOpInPlace(
          insertOp, [&]() { transferWriteOp->moveBefore(insertOp); });
    }

    // 4. Create a matching extract_slice from the insert_slice's destination.
    // This allows the transfer_write to operate on the actual slice of the
    // target.
    auto insertDest = insertOp.getDest();
    auto extractOp = rewriter.create<tensor::ExtractSliceOp>(
        transferWriteOp.getLoc(), insertOp.getSourceType(), insertDest,
        insertOp.getMixedOffsets(), insertOp.getMixedSizes(),
        insertOp.getMixedStrides());

    // 5. Create the optimized transfer_write
    auto newTransferWriteOp = rewriter.create<vector::TransferWriteOp>(
        transferWriteOp.getLoc(), transferWriteOp.getVector(),
        extractOp.getResult(), // Dst is now the extracted slice
        transferWriteOp.getIndices(), transferWriteOp.getPermutationMapAttr(),
        transferWriteOp.getInBoundsAttr());

    // 6. Replace the old transfer_write with the new result
    rewriter.replaceOp(transferWriteOp, newTransferWriteOp.getResult());
    return success();
  }
};

struct FoldInsertSliceToTransferWritePass
    : public impl::FoldInsertSliceToTransferWriteBase<
          FoldInsertSliceToTransferWritePass> {
public:
  void runOnOperation() override;
};

} // anonymous namespace

void FoldInsertSliceToTransferWritePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<FoldInsertSliceToTransferWrite>(context);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::vector::createFoldInsertSliceToTransferWritePass() {
  return std::make_unique<FoldInsertSliceToTransferWritePass>();
}
