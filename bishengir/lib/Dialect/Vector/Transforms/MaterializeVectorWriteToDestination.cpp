//===---------------- MaterializeVectorWriteToDestination.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Scope/Utils/Utils.h"
#include "bishengir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_MATERIALIZEVECTORWRITETODESTINATION
#include "bishengir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "materialize-vector-write-to-destination"

using namespace mlir;
using namespace mlir::scf;
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

// Replaces vector.transfer_write destination with the corresponding scf.for
// iter_arg if it currently writes to a tensor.empty and the result is yielded.
// This enables in-place bufferization and eliminates redundant allocations.
//
// Example:
// Before:
// - %res = scf.for %i = %c0 to %c100 step %c1 iter_args(%ia = %init) {
// -   %0 = tensor.empty() : tensor<f32> (or function arg)
// -   %new = vector.transfer_write %vec, %0[...] : vector<f32>, tensor<f32>
// -   scf.yield %new
// - }
//
// After:
// - %res = scf.for %i = %c0 to %c100 step %c1 iter_args(%ia = %init) ->
// - (tensor<f32>) {
// -   %new = vector.transfer_write %vec, %ia[...] : vector<f32>, tensor<f32>
// -   scf.yield %new
// - }
struct RedirectWriteDstToIterArg : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    bool changed = false;

    for (auto it : llvm::enumerate(yieldOp.getOperands())) {
      Value yieldedVal = it.value();

      // 1. Match transfer_write that defines the yielded value
      auto writeOp = yieldedVal.getDefiningOp<vector::TransferWriteOp>();
      if (!writeOp)
        continue;

      // 2. Match if destination is tensor.empty or function arg
      Value destTensor = writeOp.getSource();
      if (!destTensor.getDefiningOp<tensor::EmptyOp>() &&
          !isFuncArg(destTensor)) {
        // support function arg destination may cause problem: issue !1004
        // TODO: remove it after support clone-tensor-empty before outline-scope
        continue;
      }

      // 3. Get the corresponding iter_arg for this yield operand
      BlockArgument iterArg = forOp.getRegionIterArg(it.index());
      if (iterArg.getType() != destTensor.getType())
        continue;

      // 4. Replace with new transfer_write targeting iter_arg
      rewriter.setInsertionPoint(writeOp);
      auto newWriteOp = rewriter.create<vector::TransferWriteOp>(
          writeOp.getLoc(), writeOp.getVector(),
          iterArg, // Redirect destination
          writeOp.getIndices(), writeOp.getPermutationMapAttr(),
          writeOp.getMask(), writeOp.getInBoundsAttr());

      rewriter.replaceOp(writeOp, newWriteOp.getResult());
      changed = true;
    }

    return success(changed);
  }

private:
  bool isFuncArg(Value value) const {
    auto blockArg = dyn_cast<BlockArgument>(value);
    return blockArg && blockArg.getOwner()->isEntryBlock() &&
           isa<FunctionOpInterface>(blockArg.getOwner()->getParentOp());
  }
};

struct MaterializeVectorWriteToDestinationPass
    : public impl::MaterializeVectorWriteToDestinationBase<
          MaterializeVectorWriteToDestinationPass> {
public:
  void runOnOperation() override;
};

} // anonymous namespace

void MaterializeVectorWriteToDestinationPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (!scope::utils::isManualVFScope(funcOp)) {
    // This pass should only works for manual VF scope function
    return;
  }
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<FoldInsertSliceToTransferWrite>(context);
  patterns.add<RedirectWriteDstToIterArg>(context);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass>
mlir::vector::createMaterializeVectorWriteToDestinationPass() {
  return std::make_unique<MaterializeVectorWriteToDestinationPass>();
}
