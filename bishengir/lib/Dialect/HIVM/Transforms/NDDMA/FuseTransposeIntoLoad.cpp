//===- FuseTransposeIntoLoad.cpp - Fuse transpose into hivm load ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pattern that fuses linalg.transpose into hivm.hir.load
// by leveraging DMA on-the-fly transpose capability. When a load writes a tile
// that is later converted to tensor and transposed on the last two axes, this
// pass rewrites the load destination view so the hardware writes into the
// transposed result layout directly. The explicit transpose can then be
// eliminated.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/MemRef/Transforms/ComposeSubView.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/NDDMA/ComposeUnitStrideSubview.h"
#include "bishengir/Dialect/HIVM/Transforms/NDDMA/TileView.h"
#include "bishengir/Dialect/HIVM/Transforms/NDDMA/ViewPermutator.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_FUSETRANSPOSEINTOLOAD
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-fuse-transpose-into-load"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm::nddma;

namespace {

/// Walk the alias tree rooted at the allocation and accept only users that this
/// transform can reason about.
///
/// Example shape of the alias tree:
///
///   %root = memref.alloc()
///   ├─ %read = memref.subview %root[...]      // view-like, recurse
///   │  └─ bufferization.to_tensor %read       // matched tensor read
///   ├─ %dst = memref.subview %root[...]       // view-like, recurse
///   │  └─ hivm.hir.load outs(%dst)            // the single load to rewrite
///   ├─ linalg.fill outs(%root or alias)       // allowed initialization
///   └─ other users                            // return failure
LogicalResult inspectLoadUses(Value root, hivm::LoadOp &loadOp,
                              const TileView &permTile,
                              llvm::DenseSet<Value> &visited,
                              Builder &builder) {
  if (!visited.insert(root).second) {
    return success();
  }

  for (Operation *user : root.getUsers()) {

    if (isa<bufferization::ToTensorOp>(user) || hfusion::isFillOp(user)) {
      continue;
    }

    if (auto curLoadOp = dyn_cast<hivm::LoadOp>(user)) {
      auto loadTile = TileView::fromMemRef(curLoadOp.getDst(), builder);
      if (failed(loadTile) ||
          failed(verifyLoadTileCompatibleWithPermTile(*loadTile, permTile))) {
        return failure();
      }

      if (loadOp && loadOp != curLoadOp) {
        // there should be only one load
        return failure();
      }

      loadOp = curLoadOp;
      continue;
    }

    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
      for (Value result : user->getResults()) {
        if (failed(
                inspectLoadUses(result, loadOp, permTile, visited, builder))) {
          return failure();
        }
      }
      continue;
    }

    // failure for unhandled users
    return failure();
  }

  // all users are verified
  return success();
}

FailureOr<hivm::LoadOp> findLoadToFuse(Value root, const TileView &permTile,
                                       Builder &builder) {
  hivm::LoadOp loadOp = nullptr;
  llvm::DenseSet<Value> visited;
  if (failed(inspectLoadUses(root, loadOp, permTile, visited, builder)) ||
      !loadOp) {
    return failure();
  }
  return loadOp;
}

struct FuseTransposeIntoLoadPattern
    : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  /// Match `to_tensor -> linalg.transpose` and rewrite the producer load to
  /// write directly in the transposed result layout.
  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    if (!isLastTwoDimTranspose(perm)) {
      // Only fuse last-two-dim transpose to nddma.
      // TODO: should we support arbitrary permuations?
      return rewriter.notifyMatchFailure(transposeOp,
                                         "expected last-two-dim transpose");
    }

    Value input = transposeOp.getDpsInputOperand(0)->get();
    auto toTensorOp = input.getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensorOp || !toTensorOp->getResult(0).hasOneUse()) {
      return rewriter.notifyMatchFailure(
          transposeOp, "expected transpose to be the only to_tensor user");
    }

    auto permTile = TileView::fromMemRef(toTensorOp.getMemref(), rewriter);
    if (failed(permTile)) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "unsupported to_tensor memref");
    }
    LDBG("permTile: " << *permTile);

    if (!permTile->isContiguous()) {
      return rewriter.notifyMatchFailure(
          transposeOp, "expected contiguous layout to permute");
    }

    FailureOr<hivm::LoadOp> loadToFuse =
        findLoadToFuse(permTile->root, *permTile, rewriter);
    if (failed(loadToFuse)) {
      return rewriter.notifyMatchFailure(
          transposeOp, "unsupported load to fuse with transpose");
    }

    hivm::LoadOp loadOp = *loadToFuse;
    if (loadOp.getLeftPaddingNum() || loadOp.getRightPaddingNum()) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "load with padding is not supported");
    }

    // the memory write must dominate the memory read.
    DominanceInfo dominance(transposeOp->getParentOfType<func::FuncOp>());
    if (!dominance.properlyDominates(loadOp.getOperation(),
                                     toTensorOp.getOperation())) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "load does not dominate to_tensor");
    }

    auto loadSrcTile = TileView::fromMemRef(loadOp.getSrc(), rewriter);
    auto loadDstTile = TileView::fromMemRef(loadOp.getDst(), rewriter);
    if (failed(loadSrcTile) || failed(loadDstTile)) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "unsupported tile views on load");
    }
    LDBG("loadSrcTile: " << *loadSrcTile);
    LDBG("loadDstTile: " << *loadDstTile);

    rewriter.setInsertionPoint(loadOp);

    ViewPermutator srcPermutator(
        *loadSrcTile,
        getLastTwoDimPermutation(loadSrcTile->viewType.getRank()));
    ViewPermutator dstPermutator(
        *loadDstTile,
        getLastTwoDimPermutation(loadDstTile->viewType.getRank()));
    if (!srcPermutator.canPermute() || !dstPermutator.canPermute()) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "failed to permute load tiles");
    }

    TileView newLoadSrcTile = srcPermutator.permute(rewriter);
    TileView newLoadDstTile = dstPermutator.permute(rewriter);

    loadOp.setOperand(0, newLoadSrcTile.view);
    loadOp.setOperand(1, newLoadDstTile.view);

    auto newMemref = newLoadDstTile.root;
    auto toTensorSrc = toTensorOp.getMemref();
    if (auto subview = toTensorSrc.getDefiningOp<memref::SubViewOp>()) {
      TileView subviewTile{subview};
      TileView newSubviewTile =
          ViewPermutator(subviewTile, getLastTwoDimPermutation(
                                          subviewTile.viewType.getRank()))
              .permute(rewriter);
      TileView::unifyRoot(newSubviewTile, newLoadDstTile, rewriter);
      newMemref = newSubviewTile.view;
    }
    auto newToTensorOp = rewriter.create<bufferization::ToTensorOp>(
        toTensorOp.getLoc(), transposeOp->getResult(0).getType(), newMemref,
        toTensorOp.getRestrict(), toTensorOp.getWritable());
    rewriter.replaceOp(transposeOp, newToTensorOp.getResult());
    return success();
  }
};

struct FuseTransposeIntoLoadPass
    : public impl::FuseTransposeIntoLoadBase<FuseTransposeIntoLoadPass> {
  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *ctx = func.getContext();

    LDBG("First, compose subviews (unit-stride or static sizes)");
    {
      RewritePatternSet subviewPatterns(ctx);
      memref::populateComposeSubViewPatterns(subviewPatterns, ctx);
      populateComposeUnitStrideSubviewPatterns(subviewPatterns, ctx);
      if (failed(applyPatternsGreedily(func, std::move(subviewPatterns)))) {
        signalPassFailure();
        return;
      }
    }

    LDBG("Second, fuse transpose into load");
    RewritePatternSet patterns(ctx);
    patterns.add<FuseTransposeIntoLoadPattern>(ctx);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createFuseTransposeIntoLoadPass() {
  return std::make_unique<FuseTransposeIntoLoadPass>();
}
