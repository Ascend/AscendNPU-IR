//===- FuseTransposeIntoLoad.cpp - Fuse transpose into hivm load ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fuse linalg.transpose into hivm.hir.load via DMA on-the-fly transpose by
// rewriting the load dest view to the transposed layout and dropping the
// explicit transpose. Pad fills move to the new dest when needed. Last-dim-only
// pad does not expand to rootLast; drop the fill when last kept-dim offset is
// 0, otherwise keep fill guarded by offset != 0. GLA prefix still expands the
// last dim when that offset is 0 so DMA pad covers it.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/ComposeSubView.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
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
///   ├─ linalg.fill outs(%root or alias)       // transferred to the new dest
///   └─ other users                            // return failure
LogicalResult inspectLoadUses(Value root, hivm::LoadOp &loadOp,
                              const TileView &permTile,
                              llvm::DenseSet<Value> &visited,
                              Builder &builder) {
  if (!visited.insert(root).second) {
    return success();
  }

  for (Operation *user : root.getUsers()) {

    if (isa<bufferization::ToTensorOp>(user) || isa<linalg::FillOp>(user)) {
      // Fills are re-emitted on the new dest by transferFillOps, or erased
      // when DMA last-dim pad already covers the dest.
      continue;
    }

    if (isa<annotation::MarkOp>(user)) {
      // Accept any annotation.mark on the alloc; they will be transferred
      // to the new alloc by transferAnnotationMarks.
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

/// Transfer annotation.mark ops from the old alloc to the new alloc created
/// by permuteRoot(). Only AllocOp roots are handled; reinterpret_cast roots
/// (which target block arguments) are left untouched.
static void transferAnnotationMarks(Value oldAlloc, Value newAlloc,
                                    PatternRewriter &rewriter) {
  if (!isa<memref::AllocOp>(oldAlloc.getDefiningOp()))
    return;
  SmallVector<annotation::MarkOp> marks;
  for (Operation *user : oldAlloc.getUsers()) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(user))
      marks.push_back(markOp);
  }
  if (marks.empty())
    return;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(newAlloc.getDefiningOp());
  for (auto markOp : marks) {
    auto *newMarkOp = rewriter.clone(*markOp.getOperation());
    newMarkOp->setOperand(0, newAlloc);
  }
  for (auto markOp : marks)
    rewriter.eraseOp(markOp);
}

/// Collect linalg fill-like ops that write `root` or a view of it.
static void collectFillOps(Value root, llvm::DenseSet<Value> &visited,
                           SmallVectorImpl<linalg::FillOp> &fills) {
  if (!visited.insert(root).second)
    return;
  for (Operation *user : root.getUsers()) {
    if (auto fillOp = dyn_cast<linalg::FillOp>(user)) {
      fills.push_back(fillOp);
      continue;
    }
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
      for (Value result : user->getResults())
        collectFillOps(result, visited, fills);
    }
  }
}

/// Immediate parent `scf.if` if it is a no-result then-only wrapper whose
/// then-block is just fill ops. ConvertToHIVMOp emits this around pad fills
/// (`hivm.unlikely_condition` when the DMA tile already covers the dest).
static scf::IfOp enclosingThenOnlyFillIf(linalg::FillOp fillOp) {
  auto ifOp = dyn_cast<scf::IfOp>(fillOp->getParentOp());
  if (!ifOp || ifOp.getNumResults() != 0 || !ifOp.getElseRegion().empty())
    return nullptr;
  Block *thenBlock = ifOp.thenBlock();
  if (!thenBlock)
    return nullptr;
  for (Operation &op : thenBlock->without_terminator()) {
    if (!isa<linalg::FillOp>(&op))
      return nullptr;
  }
  return ifOp;
}

static SmallVector<linalg::FillOp> collectFillsOnAlloc(Value alloc) {
  SmallVector<linalg::FillOp> fills;
  if (!isa_and_nonnull<memref::AllocOp>(alloc.getDefiningOp()))
    return fills;
  llvm::DenseSet<Value> visited;
  collectFillOps(alloc, visited, fills);
  return fills;
}

static void eraseFillOps(ArrayRef<linalg::FillOp> fills,
                         PatternRewriter &rewriter) {
  llvm::SmallPtrSet<Operation *, 4> maybeDeadIfs;
  for (linalg::FillOp fillOp : fills) {
    if (isa<scf::IfOp>(fillOp->getParentOp()))
      maybeDeadIfs.insert(fillOp->getParentOp());
    rewriter.eraseOp(fillOp);
  }
  for (Operation *op : maybeDeadIfs) {
    auto ifOp = cast<scf::IfOp>(op);
    Block *thenBlock = ifOp.thenBlock();
    if (thenBlock && thenBlock->without_terminator().empty() &&
        ifOp.getElseRegion().empty() && ifOp.getNumResults() == 0)
      rewriter.eraseOp(ifOp);
  }
}

/// Re-emit pad fills on the transposed dest alloc created by permuteRoot().
/// ConvertToHIVMOp folds a full-alloc vbrc into init_out_buffer and decompose
/// restores a fill on that alloc; this pass then allocates a new dest and
/// must not leave the fill behind. If the original fill sat in a then-only
/// scf.if, keep that guard so full tiles skip the fill; unconditional fill is
/// only the fallback when the condition does not dominate the new alloc.
/// `fillIfOffsetNonZero` ANDs `offset != 0` into the guard so last-dim pad
/// tiles whose last offset is 0 at runtime skip the !3107 vector fill.
static void transferFillOps(
    Value oldAlloc, Value newAlloc, PatternRewriter &rewriter,
    std::optional<OpFoldResult> fillIfOffsetNonZero = std::nullopt) {
  if (!isa_and_nonnull<memref::AllocOp>(newAlloc.getDefiningOp()))
    return;

  SmallVector<linalg::FillOp> fills = collectFillsOnAlloc(oldAlloc);
  if (fills.empty())
    return;

  linalg::FillOp fill = fills.front();
  Value pad = fill.getInputs().front();
  Location loc = fill.getLoc();
  Operation *newAllocOp = newAlloc.getDefiningOp();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(newAllocOp);

  Value cond;
  scf::IfOp oldIf = enclosingThenOnlyFillIf(fill);
  std::optional<DominanceInfo> dom;
  if (auto func = newAllocOp->getParentOfType<func::FuncOp>())
    dom.emplace(func);

  if (oldIf && dom && dom->dominates(oldIf.getCondition(), newAllocOp))
    cond = oldIf.getCondition();

  // Attribute / static non-zero: no runtime guard (same as dropping
  // Attribute OFRs via dyn_cast<Value>). Dynamic SSA offsets get `cmpi ne`.
  if (fillIfOffsetNonZero && dom) {
    if (auto offsetVal = dyn_cast<Value>(*fillIfOffsetNonZero)) {
      if (dom->dominates(offsetVal, newAllocOp)) {
        Value zero = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                     rewriter.getIndexAttr(0));
        Value ne = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                  offsetVal, zero);
        cond = cond ? rewriter.create<arith::AndIOp>(loc, cond, ne).getResult()
                    : ne;
      }
    }
  }

  if (cond) {
    auto newIf = rewriter.create<scf::IfOp>(
        oldIf ? oldIf.getLoc() : loc, cond, [&](OpBuilder &b, Location ifLoc) {
          b.create<linalg::FillOp>(loc, ValueRange{pad}, ValueRange{newAlloc});
          b.create<scf::YieldOp>(ifLoc);
        });
    if (oldIf)
      newIf->setAttrs(oldIf->getAttrs());
  } else {
    rewriter.create<linalg::FillOp>(loc, ValueRange{pad}, ValueRange{newAlloc});
  }

  eraseFillOps(fills, rewriter);
}

static bool loadCanPadLastDim(hivm::LoadOp loadOp) {
  auto padMode = loadOp.getPadMode();
  return padMode && padMode->getPadmode() != hivm::PadMode::PadNull;
}

enum class PadFillPolicy {
  EraseFill,             // last-dim-only, offset 0: no expand
  TransferFillAndOffset, // last-dim-only, dyn/nonzero: no expand
  TransferFill,          // GLA + nonzero, or no pad
  ExpandAndTransfer,     // GLA + offset 0: expand then transfer
};

static PadFillPolicy decidePadFillPolicy(const TileView &dst,
                                         hivm::LoadOp loadOp) {
  if (!loadCanPadLastDim(loadOp))
    return PadFillPolicy::TransferFill;
  const bool off0 = dst.lastKeptDimOffsetIsZero();
  if (dst.nonLastKeptDimsCoverRoot())
    return off0 ? PadFillPolicy::EraseFill
                : PadFillPolicy::TransferFillAndOffset;
  return off0 ? PadFillPolicy::ExpandAndTransfer : PadFillPolicy::TransferFill;
}

static void applyPadFillPolicy(PadFillPolicy policy, TileView &dst,
                               Value oldRoot, PatternRewriter &rewriter) {
  switch (policy) {
  case PadFillPolicy::EraseFill:
    eraseFillOps(collectFillsOnAlloc(oldRoot), rewriter);
    return;
  case PadFillPolicy::TransferFillAndOffset:
    transferFillOps(oldRoot, dst.root, rewriter, dst.lastKeptDimOffset());
    return;
  case PadFillPolicy::ExpandAndTransfer:
  case PadFillPolicy::TransferFill:
    transferFillOps(oldRoot, dst.root, rewriter);
    return;
  }
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

    const PadFillPolicy policy = decidePadFillPolicy(newLoadDstTile, loadOp);
    if (policy == PadFillPolicy::ExpandAndTransfer)
      newLoadDstTile.expandLastKeptDimToRoot(rewriter);
    loadOp.setOperand(1, newLoadDstTile.view);

    transferAnnotationMarks(permTile->root, newLoadDstTile.root, rewriter);
    applyPadFillPolicy(policy, newLoadDstTile, permTile->root, rewriter);

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
