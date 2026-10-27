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
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/NDDMA/ComposeUnitStrideSubview.h"
#include "bishengir/Dialect/HIVM/Transforms/NDDMA/TileView.h"
#include "bishengir/Dialect/HIVM/Transforms/NDDMA/ViewPermutator.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

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

/// The load this transform fuses a transpose into.
///
/// A GM->UB load wears two forms in the pipeline: the Triton frontend emits
/// `memref.copy` into a fresh alloc, and `convert-to-hivm-op` rewrites that to
/// `hivm.hir.load` much later. This pass runs while only the copy exists - the
/// copy and the `linalg.transpose` coexist across the whole hfusion stage,
/// whereas the load and the transpose never coexist, because
/// `hfusion-auto-vectorize-v2` lowers the transpose to vector transfers long
/// before the load is formed. Both forms are accepted here: the operand
/// positions agree (0 = source, 1 = destination), and only the pad attributes
/// differ, which `memref.copy` simply does not carry.
struct FusableLoad {
  Operation *op = nullptr;

  explicit operator bool() const { return op != nullptr; }

  Value getSrc() const { return op->getOperand(0); }
  Value getDst() const { return op->getOperand(1); }
  void setSrc(Value v) const { op->setOperand(0, v); }
  void setDst(Value v) const { op->setOperand(1, v); }

  hivm::LoadOp asLoadOp() const { return dyn_cast<hivm::LoadOp>(op); }

  /// Whether a pad count rides on the innermost axis of this load.
  ///
  /// `hivm.hir.load` carries the counts as operands. `memref.copy` has none,
  /// but it still becomes a padded load: `convert-to-hivm-op` derives
  /// `left_padding_num` from the last-dim offset of the destination subview,
  /// modulo the elements per 32B block, so a window that starts partway into
  /// its innermost axis arrives with a non-zero count.
  ///
  /// The pad the frontend materializes itself is a different thing and is not
  /// reported here: a masked load becomes a fill over the whole buffer plus a
  /// copy into a valid window, which carries no count and survives any
  /// permutation, because buffer and window are permuted together.
  bool padsInnermostAxis() const {
    if (auto loadOp = asLoadOp()) {
      return loadOp.getLeftPaddingNum() || loadOp.getRightPaddingNum();
    }
    auto subviewOp = getDst().getDefiningOp<memref::SubViewOp>();
    if (!subviewOp) {
      return false;
    }
    return !isConstantIntValue(subviewOp.getMixedOffsets().back(), 0);
  }
};

/// Whether this copy is the GM->UB load a descriptor can carry.
///
/// `memref.copy` names no address space, and at this point in the pipeline the
/// operands carry none either - the destination is a plain `memref.alloc` and
/// the kernel arguments are not yet annotated - so each end has to be
/// established another way.
///
/// The source is structural: `isFromGMSpace` traces it back and accepts only
/// roots that are kernel arguments or pointer casts, which also rules out a
/// copy between two local buffers.
///
/// The destination cannot be read off the IR at all, so the kernel settles it.
/// In a vector kernel the only place GM data can land is UB. A kernel with a
/// cube half also moves GM into L1 for the matmul operands, and a transposed
/// L1 operand has to stay a transpose, because NDDMA only writes UB. Once
/// InferFuncCoreType has run the core type is authoritative; before that the
/// frontend's `mix_mode` is the signal available. An unknown core type is
/// refused rather than assumed.
static bool copyIsGMToUB(memref::CopyOp copyOp) {
  if (!utils::isFromGMSpace(copyOp.getSource())) {
    return false;
  }

  auto funcOp = copyOp->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return false;
  }

  if (std::optional<hivm::TFuncCoreType> coreType =
          hivm::queryFuncCoreType(funcOp)) {
    return *coreType == hivm::TFuncCoreType::AIV;
  }

  auto mixMode = funcOp->getAttrOfType<StringAttr>("mix_mode");
  return mixMode && mixMode.getValue() == "aiv";
}

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
LogicalResult inspectLoadUses(Value root, FusableLoad &load,
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

    if (isa<hivm::LoadOp, memref::CopyOp>(user)) {
      if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
        // Only the copy that fills this buffer is a load of it; one that
        // reads it is a store.
        if (!visited.contains(copyOp.getTarget()) || !copyIsGMToUB(copyOp)) {
          return failure();
        }
      }

      auto loadTile = TileView::fromMemRef(user->getOperand(1), builder);
      if (failed(loadTile) ||
          failed(verifyLoadTileCompatibleWithPermTile(*loadTile, permTile))) {
        return failure();
      }

      if (load && load.op != user) {
        // there should be only one load
        return failure();
      }

      load.op = user;
      continue;
    }

    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
      for (Value result : user->getResults()) {
        if (failed(inspectLoadUses(result, load, permTile, visited, builder))) {
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

FailureOr<FusableLoad> findLoadToFuse(Value root, const TileView &permTile,
                                      Builder &builder) {
  FusableLoad load;
  llvm::DenseSet<Value> visited;
  if (failed(inspectLoadUses(root, load, permTile, visited, builder)) ||
      !load || !!load.asLoadOp()) { // hivm::LoadOp is not supported yet
    return failure();
  }
  return load;
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

static bool loadCanPadLastDim(FusableLoad load) {
  auto loadOp = load.asLoadOp();
  if (!loadOp) {
    // memref.copy has no pad mode, so there is no last-dim fill to reason
    // about.
    return false;
  }
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
                                         FusableLoad load) {
  if (!loadCanPadLastDim(load))
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
    if (!isSupportedNDDMAPermutation(perm)) {
      // NDDMA drives loop0 .. loop4, so one descriptor expresses any
      // non-identity permutation up to rank kMaxNDDMARank.
      return rewriter.notifyMatchFailure(
          transposeOp, "expected a non-identity permutation NDDMA can drive");
    }
    const int64_t permRank = static_cast<int64_t>(perm.size());

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

    if (permTile->viewType.getRank() != permRank) {
      return rewriter.notifyMatchFailure(
          transposeOp, "transpose rank differs from the read tile rank");
    }

    if (!permTile->isContiguous()) {
      return rewriter.notifyMatchFailure(
          transposeOp, "expected contiguous layout to permute");
    }

    FailureOr<FusableLoad> loadToFuse =
        findLoadToFuse(permTile->root, *permTile, rewriter);
    if (failed(loadToFuse)) {
      return rewriter.notifyMatchFailure(
          transposeOp, "unsupported load to fuse with transpose");
    }

    FusableLoad load = *loadToFuse;
    // The pad counts address the dst last axis, and NDDMA applies them on
    // loop0 (see load_gm_to_ubuf_*_by_nddma). They keep their meaning only
    // while the permutation leaves that axis in place.
    if (load.padsInnermostAxis() && !permutationKeepsLastAxis(perm)) {
      return rewriter.notifyMatchFailure(
          transposeOp, "padded load needs a permutation keeping the last axis");
    }

    // the memory write must dominate the memory read.
    DominanceInfo dominance(transposeOp->getParentOfType<func::FuncOp>());
    if (!dominance.properlyDominates(load.op, toTensorOp.getOperation())) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "load does not dominate to_tensor");
    }

    auto loadSrcTile = TileView::fromMemRef(load.getSrc(), rewriter);
    auto loadDstTile = TileView::fromMemRef(load.getDst(), rewriter);
    if (failed(loadSrcTile) || failed(loadDstTile)) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "unsupported tile views on load");
    }
    LDBG("loadSrcTile: " << *loadSrcTile);
    LDBG("loadDstTile: " << *loadDstTile);

    if (loadSrcTile->viewType.getRank() != permRank ||
        loadDstTile->viewType.getRank() != permRank) {
      return rewriter.notifyMatchFailure(
          transposeOp, "transpose rank differs from the load tile ranks");
    }

    // The tile `to_tensor` reads may be a window of the dst alloc. It takes
    // the same permutation so that it keeps naming the same elements.
    std::optional<TileView> readTile;
    if (auto readSubview =
            toTensorOp.getMemref().getDefiningOp<memref::SubViewOp>()) {
      readTile.emplace(readSubview);
      if (readTile->viewType.getRank() != permRank) {
        return rewriter.notifyMatchFailure(
            transposeOp, "transpose rank differs from the read tile rank");
      }
    }

    rewriter.setInsertionPoint(load.op);

    ViewPermutator srcPermutator(*loadSrcTile, perm);
    ViewPermutator dstPermutator(*loadDstTile, perm);
    if (!srcPermutator.canPermute() || !dstPermutator.canPermute()) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "failed to permute load tiles");
    }

    // Check before the first mutation below: ViewPermutator::permute aborts
    // on a permutation it cannot materialize.
    std::optional<ViewPermutator> readPermutator;
    if (readTile) {
      readPermutator.emplace(*readTile, perm);
      if (!readPermutator->canPermute()) {
        return rewriter.notifyMatchFailure(transposeOp,
                                           "failed to permute the read tile");
      }
    }

    TileView newLoadSrcTile = srcPermutator.permute(rewriter);
    TileView newLoadDstTile = dstPermutator.permute(rewriter);

    load.setSrc(newLoadSrcTile.view);

    auto newMemref = newLoadDstTile.root;
    if (readPermutator) {
      TileView newSubviewTile = readPermutator->permute(rewriter);
      TileView::unifyRoot(newSubviewTile, newLoadDstTile, rewriter);
      newMemref = newSubviewTile.view;
    }

    const PadFillPolicy policy = decidePadFillPolicy(newLoadDstTile, load);
    if (policy == PadFillPolicy::ExpandAndTransfer)
      newLoadDstTile.expandLastKeptDimToRoot(rewriter);
    load.setDst(newLoadDstTile.view);

    transferAnnotationMarks(permTile->root, newLoadDstTile.root, rewriter);
    applyPadFillPolicy(policy, newLoadDstTile, permTile->root, rewriter);

    // The read has to follow the write. The insertion point is still at the
    // load, so creating the read here would place it before the very op that
    // fills the buffer; plan-memory then finds no live range for the alloc and
    // alloc-to-pointer-cast rejects it with "read before first write".
    // Verified for both forms of the load: swapping these two ops in otherwise
    // identical IR is on its own enough to fail the compile.
    rewriter.setInsertionPoint(toTensorOp);
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
