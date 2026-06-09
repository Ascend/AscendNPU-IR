//===-------------------- CombineOptimizedConvertLayout.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h>

#define DEBUG_TYPE "hivm-combine-optimized-convert-layout"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_COMBINEOPTIMIZEDCONVERTLAYOUT
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace mlir::hivm {

namespace {

/// Tracks whether a producer value is only consumed by the fold candidate
/// `convertLayoutOp` plus optional `annotation.mark` ops.
struct ConvertLayoutProducerUseInfo {
  SmallVector<annotation::MarkOp> annotationUsers;
  bool canEraseProducerChain = true;
};

/// Walk users of `producer` and record annotation marks. Returns
/// `canEraseProducerChain = true` when every user is either
/// `convertLayoutOp` or an `annotation.mark` on `producer`.
ConvertLayoutProducerUseInfo
getConvertLayoutProducerUseInfo(Value producer,
                                ConvertLayoutOp convertLayoutOp) {
  ConvertLayoutProducerUseInfo info;
  for (Operation *user : producer.getUsers()) {
    if (user == convertLayoutOp.getOperation())
      continue;
    if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
      info.annotationUsers.push_back(markOp);
      continue;
    }
    info.canEraseProducerChain = false;
    break;
  }
  return info;
}

/// Repoint `annotation.mark` ops from `oldValue` to `newValue`.
void repointAnnotationMarks(PatternRewriter &rewriter,
                            ArrayRef<annotation::MarkOp> annotationUsers,
                            Value oldValue, Value newValue) {
  for (annotation::MarkOp markOp : annotationUsers) {
    rewriter.modifyOpInPlace(
        markOp, [&]() { markOp->replaceUsesOfWith(oldValue, newValue); });
  }
}

/// Create `hivm.hir.nd2nz` that fuses `loadOp`'s GM->L1 movement with ND to
/// Fractal layout conversion. Pass an empty `resultTypes` for memref outs; pass
/// the fractal tensor type when outs is a `tensor.empty`.
ND2NZOp createND2NZFromLoad(PatternRewriter &rewriter, Location loc,
                            LoadOp loadOp, Value src, Value dst,
                            TypeRange resultTypes) {
  bool hasInitOutBuffer = loadOp.getInitOutBuffer();
  return rewriter.create<ND2NZOp>(
      loc, resultTypes, src, dst, rewriter.getUnitAttr(), hasInitOutBuffer,
      hasInitOutBuffer ? loadOp.getPadValue() : Value{},
      loadOp.getInitCondition());
}

} // namespace

//===----------------------------------------------------------------------===//
// Pattern 1: Fold ToTensor + ConvertLayout into ND2NZ (Direct Load)
//
// This pattern targets the common case where a LoadOp reads data directly
// from a source memref (e.g., a reinterpret_cast of global memory) into a
// local allocation, which is then materialized as a tensor and undergoes
// layout conversion from ND to a fractal format (nZ or zN).
//
// By fusing the layout conversion into the data movement (replacing LoadOp
// with ND2NZOp), we eliminate the intermediate ND-layout buffer and the
// separate convert_layout step.
//
// Preconditions:
//   - convert_layout source comes from bufferization.to_tensor
//   - to_tensor wraps a memref (%alloc) with exactly two users:
//     1. The to_tensor op itself
//     2. A single LoadOp (using %alloc as its destination)
//   - The LoadOp source is a global memory reference (e.g., reinterpret_cast)
//   - The convert_layout srcLayout is ND
//   - The to_tensor result has exactly one use (the convert_layout)
//
// Input IR:
//   %reinterpret_cast = memref.reinterpret_cast %gm_buf ...
//       : memref<...> to memref<MxNxelem_type>
//   %alloc = memref.alloc() : memref<MxNxelem_type>
//   %load = hivm.hir.load
//       ins(%reinterpret_cast : memref<MxNxelem_type>)
//       outs(%alloc : memref<MxNxelem_type>)
//   %to_tensor = bufferization.to_tensor %alloc restrict writable
//       : memref<MxNxelem_type> -> tensor<MxNxelem_type>
//   %result = hivm.hir.convert_layout %to_tensor
//       {srcLayout = ND, dstLayout = nZ}
//       : tensor<MxNxelem_type> -> tensor<fractal_shape x elem_type>
//
// Output IR:
//   %reinterpret_cast = memref.reinterpret_cast %gm_buf ...
//       : memref<...> to memref<MxNxelem_type>
//   %alloc_fractal = memref.alloc() : memref<fractal_shape x elem_type>
//   %nd2nz = hivm.hir.nd2nz
//       ins(%reinterpret_cast : memref<MxNxelem_type>)
//       outs(%alloc_fractal : memref<fractal_shape x elem_type>)
//   %to_tensor = bufferization.to_tensor %alloc_fractal restrict writable
//       : memref<fractal_shape x elem_type>
//
// The convert_layout is eliminated. The allocation is reshaped to fractal
// layout, and the LoadOp is replaced with ND2NZOp which performs DMA data
// movement and layout conversion in a single fused operation.
//===----------------------------------------------------------------------===//

struct FoldToTensorConvertLayoutPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  FoldToTensorConvertLayoutPattern(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    // Get the source tensor of the convert_layout
    Value convertSrc = op.getSource();

    // Check if source comes from a to_tensor operation
    auto toTensorOp = convertSrc.getDefiningOp<bufferization::ToTensorOp>();

    if (!toTensorOp)
      return rewriter.notifyMatchFailure(
          op, "source is not from a to_tensor operation");

    auto toTensorMemref = toTensorOp.getMemref();
    int32_t userCount = 0;
    LoadOp loadOp = nullptr;
    for (auto user : toTensorMemref.getUsers()) {
      if (user == toTensorOp)
        continue;
      userCount++;
      if (isa<LoadOp>(user)) {
        loadOp = cast<LoadOp>(user);
        continue;
      }
      return rewriter.notifyMatchFailure(
          user, "Unwanted user of cbuf convert layout");
    }

    if (userCount > 1 || loadOp == nullptr) {
      LDBG(toTensorMemref);
      return rewriter.notifyMatchFailure(
          toTensorMemref.getDefiningOp(),
          "More than one user of to tensor memref");
    }

    // Verify this is ND -> fractal conversion
    auto srcLayout = op.getSrcLayout();

    if (srcLayout.getDataLayout() != DataLayout::ND)
      return rewriter.notifyMatchFailure(op, "source layout is not ND");

    // Multi-use to_tensor is allowed: fold only this convert_layout and keep
    // original ND path for other users.
    // Only erase original ND producer chain if no other users besides this
    // convert_layout and annotation.mark ops.
    ConvertLayoutProducerUseInfo useInfo =
        getConvertLayoutProducerUseInfo(toTensorOp.getResult(), op);

    rewriter.setInsertionPointAfter(loadOp);
    // Get the result tensor type (fractal shape)
    auto resultTensorType = cast<RankedTensorType>(op.getType());
    auto memrefDestType = MemRefType::get(resultTensorType.getShape(),
                                          resultTensorType.getElementType());
    auto allocOp =
        rewriter.create<memref::AllocOp>(op.getLoc(), memrefDestType);

    // Create ND2NZ op: fuses load + layout conversion
    createND2NZFromLoad(rewriter, op.getLoc(), loadOp, loadOp.getSource(),
                        allocOp.getResult(), TypeRange());

    auto newToTensorOp =
        rewriter.create<bufferization::ToTensorOp>(op.getLoc(), allocOp);
    newToTensorOp->setAttrs(toTensorOp->getAttrs());

    // Replace only this convert_layout.
    rewriter.replaceOp(op, newToTensorOp.getResult());

    // Single-use or annotation-only case: old ND chain is now dead, erase it.
    // Re-point any annotation.mark ops to the new fractal tensor first.
    // Multi-use case: keep load/to_tensor/memref alloc for other users.
    if (useInfo.canEraseProducerChain) {
      repointAnnotationMarks(rewriter, useInfo.annotationUsers,
                             toTensorOp.getResult(), newToTensorOp.getResult());
      auto oldAllocOp = toTensorMemref.getDefiningOp<memref::AllocOp>();
      rewriter.eraseOp(loadOp);
      rewriter.eraseOp(toTensorOp);
      if (oldAllocOp && oldAllocOp->use_empty())
        rewriter.eraseOp(oldAllocOp);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern 2: Fold ToTensor + ConvertLayout into ND2NZ (Subview Load)
//
// This pattern is a variant of Pattern 1 that handles the case where the
// LoadOp operates through subviews of the source and destination memrefs,
// rather than on the full memrefs directly. This pattern example can be found
// in HSTU models.
//
// The key transformation is:
//   - The destination alloc is reshaped from ND to fractal layout.
//   - The output subview (subview of alloc) is recomputed in fractal space:
//     outer-dim offsets are divided by block sizes, inner-dim offsets are 0,
//     and sizes are recomputed using ceil-div fractal tiling.
//   - The LoadOp is replaced with ND2NZOp (fused load + layout conversion).
//   - The to_tensor now wraps the fractal alloc directly.
//   - The convert_layout is removed entirely.
//
// Preconditions:
//   - convert_layout source comes from bufferization.to_tensor
//   - to_tensor wraps a memref (%alloc) with exactly two users:
//     1. The to_tensor op itself
//     2. A single memref.subview (%subview_out)
//   - %subview_out has exactly one user: a LoadOp (as destination)
//   - The LoadOp source (%subview_in) is a subview of a global memory ref
//   - The convert_layout srcLayout is ND
//   - The to_tensor result has exactly one use (the convert_layout)
//   - Subview offsets on %alloc must be tile-aligned to fractal block sizes
//
// Input IR:
//   %reinterpret_cast = memref.reinterpret_cast %gm_buf ...
//       : memref<...> to memref<MxNxelem_type>
//   %alloc = memref.alloc() : memref<MxNxelem_type>
//   %subview_in = memref.subview %reinterpret_cast [o0, o1] [s0, s1] [1, 1]
//       : memref<MxNxelem_type> to memref<s0 x s1 x elem_type>
//   %subview_out = memref.subview %alloc [o2, o3] [s2, s3] [1, 1]
//       : memref<MxNxelem_type> to memref<s2 x s3 x elem_type>
//   %load = hivm.hir.load
//       ins(%subview_in : memref<s0 x s1 x elem_type>)
//       outs(%subview_out : memref<s2 x s3 x elem_type>)
//   %to_tensor = bufferization.to_tensor %alloc restrict writable
//       : memref<MxNxelem_type> -> tensor<MxNxelem_type>
//   %result = hivm.hir.convert_layout %to_tensor
//       {srcLayout = ND, dstLayout = nZ}
//       : tensor<MxNxelem_type> -> tensor<fractal_shape x elem_type>
//
// Output IR:
//   %reinterpret_cast = memref.reinterpret_cast %gm_buf ...
//       : memref<...> to memref<MxNxelem_type>
//   %alloc_fractal = memref.alloc() : memref<fractal_shape x elem_type>
//   %subview_in = memref.subview %reinterpret_cast [o0, o1] [s0, s1] [1, 1]
//       : memref<MxNxelem_type> to memref<s0 x s1 x elem_type>
//   %subview_out_fractal = memref.subview %alloc_fractal
//       [o2/a0, o3/b0, 0, 0]                       // fractal offsets
//       [ceil(s2/a0), ceil(s3/b0), b0, a0]          // fractal sizes
//       [1, 1, 1, 1]                                // unit strides
//       : memref<fractal_shape x elem_type> to memref<fractal_tile x elem_type>
//   %nd2nz = hivm.hir.nd2nz
//       ins(%subview_in : memref<s0 x s1 x elem_type>)
//       outs(%subview_out_fractal : memref<fractal_tile x elem_type>)
//   %to_tensor = bufferization.to_tensor %alloc_fractal restrict writable
//       : memref<fractal_shape x elem_type>
//
// Note: %subview_in is UNCHANGED (still in ND layout on the source side).
// The ND2NZOp handles the layout conversion during the data movement.
// The fractal offsets/sizes shown above assume nZ layout; for zN, the
// dimension ordering differs (see FractalLayoutType enum).
//===----------------------------------------------------------------------===//
struct FoldToTensorConvertLayoutSubviewPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  FoldToTensorConvertLayoutSubviewPattern(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    Value convertSrc = op.getSource();
    auto toTensorOp = convertSrc.getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensorOp)
      return rewriter.notifyMatchFailure(
          op, "source is not from a to_tensor operation");

    // Multi-use to_tensor is allowed: fold only this convert_layout and keep
    // original ND path for other users.
    // Only erase original ND producer chain if no other users besides this
    // convert_layout and annotation.mark ops.
    ConvertLayoutProducerUseInfo useInfo =
        getConvertLayoutProducerUseInfo(toTensorOp.getResult(), op);

    Value allocMemref = toTensorOp.getMemref();
    auto origAllocOp = allocMemref.getDefiningOp<memref::AllocOp>();
    if (!origAllocOp)
      return rewriter.notifyMatchFailure(
          op, "to_tensor source is not a memref.alloc");

    memref::SubViewOp subviewOut = nullptr;
    for (auto user : allocMemref.getUsers()) {
      if (user == toTensorOp)
        continue;
      if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
        if (subviewOut)
          return rewriter.notifyMatchFailure(
              user, "multiple subview users of alloc not yet supported");
        subviewOut = sv;
        continue;
      }
      return rewriter.notifyMatchFailure(
          user, "unexpected non-subview user of alloc (use Pattern 1 for "
                "direct LoadOp)");
    }

    if (!subviewOut)
      return rewriter.notifyMatchFailure(op, "no subview user found on alloc");

    LoadOp loadOp = nullptr;
    for (auto user : subviewOut.getResult().getUsers()) {
      if (auto load = dyn_cast<LoadOp>(user)) {
        if (loadOp)
          return rewriter.notifyMatchFailure(
              user, "multiple LoadOp users of subview_out not supported");
        loadOp = load;
        continue;
      }
      return rewriter.notifyMatchFailure(
          user, "unexpected non-LoadOp user of subview_out");
    }

    if (!loadOp)
      return rewriter.notifyMatchFailure(
          op, "no LoadOp found using subview_out as destination");

    auto srcLayout = op.getSrcLayout();
    auto dstLayout = op.getDstLayout();

    if (srcLayout.getDataLayout() != DataLayout::ND)
      return rewriter.notifyMatchFailure(op, "source layout is not ND");

    auto blockSizesOrFailure = dstLayout.getFractalBlockSizes();
    if (failed(blockSizesOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to extract block sizes from destination layout");

    // Derive from convert_layout result type (guaranteed to match).
    auto resultTensorType = cast<RankedTensorType>(op.getType());
    auto fractalAllocType = MemRefType::get(resultTensorType.getShape(),
                                            resultTensorType.getElementType());

    auto ndOffsets = subviewOut.getMixedOffsets();
    auto ndSizes = subviewOut.getMixedSizes();
    rewriter.setInsertionPointAfter(loadOp);

    auto fractalSizesOrFailure = computeMixedNDToFractalShape(
        ndSizes, srcLayout, dstLayout, rewriter, op.getLoc());
    if (failed(fractalSizesOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to compute fractal subview sizes");

    auto fractalOffsetsOrFailure = computeTargetLayoutOffset(
        ndOffsets, srcLayout, dstLayout, rewriter, op.getLoc());
    if (failed(fractalOffsetsOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to compute fractal subview offsets");

    SmallVector<OpFoldResult> fractalStrides(resultTensorType.getRank(),
                                             rewriter.getIndexAttr(1));

    auto newAllocOp = rewriter.create<memref::AllocOp>(origAllocOp.getLoc(),
                                                       fractalAllocType);
    auto newSubviewOut = rewriter.create<memref::SubViewOp>(
        subviewOut.getLoc(), newAllocOp.getResult(), *fractalOffsetsOrFailure,
        *fractalSizesOrFailure, fractalStrides);

    //     Source (ins) is subview_in — unchanged, still in ND layout.
    //     Dest (outs) is the new fractal subview of the fractal alloc.
    Value loadSrc = loadOp.getSource(); // subview_in (ND layout, unchanged)
    createND2NZFromLoad(rewriter, op.getLoc(), loadOp, loadSrc,
                        newSubviewOut.getResult(), TypeRange());

    auto newToTensorOp = rewriter.create<bufferization::ToTensorOp>(
        toTensorOp.getLoc(), newAllocOp);
    newToTensorOp->setAttrs(toTensorOp->getAttrs());

    // Replace only this convert_layout.
    rewriter.replaceOp(op, newToTensorOp.getResult());

    // Single-use or annotation-only case: old ND chain is dead and can be
    // erased. Re-point any annotation.mark ops to the new fractal tensor first.
    // Multi-use case: preserve old chain for remaining users.
    if (useInfo.canEraseProducerChain) {
      repointAnnotationMarks(rewriter, useInfo.annotationUsers,
                             toTensorOp.getResult(), newToTensorOp.getResult());
      rewriter.eraseOp(loadOp);     // user of subview_out
      rewriter.eraseOp(subviewOut); // user of origAllocOp
      rewriter.eraseOp(toTensorOp); // user of origAllocOp
      if (origAllocOp->use_empty())
        rewriter.eraseOp(origAllocOp);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Fold Tensor Load + ConvertLayout into ND2NZ
//
// Fuses tensor loads, under the assumption that hivm.hir.load was loaded
// manually from global memory. (hivm.hir.store)
//
// Matches:
//   %load = hivm.hir.load ins(%src : tensor<MxN>) outs(%init : tensor<MxN>)
//             -> tensor<MxN>
//   %conv = hivm.hir.convert_layout %load {ND -> Fractal}
//
// Transforms to:
//   %empty = tensor.empty() : tensor<fractal_shape>
//   %result = hivm.hir.nd2nz {dst_continuous} ins(%src) outs(%empty)
//             -> tensor<fractal_shape>
//===----------------------------------------------------------------------===//

struct FoldTensorLoadConvertLayoutPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  FoldTensorLoadConvertLayoutPattern(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    auto loadOp = op.getSource().getDefiningOp<LoadOp>();
    if (!loadOp)
      return rewriter.notifyMatchFailure(op, "source is not a LoadOp");

    if (!loadOp.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(op, "load is not tensor-based");

    if (op.getSrcLayout().getDataLayout() != DataLayout::ND)
      return rewriter.notifyMatchFailure(op, "source layout is not ND");

    if (op.getDstLayout().getDataLayout() != DataLayout::Fractal)
      return rewriter.notifyMatchFailure(op,
                                         "destination layout is not Fractal");

    ConvertLayoutProducerUseInfo useInfo =
        getConvertLayoutProducerUseInfo(loadOp.getResult(0), op);

    auto resultTensorType = cast<RankedTensorType>(op.getType());
    rewriter.setInsertionPointAfter(loadOp);
    auto emptyOp =
        rewriter.create<tensor::EmptyOp>(op.getLoc(), op.getMixedOutputShape(),
                                         resultTensorType.getElementType());
    ND2NZOp nd2nzOp =
        createND2NZFromLoad(rewriter, op.getLoc(), loadOp, loadOp.getSrc(),
                            emptyOp.getResult(), TypeRange(resultTensorType));

    rewriter.replaceOp(op, nd2nzOp.getResult(0));
    if (useInfo.canEraseProducerChain) {
      repointAnnotationMarks(rewriter, useInfo.annotationUsers,
                             loadOp.getResult(0), nd2nzOp.getResult(0));
      rewriter.eraseOp(loadOp);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Fold ConvertLayout + Fixpipe
//
// fixpipe output is 2D (ND), but src comes from mmad output which is nZ
// (Fractal). fixpipe always uses dma_mode = NZ2ND to convert nZ→ND inline.
// The separate convert_layout{nZ→ND} can be eliminated by feeding the
// fractal tensor directly to fixpipe.
//
// Matches:
//   %conv = hivm.hir.convert_layout %mmad_output {Fractal -> ND}
//   hivm.hir.fixpipe {dma_mode = nz2nd} ins(%conv) outs(%dst_memref)
//
// Transforms to:
//   hivm.hir.fixpipe {dma_mode = nz2nd} ins(%mmad_output) outs(%dst_memref)
//===----------------------------------------------------------------------===//

struct FoldConvertLayoutFixpipePattern
    : public OpRewritePattern<ConvertLayoutOp> {
  FoldConvertLayoutFixpipePattern(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    auto srcLayout = op.getSrcLayout();
    auto dstLayout = op.getDstLayout();
    if (srcLayout.getDataLayout() != DataLayout::Fractal ||
        !dstLayout.isNDLayout())
      return rewriter.notifyMatchFailure(op, "not a Fractal->ND conversion");

    if (!op.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(op,
                                         "convert_layout has multiple uses");

    auto fixpipeOp = dyn_cast<FixpipeOp>(*op.getResult().user_begin());
    if (!fixpipeOp)
      return rewriter.notifyMatchFailure(op, "user is not a fixpipe");

    if (fixpipeOp.getDmaMode() != FixpipeDMAMode::NZ2ND &&
        fixpipeOp.getDmaMode() != FixpipeDMAMode::NZ2DN)
      return rewriter.notifyMatchFailure(
          fixpipeOp, "fixpipe dma_mode is not nz2nd or nz2dn");

    rewriter.modifyOpInPlace(
        fixpipeOp, [&]() { fixpipeOp.getSrcMutable().assign(op.getSource()); });

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Fold ConvertLayout + ExtractSlice + Fixpipe
//
// Same as FoldConvertLayoutFixpipePattern but with an extract_slice in
// between. The 2D (ND) slice offsets/sizes are converted to Fractal space
// so the extract_slice operates directly on the nZ tensor.
//
// Matches:
//   %conv = hivm.hir.convert_layout %mmad_output {Fractal -> ND}
//   %slice = tensor.extract_slice %conv [nd_offsets] [nd_sizes] [1, 1]
//   hivm.hir.fixpipe {dma_mode = nz2nd} ins(%slice) outs(%dst_memref)
//
// Transforms to:
//   %fractal_slice = tensor.extract_slice %mmad_output
//       [fractal_offsets] [fractal_sizes] [1, 1, 1, 1]
//   hivm.hir.fixpipe {dma_mode = nz2nd} ins(%fractal_slice) outs(%dst_memref)
//===----------------------------------------------------------------------===//

struct FoldConvertLayoutExtractSliceFixpipePattern
    : public OpRewritePattern<ConvertLayoutOp> {
  FoldConvertLayoutExtractSliceFixpipePattern(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    auto srcLayout = op.getSrcLayout();
    auto dstLayout = op.getDstLayout();
    if (srcLayout.getDataLayout() != DataLayout::Fractal ||
        !dstLayout.isNDLayout())
      return rewriter.notifyMatchFailure(op, "not a Fractal->ND conversion");

    if (!op.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(
          op, "convert_layout result has multiple uses");

    auto extractSliceOp =
        dyn_cast<tensor::ExtractSliceOp>(*op.getResult().user_begin());
    if (!extractSliceOp)
      return rewriter.notifyMatchFailure(op,
                                         "user is not a tensor.extract_slice");

    if (!extractSliceOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(
          extractSliceOp, "extract_slice result has multiple uses");

    auto fixpipeOp =
        dyn_cast<FixpipeOp>(*extractSliceOp.getResult().user_begin());
    if (!fixpipeOp)
      return rewriter.notifyMatchFailure(
          extractSliceOp, "user of extract_slice is not a fixpipe");

    if (fixpipeOp.getDmaMode() != FixpipeDMAMode::NZ2ND &&
        fixpipeOp.getDmaMode() != FixpipeDMAMode::NZ2DN)
      return rewriter.notifyMatchFailure(
          fixpipeOp, "fixpipe dma_mode is not nz2nd or nz2dn");

    for (OpFoldResult stride : extractSliceOp.getMixedStrides()) {
      std::optional<int64_t> strideVal = getConstantIntValue(stride);
      if (!strideVal || *strideVal != 1)
        return rewriter.notifyMatchFailure(
            extractSliceOp, "extract_slice has non-unit strides");
    }

    Location loc = extractSliceOp.getLoc();

    // Set insertion point before computing offsets/sizes so that any
    // affine.apply ops created by the helpers dominate the new extract_slice.
    rewriter.setInsertionPoint(extractSliceOp);

    // Convert ND offsets/sizes to Fractal offsets/sizes.
    // convert_layout is Fractal -> ND, so srcLayout = Fractal, dstLayout = ND.
    // The extract_slice operates on the ND tensor (op's result).
    // We convert from ND (dstLayout) to Fractal (srcLayout).
    auto newSizes = computeMixedTargetLayoutShape(
        extractSliceOp.getMixedSizes(), op.getDstLayout(), op.getSrcLayout(),
        rewriter, loc);
    if (failed(newSizes))
      return rewriter.notifyMatchFailure(
          op, "failed to compute fractal slice sizes");

    auto newOffsets = computeTargetLayoutOffset(
        extractSliceOp.getMixedOffsets(), op.getDstLayout(), op.getSrcLayout(),
        rewriter, loc);
    if (failed(newOffsets))
      return rewriter.notifyMatchFailure(
          op, "failed to compute fractal slice offsets");

    Value source = op.getSource();
    auto sourceType = cast<RankedTensorType>(source.getType());
    int64_t sourceRank = sourceType.getRank();
    SmallVector<OpFoldResult> newStrides(sourceRank, rewriter.getIndexAttr(1));

    auto newSliceType = RankedTensorType::get(
        decomposeMixedValues(*newSizes).first, sourceType.getElementType());

    auto newExtractSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, newSliceType, source, *newOffsets, *newSizes, newStrides);

    // Replace the fixpipe's src with the fractal slice, erase the ND ops
    rewriter.modifyOpInPlace(fixpipeOp, [&]() {
      fixpipeOp.getSrcMutable().assign(newExtractSlice.getResult());
    });

    rewriter.eraseOp(extractSliceOp);
    rewriter.eraseOp(op);
    return success();
  }
};
//
// Matches:
//   %fixpipe_result = hivm.hir.fixpipe ins(%src) outs(%dst) -> tensor<nZ>
//   %result = hivm.hir.convert_layout %fixpipe_result {srcLayout = nZ,
//   dstLayout = ND}
//
// Transforms to:
//   %empty = tensor.empty() : tensor<ND_shape>
//   %result = hivm.hir.fixpipe ins(%src) outs(%empty) {dma_mode = NZ2ND} ->
//   tensor<ND>
//===----------------------------------------------------------------------===//

struct FoldFixpipeConvertLayoutPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  FoldFixpipeConvertLayoutPattern(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    // Check if source is from a fixpipe
    auto fixpipeOp = op.getSource().getDefiningOp<FixpipeOp>();
    if (!fixpipeOp)
      return rewriter.notifyMatchFailure(op, "source is not a FixpipeOp");

    if (fixpipeOp.getDmaMode() != FixpipeDMAMode::NZ2NZ)
      return rewriter.notifyMatchFailure(fixpipeOp,
                                         "fixpipe already has a DMA mode set");

    // Verify this is NZ -> ND conversion
    auto dstLayout = op.getDstLayout();

    // if (srcLayout.getDataLayout() != DataLayout::nZ)
    //   return rewriter.notifyMatchFailure(op, "source layout is not nZ");

    // Determine the appropriate DMA mode based on destination layout
    FixpipeDMAMode dmaMode;
    if (dstLayout.getDataLayout() == DataLayout::ND ||
        dstLayout.getDataLayout() == DataLayout::DOTA_ND) {
      // Check if transpose is needed
      if (dstLayout.getTranspose() && *dstLayout.getTransposeValue())
        dmaMode = FixpipeDMAMode::NZ2DN;
      else
        dmaMode = FixpipeDMAMode::NZ2ND;
    } else if (dstLayout.getDataLayout() == DataLayout::DOTB_ND) {
      // DOTB typically needs transpose
      dmaMode = (dstLayout.getTranspose() && *dstLayout.getTransposeValue())
                    ? FixpipeDMAMode::NZ2ND
                    : FixpipeDMAMode::NZ2DN;
    } else {
      // DOTC is fine with NZ2ND
      dmaMode = FixpipeDMAMode::NZ2ND;
    }

    // Get the result tensor type (ND shape from convert_layout)
    auto resultTensorType = cast<RankedTensorType>(op.getType());
    auto ofrSize = tensor::reshape_utils::getMixedSizesOrOutputShape(
        rewriter, fixpipeOp.getSource());

    // Src should be the fractal
    assert(dstLayout.isNDLayout());
    auto mixedFractalShape = op.getMixedOutputShape();

    // Create an empty tensor for the new fixpipe output (ND shape)
    // This is required because fixpipe requires outs type == result type
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), mixedFractalShape, resultTensorType.getElementType());

    // Create new fixpipe with enhanced DMA mode
    auto newFixpipe = rewriter.create<FixpipeOp>(
        fixpipeOp.getLoc(),
        resultTensorType,    // Result type: ND shape
        fixpipeOp.getSrc(),  // Same source
        emptyOp.getResult(), // New dst with ND shape (must match result)
        FixpipeDMAModeAttr::get(rewriter.getContext(), dmaMode),
        fixpipeOp.getDualDstModeAttr(), fixpipeOp.getPreQuantAttr(),
        fixpipeOp.getPreReluAttr(), fixpipeOp.getChannelSplitAttr());

    if (fixpipeOp.getUnitFlagMode())
      newFixpipe.setUnitFlagModeAttr(fixpipeOp.getUnitFlagModeAttr());
    rewriter.replaceOp(op, newFixpipe.getResults());
    rewriter.eraseOp(fixpipeOp);
    return success();
  }
};

void populateCombineOptimizedConvertLayoutPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *context) {
  ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
  patterns.add<FoldToTensorConvertLayoutPattern>(context);
  patterns.add<FoldToTensorConvertLayoutSubviewPattern>(context);
  patterns.add<FoldFixpipeConvertLayoutPattern>(context);
  patterns.add<FoldConvertLayoutFixpipePattern>(context);
  patterns.add<FoldConvertLayoutExtractSliceFixpipePattern>(context);
  patterns.add<FoldTensorLoadConvertLayoutPattern>(context);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct CombineOptimizedConvertLayoutPass
    : public impl::CombineOptimizedConvertLayoutBase<
          CombineOptimizedConvertLayoutPass> {
  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    populateCombineOptimizedConvertLayoutPatterns(patterns, context);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (failed(applyPatternsGreedily(module, std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace mlir::hivm

std::unique_ptr<Pass> mlir::hivm::createCombineOptimizedConvertLayoutPass() {
  return std::make_unique<CombineOptimizedConvertLayoutPass>();
}