//===-------------------- CombineOptimizedConvertLayout.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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
    : OpRewritePattern(context) {
  }

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
      return rewriter.notifyMatchFailure(
          op, "source layout is not ND");

    // Verify single use of to_tensor result
    if (!toTensorOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(
          op, "to_tensor result has multiple uses");

    rewriter.setInsertionPointAfter(loadOp);
    // Get the result tensor type (fractal shape)
    auto resultTensorType = cast<RankedTensorType>(op.getType());
    auto memrefDestType = MemRefType::get(resultTensorType.getShape(),
                                          resultTensorType.getElementType());
    auto allocOp = rewriter.create<memref::AllocOp>(
        op.getLoc(), memrefDestType);

    // Create ND2NZ op: fuses load + layout conversion
    // ins: source memref (reinterpret_cast, ND layout)
    // outs: destination memref (fractal layout)
    rewriter.create<ND2NZOp>(
        op.getLoc(),
        TypeRange(),
        loadOp.getSource(), // src (ins)
        allocOp.getResult(), // dst (outs)
        /*dst_continuous=*/rewriter.getUnitAttr()
        );

    auto newToTensorOp = rewriter.create<bufferization::ToTensorOp>(
        op.getLoc(), allocOp);
    newToTensorOp->setAttrs(toTensorOp->getAttrs());

    // Replace the convert_layout with ND2NZ result
    rewriter.replaceAllUsesWith(op, newToTensorOp);
    rewriter.eraseOp(loadOp);
    rewriter.eraseOp(op);
    rewriter.eraseOp(toTensorOp);
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
    : OpRewritePattern(context) {
  }

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    Value convertSrc = op.getSource();
    auto toTensorOp = convertSrc.getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensorOp)
      return rewriter.notifyMatchFailure(
          op, "source is not from a to_tensor operation");

    if (!toTensorOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(
          op, "to_tensor result has multiple uses");

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
      return rewriter.notifyMatchFailure(
          op, "no subview user found on alloc");

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
      return rewriter.notifyMatchFailure(
          op, "source layout is not ND");

    auto blockSizesOrFailure = dstLayout.getFractalBlockSizes();
    if (failed(blockSizesOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to extract block sizes from destination layout");

    // Derive from convert_layout result type (guaranteed to match).
    auto resultTensorType = cast<RankedTensorType>(op.getType());
    auto fractalAllocType = MemRefType::get(
        resultTensorType.getShape(), resultTensorType.getElementType());

    auto ndOffsets = subviewOut.getMixedOffsets();
    auto ndSizes = subviewOut.getMixedSizes();
    rewriter.setInsertionPointAfter(loadOp);

    auto fractalSizesOrFailure = computeMixedNDToFractalShape(
        ndSizes, srcLayout, dstLayout, rewriter, op.getLoc());
    if (failed(fractalSizesOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to compute fractal subview sizes");

    auto fractalOffsetsOrFailure = computeTargetLayoutOffset(ndOffsets, srcLayout, dstLayout, rewriter, op.getLoc());
    if (failed(fractalOffsetsOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to compute fractal subview offsets");

    SmallVector<OpFoldResult> fractalStrides(
        resultTensorType.getRank(), rewriter.getIndexAttr(1));

    auto newAllocOp = rewriter.create<memref::AllocOp>(
        origAllocOp.getLoc(), fractalAllocType);
    auto newSubviewOut = rewriter.create<memref::SubViewOp>(
        subviewOut.getLoc(),
        newAllocOp.getResult(),
        *fractalOffsetsOrFailure,
        *fractalSizesOrFailure,
        fractalStrides);

    //     Source (ins) is subview_in — unchanged, still in ND layout.
    //     Dest (outs) is the new fractal subview of the fractal alloc.
    Value loadSrc = loadOp.getSource(); // subview_in (ND layout, unchanged)
    rewriter.create<ND2NZOp>(
        op.getLoc(),
        TypeRange(),
        loadSrc,                // src (ins) — subview of reinterpret_cast
        newSubviewOut.getResult(),  // dst (outs) — fractal subview of alloc
        /*dst_continuous=*/rewriter.getUnitAttr());

    auto newToTensorOp = rewriter.create<bufferization::ToTensorOp>(
        toTensorOp.getLoc(), newAllocOp);
    newToTensorOp->setAttrs(toTensorOp->getAttrs());

    rewriter.replaceAllUsesWith(op.getResult(), newToTensorOp.getResult());

    rewriter.eraseOp(op);          // convert_layout (no users after replace)
    rewriter.eraseOp(toTensorOp);  // to_tensor (user was convert_layout)
    rewriter.eraseOp(loadOp);      // load (no SSA result / users erased)
    rewriter.eraseOp(subviewOut);  // subview_out (user was loadOp)
    rewriter.eraseOp(origAllocOp); // original ND alloc (all users erased)

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Fold Fixpipe + ConvertLayout into Enhanced Fixpipe
//
// Matches:
//   %fixpipe_result = hivm.hir.fixpipe ins(%src) outs(%dst) -> tensor<nZ>
//   %result = hivm.hir.convert_layout %fixpipe_result {srcLayout = nZ, dstLayout = ND}
//
// Transforms to:
//   %empty = tensor.empty() : tensor<ND_shape>
//   %result = hivm.hir.fixpipe ins(%src) outs(%empty) {dma_mode = NZ2ND} -> tensor<ND>
//===----------------------------------------------------------------------===//

struct FoldFixpipeConvertLayoutPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  FoldFixpipeConvertLayoutPattern(MLIRContext *context)
    : OpRewritePattern(context) {
  }

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
    auto srcLayout = op.getSrcLayout();

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
        resultTensorType, // Result type: ND shape
        fixpipeOp.getSrc(), // Same source
        emptyOp.getResult(), // New dst with ND shape (must match result)
        FixpipeDMAModeAttr::get(rewriter.getContext(), dmaMode),
        fixpipeOp.getDualDstModeAttr(),
        fixpipeOp.getPreQuantAttr(),
        fixpipeOp.getPreReluAttr(),
        fixpipeOp.getChannelSplitAttr());

    if (fixpipeOp.getUnitFlagMode())
      newFixpipe.setUnitFlagModeAttr(fixpipeOp.getUnitFlagModeAttr());
    rewriter.replaceOp(op, newFixpipe.getResults());
    rewriter.eraseOp(fixpipeOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Fold VCast + ConvertLayout into Fixpipe with PreQuant
//
// Matches:
//   %cast = hivm.hir.vcast ins(%src : tensor<nZ_shape_f32>)
//                          outs(%dst : tensor<nZ_shape_f16>) -> tensor<nZ_shape_f16>
//   %result = hivm.hir.convert_layout %cast {srcLayout = nZ, dstLayout = ND}
//
// Transforms to:
//   %empty = tensor.empty() : tensor<ND_shape_f16>
//   %result = hivm.hir.fixpipe ins(%src) outs(%empty)
//             {dma_mode = NZ2ND, pre_quant = F322F16} -> tensor<ND_f16>
//===----------------------------------------------------------------------===//

struct FoldVCastConvertLayoutPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  FoldVCastConvertLayoutPattern(MLIRContext *context)
      : OpRewritePattern(context) {}

  /// Determine the fixpipe pre_quant mode from VCast input/output types
  std::optional<FixpipePreQuantMode> getQuantModeFromVCast(VCastOp vcastOp) const {
    auto inputType = getElementTypeOrSelf(vcastOp.getSrc()[0].getType());
    auto outputType = getElementTypeOrSelf(vcastOp.getDst()[0].getType());

    if (inputType.isF32() && outputType.isF16())
      return FixpipePreQuantMode::F322F16;
    if (inputType.isF32() && outputType.isBF16())
      return FixpipePreQuantMode::F322BF16;
    if (inputType.isInteger(32) && outputType.isInteger(8))
      return FixpipePreQuantMode::S322I8;

    return std::nullopt;
  }

  /// Determine DMA mode from destination layout
  std::optional<FixpipeDMAMode> getDMAModeFromLayout(
      DataLayoutAttr dstLayout) const {
    auto layoutType = dstLayout.getDataLayout();

    if (layoutType == DataLayout::ND || layoutType == DataLayout::DOTA_ND) {
      if (dstLayout.getTranspose() && *dstLayout.getTransposeValue())
        return FixpipeDMAMode::NZ2DN;
      return FixpipeDMAMode::NZ2ND;
    }

    if (layoutType == DataLayout::DOTB_ND) {
      if (dstLayout.getTranspose() && *dstLayout.getTransposeValue())
        return FixpipeDMAMode::NZ2ND;
      return FixpipeDMAMode::NZ2DN;
    }

    return std::nullopt;
  }

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    // Check if source is from a VCastOp
    auto vcastOp = op.getSource().getDefiningOp<VCastOp>();
    if (!vcastOp)
      return rewriter.notifyMatchFailure(op, "source is not a VCastOp");

    // Verify single use of vcast result
    if (!vcastOp.getResult()[0].hasOneUse())
      return rewriter.notifyMatchFailure(op,
                                         "vcast result has multiple uses");

    // Verify source layout is nZ
    auto srcLayout = op.getSrcLayout();
    if (srcLayout.getDataLayout() != DataLayout::Fractal)
      return rewriter.notifyMatchFailure(op, "source layout is not fractal");

    // Get DMA mode from destination layout
    auto dstLayout = op.getDstLayout();
    auto dmaMode = getDMAModeFromLayout(dstLayout);
    if (!dmaMode)
      return rewriter.notifyMatchFailure(
          op, "destination layout is not supported for folding into fixpipe");

    // Get quantization mode from VCast types
    auto quantMode = getQuantModeFromVCast(vcastOp);
    if (!quantMode)
      return rewriter.notifyMatchFailure(
          op, "unsupported VCast type conversion for fixpipe pre_quant");

    // Get the result tensor type (ND shape)
    auto resultTensorType = cast<RankedTensorType>(op.getType());

    // Create an empty tensor for the fixpipe output
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultTensorType.getShape(),
        resultTensorType.getElementType());

    // Create fixpipe with pre_quant and DMA mode
    // Source is the input to vcast (e.g., f32 tensor)
    auto fixpipeOp = rewriter.create<FixpipeOp>(
        op.getLoc(),
        resultTensorType,
        vcastOp.getSrc()[0],  // Original source before vcast
        emptyOp.getResult(),
        FixpipeDMAModeAttr::get(rewriter.getContext(), *dmaMode),
        /*dual_dst_mode=*/FixpipeDualDstModeAttr{},
        FixpipePreQuantModeAttr::get(rewriter.getContext(), *quantMode),
        /*pre_relu=*/nullptr,
        /*channel_split=*/nullptr);

    LDBG("Folding VCast + ConvertLayout into Fixpipe with pre_quant");

    rewriter.replaceOp(op, fixpipeOp.getResults());
    rewriter.eraseOp(vcastOp);

    return success();
  }
};

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
    patterns.add<FoldToTensorConvertLayoutPattern>(context);
    patterns.add<FoldToTensorConvertLayoutSubviewPattern>(context);
    patterns.add<FoldFixpipeConvertLayoutPattern>(context);
    // patterns.add<FoldVCastConvertLayoutPattern>(context);
    ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);

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