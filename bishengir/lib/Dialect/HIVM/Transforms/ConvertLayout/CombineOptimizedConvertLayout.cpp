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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h>

#include "bishengir/Dialect/HIVM/Utils/Utils.h"

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

LogicalResult verifyLoadDominatesConvertLayout(LoadOp loadOp,
                                               ConvertLayoutOp op,
                                               PatternRewriter &rewriter) {
  Operation *loadOperation = loadOp.getOperation();
  Operation *convertLayoutOperation = op.getOperation();
  DominanceInfo dominance;
  if (!dominance.properlyDominates(loadOperation, convertLayoutOperation,
                                   /*enclosingOpOk=*/false))
    return rewriter.notifyMatchFailure(
        op, "load op does not dominate convert_layout");

  return success();
}

/// ND → Fractal matrix layout conversion can fuse into `hivm.hir.nd2nz`.
bool isMatrixND2NZConversion(ConvertLayoutOp op) {
  return op.getSrcLayout().getDataLayout() == DataLayout::ND &&
         op.getDstLayout().getDataLayout() == DataLayout::Fractal;
}

/// Non-transposed scale conversions that can fuse into `hivm.hir.load_scale`.
bool isScaleLoadMXConversion(ConvertLayoutOp op) {
  auto src = op.getSrcLayout().getDataLayout();
  auto dst = op.getDstLayout().getDataLayout();
  return (src == DataLayout::SCALEA_ND && dst == DataLayout::SCALEA_zZ) ||
         (src == DataLayout::SCALEB_DN && dst == DataLayout::SCALEB_nN);
}

LogicalResult verifyLoadMXScaleSource(Value src, PatternRewriter &rewriter,
                                      Operation *op) {
  auto shapedTy = dyn_cast<ShapedType>(src.getType());
  if (!shapedTy || shapedTy.getRank() < 1)
    return rewriter.notifyMatchFailure(op, "load_scale source is not shaped");

  int64_t lastDim = shapedTy.getDimSize(shapedTy.getRank() - 1);
  if (ShapedType::isDynamic(lastDim) || lastDim % 2 != 0)
    return rewriter.notifyMatchFailure(
        op, "load_scale requires a static last dim divisible by 2");

  if (isa<RankedTensorType>(src.getType()))
    return success();

  auto memrefTy = cast<MemRefType>(src.getType());
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(getStridesAndOffset(memrefTy, strides, offset)))
    return rewriter.notifyMatchFailure(
        op, "cannot determine strides for load_scale source");
  if (ShapedType::isDynamic(strides.back()) || strides.back() != 1)
    return rewriter.notifyMatchFailure(
        op, "load_scale requires a contiguous last dim (unit stride)");
  return success();
}

using CreateFusedMemrefDmaFn =
    llvm::function_ref<void(PatternRewriter &, Location, LoadOp, Value /*src*/,
                            Value /*dst*/)>;

using CreateFusedTensorDmaFn = llvm::function_ref<Value(
    PatternRewriter &, Location, LoadOp, Value /*empty*/,
    RankedTensorType /*resultTy*/)>;

void createND2NZMemrefDma(PatternRewriter &rewriter, Location loc,
                          LoadOp loadOp, Value src, Value dst) {
  createND2NZFromLoad(rewriter, loc, loadOp, src, dst, TypeRange());
}

Value createND2NZTensorDma(PatternRewriter &rewriter, Location loc,
                           LoadOp loadOp, Value empty,
                           RankedTensorType resultTy) {
  return createND2NZFromLoad(rewriter, loc, loadOp, loadOp.getSrc(), empty,
                             TypeRange(resultTy))
      .getResult(0);
}

void createLoadMXScaleMemrefDma(PatternRewriter &rewriter, Location loc,
                                LoadOp /*loadOp*/, Value src, Value dst) {
  rewriter.create<LoadMXScaleOp>(loc, TypeRange(), src, dst,
                                 /*isTransposed=*/false);
}

Value createLoadMXScaleTensorDma(PatternRewriter &rewriter, Location loc,
                                 LoadOp loadOp, Value empty,
                                 RankedTensorType resultTy) {
  return rewriter
      .create<LoadMXScaleOp>(loc, TypeRange(resultTy), loadOp.getSrc(), empty,
                             /*isTransposed=*/false)
      .getResult(0);
}

struct DirectLoadConvertMatch {
  bufferization::ToTensorOp toTensorOp;
  LoadOp loadOp;
  ConvertLayoutProducerUseInfo useInfo;
};

FailureOr<DirectLoadConvertMatch>
matchDirectLoadConvertLayout(ConvertLayoutOp op, PatternRewriter &rewriter) {
  auto toTensorOp =
      op.getSource().getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensorOp)
    return rewriter.notifyMatchFailure(
        op, "source is not from a to_tensor operation");

  Value toTensorMemref = toTensorOp.getMemref();
  int32_t userCount = 0;
  LoadOp loadOp = nullptr;
  for (Operation *user : toTensorMemref.getUsers()) {
    if (user == toTensorOp)
      continue;
    userCount++;
    if (isa<LoadOp>(user)) {
      loadOp = cast<LoadOp>(user);
      continue;
    }
    return rewriter.notifyMatchFailure(
        user, "Unwanted user of convert_layout to_tensor memref");
  }
  if (userCount > 1 || !loadOp)
    return rewriter.notifyMatchFailure(
        toTensorMemref.getDefiningOp(),
        "More than one user of to_tensor memref");

  if (failed(verifyLoadDominatesConvertLayout(loadOp, op, rewriter)))
    return failure();

  return DirectLoadConvertMatch{
      toTensorOp, loadOp,
      getConvertLayoutProducerUseInfo(toTensorOp.getResult(), op)};
}

LogicalResult
rewriteDirectLoadConvertLayout(ConvertLayoutOp op, PatternRewriter &rewriter,
                               DirectLoadConvertMatch match,
                               CreateFusedMemrefDmaFn createFusedDma) {
  auto resultTensorType = cast<RankedTensorType>(op.getType());
  auto memrefDestType = MemRefType::get(resultTensorType.getShape(),
                                        resultTensorType.getElementType());
  rewriter.setInsertionPointAfter(match.loadOp);
  auto allocOp = rewriter.create<memref::AllocOp>(op.getLoc(), memrefDestType);
  createFusedDma(rewriter, op.getLoc(), match.loadOp, match.loadOp.getSource(),
                 allocOp.getResult());

  auto newToTensorOp =
      rewriter.create<bufferization::ToTensorOp>(op.getLoc(), allocOp);
  newToTensorOp->setAttrs(match.toTensorOp->getAttrs());
  rewriter.replaceOp(op, newToTensorOp.getResult());

  if (match.useInfo.canEraseProducerChain) {
    repointAnnotationMarks(rewriter, match.useInfo.annotationUsers,
                           match.toTensorOp.getResult(),
                           newToTensorOp.getResult());
    Value oldMemref = match.toTensorOp.getMemref();
    auto oldAllocOp = oldMemref.getDefiningOp<memref::AllocOp>();
    rewriter.eraseOp(match.loadOp);
    rewriter.eraseOp(match.toTensorOp);
    if (oldAllocOp && oldAllocOp->use_empty())
      rewriter.eraseOp(oldAllocOp);
  }
  return success();
}

struct SubviewLoadConvertMatch {
  bufferization::ToTensorOp toTensorOp;
  memref::AllocOp origAllocOp;
  memref::SubViewOp subviewOut;
  LoadOp loadOp;
  ConvertLayoutProducerUseInfo useInfo;
};

FailureOr<SubviewLoadConvertMatch>
matchSubviewLoadConvertLayout(ConvertLayoutOp op, PatternRewriter &rewriter) {
  auto toTensorOp =
      op.getSource().getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensorOp)
    return rewriter.notifyMatchFailure(
        op, "source is not from a to_tensor operation");

  ConvertLayoutProducerUseInfo useInfo =
      getConvertLayoutProducerUseInfo(toTensorOp.getResult(), op);

  Value allocMemref = toTensorOp.getMemref();
  auto origAllocOp = allocMemref.getDefiningOp<memref::AllocOp>();
  if (!origAllocOp)
    return rewriter.notifyMatchFailure(
        op, "to_tensor source is not a memref.alloc");

  memref::SubViewOp subviewOut = nullptr;
  for (Operation *user : allocMemref.getUsers()) {
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
        user, "unexpected non-subview user of alloc");
  }
  if (!subviewOut)
    return rewriter.notifyMatchFailure(op, "no subview user found on alloc");

  LoadOp loadOp = nullptr;
  for (Operation *user : subviewOut.getResult().getUsers()) {
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
  if (failed(verifyLoadDominatesConvertLayout(loadOp, op, rewriter)))
    return failure();

  return SubviewLoadConvertMatch{toTensorOp, origAllocOp, subviewOut, loadOp,
                                 useInfo};
}

LogicalResult
rewriteSubviewLoadConvertLayout(ConvertLayoutOp op, PatternRewriter &rewriter,
                                SubviewLoadConvertMatch match,
                                CreateFusedMemrefDmaFn createFusedDma) {
  auto srcLayout = op.getSrcLayout();
  auto dstLayout = op.getDstLayout();
  auto resultTensorType = cast<RankedTensorType>(op.getType());
  auto fractalAllocType = MemRefType::get(resultTensorType.getShape(),
                                          resultTensorType.getElementType());

  auto ndOffsets = match.subviewOut.getMixedOffsets();
  auto ndSizes = match.subviewOut.getMixedSizes();
  rewriter.setInsertionPointAfter(match.loadOp);

  auto fractalSizesOrFailure = computeMixedTargetLayoutShape(
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
  auto newAllocOp = rewriter.create<memref::AllocOp>(
      match.origAllocOp.getLoc(), fractalAllocType);
  auto newSubviewOut = rewriter.create<memref::SubViewOp>(
      match.subviewOut.getLoc(), newAllocOp.getResult(),
      *fractalOffsetsOrFailure, *fractalSizesOrFailure, fractalStrides);

  createFusedDma(rewriter, op.getLoc(), match.loadOp, match.loadOp.getSource(),
                 newSubviewOut.getResult());

  auto newToTensorOp = rewriter.create<bufferization::ToTensorOp>(
      match.toTensorOp.getLoc(), newAllocOp);
  newToTensorOp->setAttrs(match.toTensorOp->getAttrs());
  rewriter.replaceOp(op, newToTensorOp.getResult());

  if (match.useInfo.canEraseProducerChain) {
    repointAnnotationMarks(rewriter, match.useInfo.annotationUsers,
                           match.toTensorOp.getResult(),
                           newToTensorOp.getResult());
    rewriter.eraseOp(match.loadOp);
    rewriter.eraseOp(match.subviewOut);
    rewriter.eraseOp(match.toTensorOp);
    if (match.origAllocOp->use_empty())
      rewriter.eraseOp(match.origAllocOp);
  }
  return success();
}

LogicalResult
rewriteTensorLoadConvertLayout(ConvertLayoutOp op, PatternRewriter &rewriter,
                               CreateFusedTensorDmaFn createFusedDma) {
  auto loadOp = op.getSource().getDefiningOp<LoadOp>();
  if (!loadOp)
    return rewriter.notifyMatchFailure(op, "source is not a LoadOp");
  if (!loadOp.hasPureTensorSemantics())
    return rewriter.notifyMatchFailure(op, "load is not tensor-based");

  ConvertLayoutProducerUseInfo useInfo =
      getConvertLayoutProducerUseInfo(loadOp.getResult(0), op);
  auto resultTensorType = cast<RankedTensorType>(op.getType());
  rewriter.setInsertionPointAfter(loadOp);
  auto emptyOp =
      rewriter.create<tensor::EmptyOp>(op.getLoc(), op.getMixedOutputShape(),
                                       resultTensorType.getElementType());

  Value fusedResult = createFusedDma(rewriter, op.getLoc(), loadOp,
                                     emptyOp.getResult(), resultTensorType);

  rewriter.replaceOp(op, fusedResult);
  if (useInfo.canEraseProducerChain) {
    repointAnnotationMarks(rewriter, useInfo.annotationUsers,
                           loadOp.getResult(0), fusedResult);
    rewriter.eraseOp(loadOp);
  }
  return success();
}

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

struct FoldDirectLoadToND2NZPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    if (!isMatrixND2NZConversion(op))
      return rewriter.notifyMatchFailure(op, "not an ND→Fractal conversion");
    auto match = matchDirectLoadConvertLayout(op, rewriter);
    if (failed(match))
      return failure();
    return rewriteDirectLoadConvertLayout(op, rewriter, *match,
                                          createND2NZMemrefDma);
  }
};

struct FoldDirectLoadToLoadMXScalePattern
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    if (!isScaleLoadMXConversion(op))
      return rewriter.notifyMatchFailure(
          op, "not a SCALEA_ND→SCALEA_zZ / SCALEB_DN→SCALEB_nN conversion");
    auto match = matchDirectLoadConvertLayout(op, rewriter);
    if (failed(match))
      return failure();
    if (failed(verifyLoadMXScaleSource(match->loadOp.getSource(), rewriter, op)))
      return failure();
    return rewriteDirectLoadConvertLayout(op, rewriter, *match,
                                          createLoadMXScaleMemrefDma);
  }
};

struct FoldSubviewLoadToND2NZPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    if (!isMatrixND2NZConversion(op))
      return rewriter.notifyMatchFailure(op, "not an ND→Fractal conversion");
    auto match = matchSubviewLoadConvertLayout(op, rewriter);
    if (failed(match))
      return failure();
    return rewriteSubviewLoadConvertLayout(op, rewriter, *match,
                                           createND2NZMemrefDma);
  }
};

struct FoldSubviewLoadToLoadMXScalePattern
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    if (!isScaleLoadMXConversion(op))
      return rewriter.notifyMatchFailure(
          op, "not a SCALEA_ND→SCALEA_zZ / SCALEB_DN→SCALEB_nN conversion");
    auto match = matchSubviewLoadConvertLayout(op, rewriter);
    if (failed(match))
      return failure();
    if (failed(verifyLoadMXScaleSource(match->loadOp.getSource(), rewriter, op)))
      return failure();
    return rewriteSubviewLoadConvertLayout(op, rewriter, *match,
                                           createLoadMXScaleMemrefDma);
  }
};

struct FoldTensorLoadToND2NZPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    if (!isMatrixND2NZConversion(op))
      return rewriter.notifyMatchFailure(op, "not an ND→Fractal conversion");
    return rewriteTensorLoadConvertLayout(op, rewriter, createND2NZTensorDma);
  }
};

struct FoldTensorLoadToLoadMXScalePattern
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    if (!isScaleLoadMXConversion(op))
      return rewriter.notifyMatchFailure(
          op, "not a SCALEA_ND→SCALEA_zZ / SCALEB_DN→SCALEB_nN conversion");
    auto loadOp = op.getSource().getDefiningOp<LoadOp>();
    if (!loadOp)
      return rewriter.notifyMatchFailure(op, "source is not a LoadOp");
    if (failed(verifyLoadMXScaleSource(loadOp.getSrc(), rewriter, op)))
      return failure();
    return rewriteTensorLoadConvertLayout(op, rewriter,
                                          createLoadMXScaleTensorDma);
  }
};

//===----------------------------------------------------------------------===//
// Fold ConvertLayout + Fixpipe
//
// fixpipe output is 2D (ND), but src comes from mmad output which is nZ
// (Fractal). fixpipe always uses dma_mode = NZ2ND to convert nZ→ND inline.
// The separate convert_layout{nZ→ND} can be bypassed by feeding the fractal
// tensor directly to each compatible fixpipe. If it has no other users, the
// convert_layout is eliminated.
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

    bool changed = false;
    for (Operation *user :
        llvm::make_early_inc_range(op.getResult().getUsers())) {
      auto fixpipeOp = dyn_cast<FixpipeOp>(user);
      if (!fixpipeOp)
        continue;

      if (fixpipeOp.getDmaMode() != FixpipeDMAMode::NZ2ND &&
          fixpipeOp.getDmaMode() != FixpipeDMAMode::NZ2DN &&
          fixpipeOp.getDmaMode() != FixpipeDMAMode::NZ2NZ)
        continue;

      rewriter.modifyOpInPlace(
          fixpipeOp, [&]() { fixpipeOp.getSrcMutable().assign(op.getSource()); });
      changed = true;
    }

    if (!changed)
      return rewriter.notifyMatchFailure(
          op, "no compatible fixpipe user to fold");

    if (op.getResult().use_empty())
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

//===----------------------------------------------------------------------===//
// Fold Fixpipe(NZ2NZ) + convert_layout(→Fractal)
//
// A rank-2 NZ2NZ Fixpipe result is already physically fractal.
// InsertConvertLayout may insert convert_layout(Fractal→Fractal) (or a leftover
// ND→Fractal) only to materialize the rank-4 SSA type for the consumer mmad.
// Fold that convert into the Fixpipe by making the Fixpipe produce the fractal
// tensor directly.
//
// Matches:
//   %fix = hivm.hir.fixpipe {NZ2NZ} ... -> tensor<MxN>
//   %fr  = hivm.hir.convert_layout %fix {src=Fractal|ND, dst=Fractal}
//       -> tensor<4D>
//
// Transforms to:
//   %fr = hivm.hir.fixpipe {NZ2NZ} ... -> tensor<4D>
//===----------------------------------------------------------------------===//

struct FoldFixpipeNz2NzToFractalConvertLayoutPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  FoldFixpipeNz2NzToFractalConvertLayoutPattern(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    auto fixpipeOp = op.getSource().getDefiningOp<FixpipeOp>();
    if (!fixpipeOp)
      return rewriter.notifyMatchFailure(op, "source is not a FixpipeOp");

    if (fixpipeOp.getDmaMode() != FixpipeDMAMode::NZ2NZ)
      return rewriter.notifyMatchFailure(fixpipeOp, "fixpipe is not NZ2NZ");

    if (!fixpipeOp.getResultTensor())
      return rewriter.notifyMatchFailure(fixpipeOp,
                                         "fixpipe has no tensor result");

    auto dstLayout = op.getDstLayout();
    if (dstLayout.getDataLayout() != DataLayout::Fractal)
      return rewriter.notifyMatchFailure(op, "dst layout is not Fractal");

    auto srcLayout = op.getSrcLayout();
    if (srcLayout.getDataLayout() != DataLayout::Fractal &&
        !srcLayout.isNDLayout())
      return rewriter.notifyMatchFailure(
          op, "src layout is neither Fractal nor ND");

    auto resultTensorType = cast<RankedTensorType>(op.getType());
    if (resultTensorType.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "convert result is not rank-4");

    auto fixpipeType =
        dyn_cast<RankedTensorType>(fixpipeOp.getResultTensor().getType());
    if (!fixpipeType || fixpipeType.getRank() == 4)
      return rewriter.notifyMatchFailure(
          fixpipeOp, "fixpipe result is already rank-4 or not a ranked tensor");

    // Only fold when the convert is the sole real user of the Fixpipe result
    // (aside from the convert itself being fed by it).
    if (!op.getSource().hasOneUse())
      return rewriter.notifyMatchFailure(
          fixpipeOp, "fixpipe result has multiple uses; cannot retarget");

    auto mixedFractalShape = op.getMixedOutputShape();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), mixedFractalShape, resultTensorType.getElementType());

    auto newFixpipe = rewriter.create<FixpipeOp>(
        fixpipeOp.getLoc(), resultTensorType, fixpipeOp.getSrc(),
        emptyOp.getResult(), fixpipeOp.getDmaModeAttr(),
        fixpipeOp.getDualDstModeAttr(), fixpipeOp.getSubBlockIdxAttr(),
        fixpipeOp.getPreQuantAttr(), fixpipeOp.getPreReluAttr(),
        fixpipeOp.getChannelSplitAttr(), fixpipeOp.getC0PadEnAttr(),
        fixpipeOp.getQuantScale());
    if (fixpipeOp.getUnitFlagMode())
      newFixpipe.setUnitFlagModeAttr(fixpipeOp.getUnitFlagModeAttr());

    rewriter.replaceOp(op, newFixpipe.getResultTensor());
    rewriter.eraseOp(fixpipeOp);
    return success();
  }
};

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

    // Verify this is NZ -> ND conversion. Fractal destinations are handled by
    // FoldFixpipeNz2NzToFractalConvertLayoutPattern.
    auto dstLayout = op.getDstLayout();
    if (!dstLayout.isNDLayout())
      return rewriter.notifyMatchFailure(op, "not an NZ->ND conversion");

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
        fixpipeOp.getDualDstModeAttr(), fixpipeOp.getSubBlockIdxAttr(),
        fixpipeOp.getPreQuantAttr(), fixpipeOp.getPreReluAttr(),
        fixpipeOp.getChannelSplitAttr(), fixpipeOp.getC0PadEnAttr());

    if (fixpipeOp.getUnitFlagMode())
      newFixpipe.setUnitFlagModeAttr(fixpipeOp.getUnitFlagModeAttr());
    rewriter.replaceOp(op, newFixpipe.getResults());
    rewriter.eraseOp(fixpipeOp);
    return success();
  }
};

// Fold ND-to-Fractal convert_layout through GM workspace for K-padded vectors.
//
// A convert_layout(ND→Fractal) whose result is copied into an L1 (cbuf)
// buffer for the cube cannot be fractalized on-chip when the ND source is
// K-padded: the tensor decomposition needs a UB transpose the memory planner
// cannot fit, and an in-L1 pad copy would be an illegal L1↔L1 DMA. Instead,
// spill the unpadded ND tensor to a GM workspace and reload with nd2nz, which
// both pads and fractalizes during the GM→L1 DMA, removing the ND pad buffer
// chain entirely.
struct RouteVectorFractalizeViaGMPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSrcLayout().getDataLayout() != DataLayout::ND ||
        op.getDstLayout().getDataLayout() != DataLayout::Fractal)
      return rewriter.notifyMatchFailure(op, "not ND -> Fractal");

    auto srcTy = dyn_cast<RankedTensorType>(op.getSource().getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!srcTy || !dstTy || srcTy.getRank() != 2 || dstTy.getRank() != 4 ||
        !srcTy.hasStaticShape() || !dstTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static 2D -> 4D");

    if (srcTy.getShape()[0] != dstTy.getShape()[1] * dstTy.getShape()[2] ||
        srcTy.getShape()[1] != dstTy.getShape()[0] * dstTy.getShape()[3])
      return rewriter.notifyMatchFailure(op, "not a zN dot fractal");

    if (!op.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(op, "convert result has many uses");
    auto copyOp =
        dyn_cast<CopyOp>(*op.getResult().getUsers().begin());
    if (!copyOp || copyOp.getSrc() != op.getResult())
      return rewriter.notifyMatchFailure(op, "result not consumed by a copy");
    if (!isa<MemRefType>(copyOp.getDst().getType()))
      return rewriter.notifyMatchFailure(op, "copy target is not a memref");
    auto l1Space = dyn_cast_or_null<AddressSpaceAttr>(
        cast<MemRefType>(copyOp.getDst().getType()).getMemorySpace());
    if (!l1Space || l1Space.getAddressSpace() != AddressSpace::L1)
      return rewriter.notifyMatchFailure(op, "copy target is not L1");

    auto toTensor = op.getSource().getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensor)
      return rewriter.notifyMatchFailure(op, "source is not a K-padded buffer");
    Value padAlloc = toTensor.getMemref();
    VBrcOp vbrc;
    CopyOp padCopy;
    memref::SubViewOp subView;
    bool ok = padAlloc.getDefiningOp<memref::AllocOp>() != nullptr;
    for (Operation *user : padAlloc.getUsers()) {
      if (user == toTensor)
        continue;
      if (auto b = dyn_cast<VBrcOp>(user))
        vbrc = b;
      else if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
        subView = sv;
        for (Operation *svUser : sv.getResult().getUsers())
          if (auto c = dyn_cast<CopyOp>(svUser))
            padCopy = c;
      } else
        ok = false;
    }
    auto toMemref =
        (ok && vbrc && padCopy && subView)
            ? padCopy.getSrc().getDefiningOp<bufferization::ToMemrefOp>()
            : nullptr;
    if (!toMemref)
      return rewriter.notifyMatchFailure(op, "no K-pad chain to spill");
    Value storeSrc = toMemref.getTensor();
    ArrayRef<int64_t> storeShape =
        cast<RankedTensorType>(storeSrc.getType()).getShape();
    Value padValue = vbrc.getSrc();
    // Collect the K-pad chain for erasure, ordered users-before-defs.
    SmallVector<Operation *> padOpsToErase{
        padCopy.getOperation(), toMemref.getOperation(), vbrc.getOperation(),
        subView.getOperation(), toTensor.getOperation(),
        padAlloc.getDefiningOp()};

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(copyOp);
    Value gm = createAllocLocalWorkSpace(rewriter, loc, storeShape,
                                         srcTy.getElementType());
    rewriter.create<StoreOp>(loc, TypeRange{}, storeSrc, gm);
    rewriter.create<ND2NZOp>(loc, TypeRange{}, gm, copyOp.getDst(),
                             rewriter.getUnitAttr(),
                             /*init_out_buffer=*/padValue != Value(),
                             /*pad_value=*/padValue);
    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(op);
    // Erase the now-dead ND pad chain (uses before defs).
    for (Operation *dead : padOpsToErase)
      if (dead && dead->use_empty())
        rewriter.eraseOp(dead);
    return success();
  }
};

void populateCombineOptimizedConvertLayoutPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *context) {
  ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
  patterns
      .add<FoldDirectLoadToND2NZPattern, FoldDirectLoadToLoadMXScalePattern,
           FoldSubviewLoadToND2NZPattern, FoldSubviewLoadToLoadMXScalePattern,
           FoldTensorLoadToND2NZPattern, FoldTensorLoadToLoadMXScalePattern,
           FoldFixpipeNz2NzToFractalConvertLayoutPattern,
           FoldFixpipeConvertLayoutPattern, FoldConvertLayoutFixpipePattern,
           FoldConvertLayoutExtractSliceFixpipePattern,
           RouteVectorFractalizeViaGMPattern>(context);
}

} // namespace

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
