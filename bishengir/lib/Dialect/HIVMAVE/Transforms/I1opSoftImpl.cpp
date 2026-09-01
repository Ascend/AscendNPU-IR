//===------------- I1opSoftImpl.cpp - soft impl i1 op ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

#define DEBUG_TYPE "ave-i1op-soft-impl"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_I1OPSOFTIMPL
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivmave;

static constexpr llvm::StringLiteral i1ProcessedAttr = "1xi1 processed";

namespace {
/// Describes a singleton i1 access as a root memref plus the corresponding
/// linearized bit offset inside that root storage.
struct LinearizedI1Access {
  Value rootMemref;
  Value linearBitOffset;
  std::optional<OpFoldResult> linearBitSize;
};
} // namespace

/// Returns the static strides of a memref. Falls back to row-major strides
/// if the layout map does not expose explicit stride information.
static SmallVector<int64_t> getStaticStrides(MemRefType memRefTy) {
  SmallVector<int64_t> strides(memRefTy.getRank());
  int64_t offset = 0;
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  if (succeeded(getStridesAndOffset(memRefTy, strides, offset)))
#else
  if (succeeded(memRefTy.getStridesAndOffset(strides, offset)))
#endif
    return strides;

  int64_t runningStride = 1;
  for (int64_t i = memRefTy.getRank() - 1; i >= 0; --i) {
    strides[i] = runningStride;
    runningStride *= memRefTy.getDimSize(i);
  }
  return strides;
}

/// Returns whether a memref is a unit-stride rank-1 byte buffer that can
/// legally serve as the base of memref.view.
static bool isSupportedByteViewSource(MemRefType memRefTy) {
  if (memRefTy.getRank() != 1 || !memRefTy.getElementType().isInteger(8))
    return false;
  if (memRefTy.getLayout().isIdentity())
    return true;

  SmallVector<int64_t> strides(memRefTy.getRank());
  int64_t offset = 0;
  if (failed(getStridesAndOffset(memRefTy, strides, offset)))
    return false;
  return offset == 0 && strides[0] == 1;
}

static Value buildStridedLinearOffset(PatternRewriter &rewriter, Location loc,
                                      OpFoldResult baseOffset,
                                      ValueRange indices,
                                      ArrayRef<OpFoldResult> strides) {
  Value linearOffset =
      getValueOrCreateConstantIndexOp(rewriter, loc, baseOffset);
  for (auto [index, stride] : llvm::zip(indices, strides)) {
    Value strideValue = getValueOrCreateConstantIndexOp(rewriter, loc, stride);
    Value scaledIndex = rewriter.create<arith::MulIOp>(loc, index, strideValue);
    linearOffset =
        rewriter.create<arith::AddIOp>(loc, linearOffset, scaledIndex);
  }
  return linearOffset;
}

static OpFoldResult buildStridedStorageSpan(PatternRewriter &rewriter,
                                            Location loc,
                                            OpFoldResult baseOffset,
                                            ArrayRef<OpFoldResult> sizes,
                                            ArrayRef<OpFoldResult> strides) {
  bool allStatic = true;
  int64_t staticSpan = 1;
  if (auto staticOffset = mlir::getConstantIntValue(baseOffset)) {
    staticSpan += *staticOffset;
  } else {
    allStatic = false;
  }

  for (auto [size, stride] : llvm::zip(sizes, strides)) {
    auto staticSize = mlir::getConstantIntValue(size);
    auto staticStride = mlir::getConstantIntValue(stride);
    if (!staticSize || !staticStride) {
      allStatic = false;
      break;
    }
    staticSpan += (*staticSize - 1) * *staticStride;
  }

  if (allStatic)
    return rewriter.getIndexAttr(staticSpan);

  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value span = rewriter.create<arith::AddIOp>(
      loc, getValueOrCreateConstantIndexOp(rewriter, loc, baseOffset), one);
  for (auto [size, stride] : llvm::zip(sizes, strides)) {
    Value sizeValue = getValueOrCreateConstantIndexOp(rewriter, loc, size);
    Value strideValue = getValueOrCreateConstantIndexOp(rewriter, loc, stride);
    Value dimMinusOne = rewriter.create<arith::SubIOp>(loc, sizeValue, one);
    Value dimSpan =
        rewriter.create<arith::MulIOp>(loc, dimMinusOne, strideValue);
    span = rewriter.create<arith::AddIOp>(loc, span, dimSpan);
  }
  return getAsOpFoldResult(span);
}

static OpFoldResult buildByteBufferBitSize(PatternRewriter &rewriter,
                                           Location loc, Value byteBuffer,
                                           MemRefType byteBufferTy) {
  if (byteBufferTy.hasStaticShape())
    return rewriter.getIndexAttr(byteBufferTy.getDimSize(0) *
                                 util::BITS_PER_BYTE);

  Value byteSize = rewriter.create<memref::DimOp>(loc, byteBuffer, 0);
  Value bitsPerByte =
      rewriter.create<arith::ConstantIndexOp>(loc, util::BITS_PER_BYTE);
  Value bitSize = rewriter.create<arith::MulIOp>(loc, byteSize, bitsPerByte);
  return getAsOpFoldResult(bitSize);
}

/// Returns the row-major logical strides of a shaped value based only on shape.
///
/// Unlike getStaticStrides(), this helper intentionally ignores the memref
/// layout map. It is used for shape-changing ops whose semantics preserve the
/// logical element order while changing rank/shape.
static SmallVector<int64_t> getLogicalRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size());
  int64_t runningStride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = runningStride;
    runningStride *= shape[i];
  }
  return strides;
}

/// Computes the row-major logical linear index for a set of indices.
///
/// This is different from computeLinearMemRefOffset(): it ignores the current
/// layout map and only follows the logical shape order of the memref.
static FailureOr<Value> computeLogicalLinearIndex(PatternRewriter &rewriter,
                                                  Location loc,
                                                  MemRefType memRefTy,
                                                  ValueRange indices) {
  if (!memRefTy.hasStaticShape() ||
      indices.size() != static_cast<size_t>(memRefTy.getRank()))
    return failure();

  SmallVector<int64_t> strides = getLogicalRowMajorStrides(memRefTy.getShape());
  Value linearIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  for (auto [dim, index] : llvm::enumerate(indices)) {
    Value strideVal =
        rewriter.create<arith::ConstantIndexOp>(loc, strides[dim]);
    Value scaledIndex = rewriter.create<arith::MulIOp>(loc, index, strideVal);
    linearIndex = rewriter.create<arith::AddIOp>(loc, linearIndex, scaledIndex);
  }
  return linearIndex;
}

/// Reconstructs source indices from a logical linear index and the source
/// shape, assuming row-major logical element order.
static FailureOr<SmallVector<Value>>
buildIndicesFromLogicalLinearIndex(PatternRewriter &rewriter, Location loc,
                                   MemRefType sourceTy, Value linearIndex) {
  if (!sourceTy.hasStaticShape())
    return failure();

  SmallVector<Value> sourceIndices;
  sourceIndices.reserve(sourceTy.getRank());
  if (sourceTy.getRank() == 0)
    return sourceIndices;

  SmallVector<int64_t> strides = getLogicalRowMajorStrides(sourceTy.getShape());
  Value remaining = linearIndex;
  for (int64_t dim = 0, e = sourceTy.getRank(); dim < e; ++dim) {
    Value indexAtDim = remaining;
    if (dim + 1 < e) {
      Value strideVal =
          rewriter.create<arith::ConstantIndexOp>(loc, strides[dim]);
      indexAtDim = rewriter.create<arith::DivSIOp>(loc, remaining, strideVal);
      Value consumed =
          rewriter.create<arith::MulIOp>(loc, indexAtDim, strideVal);
      remaining = rewriter.create<arith::SubIOp>(loc, remaining, consumed);
    }
    sourceIndices.push_back(indexAtDim);
  }
  return sourceIndices;
}

/// Maps a shape-changing result access back to the source indices by going
/// through the shared row-major logical linear index.
static FailureOr<SmallVector<Value>>
buildShapeChangingSourceIndices(PatternRewriter &rewriter, Location loc,
                                MemRefType resultTy, ValueRange resultIndices,
                                MemRefType sourceTy) {
  if (!resultTy.hasStaticShape() || !sourceTy.hasStaticShape() ||
      resultTy.getNumElements() != sourceTy.getNumElements())
    return failure();

  auto linearIndex =
      computeLogicalLinearIndex(rewriter, loc, resultTy, resultIndices);
  if (failed(linearIndex))
    return failure();
  return buildIndicesFromLogicalLinearIndex(rewriter, loc, sourceTy,
                                            *linearIndex);
}

/// Recursively traces a singleton i1 access through supported view-like ops
/// and returns the root memref plus the linearized bit offset of the unique
/// accessed element.
static FailureOr<LinearizedI1Access>
linearizeSingleElementI1Access(PatternRewriter &rewriter, Location loc,
                               Value baseMemref, ValueRange indices) {
  auto memRefTy = dyn_cast<MemRefType>(baseMemref.getType());
  if (!memRefTy || !memRefTy.hasStaticShape() ||
      indices.size() != static_cast<size_t>(memRefTy.getRank()))
    return failure();

  Operation *defOp = baseMemref.getDefiningOp();
  if (!defOp) {
    return LinearizedI1Access{
        baseMemref,
        computeLinearMemRefOffset(rewriter, loc, baseMemref, indices,
                                  rewriter.getIndexType()),
        std::nullopt};
  }

  if (auto castOp = dyn_cast<memref::CastOp>(defOp))
    return linearizeSingleElementI1Access(rewriter, loc, castOp.getSource(),
                                          indices);

  if (auto subViewOp = dyn_cast<memref::SubViewOp>(defOp)) {
    SmallVector<Value> sourceIndices;
    sourceIndices.reserve(subViewOp.getSourceType().getRank());

    auto mixedOffsets = subViewOp.getMixedOffsets();
    auto mixedStrides = subViewOp.getMixedStrides();
    llvm::SmallBitVector droppedDims = subViewOp.getDroppedDims();
    size_t resultIdx = 0;
    for (int64_t dim = 0, e = subViewOp.getSourceType().getRank(); dim < e;
         ++dim) {
      Value offset =
          getValueOrCreateConstantIndexOp(rewriter, loc, mixedOffsets[dim]);
      if (droppedDims[dim]) {
        sourceIndices.push_back(offset);
        continue;
      }

      Value stride =
          getValueOrCreateConstantIndexOp(rewriter, loc, mixedStrides[dim]);
      Value scaledIndex =
          rewriter.create<arith::MulIOp>(loc, indices[resultIdx++], stride);
      sourceIndices.push_back(
          rewriter.create<arith::AddIOp>(loc, offset, scaledIndex));
    }
    return linearizeSingleElementI1Access(rewriter, loc, subViewOp.getSource(),
                                          sourceIndices);
  }

  if (auto transposeOp = dyn_cast<memref::TransposeOp>(defOp)) {
    auto sourceTy = cast<MemRefType>(transposeOp.getIn().getType());
    SmallVector<Value> sourceIndices(sourceTy.getRank());
    AffineMap permutation = transposeOp.getPermutation();
    for (auto [resultDim, expr] : llvm::enumerate(permutation.getResults())) {
      unsigned sourceDim = cast<AffineDimExpr>(expr).getPosition();
      sourceIndices[sourceDim] = indices[resultDim];
    }
    return linearizeSingleElementI1Access(rewriter, loc, transposeOp.getIn(),
                                          sourceIndices);
  }

  if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(defOp)) {
    MemRefType sourceTy = collapseOp.getSrcType();
    auto sourceIndices = buildShapeChangingSourceIndices(
        rewriter, loc, memRefTy, indices, sourceTy);
    if (failed(sourceIndices))
      return failure();
    return linearizeSingleElementI1Access(rewriter, loc, collapseOp.getSrc(),
                                          *sourceIndices);
  }

  if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(defOp)) {
    MemRefType sourceTy = expandOp.getSrcType();
    auto sourceIndices = buildShapeChangingSourceIndices(
        rewriter, loc, memRefTy, indices, sourceTy);
    if (failed(sourceIndices))
      return failure();
    return linearizeSingleElementI1Access(rewriter, loc, expandOp.getSrc(),
                                          *sourceIndices);
  }

  if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(defOp)) {
    auto sourceTy = dyn_cast<MemRefType>(reshapeOp.getSource().getType());
    if (!sourceTy)
      return failure();
    auto sourceIndices = buildShapeChangingSourceIndices(
        rewriter, loc, memRefTy, indices, sourceTy);
    if (failed(sourceIndices))
      return failure();
    return linearizeSingleElementI1Access(rewriter, loc, reshapeOp.getSource(),
                                          *sourceIndices);
  }

  if (auto reinterpretCastOp = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
    auto sourceTy =
        dyn_cast<MemRefType>(reinterpretCastOp.getSource().getType());
    if (!sourceTy || !memRefTy.getElementType().isInteger(1))
      return failure();

    SmallVector<OpFoldResult> mixedOffsets =
        reinterpretCastOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = reinterpretCastOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides =
        reinterpretCastOp.getMixedStrides();

    Value linearOffset = buildStridedLinearOffset(
        rewriter, loc, mixedOffsets.front(), indices, mixedStrides);
    OpFoldResult linearSize = buildStridedStorageSpan(
        rewriter, loc, mixedOffsets.front(), mixedSizes, mixedStrides);
    return LinearizedI1Access{reinterpretCastOp.getSource(), linearOffset,
                              linearSize};
  }

  if (auto viewOp = dyn_cast<memref::ViewOp>(defOp)) {
    auto sourceTy = cast<MemRefType>(viewOp.getSource().getType());
    if (!memRefTy.getElementType().isInteger(1))
      return failure();

    Value byteShift =
        getValueOrCreateConstantIndexOp(rewriter, loc, viewOp.getByteShift());
    Value bitsPerByte =
        rewriter.create<arith::ConstantIndexOp>(loc, util::BITS_PER_BYTE);
    Value bitShift =
        rewriter.create<arith::MulIOp>(loc, byteShift, bitsPerByte);
    Value viewElementOffset = computeLinearMemRefOffset(
        rewriter, loc, baseMemref, indices, rewriter.getIndexType());
    Value linearOffset =
        rewriter.create<arith::AddIOp>(loc, bitShift, viewElementOffset);
    OpFoldResult linearSize =
        buildByteBufferBitSize(rewriter, loc, viewOp.getSource(), sourceTy);
    return LinearizedI1Access{viewOp.getSource(), linearOffset, linearSize};
  }

  return LinearizedI1Access{baseMemref,
                            computeLinearMemRefOffset(rewriter, loc, baseMemref,
                                                      indices,
                                                      rewriter.getIndexType()),
                            std::nullopt};
}

/// Reinterprets the root singleton source as a linear 1D i1 memref so the
/// aligned VL load path can be reused for higher-rank or unaligned accesses.
static FailureOr<Value>
buildRawLinearI1View(PatternRewriter &rewriter, Location loc, Value rootMemref,
                     std::optional<OpFoldResult> linearBitSize = std::nullopt) {
  auto rootTy = dyn_cast<MemRefType>(rootMemref.getType());
  if (!rootTy)
    return failure();

  if (rootTy.getElementType().isInteger(1)) {
    int64_t staticSpan = ShapedType::kDynamic;
    SmallVector<OpFoldResult> linearSizes;

    if (linearBitSize) {
      staticSpan = mlir::getConstantIntValue(*linearBitSize)
                       .value_or(ShapedType::kDynamic);
      linearSizes.push_back(*linearBitSize);
    } else {
      if (!rootTy.hasStaticShape())
        return failure();

      SmallVector<int64_t> strides = getStaticStrides(rootTy);
      bool hasDynamicStride = false;
      for (int64_t stride : strides) {
        if (ShapedType::isDynamic(stride)) {
          hasDynamicStride = true;
          break;
        }
      }

      if (!hasDynamicStride) {
        staticSpan = 1;
        for (auto [dim, stride] : llvm::zip(rootTy.getShape(), strides))
          staticSpan += (dim - 1) * stride;
        linearSizes.push_back(rewriter.getIndexAttr(staticSpan));
      }
    }

    auto metadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, rootMemref);
    if (linearSizes.empty()) {
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value span = one;
      for (auto [size, stride] :
           llvm::zip(metadata.getSizes(), metadata.getStrides())) {
        Value dimMinusOne = rewriter.create<arith::SubIOp>(loc, size, one);
        Value dimSpan =
            rewriter.create<arith::MulIOp>(loc, dimMinusOne, stride);
        span = rewriter.create<arith::AddIOp>(loc, span, dimSpan);
      }
      linearSizes.push_back(span);
    }

    auto linearTy =
        MemRefType::get({staticSpan}, rootTy.getElementType(),
                        StridedLayoutAttr::get(rewriter.getContext(),
                                               ShapedType::kDynamic, {1}),
                        rootTy.getMemorySpace());
    auto rawView = rewriter.create<memref::ReinterpretCastOp>(
        loc, linearTy, metadata.getBaseBuffer(),
        getAsOpFoldResult(metadata.getOffset()), linearSizes,
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)});
    return rawView.getResult();
  }

  if (!rootTy.getElementType().isInteger(8) ||
      !isSupportedByteViewSource(rootTy))
    return failure();

  OpFoldResult bitSize =
      linearBitSize ? *linearBitSize
                    : buildByteBufferBitSize(rewriter, loc, rootMemref, rootTy);
  int64_t staticBitSize =
      mlir::getConstantIntValue(bitSize).value_or(ShapedType::kDynamic);
  SmallVector<Value> dynamicSizes;
  if (ShapedType::isDynamic(staticBitSize))
    dynamicSizes.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, bitSize));

  auto linearBitTy = MemRefType::get({staticBitSize}, rewriter.getI1Type(),
                                     AffineMap(), rootTy.getMemorySpace());
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto rawBitView = rewriter.create<memref::ViewOp>(
      loc, linearBitTy, rootMemref, zero, dynamicSizes);
  return rawBitView.getResult();
}

// Decompose a memory offset into a VL-aligned base and an intra-VL offset.
//
// Given `currOffset`, computes:
//   base = floor(currOffset / VL) * VL   // VL-aligned base address
//   offsetInVL = currOffset - base           // remainder within [0, VL)
//
// This is used when loading i1 vectors: the hardware vector load must start
// at a VL-aligned boundary, and the actual data position within the loaded
// vector is tracked separately via `offsetInVL`.  For example, with VL=256
// and offset=300:
//   base = 256, offsetInVL = 44
//
// Returns {base, offsetInVL} both cast back to index type.
std::pair<Value, Value> getBaseAndOffetInVL(PatternRewriter &rewriter,
                                            Location &loc, Value currOffset) {
  Value constVL = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(util::VL));
  Value constByteSize = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(util::BITS_PER_BYTE));
  Value i32Offset = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI32Type(), currOffset);
  Value numVL = rewriter.create<arith::DivSIOp>(loc, i32Offset, constVL);
  Value newBase = rewriter.create<arith::MulIOp>(loc, numVL, constVL);
  Value newOffset = rewriter.create<arith::SubIOp>(loc, i32Offset, newBase);
  // Load indice will be convert to llvm.gep.
  // The offset of the gep instruction is measured in bytes.
  Value newBaseInByte =
      rewriter.create<arith::DivSIOp>(loc, newBase, constByteSize);
  Value indexBase = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), newBaseInByte);
  Value indexOffset = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), newOffset);
  return {indexBase, indexOffset};
}

/// Convert an i1 vector to a predicate register via i8 expansion:
///   i1 mask → select(i8 0/1) → broadcast → cmp(NE, 0) → constrain(B8)
/// If `offsetInVL` has value, inserts a PregXor + Reduction(XORI) before
/// the broadcast to shift the active element to the lowest position
/// (used for unaligned loads).
static Value convertI1ToPreg(VectorType orgVectorTy, Value i1Val,
                             Value offsetInVL, PatternRewriter &rewriter,
                             Location loc) {
  int64_t vecSize = orgVectorTy.getNumElements();
  VectorType i8VecTy = VectorType::get({util::VL}, rewriter.getI8Type());
  VectorType i8MaskTy = VectorType::get({util::VL}, rewriter.getI1Type());
  if (vecSize != util::VL)
    i1Val = rewriter.create<UnrealizedConversionCastOp>(loc, i8MaskTy, i1Val)
                .getResult(0);
  Value allI8Mask = rewriter.create<hivmave::VFPgeOp>(
      loc, i8MaskTy,
      PgePatternAttr::get(rewriter.getContext(), PgePattern::ALL));
  Value constZeroI8 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(rewriter.getI8Type()));
  Value constOneI8 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getOneAttr(rewriter.getI8Type()));

  // i1 → i8: select 1 for true lanes, 0 for false lanes.
  VFBroadcastScalarOp brcZero =
      rewriter.create<hivmave::VFBroadcastScalarOp>(loc, i8VecTy, constZeroI8);
  VFBroadcastScalarOp brcOne =
      rewriter.create<hivmave::VFBroadcastScalarOp>(loc, i8VecTy, constOneI8);
  Value selI8 = rewriter.create<hivmave::VFSelectOp>(loc, i8VecTy, i1Val,
                                                     brcOne, brcZero);

  // For unaligned loads, shift the active element to position 0 via
  // reduce-xori so that the broadcast propagates the correct value.
  Value shiftResult = selI8;
  if (offsetInVL) {
    Value indexOne =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    auto postOffsetInVL =
        rewriter.create<arith::AddIOp>(loc, offsetInVL, indexOne);
    VFPltOp p1 = rewriter.create<hivmave::VFPltOp>(
        loc, i8MaskTy, rewriter.getIndexType(), postOffsetInVL);
    VFPltOp p2 = rewriter.create<hivmave::VFPltOp>(
        loc, i8MaskTy, rewriter.getIndexType(), offsetInVL);
    PregXorOp pXor = rewriter.create<hivmave::PregXorOp>(
        loc, i8MaskTy, MaskWidthAttr::get(rewriter.getContext(), MaskWidth::B8),
        p1.getResults()[0], p2.getResults()[0], allI8Mask);
    shiftResult = rewriter.create<hivmave::ReductionOp>(
        loc, i8VecTy, hivmave::CombiningKind::XORI, selI8, pXor);
  }

  // Broadcast the (possibly shifted) i8 value, then compare NE with zero
  // to produce the final predicate register.
  VFBroadcastVectorOp brcI8 = rewriter.create<hivmave::VFBroadcastVectorOp>(
      loc, i8VecTy, shiftResult, allI8Mask, rewriter.getBoolAttr(true));
  Value newPreg = rewriter.create<hivmave::VFCmpOp>(
      loc, i8MaskTy, hivmave::CmpType::NE, brcI8, brcZero, allI8Mask);

  // Constrain the layout to B8 for VectorLayout analysis.
  newPreg = hivmave::constrainVectorLayout(newPreg, hivmave::VecMemType::B8,
                                           rewriter);
  if (vecSize != util::VL)
    newPreg =
        rewriter.create<UnrealizedConversionCastOp>(loc, orgVectorTy, newPreg)
            .getResult(0);
  return newPreg;
}

// process load + brc i1
struct loadBroadcastPattern : public OpRewritePattern<hivmave::VFLoadOp> {
  loadBroadcastPattern(MLIRContext *context)
      : OpRewritePattern<hivmave::VFLoadOp>(context, /*benefit=*/10) {}

  void rewriteLoadI1(mlir::hivmave::VFLoadOp loadOp,
                     PatternRewriter &rewriter) const {
    Location loc = loadOp.getLoc();
    rewriter.setInsertionPointAfter(loadOp);
    SmallVector<Operation *> oldUsers(loadOp.getRes().getUsers());
    Value newPreg = convertI1ToPreg(loadOp.getVectorType(), loadOp.getRes(),
                                    Value(), rewriter, loc);
    for (Operation *user : oldUsers)
      user->replaceUsesOfWith(loadOp.getRes(), newPreg);
    loadOp->setAttr(i1ProcessedAttr, rewriter.getUnitAttr());
  }

  /// Rewrites a singleton i1 load by:
  ///   1. tracing the load back to a supported root memref,
  ///   2. building a raw 1D i1 view on that root storage,
  ///   3. reloading from a VL-aligned base, and
  ///   4. reusing convertI1ToPreg() with offsetInVL to broadcast the selected
  ///      bit into the final predicate vector.
  LogicalResult rewriteLoadI1Linearized(mlir::hivmave::VFLoadOp loadOp,
                                        PatternRewriter &rewriter) const {
    Location loc = loadOp.getLoc();
    rewriter.setInsertionPointAfter(loadOp);
    auto linearized = linearizeSingleElementI1Access(
        rewriter, loc, loadOp.getBase(), loadOp.getIndices());
    if (failed(linearized))
      return failure();
    auto rawLinearView = buildRawLinearI1View(
        rewriter, loc, linearized->rootMemref, linearized->linearBitSize);
    if (failed(rawLinearView))
      return failure();

    // Split the linearized bit position into a VL-aligned base and the lane
    // offset inside that loaded VL chunk.
    auto [baseIndices, offsetInVL] =
        getBaseAndOffetInVL(rewriter, loc, linearized->linearBitOffset);
    VectorType orgVectorTy = loadOp.getVectorType();
    VFLoadOp newLoad = rewriter.create<hivmave::VFLoadOp>(
        loc, orgVectorTy, *rawLinearView, baseIndices);
    Value newPreg = convertI1ToPreg(orgVectorTy, newLoad.getResult(0),
                                    offsetInVL, rewriter, loc);
    rewriter.replaceAllUsesWith(loadOp.getResult(0), newPreg);
    rewriter.eraseOp(loadOp);
    newLoad->setAttr(i1ProcessedAttr, rewriter.getUnitAttr());
    return success();
  }

  LogicalResult matchAndRewrite(mlir::hivmave::VFLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    VectorType orgVectorTy = loadOp.getVectorType();
    Type vecElemTy = orgVectorTy.getElementType();
    MemRefType memRefTy = loadOp.getMemRefType();
    if (loadOp->hasAttr(i1ProcessedAttr))
      return failure();
    if (!vecElemTy.isInteger(1) || !memRefTy.hasStaticShape() ||
        memRefTy.getNumElements() != 1)
      return failure();
    LDBG("Process operation : " << loadOp);

    if (!loadOp->hasAttr(UnalignedAttr::name) && memRefTy.getRank() == 1) {
      rewriteLoadI1(loadOp, rewriter);
    } else {
      if (failed(rewriteLoadI1Linearized(loadOp, rewriter)))
        return failure();
    }
    return success();
  }
};

/// Rewrites an unaligned singleton i1 store as a predicate-block
/// read-modify-write.
///
/// Before conversion:
/// %value = ave.hir.vcmp <EQ> %lhs, %rhs, %all
///     : vector<64xf32>, vector<64xi1> -> vector<64xi1>
/// ave.hir.masked_store <NORM_B8> %subview[%c0, %c0], %mask, %value
///       {ave.unaligned_ub_access = #ave.unaligned_ub_access} :
///       memref<1x1xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>,
///       vector<64xi1>, vector<64xi1>
///
/// After conversion:
/// %linear = memref.reinterpret_cast %singleton[...] to memref<?xi1>
/// %old = ave.hir.vload <NORM> %linear[%base]
///     {"1xi1 processed"} : memref<?xi1, strided<[1], offset: ?>>
///     into vector<256xi1>
///
/// %p1 = ave.hir.plt (%offsetInVL + 1) : vector<256xi1>
/// %p2 = ave.hir.plt %offsetInVL : vector<256xi1>
/// %oneHot = ave.hir.preg.xor <B8> %p1, %p2, %all : vector<256xi1>
///
/// %valueAll = convertI1ToPreg(%value) : vector<256xi1>
/// %diff = ave.hir.preg.xor <B8> %old, %valueAll, %all : vector<256xi1>
/// %delta = ave.hir.preg.and <B8> %diff, %oneHot, %all : vector<256xi1>
/// %new = ave.hir.preg.xor <B8> %old, %delta, %all : vector<256xi1>
/// ave.hir.masked_store <NORM_B8> %linear[%base], %all, %new
///     {"1xi1 processed"} : memref<?xi1, strided<[1], offset: ?>>,
///                           vector<256xi1>, vector<256xi1>
struct storeBroadcastPattern
    : public OpRewritePattern<hivmave::VFMaskedStoreOp> {
  storeBroadcastPattern(MLIRContext *context)
      : OpRewritePattern<hivmave::VFMaskedStoreOp>(context, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(hivmave::VFMaskedStoreOp store,
                                PatternRewriter &rewriter) const override {
    // Step 1: Restrict the pattern to original unaligned singleton i1 stores.
    if (!store->hasAttr(UnalignedAttr::name) ||
        store->hasAttr(i1ProcessedAttr))
      return failure();

    VectorType valueTy = store.getVectorType();
    MemRefType memRefTy = store.getMemRefType();
    if (!valueTy.getElementType().isInteger(1) || !memRefTy.hasStaticShape() ||
        memRefTy.getNumElements() != 1 || valueTy.getRank() != 1)
      return failure();

    // Step 2: Linearize the destination bit address and create a flat i1 view.
    Location loc = store.getLoc();
    auto linearized = linearizeSingleElementI1Access(
        rewriter, loc, store.getBase(), store.getIndices());
    if (failed(linearized))
      return failure();
    auto rawLinearView = buildRawLinearI1View(
        rewriter, loc, linearized->rootMemref, linearized->linearBitSize);
    if (failed(rawLinearView))
      return failure();

    // Step 3: Load the containing predicate block and compute its lane offset.
    auto [baseIndex, offsetInVL] =
        getBaseAndOffetInVL(rewriter, loc, linearized->linearBitOffset);
    VectorType pregTy = VectorType::get({util::VL}, rewriter.getI1Type());

    // Load the old block; the update must preserve every other bit.
    auto oldBlockLoad = rewriter.create<hivmave::VFLoadOp>(
        loc, pregTy, *rawLinearView, baseIndex);
    oldBlockLoad->setAttr(i1ProcessedAttr, rewriter.getUnitAttr());
    Value oldBlock = oldBlockLoad->getResult(0);

    // Step 4: Build a one-hot predicate and normalize the incoming value.
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value nextOffset = rewriter.create<arith::AddIOp>(loc, offsetInVL, one);
    auto p1 = rewriter.create<hivmave::VFPltOp>(
        loc, pregTy, rewriter.getIndexType(), nextOffset);
    auto p2 = rewriter.create<hivmave::VFPltOp>(
        loc, pregTy, rewriter.getIndexType(), offsetInVL);
    Value allMask = rewriter.create<hivmave::VFPgeOp>(
        loc, pregTy,
        PgePatternAttr::get(rewriter.getContext(), PgePattern::ALL));

    auto oneHot = rewriter.create<hivmave::PregXorOp>(
        loc, pregTy, MaskWidthAttr::get(rewriter.getContext(), MaskWidth::B8),
        p1->getResult(0), p2->getResult(0), allMask);

    // Convert the source to the predicate used by the RMW. B16/B32 predicate
    // results must be packed to B8 first; the cast mode follows the comparison
    // element width. Other sources may already be B8.
    Value value = store.getVal();
    if (auto cmp = value.getDefiningOp<hivmave::VFCmpOp>()) {
      auto lhsTy = cast<VectorType>(cmp.getLhs().getType());
      auto bitWidth = lhsTy.getElementTypeBitWidth();
      if (bitWidth == 16) {
        value = rewriter.create<hivmave::VFPregTypeCastOp>(
            loc, valueTy, value, hivmave::PregCastMode::PK_B16);
      } else if (bitWidth == 32 || bitWidth == 64) {
        // AVE uses B32 predicate granularity for 32/64-bit comparisons.
        value = rewriter.create<hivmave::VFPregTypeCastOp>(
            loc, valueTy, value, hivmave::PregCastMode::PK4_B32);
      }
    }
    Value valueAll = convertI1ToPreg(valueTy, value, Value(), rewriter, loc);
    if (valueAll.getType() != pregTy)
      valueAll =
          rewriter.create<UnrealizedConversionCastOp>(loc, pregTy, valueAll)
              .getResult(0);

    // TODO: Replace this RMW sequence with PSEL once the PSEL op is available.
    // Step 5: Replace only the selected bit:
    //   newBlock = oldBlock XOR ((oldBlock XOR valueAll) AND oneHot).
    auto diff = rewriter.create<hivmave::PregXorOp>(
        loc, pregTy, MaskWidthAttr::get(rewriter.getContext(), MaskWidth::B8),
        oldBlock, valueAll, allMask);
    auto delta = rewriter.create<hivmave::PregAndOp>(
        loc, pregTy, MaskWidthAttr::get(rewriter.getContext(), MaskWidth::B8),
        diff->getResult(0), oneHot->getResult(0), allMask);
    auto newBlock = rewriter.create<hivmave::PregXorOp>(
        loc, pregTy, MaskWidthAttr::get(rewriter.getContext(), MaskWidth::B8),
        oldBlock, delta->getResult(0), allMask);

    // Step 6: Store the updated B8 block and mark it processed to avoid
    // re-rewriting.
    auto newStore = rewriter.create<hivmave::VFMaskedStoreOp>(
        loc, hivmave::StoreDist::NORM_B8, *rawLinearView, ValueRange{baseIndex},
        allMask, newBlock->getResult(0));
    newStore->setAttr(i1ProcessedAttr, rewriter.getUnitAttr());
    rewriter.eraseOp(store);
    return success();
  }
};

namespace {
struct i1opSoftImplPass : public impl::I1opSoftImplBase<i1opSoftImplPass> {
  using Base::Base;

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<loadBroadcastPattern>(context);
    patterns.add<storeBroadcastPattern>(context);
    mlir::GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> hivmave::createI1opSoftImplPass() {
  return std::make_unique<i1opSoftImplPass>();
}
