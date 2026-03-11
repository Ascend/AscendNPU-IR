//===-------------------- ConvertLayoutOffsetUtils.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"

#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#define DEBUG_TYPE "convert-layout-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace mlir::hivm {
//===----------------------------------------------------------------------===//
// Offset Computation (Value-based, for index transformations)
//===----------------------------------------------------------------------===//


/// Helper struct bundling all operands required to transform an ND index tuple
/// into a fractal index tuple.
///
/// All fields are populated by `extractOffsetConversionParams` and consumed
/// by `computeFractalIndices` and `computeNDToFractalOffsetImpl`. Grouping
/// them in a single struct avoids threading many individual arguments through
/// the call chain.
struct OffsetConversionParams {
  /// Row index of the source ND coordinate, expressed as a mixed
  /// static/dynamic `OpFoldResult`.
  OpFoldResult a;

  /// Column index of the source ND coordinate, expressed as a mixed
  /// static/dynamic `OpFoldResult`.
  OpFoldResult b;

  /// The two fractal tile dimensions $(f_0, f_1)$ extracted from the
  /// destination layout attribute. Both dimensions are always static.
  FractalSize fractalSize{};

  /// Index offset applied when the source offset is rank-3 (batched).
  /// Set to 1 for rank-3 offsets and 0 for rank-2 offsets by
  /// `computeBatchIndexBias`.
  int batchIndexBias{};

  /// Leading batch index of a rank-3 source offset, expressed as a mixed
  /// static/dynamic `OpFoldResult`. Only valid when `batchIndexBias == 1`.
  OpFoldResult batch;
};

/// Extract and validate all parameters needed for the ND→fractal offset
/// computation from the given index tuple and layout attributes.
static FailureOr<OffsetConversionParams> extractOffsetConversionParams(
    ArrayRef<OpFoldResult> currentOffset,
    DataLayoutAttr srcLayout,
    DataLayoutAttr dstLayout) {

  auto blockSizesResult = dstLayout.getFractalBlockSizes();
  if (failed(blockSizesResult))
    return failure();

  OffsetConversionParams params;
  params.fractalSize = *blockSizesResult;

  params.batchIndexBias = computeBatchIndexBias(currentOffset.size());
  LDBG("Batch index bias: " << params.batchIndexBias);

  params.a = currentOffset[params.batchIndexBias + 0];
  params.b = currentOffset[params.batchIndexBias + 1];
  if (params.batchIndexBias) params.batch = currentOffset[0];
  return params;
}


/// Decompose a pair of ND spatial indices into four fractal indices.
///
/// Given the row index $a$ and the column index $b$ from `params`, together
/// with the static tile sizes $(f_0, f_1)$ from `params.fractalSize`.
///
/// @param params  Conversion parameters providing the spatial indices and
///                the static fractal tile sizes.
/// @param builder `OpBuilder` used to emit `affine.apply` ops for any
///                dynamic operands.
/// @param loc     Location attached to any newly created operations.
/// @param a1Idx   Output: tile-row index $\lfloor a / f_0 \rfloor$.
/// @param b1Idx   Output: tile-column index $\lfloor b / f_1 \rfloor$.
/// @param a0Idx   Output: intra-tile row index $a \bmod f_0$.
/// @param b0Idx   Output: intra-tile column index $b \bmod f_1$.
static void computeFractalIndices(
    const OffsetConversionParams &params,
    OpBuilder &builder,
    Location loc,
    OpFoldResult &a1Idx, OpFoldResult &b1Idx,
    OpFoldResult &a0Idx, OpFoldResult &b0Idx) {

  AffineExpr d0 = builder.getAffineDimExpr(0);

  AffineMap divByf0 = AffineMap::get(1, 0, d0.floorDiv(params.fractalSize.first),
                                     builder.getContext());
  AffineMap modByf0 = AffineMap::get(1, 0, d0 % params.fractalSize.first,
                                     builder.getContext());
  AffineMap divByf1 = AffineMap::get(1, 0, d0.floorDiv(params.fractalSize.second),
                                     builder.getContext());
  AffineMap modByf1 = AffineMap::get(1, 0, d0 % params.fractalSize.second,
                                     builder.getContext());

  a1Idx = affine::makeComposedFoldedAffineApply(
      builder, loc, divByf0, {params.a});
  b1Idx = affine::makeComposedFoldedAffineApply(
      builder, loc, divByf1, {params.b});
  a0Idx = affine::makeComposedFoldedAffineApply(
      builder, loc, modByf0, {params.a});
  b0Idx = affine::makeComposedFoldedAffineApply(
      builder, loc, modByf1, {params.b});

  LDBG("Computed fractal indices");
}

/// Core implementation that converts an ND offset to a fractal offset.
///
/// Given a source offset of rank 2 $(a, b)$ or rank 3 $(batch, a, b)$ and a
/// destination fractal layout with tile sizes $(f_0, f_1)$.
///
/// where $a_1$, $b_1$, $a_0$, $b_0$ are computed by `computeFractalIndices`.
static FailureOr<SmallVector<OpFoldResult>> computeNDToFractalOffsetImpl(
    ArrayRef<OpFoldResult> currentOffset,
    DataLayoutAttr srcLayout,
    DataLayoutAttr dstLayout,
    OpBuilder &builder,
    Location loc) {
  LDBG("Compute ND to Fractal Offset");
  auto paramsResult = extractOffsetConversionParams(
      currentOffset, srcLayout, dstLayout);
  if (failed(paramsResult))
    return failure();

  auto &params = *paramsResult;

  OpFoldResult a1Idx, b1Idx, a0Idx, b0Idx;
  computeFractalIndices(params, builder, loc, a1Idx, b1Idx, a0Idx, b0Idx);
  SmallVector<OpFoldResult> fractalOffset = {b1Idx, a1Idx, a0Idx, b0Idx};
  // Add batch dimension if present
  if (params.batchIndexBias) {
    fractalOffset.insert(fractalOffset.begin(), params.batch);
  }

  return fractalOffset;
}

/// Compute target layout offset based on layout conversion
FailureOr<SmallVector<OpFoldResult>> computeTargetLayoutOffset(
    ArrayRef<OpFoldResult> currentOffset,
    DataLayoutAttr srcLayout,
    DataLayoutAttr dstLayout,
    PatternRewriter &rewriter,
    Location loc) {

  if (!srcLayout || !dstLayout)
    llvm::report_fatal_error("Layout cannot be found!");
  if (srcLayout.getDataLayout() != DataLayout::ND || dstLayout.getDataLayout()
      != DataLayout::Fractal)
    llvm::report_fatal_error("Source and destination layout is incorrect!");

  return computeNDToFractalOffsetImpl(
      currentOffset, srcLayout, dstLayout, rewriter, loc);
}
}
