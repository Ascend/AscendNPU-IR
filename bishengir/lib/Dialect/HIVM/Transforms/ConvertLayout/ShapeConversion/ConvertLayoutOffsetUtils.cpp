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

/// Helper struct for offset conversion parameters
struct OffsetConversionParams {
  OpFoldResult a; // dimension a value (i index)
  OpFoldResult b; // dimension b value (j index)
  FractalSize fractalSize{};
  int batchIndexBias{};
  OpFoldResult batch;
};

/// Extract parameters for offset conversion
static FailureOr<OffsetConversionParams> extractOffsetConversionParams(
    ArrayRef<OpFoldResult> currentOffset,
    DataLayoutAttr srcLayout,
    DataLayoutAttr dstLayout) {

  auto blockSizesResult = extractBlockSizes(dstLayout);
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

/// Compute fractal indices from ND coordinates
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

/// Common implementation for ND to fractal offset conversion
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
