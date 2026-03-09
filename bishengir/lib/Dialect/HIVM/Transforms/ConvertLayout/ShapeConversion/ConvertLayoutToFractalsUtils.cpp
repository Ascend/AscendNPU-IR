//===-------------------- ConvertLayoutToFractalsUtils.cpp ----------------===//
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
// Shape Computation - ND to Fractal
//===----------------------------------------------------------------------===//

/// Helper struct for ND to fractal shape conversion parameters.
struct NDToFractalConversionParams {
  OpFoldResult a;        // dimension a value
  OpFoldResult b;        // dimension b value
  FractalSize fractalSize{};
  int batchIndexBias{};
  OpFoldResult batch;
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

/// Build an AffineMap for: d0 ceildiv Constant.
AffineMap getCeilDivMap(int64_t divisor, MLIRContext *ctx) {
  AffineExpr d0 = getAffineDimExpr(0, ctx);
  return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                        d0.ceilDiv(divisor), ctx);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Core Implementation
//===----------------------------------------------------------------------===//

/// Extract parameters for ND to fractal shape conversion.
static FailureOr<NDToFractalConversionParams>
extractNDToFractalConversionParams(ArrayRef<OpFoldResult> currentShape,
                                   DataLayoutAttr srcLayout,
                                   DataLayoutAttr dstLayout,
                                   MLIRContext *ctx) {
  LDBG("=== extractNDToFractalConversionParams ===");
  auto blockSizesResult = extractBlockSizes(dstLayout);
  if (failed(blockSizesResult))
    return failure();

  NDToFractalConversionParams params;
  params.fractalSize = *blockSizesResult;

  params.batchIndexBias = computeBatchIndexBias(currentShape.size());
  LDBG("Batch index bias: " << params.batchIndexBias);

  if (params.batchIndexBias) {
    params.batch = currentShape[0];
  }
  params.a = currentShape[0 + params.batchIndexBias];
  params.b = currentShape[1 + params.batchIndexBias];

  LDBG("Applied params");
  return params;
}

/// Assemble fractal shape based on layout type.
/// Block sizes (a0, b0) are always static, so no builder is needed.
static SmallVector<OpFoldResult>
assembleFractalShape(OpFoldResult aTiles, OpFoldResult bTiles,
                     const NDToFractalConversionParams &params, MLIRContext *ctx) {

  LDBG("=== assembling fractal shape ===");
  LDBG("Tile a: " << aTiles);
  LDBG("Tile b: " << bTiles);
  SmallVector<OpFoldResult> fractalShape = {
      aTiles, bTiles, getAsIndexOpFoldResult(ctx, params.fractalSize.first),
      getAsIndexOpFoldResult(ctx, params.fractalSize.second)};
  if (params.batchIndexBias) {
    LDBG("Inserting batch dimension");
    fractalShape.insert(fractalShape.begin(), params.batch);
  }
  return fractalShape;
}

//===----------------------------------------------------------------------===//
// Public API - Mixed Shape Computation
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<OpFoldResult>>
computeMixedNDToFractalShape(ArrayRef<OpFoldResult> currentShape,
                             DataLayoutAttr srcLayout,
                             DataLayoutAttr dstLayout, OpBuilder &builder,
                             Location loc) {
  MLIRContext *ctx = srcLayout.getContext();

  auto paramsResult = extractNDToFractalConversionParams(
      currentShape, srcLayout,
      dstLayout, ctx);
  if (failed(paramsResult))
    return failure();

  auto &params = *paramsResult;

  // Use affine::makeComposedFoldedAffineApply for ceildiv computation.
  // This automatically folds static operands to attributes and emits
  // affine.apply ops only for dynamic operands.
  AffineMap f0Map = getCeilDivMap(params.fractalSize.first, ctx);
  AffineMap f1Map = getCeilDivMap(params.fractalSize.second, ctx);

  OpFoldResult bTiles =
      affine::makeComposedFoldedAffineApply(builder, loc, f0Map, {params.a});
  OpFoldResult aTiles =
      affine::makeComposedFoldedAffineApply(builder, loc, f1Map, {params.b});

  return assembleFractalShape(aTiles, bTiles, params, ctx);
}

} // namespace mlir::hivm