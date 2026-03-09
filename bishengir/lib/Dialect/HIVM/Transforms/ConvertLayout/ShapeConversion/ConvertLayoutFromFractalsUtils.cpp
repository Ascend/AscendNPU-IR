//===-------------------- ConvertLayoutFromFractalsUtils.cpp --------------===//
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
// Shape Computation - Fractal to ND
//===----------------------------------------------------------------------===//

/// Helper struct for fractal to ND shape conversion parameters.
struct FractalToNDConversionParams {
  // Fractal = [a, b, f[0], f[1]]
  // zN shape: (b' ceildiv f[0], a' ceildiv f[1], f[0], f[1]) (no transpose case)
  // nZ shape: (a' ceildiv f[0], b' ceildiv f[1], f[0], f[1])
  // nD shape: (a', b') (no transpose)
  // nD shape: (a * f[0], b * f[1])  (no transpose)
  OpFoldResult aTiles;       // tiled A value
  OpFoldResult bTiles;       // tiled B value
  FractalSize fractalSize{};
  int batchIndexBias{};
  OpFoldResult batchDim;
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

/// Build an AffineMap for: d0 * C.
AffineMap getMulConstMap(int64_t factor, MLIRContext *ctx) {
  AffineExpr d0 = getAffineDimExpr(0, ctx);
  return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, d0 * factor, ctx);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Core Implementation
//===----------------------------------------------------------------------===//

/// Extract parameters for fractal to ND shape conversion.
static FailureOr<FractalToNDConversionParams>
extractFractalToNDConversionParams(ArrayRef<OpFoldResult> currentShape,
                                   DataLayoutAttr srcLayout,
                                   DataLayoutAttr dstLayout,
                                   MLIRContext *ctx) {
  LDBG("=== extractFractalToNDConversionParams ===");

  if (currentShape.size() < 4) {
    LDBG("ERROR: Insufficient dimensions for fractal shape (need at least 4, got "
         << currentShape.size() << ")");
    return failure();
  }
  LDBG(srcLayout);
  auto blockSizesResult = extractBlockSizes(srcLayout);
  if (failed(blockSizesResult))
    return failure();

  FractalToNDConversionParams params;
  params.fractalSize = *blockSizesResult;
  params.batchIndexBias = (currentShape.size() == 5) ? 1 : 0;
  LDBG("Batch index bias: " << params.batchIndexBias);
  if (params.batchIndexBias)
    params.batchDim = currentShape[0];

  params.aTiles = currentShape[0 + params.batchIndexBias];
  params.bTiles = currentShape[1 + params.batchIndexBias];

  return params;
}

/// Assemble ND shape from fractal parameters.
/// Block sizes (a0, b0) are always static, so no builder is needed here.
static SmallVector<OpFoldResult>
assembleNDShape(OpFoldResult a, OpFoldResult b,
                const FractalToNDConversionParams &params,
                DataLayoutAttr dstLayout) {
  SmallVector<OpFoldResult> ndShape;
  if (params.batchIndexBias)
    ndShape.push_back(params.batchDim);

  ndShape.push_back(a);
  ndShape.push_back(b);

  return ndShape;
}

//===----------------------------------------------------------------------===//
// Public API - Mixed Shape Computation
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<OpFoldResult>>
computeMixedFractalToNDShape(ArrayRef<OpFoldResult> currentShape,
                             DataLayoutAttr srcLayout,
                             DataLayoutAttr dstLayout, OpBuilder &builder,
                             Location loc) {
  MLIRContext *ctx = srcLayout.getContext();

  auto paramsResult =
      extractFractalToNDConversionParams(currentShape, srcLayout, dstLayout, ctx);
  if (failed(paramsResult))
    return failure();

  auto &params = *paramsResult;

  // Use affine::makeComposedFoldedAffineApply for product computation.
  // This automatically folds static operands to attributes and emits
  // affine.apply ops only for dynamic operands.
  AffineMap f0Map = getMulConstMap(params.fractalSize.first, ctx);
  AffineMap f1Map = getMulConstMap(params.fractalSize.second, ctx);

  OpFoldResult ndA =
      affine::makeComposedFoldedAffineApply(builder, loc, f0Map,
                                            {params.bTiles});
  OpFoldResult ndB =
      affine::makeComposedFoldedAffineApply(builder, loc, f1Map,
                                            {params.aTiles});

  return assembleNDShape(ndA, ndB, params, dstLayout);
}

} // namespace mlir::hivm