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

/// Aggregates the operands required to compute the ND shape that results from
/// converting a fractal-tiled tensor back to a plain ND layout.
///
/// All fields are populated by `extractFractalToNDConversionParams` and
/// consumed by `assembleNDShape` and `computeMixedFractalToNDShape`. Grouping
/// them in a single struct avoids threading many individual arguments through
/// the call chain.
struct FractalToNDConversionParams {
  /// Number of tiles along the A (row) dimension of the fractal tensor,
  /// expressed as a mixed static/dynamic `OpFoldResult`.
  OpFoldResult aTiles;

  /// Number of tiles along the B (column) dimension of the fractal tensor,
  /// expressed as a mixed static/dynamic `OpFoldResult`.
  OpFoldResult bTiles;

  /// The two fractal tile dimensions $(f_0, f_1)$ extracted from the source
  /// layout attribute. Both dimensions are always static.
  FractalSize fractalSize{};

  /// Index offset applied when the source tensor is rank-5 (batched).
  /// Set to 1 for rank-5 tensors and 0 for rank-4 tensors.
  int batchIndexBias{};

  /// Leading batch dimension of a rank-5 source tensor, expressed as a mixed
  /// static/dynamic `OpFoldResult`. Only valid when `batchIndexBias == 1`.
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

/// Extract and validate all parameters needed for the fractal→ND shape
/// computation from the given shapes and layout attributes.
static FailureOr<FractalToNDConversionParams>
extractFractalToNDConversionParams(ArrayRef<OpFoldResult> currentShape,
                                   DataLayoutAttr srcLayout,
                                   DataLayoutAttr dstLayout,
                                   MLIRContext *ctx) {
  LDBG("=== extractFractalToNDConversionParams ===");

  if (currentShape.size() < kFractalDims) {
    LDBG("ERROR: Insufficient dimensions for fractal shape (need at least 4, got "
         << currentShape.size() << ")");
    return failure();
  }
  LDBG(srcLayout);
  auto blockSizesResult = srcLayout.getFractalBlockSizes();
  if (failed(blockSizesResult))
    return failure();

  FractalToNDConversionParams params;
  params.fractalSize = *blockSizesResult;
  params.batchIndexBias = static_cast<int>(currentShape.size()) % 2;
  LDBG("Batch index bias: " << params.batchIndexBias);
  if (params.batchIndexBias)
    params.batchDim = currentShape[0];

  params.aTiles = currentShape[0 + params.batchIndexBias];
  params.bTiles = currentShape[1 + params.batchIndexBias];

  return params;
}

/// Assemble the ND output shape from the recovered spatial extents and the
/// batch dimension stored in `params`.
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

/// Compute the mixed static/dynamic result shape for a fractal → ND layout
/// conversion.
///
/// This helper takes a source fractal tensor shape (`currentShape`) encoded as
/// `OpFoldResult`s and reconstructs the corresponding plain ND shape, folding
/// affine arithmetic whenever possible.
///
/// Supported source ranks:
/// - Rank-4: `[aTiles, bTiles, f0, f1]`
/// - Rank-5 (batched): `[batch, aTiles, bTiles, f0, f1]`
///
/// Let `(f0, f1)` be the fractal tile sizes from `srcLayout`. The recovered ND
/// extents are computed as:
/// - `ndA = bTiles * f0`
/// - `ndB = aTiles * f1`
///
/// Therefore, the output shape is:
/// - Rank-4 input: `[ndA, ndB]`
/// - Rank-5 input: `[batch, ndA, ndB]`
///
/// Implementation notes:
/// - Parameters (tile counts, tile sizes, optional batch dim) are extracted by
///   `extractFractalToNDConversionParams`.
/// - `affine::makeComposedFoldedAffineApply` is used for multiplication so
///   static values are folded to attributes and dynamic values emit
///   `affine.apply` only when needed.
///
/// Returns:
/// - `failure()` if the input shape is not a valid fractal rank or if fractal
///   block sizes cannot be extracted from `srcLayout`.
/// - Otherwise, the assembled ND shape as `SmallVector<OpFoldResult>`.
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