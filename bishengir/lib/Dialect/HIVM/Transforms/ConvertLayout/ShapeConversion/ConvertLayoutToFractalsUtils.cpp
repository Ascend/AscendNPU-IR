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

/// Aggregates the operands required to compute the fractal shape that results
/// from converting an ND tensor to a fractal-tiled layout.
///
/// All fields are populated by extractNDToFractalConversionParams and then
/// consumed by assembleFractalShape. Keeping them in a single struct avoids
/// threading a large number of individual arguments through the call chain.
struct NDToFractalConversionParams {

  /// The size of the first spatial dimension (rows) of the source ND tensor.
  OpFoldResult a;

  /// The size of the second spatial dimension (columns) of the source ND
  /// tensor.
  OpFoldResult b;

  /// The two fractal tile dimensions $(f_0, f_1)$ extracted from the
  /// destination layout attribute. All tile dimensions are static.
  FractalSize fractalSize{};

  /// Offset applied to dimension indices when the source tensor is rank-3
  /// (batched). Set to 1 for rank-3 tensors and 0 for rank-2 tensors.
  int batchIndexBias{};

  /// The leading batch dimension of a rank-3 source tensor, expressed as a
  /// mixed static/dynamic OpFoldResult. Only valid when batchIndexBias == 1.
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

/// Extract and validate all parameters needed for the ND→fractal shape
/// computation.
///
/// The function performs:
/// - Retrieves the fractal tile sizes $(f_0, f_1)$ from dstLayout via
///   extractBlockSizes.
/// - Determines whether the tensor is batched (rank 3) or not (rank 2) and
///   records the corresponding index bias.
/// - Slices the spatial dimensions a and b out of currentShape,
///   skipping the leading batch dimension when present.
static FailureOr<NDToFractalConversionParams>
extractNDToFractalConversionParams(ArrayRef<OpFoldResult> currentShape,
                                   DataLayoutAttr srcLayout,
                                   DataLayoutAttr dstLayout,
                                   MLIRContext *ctx) {
  LDBG("=== extractNDToFractalConversionParams ===");
  auto blockSizesResult = dstLayout.getFractalBlockSizes();
  if (failed(blockSizesResult))
    return failure();

  NDToFractalConversionParams params;
  params.fractalSize = *blockSizesResult;

  params.batchIndexBias = computeBatchIndexBias(currentShape.size());
  LDBG("Batch index bias: " << params.batchIndexBias);

  if (params.batchIndexBias > 0) {
    params.batch = currentShape[0];
  }
  params.a = currentShape[0 + params.batchIndexBias];
  params.b = currentShape[1 + params.batchIndexBias];

  LDBG("Applied params");
  return params;
}

/// Assemble the fractal output shape from pre-computed tile counts and the
/// static block dimensions stored in params.
///
/// The resulting shape always has the form:
/// [aTiles, bTiles, f_0, f_1]
/// for rank-2 inputs, or
/// [batch, aTiles, bTiles, f_0, f_1]
/// for rank-3 (batched) inputs, where $f_0$ and $f_1$ are the static fractal
/// tile sizes taken from params.fractalSize.
///
/// Because $f_0$ and $f_1$ are always static, they are represented as
/// IntegerAttr fold results and no OpBuilder is required.
static SmallVector<OpFoldResult>
assembleFractalShape(OpFoldResult aTiles, OpFoldResult bTiles,
                     const NDToFractalConversionParams &params, MLIRContext *ctx) {

  LDBG("=== assembling fractal shape ===");
  LDBG("Tile a: " << aTiles);
  LDBG("Tile b: " << bTiles);
  SmallVector<OpFoldResult> fractalShape = {
      aTiles, bTiles, getAsIndexOpFoldResult(ctx, params.fractalSize.first),
      getAsIndexOpFoldResult(ctx, params.fractalSize.second)};
  if (params.batchIndexBias > 0) {
    LDBG("Inserting batch dimension");
    fractalShape.insert(fractalShape.begin(), params.batch);
  }
  return fractalShape;
}

//===----------------------------------------------------------------------===//
// Public API - Mixed Shape Computation
//===----------------------------------------------------------------------===//

/// Compute the mixed static/dynamic result shape for an ND → fractal layout
/// conversion.
///
/// This helper accepts a source tensor shape (`currentShape`) represented as
/// `OpFoldResult`s and produces the destination fractal shape with folded
/// affine arithmetic where possible. (Affine automatically supports
/// OpFoldResult)
///
/// Supported source ranks:
/// - Rank-2: `[a, b]`
/// - Rank-3 (batched): `[batch, a, b]`
///
/// Let the destination fractal tile sizes be `(f0, f1)`. The output shape is:
/// - Rank-2 input: `[ceildiv(b, f1), ceildiv(a, f0), f0, f1]`
/// - Rank-3 input: `[batch, ceildiv(b, f1), ceildiv(a, f0), f0, f1]`
///
/// Implementation notes:
/// - Tile sizes are extracted from `dstLayout` via
///   `extractNDToFractalConversionParams`.
/// - `affine::makeComposedFoldedAffineApply` is used for ceildiv so static
///   values are folded to attributes and dynamic values materialize
///   `affine.apply` only when needed.
///
/// Returns:
/// - `failure()` if fractal block sizes cannot be extracted from `dstLayout`.
/// - Otherwise, the assembled fractal shape as `SmallVector<OpFoldResult>`.
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