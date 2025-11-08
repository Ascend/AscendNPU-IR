//===- ElementwisePattern.cpp ---------------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//============================================================================//

#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/Pattern.h"
#include "bishengir/Dialect/HIVM/Transforms/HIVMTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"

#define DEBUG_TYPE "elementwise-bubble-up-extract-slice"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << (X) << "\n")

namespace mlir::hivm::detail {

namespace {

static constexpr llvm::StringLiteral toBeBubbleUpSlice = "to_be_bubbled_slice";

static bool isTiled(AffineExpr expr, ArrayRef<OpFoldResult> tileSizes) {
  if (!expr)
    return false;
  mlir::hivm::tiling_helper::TileCheck t(tileSizes);
  t.visit(expr);
  return t.isTiled;
}

// Checks whether the `map  varies with respect to a non-zero `tileSize`.
static bool isTiled(AffineMap map, ArrayRef<OpFoldResult> tileSizes) {
  if (!map)
    return false;
  for (unsigned r = 0; r < map.getNumResults(); ++r)
    if (isTiled(map.getResult(r), tileSizes))
      return true;
  return false;
}

struct SliceParameters {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
};

// This function refer on the implmentation of linalg bubbleUpExtractSlice.
static SliceParameters
computeSliceParameters(OpBuilder &builder, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> tileSizes, AffineMap map,
                       ArrayRef<OpFoldResult> lbs, ArrayRef<OpFoldResult> ubs,
                       ArrayRef<OpFoldResult> subShapeSizes,
                       bool omitPartialTileCheck) {
  auto shapedType = dyn_cast<ShapedType>(valueToTile.getType());
  assert(shapedType && "only shaped types can be tiled");
  ArrayRef<int64_t> shape = shapedType.getShape();
  int64_t rank = shapedType.getRank();

  // Compute offsets/sizes/strides for the tile.
  SliceParameters sliceParams;
  sliceParams.offsets.reserve(rank);
  sliceParams.sizes.reserve(rank);
  sliceParams.strides.reserve(rank);

  DenseSet<size_t> inlineBrcAxes;
  for (auto [i, result] : llvm::enumerate(map.getResults())) {
    auto constExpr = dyn_cast<AffineConstantExpr>(result);
    if (!constExpr)
      continue;
    if (constExpr.getValue() != 0)
      continue;
    inlineBrcAxes.insert(i);
  }

  for (unsigned r = 0; r < rank; ++r) {
    LLVM_DEBUG(llvm::dbgs() << "computeSliceParameters: for dim#" << r);
    if (!isTiled(map.getSubMap({r}), tileSizes)) {
      sliceParams.offsets.push_back(builder.getIndexAttr(0));
      OpFoldResult dim =
          mlir::linalg::createFoldedDimOp(builder, loc, valueToTile, r);
      if (inlineBrcAxes.contains(r)) {
        sliceParams.sizes.push_back(builder.getIndexAttr(1));
      } else {
        sliceParams.sizes.push_back(dim);
      }
      sliceParams.strides.push_back(builder.getIndexAttr(1));
      LLVM_DEBUG(llvm::dbgs() << ": not tiled: use size: " << dim << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subsize...\n");

    // Tiling creates a new slice at the proper index, the slice step is 1
    // (i.e. the op does not subsample, stepping occurs in the loop).
    auto m = map.getSubMap({r});
    LLVM_DEBUG(llvm::dbgs() << "computeSliceParameters: submap: " << m << "\n");
    IRRewriter rewriter(builder);
    OpFoldResult offset =
        mlir::affine::makeComposedFoldedAffineApply(rewriter, loc, m, lbs);
    sliceParams.offsets.push_back(offset);
    OpFoldResult closedIntSize = mlir::affine::makeComposedFoldedAffineApply(
        rewriter, loc, m, subShapeSizes);
    // Resulting size needs to be made half open interval again.
    AffineExpr s0 = getAffineSymbolExpr(0, builder.getContext());
    OpFoldResult size = mlir::affine::makeComposedFoldedAffineApply(
        rewriter, loc, s0 + 1, closedIntSize);
    LLVM_DEBUG(llvm::dbgs()
               << "computeSliceParameters: raw size: " << size << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "computeSliceParameters: new offset: " << offset << "\n");
    sliceParams.strides.push_back(builder.getIndexAttr(1));

    if (omitPartialTileCheck) {
      // We statically know that the partial/boundary tile condition is
      // unnecessary.
      if (inlineBrcAxes.contains(r)) {
        sliceParams.sizes.push_back(builder.getIndexAttr(1));
        LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: new size: " << 1 << "\n");
      } else {
        sliceParams.sizes.push_back(size);
        LLVM_DEBUG(llvm::dbgs()
                   << "makeTiledShape: new size: " << size << "\n");
      }
      continue;
    }

    // The size of the subview / extract_slice should be trimmed to avoid
    // out-of-bounds accesses, unless:
    // a. We statically know the subshape size divides the shape size evenly.
    // b. The subshape size is 1. According to the way the loops are set up,
    //    tensors with "0" dimensions would never be constructed.
    int64_t shapeSize = shape[r];
    std::optional<int64_t> sizeCst = getConstantIntValue(size);
    auto hasTileSizeOne = sizeCst && *sizeCst == 1;
    auto dividesEvenly = sizeCst && !ShapedType::isDynamic(shapeSize) &&
                         ((shapeSize % *sizeCst) == 0);
    if (!hasTileSizeOne && !dividesEvenly) {
      LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: shapeSize=" << shapeSize
                              << ", size: " << size
                              << ": make sure in bound with affine.min\n");

      AffineExpr dim0, dim1, dim2;
      MLIRContext *context = builder.getContext();
      bindDims(context, dim0, dim1, dim2);

      // Get the dimension size for this dimension. We need to first calculate
      // the max index and then plus one. This is important because for
      // convolution ops, we have its input window dimension's affine map of the
      // form `(d0 * s0 + d1)`, where `d0`/`d1 is an output/filter window
      // dimension and `s0` is stride. Directly use the dimension size of
      // output/filer window dimensions will cause incorrect calculation.
      AffineMap minusOneMap = AffineMap::inferFromExprList(
                                  {ArrayRef<AffineExpr>{dim0 - 1}}, context)
                                  .front();
      AffineMap plusOneMap = AffineMap::inferFromExprList(
                                 {ArrayRef<AffineExpr>{dim0 + 1}}, context)
                                 .front();
      SmallVector<OpFoldResult> maxIndices = llvm::to_vector(
          llvm::map_range(ubs, [&rewriter, &loc, &minusOneMap](OpFoldResult ub) {
            return mlir::affine::makeComposedFoldedAffineApply(
                rewriter, loc, minusOneMap, {ub});
          }));
      OpFoldResult maxIndex = mlir::affine::makeComposedFoldedAffineApply(
          rewriter, loc, m, maxIndices);
      OpFoldResult d = mlir::affine::makeComposedFoldedAffineApply(
          rewriter, loc, plusOneMap, {maxIndex});

      // Compute min(dim - offset, size) to avoid out-of-bounds accesses.
      AffineMap minMap = AffineMap::inferFromExprList(
                             {ArrayRef<AffineExpr>{dim1 - dim2, dim0}}, context)
                             .front();
      size = mlir::affine::makeComposedFoldedAffineMin(rewriter, loc, minMap,
                                                       {size, d, offset});
    }
    if (inlineBrcAxes.contains(r)) {
      sliceParams.sizes.push_back(builder.getIndexAttr(1));
      LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: new size: " << 1 << "\n");
    } else {
      sliceParams.sizes.push_back(size);
      LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: new size: " << size << "\n");
    }
  }
  return sliceParams;
}

// This function refer on the implmentation of linalg bubbleUpExtractSlice.
static SmallVector<std::optional<SliceParameters>> computeAllSliceParameters(
    OpBuilder &builder, Location loc, HIVMStructuredOp hivmOp,
    ValueRange valuesToTile, ArrayRef<OpFoldResult> ivs,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> sizeBounds,
    bool omitPartialTileCheck) {
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](OpFoldResult v) { return !isZeroIndex(v); })) &&
         "expected as many ivs as non-zero sizes");

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subshapes.
  SmallVector<OpFoldResult> lbs =
      mlir::linalg::computeTileOffsets(builder, loc, ivs, tileSizes);
  SmallVector<OpFoldResult> subShapeSizes =
      mlir::linalg::computeTileSizes(builder, loc, tileSizes, sizeBounds);

  assert(static_cast<int64_t>(valuesToTile.size()) <=
             hivmOp->getNumOperands() &&
         "more value to tile than operands.");
  SmallVector<std::optional<SliceParameters>> allSliceParams;
  allSliceParams.reserve(valuesToTile.size());
  for (auto [opOperand, val] :
       llvm::zip(hivmOp->getOpOperands(), valuesToTile)) {
    Value shapedOp = val;
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for operand " << shapedOp);
    AffineMap map = hivmOp.getMatchingIndexingMap(&opOperand);
    // Use `opOperand` as is if it is not tiled and not an output tensor. Having
    // an extract/insert slice pair for all output tensors simplifies follow up
    // transformations such as padding and bufferization since the
    // extract/insert slice pairs make the accessed iteration argument
    // subdomains explicit.

    Type operandType = opOperand.get().getType();
    if (!isTiled(map, tileSizes) &&
        !(isa<RankedTensorType>(operandType) && hivmOp.isDpsInit(&opOperand))) {
      allSliceParams.push_back(std::nullopt);
      LLVM_DEBUG(llvm::dbgs()
                 << ": not tiled: use shape: " << operandType << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subshape...\n");

    allSliceParams.push_back(computeSliceParameters(
        builder, loc, shapedOp, tileSizes, map, lbs, sizeBounds, subShapeSizes,
        omitPartialTileCheck));
  }

  return allSliceParams;
}

// This function refer on the implmentation of linalg bubbleUpExtractSlice.
static Value materializeTiledShape(OpBuilder &builder, Location loc,
                                   Value valueToTile,
                                   const SliceParameters &sliceParams) {
  auto shapedType = dyn_cast<ShapedType>(valueToTile.getType());
  auto *sliceOp = TypeSwitch<ShapedType, Operation *>(shapedType)
                      .Case([&builder, &loc, &valueToTile,
                             &sliceParams](MemRefType) {
                        return builder.create<memref::SubViewOp>(
                            loc, valueToTile, sliceParams.offsets,
                            sliceParams.sizes, sliceParams.strides);
                      })
                      .Case([&builder, &loc, &valueToTile,
                             &sliceParams](RankedTensorType) {
                        return builder.create<tensor::ExtractSliceOp>(
                            loc, valueToTile, sliceParams.offsets,
                            sliceParams.sizes, sliceParams.strides);
                      })
                      .Default([](ShapedType) -> Operation * {
                        llvm_unreachable("Unexpected shaped type");
                      });
  if (isa<tensor::ExtractSliceOp>(sliceOp)) {
    sliceOp->setAttr(toBeBubbleUpSlice, UnitAttr::get(builder.getContext()));
  }
  return sliceOp->getResult(0);
}

// This function refer on the implmentation of linalg bubbleUpExtractSlice.
SmallVector<Value>
makeTiledShapes(OpBuilder &builder, Location loc, HIVMStructuredOp hivmOp,
                ValueRange valuesToTile, ArrayRef<OpFoldResult> ivs,
                ArrayRef<OpFoldResult> tileSizes,
                ArrayRef<OpFoldResult> sizeBounds, bool omitPartialTileCheck) {
  SmallVector<std::optional<SliceParameters>> allSliceParameter =
      computeAllSliceParameters(builder, loc, hivmOp, valuesToTile, ivs,
                                tileSizes, sizeBounds, omitPartialTileCheck);
  SmallVector<Value> tiledShapes;
  for (auto item : llvm::zip(valuesToTile, allSliceParameter)) {
    Value valueToTile = std::get<0>(item);
    std::optional<SliceParameters> sliceParams = std::get<1>(item);
    tiledShapes.push_back(
        sliceParams.has_value()
            ? materializeTiledShape(builder, loc, valueToTile, *sliceParams)
            : valueToTile);
  }
  return tiledShapes;
}
} // namespace

static bool isDynamicSlice(tensor::ExtractSliceOp op) {
  return ShapedType::isDynamicShape(op.getStaticSizes());
}

bool ElementwiseBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  // Check if source is a block argument of a for loop
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  if (!sourceOp) {
    return false;
  }
  bool isValidElementwise =
      (isElemwiseNaryOpImpl(sourceOp) ||
       isa<hivm::LoadOp, hivm::StoreOp, hivm::CopyOp>(sourceOp)) &&
      !isDynamicSlice(sliceOp);
  return isValidElementwise;
}

LogicalResult
ElementwiseBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                     PatternRewriter &rewriter) const {
  auto hivmOp = dyn_cast<HIVMStructuredOp>(sliceOp.getSource().getDefiningOp());
  if (!hivmOp)
    return failure();
  rewriter.setInsertionPoint(hivmOp);

  OpOperand *outOperand = hivmOp.getDpsInitOperand(0);
  AffineMap indexingMap = hivmOp.getMatchingIndexingMap(outOperand);
  if (!indexingMap.isProjectedPermutation()) {
    return rewriter.notifyMatchFailure(
        sliceOp, "expected a projected permutation for output");
  }

  auto hivmLoc = hivmOp.getLoc();
  SmallVector<OpFoldResult> allShapeSizes =
      hivmOp.createFlatListOfOperandDims(rewriter, hivmLoc);
  AffineMap shapeSizesToLoopsMap = hivmOp.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap) {
    return rewriter.notifyMatchFailure(
        hivmOp, "failed to get loops map from shape sizes");
  }
  SmallVector<OpFoldResult> sizeBounds =
      affine::makeComposedFoldedMultiResultAffineApply(
          rewriter, hivmLoc, shapeSizesToLoopsMap, allShapeSizes);

  // The offsets and sizes from the slice operation only give you the tile
  // size of the output. Use that compute the tile sizes and offsets of the
  // loops. For loops not used to access the output, set the tile sizes to
  // loop bounds and set the offset to 0.
  SmallVector<OpFoldResult> tileOffsets(sizeBounds.size(),
                                        rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> tileSizes = sizeBounds;
  for (auto const &result : enumerate(indexingMap.getResults())) {
    unsigned position = cast<AffineDimExpr>(result.value()).getPosition();
    tileOffsets[position] = sliceOp.getMixedOffsets()[result.index()];
    tileSizes[position] = sliceOp.getMixedSizes()[result.index()];
  }

  SmallVector<Value> valuesToTile = hivmOp->getOperands();
  SmallVector<Value> tiledOperands =
      makeTiledShapes(rewriter, hivmLoc, hivmOp, valuesToTile, tileOffsets,
                      tileSizes, sizeBounds,
                      /*omitPartialTileCheck=*/true);

  SmallVector<Type, 4> resultTensorTypes;
  for (OpOperand &opOperand : hivmOp.getDpsInitsMutable())
    resultTensorTypes.push_back(
        tiledOperands[opOperand.getOperandNumber()].getType());

  rewriter.setInsertionPointAfter(hivmOp);
  Operation *newOp = clone(rewriter, hivmOp, resultTensorTypes, tiledOperands);
  rewriter.replaceOp(sliceOp, newOp->getResults());
  return success();
}

} // namespace mlir::hivm::detail
