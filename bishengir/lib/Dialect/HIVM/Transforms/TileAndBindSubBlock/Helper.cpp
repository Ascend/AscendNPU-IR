//===- Helper.cpp --Helper functions for HIVMTileAndBindSubBlock pass -----===//
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

#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/LogicalResult.h"
#include <cstddef>
#include <optional>

namespace mlir {
namespace hivm {

void markCreatedExtractSliceOp(RewriterBase &rewriter, Operation *op) {
  rewriter.modifyOpInPlace(op, [&]() {
    op->setAttr(toBeBubbleUpSlice, UnitAttr::get(rewriter.getContext()));
  });
}

bool isMarkedExtractSliceOp(Operation *op) {
  return op->hasAttrOfType<UnitAttr>(toBeBubbleUpSlice);
}

OpFoldResult calculateOffsetAtTilingDim(RewriterBase &rewriter, Location loc,
                                        scf::ForOp containingLoop,
                                        OpFoldResult singleTileSize) {
  AffineExpr mulExpr =
      rewriter.getAffineSymbolExpr(0) * rewriter.getAffineSymbolExpr(1);
  OpFoldResult offsetAtTileDim = affine::makeComposedFoldedAffineApply(
      rewriter, loc, mulExpr,
      {containingLoop.getInductionVar(), singleTileSize});
  return offsetAtTileDim;
}

/// This function calculates the tile size by dividing the dimension size
/// by kSubBlockDim (using ceiling division).
///
/// For static dimensions: tile_size = ceil(dim_size / kSubBlockDim)
/// For dynamic dimensions: creates affine operations to compute at runtime
///
/// @param input The input tensor to be tiled
/// @return The computed tile size as an OpFoldResult, or failure if the
///         static dimension size is less than kSubBlockDim
FailureOr<OpFoldResult> getSingleTileSize(OpBuilder &builder, Location loc,
                                          Value input, int64_t tileDimension,
                                          scf::ForOp containingLoop) {
  // Extract the dimension size to be tiled
  auto inputType = dyn_cast<ShapedType>(input.getType());
  if (!inputType)
    return failure();
  auto inputShape = inputType.getShape();

  if (tileDimension > inputType.getRank())
    return failure();

  auto upperBound =
      containingLoop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  if (!upperBound)
    return failure();

  size_t dimensionSize = inputShape[tileDimension];
  if (upperBound.value() < 0)
    return containingLoop.emitError("UpperBound is less than 0");
  size_t denominator = static_cast<size_t>(upperBound.value());

  // Case 1: Static dimension - compute tile size at compile time
  if (!ShapedType::isDynamic(dimensionSize)) {
    if (dimensionSize < denominator) {
      return emitError(loc)
             << "dimension size (" << dimensionSize
             << ") is less than minimum tile size (" << denominator << ")";
    }
    // can be fully divided
    size_t tileSize = llvm::divideCeil(dimensionSize, denominator);
    if (dimensionSize % denominator == 0) {
      return getAsIndexOpFoldResult(builder.getContext(), tileSize);
    }
    if (denominator != static_cast<size_t>(kSubBlockDim))
      return containingLoop.emitError(
          "Tile size not divisible by number of tiles, and number of tiles "
          "doesn't equal to kSubBlockDim");
    auto tailsize =
        dimensionSize - tileSize * (static_cast<size_t>(kSubBlockDim) - 1);
    // Using (1-x) * a + (x) * b
    // will return a if x == 0, and return b if x == 1
    // This can deal with 1:2, need to think of a better formular when doing
    // 1:N
    AffineExpr tileSizeExpr = (1 - builder.getAffineSymbolExpr(0)) * tileSize +
                              (builder.getAffineSymbolExpr(0) * tailsize);
    Value inductionVar = containingLoop.getBody()->getArgument(0);
    auto finalTileSize = affine::makeComposedAffineApply(
        builder, loc, tileSizeExpr, {getAsOpFoldResult(inductionVar)});
    return getAsOpFoldResult(finalTileSize);
  }

  // Case 2: Dynamic dimension - generate runtime computation
  // Create affine expression: ceil(dim0 / kSubBlockDim)
  AffineExpr dim0;
  bindDims(builder.getContext(), dim0);
  auto ceilDivMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                   dim0.ceilDiv(kSubBlockDim));
  auto dimSizeOp = builder.create<tensor::DimOp>(loc, input, tileDimension);
  auto tileSizeOp = builder.create<affine::AffineApplyOp>(
      loc, ceilDivMap, ValueRange{dimSizeOp});
  return getAsOpFoldResult(tileSizeOp);
}

LogicalResult findCorrespondingSizesOffsetsStrides(
    RewriterBase &rewriter, ShapedType rankType, int64_t tilingDim,
    OpFoldResult offsetAtTileDim, OpFoldResult tileSize,
    SmallVector<OpFoldResult, 4> &mixedStrides,
    SmallVector<OpFoldResult, 4> &mixedOffsets,
    SmallVector<OpFoldResult, 4> &mixedSize,
    SmallVector<int64_t, 4> &newShape) {
  for (int i = 0; i < rankType.getRank(); i++) {
    mixedStrides.push_back(rewriter.getIndexAttr(1));
    if (i != tilingDim) {
      mixedOffsets.push_back(rewriter.getIndexAttr(0));
      mixedSize.push_back(getAsIndexOpFoldResult(rewriter.getContext(),
                                                 rankType.getDimSize(i)));
      newShape.push_back(rankType.getDimSize(i));
    } else {
      mixedOffsets.push_back(offsetAtTileDim);
      mixedSize.push_back(tileSize);
      if (!getConstantIntValue(tileSize)) {
        return failure();
      }
      newShape.push_back(getConstantIntValue(tileSize).value());
    }
  }
  return success();
}

static std::optional<ShapedType>
getOriginalType(OffsetSizeAndStrideOpInterface offsetSizeAndStrideOp) {
  if (auto op = dyn_cast<tensor::ExtractSliceOp>(
          offsetSizeAndStrideOp.getOperation()))
    return op.getSourceType();
  if (auto op =
          dyn_cast<tensor::InsertSliceOp>(offsetSizeAndStrideOp.getOperation()))
    return op.getDestType();
  if (auto op =
          dyn_cast<memref::SubViewOp>(offsetSizeAndStrideOp.getOperation()))
    return op.getSourceType();
  llvm_unreachable("There should not be such case");
  return std::nullopt;
}

DenseSet<size_t> getExtractOrInsertDim(OffsetSizeAndStrideOpInterface op) {
  auto originalType = getOriginalType(op);
  if (!originalType.has_value()) {
    return {};
  }
  auto extractSize = op.getStaticSizes();
  DenseSet<size_t> extractDims;
  for (size_t dim = 0; dim < extractSize.size(); dim++) {
    if (ShapedType::isDynamic(extractSize[dim]) ||
        extractSize[dim] != originalType->getDimSize(dim)) {
      extractDims.insert(dim);
    }
  }
  return extractDims;
}

DenseSet<size_t> getIntersectionDims(DenseSet<size_t> dims1,
                                     const DenseSet<size_t> &dims2) {
  llvm::set_intersect(dims1, dims2);
  return dims1;
}

static bool checkStridesCreatedByTiling(llvm::ArrayRef<int64_t> strides) {
  // Check Strides
  // We only support unstrided slice now
  for (auto stride : strides) {
    if (stride != 1)
      return false;
  }
  return true;
}

static FailureOr<std::pair<int64_t, int64_t>>
checkSizesCreatedByTiling(ArrayRef<int64_t> sizes, ArrayRef<int64_t> srcShape,
                          size_t tilingDim, scf::ForOp tilingLoop) {
  // Check Sizes
  // All sizes should match source shape except at tiling dim
  assert(srcShape.size() == sizes.size());
  for (size_t dim = 0; dim < srcShape.size(); dim++) {
    if (dim == tilingDim)
      continue;
    if (srcShape[dim] != sizes[dim])
      return failure();
  }
  // At tiling dim, size should be source shape divided by tileCounts
  // Find tileCount, which should be upper bound of normalized tiling loop.
  if (!scf::utils::isNormalized(tilingLoop))
    return failure();
  auto upperBoundOp =
      tilingLoop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  if (!upperBoundOp)
    return failure();
  int64_t tileCounts = upperBoundOp.value();
  // Can be fully divided.
  if (srcShape[tilingDim] % tileCounts != 0)
    return failure();
  int64_t tileSize = srcShape[tilingDim] / tileCounts;
  // And extractSlize at tiling dim should be tile size.
  if (tileSize != sizes[tilingDim])
    return failure();
  return std::make_pair(tileSize, tileCounts);
}

static FailureOr<int64_t>
calculateMapWithN(OpBuilder &builder, affine::AffineApplyOp offsetAffineMap,
                  int64_t n) {
  // Compute the offset when loop index is 2
  SmallVector<Attribute> tileOffsets2;
  if (failed(offsetAffineMap.getAffineMap().constantFold(
          {builder.getI64IntegerAttr(n)}, tileOffsets2))) {
    return false;
  }
  auto tileOffsetAttr2 = dyn_cast<IntegerAttr>(tileOffsets2[0]);
  if (!tileOffsetAttr2)
    return false;
  return tileOffsetAttr2.getInt();
}

static bool calculateMapAndVerifyResult(OpBuilder &builder,
                                        affine::AffineApplyOp offsetAffineMap,
                                        int64_t n, int64_t expectedResult) {
  // Compute the offset when loop index is 0
  auto calculatedResult = calculateMapWithN(builder, offsetAffineMap, n);
  // Offset at loop 0 should be 0
  return !failed(calculatedResult) && calculatedResult == expectedResult;
}

static bool checkOffsetsCreatedByTiling(ArrayRef<int64_t> staticOffsets,
                                        ValueRange offsets,
                                        ArrayRef<int64_t> srcShape,
                                        size_t tilingDim, scf::ForOp tilingLoop,
                                        int64_t tileSize, int64_t tileCounts) {
  OpBuilder builder(tilingLoop->getContext());
  // Check offset
  // Offsets at non-tiling dim should be 0
  for (size_t dim = 0; dim < staticOffsets.size(); dim++) {
    if (dim == tilingDim)
      continue;
    if (staticOffsets[dim] != 0)
      return false;
  }

  // Offset at tiling dim should be tileSize * loop index;
  // First check the affine map is calculating N * tileSize;
  auto offsetAffineMap =
      offsets[tilingDim].getDefiningOp<affine::AffineApplyOp>();
  // If it's created by tiling, then the offset at tiling dim must be
  // calculated by AffineApplyOp.
  if (!offsetAffineMap)
    return false;
  // This map should only take 1 operand
  if (offsetAffineMap.getMapOperands().size() != 1)
    return false;
  // and the operand has to be the loop index
  auto blockArg = dyn_cast<BlockArgument>(offsetAffineMap.getMapOperands()[0]);
  if (!blockArg || blockArg != tilingLoop.getInductionVar())
    return false;

  // Verify offsets at multiple loop index
  if ( // Offset at loop 0 should be 0.
      !calculateMapAndVerifyResult(builder, offsetAffineMap, 0, 0) ||
      // At iteration 1, offset should equals to tile size.
      !calculateMapAndVerifyResult(builder, offsetAffineMap, 1, tileSize) ||
      // The diff of offset between tiling loop 1 and 2 should be tile size too.
      !calculateMapAndVerifyResult(builder, offsetAffineMap, 2,
                                   tileSize + tileSize) ||
      // Offset at last iteration should be shape - tilesize.
      !calculateMapAndVerifyResult(builder, offsetAffineMap, tileCounts - 1,
                                   srcShape[tilingDim] - tileSize)) {
    return false;
  }

  // All checked
  return true;
}

bool createdByTiling(OffsetSizeAndStrideOpInterface offsetSizeAndStrideOp) {
  // Get tiling loop
  auto tilingLoop = offsetSizeAndStrideOp->getParentOfType<scf::ForOp>();
  if (!tilingLoop)
    return false;

  // Get tiling Dim
  auto extractDims = getExtractOrInsertDim(offsetSizeAndStrideOp);
  // We can only handle tiling 1 dimension for now.
  if (extractDims.size() != 1)
    return false;
  auto tilingDim = *extractDims.begin();

  if (!checkStridesCreatedByTiling(offsetSizeAndStrideOp.getStaticStrides()))
    return false;

  auto originalShape = getOriginalType(offsetSizeAndStrideOp);
  auto maybeTileSizeCountPair = checkSizesCreatedByTiling(
      offsetSizeAndStrideOp.getStaticSizes(), originalShape->getShape(),
      tilingDim, tilingLoop);
  if (failed(maybeTileSizeCountPair))
    return false;
  auto tileSize = maybeTileSizeCountPair.value().first;
  auto tileCount = maybeTileSizeCountPair.value().second;
  if (!checkOffsetsCreatedByTiling(offsetSizeAndStrideOp.getStaticOffsets(),
                                   offsetSizeAndStrideOp.getOffsets(),
                                   originalShape->getShape(), tilingDim,
                                   tilingLoop, tileSize, tileCount)) {
    return false;
  }

  // All checked, we are safe to conclude this insert is from tiling.
  return true;
}

} // namespace hivm
} // namespace mlir