//===- TilingInterfaceImpl.cpp - Implementation of TilingInterface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/TilingInterfaceImpl.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/Support/Debug.h"

#include <optional>

using namespace mlir;
using namespace mlir::tensor;

#define DEBUG_TYPE "bishengir-tensor-tiling"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
/// Flag for uninitialized value.
static constexpr int64_t kNotInited = -1;

//----------------------------------------------------------------------------//
// ConcatOpTiling
//----------------------------------------------------------------------------//

struct Segment {
  OpFoldResult offset;
  OpFoldResult end;
};

SmallVector<Segment> getConcatSegments(tensor::ConcatOp concatOp,
                                       OpBuilder &b) {
  // create affine add using lambda
  AffineExpr dim0;
  AffineExpr dim1;
  bindDims(b.getContext(), dim0, dim1);
  auto addMap = AffineMap::get(2, 0, {dim0 + dim1});
  auto addFn = [&b, &addMap](OpFoldResult v1, OpFoldResult v2,
                             Location inputLoc) {
    return affine::makeComposedFoldedAffineApply(b, inputLoc, addMap, {v1, v2});
  };

  uint64_t concatDim = concatOp.getDim();
  SmallVector<Segment> segments;
  OpFoldResult start = b.getIndexAttr(0);
  for (auto [idx, input] : llvm::enumerate(concatOp.getInputs())) {
    ShapedType inputShape = cast<ShapedType>(input.getType());
    Location loc = input.getLoc();
    OpFoldResult size;
    if (inputShape.isDynamicDim(concatDim)) {
      SmallVector<OpFoldResult> inputShapeValues =
          tensor::getMixedSizes(b, loc, input);
      size = inputShapeValues[concatDim];
    } else {
      int64_t staticSize = inputShape.getDimSize(concatDim);
      size = b.getIndexAttr(staticSize);
    }

    OpFoldResult end = addFn(start, size, loc);
    segments.push_back({start, end});
    start = end;
  }
  return segments;
}

FailureOr<TilingResult> bubbleUpConcatSlice(OpBuilder &b,
                                            tensor::ConcatOp concatOp,
                                            ArrayRef<OpFoldResult> offsets,
                                            ArrayRef<OpFoldResult> sizes) {
  // only support concat with static sizes.
  SmallVector<Segment> segments = getConcatSegments(concatOp, b);

  // Helper variables and functions for various arithmetic operations. These
  // are used extensively for computing new offset/length and padding values.
  Location loc = concatOp->getLoc();
  AffineExpr dim0;
  AffineExpr dim1;
  bindDims(b.getContext(), dim0, dim1);
  // Subtract two integers.
  auto subMap = AffineMap::get(2, 0, {dim0 - dim1});
  auto sub = [&b, &loc, &subMap](OpFoldResult v1, OpFoldResult v2) {
    return affine::makeComposedFoldedAffineApply(b, loc, subMap, {v1, v2});
  };
  auto addMap = AffineMap::get(2, 0, {dim0 + dim1});
  auto add = [&b, &loc, &addMap](OpFoldResult v1, OpFoldResult v2) {
    return affine::makeComposedFoldedAffineApply(b, loc, addMap, {v1, v2});
  };
  // Take the minimum of two integers.
  auto idMap = AffineMap::getMultiDimIdentityMap(2, b.getContext());
  auto min = [&b, &loc, &idMap](OpFoldResult v1, OpFoldResult v2) {
    return affine::makeComposedFoldedAffineMin(b, loc, idMap, {v1, v2});
  };
  // Take the maximum of two integers.
  auto max = [&b, &loc, &idMap](OpFoldResult v1, OpFoldResult v2) {
    return affine::makeComposedFoldedAffineMax(b, loc, idMap, {v1, v2});
  };
  // Zero index-typed integer.
  OpFoldResult zero = b.getIndexAttr(0);
  // Select based on condition.
  auto select = [&b, &loc](Value condV, Value trueV, Value falseV) {
    return b.create<arith::SelectOp>(loc, b.getIndexType(), condV, trueV,
                                     falseV);
  };
  // Take the logical and based on two conditions.
  auto logicAnd = [&b, &loc](Value cond1, Value cond2) {
    return b.create<arith::AndIOp>(loc, cond1, cond2);
  };
  // Compare greater or equal than, i.e. left >= right.
  auto ge = [&b, &loc](Value left, Value right) {
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, left, right);
  };
  // Compare less than, i.e. left < right.
  auto lt = [&b, &loc](Value left, Value right) {
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, left, right);
  };
  // Convert OpFoldResult to Value.
  auto toValue = [&b, &loc](OpFoldResult ofr) {
    if (ofr.is<Value>())
      return ofr.get<Value>();
    return b.create<arith::ConstantIndexOp>(loc, *getConstantIntValue(ofr))
        .getResult();
  };

  int64_t rank = concatOp.getResultType().getRank();
  SmallVector<SmallVector<OpFoldResult>> allNewOffsets;
  SmallVector<SmallVector<OpFoldResult>> allNewLengths;
  SmallVector<SmallVector<OpFoldResult>> allNewStrides;
  uint64_t concatDim = concatOp.getDim();
  for (auto [idx, input] : llvm::enumerate(concatOp.getInputs())) {
    SmallVector<OpFoldResult> newOffsets, newLengths, newStrides;

    for (unsigned dim = 0; dim < rank; ++dim) {
      auto offset = offsets[dim];
      auto length = sizes[dim];
      if (dim == concatDim) {
        // Tile concat dim.
        // concat: |  arg0  |     arg1     |   arg2   |     arg3     |
        //         |--------|--------------|----------|--------------|
        // tile:                  ^offset                   ^offset+length
        // ------------------------------------------------------------------
        // for each segment: overlap means the conjunction of:
        // 1. tile.offset < seg.end
        // 2. tile.offset + tile.length >= seg.offset
        // ------------------------------------------------------------------
        // arg0 and arg3: if tile has no overlap:
        // 1. newOffset: 0
        // 2. newLength: 0
        // ------------------------------------------------------------------
        // arg1: if tile has overlap with seg right part:
        // -- tile.offset >= seg.offset
        // 1. newOffset: tile.offset - seg.offset
        // 2. newLength: min(tile.length, seg.end - tile.offset)
        // ------------------------------------------------------------------
        // arg2: if tile has overlap with seg whole part:
        // -- tile.offset < seg.offset
        // -- tile.end >= seg.end
        // 1. newOffset: 0
        // 2. newLength: seg.length
        // ------------------------------------------------------------------
        // arg3: if tile has overlap with seg left part:
        // -- tile.offset < seg.offset
        // -- tile.end <= seg.end
        // 1. newOffset: 0
        // 2. newLength: tile.end - seg.offset
        // ------------------------------------------------------------------
        // arg2 & arg3: if tile has overlap with seg left or whole part:
        // -- tile.offset < seg.offset
        // 1. newOffset: 0
        // 2. newLength: min(tile.end - seg.offset, seg.length)
        // ------------------------------------------------------------------
        // the overlap conditions for offset can be combined:
        // -  newOffset: max(tile.offset - seg.offset, 0)

        Segment seg = segments[idx];
        Value zeroV = toValue(zero);
        Value tileOffsetV = toValue(offset);
        Value tileEndV = toValue(add(offset, length));
        Value segOffsetV = toValue(seg.offset);
        Value segEndV = toValue(seg.end);
        Value segLengthV = toValue(sub(seg.end, seg.offset));

        // (tile.offset < seg.end) and (tile.end >= seg.offset)
        Value hasOverlap =
            logicAnd(lt(tileOffsetV, segEndV), ge(tileEndV, segOffsetV));
        // Since tile has overlap with segment, offset will be:
        // 1. zero, when tile.offset < seg.offset
        // 2. tile.offset - seg.offset, otherwise
        // Use max() to combine the two cases
        Value curOffsetV = toValue(max(zero, sub(offset, segOffsetV)));
        Value newOffset = select(hasOverlap, curOffsetV, zeroV);

        // if has overlap, cur length will be:
        // 1. min(tile.end - seg.offset, seg.length), if tile start before seg
        // 2. min(tile.length, seg.end - tile.offset), if tile start within seg
        Value tileBeforeSeg = lt(tileOffsetV, segOffsetV);
        Value length1 = toValue(min(sub(tileEndV, segOffsetV), segLengthV));
        Value length2 = toValue(min(length, sub(segEndV, tileOffsetV)));
        Value curLengthV = select(tileBeforeSeg, length1, length2);
        Value newLength = select(hasOverlap, curLengthV, zeroV);
        newOffsets.push_back(newOffset);
        newLengths.push_back(newLength);
      } else {
        // Tile non-concat dim.
        auto dimSize = tensor::getMixedSize(b, loc, input, dim);
        // In case tile.offset exceeds dimSize:
        // newOffset = min(tile.offset, dimSize)
        OpFoldResult newOffset = min(offset, dimSize);
        // In case tile range exceeds dimSize:
        // newLength = min(dimSize - newOffset, tile.length)
        // optimize `newLength` of dynamic shape if tile size is constant one
        OpFoldResult newLength = isConstantIntValue(length, 1)
                                     ? length
                                     : min(sub(dimSize, newOffset), length);
        newOffsets.push_back(newOffset);
        newLengths.push_back(newLength);
      }
      // Only unit stride supported.
      newStrides.push_back(b.getIndexAttr(1));
    }
    allNewOffsets.push_back(newOffsets);
    allNewLengths.push_back(newLengths);
    allNewStrides.push_back(newStrides);
  }

  // The shape of the result can be obtained from the sizes passed in.
  SmallVector<Value> dynConcatResShapes;
  SmallVector<int64_t> staticConcatResShapes;
  dispatchIndexOpFoldResults(sizes, dynConcatResShapes, staticConcatResShapes);
  RankedTensorType resultType = RankedTensorType::get(
      staticConcatResShapes, concatOp.getResultType().getElementType());

  // Create concat(extract_slice(x), extract_slice(y), ...).
  SmallVector<Value> newSliceOps;
  for (auto [idx, input] : llvm::enumerate(concatOp.getInputs())) {
    Value newSliceOp = b.create<tensor::ExtractSliceOp>(
        loc, input, allNewOffsets[idx], allNewLengths[idx], allNewStrides[idx]);
    newSliceOps.push_back(newSliceOp);
  }
  // use the extract_slice sizes
  auto newConcatOp =
      b.create<ConcatOp>(loc, resultType, concatOp.getDim(), newSliceOps);
  newConcatOp->setAttrs(
      getPrunedAttributeList(concatOp, ConcatOp::getAttributeNames()));
  return TilingResult{{newConcatOp}, {newConcatOp->getResult(0)}};
}

struct ConcatOpTiling
    : public TilingInterface::ExternalModel<ConcatOpTiling, ConcatOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto concatOp = cast<ConcatOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        concatOp.getResultType().getRank(), utils::IteratorType::concat);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    ReifiedRankedShapedTypeDims reifiedShapes;
    (void)reifyResultShapes(b, op, reifiedShapes);
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    // Initialize all the ranges to {zero, one, one}. All the `ub`s are
    // overwritten.
    SmallVector<Range> loopRanges(reifiedShapes[0].size(), {zero, one, one});
    for (const auto &ub : enumerate(reifiedShapes[0]))
      loopRanges[ub.index()].size = ub.value();
    return loopRanges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    FailureOr<TilingResult> result =
        bubbleUpConcatSlice(b, cast<ConcatOp>(op), offsets, sizes);
    if (failed(result))
      return failure();
    return result.value();
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    return getTiledImplementation(op, b, offsets, sizes);
  }
};

//===----------------------------------------------------------------------===//
//
// ExpandShapeOp and CollapseShapeOp Tiling.
// Original implementation is from:
// https://github.com/bytedance/byteir/blob/3672e3b858a06b7603f7c48806702b108237ca4c/compiler/lib/Dialect/Tensor/IR/TilingInterfaceImpl.cpp
//
//===----------------------------------------------------------------------===//

OpFoldResult canonicalizeOpFoldResult(OpFoldResult ofr, bool enableFold) {
  auto val = dyn_cast<Value>(ofr);
  if (!val)
    return ofr;

  if (!enableFold)
    return getAsOpFoldResult(val);

  auto opResult = dyn_cast<OpResult>(val);
  if (!opResult)
    return getAsOpFoldResult(val);

  OpBuilder builder(opResult.getOwner());
  SmallVector<Value> foldResults;
  if (succeeded(builder.tryFold(opResult.getOwner(), foldResults)))
    val = foldResults[opResult.getResultNumber()];

  auto constInt = getConstantIntValue(val);
  if (constInt.has_value())
    return getAsIndexOpFoldResult(ofr.getContext(), constInt.value());

  return val;
}

SmallVector<OpFoldResult> canonicalizeOpFoldResult(ArrayRef<OpFoldResult> ofrs,
                                                   bool enableFold = true) {
  return llvm::to_vector(llvm::map_range(ofrs, [&](OpFoldResult ofr) {
    return canonicalizeOpFoldResult(ofr, enableFold);
  }));
}

struct TensorSliceParameters {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
};

static bool isNoTile(OpFoldResult tileSize, OpFoldResult offset,
                     ArrayRef<int64_t> shape, int64_t dim) {
  std::optional<int64_t> maybeIntTileSize = getConstantIntValue(tileSize);
  if (maybeIntTileSize.has_value()) {
    return maybeIntTileSize.value() == 0 ||
           maybeIntTileSize.value() == shape[dim];
  }
  std::optional<int64_t> maybeIntOffset = getConstantIntValue(offset);
  return maybeIntOffset.has_value();
}

static bool isUnitTile(OpFoldResult tileSize, int64_t dim) {
  std::optional<int64_t> maybeIntTileSize = getConstantIntValue(tileSize);
  if (maybeIntTileSize.has_value()) {
    return maybeIntTileSize.value() == 1;
  }
  return false;
}

static bool isValidTile(OpFoldResult tileSize, ArrayRef<int64_t> shape,
                        int64_t dim) {
  std::optional<int64_t> maybeIntTileSize = getConstantIntValue(tileSize);
  if (maybeIntTileSize.has_value()) {
    return shape[dim] % maybeIntTileSize.value() == 0;
  }
  return false;
}

static FailureOr<TensorSliceParameters> getExpandedSliceParameters(
    OpBuilder &b, Location loc, ArrayRef<ReassociationIndices> associations,
    const TensorSliceParameters &collapsedSliceParams,
    ArrayRef<int64_t> collapsedShape, Value expandedValue) {
  MLIRContext *ctx = expandedValue.getContext();
  ArrayRef<int64_t> expandedShape =
      cast<ShapedType>(expandedValue.getType()).getShape();
  TensorSliceParameters resSliceParameters;
  resSliceParameters.offsets.reserve(expandedShape.size());
  resSliceParameters.sizes.reserve(expandedShape.size());

  for (auto [collapsedIdx, expandedIndices] : llvm::enumerate(associations)) {
    OpFoldResult collapsedTileSize = collapsedSliceParams.sizes[collapsedIdx];
    OpFoldResult collapsedOffset = collapsedSliceParams.offsets[collapsedIdx];

    // Case 0a: if a dimension of the collapsed value isn't tiled, all the
    // correspond dimensions of the expanded value won't be tiled.
    if (isNoTile(collapsedTileSize, collapsedOffset, collapsedShape,
                 collapsedIdx)) {
      for (int64_t expandedIdx : expandedIndices) {
        resSliceParameters.offsets.push_back(b.getIndexAttr(0));
        resSliceParameters.sizes.push_back(
            utils::getDimOFR(b, loc, expandedValue, expandedIdx));
      }
      continue;
    }

    ArrayRef<int64_t> expandedIndicesRef = expandedIndices;
    // Case 0b: if the last dimension of the expanded value was the multiple of
    // the tileSize N of the collapsed dimension, the expanded value could be
    // tiled by [1, ...,1, N]
    if (isValidTile(collapsedTileSize, expandedShape,
                    expandedIndicesRef.back())) {
      std::optional<int64_t> maybeIntOffset =
          getConstantIntValue(collapsedOffset);
      AffineExpr offsetExpr;
      if (!maybeIntOffset.has_value()) {
        offsetExpr = getAffineDimExpr(0, ctx);
      } else {
        offsetExpr = getAffineConstantExpr(*maybeIntOffset, ctx);
      }
      SmallVector<AffineExpr> offsetExprs;
      for (auto &&dim : llvm::reverse(expandedIndicesRef)) {
        offsetExprs.push_back({offsetExpr % expandedShape[dim]});
        offsetExpr = offsetExpr.floorDiv(expandedShape[dim]);
      }

      for (auto &&expr : llvm::reverse(offsetExprs)) {
        if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
          resSliceParameters.offsets.push_back(
              b.getIndexAttr(constExpr.getValue()));
        } else {
          resSliceParameters.offsets.push_back(
              b.create<affine::AffineApplyOp>(
                   loc, AffineMap::inferFromExprList({expr}, ctx).front(),
                   dyn_cast<Value>(collapsedOffset))
                  ->getResult(0));
        }
      }
      resSliceParameters.sizes.append(expandedIndicesRef.size() - 1,
                                      b.getIndexAttr(1));
      resSliceParameters.sizes.push_back(collapsedTileSize);

      continue;
    }

    // handle the leading dimensions whose size is equal to 1
    expandedIndicesRef = expandedIndicesRef.drop_while([&](int64_t idx) {
      bool isOne = expandedShape[idx] == 1;
      if (!isOne)
        return false;
      // should also add correct tile size and offset to the result
      resSliceParameters.offsets.push_back(b.getIndexAttr(0));
      resSliceParameters.sizes.push_back(b.getIndexAttr(1));
      return true;
    });

    // Case 1: No more index left
    if (expandedIndicesRef.empty())
      continue;

    // Case 2: If only one index left, the tile size on the expanded side is
    // equal to that on the collapsed side
    if (expandedIndicesRef.size() == 1) {
      resSliceParameters.offsets.push_back(collapsedOffset);
      resSliceParameters.sizes.push_back(collapsedTileSize);
      continue;
    }

    expandedIndicesRef = expandedIndicesRef.drop_front(1);
    // Case 3: If all the remaining dimention sizes except the leading one is
    // equal to one, the situation is similar to above.
    if (llvm::all_of(expandedIndicesRef,
                     [&](int64_t dim) { return expandedShape[dim] == 1; })) {
      resSliceParameters.offsets.push_back(collapsedOffset);
      resSliceParameters.sizes.push_back(collapsedTileSize);
      resSliceParameters.offsets.append(expandedIndicesRef.size(),
                                        b.getIndexAttr(0));
      resSliceParameters.sizes.append(expandedIndicesRef.size(),
                                      b.getIndexAttr(1));
      continue;
    }

    // Case 4: If more than 1 indices are left, the tile size must be a multiple
    // of the product of the dimension size except the first one, which also
    // requires that the tile size and all the dimension size of the first one
    // must be static.
    if (!llvm::all_of(expandedIndicesRef, [&](int64_t dim) {
          return expandedShape[dim] != ShapedType::kDynamic;
        })) {
      LLVM_DEBUG(
          DBGS() << "Not all of the remaining dimension size is equal to 1.\n");
      return failure();
    }

    std::optional<int64_t> maybeIntTileSize =
        getConstantIntValue(collapsedTileSize);
    if (!maybeIntTileSize.has_value()) {
      LLVM_DEBUG(DBGS() << "the tile size must be static: " << collapsedTileSize
                        << ".\n");
      return failure();
    }
    int64_t collapsedIntTileSize = maybeIntTileSize.value();
    int64_t productOfDimSizes = 1;
    for (int64_t dim : expandedIndicesRef) {
      if (collapsedIntTileSize % expandedShape[dim] != 0) {
        LLVM_DEBUG(DBGS() << "the tile size is not a multiple of the product "
                             "of the dimension size except the first one.\n");
        return failure();
      }
      collapsedIntTileSize /= expandedShape[dim];
      productOfDimSizes *= expandedShape[dim];
    }
    Value collapsedOffsetVal = dyn_cast<Value>(collapsedOffset);
    if (!collapsedOffsetVal) {
      return failure();
    }

    // add the size and offset of the first the dimensions after dropping those
    // of dimension size one
    resSliceParameters.sizes.push_back(b.getIndexAttr(collapsedIntTileSize));
    AffineMap map =
        AffineMap::inferFromExprList(
            {mlir::getAffineDimExpr(0, ctx).floorDiv(productOfDimSizes)}, ctx)
            .front();
    resSliceParameters.offsets.push_back(
        b.create<affine::AffineApplyOp>(loc, map, collapsedOffsetVal)
            ->getResult(0));

    // add the size and offset of the remaining dimensions
    for (int64_t expandedIdx : expandedIndicesRef) {
      resSliceParameters.offsets.push_back(b.getIndexAttr(0));
      resSliceParameters.sizes.push_back(
          utils::getDimOFR(b, loc, expandedValue, expandedIdx));
    }
  }

  return resSliceParameters;
}

static FailureOr<TensorSliceParameters> getCollapsedSliceParameters(
    OpBuilder &b, Location loc, ArrayRef<ReassociationIndices> associations,
    const TensorSliceParameters &expandedSliceParams,
    ArrayRef<int64_t> expandedShape, Value collapsedValue) {
  MLIRContext *ctx = collapsedValue.getContext();
  ArrayRef<int64_t> collapsedShape =
      cast<ShapedType>(collapsedValue.getType()).getShape();
  TensorSliceParameters resSliceParameters;
  resSliceParameters.offsets.reserve(collapsedShape.size());
  resSliceParameters.sizes.reserve(collapsedShape.size());

  for (auto [collapsedIdx, expandedIndices] : llvm::enumerate(associations)) {
    // Case 0a: If all the dimensions of the expanded value aren't tiled, the
    // corresponding collapsed dimension of the collapsed value won't be tiled.
    if (llvm::all_of(expandedIndices, [&](int64_t dim) {
          OpFoldResult expandedTileSize = expandedSliceParams.sizes[dim];
          OpFoldResult expandedOffset = expandedSliceParams.offsets[dim];
          return isNoTile(expandedTileSize, expandedOffset, expandedShape, dim);
        })) {
      resSliceParameters.offsets.push_back(b.getIndexAttr(0));
      resSliceParameters.sizes.push_back(
          utils::getDimOFR(b, loc, collapsedValue, collapsedIdx));
      continue;
    }

    ArrayRef<int64_t> expandedIndicesRef = expandedIndices;
    // Case 0b: If expanded value are tiled by (1, ...,1, N), the corresponding
    // collapsed dimension of the collapsed value will be tiled by N
    if (llvm::all_of(expandedIndicesRef.drop_back(1), [&](int64_t dim) {
          OpFoldResult expandedTileSize = expandedSliceParams.sizes[dim];
          return isUnitTile(expandedTileSize, dim);
        })) {
      auto offsetExpr = getAffineConstantExpr(0, ctx);
      SmallVector<Value> offsetValues;
      int64_t ind = 0;
      for (auto &&dim : expandedIndicesRef) {
        offsetExpr =
            offsetExpr * getAffineConstantExpr(expandedShape[dim], ctx);
        std::optional<int64_t> maybeIntOffset =
            getConstantIntValue(expandedSliceParams.offsets[dim]);
        if (!maybeIntOffset.has_value()) {
          offsetExpr = offsetExpr + getAffineDimExpr(ind++, ctx);
          offsetValues.push_back(
              dyn_cast<Value>(expandedSliceParams.offsets[dim]));
        } else {
          offsetExpr = offsetExpr + getAffineConstantExpr(*maybeIntOffset, ctx);
        }
      }

      resSliceParameters.sizes.push_back(
          expandedSliceParams.sizes[expandedIndicesRef.back()]);
      resSliceParameters.offsets.push_back(
          b.create<affine::AffineApplyOp>(
               loc, AffineMap::inferFromExprList({offsetExpr}, ctx).front(),
               offsetValues)
              ->getResult(0));
      continue;
    }

    // handle the leading dimensions whose size is equal to 1
    expandedIndicesRef = expandedIndicesRef.drop_while([&](int64_t idx) {
      bool isOne = expandedShape[idx] == 1;
      return isOne;
    });

    // Case 1: No more index left
    if (expandedIndicesRef.empty()) {
      resSliceParameters.offsets.push_back(b.getIndexAttr(0));
      resSliceParameters.sizes.push_back(b.getIndexAttr(1));
      continue;
    }

    int64_t firstNotOneDim = expandedIndicesRef[0];
    // Case 2: If only one index left, the tile size on the expanded side is
    // equal to that on the collapsed side
    if (expandedIndicesRef.size() == 1) {
      resSliceParameters.offsets.push_back(
          expandedSliceParams.offsets[firstNotOneDim]);
      resSliceParameters.sizes.push_back(
          expandedSliceParams.sizes[firstNotOneDim]);
      continue;
    }

    expandedIndicesRef = expandedIndicesRef.drop_front(1);
    // Case 3: If all the remaining dimention sizes except the leading one is
    // equal to one, the situation is similar to above.
    if (llvm::all_of(expandedIndicesRef,
                     [&](int64_t dim) { return expandedShape[dim] == 1; })) {
      resSliceParameters.offsets.push_back(
          expandedSliceParams.offsets[firstNotOneDim]);
      resSliceParameters.sizes.push_back(
          expandedSliceParams.sizes[firstNotOneDim]);
      continue;
    }

    // Case 4: If more than 1 indices are left, the tile size must be a multiple
    // of the product of the dimension size except the first one, which also
    // requires that the tile size and all the dimension size of the first one
    // must be static.
    if (!llvm::all_of(expandedIndicesRef, [&](int64_t dim) {
          return expandedShape[dim] != ShapedType::kDynamic;
        })) {
      LLVM_DEBUG(
          DBGS() << "Not all of the remaining dimension size is equal to 1.\n");
      return failure();
    }

    int64_t productOfExpandedTileSize = 1;
    // If any of the remaining dimension is tiled, return failure
    for (int64_t dim : expandedIndicesRef) {
      OpFoldResult expandedTileSize = expandedSliceParams.sizes[dim];
      OpFoldResult expandedOffset = expandedSliceParams.offsets[dim];
      if (!isNoTile(expandedTileSize, expandedOffset, expandedShape, dim) ||
          expandedShape[dim] == ShapedType::kDynamic)
        return failure();
      productOfExpandedTileSize *= expandedShape[dim];
    }

    Value firstNotOneOffsetVal =
        dyn_cast<Value>(expandedSliceParams.offsets[firstNotOneDim]);
    if (!firstNotOneOffsetVal) {
      return failure();
    }
    OpFoldResult firstNotOneTileSize =
        expandedSliceParams.sizes[firstNotOneDim];
    std::optional<int64_t> maybeIntFirstNotOneTileSize =
        getConstantIntValue(firstNotOneTileSize);
    if (!maybeIntFirstNotOneTileSize.has_value()) {
      LLVM_DEBUG(
          DBGS() << "the tile size of the first not-one should be static.\n");
      return failure();
    }
    int64_t collaspedTileSize =
        (*maybeIntFirstNotOneTileSize) * productOfExpandedTileSize;

    // add the size and offset of the first the dimensions after dropping those
    // of dimension size one
    resSliceParameters.sizes.push_back(b.getIndexAttr(collaspedTileSize));
    AffineMap map =
        AffineMap::inferFromExprList(
            {mlir::getAffineDimExpr(0, ctx) * productOfExpandedTileSize}, ctx)
            .front();
    resSliceParameters.offsets.push_back(
        b.create<affine::AffineApplyOp>(loc, map, firstNotOneOffsetVal)
            ->getResult(0));
  }

  return resSliceParameters;
}

static FailureOr<TilingResult> commonGenerateResultTileValue(
    Operation *op, OpBuilder &b, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  auto tilingInterfaceOp = cast<TilingInterface>(op);
  FailureOr<TilingResult> tilingResult =
      tilingInterfaceOp.getTiledImplementation(b, offsets, sizes);
  if (failed(tilingResult))
    return failure();
  return tilingResult.value();
}

//----------------------------------------------------------------------------//
// ExpandShapeOpTiling
//----------------------------------------------------------------------------//

struct ExpandShapeOpTiling
    : public TilingInterface::ExternalModel<ExpandShapeOpTiling,
                                            tensor::ExpandShapeOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto expandShapeOp = cast<tensor::ExpandShapeOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        expandShapeOp.getResultType().getRank(), utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    MLIRContext *ctx = op->getContext();
    auto expandShapeOp = cast<tensor::ExpandShapeOp>(op);
    int64_t outRank = expandShapeOp.getResultType().getRank();
    Location loc = op->getLoc();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(op);
    IntegerAttr zero = b.getIndexAttr(0);
    IntegerAttr one = b.getIndexAttr(1);
    ArrayRef<int64_t> resShape = expandShapeOp.getResultType().getShape();
    SmallVector<Range> loopRanges(outRank, {zero, one, one});
    SmallVector<ReassociationIndices, 4> associations =
        expandShapeOp.getReassociationIndices();
    for (auto [collapsedIdx, expandedIndices] : llvm::enumerate(associations)) {
      int64_t product = 1;
      int64_t dynamicDim = kNotInited;
      for (int64_t dim : expandedIndices) {
        if (resShape[dim] != ShapedType::kDynamic) {
          loopRanges[dim].size = b.getIndexAttr(resShape[dim]);
          product *= resShape[dim];
        } else {
          assert(dynamicDim == kNotInited && "at most one dynamic dimension");
          dynamicDim = dim;
        }
      }
      if (dynamicDim != kNotInited) {
        Value dynDimSize =
            utils::getDimValue(b, loc, expandShapeOp.getSrc(), collapsedIdx);
        if (product == 1)
          loopRanges[dynamicDim].size = dynDimSize;
        else {
          AffineMap map =
              AffineMap::inferFromExprList(
                  {mlir::getAffineDimExpr(0, ctx).floorDiv(product)}, ctx)
                  .front();
          loopRanges[dynamicDim].size =
              b.create<affine::AffineApplyOp>(loc, map, dynDimSize)
                  ->getResult(0);
        }
      }
    }
    return loopRanges;
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    SmallVector<OpFoldResult> canonOffsets = canonicalizeOpFoldResult(offsets);
    resultOffsets.assign(canonOffsets.begin(), canonOffsets.end());

    SmallVector<OpFoldResult> canonSizes = canonicalizeOpFoldResult(sizes);
    resultSizes.assign(canonSizes.begin(), canonSizes.end());
    return success();
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto expandShapeOp = cast<tensor::ExpandShapeOp>(op);
    SmallVector<OpFoldResult> canonOffsets = canonicalizeOpFoldResult(offsets);
    SmallVector<OpFoldResult> canonSizes = canonicalizeOpFoldResult(sizes);
    Location loc = op->getLoc();
#ifndef NDEBUG
    int64_t outRank = expandShapeOp.getResultType().getRank();
#endif
    int64_t srcRank = expandShapeOp.getSrcType().getRank();
    SmallVector<ReassociationIndices, 4> associations =
        expandShapeOp.getReassociationIndices();
    assert(offsets.size() == static_cast<size_t>(outRank) &&
           sizes.size() == static_cast<size_t>(outRank));

    // create tiled source
    SmallVector<OpFoldResult> srcStrides(srcRank, b.getIndexAttr(1));
    TensorSliceParameters expandedSliceParams;
    expandedSliceParams.offsets = canonOffsets;
    expandedSliceParams.sizes = canonSizes;
    FailureOr<TensorSliceParameters> collapsedSliceParam =
        getCollapsedSliceParameters(b, loc, associations, expandedSliceParams,
                                    expandShapeOp.getResultType().getShape(),
                                    expandShapeOp.getSrc());
    if (failed(collapsedSliceParam)) {
      LLVM_DEBUG(DBGS() << "Check tile size failed.\n");
      return {};
    }
    Value tiledSrc = utils::getSlice(b, loc, expandShapeOp.getSrc(),
                                     (*collapsedSliceParam).offsets,
                                     (*collapsedSliceParam).sizes, srcStrides);

    // create result type
    SmallVector<int64_t> resShape =
        llvm::to_vector(llvm::map_range(sizes, [](OpFoldResult ofr) {
          std::optional<int64_t> maybeIntSize = getConstantIntValue(ofr);
          if (!maybeIntSize.has_value())
            return ShapedType::kDynamic;
          return maybeIntSize.value();
        }));
    auto resType = expandShapeOp.getResultType().clone(resShape);

    Operation *tiledExpandShapeOp = b.create<tensor::ExpandShapeOp>(
        loc, resType, tiledSrc, expandShapeOp.getReassociation(),
        ValueRange(expandShapeOp.getOutputShape()),
        expandShapeOp.getStaticOutputShapeAttr());

    return TilingResult{{tiledExpandShapeOp},
                        SmallVector<Value>(tiledExpandShapeOp->getResults())};
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    return commonGenerateResultTileValue(op, b, resultNumber, offsets, sizes);
  }
};

//----------------------------------------------------------------------------//
// CollapseShapeOpTiling
//----------------------------------------------------------------------------//

struct CollapseShapeOpTiling
    : public TilingInterface::ExternalModel<CollapseShapeOpTiling,
                                            tensor::CollapseShapeOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto collapseShapeOp = cast<tensor::CollapseShapeOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        collapseShapeOp.getResultType().getRank(),
        utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto collapseShapeOp = cast<tensor::CollapseShapeOp>(op);
    int64_t resRank = collapseShapeOp.getResultType().getRank();
    Location loc = op->getLoc();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(op);
    IntegerAttr zero = b.getIndexAttr(0);
    IntegerAttr one = b.getIndexAttr(1);
    SmallVector<Range> loopRanges(resRank, {zero, one, one});
    ArrayRef<int64_t> resShape = collapseShapeOp.getResultType().getShape();
    ArrayRef<int64_t> srcShape = collapseShapeOp.getSrcType().getShape();
    MLIRContext *ctx = op->getContext();
    for (auto dim : llvm::seq<int64_t>(0, resRank)) {
      if (resShape[dim] != ShapedType::kDynamic)
        loopRanges[dim].size = b.getIndexAttr(resShape[dim]);
      else {
        // When it is dynamic, we should get the dimension info from the input
        ReassociationIndices singleAssociation =
            collapseShapeOp.getReassociationIndices()[dim];
        int64_t product = 1;
        int64_t dynamicDim = kNotInited;
        for (int64_t idx : singleAssociation) {
          if (srcShape[idx] == ShapedType::kDynamic) {
            assert(dynamicDim == kNotInited && "at most one dynamic dimension");
            dynamicDim = idx;
          } else {
            product *= srcShape[idx];
          }
        }
        Value dynDimSize =
            utils::getDimValue(b, loc, collapseShapeOp.getSrc(), dynamicDim);
        if (product == 1)
          loopRanges[dim].size = dynDimSize;
        else {
          AffineMap map = AffineMap::inferFromExprList(
                              {mlir::getAffineDimExpr(0, ctx) * product}, ctx)
                              .front();
          loopRanges[dim].size =
              b.create<affine::AffineApplyOp>(loc, map, dynDimSize)
                  ->getResult(0);
        }
      }
    }

    return loopRanges;
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    SmallVector<OpFoldResult> canonOffsets = canonicalizeOpFoldResult(offsets);
    resultOffsets.assign(canonOffsets.begin(), canonOffsets.end());

    SmallVector<OpFoldResult> canonSizes = canonicalizeOpFoldResult(sizes);
    resultSizes.assign(canonSizes.begin(), canonSizes.end());
    return success();
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto collapseShapeOp = cast<tensor::CollapseShapeOp>(op);
    Location loc = op->getLoc();
#ifndef NDEBUG
    int64_t outRank = collapseShapeOp.getResultType().getRank();
#endif
    int64_t srcRank = collapseShapeOp.getSrcType().getRank();
    assert(offsets.size() == static_cast<size_t>(outRank) &&
           sizes.size() == static_cast<size_t>(outRank));
    SmallVector<ReassociationIndices, 4> associations =
        collapseShapeOp.getReassociationIndices();
    SmallVector<OpFoldResult> canonOffsets = canonicalizeOpFoldResult(offsets);
    SmallVector<OpFoldResult> canonSizes = canonicalizeOpFoldResult(sizes);

    // create tiled source
    TensorSliceParameters collapsedSliceParams;
    collapsedSliceParams.offsets = canonOffsets;
    collapsedSliceParams.sizes = canonSizes;
    FailureOr<TensorSliceParameters> expandedSliceParam =
        getExpandedSliceParameters(b, loc, associations, collapsedSliceParams,
                                   collapseShapeOp.getResultType().getShape(),
                                   collapseShapeOp.getSrc());
    if (failed(expandedSliceParam)) {
      LLVM_DEBUG(DBGS() << "Check tile size failed.\n");
      return {};
    }
    SmallVector<OpFoldResult> srcStrides(srcRank, b.getIndexAttr(1));
    Value tiledSrc = utils::getSlice(b, loc, collapseShapeOp.getSrc(),
                                     (*expandedSliceParam).offsets,
                                     (*expandedSliceParam).sizes, srcStrides);

    // create result type
    SmallVector<int64_t> resShape =
        llvm::to_vector(llvm::map_range(canonSizes, [](OpFoldResult ofr) {
          std::optional<int64_t> maybeIntSize = getConstantIntValue(ofr);
          if (!maybeIntSize.has_value())
            return ShapedType::kDynamic;
          return maybeIntSize.value();
        }));
    auto resType = collapseShapeOp.getResultType().clone(resShape);

    Operation *tiledCollapseShapeOp = b.create<tensor::CollapseShapeOp>(
        loc, resType, tiledSrc, op->getAttrs());

    return TilingResult{{tiledCollapseShapeOp},
                        SmallVector<Value>(tiledCollapseShapeOp->getResults())};
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    return commonGenerateResultTileValue(op, b, resultNumber, offsets, sizes);
  }
};

} // namespace

void bishengir::tensor::registerTilingInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TensorDialect *dialect) {
    mlir::tensor::ConcatOp::attachInterface<ConcatOpTiling>(*ctx);
    mlir::tensor::ExpandShapeOp::attachInterface<ExpandShapeOpTiling>(*ctx);
    mlir::tensor::CollapseShapeOp::attachInterface<CollapseShapeOpTiling>(*ctx);
  });
}
