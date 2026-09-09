//===-------------------- PropagateConvertLayoutInsertSlice.cpp -----------===//
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
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#define DEBUG_TYPE "hivm-propagate-convert-layout"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace {

/// Return true when `value` is statically known to be a multiple of `divisor`.
/// Walks through index casts and simple arith/affine producers so dynamic
/// offsets such as `arith.muli %iv, %c32` can be proven tile-aligned.
bool isKnownMultipleOf(OpFoldResult value, int64_t divisor) {
  if (divisor == 1)
    return true;
  if (divisor <= 0)
    return false;
  if (std::optional<int64_t> cst = getConstantIntValue(value))
    return *cst % divisor == 0;

  auto v = dyn_cast<Value>(value);
  if (!v)
    return false;

  Operation *def = v.getDefiningOp();
  if (!def)
    return false;

  if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
          arith::ExtUIOp>(def))
    return isKnownMultipleOf(def->getOperand(0), divisor);

  if (auto mul = dyn_cast<arith::MulIOp>(def))
    return isKnownMultipleOf(mul.getLhs(), divisor) ||
           isKnownMultipleOf(mul.getRhs(), divisor);

  if (isa<arith::AddIOp, arith::SubIOp, arith::MaxSIOp, arith::MinSIOp,
          arith::MaxUIOp, arith::MinUIOp>(def))
    return isKnownMultipleOf(def->getOperand(0), divisor) &&
           isKnownMultipleOf(def->getOperand(1), divisor);

  if (auto apply = dyn_cast<affine::AffineApplyOp>(def)) {
    if (apply.getAffineMap().getNumResults() != 1)
      return false;
    AffineExpr expr = apply.getAffineMap().getResult(0);
    if (expr.isMultipleOf(divisor))
      return true;
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
      return isKnownMultipleOf(apply.getOperand(dimExpr.getPosition()),
                               divisor);
    if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr))
      return isKnownMultipleOf(
          apply.getOperand(apply.getAffineMap().getNumDims() +
                           symExpr.getPosition()),
          divisor);
  }
  return false;
}

LogicalResult checkInsertSliceHasUnitStrides(tensor::InsertSliceOp insertSliceOp,
                                             PatternRewriter &rewriter,
                                             ConvertLayoutOp convertOp) {
  for (OpFoldResult stride : insertSliceOp.getMixedStrides()) {
    std::optional<int64_t> strideVal = getConstantIntValue(stride);
    if (!strideVal || *strideVal != 1)
      return rewriter.notifyMatchFailure(
          convertOp, "insert_slice has non-unit or dynamic strides");
  }
  return success();
}

/// Per spatial dim. `block` is that dim's fractal size (M: f0, N: f1).
/// FullyAligned: offset and size are both multiples of `block` (a0/b0 = 0).
/// IntraTile: size divides `block` and offset is a multiple of size, so
/// [off, off+size) stays in one tile. sap: size 8, block 16, offset iv*8
/// → a0 is 0 or 8.
enum class TileDimFit { FullyAligned, IntraTile };

struct InsertSliceTileFit {
  TileDimFit mFit = TileDimFit::FullyAligned;
  TileDimFit nFit = TileDimFit::FullyAligned;
  bool anyIntraTile() const {
    return mFit == TileDimFit::IntraTile || nFit == TileDimFit::IntraTile;
  }
};

bool classifyTileDim(OpFoldResult offset, OpFoldResult size, int64_t block,
                     TileDimFit &fit) {
  if (isKnownMultipleOf(offset, block) && isKnownMultipleOf(size, block)) {
    fit = TileDimFit::FullyAligned;
    return true;
  }
  std::optional<int64_t> sizeCst = getConstantIntValue(size);
  if (!sizeCst || *sizeCst <= 0 || *sizeCst > block || (block % *sizeCst) != 0)
    return false;
  if (!isKnownMultipleOf(offset, *sizeCst))
    return false;
  fit = TileDimFit::IntraTile;
  return true;
}

LogicalResult checkInsertSliceTileFit(tensor::InsertSliceOp insertSliceOp,
                                      DataLayoutAttr fractalLayout,
                                      PatternRewriter &rewriter,
                                      ConvertLayoutOp convertOp,
                                      InsertSliceTileFit &fit) {
  auto sourceType =
      dyn_cast<RankedTensorType>(insertSliceOp.getSource().getType());
  auto destType = dyn_cast<RankedTensorType>(insertSliceOp.getDest().getType());
  if (!sourceType || !destType || sourceType.getRank() != destType.getRank())
    return rewriter.notifyMatchFailure(
        convertOp, "rank-reduced insert_slice is not supported");

  int64_t rank = destType.getRank();
  if (rank != 2 && rank != 3)
    return rewriter.notifyMatchFailure(
        convertOp, "insert_slice must have rank two or three");

  FailureOr<FractalSize> blockSizes = fractalLayout.getFractalBlockSizes();
  if (failed(blockSizes))
    return rewriter.notifyMatchFailure(convertOp,
                                       "failed to get fractal block sizes");

  int64_t spatialStart = rank == 3 ? 1 : 0;
  SmallVector<OpFoldResult> offsets = insertSliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = insertSliceOp.getMixedSizes();
  int64_t blocks[2] = {blockSizes->first, blockSizes->second};
  TileDimFit dimFits[2];
  for (int i = 0; i < 2; ++i) {
    int64_t dim = spatialStart + i;
    if (!classifyTileDim(offsets[dim], sizes[dim], blocks[i], dimFits[i]))
      return rewriter.notifyMatchFailure(
          convertOp, "insert_slice offsets and sizes must be tile-aligned or "
                     "contained in one fractal tile");
    // Dest already one tile (M=16). Stop. Else this rewrite loops.
    if (dimFits[i] == TileDimFit::IntraTile) {
      int64_t destSize = destType.getDimSize(dim);
      if (destSize != ShapedType::kDynamic && destSize <= blocks[i])
        return rewriter.notifyMatchFailure(
            convertOp,
            "intra-tile insert dest is already a single fractal tile");
    }
  }
  fit.mFit = dimFits[0];
  fit.nFit = dimFits[1];
  return success();
}

FailureOr<Value> createConvertLayoutForOperand(PatternRewriter &rewriter,
                                               Location loc,
                                               DataLayoutAttr srcLayout,
                                               DataLayoutAttr dstLayout,
                                               Value operand) {
  auto operandType = cast<RankedTensorType>(operand.getType());
  SmallVector<OpFoldResult> operandShape = llvm::map_to_vector(
      operandType.getShape(), [&](int64_t dim) -> OpFoldResult {
        return getAsIndexOpFoldResult(rewriter.getContext(), dim);
      });

  auto mixedShape = computeMixedTargetLayoutShape(operandShape, srcLayout,
                                                  dstLayout, rewriter, loc);
  if (failed(mixedShape))
    return failure();

  auto convertedType = RankedTensorType::get(
      decomposeMixedValues(*mixedShape).first, operandType.getElementType());
  return rewriter
      .create<ConvertLayoutOp>(loc, convertedType, operand, srcLayout,
                               dstLayout, *mixedShape)
      .getResult();
}

FailureOr<Value> createFractalInsertSlice(PatternRewriter &rewriter,
                                          Location loc,
                                          tensor::InsertSliceOp insertSliceOp,
                                          Value fractalSource,
                                          Value fractalDest,
                                          DataLayoutAttr ndLayout,
                                          DataLayoutAttr fractalLayout) {
  auto newOffsets = computeTargetLayoutOffset(insertSliceOp.getMixedOffsets(),
                                              ndLayout, fractalLayout, rewriter,
                                              loc);
  if (failed(newOffsets))
    return failure();

  // Force a0=b0=0. Safe only when offset is a multiple of the tile.
  (*newOffsets)[newOffsets->size() - 2] = rewriter.getIndexAttr(0);
  (*newOffsets)[newOffsets->size() - 1] = rewriter.getIndexAttr(0);

  auto newSizes = computeMixedTargetLayoutShape(
      insertSliceOp.getMixedSizes(), ndLayout, fractalLayout, rewriter, loc);
  if (failed(newSizes))
    return failure();

  int64_t fractalRank = cast<RankedTensorType>(fractalDest.getType()).getRank();
  SmallVector<OpFoldResult> newStrides(fractalRank, rewriter.getIndexAttr(1));
  return rewriter
      .create<tensor::InsertSliceOp>(loc, fractalSource, fractalDest,
                                     *newOffsets, *newSizes, newStrides)
      .getResult();
}

OpFoldResult tileCountOFR(OpBuilder &builder, Location loc, OpFoldResult size,
                          int64_t block, TileDimFit fit) {
  // floorDiv(8, 16) is 0. Intra-tile still occupies one outer tile.
  if (fit == TileDimFit::IntraTile)
    return builder.getIndexAttr(1);
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineMap map =
      AffineMap::get(1, 0, d0.floorDiv(block), builder.getContext());
  return affine::makeComposedFoldedAffineApply(builder, loc, map, {size});
}

/// Convert src to the live fractal slice [N1,1,a0,N0] and insert at
/// [0, M1, a0, 0]. Do not convert dest. Output shape is the live inner so
/// combine can nd2nz into dest's extract (no padded temp, no L1→L1 copy).
FailureOr<Value>
createIntraTileFractalInsertSlice(PatternRewriter &rewriter, Location loc,
                                  tensor::InsertSliceOp insertSliceOp,
                                  Value fractalDest, DataLayoutAttr ndLayout,
                                  DataLayoutAttr fractalLayout,
                                  const InsertSliceTileFit &fit) {
  FailureOr<FractalSize> blockSizes = fractalLayout.getFractalBlockSizes();
  if (failed(blockSizes))
    return failure();
  int64_t f0 = blockSizes->first;
  int64_t f1 = blockSizes->second;

  // Offsets before convert so they dominate it; combine nd2nz uses them.
  auto newOffsets = computeTargetLayoutOffset(
      insertSliceOp.getMixedOffsets(), ndLayout, fractalLayout, rewriter, loc);
  if (failed(newOffsets))
    return failure();

  int64_t ndRank =
      cast<RankedTensorType>(insertSliceOp.getDest().getType()).getRank();
  int64_t spatialStart = ndRank == 3 ? 1 : 0;
  int64_t fractalRank = cast<RankedTensorType>(fractalDest.getType()).getRank();
  int64_t fractalSpatialStart = fractalRank == 5 ? 1 : 0;

  SmallVector<OpFoldResult> ndSizes = insertSliceOp.getMixedSizes();
  OpFoldResult m1Count =
      tileCountOFR(rewriter, loc, ndSizes[spatialStart], f0, fit.mFit);
  OpFoldResult n1Count =
      tileCountOFR(rewriter, loc, ndSizes[spatialStart + 1], f1, fit.nFit);
  OpFoldResult m0Size = fit.mFit == TileDimFit::IntraTile
                            ? ndSizes[spatialStart]
                            : rewriter.getIndexAttr(f0);
  OpFoldResult n0Size = fit.nFit == TileDimFit::IntraTile
                            ? ndSizes[spatialStart + 1]
                            : rewriter.getIndexAttr(f1);

  SmallVector<OpFoldResult> insertSizes;
  if (fractalSpatialStart)
    insertSizes.push_back(ndSizes[0]);
  if (fractalLayout.isScaleFractalLayout())
    insertSizes.append({m1Count, n1Count, m0Size, n0Size});
  else
    insertSizes.append({n1Count, m1Count, m0Size, n0Size});

  auto sourceType = cast<RankedTensorType>(insertSliceOp.getSource().getType());
  auto convertedType = RankedTensorType::get(
      decomposeMixedValues(insertSizes).first, sourceType.getElementType());
  Value sourceFr = rewriter
                       .create<ConvertLayoutOp>(
                           loc, convertedType, insertSliceOp.getSource(),
                           ndLayout, fractalLayout, insertSizes)
                       .getResult();

  SmallVector<OpFoldResult> unitStrides(fractalRank, rewriter.getIndexAttr(1));
  return rewriter
      .create<tensor::InsertSliceOp>(loc, sourceFr, fractalDest, *newOffsets,
                                     insertSizes, unitStrides)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Propagate UP through InsertSlice Operations
//===----------------------------------------------------------------------===//

/// Push convert_layout up through insert_slice.
/// Whole tiles: convert src+dest, insert in fractal.
/// Half tile: convert src to the live fractal slice, insert at a0. No dest
/// convert.
struct PropagateConvertLayoutUpThroughInsertSlice
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (!isPropagatingUp(convertOp))
      return failure();

    auto insertSliceOp =
        convertOp.getSource().getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSliceOp)
      return failure();

    if (failed(checkInsertSliceHasUnitStrides(insertSliceOp, rewriter,
                                              convertOp)))
      return failure();

    Location loc = insertSliceOp.getLoc();
    auto srcLayout = convertOp.getSrcLayoutAttr();
    auto dstLayout = convertOp.getDstLayoutAttr();

    InsertSliceTileFit fit;
    if (failed(checkInsertSliceTileFit(insertSliceOp, dstLayout, rewriter,
                                       convertOp, fit)))
      return rewriter.notifyMatchFailure(
          convertOp, "insert_slice offsets or sizes are not tile-aligned");

    rewriter.setInsertionPoint(insertSliceOp);

    FailureOr<Value> destConverted = createConvertLayoutForOperand(
        rewriter, loc, srcLayout, dstLayout, insertSliceOp.getDest());
    if (failed(destConverted))
      return rewriter.notifyMatchFailure(convertOp,
                                         "failed to convert dest operand");

    FailureOr<Value> newInsertSlice;
    if (fit.anyIntraTile()) {
      newInsertSlice = createIntraTileFractalInsertSlice(
          rewriter, loc, insertSliceOp, *destConverted, srcLayout, dstLayout,
          fit);
    } else {
      FailureOr<Value> sourceConverted = createConvertLayoutForOperand(
          rewriter, loc, srcLayout, dstLayout, insertSliceOp.getSource());
      if (failed(sourceConverted))
        return rewriter.notifyMatchFailure(convertOp,
                                           "failed to convert source operand");
      newInsertSlice = createFractalInsertSlice(
          rewriter, loc, insertSliceOp, *sourceConverted, *destConverted,
          srcLayout, dstLayout);
    }
    if (failed(newInsertSlice))
      return rewriter.notifyMatchFailure(
          convertOp, "failed to create fractal insert_slice");

    rewriter.replaceOp(convertOp, *newInsertSlice);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Propagate DOWN through InsertSlice Operations
//===----------------------------------------------------------------------===//

/// Push convert_layout down through insert_slice dest.
/// Whole tiles: convert src, insert in fractal, convert result down.
/// Half tile: convert src to the live fractal slice, insert at a0. No dest
/// convert.
struct PropagateConvertLayoutDownThroughInsertSlice
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (!isPropagatingDown(convertOp))
      return failure();

    if (convertOp->use_empty())
      return rewriter.notifyMatchFailure(convertOp,
                                         "convert_layout has no uses");

    auto findIt = llvm::find_if(convertOp->getUsers(), [](Operation *user) {
      return isa<tensor::InsertSliceOp>(user);
    });
    if (findIt == convertOp->getUsers().end())
      return rewriter.notifyMatchFailure(convertOp,
                                         "no tensor.insert_slice user found");

    auto insertSliceOp = cast<tensor::InsertSliceOp>(*findIt);
    if (insertSliceOp.getDest() != convertOp.getResult())
      return rewriter.notifyMatchFailure(
          convertOp, "convert_layout is not the insert_slice dest");

    if (failed(checkInsertSliceHasUnitStrides(insertSliceOp, rewriter,
                                              convertOp)))
      return failure();

    auto ndLayout = convertOp.getDstLayoutAttr();
    auto fractalLayout = convertOp.getSrcLayoutAttr();
    InsertSliceTileFit fit;
    if (failed(checkInsertSliceTileFit(insertSliceOp, fractalLayout, rewriter,
                                       convertOp, fit)))
      return rewriter.notifyMatchFailure(
          convertOp, "insert_slice offsets or sizes are not tile-aligned");

    Location loc = insertSliceOp.getLoc();
    rewriter.setInsertionPoint(insertSliceOp);

    FailureOr<Value> newInsertSlice;
    if (fit.anyIntraTile()) {
      newInsertSlice = createIntraTileFractalInsertSlice(
          rewriter, loc, insertSliceOp, convertOp.getSource(), ndLayout,
          fractalLayout, fit);
    } else {
      FailureOr<Value> sourceConverted = createConvertLayoutForOperand(
          rewriter, loc, ndLayout, fractalLayout, insertSliceOp.getSource());
      if (failed(sourceConverted))
        return rewriter.notifyMatchFailure(convertOp,
                                           "failed to convert source operand");
      newInsertSlice = createFractalInsertSlice(
          rewriter, loc, insertSliceOp, *sourceConverted, convertOp.getSource(),
          ndLayout, fractalLayout);
    }
    if (failed(newInsertSlice))
      return rewriter.notifyMatchFailure(
          convertOp, "failed to create fractal insert_slice");

    Value resultConvert =
        createConvertLayoutLike(rewriter, convertOp, *newInsertSlice);
    rewriter.replaceOp(insertSliceOp, resultConvert);
    return success();
  }
};

} // namespace

void mlir::hivm::populateConvertLayoutInsertSlice(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<PropagateConvertLayoutUpThroughInsertSlice,
               PropagateConvertLayoutDownThroughInsertSlice>(context);
}
