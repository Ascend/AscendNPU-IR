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

LogicalResult checkInsertSliceTileAlignment(tensor::InsertSliceOp insertSliceOp,
                                            DataLayoutAttr fractalLayout,
                                            PatternRewriter &rewriter,
                                            ConvertLayoutOp convertOp) {
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
  for (int64_t dim = spatialStart; dim < rank; ++dim) {
    int64_t blockSize =
        dim == spatialStart ? blockSizes->first : blockSizes->second;
    if (!isKnownMultipleOf(offsets[dim], blockSize) ||
        !isKnownMultipleOf(sizes[dim], blockSize))
      return rewriter.notifyMatchFailure(
          convertOp,
          "insert_slice offsets and sizes must be tile-aligned");
  }
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

  // Tile alignment guarantees intra-tile offsets are zero. Force them so
  // dynamic-but-aligned ND offsets do not leave a residual `mod` apply.
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

//===----------------------------------------------------------------------===//
// Propagate UP through InsertSlice Operations
//===----------------------------------------------------------------------===//

/// Pattern: Push convert_layout UP through tensor.insert_slice operations
/// Before:
///   %inserted = tensor.insert_slice %source into %dest[off][sz][1,1]
///   %fractal = hivm.hir.convert_layout %inserted {up}  // ND -> Fractal
/// After:
///   %dest_fractal = hivm.hir.convert_layout %dest {up}
///   %source_fractal = hivm.hir.convert_layout %source {up}
///   %inserted_fractal = tensor.insert_slice %source_fractal into %dest_fractal
///       [off'][sz'][1,1,1,1]
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

    if (failed(checkInsertSliceTileAlignment(insertSliceOp, dstLayout, rewriter,
                                             convertOp)))
      return rewriter.notifyMatchFailure(
          convertOp, "insert_slice offsets or sizes are not tile-aligned");

    rewriter.setInsertionPoint(insertSliceOp);

    FailureOr<Value> destConverted = createConvertLayoutForOperand(
        rewriter, loc, srcLayout, dstLayout, insertSliceOp.getDest());
    if (failed(destConverted))
      return rewriter.notifyMatchFailure(convertOp,
                                         "failed to convert dest operand");

    FailureOr<Value> sourceConverted = createConvertLayoutForOperand(
        rewriter, loc, srcLayout, dstLayout, insertSliceOp.getSource());
    if (failed(sourceConverted))
      return rewriter.notifyMatchFailure(convertOp,
                                         "failed to convert source operand");

    FailureOr<Value> newInsertSlice = createFractalInsertSlice(
        rewriter, loc, insertSliceOp, *sourceConverted, *destConverted,
        srcLayout, dstLayout);
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

/// Pattern: Push convert_layout DOWN through tensor.insert_slice users
/// Before:
///   %dest_nd = hivm.hir.convert_layout %dest_fr {down}  // Fractal -> ND
///   %inserted = tensor.insert_slice %source into %dest_nd[off][sz][1,1]
/// After:
///   %source_fr = hivm.hir.convert_layout %source {up}
///   %inserted_fr = tensor.insert_slice %source_fr into %dest_fr[off'][sz']
///   %inserted = hivm.hir.convert_layout %inserted_fr {down}
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
    if (failed(checkInsertSliceTileAlignment(insertSliceOp, fractalLayout,
                                             rewriter, convertOp)))
      return rewriter.notifyMatchFailure(
          convertOp, "insert_slice offsets or sizes are not tile-aligned");

    Location loc = insertSliceOp.getLoc();
    rewriter.setInsertionPoint(insertSliceOp);

    FailureOr<Value> sourceConverted = createConvertLayoutForOperand(
        rewriter, loc, ndLayout, fractalLayout, insertSliceOp.getSource());
    if (failed(sourceConverted))
      return rewriter.notifyMatchFailure(convertOp,
                                         "failed to convert source operand");

    FailureOr<Value> newInsertSlice = createFractalInsertSlice(
        rewriter, loc, insertSliceOp, *sourceConverted, convertOp.getSource(),
        ndLayout, fractalLayout);
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
