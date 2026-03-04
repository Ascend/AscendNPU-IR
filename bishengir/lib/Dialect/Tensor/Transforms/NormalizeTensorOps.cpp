//===--- NormalizeTensorOps.cpp -  optimize tensor ops --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to canonicalize pad operation sourced from
// Insert_slice operation
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/SwitchLoweringUtils.h"
#include <cstdint>

namespace mlir {
#define GEN_PASS_DEF_NORMALIZETENSOROPS
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "normalize-tensor-ops"

using namespace mlir;
using namespace mlir::tensor;

namespace {
struct NormalizeTensorOps
    : public impl::NormalizeTensorOpsBase<NormalizeTensorOps> {
  explicit NormalizeTensorOps(bool skipAlignedSlice)
      : skipAlignedSlice(skipAlignedSlice) {}
  void runOnOperation() override;

private:
  // option for `FoldInsertSliceToConcat` to decide whether we can skip folding
  // aligned insert_slice to concat
  bool skipAlignedSlice{false};
};
} // namespace

bool isStatic(llvm::ArrayRef<int64_t> arr, unsigned int padrank) {
  return arr.size() == padrank && llvm::none_of(arr, [](int64_t val) {
           return ShapedType::isDynamic(val);
         });
}

LogicalResult canFoldInsertPad(tensor::PadOp padOp, PatternRewriter &rewriter) {
  auto insertOp = padOp.getSource().getDefiningOp<InsertSliceOp>();
  if (!insertOp)
    return rewriter.notifyMatchFailure(padOp, "not defined by insert");
  // only apply if stride is 1
  auto insertStride = insertOp.getStaticStrides();
  if (!llvm::all_of(insertStride,
                    [&](int64_t sValue) { return sValue == 1; })) {
    return rewriter.notifyMatchFailure(insertOp, "stride not ones");
  }
  // Check if insert destination is uniform tensor (All elements same)
  auto insertDest = insertOp.getDest();
  Operation *defOp = insertDest.getDefiningOp();
  if (!defOp) {
    return rewriter.notifyMatchFailure(insertOp,
                                       "insert destination not defined");
  }
  auto constantOp = dyn_cast_if_present<arith::ConstantOp>(defOp);
  auto fillOp = dyn_cast_if_present<linalg::FillOp>(defOp);
  Value padValue = padOp.getConstantPaddingValue();
  if (!padValue)
    return rewriter.notifyMatchFailure(padOp, "pad value not constant");
  auto padConstantOp = padValue.getDefiningOp<arith::ConstantOp>();
  if (!padConstantOp)
    return rewriter.notifyMatchFailure(padOp, "pad value not defined constant");
  Attribute padAttr = padConstantOp.getValue();
  if (constantOp) {
    auto constValue = constantOp.getValue();
    auto denseAttr = dyn_cast_if_present<DenseIntOrFPElementsAttr>(constValue);
    if (!denseAttr || !denseAttr.isSplat()) {
      return rewriter.notifyMatchFailure(insertOp,
                                         "insert destination not splat");
    }
    auto splatAttr = denseAttr.getSplatValue<TypedAttr>();
    if (padAttr != splatAttr) {
      return rewriter.notifyMatchFailure(padOp, "pad value doesnt match splat");
    }
  } else if (fillOp) {
    auto fillValue = fillOp.value();
    if (padValue != fillValue) {
      return rewriter.notifyMatchFailure(padOp, "pad value doesnt match fill");
    }
  } else {
    return rewriter.notifyMatchFailure(
        insertOp, "insert destination not constant or fill operation");
  }
  return success();
}

/// This pattern identifies pad operations where the source is a insert_slice
/// operation where the destination of the insert is a uniform tensor
/// in this case insert can be treated as general pad and fold with the old
/// padding into a single padding, if the pad value is also the same
/// %fill = linalg.fill ins(%cst : f32) outs(%0 :tensor<6140xf32>)->tensor<f32>
/// %inserted_slice = tensor.insert_slice %source into %fill[2046] [2047] [1] :
/// tensor<2047xf32> into tensor<6140xf32>
/// %padded = tensor.pad %inserted_slice low[0] high[-2047]{pad value = %cst} :
/// tensor<6140xf32> to tensor<4093xf32>
/// into:
/// %padded = tensor.pad %source low[2046] high[0] {pad value = %cst} :
/// tensor<2047xf32> to tensor<4093xf32>
struct FoldInsertPadPattern : public OpRewritePattern<tensor::PadOp> {
public:
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    if (failed(canFoldInsertPad(padOp, rewriter)))
      return failure();

    auto insertOp = padOp.getSource().getDefiningOp<InsertSliceOp>();
    auto newSource = insertOp.getSource();
    auto insertDest = insertOp.getDest();
    auto destType = insertDest.getType();
    int64_t rank = destType.getRank();
    SmallVector<OpFoldResult> newHigh;
    SmallVector<OpFoldResult> newLow;
    SmallVector<int64_t> newHighInt;
    SmallVector<int64_t> newLowInt;
    llvm::ArrayRef<int64_t> oldHighStatic = padOp.getStaticHigh();
    llvm::ArrayRef<int64_t> oldLowStatic = padOp.getStaticLow();
    llvm::ArrayRef<int64_t> insertOffsets = insertOp.getStaticOffsets();
    bool isStaticOffset = isStatic(insertOffsets, rank);
    bool isStaticLow = isStatic(oldHighStatic, rank);
    bool isStaticHigh = isStatic(oldLowStatic, rank);
    if (!isStaticLow || !isStaticHigh || !isStaticOffset)
      return rewriter.notifyMatchFailure(insertOp,
                                         "offsets or pad nums not static");
    for (unsigned i = 0; i < rank; ++i) {
      // Insert + pad becomes pad with :
      int64_t offsetCurrent = insertOp.getStaticOffset(i);
      // low = offset + oldLowPad
      auto newLowElement = offsetCurrent + oldLowStatic[i];
      // high = InsertDimSize - offset - insertSize + oldPadHigh
      auto newHighElement = destType.getDimSize(i) - offsetCurrent -
                            insertOp.getStaticSize(i) + oldHighStatic[i];
      newHighInt.push_back(newHighElement);
      newLowInt.push_back(newLowElement);
    }
    newHigh = getAsIndexOpFoldResult(rewriter.getContext(), newHighInt);
    newLow = getAsIndexOpFoldResult(rewriter.getContext(), newLowInt);
    auto padLoc = insertOp.getLoc();
    auto resultType = padOp.getResult().getType();
    Value padValue = padOp.getConstantPaddingValue();
    auto newPadOp = rewriter.create<PadOp>(padLoc, resultType, newSource,
                                           newLow, newHigh, padValue);
    rewriter.replaceOp(padOp, newPadOp.getResult());
    return success();
  }
};

/// This pattern identifies a pad operation whose source is another pad
/// we can fold it into single pad with summed pad values
/// %padded = tensor.pad %source low[2046] high[2047] {pad value = %cst} :
/// tensor<2047xf32> to tensor<6140xf32>
/// %padded2 = tensor.pad %padded low[0] high[-2047] {pad value = %cst} :
/// tensor<6140xf32> to tensor<4093xf32>
/// becomes:
/// %padded = tensor.pad %source low[2046] high[0] {pad value = %cst} :
/// tensor<2047xf32> to tensor<4093xf32>
struct FoldDoublePadPattern : public OpRewritePattern<tensor::PadOp> {
public:
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto oldpadOp = padOp.getSource().getDefiningOp<tensor::PadOp>();
    if (!oldpadOp)
      return rewriter.notifyMatchFailure(padOp, "not a defined by other pad");

    // We can only fold if the padding value is the same as the dst splat value
    Value newPadValue = padOp.getConstantPaddingValue();
    if (!newPadValue)
      return rewriter.notifyMatchFailure(padOp, "pad value not constant");
    Value oldPadValue = padOp.getConstantPaddingValue();
    if (!oldPadValue)
      return rewriter.notifyMatchFailure(padOp, "pad value not constant");
    if (oldPadValue != newPadValue) {
      return rewriter.notifyMatchFailure(padOp, "pad values are not equal");
    }

    unsigned padrank = static_cast<unsigned>(padOp.getResultType().getRank());
    unsigned oldPadrank =
        static_cast<unsigned>(oldpadOp.getResultType().getRank());
    if (padrank != oldPadrank)
      return failure();
    SmallVector<long> newTotalHigh;
    SmallVector<long> newTotalLow;
    SmallVector<OpFoldResult> newHighFold;
    SmallVector<OpFoldResult> newLowFold;
    llvm::ArrayRef<int64_t> oldHighStatic = oldpadOp.getStaticHigh();
    llvm::ArrayRef<int64_t> oldLowStatic = oldpadOp.getStaticLow();
    llvm::ArrayRef<int64_t> newHighStatic = padOp.getStaticHigh();
    llvm::ArrayRef<int64_t> newLowStatic = padOp.getStaticLow();
    bool isStaticLow = isStatic(oldLowStatic, oldPadrank);
    bool isStaticHigh = isStatic(oldHighStatic, oldPadrank);
    bool isStaticLowToo = isStatic(newLowStatic, padrank);
    bool isStaticHighToo = isStatic(newHighStatic, padrank);
    if (!isStaticLow || !isStaticHigh || !isStaticLowToo || !isStaticHighToo)
      return rewriter.notifyMatchFailure(padOp, "Pad not static");

    for (unsigned i = 0; i < padrank; ++i) {
      auto newLowElement = newLowStatic[i] + oldLowStatic[i];
      auto newHighElement = newHighStatic[i] + oldHighStatic[i];
      newTotalHigh.push_back(newHighElement);
      newTotalLow.push_back(newLowElement);
    }
    newHighFold = getAsIndexOpFoldResult(rewriter.getContext(), newTotalHigh);
    newLowFold = getAsIndexOpFoldResult(rewriter.getContext(), newTotalLow);
    auto padLoc = padOp.getLoc();
    auto resultType = padOp.getResult().getType();
    auto newSource = oldpadOp.getSource();
    auto newPadOp =
        rewriter.create<PadOp>(padLoc, resultType, newSource, newLowFold,
                               newHighFold, newPadValue, padOp.getNofold());
    rewriter.replaceOp(padOp, newPadOp.getResult());

    return success();
  }
};

struct FoldStaticNegativeHighPad : public OpRewritePattern<tensor::PadOp> {
public:
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> highs = padOp.getStaticHigh();
    if (llvm::any_of(highs, [](int64_t s) {
          return ShapedType::isDynamic(s) || s > 0;
        })) {
      // TODO: support cases with mixed positive/negative high paddings.
      // e.g.
      // tensor.pad %arg low[0, 0, 0] high[0, 1, -1] :
      //                 tensor<8x8x8xf32> to tensor<8x9x7xf32>
      // ==>
      // %slice = tensor.extract_slice %arg [0, 0, 0] [0, 0, 7]
      //                 tensor<8x8x8xf32> to tensor<8x8x7xf32>
      // tensor.pad %slice low[0, 0, 0] high[0, 1, 0] :
      //                 tensor<8x8x7xf32> to tensor<8x9x7xf32>
      return rewriter.notifyMatchFailure(
          padOp, "cannot fold pad with dynamic or positive high padding num");
    }

    ArrayRef<int64_t> sizes = padOp.getSourceType().getShape();
    if (ShapedType::isDynamicShape(sizes)) {
      return rewriter.notifyMatchFailure(
          padOp, "cannot fold pad with dynamic source size");
    }

    ArrayRef<int64_t> lows = padOp.getStaticLow();
    if (llvm::any_of(lows, [](int64_t s) { return s != 0; })) {
      return rewriter.notifyMatchFailure(
          padOp, "only fold pad with zero low padding num");
    }

    if (llvm::none_of(highs, [](int64_t s) { return s < 0; })) {
      return rewriter.notifyMatchFailure(
          padOp, "only fold pad with static negative high padding num");
    }

    int64_t rank = padOp.getSourceType().getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> newSizes;
    for (int64_t i = 0; i < rank; ++i) {
      int64_t curSize = sizes[i] + highs[i];
      newSizes.push_back(rewriter.getIndexAttr(curSize));
    }

    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        padOp->getLoc(), padOp.getResultType(), padOp.getSource(), offsets,
        newSizes, strides);
    rewriter.replaceOp(padOp, newSliceOp.getResult());
    return success();
  }
};

struct FoldInsertSliceToConcat
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  explicit FoldInsertSliceToConcat(mlir::MLIRContext *ctx,
                                   bool skipAlignedSlice = false)
      : OpRewritePattern<tensor::InsertSliceOp>(ctx),
        skipAlignedSlice(skipAlignedSlice) {}

  LogicalResult matchAndRewrite(tensor::InsertSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    // sliceOp must have strides size one
    ArrayRef<int64_t> strides = sliceOp.getStaticStrides();
    if (llvm::any_of(strides, [](int64_t s) { return s != 1; })) {
      return rewriter.notifyMatchFailure(
          sliceOp, "only can fold insert_slice with strides equal to one");
    }

    // sliceOp must have static offsets and sizes
    ArrayRef<int64_t> offsets = sliceOp.getStaticOffsets();
    ArrayRef<int64_t> sizes = sliceOp.getStaticSizes();
    auto isDynamic = [](int64_t s) { return ShapedType::isDynamic(s); };
    if (llvm::any_of(offsets, isDynamic)) {
      return rewriter.notifyMatchFailure(
          sliceOp, "only can fold insert_slice with static offsets");
    }
    if (llvm::any_of(offsets, isDynamic)) {
      return rewriter.notifyMatchFailure(
          sliceOp, "only can fold insert_slice with static sizes");
    }

    if (skipAlignedSlice && isOffsetsAligned(sliceOp)) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "not fold slice with aligned offsets");
    }

    // slice op must have static shaped src and dst tensor with same rank
    RankedTensorType srcType = sliceOp.getSourceType();
    RankedTensorType dstType = sliceOp.getResultType();
    if (srcType.getNumDynamicDims() != 0 || dstType.getNumDynamicDims() != 0) {
      return rewriter.notifyMatchFailure(
          sliceOp, "cannot fold insert_slice with dynamic shaped src or dst");
    }
    if (srcType.getRank() != dstType.getRank()) {
      // insert_slice supports insertion into a tensor of higher rank than the
      // source tensor, we should avoid this situation
      return rewriter.notifyMatchFailure(
          sliceOp, "cannot fold insert_slice with different src and dst rank");
    }

    // find out on which dimension to concat the slices
    auto sliceDimMaybe = getUniqueConcatDim(srcType, dstType, offsets, sizes);
    if (!sliceDimMaybe.has_value()) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "should slice exactly one dimension");
    }
    int64_t sliceDim = sliceDimMaybe.value();
    Value newConcat = convertInsertSliceToConcatOp(sliceOp, sliceDim, rewriter);
    rewriter.replaceOp(sliceOp, {newConcat});
    return success();
  }

private:
  bool isOffsetsAligned(InsertSliceOp sliceOp) const {
    ArrayRef<int64_t> offsets = sliceOp.getStaticOffsets();
    int64_t bytesPerElem =
        sliceOp.getResultType().getElementTypeBitWidth() / utils::kBitsToByte;
    return llvm::all_of(offsets, [&](int64_t offset) {
      if (ShapedType::isDynamic(offset)) {
        return false;
      }
      int64_t offsetInBytes = offset * bytesPerElem;
      return offsetInBytes % utils::INTR_BYTES_PER_BLOCK == 0;
    });
  }

  void appendIfValueNotEmpty(SmallVector<Value> &vec, Value v) const {
    // helper func to make sure not concat zero sized tensor.
    // v will be empty Value() if it is zero sized tensor.
    if (v) {
      vec.push_back(v);
    }
  }

  std::optional<int64_t> getUniqueConcatDim(RankedTensorType srcType,
                                            RankedTensorType dstType,
                                            ArrayRef<int64_t> offsets,
                                            ArrayRef<int64_t> sizes) const {
    int64_t rank = srcType.getRank();
    int64_t sliceDim = -1;
    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t dstDimSize = dstType.getDimSize(dim);
      if (offsets[dim] == 0 && (sizes[dim] == dstDimSize)) {
        // if slice the whole dimension, not interested
        continue;
      }
      // according to the definition of torch scatter_slice, there should only
      // be one dimension to slice
      if (sliceDim == -1) {
        sliceDim = dim;
      } else {
        return std::nullopt;
      }
    }
    if (sliceDim == -1) {
      // no slice dim found
      return std::nullopt;
    }
    return sliceDim;
  }

  SmallVector<OpFoldResult>
  cloneWithSubstitution(ArrayRef<int64_t> data, int64_t substituteDim,
                        int64_t substituteData,
                        PatternRewriter &rewriter) const {
    SmallVector<int64_t> dataVec{data};
    SmallVector<OpFoldResult> result;
    dataVec[substituteDim] = substituteData;
    for (int64_t value : dataVec) {
      result.push_back(OpFoldResult(rewriter.getIndexAttr(value)));
    }
    return result;
  }

  // extract slices from sliceOp src/dst and concat
  Value convertInsertSliceToConcatOp(tensor::InsertSliceOp sliceOp,
                                     int64_t sliceDim,
                                     PatternRewriter &rewriter) const {
    Location loc = sliceOp->getLoc();
    ArrayRef<int64_t> offsets = sliceOp.getStaticOffsets();
    ArrayRef<int64_t> sizes = sliceOp.getStaticSizes();
    ArrayRef<int64_t> strides = sliceOp.getStaticStrides();
    auto sliceFrom = [&](Value from, int64_t curOffset,
                         int64_t curSize) -> Value {
      if (curSize == 0) {
        // should not extract and concat zero sized tensor.
        // if zero sized, return empty Value()
        return Value();
      }
      SmallVector<OpFoldResult> newOffsets =
          cloneWithSubstitution(offsets, sliceDim, curOffset, rewriter);
      SmallVector<OpFoldResult> newSizes =
          cloneWithSubstitution(sizes, sliceDim, curSize, rewriter);
      SmallVector<OpFoldResult> newStrides =
          cloneWithSubstitution(strides, sliceDim, 1, rewriter);
      auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
          loc, from, newOffsets, newSizes, newStrides);
      return newSliceOp;
    };
    RankedTensorType dstType = sliceOp.getResultType();
    int64_t sliceOffset = offsets[sliceDim];
    int64_t sliceSize = sizes[sliceDim];
    int64_t sliceDstSize = dstType.getDimSize(sliceDim);
    Value src = sliceOp.getSource();
    Value dst = sliceOp->getOperand(1);
    SmallVector<Value> concatInputs;
    // extract `0 ~ offset` from dst tensor
    appendIfValueNotEmpty(concatInputs, sliceFrom(dst, 0, sliceOffset));
    // extract `offset ~ offset+size` from src tensor, exactly src tensor
    appendIfValueNotEmpty(concatInputs, src);
    // extract `offset + size ~ sliceDstSize` from dst tensor
    appendIfValueNotEmpty(concatInputs,
                          sliceFrom(dst, sliceOffset + sliceSize,
                                    sliceDstSize - (sliceOffset + sliceSize)));
    return rewriter.create<tensor::ConcatOp>(loc, sliceOp.getResultType(),
                                             sliceDim, concatInputs);
  }

  bool skipAlignedSlice{false};
};

struct NormalizeLastDimConcatToInterleave
    : public OpRewritePattern<tensor::ConcatOp> {
public:
  using OpRewritePattern<tensor::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    // TODO: remove this check after support interleave with channel_num >= 2
    if (concatOp->getNumOperands() != 2) {
      return rewriter.notifyMatchFailure(
          concatOp, "hfusion interleave op currently only supports channel num "
                    "equals to 2");
    }

    int64_t concatDim = static_cast<int64_t>(concatOp.getDim());
    int64_t rank = concatOp.getRank();
    if (concatDim != rank - 1) {
      return rewriter.notifyMatchFailure(
          concatOp, "only can normalize last dim concat op to interleave op");
    }

    SmallVector<Value> inputs = concatOp.getInputs();
    if (!llvm::all_of(inputs, [&](Value value) {
          return isTensorWithOneSizeDim(value, concatDim);
        })) {
      return rewriter.notifyMatchFailure(
          concatOp, "only can normalize last dim concat op to interleave op if "
                    "last dim is one size");
    }

    Location loc = concatOp->getLoc();
    auto newOp = rewriter.create<hfusion::InterleaveOp>(
        loc, ValueRange{concatOp->getResults()}, ValueRange{inputs});
    rewriter.replaceOp(concatOp, newOp);
    return success();
  }

private:
  bool isTensorWithOneSizeDim(Value value, int64_t dim) const {
    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType) {
      return false;
    }
    return tensorType.getDimSize(dim) == 1;
  }
};

struct FoldGenerateToFill : public OpRewritePattern<tensor::GenerateOp> {
public:
  using OpRewritePattern<tensor::GenerateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::GenerateOp genOp,
                                PatternRewriter &rewriter) const override {
    Block &body = genOp.getBody().front();
    if (!llvm::hasSingleElement(body))
      return rewriter.notifyMatchFailure(genOp, "Body not single element");

    auto yieldOp = dyn_cast<tensor::YieldOp>(body.getTerminator());
    if (!yieldOp)
      return rewriter.notifyMatchFailure(genOp, "Terminator not a YieldOp");

    Value fillValue =
        yieldOp.getValue(); // TODO: Might need some check if constant scalar
    auto dynSizes = genOp.getDynamicExtents();
    auto resType = genOp.getType();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        genOp.getLoc(), cast<RankedTensorType>(resType).getShape(),
        cast<RankedTensorType>(resType).getElementType(), ValueRange(dynSizes));
    auto fillOp = rewriter.create<linalg::FillOp>(
        genOp.getLoc(), ValueRange(fillValue), ValueRange(emptyOp));
    rewriter.replaceOp(genOp, fillOp);
    return success();
  }
};

struct NormalizeInterleaveExpandReshapePattern
    : public OpRewritePattern<hfusion::InterleaveOp> {
  using OpRewritePattern<hfusion::InterleaveOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::InterleaveOp interleaveOp,
                                PatternRewriter &rewriter) const override {
    Value lhs = interleaveOp.getOperand(0);
    Value rhs = interleaveOp.getOperand(1);
    auto lhsExpandOp = lhs.getDefiningOp<tensor::ExpandShapeOp>();
    auto rhsExpandOp = rhs.getDefiningOp<tensor::ExpandShapeOp>();
    if (!lhsExpandOp || !rhsExpandOp)
      return failure();

    // get types dynamically
    auto lhsSourceType =
        mlir::dyn_cast<RankedTensorType>(lhsExpandOp.getSrc().getType());
    auto rhsSourceType =
        mlir::dyn_cast<RankedTensorType>(rhsExpandOp.getSrc().getType());
    auto lhsExpandedType = mlir::dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsExpandedType = mlir::dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsSourceType || !rhsSourceType || !lhsExpandedType ||
        !rhsExpandedType)
      return failure();

    if (!validateExpandOperations(lhsSourceType, rhsSourceType, lhsExpandedType,
                                  rhsExpandedType))
      return failure();

    if (lhsSourceType != rhsSourceType)
      return failure();

    // check single reshape user with valid target shape
    auto users = llvm::to_vector(interleaveOp->getUsers());
    if (users.size() != 1)
      return failure();
    auto reshapeOp = dyn_cast<tensor::ReshapeOp>(users[0]);
    if (!reshapeOp)
      return failure();
    auto reshapeType =
        mlir::dyn_cast<RankedTensorType>(reshapeOp.getResultType());
    if (!reshapeType)
      return failure();

    if (!validateReshapeOperation(reshapeType, lhsSourceType))
      return failure();

    // create optimized interleave directly producing final shape
    auto newInterleave = rewriter.create<hfusion::InterleaveOp>(
        interleaveOp.getLoc(), reshapeOp.getResultType(),
        ValueRange{lhsExpandOp.getSrc(), rhsExpandOp.getSrc()});
    rewriter.replaceOp(reshapeOp, newInterleave.getResult());
    return success();
  }

private:
  bool validateExpandOperations(RankedTensorType lhsSourceType,
                                RankedTensorType rhsSourceType,
                                RankedTensorType lhsExpandedType,
                                RankedTensorType rhsExpandedType) const {
    // check expanded rank = source rank + 1
    if (lhsExpandedType.getRank() != lhsSourceType.getRank() + 1 ||
        rhsExpandedType.getRank() != rhsSourceType.getRank() + 1)
      return false;

    // check last dimension is 1
    const int64_t lastDimLhs = lhsExpandedType.getRank() - 1;
    const int64_t lastDimRhs = rhsExpandedType.getRank() - 1;
    if (lhsExpandedType.getDimSize(lastDimLhs) != 1 ||
        rhsExpandedType.getDimSize(lastDimRhs) != 1)
      return false;

    // verify all preceding dimensions match source
    for (int64_t i = 0; i < lhsSourceType.getRank(); ++i) {
      if (lhsExpandedType.getDimSize(i) != lhsSourceType.getDimSize(i) ||
          rhsExpandedType.getDimSize(i) != rhsSourceType.getDimSize(i))
        return false;
    }
    return true;
  }

  bool validateReshapeOperation(RankedTensorType reshapeType,
                                RankedTensorType lhsSourceType) const {
    const int64_t sourceRank = lhsSourceType.getRank();
    if (reshapeType.getRank() != sourceRank)
      return false;

    // check all but last dimension match source
    for (int64_t i = 0; i < sourceRank - 1; ++i) {
      if (reshapeType.getDimSize(i) != lhsSourceType.getDimSize(i))
        return false;
    }

    // check last dimension = source last dim * 2
    return reshapeType.getDimSize(sourceRank - 1) ==
           lhsSourceType.getDimSize(sourceRank - 1) * 2;
  }
};

// not used but this will make the alloc 1D for the input of the deinterleave
struct NormalizeAllocaForDeinterleavePattern
    : public OpRewritePattern<hfusion::DeinterleaveOp> {
  explicit NormalizeAllocaForDeinterleavePattern(MLIRContext *context)
      : OpRewritePattern<hfusion::DeinterleaveOp>(context, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(hfusion::DeinterleaveOp deinterleaveOp,
                                PatternRewriter &rewriter) const override {
    Location loc = deinterleaveOp.getLoc();
    Value src = deinterleaveOp.getInput();

    auto [toTensorOp, allocOp] = validateOperationChain(src);
    if (!toTensorOp || !allocOp)
      return failure();

    auto [srcType, deintResultType] = getTensorTypes(deinterleaveOp, src);
    if (!srcType || !deintResultType)
      return failure();

    if (!isValidRankAndDimensions(srcType, deintResultType))
      return failure();

    auto allocType = getValidAllocType(allocOp);
    if (!allocType)
      return failure();

    // calculate total elements
    int64_t totalElems = computeTotalElements(allocType);

    // create new 1D alloc
    rewriter.setInsertionPoint(allocOp);
    auto newAllocType =
        MemRefType::get({totalElems}, allocType.getElementType());
    Value newAlloc = rewriter.create<memref::AllocOp>(loc, newAllocType);

    // create reinterpret_cast for 1D alloc to original shape
    SmallVector<int64_t> strides = computeStrides(allocType);

    Value reinterpretedAlloc = rewriter.create<memref::ReinterpretCastOp>(
        loc,
        MemRefType::get(allocType.getShape(), allocType.getElementType(),
                        StridedLayoutAttr::get(getContext(), 0, strides)),
        newAlloc, 0, allocType.getShape(), strides);

    // replace all uses of original alloc with reinterpreted alloc
    rewriter.replaceOp(allocOp, reinterpretedAlloc);

    return success();
  }

private:
  std::pair<bufferization::ToTensorOp, memref::AllocOp>
  validateOperationChain(Value src) const {
    auto toTensorOp = src.getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensorOp)
      return {nullptr, nullptr};

    auto allocOp = toTensorOp.getMemref().getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return {nullptr, nullptr};

    return {toTensorOp, allocOp};
  }

  std::pair<RankedTensorType, RankedTensorType>
  getTensorTypes(hfusion::DeinterleaveOp deinterleaveOp, Value src) const {
    auto srcType = mlir::dyn_cast<RankedTensorType>(src.getType());
    auto deintResultType =
        mlir::dyn_cast<RankedTensorType>(deinterleaveOp.getResult(0).getType());
    return {srcType, deintResultType};
  }

  bool isValidRankAndDimensions(RankedTensorType srcType,
                                RankedTensorType deintResultType) const {
    int64_t rank = srcType.getRank();
    if (rank < 2)
      return false;
    if (srcType.getDimSize(rank - 1) != 2 ||
        deintResultType.getDimSize(rank - 1) != 1)
      return false;
    return true;
  }

  MemRefType getValidAllocType(memref::AllocOp allocOp) const {
    auto allocType = mlir::dyn_cast<MemRefType>(allocOp.getType());
    if (!allocType || !allocType.hasStaticShape())
      return MemRefType();
    return allocType;
  }

  int64_t computeTotalElements(MemRefType allocType) const {
    int64_t totalElems = 1;
    for (int64_t dim : allocType.getShape()) {
      totalElems *= dim;
    }
    return totalElems;
  }

  SmallVector<int64_t> computeStrides(MemRefType allocType) const {
    int64_t rank = allocType.getRank();
    SmallVector<int64_t> strides;
    int64_t stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
      strides.insert(strides.begin(), stride);
      stride *= allocType.getDimSize(i);
    }
    return strides;
  }
};

// not used but this will collapse deinterleave to just use input and output 1D
struct NormalizeDeInterleavePattern
    : public OpRewritePattern<hfusion::DeinterleaveOp> {
  using OpRewritePattern<hfusion::DeinterleaveOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::DeinterleaveOp deinterleaveOp,
                                PatternRewriter &rewriter) const override {
    Value src = deinterleaveOp.getInput();
    auto srcType = mlir::dyn_cast<RankedTensorType>(src.getType());
    auto deintResultType =
        mlir::dyn_cast<RankedTensorType>(deinterleaveOp.getResult(0).getType());

    if (failed(validateInputs(deinterleaveOp, src, srcType, deintResultType)))
      return failure();

    Operation *userOp = *deinterleaveOp.getResult(0).user_begin();
    tensor::CollapseShapeOp collapseOp;
    RankedTensorType collapseResultType;

    if (failed(validateCollapseOp(userOp, collapseOp, collapseResultType)))
      return failure();

    // Compute total source elements
    ArrayRef<int64_t> srcShape = srcType.getShape();
    int64_t totalSrcElems;
    if (failed(computeTotalElements(srcShape, totalSrcElems)))
      return failure();

    // build target (1D) shape vector
    SmallVector<int64_t, 1> targetShapeVec{totalSrcElems};

    // get reassociation indices to collapse srcShape -> [totalSrcElems]
    auto collapseReassoc =
        mlir::getReassociationIndicesForCollapse(srcShape, targetShapeVec);
    if (!collapseReassoc.has_value())
      return failure();

    // create the collapse to 1D
    Value fullyCollapsedSrc = rewriter.create<tensor::CollapseShapeOp>(
        deinterleaveOp.getLoc(),
        RankedTensorType::get(targetShapeVec, srcType.getElementType()), src,
        collapseReassoc.value());

    // Compute total collapsed result elements
    ArrayRef<int64_t> collapsedResultShape = collapseResultType.getShape();
    int64_t totalCollapsedResultElems;
    if (failed(computeTotalElements(collapsedResultShape,
                                    totalCollapsedResultElems)))
      return failure();

    auto elemType = collapseResultType.getElementType();
    RankedTensorType newDeintResultTy =
        RankedTensorType::get({totalCollapsedResultElems}, elemType);
    auto channelAttr = deinterleaveOp.getChannelIndexAttr();

    // create the new deinterleave that takes the fully-collapsed 1D input
    auto newDeint = rewriter.create<hfusion::DeinterleaveOp>(
        deinterleaveOp.getLoc(), newDeintResultTy, fullyCollapsedSrc,
        channelAttr);

    auto expandReassoc = mlir::getReassociationIndicesForReshape(
        newDeintResultTy, collapseResultType);
    if (!expandReassoc.has_value())
      return failure();

    // create tensor.expand_shape from 1D back to collapseOp result type.
    Value expandedResult = rewriter
                               .create<tensor::ExpandShapeOp>(
                                   deinterleaveOp.getLoc(), collapseResultType,
                                   newDeint.getResult(0), expandReassoc.value())
                               .getResult();

    // replace the original collapse op's use with the expanded result.
    rewriter.replaceOp(collapseOp, expandedResult);
    return success();
  }

private:
  LogicalResult validateInputs(hfusion::DeinterleaveOp deinterleaveOp,
                               Value src, RankedTensorType srcType,
                               RankedTensorType deintResultType) const {
    if (!srcType || !deintResultType)
      return failure();

    int64_t rank = srcType.getRank();
    if (rank < 2)
      return failure();

    // require last dim == 2 and deinterleave result's last dim == 1 (same as
    // before)
    if (srcType.getDimSize(rank - 1) != 2 ||
        deintResultType.getDimSize(rank - 1) != 1)
      return failure();

    // only user of the deinterleave result is the collapse
    if (!llvm::hasSingleElement(deinterleaveOp.getResult(0).getUsers()))
      return failure();

    return success();
  }

  LogicalResult validateCollapseOp(Operation *userOp,
                                   tensor::CollapseShapeOp &collapseOp,
                                   RankedTensorType &collapseResultType) const {
    collapseOp = dyn_cast<tensor::CollapseShapeOp>(userOp);
    if (!collapseOp)
      return failure();

    collapseResultType =
        mlir::dyn_cast<RankedTensorType>(collapseOp.getResult().getType());
    if (!collapseResultType)
      return failure();

    return success();
  }

  LogicalResult computeTotalElements(ArrayRef<int64_t> shape,
                                     int64_t &totalElems) const {
    totalElems = 1;
    for (int64_t d : shape) {
      if (ShapedType::isDynamic(d))
        return failure(); // conservative
      totalElems *= d;
    }
    return success();
  }
};

void NormalizeTensorOps::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<FoldInsertPadPattern>(patterns.getContext());
  patterns.insert<FoldDoublePadPattern>(patterns.getContext());
  // TODO: we can disable it if insert_slice implements tiling interface
  patterns.insert<FoldInsertSliceToConcat>(patterns.getContext(),
                                           this->skipAlignedSlice);
  patterns.insert<FoldGenerateToFill>(patterns.getContext());
  patterns.insert<NormalizeLastDimConcatToInterleave>(patterns.getContext());
  patterns.insert<FoldStaticNegativeHighPad>(patterns.getContext());
  patterns.insert<NormalizeInterleaveExpandReshapePattern>(
      patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass>
mlir::tensor::createNormalizeTensorOpsPass(bool skipAlignedSlice) {
  return std::make_unique<NormalizeTensorOps>(skipAlignedSlice);
}
