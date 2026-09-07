//===--- NormalizeTensorOps.cpp -  optimize tensor ops --------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025~2026. All rights reserved.
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
//
// This file implements a pass to canonicalize pad operation sourced from
// Insert_slice operation
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
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
  using NormalizeTensorOpsBase::NormalizeTensorOpsBase;

  // Programmatic construction: set the TableGen option (do not shadow it).
  explicit NormalizeTensorOps(bool skipAlignedSliceArg) {
    skipAlignedSlice = skipAlignedSliceArg;
  }

  void runOnOperation() override;
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
    SmallVector<int64_t> newTotalHigh;
    SmallVector<int64_t> newTotalLow;
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

struct FoldInsertSliceToConcat : OpRewritePattern<tensor::InsertSliceOp> {
  FoldInsertSliceToConcat(MLIRContext *context, bool skipAlignedSlice)
      : OpRewritePattern(context), skipAlignedSlice(skipAlignedSlice) {}

  LogicalResult matchAndRewrite(tensor::InsertSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> strides = sliceOp.getStaticStrides();
    if (llvm::any_of(strides, [](int64_t stride) { return stride != 1; }))
      return rewriter.notifyMatchFailure(
          sliceOp, "only folds insert_slice operations with unit strides");

    ArrayRef<int64_t> offsets = sliceOp.getStaticOffsets();
    ArrayRef<int64_t> sizes = sliceOp.getStaticSizes();
    auto isDynamic = [](int64_t value) { return ShapedType::isDynamic(value); };
    if (llvm::any_of(offsets, isDynamic) || llvm::any_of(sizes, isDynamic))
      return rewriter.notifyMatchFailure(
          sliceOp, "only folds insert_slice operations with static offsets and "
                   "sizes");

    if (skipAlignedSlice && areOffsetsBlockAligned(sliceOp))
      return rewriter.notifyMatchFailure(
          sliceOp, "aligned offsets are disabled by skip-aligned-slice");

    RankedTensorType sourceType = sliceOp.getSourceType();
    RankedTensorType resultType = sliceOp.getResultType();
    if (sourceType.getNumDynamicDims() != 0 ||
        resultType.getNumDynamicDims() != 0)
      return rewriter.notifyMatchFailure(
          sliceOp, "cannot fold insert_slice with dynamic source or result");
    if (sourceType.getRank() != resultType.getRank())
      return rewriter.notifyMatchFailure(
          sliceOp, "cannot fold insert_slice with different source and result "
                   "ranks");

    std::optional<int64_t> concatDim =
        getUniqueConcatDim(sourceType, resultType, offsets, sizes);
    if (!concatDim)
      return rewriter.notifyMatchFailure(
          sliceOp, "the insertion must slice exactly one dimension");

    // Only a last-dim concat has a copy-free lowering (interleave); on any
    // other dimension the concat materializes a new buffer plus copies,
    // whereas the insert_slice form bufferizes in place.
    if (*concatDim != resultType.getRank() - 1)
      return rewriter.notifyMatchFailure(
          sliceOp, "only folds insertions along the innermost dimension");

    rewriter.replaceOp(sliceOp, convertToConcat(sliceOp, *concatDim, rewriter));
    return success();
  }

private:
  static bool areOffsetsBlockAligned(tensor::InsertSliceOp sliceOp) {
    int64_t bitWidth = sliceOp.getResultType().getElementTypeBitWidth();
    // Sub-byte elements make bytesPerElement 0 via integer division, which
    // would treat every offset as aligned. Decline the heuristic instead.
    if (bitWidth < static_cast<int64_t>(utils::kBitsToByte) ||
        bitWidth % utils::kBitsToByte != 0)
      return false;
    int64_t bytesPerElement = bitWidth / utils::kBitsToByte;
    return llvm::all_of(sliceOp.getStaticOffsets(), [&](int64_t offset) {
      if (ShapedType::isDynamic(offset))
        return false;
      return offset * bytesPerElement % utils::INTR_BYTES_PER_BLOCK == 0;
    });
  }

  static std::optional<int64_t> getUniqueConcatDim(RankedTensorType sourceType,
                                                   RankedTensorType resultType,
                                                   ArrayRef<int64_t> offsets,
                                                   ArrayRef<int64_t> sizes) {
    std::optional<int64_t> concatDim;
    for (int64_t dim = 0, rank = sourceType.getRank(); dim < rank; ++dim) {
      if (offsets[dim] == 0 && sizes[dim] == resultType.getDimSize(dim))
        continue;
      if (concatDim)
        return std::nullopt;
      concatDim = dim;
    }
    return concatDim;
  }

  static SmallVector<OpFoldResult>
  withSubstitutedDimension(ArrayRef<int64_t> values, int64_t dimension,
                           int64_t replacement, PatternRewriter &rewriter) {
    SmallVector<OpFoldResult> results;
    results.reserve(values.size());
    for (auto [index, value] : llvm::enumerate(values))
      results.push_back(
          rewriter.getIndexAttr(index == dimension ? replacement : value));
    return results;
  }

  static Value createSlice(Value source, Location loc,
                           ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes,
                           ArrayRef<int64_t> strides, int64_t dimension,
                           int64_t offset, int64_t size,
                           PatternRewriter &rewriter) {
    if (size == 0)
      return {};
    return rewriter.create<tensor::ExtractSliceOp>(
        loc, source,
        withSubstitutedDimension(offsets, dimension, offset, rewriter),
        withSubstitutedDimension(sizes, dimension, size, rewriter),
        withSubstitutedDimension(strides, dimension, 1, rewriter));
  }

  static Value convertToConcat(tensor::InsertSliceOp sliceOp, int64_t concatDim,
                               PatternRewriter &rewriter) {
    ArrayRef<int64_t> offsets = sliceOp.getStaticOffsets();
    ArrayRef<int64_t> sizes = sliceOp.getStaticSizes();
    ArrayRef<int64_t> strides = sliceOp.getStaticStrides();
    int64_t insertOffset = offsets[concatDim];
    int64_t insertSize = sizes[concatDim];
    int64_t resultSize = sliceOp.getResultType().getDimSize(concatDim);
    Location loc = sliceOp.getLoc();

    SmallVector<Value> inputs;
    if (Value prefix =
            createSlice(sliceOp.getDest(), loc, offsets, sizes, strides,
                        concatDim, 0, insertOffset, rewriter))
      inputs.push_back(prefix);
    inputs.push_back(sliceOp.getSource());
    if (Value suffix =
            createSlice(sliceOp.getDest(), loc, offsets, sizes, strides,
                        concatDim, insertOffset + insertSize,
                        resultSize - insertOffset - insertSize, rewriter))
      inputs.push_back(suffix);

    return rewriter.create<tensor::ConcatOp>(loc, sliceOp.getResultType(),
                                             concatDim, inputs);
  }

  bool skipAlignedSlice;
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

struct NormalizeInterleaveExpandReshape
    : OpRewritePattern<hfusion::InterleaveOp> {
  using OpRewritePattern<hfusion::InterleaveOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::InterleaveOp interleaveOp,
                                PatternRewriter &rewriter) const override {
    if (interleaveOp->getNumOperands() != 2)
      return failure();

    auto lhsExpand =
        interleaveOp.getOperand(0).getDefiningOp<tensor::ExpandShapeOp>();
    auto rhsExpand =
        interleaveOp.getOperand(1).getDefiningOp<tensor::ExpandShapeOp>();
    if (!lhsExpand || !rhsExpand)
      return failure();

    auto sourceType = dyn_cast<RankedTensorType>(lhsExpand.getSrc().getType());
    auto rhsSourceType =
        dyn_cast<RankedTensorType>(rhsExpand.getSrc().getType());
    auto lhsType = dyn_cast<RankedTensorType>(lhsExpand.getResult().getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhsExpand.getResult().getType());
    if (!sourceType || sourceType != rhsSourceType || !lhsType ||
        lhsType != rhsType || !isUnitLastDimExpansion(sourceType, lhsType))
      return failure();

    if (!interleaveOp.getResult().hasOneUse())
      return failure();
    auto reshapeOp = dyn_cast<tensor::ReshapeOp>(
        *interleaveOp.getResult().getUsers().begin());
    auto reshapeType =
        reshapeOp ? dyn_cast<RankedTensorType>(reshapeOp.getResult().getType())
                  : RankedTensorType();
    if (!reshapeType || !isInterleaveResultShape(sourceType, reshapeType))
      return failure();

    auto normalized = rewriter.create<hfusion::InterleaveOp>(
        interleaveOp.getLoc(), reshapeType,
        ValueRange{lhsExpand.getSrc(), rhsExpand.getSrc()});
    rewriter.replaceOp(reshapeOp, normalized.getResult());
    return success();
  }

private:
  static bool isUnitLastDimExpansion(RankedTensorType sourceType,
                                     RankedTensorType expandedType) {
    if (expandedType.getRank() != sourceType.getRank() + 1 ||
        expandedType.getDimSize(expandedType.getRank() - 1) != 1)
      return false;
    return llvm::equal(sourceType.getShape(),
                       expandedType.getShape().drop_back());
  }

  static bool isInterleaveResultShape(RankedTensorType sourceType,
                                      RankedTensorType reshapeType) {
    if (reshapeType.getRank() != sourceType.getRank())
      return false;
    int64_t lastDim = sourceType.getRank() - 1;
    return llvm::equal(sourceType.getShape().take_front(lastDim),
                       reshapeType.getShape().take_front(lastDim)) &&
           reshapeType.getDimSize(lastDim) ==
               sourceType.getDimSize(lastDim) * 2;
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

void NormalizeTensorOps::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<FoldInsertPadPattern>(patterns.getContext());
  patterns.insert<FoldDoublePadPattern>(patterns.getContext());
  patterns.insert<FoldInsertSliceToConcat>(patterns.getContext(),
                                           skipAlignedSlice);
  patterns.insert<FoldGenerateToFill>(patterns.getContext());
  patterns.insert<NormalizeLastDimConcatToInterleave>(patterns.getContext());
  patterns.insert<FoldStaticNegativeHighPad>(patterns.getContext());
  patterns.insert<NormalizeInterleaveExpandReshape>(patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass>
mlir::tensor::createNormalizeTensorOpsPass(bool skipAlignedSlice) {
  return std::make_unique<NormalizeTensorOps>(skipAlignedSlice);
}
