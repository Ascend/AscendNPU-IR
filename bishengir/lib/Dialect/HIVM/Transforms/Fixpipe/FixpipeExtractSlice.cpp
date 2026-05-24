//===-------------------- FixpipeExtractSlice.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#define DEBUG_TYPE "fixpipe-opts"

using namespace mlir;
using namespace mlir::hivm;

namespace {
//===----------------------------------------------------------------------===//
// Helper: Compute slice parameters in nZ layout from ND parameters
//===----------------------------------------------------------------------===//

FailureOr<std::pair<SmallVector<Value>, SmallVector<Value>>>
computeSliceParamsInNZLayout(
    ArrayRef<OpFoldResult> mixedOffsets,
    ArrayRef<OpFoldResult> mixedSizes,
    RankedTensorType sourceType,
    OpBuilder &builder,
    Location loc) {

  auto sourceShape = sourceType.getShape();
  int64_t sourceRank = sourceType.getRank();

  if (sourceRank < 4)
    return failure();

  int batchIndexBias = (sourceRank == 5) ? 1 : 0;

  // For nZ layout: shape is (a1, b1, b0, a0) or (batch, a1, b1, b0, a0)
  // Extract block sizes from shape
  int64_t a0 = sourceShape[3 + batchIndexBias];
  int64_t b0 = sourceShape[2 + batchIndexBias];

  // Convert ND offsets to Values
  SmallVector<Value> offsetValues;
  for (OpFoldResult ofr : mixedOffsets) {
    offsetValues.push_back(getValueOrCreateConstantIndexOp(builder, loc, ofr));
  }

  Value a0Val = builder.create<arith::ConstantIndexOp>(loc, a0);
  Value b0Val = builder.create<arith::ConstantIndexOp>(loc, b0);

  Value i = offsetValues[0 + batchIndexBias]; // a dimension offset in ND
  Value j = offsetValues[1 + batchIndexBias]; // b dimension offset in ND

  // Transform ND offsets to nZ offsets
  // ND: (i, j) -> nZ: (i/a0, j/b0, j%b0, i%a0)
  Value a1Idx = builder.create<arith::DivUIOp>(loc, i, a0Val);
  Value b1Idx = builder.create<arith::DivUIOp>(loc, j, b0Val);
  Value b0Idx = builder.create<arith::RemUIOp>(loc, j, b0Val);
  Value a0Idx = builder.create<arith::RemUIOp>(loc, i, a0Val);

  SmallVector<Value> newOffsets;
  if (batchIndexBias) {
    newOffsets.push_back(offsetValues[0]);
  }
  newOffsets.append({a1Idx, b1Idx, b0Idx, a0Idx});

  // Get ND sizes and transform to nZ sizes
  Value size_a = getValueOrCreateConstantIndexOp(
      builder, loc, mixedSizes[0 + batchIndexBias]);
  Value size_b = getValueOrCreateConstantIndexOp(
      builder, loc, mixedSizes[1 + batchIndexBias]);

  // Compute tiled sizes: a1_size = size_a / a0, b1_size = size_b / b0
  // Inner tile sizes are full (a0, b0)
  Value a1_size = builder.create<arith::DivUIOp>(loc, size_a, a0Val);
  Value b1_size = builder.create<arith::DivUIOp>(loc, size_b, b0Val);

  SmallVector<Value> newSizes;
  if (batchIndexBias) {
    newSizes.push_back(
        getValueOrCreateConstantIndexOp(builder, loc, mixedSizes[0]));
  }
  // nZ sizes: (a1_size, b1_size, b0, a0)
  newSizes.append({a1_size, b1_size, b0Val, a0Val});

  return std::make_pair(std::move(newOffsets), std::move(newSizes));
}

//===----------------------------------------------------------------------===//
// Swap Fixpipe (nZ2ND) and ExtractSlice
//===----------------------------------------------------------------------===//

/// Pattern: Push fixpipe DOWN through tensor.extract_slice operations
/// Before:
///   %t1 = hivm.hir.fixpipe {dma_mode = nz2nd} ins(%t0) outs(...)  // nZ -> ND
///   %slice = tensor.extract_slice %t1[...][...][1,1]
/// After:
///   %new_slice = tensor.extract_slice %t0[...'][...'][1,1,1,1]  // slice in nZ
///   %result = hivm.hir.fixpipe {dma_mode = nz2nd} ins(%new_slice) outs(...)
struct SwapFixpipeDownThroughExtractSlice
    : public OpRewritePattern<FixpipeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FixpipeOp fixpipeOp,
                                PatternRewriter &rewriter) const override {
    // Only handle nZ to ND conversion
    auto dmaModeAttr = fixpipeOp.getDmaModeAttr();
    if (!dmaModeAttr || dmaModeAttr.getValue() != FixpipeDMAMode::NZ2ND)
      return failure();

    if (fixpipeOp->use_empty())
      return rewriter.notifyMatchFailure(fixpipeOp, "fixpipe has no uses");

    // Find an extract_slice user
    auto findIt = llvm::find_if(fixpipeOp->getUsers(), [](Operation *user) {
      return isa<tensor::ExtractSliceOp>(user);
    });

    if (findIt == fixpipeOp->getUsers().end())
      return rewriter.notifyMatchFailure(fixpipeOp,
                                         "no tensor.extract_slice user found");

    auto extractSliceOp = cast<tensor::ExtractSliceOp>(*findIt);

    // Only support unit strides
    for (OpFoldResult stride : extractSliceOp.getMixedStrides()) {
      std::optional<int64_t> strideVal = getConstantIntValue(stride);
      if (!strideVal || *strideVal != 1)
        return rewriter.notifyMatchFailure(fixpipeOp,
                                           "extract_slice has non-unit or dynamic strides");
    }

    Location loc = extractSliceOp.getLoc();
    Value source = fixpipeOp.getDpsInputOperand(0)->get();
    auto sourceType = cast<RankedTensorType>(source.getType());
    int64_t sourceRank = sourceType.getRank();

    rewriter.setInsertionPoint(extractSliceOp);

    // Compute slice parameters in source (nZ) layout
    auto sliceParamsResult = computeSliceParamsInNZLayout(
        extractSliceOp.getMixedOffsets(),
        extractSliceOp.getMixedSizes(),
        sourceType,
        rewriter,
        loc);

    if (failed(sliceParamsResult))
      return rewriter.notifyMatchFailure(fixpipeOp,
                                         "failed to compute slice parameters in nZ layout");

    auto &[newOffsets, newSizes] = *sliceParamsResult;

    // Convert to OpFoldResult
    SmallVector<OpFoldResult> newOffsetsOFR, newSizesOFR;
    for (Value v : newOffsets)
      newOffsetsOFR.push_back(v);
    for (Value v : newSizes)
      newSizesOFR.push_back(v);

    // Create unit strides for source rank
    SmallVector<OpFoldResult> newStrides(sourceRank, rewriter.getIndexAttr(1));

    // Result type for new extract_slice (dynamic since sizes may be dynamic)
    SmallVector<int64_t> newSliceShape(sourceRank, ShapedType::kDynamic);
    auto newSliceType = RankedTensorType::get(
        newSliceShape, sourceType.getElementType());

    // Create new extract_slice on source (in nZ layout)
    auto newExtractSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, newSliceType, source,
        newOffsetsOFR, newSizesOFR, newStrides);

    // Create new fixpipe to convert sliced nZ to ND
    auto quantModeAttr = fixpipeOp.getPreQuantAttr();
    auto reluModeAttr = fixpipeOp.getPreReluAttr();

    auto newExtractSliceResult = newExtractSlice.getResult();

    // Determine output element type from original fixpipe output
    Type outputElemType = extractSliceOp.getResultType().getElementType();

    Value fixpipeInit = utils::createEmptyOpWithTargetElemType(
        rewriter, loc, newExtractSliceResult, outputElemType);

    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr newDmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);

    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        loc, fixpipeInit.getType(),
        /*src=*/newExtractSliceResult,
        /*dst=*/fixpipeInit,
        newDmaModeAttr,
        FixpipeDualDstModeAttr{},
        quantModeAttr,
        reluModeAttr);

    // Replace original extract_slice with new fixpipe result
    rewriter.replaceOp(extractSliceOp, newFixpipeOp.getResultTensor());

    // Clean up old fixpipe if it has no more users
    if (fixpipeOp->use_empty())
      rewriter.eraseOp(fixpipeOp);

    return success();
  }
};
}

void mlir::hivm::populateFixpipeExtractSlice(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add<SwapFixpipeDownThroughExtractSlice>(context);
}