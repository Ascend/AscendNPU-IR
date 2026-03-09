//===-------------------- PropagateConvertLayoutExtractSlice.cpp ----------===//
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "hivm-propagate-convert-layout"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace {

//===----------------------------------------------------------------------===//
// Propagate DOWN through ExtractSlice Operations
//===----------------------------------------------------------------------===//

/// Pattern: Push convert_layout DOWN through tensor.extract_slice operations
/// Before:
///   %t1 = hivm.hir.convert_layout %t0 {down}  // nZ -> ND
///   %slice = tensor.extract_slice %t1[...][...][1,1]
/// After:
///   %new_slice = tensor.extract_slice %t0[...'][...'][1,1,1,1]  // slice in nZ
///   %result = hivm.hir.convert_layout %new_slice {down}  // nZ -> ND
struct PropagateConvertLayoutDownThroughExtractSlice
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    // Only handle down propagation
    if (!isPropagatingDown(convertOp))
      return failure();

    if (convertOp->use_empty())
      return rewriter.notifyMatchFailure(convertOp,
                                         "convert_layout has no uses");

    // Find an extract_slice user
    auto findIt = llvm::find_if(convertOp->getUsers(), [](Operation *user) {
      return isa<tensor::ExtractSliceOp>(user);
    });

    if (findIt == convertOp->getUsers().end())
      return rewriter.notifyMatchFailure(convertOp,
                                         "no tensor.extract_slice user found");

    auto extractSliceOp = cast<tensor::ExtractSliceOp>(*findIt);

    // Only support unit strides
    for (OpFoldResult stride : extractSliceOp.getMixedStrides()) {
      std::optional<int64_t> strideVal = getConstantIntValue(stride);
      if (!strideVal || *strideVal != 1)
        return rewriter.notifyMatchFailure(convertOp,
                                           "extract_slice has non-unit or dynamic strides");
    }

    Location loc = extractSliceOp.getLoc();
    Value source = convertOp.getSource();
    auto sourceType = cast<RankedTensorType>(source.getType());
    int64_t sourceRank = sourceType.getRank();

    // Compute slice parameters in source layout
    rewriter.setInsertionPoint(extractSliceOp);

    auto newSizes = computeMixedTargetLayoutShape(
        extractSliceOp.getMixedSizes(), convertOp.getDstLayout(),
        convertOp.getSrcLayout(), rewriter, loc);

    if (failed(newSizes))
      return rewriter.notifyMatchFailure(convertOp,
                                         "failed to compute size in source layout");

    auto newOffsets = computeTargetLayoutOffset(extractSliceOp.getMixedOffsets(),
                                                convertOp.getDstLayout(),
                                                convertOp.getSrcLayout(),
                                                rewriter, loc);

    if (failed(newOffsets))
      return rewriter.notifyMatchFailure(convertOp,
                                         "failed to compute offset in source layout");

    // Create unit strides for source rank
    SmallVector<OpFoldResult> newStrides(sourceRank, rewriter.getIndexAttr(1));

    // Result type for new extract_slice (dynamic since sizes may be dynamic)
    auto newSliceType = RankedTensorType::get(
        decomposeMixedValues(*newSizes).first, sourceType.getElementType());

    // Create new extract_slice on source (in srcLayout)
    auto newExtractSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, newSliceType, source,
        *newOffsets, *newSizes, newStrides);

    // Create convert_layout to match original extract_slice result type
    auto originalResultType = extractSliceOp.getResult().getType();
    auto newConvert = rewriter.create<ConvertLayoutOp>(
        loc, originalResultType, newExtractSlice.getResult(),
        convertOp.getSrcLayoutAttr(),
        convertOp.getDstLayoutAttr(), extractSliceOp.getMixedSizes());

    // Replace original extract_slice
    rewriter.replaceOp(extractSliceOp, newConvert.getResult());

    return success();
  }
};
}

void mlir::hivm::populateConvertLayoutExtractSlice(RewritePatternSet &patterns,
                                                   MLIRContext *context) {
  patterns.add<PropagateConvertLayoutDownThroughExtractSlice>(context);
}