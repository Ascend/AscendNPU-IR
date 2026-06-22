//===-------------------- PropagateConvertLayoutElementwise.cpp -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include <llvm/Frontend/OpenMP/OMPConstants.h>

#define DEBUG_TYPE "hivm-propagate-convert-layout"

using namespace mlir;
using namespace mlir::hivm;

namespace {
//===----------------------------------------------------------------------===//
// Propagate UP through Elementwise Operations
//===----------------------------------------------------------------------===//

bool checkIsAgnostic(Operation *op,
                     const PropagateConvertLayoutInternalOptions &options) {
  if (!op) return false;
  if (options.allowAgnosticOps)
    return isLayoutAgnosticOp(op);
  return isa<hivm::VCastOp>(op);
}

/// Pattern: Push convert_layout UP through elementwise operations
/// Before:
///   %b = hivm.pointwise %a
///   %t1 = hivm.hir.convert_layout %b {up}
/// After:
///   %new_a = hivm.hir.convert_layout %a {up}
///   %new_b = hivm.pointwise %new_a
struct PropagateConvertLayoutUpThroughElementwise
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  PropagateConvertLayoutInternalOptions options;
  PropagateConvertLayoutUpThroughElementwise(MLIRContext *context,
                       const PropagateConvertLayoutInternalOptions &options = {},
                       PatternBenefit benefit = 1)
      : OpRewritePattern<ConvertLayoutOp>(context, benefit),
        options(std::move(options)) {}
  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (!isPropagatingUp(convertOp))
      return failure();

    Value sourceOperand = convertOp.getSource();
    Operation *definingOp = sourceOperand.getDefiningOp();

    if (!checkIsAgnostic(definingOp, options))
      return rewriter.notifyMatchFailure(
          convertOp, "defining op is not layout-agnostic");

    // Clone the defining op
    rewriter.setInsertionPointAfter(definingOp);

    Operation *newOp = rewriter.clone(*definingOp);

    Value targetOperand = convertOp.getResult();
    auto sourceShape = cast<ShapedType>(sourceOperand.getType()).getShape();
    // Convert DPS inputs
    rewriter.setInsertionPoint(newOp);
    for (OpOperand &opr : newOp->getOpOperands()) {
      // Only convert if ranks match (handle scalar operands)
      auto oprType = dyn_cast<ShapedType>(opr.get().getType());
      if (!oprType)
        continue;
      if (oprType.getShape() != sourceShape)
        continue;
      auto convertedOpr = createConvertLayoutLike(
          rewriter, convertOp, opr.get());
      opr.assign(convertedOpr);
    }

    // Update result type to match converted layout
    auto newResultType = cast<ShapedType>(targetOperand.getType()).clone(
        getElementTypeOrSelf(newOp->getResult(0)));
    newOp->getResult(0).setType(newResultType);
    rewriter.setInsertionPointAfter(newOp);
    // Replace convert_layout with new op's result
    rewriter.replaceAllUsesWith(convertOp.getResult(), newOp->getResult(0));
    auto convertBackOp = createInverseConvertLayout(
          rewriter, convertOp, newOp->getResult(0));
    rewriter.eraseOp(convertOp);
    rewriter.replaceOp(definingOp, convertBackOp.getDefiningOp());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Propagate DOWN through Elementwise Operations (NOT through fixpipe)
//===----------------------------------------------------------------------===//

/// Pattern: Push convert_layout DOWN through elementwise operations
/// Before:
///   %t1 = hivm.hir.convert_layout %t0 {down}
///   %b = hivm.pointwise %t1
/// After:
///   %new_b = hivm.pointwise %t0
///   %result = hivm.hir.convert_layout %new_b {down}
struct PropagateConvertLayoutDownThroughElementwise
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;
  PropagateConvertLayoutInternalOptions options;
  PropagateConvertLayoutDownThroughElementwise(MLIRContext *context,
                       const PropagateConvertLayoutInternalOptions &options = {},
                       PatternBenefit benefit = 1)
      : OpRewritePattern<ConvertLayoutOp>(context, benefit),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (!isPropagatingDown(convertOp))
      return failure();

    if (convertOp->use_empty())
      return rewriter.notifyMatchFailure(
          convertOp, "convert has no use");

    auto findIt = llvm::find_if(convertOp->getUsers(), [this](Operation *user) {
      return checkIsAgnostic(user, options);
    });

    if (findIt == convertOp->getUsers().end())
      return rewriter.notifyMatchFailure(
          convertOp, "elementwise operation is not found");
    auto userOp = *findIt;
    Value targetOperand = convertOp.getResult();
    auto targetShape = cast<ShapedType>(targetOperand.getType()).getShape();
    rewriter.setInsertionPoint(userOp);
    Operation *newOp = rewriter.clone(*userOp);
    rewriter.setInsertionPoint(newOp);
    for (OpOperand &opr : newOp->getOpOperands()) {
      // Only convert if ranks match (handle scalar operands)
      auto oprType = dyn_cast<ShapedType>(opr.get().getType());
      if (!oprType)
        continue;
      if (oprType.getShape() != targetShape)
        continue;
      auto convertedOpr = createInverseConvertLayout(
          rewriter, convertOp, opr.get());
      opr.assign(convertedOpr);
    }
    Value sourceOperand = convertOp.getSource();

    // Update result type to match source layout
    auto newResultType = cast<ShapedType>(sourceOperand.getType()).clone(
        getElementTypeOrSelf(newOp->getResult(0)));
    newOp->getResult(0).setType(newResultType);
    // Insert conversion after the new op
    rewriter.setInsertionPointAfter(newOp);
    Value resultConvert = createConvertLayoutLike(
        rewriter, convertOp,newOp->getResult(0));
    rewriter.replaceOp(userOp, resultConvert.getDefiningOp());
    return success();
  }
};
} // namespace

void mlir::hivm::populateConvertLayoutElementwise(RewritePatternSet &patterns,
                                                  MLIRContext *context,
                                                  const
                                                  PropagateConvertLayoutInternalOptions
                                                  &options) {
  patterns.add<
    PropagateConvertLayoutUpThroughElementwise,
    PropagateConvertLayoutDownThroughElementwise
  >(context, options);
}