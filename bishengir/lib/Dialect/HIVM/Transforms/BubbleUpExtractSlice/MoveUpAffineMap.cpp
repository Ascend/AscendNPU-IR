//===- MoveUpAffineMap.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/MoveUpAffineMap.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Transforms/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AsmState.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::hivm::detail {

// We need this move up affine map patterns because
// when we bubble up extract slice, extract slice might get bubbled
// higher than it's map.
// For example:
// for {
//   xxx
//   xxx
//   %0 = affine.apply
//   extract_slice [%0]
// }
// ->
// for {
//   extract_slice [%0]
//   xxx
//   xxx
//   %0 = affine.apply
// }
// so we need to move up the affine apply too
// ->
// for {
//   %0 = affine.apply
//   extract_slice [%0]
//   xxx
//   xxx
// }
struct MoveUpAffineMapPattern : public OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;

  explicit MoveUpAffineMapPattern(MLIRContext *ctx, PatternBenefit benefit = 100)
      : OpRewritePattern<affine::AffineApplyOp>(ctx, benefit){};

  LogicalResult matchAndRewrite(affine::AffineApplyOp affineApplyOp,
                                PatternRewriter &rewriter) const final {
    if (!affineApplyOp->getPrevNode() ||
        isa<affine::AffineApplyOp>(affineApplyOp->getPrevNode())) {
      return rewriter.notifyMatchFailure(affineApplyOp,
                                         "previous node doesn't exist");
    }

    // If this affine map is only using block arguments as operand
    // move it to the start of the block.
    bool allArgsArguments =
        llvm::all_of(affineApplyOp->getOperands(),
                     [](const Value opr) { return isa<BlockArgument>(opr); });
    if (!allArgsArguments) {
      return rewriter.notifyMatchFailure(affineApplyOp,
                                         "not all operands are blockArguments");
    }
    auto applyParentBlock = affineApplyOp->getBlock();
    rewriter.moveOpBefore(affineApplyOp, applyParentBlock,
                          applyParentBlock->begin());
    auto *newOp = rewriter.clone(*affineApplyOp);
    rewriter.replaceOp(affineApplyOp, newOp);
    return success();
  }
};

void populateMoveUpAffineMapPattern(RewritePatternSet &patterns) {
  patterns.add<MoveUpAffineMapPattern>(patterns.getContext());
}
} // namespace mlir::hivm::detail