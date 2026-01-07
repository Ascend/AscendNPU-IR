//===- HoistAffine.cpp ------------------------------------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/HoistAffine.h"
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
template <typename AffineOpTy>
struct HoistAffinePattern : public OpRewritePattern<AffineOpTy> {
  using OpRewritePattern<AffineOpTy>::OpRewritePattern;

  explicit HoistAffinePattern(MLIRContext *ctx, PatternBenefit benefit = 100)
      : OpRewritePattern<AffineOpTy>(ctx, benefit){};

  LogicalResult matchAndRewrite(AffineOpTy op,
                                PatternRewriter &rewriter) const final {
    // Get the lowest dominating operation and block argument.
    Operation *lastDefOp = nullptr;
    Value lastDefVal = Value();
    Value lastBlockArg = Value();
    DominanceInfo dominance;
    for (Value operand : op->getOperands()) {
      if (auto ba = dyn_cast<BlockArgument>(operand)) {
        if (!lastBlockArg) {
          lastBlockArg = ba;
        } else {
          Block *currentBlock = ba.getParentBlock();
          Block *lastBlock = lastBlockArg.getParentBlock();
          
          if (dominance.dominates(lastBlock, currentBlock)) {
            lastBlockArg = ba;
          }
        }
        continue;
      }

      auto *definingOp = operand.getDefiningOp();
      if (!definingOp)
        continue;
      if (!lastDefOp || dominance.dominates(lastDefOp, definingOp)) {
        lastDefOp = definingOp;
        lastDefVal = operand;
      }
    }

    Operation *insertPoint = nullptr;

    // If 'lastDefOp' is null, it means that the lowest dominating value is
    // a block argument. So we use 'lastBlockArg' as 'lastDefVal' 
    // and set the insertion point to the front of the block.
    if (!lastDefOp && lastBlockArg) {
      lastDefVal = lastBlockArg;
      Block *anchorBlock = lastBlockArg.getParentBlock();
      insertPoint = &anchorBlock->front();
    }

    // If the 'lastDefOp' is null, after updating 'lastDefVal' with 'lastBlockArg',
    // if 'lastDefVal' is still null, it is abnormal.
    if (!lastDefOp && !lastDefVal)
      return rewriter.notifyMatchFailure(
          op, "no valid operands found");

    // If we have 'lastDefOp' and 'lastBlockArg' is null, 
    // it means all operands have defining Op,
    // we can directly use 'lastDefOp' and 'lastDefVal'
    // and set the insertion point to the next node of 'lastDefOp'.
    if (lastDefOp && !lastBlockArg) {
      insertPoint = lastDefOp->getNextNode();
    }

    // If we have both 'lastDefOp' and 'lastBlockArg', we might either 
    // set the insertion point to the next node of 'lastDefOp' or to the front of 'argBlock',
    // so we need to verify the dominance relationship between 'defOpBlock' and 'argBlock'
    // and might update 'lastDefVal' with 'lastBlockArg'.
    if (lastDefOp && lastBlockArg) {
      Block *defOpBlock = lastDefOp->getBlock();
      Block *argBlock = lastBlockArg.getParentBlock();
      insertPoint = dominance.dominates(argBlock, defOpBlock) ? lastDefOp->getNextNode() : &argBlock->front();
      lastDefVal = dominance.dominates(argBlock, defOpBlock) ? lastDefVal : lastBlockArg;
    }

    if (insertPoint->getBlock() != op->getBlock()) {
      rewriter.moveOpBefore(op, insertPoint);
      return success();
    }

    // Adjust insertion point down to avoid oscillation.
    // For example:
    //
    // ```mlir
    // %def = some_op
    // first_use(%def)
    // second_use(%def)
    // ...
    // third_use(%def)
    // ```
    // Potential problem: If we matched the "second_use", the `insertPoint` 
    // will be "first_use", and vice versa. Because there is no domination 
    // relationship between the two.
    // We can break the tie by moving the insertion point to end of the
    // consecutive chain of users before current op.
    auto lastDefValUser = SetVector<Operation *>{lastDefVal.getUsers().begin(),
                                                 lastDefVal.getUsers().end()};
    
    while (insertPoint && lastDefValUser.contains(insertPoint))
      insertPoint = insertPoint->getNextNode();

    if (!insertPoint || insertPoint->getBlock() != op->getBlock())
      return rewriter.notifyMatchFailure(op, "invalid insertion point");

    if (insertPoint == op || op->isBeforeInBlock(insertPoint))
      return rewriter.notifyMatchFailure(
          op, "op cannot be moved to a higher place in block");

    rewriter.moveOpBefore(op, insertPoint);
    return success();
  }
};

void populateHoistAffinePattern(RewritePatternSet &patterns) {
  patterns
      .add<HoistAffinePattern<affine::AffineApplyOp>, 
           HoistAffinePattern<affine::AffineMinOp>,
           HoistAffinePattern<affine::AffineMaxOp>>(patterns.getContext());
}
} // namespace mlir::hivm::detail