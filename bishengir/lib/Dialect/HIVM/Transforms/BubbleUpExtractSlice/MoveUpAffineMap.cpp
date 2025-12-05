//===- MoveUpAffineMap.cpp ------------------------------------------------===//
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
