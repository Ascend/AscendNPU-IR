//===-------------------- HoistConvertLayout.cpp --------------------------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "hivm-hoist-convert-layout"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::hivm {
//===----------------------------------------------------------------------===//
// Move ConvertLayout Adjacent to Source Definition
//===----------------------------------------------------------------------===//

/// Pattern to move convert_layout operations right after their source
/// definition. This improves code locality and can enable other
/// canonicalization patterns.
struct MoveConvertLayoutToSourcePattern
    : public OpRewritePattern<ConvertLayoutOp> {
  MoveConvertLayoutToSourcePattern(MLIRContext *context, bool stopMoveAtScope)
      : OpRewritePattern(context, /*benefit=*/1),
        stopMoveAtScope(stopMoveAtScope) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getSource();
    Operation *sourceOp = source.getDefiningOp();
    Block *currentBlock = op->getBlock();

    Block *headDstBlock = nullptr;

    if (!sourceOp) {
      // Source is from a block argument
      auto blockArg = cast<BlockArgument>(source);
      Block *ownerBlock = blockArg.getOwner();
      headDstBlock = computeCappedTargetBlock(currentBlock, ownerBlock);
      return moveToBlockHead(op, rewriter, headDstBlock);
    }
    Block *sourceBlock = sourceOp->getBlock();
    Block *dstBlock = computeCappedTargetBlock(currentBlock, sourceBlock);

    if (dstBlock != sourceBlock) {
      headDstBlock = dstBlock; // scope-capped case: place at block head
      return moveToBlockHead(op, rewriter, headDstBlock);
    }

    Operation *beforeOp = sourceOp; // normal case: place after source
    if (isAlreadyAdjacentToSource(op, beforeOp))
      return rewriter.notifyMatchFailure(op, "already adjacent to source");

    LDBG("Moving convert_layout after source: " << *beforeOp);
    rewriter.modifyOpInPlace(op, [&]() { op->moveAfter(beforeOp); });
    return success();
  }

private:
  bool stopMoveAtScope;
  static bool isAlreadyAdjacentToSource(ConvertLayoutOp op,
                                        Operation *beforeOp) {
    Value source = op.getSource();
    Operation *previousNode = op->getPrevNode();

    bool validConvertLayoutPosition = previousNode == beforeOp;

    if (previousNode) {
      if (auto prevConvert = dyn_cast<ConvertLayoutOp>(previousNode))
        validConvertLayoutPosition |=
            (prevConvert.getSource().getDefiningOp() == source.getDefiningOp());
    }

    return validConvertLayoutPosition;
  }

  static LogicalResult moveToBlockHead(ConvertLayoutOp op,
                                       PatternRewriter &rewriter,
                                       Block *dstBlock) {
    if (op->getBlock() == dstBlock)
      return rewriter.notifyMatchFailure(
          op, "operation already at destination block");
    rewriter.modifyOpInPlace(
        op, [&]() { op->moveBefore(dstBlock, dstBlock->begin()); });
    return success();
  }

  /// Returns the earliest block we can move to while walking from `from` up to
  /// `target`. If `stopMoveAtScope` is enabled and a scope boundary is hit
  /// first, we stop at that scope's region entry block.
  Block *computeCappedTargetBlock(Block *from, Block *target) const {
    if (!stopMoveAtScope)
      return target;
    Block *cursor = from;
    while (cursor && cursor != target) {
      Region *parentRegion = cursor->getParent();
      if (!parentRegion)
        break;
      Operation *parentOp = parentRegion->getParentOp();
      if (!parentOp)
        break;
      if (isScopeBoundaryOp(parentOp)) {
        return &parentRegion->front();
      }
      cursor = parentOp->getBlock();
    }
    return target;
  }

  static bool isScopeBoundaryOp(Operation *op) {
    return isa_and_present<scope::ScopeOp>(op);
  }
};

void populateHoistConvertLayout(RewritePatternSet &patterns,
                                MLIRContext *context) {
  patterns.add<MoveConvertLayoutToSourcePattern>(context, true);
}
} // namespace mlir::hivm