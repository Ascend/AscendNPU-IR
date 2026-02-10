//===- CSEPattern.cpp ------------------------------------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/CSEPattern.h"
#include "bishengir/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "common-pattern-bubble-up-extract-slice"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

namespace mlir::hivm::detail {
static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
  auto *lhs = cast<Operation *>(lhsC);
  auto *rhs = cast<Operation *>(rhsC);
  return OperationEquivalence::isEquivalentTo(
      lhs, rhs, OperationEquivalence::IgnoreLocations);
}

struct CSEExtractSlicePattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp pivotSliceOp,
                                PatternRewriter &rewriter) const final {
    // Get the source value
    Value source = pivotSliceOp.getSource();
    if (!source)
      return failure();

    // Collect all users of the source that are ExtractSliceOps
    SmallVector<tensor::ExtractSliceOp> siblingSlices;
    for (Operation *user : source.getUsers()) {
      if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
        if (sliceOp.getSource() == source &&
            isEqual(user, pivotSliceOp.getOperation())) {
          siblingSlices.push_back(sliceOp);
        }
      }
    }

    if (siblingSlices.size() == 1)
      return rewriter.notifyMatchFailure(
          pivotSliceOp, "Slice doesn't have any sibling duplicate");

    DominanceInfo domInfo(pivotSliceOp->getParentOfType<func::FuncOp>());
    auto pivot = source;

    for (auto opr : pivotSliceOp->getOperands()) {
      if (pivot == opr)
        continue;
      auto pivotDefOp = pivot.getDefiningOp();
      auto oprDefOp = opr.getDefiningOp();
      if (pivotDefOp && oprDefOp) {
        if (domInfo.properlyDominates(pivotDefOp, oprDefOp))
          pivot = opr;
      } else if (pivot.getParentRegion()->isProperAncestor(opr.getParentRegion()) ||
                 (pivot.getParentRegion() == opr.getParentRegion() && oprDefOp)) {
        pivot = opr;
      }
    }

    rewriter.setInsertionPointAfterValue(pivot);
    auto newOp = rewriter.clone(*pivotSliceOp.getOperation());
    for (auto siblingSlice : siblingSlices)
      rewriter.replaceOp(siblingSlice, newOp);
    return success();
  }
};

struct CSEAffineApplyPattern : public OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp baseOp,
                                PatternRewriter &rewriter) const final {
    Block *blockParent = baseOp->getBlock();
    SmallVector<affine::AffineApplyOp> siblingGroup;
    blockParent->walk([&](affine::AffineApplyOp applyOp) {
      if (isEqual(applyOp, baseOp))
        siblingGroup.push_back(applyOp);
    });
    if (siblingGroup.size() == 1)
      return failure();
    for (size_t i = 1; i < siblingGroup.size(); ++i) {
      rewriter.replaceAllUsesWith(siblingGroup[i], siblingGroup[0]);
    }
    return success();
  }
};

void populateCSEPattern(RewritePatternSet &patterns) {
  patterns.add<CSEExtractSlicePattern>(patterns.getContext());
  patterns.add<CSEAffineApplyPattern>(patterns.getContext());
}

} // namespace mlir::hivm::detail
