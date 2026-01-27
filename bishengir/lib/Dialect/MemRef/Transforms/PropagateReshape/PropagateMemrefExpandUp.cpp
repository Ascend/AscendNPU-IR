//===- PropagateMemrefExpandUp.cpp ----------------------------------------===//
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
//
//  Propagate expand up will try to bubble up the expandshape operation to the
//  top
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "bishengir/Dialect/MemRef/Transforms/PropagateReshape.h"

#define DEBUG_TYPE "propagate-memref-reshape"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#include "bishengir/Dialect/Utils/Util.h"
using namespace mlir::utils::debugger;

namespace mlir {
namespace memref {
using namespace mlir::utils::debugger;

namespace {

LogicalResult handleAllocOp(memref::ExpandShapeOp expandOp,
                            PatternRewriter &rewriter, Operation *definingOp) {
  rewriter.setInsertionPointAfter(definingOp);
  SmallVector<Value, 4> newOperands;
  auto allocOp = cast<memref::AllocOp>(definingOp);
  auto reassociation = expandOp.getReassociation();
  auto collapsedRes = rewriter.create<memref::CollapseShapeOp>(
      expandOp->getLoc(), definingOp->getResults()[0].getType(),
      definingOp->getResults()[0], reassociation);
  rewriter.replaceAllUsesExcept(definingOp->getResults()[0],
                                collapsedRes.getResult(), collapsedRes);
  rewriter.modifyOpInPlace(definingOp, [&]() {
    definingOp->getResult(0).setType(expandOp.getResultType());
  });
  rewriter.replaceAllUsesWith(expandOp.getResult(), allocOp);
  return success();
}

LogicalResult computeStridesFromLayout(PatternRewriter &rewriter,
                                       memref::ExpandShapeOp expandOp,
                                       SmallVector<OpFoldResult> &stridesOfr) {
  auto expandResultRank = expandOp.getResult().getType().getRank();
  auto reassociation = expandOp.getReassociationIndices();
  stridesOfr.reserve(expandResultRank);

  auto stridedExpandLayout =
      dyn_cast<StridedLayoutAttr>(expandOp.getResult().getType().getLayout());
  if (!stridedExpandLayout)
    return failure();

  auto expandStrides = stridedExpandLayout.getStrides();
  for (auto &group : reassociation) {
    for (auto &el : group) {
      auto &curStride = expandStrides[el];
      if (!ShapedType::isDynamic(curStride)) {
        stridesOfr.push_back(
            getAsIndexOpFoldResult(rewriter.getContext(), curStride));
      } else {
        return rewriter.notifyMatchFailure(expandOp, "Has dynamic strides");
      }
    }
  }
  return success();
}

LogicalResult computeStridesFromShape(memref::ExpandShapeOp expandOp,
                                      PatternRewriter &rewriter,
                                      SmallVector<OpFoldResult> &stridesOfr) {
  auto expandSrcShape = utils::getShape(expandOp.getSrc().getType());
  auto staticOutputShape = expandOp.getStaticOutputShape();
  auto expandResultRank = expandOp.getResult().getType().getRank();
  auto total = utils::getStaticTotalSize(expandSrcShape);
  if (!total.has_value())
    return failure();
  auto totalInt = total.value();
  stridesOfr.reserve(expandResultRank);
  for (int i = 0; i < expandResultRank; i++) {
    if (ShapedType::isDynamic(staticOutputShape[i]))
      return failure();
    totalInt /= staticOutputShape[i];
    stridesOfr.push_back(rewriter.getIndexAttr(totalInt));
  }
  return success();
}

LogicalResult computeStrides(memref::ExpandShapeOp expandOp,
                             PatternRewriter &rewriter,
                             SmallVector<OpFoldResult> &stridesOfr) {
  // Try layout-based computation first
  if (succeeded(computeStridesFromLayout(rewriter, expandOp, stridesOfr))) {
    return success();
  }
  // Fall back to shape-based computation
  return computeStridesFromShape(expandOp, rewriter, stridesOfr);
}

LogicalResult handleReinterpretCast(memref::ExpandShapeOp expandOp,
                                    PatternRewriter &rewriter,
                                    Operation *definingOp) {
  auto reinterpretCast = cast<memref::ReinterpretCastOp>(definingOp);
  if (ShapedType::isDynamicShape(expandOp.getSrc().getType().getShape())) {
    return failure();
  }
  SmallVector<OpFoldResult> stridesOfr;
  if (failed(computeStrides(expandOp, rewriter, stridesOfr))) {
    return failure();
  }
  assert(stridesOfr.size() == expandOp.getResult().getType().getRank());
  SmallVector<OpFoldResult> offsetsOfr = reinterpretCast.getMixedOffsets();
  SmallVector<OpFoldResult> sizesOfr = getMixedValues(
      expandOp.getStaticOutputShape(), expandOp.getOutputShape(), rewriter);
  rewriter.setInsertionPointAfter(expandOp);
  auto newReinterpret = rewriter.create<memref::ReinterpretCastOp>(
      reinterpretCast->getLoc(), expandOp.getResultType(),
      reinterpretCast.getSource(), offsetsOfr, sizesOfr, stridesOfr);
  auto reassociation = expandOp.getReassociationIndices();
  rewriter.replaceOp(expandOp, newReinterpret);
  auto newCollapse = rewriter.create<memref::CollapseShapeOp>(
      reinterpretCast.getLoc(), reinterpretCast.getResult().getType(),
      newReinterpret.getResult(), reassociation);
  rewriter.replaceOp(reinterpretCast, newCollapse);
  return success();
}

LogicalResult handleSubView(memref::ExpandShapeOp expandOp,
                            PatternRewriter &rewriter, Operation *definingOp) {
  auto subviewOp = cast<memref::SubViewOp>(definingOp);
  auto offsets = subviewOp.getMixedOffsets();
  auto sizes = subviewOp.getMixedSizes();
  auto strides = subviewOp.getMixedStrides();
  SmallVector<OpFoldResult> newOffsets;
  SmallVector<OpFoldResult> newSizes;
  SmallVector<OpFoldResult> newStrides;
  auto inputShape = subviewOp.getSourceType().getShape();
  auto targetShape = expandOp.getStaticOutputShape();
  SmallVector<int64_t> newShape;
  auto reassociation = expandOp.getReassociationIndices();
  // only handle the [1, d] -> [d] case
  for (auto [i, indices] : llvm::enumerate(reassociation)) {
    bool isHandled = false;
    for (auto idx : indices) {
      if (targetShape[idx] != 1) {
        if (isHandled) {
          // not trivial conversion
          return failure();
        }
        isHandled = true;
        newOffsets.push_back(offsets[i]);
        newSizes.push_back(sizes[i]);
        newStrides.push_back(strides[i]);
        newShape.push_back(inputShape[i]);
      } else {
        newOffsets.push_back(rewriter.getIndexAttr(0));
        newSizes.push_back(rewriter.getIndexAttr(1));
        newStrides.push_back(rewriter.getIndexAttr(1));
        newShape.push_back(1);
      }
    }
    if (!isHandled) {
      newOffsets.back() = offsets[i];
      newSizes.back() = sizes[i];
      newStrides.back() = strides[i];
      newShape.back() = inputShape[i];
    }
  }
  rewriter.setInsertionPoint(expandOp);
  auto newExpand = rewriter.create<memref::ExpandShapeOp>(expandOp.getLoc(), newShape, subviewOp.getSource(), reassociation);
  auto newSubView = rewriter.create<memref::SubViewOp>(
      subviewOp.getLoc(), newExpand, newOffsets, newSizes,
      newStrides);
  rewriter.replaceOp(expandOp, newSubView);
  return success();
}
} // namespace

LogicalResult
PropagateMemrefExpandUp::matchAndRewrite(memref::ExpandShapeOp expandOp,
                                         PatternRewriter &rewriter) const {
  Value source = expandOp.getSrc();
  Operation *definingOp = source.getDefiningOp();
  if (!definingOp)
    return failure();
  if (definingOp->getParentOp() != expandOp->getParentOp())
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "-- Found definingOp: " << *definingOp << "\n";);
  LLVM_DEBUG(llvm::dbgs() << "Ok rewriting\n";);
  LLVM_DEBUG(llvm::dbgs() << *definingOp->getParentOp() << "\n";);
  if (isa<memref::AllocOp>(definingOp)) {
    LDBG("Ok in here");
    return handleAllocOp(expandOp, rewriter, definingOp);
  }
  if (isa<memref::ReinterpretCastOp>(definingOp)) {
    LDBG("Ok in here");
    return handleReinterpretCast(expandOp, rewriter, definingOp);
  }
  if (isa<memref::SubViewOp>(definingOp)) {
    LDBG("Ok in here");
    return handleSubView(expandOp, rewriter, definingOp);
  }
  return failure();
}
} // namespace memref
} // namespace mlir