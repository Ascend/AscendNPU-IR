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
#include "mlir/Dialect/Arith/Utils/Utils.h"
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

static OpFoldResult multiplyOFR(PatternRewriter &rewriter, Location loc,
                                OpFoldResult a, OpFoldResult b) {
  auto aConstant = getConstantIntValue(a);
  auto bConstant = getConstantIntValue(b);

  if (aConstant && bConstant)
    return rewriter.getIndexAttr(*aConstant * *bConstant);

  Value aVal = getValueOrCreateConstantIndexOp(rewriter, loc, a);
  Value bVal = getValueOrCreateConstantIndexOp(rewriter, loc, b);
  return rewriter.create<arith::MulIOp>(loc, aVal, bVal).getResult();
}

LogicalResult handleReinterpretCast(memref::ExpandShapeOp expandOp,
                                    PatternRewriter &rewriter,
                                    Operation *definingOp) {
  auto reinterpretCast = cast<memref::ReinterpretCastOp>(definingOp);
  auto expandResType = expandOp.getResult().getType().cast<MemRefType>();

  auto reassociation = expandOp.getReassociationIndices();

  SmallVector<OpFoldResult> offsetOfr = reinterpretCast.getMixedOffsets();
  SmallVector<OpFoldResult> oldStrides = reinterpretCast.getMixedStrides();
  SmallVector<OpFoldResult> sizesOfr = getMixedValues(
      expandOp.getStaticOutputShape(), expandOp.getOutputShape(), rewriter);

  SmallVector<OpFoldResult> newStridesOfr;

  rewriter.setInsertionPoint(reinterpretCast);

  for (auto [idx, group] : llvm::enumerate(reassociation)) {
    OpFoldResult currentStride = oldStrides[idx];
    SmallVector<OpFoldResult> groupStrides;

    for (size_t i = group.size(); i > 0; --i) {
      groupStrides.push_back(currentStride);
      if (i > 1) {
        currentStride = multiplyOFR(rewriter, reinterpretCast.getLoc(),
                                    currentStride, sizesOfr[group[i - 1]]);
      }
    }
    std::reverse(groupStrides.begin(), groupStrides.end());
    newStridesOfr.append(groupStrides.begin(), groupStrides.end());
  }

  expandOp->moveAfter(reinterpretCast);
  rewriter.setInsertionPointAfterValue(expandOp);

  auto newReinterpret = rewriter.create<memref::ReinterpretCastOp>(
      reinterpretCast->getLoc(), expandResType, reinterpretCast.getSource(),
      offsetOfr, sizesOfr, newStridesOfr);

  auto newCollapse = rewriter.create<memref::CollapseShapeOp>(
      reinterpretCast->getLoc(), reinterpretCast.getResult().getType(),
      newReinterpret, reassociation);

  rewriter.replaceAllUsesExcept(reinterpretCast, newCollapse, expandOp);
  rewriter.replaceOp(expandOp, newReinterpret);

  LDBG(*definingOp->getParentOp());

  rewriter.eraseOp(reinterpretCast);

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
  auto newExpand = rewriter.create<memref::ExpandShapeOp>(
      expandOp.getLoc(), newShape, subviewOp.getSource(), reassociation);
  auto newSubView = rewriter.create<memref::SubViewOp>(
      subviewOp.getLoc(), newExpand, newOffsets, newSizes, newStrides);
  rewriter.replaceOp(expandOp, newSubView);
  return success();
}

// whether expand shape dims is all 1
// eg: expand_shape<2x3> [[0][1, 2, 3]] -> <2, 1, 3, 1> true
// eg: expand_shape<2x?> [[0][1, 2, 3]] -> <2, 1, ?, 1> true
// eg: expand_shape<2x4> [[0][1, 2, 3]] -> <2, 1, 2, 2> false
bool isExpandShapeAllOne(memref::ExpandShapeOp expandOp) {
  auto targetShape = expandOp.getStaticOutputShape();
  auto reassociation = expandOp.getReassociationIndices();
  for (auto &indices : reassociation) {
    // not expand dim: continue
    if (indices.size() <= 1) {
      continue;
    }
    // expand dim: get number of none one
    int nonOneCount = 0;
    for (auto idx : indices) {
      if (targetShape[idx] != 1) {
        nonOneCount++;
      }
      if (nonOneCount > 1) {
        return false;
      }
    }
  }
  return true;
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
    auto dstType = expandOp.getResult().getType();
    auto dstMemrefType = dyn_cast<MemRefType>(dstType);
    // expand shape dims is all 1, not propagate alloc op (only when dst rank >
    // 3).
    // TODO: remove this after the logic of HIVMInferDataLayout pass is
    // refactored.
    if (isExpandShapeAllOne(expandOp) && dstMemrefType &&
        dstMemrefType.getRank() > 3) {
      return failure();
    }
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