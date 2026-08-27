//===- FoldCollapseIntoAllocWithLoadPattern.cpp ----------------------------------------===//
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

#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/FoldCollapseIntoAllocWithLoadPattern.h"

#include "bishengir/Dialect/Tensor/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "fold-collapse-into-alloc-with-load"

namespace mlir {
namespace tensor {

#define GEN_PASS_DEF_FOLDCOLLAPSEINTOALLOCWITHLOAD
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"

// True when the collapse only removes unit (size-1) dims: every reassociation
// group merges at most one non-unit (or dynamic) source dim and at least one
// group merges a unit dim away. A genuine reshape (e.g. batch flattening)
// merges >= 2 non-unit dims in some group and returns false.
static bool removesOnlyUnitDims(MemRefType srcType,
                                ArrayRef<ReassociationIndices> reassociation) {
  bool removedAnyUnitDim = false;
  for (const auto &group : reassociation) {
    unsigned nonUnitDims = 0;
    for (int64_t dim : group) {
      if (srcType.isDynamicDim(dim) || srcType.getDimSize(dim) != 1)
        ++nonUnitDims;
      else if (group.size() > 1)
        removedAnyUnitDim = true;
    }
    if (nonUnitDims > 1)
      return false;
  }
  return removedAnyUnitDim;
}

// Reassociation that merges every static unit dim into its left neighbor,
// e.g. [d0, 1, d2] -> [[0, 1], [2]].
static SmallVector<ReassociationIndices>
unitDimReassociation(ArrayRef<int64_t> shape) {
  SmallVector<ReassociationIndices> groups;
  ReassociationIndices current;
  for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
    if (shape[i] == 1 && !current.empty()) {
      current.push_back(i);
      continue;
    }
    if (!current.empty())
      groups.push_back(current);
    current = {i};
  }
  if (!current.empty())
    groups.push_back(current);
  return groups;
}

// Drop the given positions from a mixed offset/size/stride list.
static SmallVector<OpFoldResult> dropDims(ArrayRef<OpFoldResult> values,
                                          ArrayRef<int64_t> dims) {
  SmallVector<OpFoldResult> result;
  for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i)
    if (!llvm::is_contained(dims, i))
      result.push_back(values[i]);
  return result;
}

LogicalResult
FoldCollapseIntoAllocWithLoadPattern::matchAndRewrite(hivm::LoadOp loadOp,
                                         PatternRewriter &rewriter) const {
  Value src = loadOp->getOperand(0);
  Value dst = loadOp->getOperand(1);
  auto srcType = dyn_cast<MemRefType>(src.getType());
  auto dstType = dyn_cast<MemRefType>(dst.getType());
  if (!srcType || !dstType || srcType.getRank() != dstType.getRank() ||
      dstType.getRank() < 2)
    return rewriter.notifyMatchFailure(loadOp, "rank mismatch or not memref");

  auto subview = dst.getDefiningOp<memref::SubViewOp>();
  if (!subview)
    return rewriter.notifyMatchFailure(loadOp, "dst is not a subview");
  auto alloc = subview.getViewSource().getDefiningOp<memref::AllocOp>();
  if (!alloc)
    return rewriter.notifyMatchFailure(loadOp, "dst is not a subview of alloc");
  auto allocType = cast<MemRefType>(alloc.getType());
  if (!allocType.hasStaticShape() || !allocType.getLayout().isIdentity())
    return rewriter.notifyMatchFailure(loadOp,
                                       "alloc is dynamic or non-identity");

  ArrayRef<int64_t> shape = allocType.getShape();
  SmallVector<int64_t> unitDims;
  for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i)
    if (shape[i] == 1)
      unitDims.push_back(i);
  if (unitDims.empty())
    return rewriter.notifyMatchFailure(loadOp, "alloc has no static unit dim");

  // Vet every use of the alloc: subviews and one unit-dim collapse. The
  // collapse is the proof that consumers want the plain matrix form; the
  // unit-dim restriction is only the fold's mechanical capability (the
  // subview rebuild drops folded dims, which is address-exact only for
  // size-1 dims).
  SmallVector<memref::SubViewOp> subviews;
  memref::CollapseShapeOp scaffoldCollapse;
  for (OpOperand &use : alloc->getUses()) {
    Operation *owner = use.getOwner();
    if (auto sv = dyn_cast<memref::SubViewOp>(owner)) {
      subviews.push_back(sv);
      continue;
    }
    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(owner);
        collapse && removesOnlyUnitDims(cast<MemRefType>(collapse.getSrcType()),
                                        collapse.getReassociationIndices())) {
      scaffoldCollapse = collapse;
      continue;
    }
    return rewriter.notifyMatchFailure(loadOp, "alloc has unknown users");
  }
  if (!scaffoldCollapse)
    return rewriter.notifyMatchFailure(loadOp,
                                       "no unit-dim collapse consumer (maybe "
                                       "a genuine batch load)");
  for (memref::SubViewOp sv : subviews)
    for (OpOperand &use : sv->getUses())
      if (!isa<hivm::LoadOp>(use.getOwner()))
        return rewriter.notifyMatchFailure(loadOp,
                                           "subview has non-load users");

  Location loc = loadOp.getLoc();

  // 1. Recreate the alloc without the unit dims.
  SmallVector<int64_t> newShape;
  for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i)
    if (!llvm::is_contained(unitDims, i))
      newShape.push_back(shape[i]);
  // Capability check: the copy's src must collapse (by its own unit dims)
  // to the same rank as the rebuilt destination.
  if (static_cast<int64_t>(unitDimReassociation(srcType.getShape()).size()) !=
      static_cast<int64_t>(newShape.size()))
    return rewriter.notifyMatchFailure(loadOp,
                                       "copy src is not unit-collapsible to "
                                       "the folded rank");
  auto newAllocType = MemRefType::get(newShape, allocType.getElementType(),
                                      MemRefLayoutAttrInterface(),
                                      allocType.getMemorySpace());
  rewriter.setInsertionPoint(alloc);
  auto newAlloc = rewriter.create<memref::AllocOp>(
      loc, newAllocType, alloc.getDynamicSizes(), ValueRange{},
      alloc.getAlignmentAttr());

  // 2. Rebuild each subview on the new alloc with unit dims dropped, and
  // repoint its load user with a collapsed (unit-dim-free) source.
  for (memref::SubViewOp sv : subviews) {
    rewriter.setInsertionPoint(sv);
    auto newSubview = rewriter.create<memref::SubViewOp>(
        loc, newAlloc, dropDims(sv.getMixedOffsets(), unitDims),
        dropDims(sv.getMixedSizes(), unitDims),
        dropDims(sv.getMixedStrides(), unitDims));
    for (OpOperand &use : llvm::make_early_inc_range(sv->getUses())) {
      Operation *owner = use.getOwner();
      Value copySrc = owner->getOperand(0);
      auto copySrcType = cast<MemRefType>(copySrc.getType());
      rewriter.setInsertionPoint(owner);
      auto newSrc = rewriter.create<memref::CollapseShapeOp>(
          loc, copySrc, unitDimReassociation(copySrcType.getShape()));
      owner->setOperand(1, newSubview.getResult());
      owner->setOperand(0, newSrc.getResult());
    }
    if (sv->use_empty())
      rewriter.eraseOp(sv);
  }

  // 3. The scaffold collapse's users take the plain matrix-form buffer.
  rewriter.replaceOp(scaffoldCollapse, newAlloc.getResult());
  if (alloc->use_empty())
    rewriter.eraseOp(alloc);
  return success();
}

namespace {

class FoldCollapseIntoAllocWithLoadPass
    : public impl::FoldCollapseIntoAllocWithLoadBase<
          FoldCollapseIntoAllocWithLoadPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldCollapseIntoAllocWithLoadPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace tensor
} // namespace mlir

std::unique_ptr<mlir::Pass>
mlir::tensor::createFoldCollapseIntoAllocWithLoadPass() {
  return std::make_unique<FoldCollapseIntoAllocWithLoadPass>();
}
