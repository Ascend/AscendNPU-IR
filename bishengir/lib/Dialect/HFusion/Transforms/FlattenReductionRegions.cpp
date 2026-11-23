//===- FlattenReductionRegions.cpp ----------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include <limits>

#define DEBUG_TYPE "hfusion-flatten-reduction-regions"

namespace mlir {
#define GEN_PASS_DEF_FLATTENREDUCTIONREGIONS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace {

/// Keep layout-sensitive tensors and index-dependent payloads outside the
/// region. In particular, do not undo the sub-byte mask preparation before V2.
static bool hasSupportedPayload(linalg::GenericOp op) {
  if (!op.hasPureTensorSemantics() || op.hasIndexSemantics())
    return false;
  for (Value operand : op->getOperands()) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    if (!type)
      continue;
    Type elementType = type.getElementType();
    if (type.getEncoding() || !elementType.isIntOrFloat() ||
        elementType.getIntOrFloatBitWidth() < 8)
      return false;
  }
  return llvm::all_of(
      op.getRegion().front().without_terminator(), [](Operation &nested) {
        return nested.getNumRegions() == 0 && isMemoryEffectFree(&nested);
      });
}

/// A full-domain input fixes the common iteration shape. Broadcast, transpose,
/// and other projected input maps are boundaries, not candidates to duplicate
/// or materialize inside the region. Scalar inputs are unchanged.
static FailureOr<SmallVector<int64_t>> getIterationShape(linalg::GenericOp op) {
  if (!hasSupportedPayload(op))
    return failure();
  SmallVector<int64_t> shape;
  for (OpOperand *input : op.getDpsInputOperands()) {
    AffineMap map = op.getMatchingIndexingMap(input);
    auto type = dyn_cast<RankedTensorType>(input->get().getType());
    if (!type || type.getRank() == 0) {
      if (map.getNumResults() != 0)
        return failure();
      continue;
    }
    if (!map.isIdentity())
      return failure();
    if (shape.empty())
      llvm::append_range(shape, type.getShape());
    else if (ArrayRef<int64_t>(shape) != type.getShape())
      return failure();
  }
  if (shape.empty())
    return failure();
  return shape;
}

/// Reduction results retain every parallel dimension, in its original order.
/// This also accepts variadic reductions and arbitrary scalar combiners.
static bool hasProjectedOutputs(linalg::GenericOp op) {
  SmallVector<AffineExpr> parallelDims;
  for (auto [dim, iterator] : llvm::enumerate(op.getIteratorTypesArray())) {
    if (iterator == utils::IteratorType::parallel)
      parallelDims.push_back(getAffineDimExpr(dim, op.getContext()));
    else if (iterator != utils::IteratorType::reduction)
      return false;
  }
  AffineMap expected =
      AffineMap::get(op.getNumLoops(), 0, parallelDims, op.getContext());
  return llvm::all_of(op.getDpsInitsMutable(), [&](OpOperand &init) {
    return op.getMatchingIndexingMap(&init) == expected;
  });
}

/// Only adjacent reduction dimensions can join. Parallel dimensions are kept
/// separate, including unit dimensions. A group contains at most one dynamic
/// extent, matching the RegBase flatten policy.
static SmallVector<ReassociationIndices>
getReductionGroups(linalg::GenericOp op, ArrayRef<int64_t> shape) {
  auto iterators = op.getIteratorTypesArray();
  SmallVector<ReassociationIndices> groups;
  int64_t product = 1;
  unsigned dynamicDims = 0;
  for (auto [dim, extent] : llvm::enumerate(shape)) {
    bool dynamic = ShapedType::isDynamic(extent);
    bool canJoin =
        dim != 0 && iterators[dim] == utils::IteratorType::reduction &&
        iterators[dim - 1] == utils::IteratorType::reduction && extent != 0 &&
        shape[dim - 1] != 0 && (!dynamic || dynamicDims == 0) &&
        (dynamic || product <= std::numeric_limits<int64_t>::max() / extent);
    if (!canJoin) {
      groups.emplace_back();
      product = 1;
      dynamicDims = 0;
    }
    groups.back().push_back(dim);
    if (dynamic)
      ++dynamicDims;
    else
      product *= extent;
  }
  return groups;
}

struct ReductionRegion {
  linalg::GenericOp root;
  SmallVector<int64_t> shape;
  SmallVector<utils::IteratorType> iterators;
  SmallVector<ReassociationIndices> groups;
  llvm::SetVector<Operation *> operations;

  bool accepts(linalg::GenericOp op) const {
    if (!op || op->getBlock() != root->getBlock())
      return false;
    auto candidateShape = getIterationShape(op);
    if (failed(candidateShape) || *candidateShape != shape ||
        !hasProjectedOutputs(op))
      return false;
    auto candidateIterators = op.getIteratorTypesArray();
    if (candidateIterators != iterators &&
        !llvm::all_of(candidateIterators, [](utils::IteratorType iterator) {
          return iterator == utils::IteratorType::parallel;
        }))
      return false;
    return linalg::areDimSequencesPreserved(op.getIndexingMapsArray(), groups);
  }

  void collect() {
    operations.insert(root);
    for (unsigned i = 0; i < operations.size(); ++i) {
      auto op = cast<linalg::GenericOp>(operations[i]);
      auto add = [&](Operation *neighbor) {
        auto generic = dyn_cast_or_null<linalg::GenericOp>(neighbor);
        if (accepts(generic))
          operations.insert(neighbor);
      };
      // Init operands are not dataflow edges. Shared DPS initializers must not
      // pull unrelated computations into the same analysis scope.
      for (Value input : op.getDpsInputs())
        add(input.getDefiningOp());
      if (op.getNumReductionLoops() == 0) {
        for (Value result : op->getResults()) {
          for (OpOperand &use : result.getUses()) {
            auto user = dyn_cast<linalg::GenericOp>(use.getOwner());
            if (user && use.getOperandNumber() < user.getNumDpsInputs())
              add(user);
          }
        }
      }
    }

    // Do not clone producers for outside users or change their iteration
    // domain. Prune to a fixed point so their producers also stay outside the
    // region.
    bool changed;
    do {
      changed = false;
      for (Operation *candidate : operations.getArrayRef().vec()) {
        auto op = cast<linalg::GenericOp>(candidate);
        if (op.getNumReductionLoops() != 0)
          continue;
        if (llvm::any_of(op->getUsers(), [&](Operation *user) {
              return !operations.contains(user);
            })) {
          operations.remove(candidate);
          changed = true;
        }
      }
    } while (changed);
  }
};

struct FlattenReductionRegionsPass
    : public impl::FlattenReductionRegionsBase<FlattenReductionRegionsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!util::shouldApplyFlattenOpsPass(func) || hacc::utils::isHost(func) ||
        util::hasUnpropagateableCase(func, /*skipScope=*/true))
      return;

    SmallVector<linalg::GenericOp> roots;
    func.walk([&](linalg::GenericOp op) {
      if (op.getNumReductionLoops() > 1)
        roots.push_back(op);
    });
    IRRewriter rewriter(&getContext());
    DenseSet<Operation *> rewritten;
    for (linalg::GenericOp root : roots) {
      // A sibling reduction may already have been rewritten with this root's
      // common producer. Test pointer membership before inspecting the old op.
      if (rewritten.contains(root.getOperation()))
        continue;
      auto shape = getIterationShape(root);
      if (failed(shape) || !hasProjectedOutputs(root))
        continue;
      auto groups = getReductionGroups(root, *shape);
      if (groups.size() == shape->size())
        continue;
      ReductionRegion region{
          root, *shape, root.getIteratorTypesArray(), groups, {}};
      if (!region.accepts(root))
        continue;
      region.collect();
      // Same-block SSA order is topological. Collapsing a producer first lets
      // canonicalization fold the expand/collapse pairs at internal edges.
      SmallVector<linalg::GenericOp> ordered;
      for (Operation &op : *root->getBlock())
        if (region.operations.contains(&op))
          ordered.push_back(cast<linalg::GenericOp>(&op));
      for (linalg::GenericOp op : ordered) {
        rewriter.setInsertionPoint(op);
        auto collapsed = linalg::collapseOpIterationDims(op, groups, rewriter);
        if (failed(collapsed)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Skipping reduction collapse rejected by the helper: "
                     << op << "\n");
          continue;
        }
        rewritten.insert(op.getOperation());
        rewriter.replaceOp(op, collapsed->results);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createFlattenReductionRegionsPass() {
  return std::make_unique<FlattenReductionRegionsPass>();
}
