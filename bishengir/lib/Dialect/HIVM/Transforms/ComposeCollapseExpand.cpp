//===- ComposeCollapseExpand.cpp ------------------------------------------===//
//
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
// This file implement compose of collapse and expand.
//
//===----------------------------------------------------------------------===//
// This file contains code from the LLVM Project.
// Original License: Apache License v2.0 with LLVM Exceptions
// Original Copyright: NA
// Original Source:
// https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Symbol/IR/Symbol.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "compose-collapse-expand"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

namespace mlir {
#define GEN_PASS_DEF_COMPOSECOLLAPSEEXPAND
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct ComposeCollapseExpandPass
    : public impl::ComposeCollapseExpandBase<ComposeCollapseExpandPass> {
  void runOnOperation() override;
};

struct ComposeExpandOfCollapseOpPattern
    : public OpRewritePattern<mlir::memref::ExpandShapeOp> {
public:
  using OpRewritePattern<mlir::memref::ExpandShapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    auto collapseOp =
        expandOp.getSrc().getDefiningOp<memref::CollapseShapeOp>();
    if (!collapseOp)
      return failure();

    ShapedType srcType = collapseOp.getSrcType();
    ShapedType resultType = expandOp.getResultType();

    if (hasNonIdentityLayout(expandOp.getSrc().getType()) ||
        hasNonIdentityLayout(collapseOp.getSrc().getType()) ||
        hasNonIdentityLayout(collapseOp.getResult().getType()))
      return failure();

    int64_t srcRank = srcType.getRank();
    int64_t resultRank = resultType.getRank();

    auto srcReassociation = collapseOp.getReassociationIndices();
    auto resultReassociation = expandOp.getReassociationIndices();
    if (srcRank > resultRank) {
      auto composedReassociation = findCollapsingReassociation(
          srcReassociation, resultReassociation, srcType.getShape(),
          resultType.getShape(), collapseOp, expandOp);
      if (!composedReassociation)
        return failure();

      rewriter.replaceOpWithNewOp<memref::CollapseShapeOp>(
          expandOp, resultType, collapseOp.getSrc(), *composedReassociation);
      return success();
    }
    auto composedReassociation = findCollapsingReassociation(
        resultReassociation, srcReassociation, resultType.getShape(),
        srcType.getShape(), collapseOp, expandOp);
    if (!composedReassociation)
      return failure();

    SmallVector<OpFoldResult> outputShape(getMixedValues(
        expandOp.getStaticOutputShape(), expandOp.getOutputShape(), rewriter));
    rewriter.replaceOpWithNewOp<memref::ExpandShapeOp>(
        expandOp, resultType, collapseOp.getSrc(), *composedReassociation,
        outputShape);
    return success();
  }

private:
  /// Finds the index of a dynamic dimension in the list of dynamic dimensions.
  ///
  /// Given a static shape and an index into its full shape, returns the
  /// corresponding index in the list of only dynamic dimensions.
  ///
  /// Example: For shape [2, ?, 3, ?] and idx=3, returns 1
  /// (the second dynamic dimension).
  ///
  /// @param staticShape The static shape to analyze
  /// @param idx The index in the full shape (0-based from the end)
  /// @return The index in the list of dynamic dimensions, -1 if shape at axis
  /// idx is static
  int64_t findDynamicShapeIndex(ArrayRef<int64_t> staticShape,
                                int64_t idx) const {
    if (!ShapedType::isDynamic(staticShape[idx]))
      return -1;
    int64_t res = 0;
    for (int64_t i = 0; i < idx; ++i)
      if (ShapedType::isDynamic(staticShape[i]))
        ++res;
    return res;
  }

  /// Checks if two dynamic shapes at specified indices are symbolically
  /// equivalent.
  ///
  /// Verifies that the dynamic dimension at srcIndex in srcOp's input
  /// is the same as the dynamic dimension at resultIndex in resultOp's output.
  /// This is done by checking if they're bound to the same symbolic value.
  ///
  /// @param srcOp The collapse shape operation
  /// @param srcIndex Index of the dimension in srcOp's input
  /// @param resultOp The expand shape operation
  /// @param resultIndex Index of the dimension in resultOp's output
  /// @return true if the dynamic shapes are symbolically equivalent
  bool isSameDynamicShape(memref::CollapseShapeOp srcOp, int64_t srcIndex,
                          memref::ExpandShapeOp resultOp,
                          int64_t resultIndex) const {
    // we need this because when output_shape is
    // [2, 320, %dim_0, %dim_1], getShape() returns [%dim_0, %dim_1]
    // so we need to map original shape index to dynamic shape index.
    auto resDynamicIndex =
        findDynamicShapeIndex(resultOp.getResultType().getShape(), resultIndex);
    auto resultSize = resultOp.getOutputShape()[resDynamicIndex]
                          .getDefiningOp<memref::DimOp>();
    if (!resultSize) {
      return false;
    }

    // Checking whether it's the same index
    auto dimIndex = getConstantIntValue(resultSize.getDimension());
    if (!dimIndex.has_value()) {
      return false;
    }

    // Simple case.. where it's the exact same input args
    if (resultSize.getSource() == srcOp.getSrc()) {
      return true;
    }

    // Else we need to check whether two input args are binded to same symbolic
    // value.
    Value srcSymbolicValue;
    Value dstSymbolicValue;
    auto srcDynamicIndex =
        findDynamicShapeIndex(srcOp.getSrcType().getShape(), srcIndex);
    auto srcParentOp = srcOp->getParentOp();
    assert(srcParentOp != nullptr && "srcOp should have parent");
    srcParentOp->walk<WalkOrder::PreOrder>(
        [&](symbol::BindSymbolicShapeOp bindSymbolicShapeOp) {
          if (bindSymbolicShapeOp.getOperand() == srcOp.getSrc()) {
            srcSymbolicValue =
                bindSymbolicShapeOp.getShapeSymbols()[srcDynamicIndex];
          }
          if (bindSymbolicShapeOp.getOperand() == resultSize.getSource()) {
            dstSymbolicValue =
                bindSymbolicShapeOp.getShapeSymbols()[resDynamicIndex];
          }
          WalkResult::advance();
        });

    return srcSymbolicValue == dstSymbolicValue;
  }

  /// Attempts to find a reassociation pattern to collapse srcShape into
  /// resultShape.
  ///
  /// Given the reassociation indices from a CollapseShapeOp and an
  /// ExpandShapeOp, along with their shapes, this function tries to compose
  /// them into a single collapse operation that transforms srcShape directly
  /// into resultShape.
  ///
  /// @param srcReassociation Reassociation indices from the collapse operation
  /// @param resultReassociation Reassociation indices from the expand operation
  /// @param srcShape The input shape to be collapsed
  /// @param resultShape The target shape after collapsing
  /// @param srcOp The original collapse operation
  /// @param resultOp The original expand operation
  /// @return The composed reassociation indices if successful, nullopt
  /// otherwise
  std::optional<SmallVector<ReassociationIndices>> findCollapsingReassociation(
      ArrayRef<ReassociationIndices> srcReassociation,
      ArrayRef<ReassociationIndices> resultReassociation,
      ArrayRef<int64_t> srcShape, ArrayRef<int64_t> resultShape,
      memref::CollapseShapeOp srcOp, memref::ExpandShapeOp resultOp) const {
    SmallVector<ReassociationIndices, 4> composedReassociation;

    if (srcReassociation.empty())
      return {getReassociationIndicesForCollapse(srcShape, resultShape)};

    for (auto item : llvm::zip_equal(srcReassociation, resultReassociation)) {
      auto &srcIndices = std::get<0>(item);
      auto &resultIndices = std::get<1>(item);
      auto srcSubShape = srcShape.slice(srcIndices.front(), srcIndices.size());
      auto resultSubShape =
          resultShape.slice(resultIndices.front(), resultIndices.size());

      auto srcIdx = srcIndices.front();
      auto resIdx = resultIndices.front();
      if (srcSubShape.size() == resultSubShape.size()) {
        if (srcSubShape == resultSubShape) {
          for (auto shape : srcSubShape) {
            if (shape == ShapedType::kDynamic &&
                !isSameDynamicShape(srcOp, srcIdx, resultOp, resIdx)) {
              return std::nullopt;
            }
            srcIdx++;
            resIdx++;
          }
          for (auto index : llvm::seq<int64_t>(0, srcSubShape.size())) {
            composedReassociation.emplace_back(1, srcIndices.front() + index);
          }
        } else {
          return std::nullopt;
        }
        continue;
      }

      // Find reassociation to collapse `srcSubShape` into `resultSubShape`.
      auto subShapeReassociation =
          getReassociationIndicesForCollapse(srcSubShape, resultSubShape);
      if (!subShapeReassociation)
        return std::nullopt;

      // Remap the subshape indices back to the original srcShape.
      for (auto &subshape_indices : *subShapeReassociation) {
        ReassociationIndices shape_indices;
        for (int64_t index : subshape_indices)
          shape_indices.push_back(srcIndices.front() + index);
        composedReassociation.push_back(shape_indices);
      }
    }
    return {std::move(composedReassociation)};
  }
}; // Struct ComposeExpandOfCollapseOpPattern

} // namespace

void ComposeCollapseExpandPass::runOnOperation() {
  auto *funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  RewritePatternSet patterns(&getContext());
  patterns.add<ComposeExpandOfCollapseOpPattern>(patterns.getContext());
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createComposeCollapseExpandPass() {
  return std::make_unique<ComposeCollapseExpandPass>();
}
