//===- ConvertNonContiguousReshapeToCopy.cpp --------------------*- C++ -*-===//
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
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTNONCONTIGUOUSRESHAPETOCOPY
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct ConvertNonContiguousReshapeToCopyPass
    : public impl::ConvertNonContiguousReshapeToCopyBase<
          ConvertNonContiguousReshapeToCopyPass> {
  void runOnOperation() override;
};

struct ConvertMaybeNonContiguousReassociativeReshapeOpToCopy
    : public OpRewritePattern<memref::CollapseShapeOp> {
  using OpRewritePattern<memref::CollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CollapseShapeOp collapse,
                                PatternRewriter &rewriter) const override {
    std::optional<Operation *> maybeAnnotation =
        mlir::utils::getAnnotateOpWithAttr(collapse.getResult(),
                                           "maybeUnCollapsibleReshape");
    bool maybeUnCollapsibleReshape = maybeAnnotation.has_value();
    if (!maybeUnCollapsibleReshape)
      return failure();

    // Erase this attribute no matter what.
    rewriter.eraseOp(maybeAnnotation.value());

    auto enclosingFunc = collapse->getParentOfType<func::FuncOp>();
    if (!enclosingFunc)
      return success();

    std::optional<TFuncCoreType> maybeFuncCoreType =
        queryFuncCoreType(enclosingFunc);
    if (!maybeFuncCoreType.has_value())
      return success();

    // This is a cube function, no need to do this.
    if (maybeFuncCoreType.value() == TFuncCoreType::AIC)
      return success();

    TypedValue<MemRefType> collapseSrc = collapse.getSrc();
    Location loc = collapse.getLoc();
    // Create a new memref.alloc
    Value newAlloc = utils::createTmpBufferOrTensorWithTargetType(
        rewriter, loc, collapseSrc,
        /*targetElemType=*/std::nullopt, /*targetShape=*/std::nullopt,
        /*allowDynShapeAlloc=*/false);

    // Copy maybe non-contiguous data to contiguous alloc
    auto copyOp = rewriter.create<hivm::CopyOp>(loc, SmallVector<Type>{},
                                                collapseSrc, newAlloc);
    ReassociationIndices fullyCollapsed =
        llvm::to_vector(llvm::seq<int64_t>(0, collapseSrc.getType().getRank()));
    copyOp.setCollapseReassociationAttr(getReassociationIndicesAttribute(
        rewriter, SmallVector<ReassociationIndices>({fullyCollapsed})));

    // Collapse the contiguous alloc
    rewriter.modifyOpInPlace(collapse, [&collapse, &newAlloc]() {
      collapse.getSrcMutable().assign(newAlloc);
    });
    return success();
  }
};

} // namespace

void ConvertNonContiguousReshapeToCopyPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ConvertMaybeNonContiguousReassociativeReshapeOpToCopy>(
      patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::hivm::createNonContiguousReshapeToCopyPass() {
  return std::make_unique<ConvertNonContiguousReshapeToCopyPass>();
}
