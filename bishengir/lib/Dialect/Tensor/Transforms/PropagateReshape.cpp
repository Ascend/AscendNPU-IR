//===- PropagateReshape.cpp ------- Propagate Reshape Pass ----------------===//
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
// This file implements a pass to propagate reshape for expandShape
// and elemwise collapseShape.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/MemRef/Transforms/PropagateReshape.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagateCollapseDown.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagateExpandUp.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/PropagateNearEndExpandDown.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/SwapCollapseExpand.h"

#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "propagate-reshape"
namespace mlir {
namespace tensor {
using namespace mlir::hfusion;
using namespace mlir::hfusion::reshape_utils;
#define GEN_PASS_DEF_PROPAGATERESHAPE
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"

namespace {

class PropagateReshapePass
    : public impl::PropagateReshapeBase<PropagateReshapePass> {
public:
  explicit PropagateReshapePass(const PropagateReshapeOptions &options)
      : PropagateReshapeBase(options) {}
  void runOnOperation() final;
};

static bool hasScopeOperation(func::FuncOp function) {
  return function
      .walk([](Operation *op) {
        return op->getName().getStringRef() == "scope.scope"
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

void PropagateReshapePass::runOnOperation() {
  func::FuncOp f = getOperation();
  MLIRContext *context = &getContext();

  PropagateReshapeOptions options;
  options.forHIVM = forHIVM;
  options.forRegbased = forRegbased;
  options.skipScope = skipScope;
  options.maxUnitDimsForPropagation = maxUnitDimsForPropagation;

  if (auto coreType = mlir::hivm::queryFuncCoreType(f);
      coreType && *coreType == mlir::hivm::TFuncCoreType::AIC)
    return;

  if (options.forRegbased && options.skipScope && hasScopeOperation(f))
    return;

  RewritePatternSet patterns(context);
  patterns.add<PropagateCollapseDownToI1Cast>(context);
  if (!options.forRegbased)
    patterns.add<PropagateNearEndExpandDown>(context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
  memref::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
  memref::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
  patterns.add<SwapCollapseExpand>(context, options);
  patterns.add<PropagateExpandUp>(context, options);
  patterns.add<PropagateCollapseDown>(context, options);
  if (!options.forRegbased)
    patterns.add<memref::SwapMemrefCollapseExpand>(context);
  patterns.add<memref::PropagateMemrefExpandUp>(context, options);
  patterns.add<memref::PropagateMemrefCollapseDown>(context, options);

  if (failed(applyPatternsGreedily(f, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<Pass>
createPropagateReshapePass(const PropagateReshapeOptions &options) {
  return std::make_unique<PropagateReshapePass>(options);
}

} // namespace tensor
} // namespace mlir