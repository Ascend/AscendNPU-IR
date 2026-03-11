//===- PropagateReshape.cpp ------- Propagate Reshape Pass ----------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "bishengir/Dialect/Utils/Util.h"

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

void PropagateReshapePass::runOnOperation() {
  func::FuncOp f = getOperation();
  MLIRContext *context = &getContext();

  PropagateReshapeOptions opts;
  opts.forHIVM = forHIVM;
  opts.forRegbased = forRegbased;

  std::optional<mlir::hivm::TFuncCoreType> funcCoreType =
      mlir::hivm::queryFuncCoreType(f);
  if (funcCoreType.has_value()) {
    if (funcCoreType.value() == mlir::hivm::TFuncCoreType::AIC) {
      return;
    }
  }

  // Experimental propagate reshape, can remove this if
  if (opts.forRegbased && util::hasUnpropagateableCase(f, this->skipScope)) {
    return;
  }

  RewritePatternSet patterns(context);
  if (!opts.forRegbased) {
    patterns.add<PropagateNearEndExpandDown>(context);
  }
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
  ReduceWithIndexOp::getCanonicalizationPatterns(patterns, context);
  patterns.add<SwapCollapseExpand>(context);
  patterns.add<PropagateExpandUp>(context, opts);
  patterns.add<PropagateCollapseDown>(context, opts);
  patterns.add<memref::PropagateMemrefExpandUp>(context);
  patterns.add<memref::PropagateMemrefCollapseDown>(context);

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
