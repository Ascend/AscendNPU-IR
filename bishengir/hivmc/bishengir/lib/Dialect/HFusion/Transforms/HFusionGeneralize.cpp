//===- HFusionGeneralize.cpp ---- convert hfusionOp To linalg.generic ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
namespace mlir {
#define GEN_PASS_DEF_HFUSIONGENERALIZEPASS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hfusion-generalize"

using namespace mlir;

namespace {
struct HFusionGeneralizePass
    : public impl::HFusionGeneralizePassBase<HFusionGeneralizePass> {
public:
  void runOnOperation() override;
};

void HFusionGeneralizePass::runOnOperation() {
  auto module = getOperation();
  module.walk([&](hfusion::GatherOp op) {
    IRRewriter rewriter(op->getContext());
    rewriter.setInsertionPoint(op);
    (void)generalizeNamedOp(rewriter, op);
  });
}
} // anonymous namespace

std::unique_ptr<Pass> mlir::hfusion::createHFusionGeneralizePass() {
  return std::make_unique<HFusionGeneralizePass>();
}
