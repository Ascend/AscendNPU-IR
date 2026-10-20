//===- HIVMCanonicalizationPipeline.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/ArithToAffine/ArithToAffine.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "bishengir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMCANONICALIZATIONPIPELINE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-canonicalization-pipeline"

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct HIVMCanonicalizationPipelinePass
    : public impl::HIVMCanonicalizationPipelineBase<
          HIVMCanonicalizationPipelinePass> {
  void runOnOperation() override;
};
} // namespace

void HIVMCanonicalizationPipelinePass::runOnOperation() {
  auto funcOp = getOperation();

  CanonicalizerOptions canonicalizerOptions;
  canonicalizerOptions.enableExtendedPattern = true;

  OpPassManager pm(func::FuncOp::getOperationName());
  pm.addPass(createArithToAffineConversionPass());
  pm.addPass(scf::createCanonicalizeIterArgPass());
  pm.addPass(createCanonicalizerPass(canonicalizerOptions));
  pm.addPass(createSCFForLoopCanonicalizationPass());
  pm.addPass(createCSEPass());
  // No canonicalizer between cse and single-point: its folds are subsumed by
  // the trailing canonicalizer of this cluster.
  pm.addPass(createHIVMOptSinglePointPass());
  // No canonicalizer between single-point and dead store elimination:
  // dead-store matching does not depend on its rewrites, and the trailing
  // canonicalizer subsumes them.
  pm.addPass(memref::createDeadStoreEliminationPass());
  pm.addPass(createCanonicalizerPass(canonicalizerOptions));

  if (failed(runPipeline(pm, funcOp)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::hivm::createHIVMCanonicalizationPipelinePass() {
  return std::make_unique<HIVMCanonicalizationPipelinePass>();
}
