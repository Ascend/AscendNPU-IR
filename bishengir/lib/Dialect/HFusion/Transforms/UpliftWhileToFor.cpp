//===- UpliftWhileToFor.cpp - Uplift scf.while to scf.for ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Thin downstream wrapper around upstream
// `mlir::scf::populateUpliftWhileToForPatterns`, exposed as the
// `hfusion-uplift-while-to-for` pass and wired into HFusion preProcess
// (see multibuffer-support-while-op/设计方案.md supplement #2).
//
// The upstream pattern only fires when the `scf.while` matches a
// canonical for-shape:
//   * `before` block: a single `arith.cmpi` feeding `scf.condition`
//   * `after`  block: a linear `arith.addi` on the induction variable
// All other while-loops (e.g. data-driven exit conditions) are left
// untouched. Downstream HIVM multi-buffer can take the simpler scf.for
// counter path whenever this pass succeeds; structurally-while loops
// still fall back to the alloca-based counter machinery in
// MultiBufferLoopAdapter.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_UPLIFTWHILETOFOR
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace {

struct UpliftWhileToForPass
    : public impl::UpliftWhileToForBase<UpliftWhileToForPass> {
  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    scf::populateUpliftWhileToForPatterns(patterns);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createUpliftWhileToForPass() {
  return std::make_unique<UpliftWhileToForPass>();
}
