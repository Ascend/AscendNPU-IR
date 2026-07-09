//===- UpliftWhileToFor.cpp - Uplift scf.while to scf.for ----------------===//
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
// Thin downstream wrapper around upstream
// `mlir::scf::populateUpliftWhileToForPatterns`, exposed as the
// `hfusion-uplift-while-to-for` pass and wired into the head of HFusion
// preProcess (HFusionPipelines.cpp).
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
