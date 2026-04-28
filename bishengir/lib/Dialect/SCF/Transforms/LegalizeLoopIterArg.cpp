//===----------------------- LegalizeLoopIterArg.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "bishengir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "hivm-legalize-loop-iter-arg"

namespace mlir {
#define GEN_PASS_DEF_LEGALIZELOOPITERARGS
#include "bishengir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct LegalizeLoopIterArgsPass
    : public impl::LegalizeLoopIterArgsBase<LegalizeLoopIterArgsPass> {
  using LegalizeLoopIterArgsBase<
      LegalizeLoopIterArgsPass>::LegalizeLoopIterArgsBase;

public:
  void runOnOperation() override;
};
} // namespace

void LegalizeLoopIterArgsPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *context = op->getContext();
  RewritePatternSet patterns(context);
  OpBuilder builder(context);
  patterns.add<hivmave::ForOpLegalization<false>>(patterns.getContext());

  // Use TopDownTraversal for compile time reasons
  GreedyRewriteConfig grc;
  grc.useTopDownTraversal = true;
  (void)applyPatternsGreedily(op, std::move(patterns), grc);
}

std::unique_ptr<Pass> scf::createLegalizeLoopIterArgsPass() {
  return std::make_unique<LegalizeLoopIterArgsPass>();
}
