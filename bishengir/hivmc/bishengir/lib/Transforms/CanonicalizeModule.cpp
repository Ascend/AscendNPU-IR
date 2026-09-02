//===----------------- CanonicalizeModeule.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "bishengir-canonicalize-module"

namespace bishengir {
using namespace mlir;

#define GEN_PASS_DEF_CANONICALIZEMODULE
#include "bishengir/Transforms/Passes.h.inc"

namespace {

/// Pattern to erase empty module.
struct EliminateEmptyModuleOp : public OpRewritePattern<ModuleOp> {
public:
  using OpRewritePattern<ModuleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModuleOp moduleOp,
                                PatternRewriter &rewriter) const override {
    if (moduleOp.getBody()->empty()) {
      rewriter.eraseOp(moduleOp);
      return success();
    }
    return failure();
  }
};

void forwardModuleAttributes(ModuleOp sourceModule, ModuleOp destModule) {
  for (auto attr : sourceModule->getAttrs()) {
    destModule->setAttr(attr.getName(), attr.getValue());
  }
}

void populateCanonicalizePatterns(RewritePatternSet &patterns) {
  patterns.add<EliminateEmptyModuleOp>(patterns.getContext());
}

} // namespace

struct CanonicalizeModule
    : public impl::CanonicalizeModuleBase<CanonicalizeModule> {
  explicit CanonicalizeModule() : CanonicalizeModuleBase() {}

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    populateCanonicalizePatterns(patterns);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns))))
      signalPassFailure();

    // Check if the top level module is a perfectly nested module.
    Region &outerRegion = moduleOp->getRegion(0);
    if (outerRegion.empty())
      return;

    // The first region should only have one block.
    Block &outerBlock = outerRegion.front();
    if (outerBlock.getOperations().size() != 1)
      return;

    // The first block should only contain a module.
    Operation &firstOp = outerBlock.front();
    ModuleOp innerModule = dyn_cast<ModuleOp>(firstOp);
    if (!innerModule)
      return;

    // Forward the inner modules attributes
    forwardModuleAttributes(innerModule, moduleOp);

    Region &innerRegion = innerModule->getRegion(0);
    if (innerRegion.empty()) {
      innerModule.erase();
      return;
    }

    Block &innerBlock = innerRegion.front();
    outerBlock.getOperations().splice(outerBlock.begin(),
                                      innerBlock.getOperations());

    innerModule.erase();

    // Recurse on the top level module again
    runOnOperation();
  }
};

} // namespace bishengir

/// Create a Canonicalize pass.
std::unique_ptr<mlir::Pass> bishengir::createCanonicalizeModulePass() {
  return std::make_unique<CanonicalizeModule>();
}
