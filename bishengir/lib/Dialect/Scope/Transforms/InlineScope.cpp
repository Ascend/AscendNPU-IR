//===- InlineScope.cpp --------- Inline Scope Pass ------------------------===//
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
// This file implements a pass to inline scope regions back into their parent
// functions.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <memory>
#include <string>

#define DEBUG_TYPE "inline-scope"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_INLINESCOPE
#include "bishengir/Dialect/Scope/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::impl;

namespace mlir {
namespace scope {

static bool isSimtScope(Operation *op) {
  if (auto vectorMode = op->getAttrOfType<StringAttr>("vector_mode")) {
    return vectorMode.getValue() == "simt";
  }
  return false;
}

class ExtractOpsFromBodyPattern : public OpRewritePattern<ScopeOp> {
public:
  using OpRewritePattern<ScopeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ScopeOp scopeOp,
                                PatternRewriter &rewriter) const override {
    if (scopeOp.getNoInline())
      return failure();

    if (!scopeOp.getRegion().hasOneBlock()) {
      return rewriter.notifyMatchFailure(
          scopeOp, "only single-block scope regions are handled");
    }

    Region &region = scopeOp.getRegion();
    Block &block = region.front();
    auto opsToMove = llvm::make_range(block.begin(), std::prev(block.end()));

    for (Operation &op : llvm::make_early_inc_range(opsToMove)) {
      LLVM_DEBUG(llvm::dbgs() << "Moving " << op << "\n";);
      rewriter.moveOpBefore(&op, scopeOp);
    }

    for (auto [res, opr] : llvm::zip_equal(
             scopeOp.getResults(),
             scopeOp.getRegion().front().getTerminator()->getOperands())) {
      rewriter.replaceAllUsesWith(res, opr);
    }

    rewriter.eraseOp(scopeOp);
    return success();
  }
};

class InlineScopePass : public InlineScopeBase<InlineScopePass> {
public:
  explicit InlineScopePass(const mlir::InlineScopeOptions &options)
      : InlineScopeBase(options) {}
  void runOnOperation() final;
};

void InlineScopePass::runOnOperation() {
  auto moduleOp = getOperation();

  if (forceInline) {
    moduleOp.walk([](scope::ScopeOp op) { op.setNoInline(false); });
  }

  // SIMT scopes are kept outlined: they are lowered to independent SIMT
  // vector functions and must not be merged back into the caller.
  if (preserveSimtScopes) {
    moduleOp.walk([](scope::ScopeOp op) {
      if (isSimtScope(op))
        op.setNoInline(true);
    });
  }

  RewritePatternSet patterns(&getContext());
  patterns.add<ExtractOpsFromBodyPattern>(&getContext());
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns))))
    return signalPassFailure();

  // FIXME: Consider moving it outside the pass
  // Inline the calls that became bodyless (e.g. outlined vector functions).
  PassManager pm(moduleOp->getContext());
  pm.addPass(createInlinerPass());
  if (failed(pm.run(moduleOp)))
    return signalPassFailure();
}

std::unique_ptr<Pass>
createInlineScopePass(const mlir::InlineScopeOptions &options) {
  return std::make_unique<InlineScopePass>(options);
}

} // namespace scope
} // namespace mlir
