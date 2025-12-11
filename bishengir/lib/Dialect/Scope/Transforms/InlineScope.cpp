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
// This file implements a pass to convert scopeOp to funcOp
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
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

#define GEN_PASS_DEF_INLINESCOPE
#include "bishengir/Dialect/Scope/Transforms/Passes.h.inc"

using namespace impl;
using namespace mlir;

namespace mlir {
namespace scope {
template <typename OpTy>
class ExtractOpsFromBodyPattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  explicit ExtractOpsFromBodyPattern(MLIRContext *context)
      : OpRewritePattern<OpTy>(context) {}

  LogicalResult matchAndRewrite(OpTy regionOp,
                                PatternRewriter &rewriter) const override {
    if (regionOp.getNoInline())
      return failure();
    assert(regionOp->getNumRegions() == 1 && regionOp->getNumResults() == 0 &&
           "handle case of op with single region without result");

    for (auto it = regionOp.getRegion().op_begin();
         it != regionOp.getRegion().op_end(); ++it) {
      if (it->template hasTrait<OpTrait::ReturnLike>())
        continue;

      LLVM_DEBUG(llvm::dbgs() << "Cloning " << *it << "\n";);
      rewriter.setInsertionPoint(regionOp);
      rewriter.clone(*it);
    }
    rewriter.eraseOp(regionOp);
    return success();
  }
};

class ExtractScopeBodyPass : public InlineScopeBase<ExtractScopeBodyPass> {
public:
  explicit ExtractScopeBodyPass() : InlineScopeBase() {}
  void runOnOperation() final;
};

void ExtractScopeBodyPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  RewritePatternSet patterns(&getContext());

  patterns.add<ExtractOpsFromBodyPattern<ScopeOp>>(&getContext());

  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

class InlineScopePass : public InlineScopeBase<InlineScopePass> {
public:
  explicit InlineScopePass() : InlineScopeBase() {}
  void runOnOperation() final;
};

void InlineScopePass::runOnOperation() {
  auto moduleOp = getOperation();
  PassManager pm(moduleOp->getContext());

  pm.addPass(std::make_unique<ExtractScopeBodyPass>());
  pm.addPass(createInlinerPass());

  if (failed(pm.run(moduleOp)))
    return signalPassFailure();
}

std::unique_ptr<Pass> createInlineScopePass() {
  return std::make_unique<InlineScopePass>();
}

} // namespace scope
} // namespace mlir