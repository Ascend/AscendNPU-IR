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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include <memory>

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
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  void runOnOperation() final;
};

void InlineScopePass::runOnOperation() {
  auto moduleOp = getOperation();

  if (forceInline) {
    moduleOp.walk([](scope::ScopeOp op) { op.setNoInline(false); });
  }

  RewritePatternSet patterns(&getContext());
  patterns.add<ExtractOpsFromBodyPattern>(&getContext());
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns))))
    return signalPassFailure();

  // Keep ordinary scope unwrapping independent of function inlining. The
  // explicit affine-helper recovery option also requests function inlining.
  if (!lowerAffineApplyForInlining)
    return;

  // Try ordinary inlining before lowering only the helpers that remain.
  PassManager pm(moduleOp->getContext());
  pm.addPass(createInlinerPass());
  if (failed(pm.run(moduleOp)))
    return signalPassFailure();

  // Late fallback for fusion helpers whose top-level affine.apply operations
  // prevent inlining at scf.for call sites. Lower those operations in remaining
  // eligible private callees and retry before memory-space inference encounters
  // the helper memref-result boundaries, avoiding module-wide affine lowering.
  // TODO: Remove this workaround once helper generation or earlier helper
  // elimination fixes the underlying issue.
  llvm::SetVector<func::FuncOp> callees;
  moduleOp.walk([&callees](func::CallOp call) {
    if (call->hasAttr("no_inline"))
      return;
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if (callee && callee.isPrivate() && !callee.isExternal() &&
        !callee->hasAttr("no_inline"))
      callees.insert(callee);
  });

  bool changed = false;
  IRRewriter rewriter(&getContext());
  for (func::FuncOp callee : callees) {
    // Only expand scalar offsets in the function body. Preserve affine loops
    // and operations inside nested regions, including no_inline VF scopes.
    for (auto apply :
         llvm::make_early_inc_range(callee.getOps<affine::AffineApplyOp>())) {
      rewriter.setInsertionPoint(apply);
      auto expanded = affine::expandAffineMap(
          rewriter, apply.getLoc(), apply.getAffineMap(), apply.getOperands());
      if (!expanded)
        return signalPassFailure();
      rewriter.replaceOp(apply, *expanded);
      changed = true;
    }
  }
  if (changed && failed(pm.run(moduleOp)))
    return signalPassFailure();
}

std::unique_ptr<Pass>
createInlineScopePass(const mlir::InlineScopeOptions &options) {
  return std::make_unique<InlineScopePass>(options);
}

} // namespace scope
} // namespace mlir
