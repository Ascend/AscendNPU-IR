//===- OutlineScope.cpp --------- Outline Scope Pass ----------------------===//
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <string>

#define DEBUG_TYPE "outline-scope"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_DEF_OUTLINESCOPE
#include "bishengir/Dialect/Scope/Transforms/Passes.h.inc"

using namespace impl;
namespace mlir {
namespace scope {

class OutlineScopePass : public OutlineScopeBase<OutlineScopePass> {
public:
  explicit OutlineScopePass() : OutlineScopeBase() {}
  void runOnOperation() final;
};

class OutlineScopeOp : public OpRewritePattern<scope::ScopeOp> {
  SmallVector<Value> getInputs(ScopeOp scopeOp) const {
    SetVector<Value> inputs;
    scopeOp.walk<WalkOrder::PreOrder>([&inputs, &scopeOp](Operation *op) {
      for (auto &opr : op->getOpOperands()) {
        auto val = opr.get();

        // Skip if defined within scope
        if (auto blockArg = dyn_cast<BlockArgument>(val)) {
          if (scopeOp->isAncestor(blockArg.getParentRegion()->getParentOp()))
            continue;
        } else if (auto defOp = val.getDefiningOp()) {
          if (scopeOp->isAncestor(defOp))
            continue;
        }

        inputs.insert(val);
      }
    });
    return inputs.takeVector();
  }

  SetVector<Operation *> getOpsOfScopeOp(ScopeOp scopeOp) const {
    SetVector<Operation *> ops;
    scopeOp.getRegion().walk([&](Operation *op) { ops.insert(op); });
    return ops;
  }

  FailureOr<func::FuncOp> outlineScope(scope::ScopeOp scopeOp,
                                       PatternRewriter &rewriter) const {
#ifndef NDEBUG
    auto numResults = scopeOp->getNumResults();
    assert(numResults == 0 && "unhandled case for scopeOp with results");
#endif

    ModuleOp moduleOp = scopeOp->getParentOfType<ModuleOp>();
    func::FuncOp parF = scopeOp->getParentOfType<func::FuncOp>();
    OpBuilder::InsertionGuard insGuard(rewriter);
    rewriter.setInsertionPoint(parF);

    const std::string prefixFunctionName =
        scopeOp->getParentOfType<func::FuncOp>().getSymName().str() + "_scope";

    SetVector<Operation *> ops = getOpsOfScopeOp(scopeOp);
    SmallVector<Value> inputs = getInputs(scopeOp);

    rewriter.setInsertionPoint(parF);
    FunctionType funcTy =
        FunctionType::get(moduleOp.getContext(), TypeRange(inputs), {});
    func::FuncOp newFuncOp = rewriter.create<func::FuncOp>(
        moduleOp->getLoc(), prefixFunctionName, funcTy, scopeOp->getAttrs());
    SymbolTable symbolTable(moduleOp);
    FailureOr<StringAttr> scopeFuncName =
        symbolTable.renameToUnique(newFuncOp, SmallVector<SymbolTable *>());
    if (failed(scopeFuncName))
      return failure();

    // Create function body
    Block *entryBB = newFuncOp.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBB);

    // Clone operations and replace usages
    LDBG("pushing outlined operations\n");
    IRMapping currentMap;
    for (auto [oldIn, newIn] : llvm::zip_equal(inputs, entryBB->getArguments()))
      currentMap.map(oldIn, newIn);

    SetVector<Operation *> newOps;
    for (auto it = scopeOp.getRegion().op_begin();
         it != scopeOp.getRegion().op_end(); ++it) {
      newOps.insert(rewriter.clone(*it, currentMap));
      LLVM_DEBUG(llvm::dbgs() << "Cloning " << *it << "\n";);
    }
    newFuncOp->walk(
        [&](scope::ReturnOp returnOp) { rewriter.eraseOp(returnOp); });
    LDBG("created FuncOp for outlined scope\n" << *newFuncOp);

    rewriter.create<func::ReturnOp>(entryBB->front().getLoc(),
                                    SmallVector<Value>());
    LDBG("created return Op\n");
    return newFuncOp;
  }

  LogicalResult replaceScopeWithInvoke(scope::ScopeOp scopeOp,
                                       func::FuncOp funcOp,
                                       PatternRewriter &rewriter) const {
    LDBG("Replacing invoke with callOp");
#ifndef NDEBUG
    auto numResults = scopeOp->getNumResults();
    assert(numResults == 0 && "unhandled case for scopeOp with results");
#endif

    SetVector<Operation *> ops = getOpsOfScopeOp(scopeOp);
    rewriter.setInsertionPoint(scopeOp);
    func::CallOp callOp = rewriter.create<func::CallOp>(
        scopeOp->getLoc(), funcOp, getInputs(scopeOp));

    rewriter.replaceOp(scopeOp, callOp);
    return success();
  }

public:
  using OpRewritePattern<scope::ScopeOp>::OpRewritePattern;
  explicit OutlineScopeOp(MLIRContext *context)
      : OpRewritePattern<scope::ScopeOp>(context) {}

  LogicalResult matchAndRewrite(scope::ScopeOp scopeOp,
                                PatternRewriter &rewriter) const override {
    if (scopeOp->getNumResults() > 0) {
      llvm_unreachable("unhandled case for scopeOp with results");
      return failure();
    }

    FailureOr<func::FuncOp> newFuncOp = outlineScope(scopeOp, rewriter);
    if (failed(newFuncOp))
      return failure();
    if (failed(replaceScopeWithInvoke(scopeOp, newFuncOp.value(), rewriter))) {
      return failure();
    }
    return success();
  }
};

void OutlineScopePass::runOnOperation() {
  ModuleOp module = getOperation();
  RewritePatternSet patterns(&getContext());

  patterns.add<OutlineScopeOp>(&getContext());

  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createOutlineScopePass() {
  return std::make_unique<OutlineScopePass>();
}

} // namespace scope
} // namespace mlir