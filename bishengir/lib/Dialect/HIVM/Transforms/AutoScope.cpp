//===------------- AutoScope.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement create scope for gather_load and scatter_store.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cstddef>

namespace mlir {
#define GEN_PASS_DEF_AUTOSCOPE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct AutoScopePass : public impl::AutoScopeBase<AutoScopePass> {
  void runOnOperation() override;
};

template <typename SIMTOP>
struct AutoScopePattern : public OpRewritePattern<SIMTOP> {
public:
  using OpRewritePattern<SIMTOP>::OpRewritePattern;
  LogicalResult matchAndRewrite(SIMTOP simtOp,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<Operation *> simtVFOps{simtOp};
    gatherSimtOps(simtVFOps);
    auto loc = simtOp->getLoc();
    auto scopeOp = createScope(simtVFOps, rewriter, loc);
    rewriter.replaceOp(simtOp, scopeOp);
    return success();
  }

private:
  scope::ScopeOp createScope(llvm::SmallVector<Operation *> &simtVFOps,
                             PatternRewriter &rewriter, Location &loc) const {
    auto scopeOp = rewriter.create<scope::ScopeOp>(
        loc, simtVFOps[0]->getResultTypes(), true);
    scopeOp->setAttr(
        hivm::VFModeAttr::getMnemonic(),
        hivm::VFModeAttr::get(rewriter.getContext(), hivm::VFMode::SIMT));
    scopeOp->setAttr("outline", rewriter.getUnitAttr());
    scopeOp->setAttr(hivm::VectorFunctionAttr::getMnemonic(),
                     rewriter.getUnitAttr());
    rewriter.createBlock(&scopeOp->getRegion(0));
    rewriter.setInsertionPointToStart(&scopeOp.getRegion().getBlocks().front());
    IRMapping mapping;
    Operation *cloned;
    for (auto iter = simtVFOps.rbegin(); iter != simtVFOps.rend(); iter++) {
      cloned = rewriter.clone(**iter, mapping);
      for (size_t i = 0; i < cloned->getNumResults(); i++) {
        mapping.map((*iter)->getResult(i), cloned->getResult(i));
      }
      if (llvm::isa<DestinationStyleOpInterface>(*iter)) {
        rewriter.eraseOp(*iter);
      }
    }
    rewriter.create<scope::ReturnOp>(loc, cloned->getResults());
    return scopeOp;
  }

  void gatherDestinationStyleOp(llvm::SmallVector<Operation *> &simtVFOps,
                                Operation *allocOp, Operation *simtOp) const {
    // currently only concern alloc op and simt op in the same block
    if (allocOp->getBlock() != simtOp->getBlock()) {
      return;
    }
    bool startMatch = false;
    auto &opList = allocOp->getBlock()->getOperations();
    // reverse iteration, find the nearest destination style op, which after
    // alloc op and before simt op
    for (auto iter = opList.rbegin(); iter != opList.rend(); iter++) {
      auto &op = *iter;
      if (&op == simtOp) {
        startMatch = true;
      }
      if (!startMatch) {
        continue;
      }
      if (&op == allocOp) {
        break;
      }
      if (auto destinationStyleOp =
              llvm::dyn_cast<DestinationStyleOpInterface>(&op)) {
        bool isOperandsFromAlloc = false;
        for (auto val : destinationStyleOp.getDpsInits()) {
          auto maybeAllocOp = utils::tracebackMemRefToAlloc(val);
          if (maybeAllocOp.has_value() && *maybeAllocOp == allocOp) {
            isOperandsFromAlloc = true;
            break;
          }
        }
        // don't gather destination style op with multiple operands
        if (destinationStyleOp.getDpsInits().size() > 1 &&
            isOperandsFromAlloc) {
          break;
        }
        auto val = destinationStyleOp.getDpsInitOperand(0)->get();
        auto maybeAllocOp = utils::tracebackMemRefToAlloc(val);
        if (maybeAllocOp && *maybeAllocOp == allocOp) {
          simtVFOps.emplace_back(&op);
          recurseIR(simtVFOps, val);
          recurseIR(simtVFOps, destinationStyleOp.getDpsInputs()[0], simtOp);
          break;
        }
      }
    }
  }

  // simply trace IR.
  void recurseIR(llvm::SmallVector<Operation *> &simtVFOps, Value val) const {
    auto defOp = val.getDefiningOp();
    if (!defOp || llvm::isa<scope::ScopeOp, memref::AllocOp>(defOp)) {
      return;
    }
    simtVFOps.emplace_back(defOp);
    for (auto operand : defOp->getOperands()) {
      recurseIR(simtVFOps, operand);
    }
  }

  // trace IR and gather destination style op
  void recurseIR(llvm::SmallVector<Operation *> &simtVFOps, Value val,
                 Operation *simtOp) const {
    auto defOp = val.getDefiningOp();
    if (!defOp || llvm::isa<scope::ScopeOp>(defOp)) {
      return;
    }
    if (llvm::isa<memref::AllocOp>(defOp)) {
      // gather the nearest destionation style op for simt op's operand
      gatherDestinationStyleOp(simtVFOps, defOp, simtOp);
      return;
    }
    simtVFOps.emplace_back(defOp);
    for (auto operand : defOp->getOperands()) {
      recurseIR(simtVFOps, operand, simtOp);
    }
  }

  void gatherSimtOps(llvm::SmallVector<Operation *> &simtVFOps) const {
    if (auto loadOp = llvm::dyn_cast<hivm::GatherLoadOp>(simtVFOps[0])) {
      recurseIR(simtVFOps, loadOp.getBase(), simtVFOps[0]);
      recurseIR(simtVFOps, loadOp.getIndices(), simtVFOps[0]);
      if (loadOp.getMask()) {
        recurseIR(simtVFOps, loadOp.getMask(), simtVFOps[0]);
      }
      if (loadOp.getOther()) {
        recurseIR(simtVFOps, loadOp.getOther(), simtVFOps[0]);
      }
    } else if (auto storeOp =
                   llvm::dyn_cast<hivm::ScatterStoreOp>(simtVFOps[0])) {
      recurseIR(simtVFOps, storeOp.getBase(), simtVFOps[0]);
      recurseIR(simtVFOps, storeOp.getIndices(), simtVFOps[0]);
      if (storeOp.getMask()) {
        recurseIR(simtVFOps, storeOp.getMask(), simtVFOps[0]);
      }
    }
  }
};

} // namespace

void AutoScopePass::runOnOperation() {
  auto mod = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<AutoScopePattern<hivm::GatherLoadOp>>(patterns.getContext());
  patterns.add<AutoScopePattern<hivm::ScatterStoreOp>>(patterns.getContext());

  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;
  (void)applyPatternsGreedily(mod, std::move(patterns), config);
}

std::unique_ptr<Pass> mlir::hivm::createAutoScopePass() {
  return std::make_unique<AutoScopePass>();
}