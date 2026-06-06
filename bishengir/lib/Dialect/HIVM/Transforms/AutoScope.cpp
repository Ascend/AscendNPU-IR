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
#include "llvm/ADT/SmallPtrSet.h"
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
    auto indices = simtOp.getIndices();
    auto indicesTy = dyn_cast<RankedTensorType>(indices.getType());
    if (!indicesTy) {
      return rewriter.notifyMatchFailure(simtOp, "indices must be ranked tensor type");
    }
    unsigned blockSize = indicesTy.getShape().front();
    if (!(blockSize  && !(blockSize & (blockSize - 1)))) {
      llvm::report_fatal_error("BLOCK size of simd_simt mode must be power of 2");
    }
    
    OrderedOps simtVFOps = gatherSimtOps(simtOp);
    auto loc = simtOp->getLoc();
    auto scopeOp = createScope(simtVFOps, rewriter, loc);
    rewriter.replaceOp(simtOp, scopeOp);
    return success();
  }

private:
  using OrderedOps = llvm::SmallVector<Operation *>;
  using VisitedOps = llvm::SmallPtrSet<Operation *, 16>;

  scope::ScopeOp createScope(const OrderedOps &simtVFOps,
                             PatternRewriter &rewriter, Location &loc) const {
    auto scopeOp = rewriter.create<scope::ScopeOp>(
        loc, simtVFOps.back()->getResultTypes(), true);
    scopeOp->setAttr("outline", rewriter.getUnitAttr());
    scopeOp->setAttr(
        TFuncCoreTypeAttr::name,
        TFuncCoreTypeAttr::get(rewriter.getContext(), TFuncCoreType::AIV));
    auto vfMode =
        hivm::VFModeAttr::get(scopeOp->getContext(), hivm::VFMode::SIMT);
    scopeOp->setAttr(hivm::VFModeAttr::name, vfMode);
    rewriter.createBlock(&scopeOp->getRegion(0));
    rewriter.setInsertionPointToStart(&scopeOp.getRegion().getBlocks().front());
    IRMapping mapping;
    Operation *cloned = nullptr;  
    for (Operation *op : simtVFOps) {
      cloned = rewriter.clone(*op, mapping);
      for (size_t i = 0; i < cloned->getNumResults(); i++) {
        mapping.map(op->getResult(i), cloned->getResult(i));
      }
      if (llvm::isa<DestinationStyleOpInterface>(op) &&
          op != simtVFOps.back() && op->use_empty()) {
        rewriter.eraseOp(op);
      }
    }
    if (cloned != nullptr) {
      rewriter.create<scope::ReturnOp>(loc, cloned->getResults());
    }
    return scopeOp;
  }

  void collectValueDependencies(OrderedOps &simtVFOps, VisitedOps &visitedOps,
                                Value val) const {
    auto defOp = val.getDefiningOp();
    if (!defOp || llvm::isa<scope::ScopeOp, memref::AllocOp>(defOp)) {
      return;
    }
    collectOpDependencies(simtVFOps, visitedOps, defOp);
  }

  void collectValueDependencies(OrderedOps &simtVFOps, VisitedOps &visitedOps,
                                Value val, Operation *simtOp) const {
    auto defOp = val.getDefiningOp();
    if (!defOp || llvm::isa<scope::ScopeOp>(defOp)) {
      return;
    }
    if (auto allocOp = llvm::dyn_cast<memref::AllocOp>(defOp)) {
      collectDestinationStyleOp(simtVFOps, visitedOps, allocOp, simtOp);
      return;
    }
    collectOpDependencies(simtVFOps, visitedOps, defOp, simtOp);
  }

  void collectOpDependencies(OrderedOps &simtVFOps, VisitedOps &visitedOps,
                             Operation *op) const {
    if (!visitedOps.insert(op).second) {
      return;
    }
    for (auto operand : op->getOperands()) {
      collectValueDependencies(simtVFOps, visitedOps, operand);
    }
    simtVFOps.emplace_back(op);
  }

  void collectOpDependencies(OrderedOps &simtVFOps, VisitedOps &visitedOps,
                             Operation *op, Operation *simtOp) const {
    if (!visitedOps.insert(op).second) {
      return;
    }
    for (auto operand : op->getOperands()) {
      collectValueDependencies(simtVFOps, visitedOps, operand, simtOp);
    }
    simtVFOps.emplace_back(op);
  }

  void collectDestinationStyleOp(OrderedOps &simtVFOps,
                                 VisitedOps &visitedOps,
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
        continue;
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
          if (!visitedOps.insert(&op).second) {
            return;
          }
          collectValueDependencies(simtVFOps, visitedOps, val);
          collectValueDependencies(simtVFOps, visitedOps,
                                   destinationStyleOp.getDpsInputs()[0],
                                   simtOp);
          simtVFOps.emplace_back(&op);
          break;
        }
      }
    }
  }

  OrderedOps gatherSimtOps(Operation *simtOp) const {
    OrderedOps simtVFOps;
    VisitedOps visitedOps;
    if (auto loadOp = llvm::dyn_cast<hivm::GatherLoadOp>(simtOp)) {
      collectValueDependencies(simtVFOps, visitedOps, loadOp.getBase(), simtOp);
      collectValueDependencies(simtVFOps, visitedOps, loadOp.getIndices(),
                               simtOp);
      collectValueDependencies(simtVFOps, visitedOps, loadOp.getDst(), simtOp);
      if (loadOp.getMask()) {
        collectValueDependencies(simtVFOps, visitedOps, loadOp.getMask(),
                                 simtOp);
      }
      if (loadOp.getOther()) {
        collectValueDependencies(simtVFOps, visitedOps, loadOp.getOther(),
                                 simtOp);
      }
    } else if (auto storeOp =
                   llvm::dyn_cast<hivm::ScatterStoreOp>(simtOp)) {
      collectValueDependencies(simtVFOps, visitedOps, storeOp.getBase(),
                               simtOp);
      collectValueDependencies(simtVFOps, visitedOps, storeOp.getIndices(),
                               simtOp);
      if (storeOp.getMask()) {
        collectValueDependencies(simtVFOps, visitedOps, storeOp.getMask(),
                                 simtOp);
      }
    }
    simtVFOps.emplace_back(simtOp);
    return simtVFOps;
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
