//===--------- DeadStoreElimination.cpp - Load Forward and DSE ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to optimize memref in a scope
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/MemRef/Transforms/Passes.h"

#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/Utils/ValueDependencyAnalyzer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/Visitors.h"

#define DEBUG_TYPE "memref-dse"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_MEMREFDEADSTOREELIMINATIONOP
#include "bishengir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct MemrefDeadStoreElimination
    : public impl::MemrefDeadStoreEliminationOpBase<
          MemrefDeadStoreElimination> {
  using Base::Base;
  void runOnOperation() override;

private:
  void handleMemoryWriteOp(
      MemoryEffectOpInterface memOp, SmallVector<Value> &storeOpMemrefs,
      DenseMap<Value, SmallVector<memref::StoreOp>> &storeOpOnMemref) {
    assert(memOp.hasEffect<MemoryEffects::Write>() &&
           "operation doesn't have memory effect write");

    SmallVector<MemoryEffects::EffectInstance> effects;
    memOp.getEffects(effects);

    // value is overwritten
    for (const MemoryEffects::EffectInstance &effect : effects) {
      if (!isa<MemoryEffects::Write>(effect.getEffect()))
        continue;
      Value value = effect.getValue();
      if (!value)
        continue;
      auto rootValue = analyzer.getAllocOf(value);
      LDBG("Clear cached of " << rootValue);
      storeOpOnMemref[rootValue].clear();
      storeOpMemrefs.push_back(rootValue);
    }
  }

  // returns overwritten allocation from current level
  SmallVector<Value> precomputeStoreToLoad(Operation *parent) {
    SmallVector<Value> storeOpMemrefs;
    DenseMap<Value, SmallVector<memref::StoreOp>> storeOpOnMemref;
    for (Region &region : parent->getRegions()) {
      for (Block &block : region) {
        for (Operation &op : block) {
          if (memref::StoreOp storeOp = dyn_cast<memref::StoreOp>(op)) {
            // only allowed to forward store to load if on the same level
            auto alloc = analyzer.getAllocOf(storeOp.getMemRef());
            LDBG("Storing " << alloc << " with " << storeOp.getValue());
            storeOpOnMemref[alloc].push_back(storeOp);
            // possible forwardable storeOp of parent should be cleared
            storeOpMemrefs.push_back(alloc);
          } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
            auto alloc = analyzer.getAllocOf(loadOp.getMemRef());
            // get the latest stored value
            for (auto it = storeOpOnMemref[alloc].rbegin();
                 it != storeOpOnMemref[alloc].rend(); ++it) {
              memref::StoreOp storeOp = *it;
              if (utils::isScalarLike(storeOp.getMemRef()) ||
                  storeOp.getIndices() == loadOp.getIndices()) {
                LDBG("Load value at " << alloc << " with "
                                      << storeOp.getValue());
                loadedOp[loadOp] = storeOp;
                break;
              }
            }
          } else if (auto memOp = dyn_cast<MemoryEffectOpInterface>(op)) {
            if (!memOp.hasEffect<MemoryEffects::Write>())
              continue;
            handleMemoryWriteOp(memOp, storeOpMemrefs, storeOpOnMemref);
          } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
            for (Value arg : callOp.getArgOperands()) {
              // assuming that the function will replace the value of the passed
              // arguments
              auto alloc = analyzer.getAllocOf(arg);
              LDBG("Clear cached of " << alloc);
              storeOpOnMemref[alloc].clear();
              storeOpMemrefs.push_back(alloc);
            }
            // TODO: tranverse the func call (possible improvement)
          } else if (op.getNumRegions() > 0) {
            // maintain the possible forwardable storeOp of parent that will be
            // cleared
            SmallVector<Value> overwrittenMemref = precomputeStoreToLoad(&op);
            storeOpMemrefs.append(overwrittenMemref.begin(),
                                  overwrittenMemref.end());

            // clear every overwritten memref on different level
            for (Value memref : overwrittenMemref)
              storeOpOnMemref[memref].clear();
          }
        }
      }
    }

    return storeOpMemrefs;
  }

  void analyzeForwardableStore(Operation *parent) {
    // build dependency to get allocation for each values.
    analyzer.buildValueDependency(parent);
    LDBG("Finish building dependency");

    // map forwardable store op to load op
    precomputeStoreToLoad(parent);
    LDBG("Finish precomputing forwardable store op to load op");
  }

  // forward loaded value if the stored value is the last changes applied to the
  // operand.
  void storeToLoadForwarding(memref::LoadOp loadOp) {
    // no forwardable store to load
    if (!loadedOp.contains(loadOp))
      return;

    auto storeOp = loadedOp[loadOp];
    LDBG("forwarding " << storeOp.getValue() << " to " << loadOp);
    loadOp.replaceAllUsesWith(storeOp.getValue());
    loadOp.erase();
  }

  DenseMap<memref::LoadOp, memref::StoreOp> loadedOp;
  utils::ValueDependencyAnalyzer analyzer;
};
} // namespace

void MemrefDeadStoreElimination::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // analyze forwardable store to load
  analyzeForwardableStore(funcOp);
  LDBG("Finish analyzing");

  // forwarding store to load
  funcOp->walk(
      [this](memref::LoadOp loadOp) { storeToLoadForwarding(loadOp); });
  LDBG("Finish forwarding");

  IRRewriter rewriter(&getContext());
  LDBG("func propagated\n" << *funcOp << "\n");
  memref::eraseDeadAllocAndStores(rewriter, funcOp);
}

std::unique_ptr<Pass> memref::createDeadStoreEliminationPass() {
  return std::make_unique<MemrefDeadStoreElimination>();
}