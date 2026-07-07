//===------------- AutoScope.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements autoscope creation for gather_load and scatter_store.
//
// Core design:
// 1. AutoScope uses a conservative boundary and only clones the tensor/SSA
//    subgraph needed by each SIMT seed.
// 2. Backward collection stops as soon as it reaches a memref-typed value, so
//    memory-access producers stay outside the SIMT scope by default.
// 3. This keeps scope placement independent from alloc sharing and matches the
//    default preference for leaving memory movement in surrounding SIMD code.
// 4. Broader scope expansion is left to future cost-model driven policies.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
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

using OrderedOps = llvm::SmallVector<Operation *>;
using VisitedOps = llvm::SmallPtrSet<Operation *, 16>;
using SeedSet = llvm::SmallPtrSet<Operation *, 16>;

// Per-seed rewrite plan. The plan only contains tensor/SSA producers and the
// seed itself; memref-side producers remain outside the scope boundary.
struct SeedInfo {
  explicit SeedInfo(Operation *seedOp) : seedOp(seedOp) {}
  SeedInfo() = default;

  Operation *seedOp = nullptr;
  OrderedOps plannedOps;
};

bool isSimtSeedOp(Operation *op) {
  return isa<hivm::GatherLoadOp, hivm::ScatterStoreOp>(op);
}

bool isMemRefBoundary(Value value) {
  return llvm::isa<BaseMemRefType>(value.getType());
}

void forEachSeedDependencyValue(Operation *seedOp,
                                llvm::function_ref<void(Value)> fn) {
  if (auto loadOp = llvm::dyn_cast<hivm::GatherLoadOp>(seedOp)) {
    fn(loadOp.getBase());
    fn(loadOp.getIndices());
    fn(loadOp.getDst());
    if (loadOp.getMask()) {
      fn(loadOp.getMask());
    }
    if (loadOp.getOther()) {
      fn(loadOp.getOther());
    }
    return;
  }
  if (auto storeOp = llvm::dyn_cast<hivm::ScatterStoreOp>(seedOp)) {
    fn(storeOp.getBase());
    fn(storeOp.getIndices());
    if (storeOp.getMask()) {
      fn(storeOp.getMask());
    }
  }
}

RankedTensorType getSeedIndicesType(Operation *seedOp) {
  if (auto loadOp = llvm::dyn_cast<hivm::GatherLoadOp>(seedOp)) {
    return dyn_cast<RankedTensorType>(loadOp.getIndices().getType());
  }
  auto storeOp = cast<hivm::ScatterStoreOp>(seedOp);
  return dyn_cast<RankedTensorType>(storeOp.getIndices().getType());
}

void collectOpDependencies(OrderedOps &simtVFOps, VisitedOps &visitedOps,
                           Operation *op, Operation *seedOp,
                           const SeedSet &allSeedOps);

void collectValueDependencies(OrderedOps &simtVFOps, VisitedOps &visitedOps,
                              Value val, Operation *seedOp,
                              const SeedSet &allSeedOps) {
  // Memref-typed values form the default autoscope boundary: keep memory-side
  // producers outside the SIMT scope and only clone the tensor/SSA subgraph.
  if (isMemRefBoundary(val)) {
    return;
  }
  auto defOp = val.getDefiningOp();
  if (!defOp || llvm::isa<scope::ScopeOp>(defOp) ||
      (allSeedOps.contains(defOp) && defOp != seedOp)) {
    return;
  }
  collectOpDependencies(simtVFOps, visitedOps, defOp, seedOp, allSeedOps);
}

void collectOpDependencies(OrderedOps &simtVFOps, VisitedOps &visitedOps,
                           Operation *op, Operation *seedOp,
                           const SeedSet &allSeedOps) {
  if (!visitedOps.insert(op).second) {
    return;
  }
  for (auto operand : op->getOperands()) {
    collectValueDependencies(simtVFOps, visitedOps, operand, seedOp,
                             allSeedOps);
  }
  simtVFOps.emplace_back(op);
}

OrderedOps gatherSimtOps(Operation *seedOp, const SeedSet &allSeedOps) {
  OrderedOps simtVFOps;
  VisitedOps visitedOps;
  forEachSeedDependencyValue(seedOp, [&](Value value) {
    collectValueDependencies(simtVFOps, visitedOps, value, seedOp, allSeedOps);
  });
  simtVFOps.emplace_back(seedOp);
  return simtVFOps;
}

scope::ScopeOp createScope(const OrderedOps &simtVFOps, IRRewriter &rewriter,
                           Location loc) {
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
  }
  if (cloned != nullptr) {
    rewriter.create<scope::ReturnOp>(loc, cloned->getResults());
  }
  return scopeOp;
}

// Check if the op is in simt scope
bool inSimtScope(Operation *op) {
  auto parentOp = op->getParentOp();
  while (parentOp) {
    if (auto scopeOp = llvm::dyn_cast<scope::ScopeOp>(parentOp)) {
      if (auto vectorType = scopeOp->getAttrOfType<StringAttr>("vector_type")) {
        if (vectorType.getValue() == "simt") {
          return true;
        }
      }
    }
    parentOp = parentOp->getParentOp();
  }
  return false;
}

} // namespace

void AutoScopePass::runOnOperation() {
  auto mod = getOperation();

  IRRewriter rewriter(&getContext());
  SeedSet allSeedOps;
  llvm::SmallVector<SeedInfo, 4> seedInfos;

  // Collect every SIMT seed that still needs autoscope materialization.
  mod->walk([&](Operation *op) {
    if (!isSimtSeedOp(op) || inSimtScope(op)) {
      return;
    }
    allSeedOps.insert(op);
    seedInfos.emplace_back(op);
  });

  // Build one conservative tensor/SSA-only rewrite plan per seed.
  for (auto &seedInfo : seedInfos) {
    auto indicesTy = getSeedIndicesType(seedInfo.seedOp);
    if (!indicesTy) {
      continue;
    }
    unsigned blockSize = indicesTy.getShape().front();
    if (!(blockSize && !(blockSize & (blockSize - 1)))) {
      llvm::report_fatal_error(
          "BLOCK size of simd_simt mode must be power of 2");
    }
    seedInfo.plannedOps = gatherSimtOps(seedInfo.seedOp, allSeedOps);
  }

  // Materialize the planned scope for each seed and replace the original op.
  for (auto &seedInfo : seedInfos) {
    rewriter.setInsertionPoint(seedInfo.seedOp);
    auto scopeOp =
        createScope(seedInfo.plannedOps, rewriter, seedInfo.seedOp->getLoc());
    rewriter.replaceOp(seedInfo.seedOp, scopeOp->getResults());
  }

  // Deal with existed scopeOps.
  mod->walk([&](scope::ScopeOp scopeOp) {
    if (auto vectorType = scopeOp->getAttrOfType<StringAttr>("vector_type")) {
      if (vectorType.getValue() == "simt") {
        scopeOp->setAttr("outline", rewriter.getUnitAttr());
        scopeOp->setAttr(
            TFuncCoreTypeAttr::name,
            TFuncCoreTypeAttr::get(rewriter.getContext(), TFuncCoreType::AIV));
        auto vfMode =
            hivm::VFModeAttr::get(scopeOp->getContext(), hivm::VFMode::SIMT);
        scopeOp->setAttr(hivm::VFModeAttr::name, vfMode);
      }
    }
  });
}

std::unique_ptr<Pass> mlir::hivm::createAutoScopePass() {
  return std::make_unique<AutoScopePass>();
}
