//===- MergeSamePreloadScopes.cpp ------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
// Merge scopes sharing identical preload_num within the same parent loop op.
// Exclusive dest / src / copy-chain sinking lives in
// hivm-sink-exclusive-preload-work (after CVPipelining, before SplitMix).
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/ShapeRegistry.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hivm-merge-same-preload-scopes"

namespace mlir {
#define GEN_PASS_DEF_MERGESAMEPRELOADSCOPES
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

static LogicalResult processForOp(scf::ForOp forOp, func::FuncOp funcOp) {
  Block *body = forOp.getBody();

  while (true) {
    DenseMap<int32_t, SmallVector<scope::ScopeOp, 4>> preloadNumToScopes;
    for (Operation &op : *body) {
      if (auto scopeOp = dyn_cast<scope::ScopeOp>(&op)) {
        if (auto attr = scopeOp->getAttrOfType<IntegerAttr>(
                hivm::PreloadNumAttr::name)) {
          preloadNumToScopes[attr.getInt()].push_back(scopeOp);
        }
      }
    }

    int32_t targetPreloadNum = -1;
    SmallVector<scope::ScopeOp, 4> targetScopes;
    for (auto &[num, scopes] : preloadNumToScopes) {
      if (scopes.size() > 1) {
        targetPreloadNum = num;
        targetScopes = std::move(scopes);
        break;
      }
    }

    if (targetPreloadNum == -1)
      break;

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "]: Merging " << targetScopes.size()
               << " scopes with preload_num = " << targetPreloadNum << "\n");

    scope::ScopeOp firstScope = targetScopes.front();
    scope::ScopeOp lastScope = targetScopes.back();

    SmallVector<Operation *> opsInSpan;
    auto it = Block::iterator(firstScope);
    for (; it != Block::iterator(lastScope); ++it) {
      opsInSpan.push_back(&*it);
    }
    opsInSpan.push_back(&*it); // lastScope
    ++it;

    // Include ops after last scope until next scope or terminator (e.g.
    // scf.yield)
    for (; it != body->end(); ++it) {
      Operation *op = &*it;
      if (isa<scope::ScopeOp>(op) || op->hasTrait<OpTrait::IsTerminator>())
        break;
      // L41: skip to_tensor ops if preload_num being merged is NOT 0
      if (targetPreloadNum != 0 && isa<bufferization::ToTensorOp>(op))
        continue;
      opsInSpan.push_back(op);
    }

    DenseSet<Operation *> opsInSpanSet(opsInSpan.begin(), opsInSpan.end());
    SmallVector<Operation *> opsToProcess;
    for (Operation *op : opsInSpan) {
      if (!isa<scope::ScopeOp>(op) && isMemoryEffectFree(op)) {
        bool dependsOnSpan =
            llvm::any_of(op->getOperands(), [&](Value operand) {
              Operation *defOp = operand.getDefiningOp();
              return defOp && opsInSpanSet.contains(defOp);
            });
        if (!dependsOnSpan) {
          bool usedOutside = llvm::any_of(op->getUsers(), [&](Operation *user) {
            return !opsInSpanSet.contains(user);
          });
          if (usedOutside) {
            op->moveBefore(firstScope);
            opsInSpanSet.erase(op);
            continue;
          }
        }
      }
      opsToProcess.push_back(op);
    }

    SmallVector<Type> combinedResultTypes;
    for (Operation *op : opsToProcess) {
      auto sOp = dyn_cast<scope::ScopeOp>(op);
      auto attr =
          sOp ? sOp->getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name)
              : nullptr;
      if (sOp && attr && attr.getInt() == targetPreloadNum) {
        combinedResultTypes.append(sOp.getResultTypes().begin(),
                                   sOp.getResultTypes().end());
      }
    }

    OpBuilder builder(firstScope);
    auto newScopeOp = builder.create<scope::ScopeOp>(firstScope.getLoc(),
                                                     combinedResultTypes);

    for (const NamedAttribute &attr : firstScope->getAttrs()) {
      newScopeOp->setAttr(attr.getName(), attr.getValue());
    }
    newScopeOp.setNoInline(true);
    newScopeOp->setAttr(hivm::PreloadNumAttr::name,
                        builder.getI32IntegerAttr(targetPreloadNum));

    for (scope::ScopeOp s : targetScopes) {
      if (s->hasAttr("hivm.has_loop_carried_dep")) {
        newScopeOp->setAttr("hivm.has_loop_carried_dep", builder.getUnitAttr());
        break;
      }
    }

    if (auto coreAttr = funcOp->getAttrOfType<hivm::TFuncCoreTypeAttr>(
            hivm::TFuncCoreTypeAttr::name)) {
      if (coreAttr.getFuncCoreType() == hivm::TFuncCoreType::AIV) {
        newScopeOp->setAttr(hivm::kPipelinedLoopCoreTypeAttrName,
                            hivm::TCoreTypeAttr::get(builder.getContext(),
                                                     hivm::TCoreType::VECTOR));
      } else if (coreAttr.getFuncCoreType() == hivm::TFuncCoreType::AIC) {
        newScopeOp->setAttr(hivm::kPipelinedLoopCoreTypeAttrName,
                            hivm::TCoreTypeAttr::get(builder.getContext(),
                                                     hivm::TCoreType::CUBE));
      }
    }

    Block *newBody = builder.createBlock(&newScopeOp.getRegion());

    SmallVector<Value> collectedReturns;
    SmallVector<scope::ScopeOp> unwrappedScopes;

    for (Operation *op : opsToProcess) {
      auto sOp = dyn_cast<scope::ScopeOp>(op);
      auto attr =
          sOp ? sOp->getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name)
              : nullptr;
      if (sOp && attr && attr.getInt() == targetPreloadNum) {
        unwrappedScopes.push_back(sOp);
        Block &sBody = sOp.getRegion().front();
        auto innerOps = llvm::make_early_inc_range(
            llvm::make_range(sBody.begin(), std::prev(sBody.end())));
        for (Operation &innerOp : innerOps) {
          innerOp.moveBefore(newBody, newBody->end());
        }
        auto *term = sBody.getTerminator();
        for (Value retVal : term->getOperands()) {
          collectedReturns.push_back(retVal);
        }
      } else {
        op->moveBefore(newBody, newBody->end());
      }
    }

    builder.setInsertionPointToEnd(newBody);
    builder.create<scope::ReturnOp>(lastScope.getLoc(), collectedReturns);

    unsigned resIdx = 0;
    for (scope::ScopeOp oldScope : unwrappedScopes) {
      for (Value oldRes : oldScope.getResults()) {
        Value innerVal = collectedReturns[resIdx];
        Value newRes = newScopeOp.getResult(resIdx);
        oldRes.replaceUsesWithIf(innerVal, [&](OpOperand &use) {
          return newScopeOp->isProperAncestor(use.getOwner());
        });
        oldRes.replaceAllUsesWith(newRes);
        resIdx++;
      }
    }

    for (scope::ScopeOp oldScope : llvm::reverse(unwrappedScopes)) {
      oldScope.erase();
    }

    if (failed(mlir::verify(forOp))) {
      forOp.emitError(
          "Verification failed after merging scopes with preload_num ")
          << targetPreloadNum;
      return failure();
    }
  }

  return success();
}

struct MergeSamePreloadScopesPass
    : public impl::MergeSamePreloadScopesBase<MergeSamePreloadScopesPass> {
  using Base = impl::MergeSamePreloadScopesBase<MergeSamePreloadScopesPass>;
  using Base::Base;
  void runOnOperation() override;
};

void MergeSamePreloadScopesPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (!hivm::allowLoopShapeHeuristics(this->bypassShapeRegistry,
                                        funcOp.getName(), this->enablePreload))
      continue;
    SmallVector<scf::ForOp> forOps;
    funcOp.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    for (scf::ForOp forOp : forOps) {
      if (failed(processForOp(forOp, funcOp)))
        return signalPassFailure();
    }
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createMergeSamePreloadScopesPass(
    const MergeSamePreloadScopesOptions &options) {
  return std::make_unique<MergeSamePreloadScopesPass>(options);
}
