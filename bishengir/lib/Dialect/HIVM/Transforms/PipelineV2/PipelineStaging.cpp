//===- PipelineStaging.cpp - Pipeline Staging pass for mix-cv ops --------===//
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

#include "Util.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <memory>

#define DEBUG_TYPE "pipeline-staging"

namespace mlir {
using namespace hivm;

#define GEN_PASS_DEF_PIPELINESTAGING
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

namespace {

struct PipelineStagingPass
    : public ::mlir::impl::PipelineStagingBase<PipelineStagingPass> {
  using Base::Base;
  void runOnOperation() final;
};

} // namespace

static void collectCVPipelinedLoops(func::FuncOp &funcOp,
                                    SmallVector<LoopLikeOpInterface> &loops) {
  funcOp.walk<WalkOrder::PreOrder>(
      [&](LoopLikeOpInterface loopOp) -> WalkResult {
        if (loopOp->hasAttr(hivm::CVPipelinedLoopAttr::name)) {
          loops.push_back(loopOp);
          return WalkResult::skip();
        }
        return WalkResult::advance();
      });
}

static void stageScopesNaively(SmallVector<scope::ScopeOp> &scopeOps) {
  int32_t numStages = llvm::range_size(scopeOps);
  int32_t stage = numStages - 1;

  for (scope::ScopeOp scopeOp : scopeOps) {
    scopeOp.setNoInline(true);
    scopeOp->setAttr(
        hivm::PreloadNumAttr::name,
        IntegerAttr::get(IntegerType::get(scopeOp->getContext(), 32), stage--));
    scopeOp->setAttr(
        hivm::MaxPreloadNumAttr::name,
        IntegerAttr::get(IntegerType::get(scopeOp->getContext(), 32),
                         numStages));
  }
}

static void stageScopes(LoopLikeOpInterface &loopOp,
                        int maxPipelineMultibuffer) {
  for (Region &region : loopOp->getRegions()) {
    SmallVector<scope::ScopeOp> scopes;
    for (scope::ScopeOp scopeOp : region.getOps<scope::ScopeOp>())
      scopes.push_back(scopeOp);
    if (scopes.empty())
      continue;
    if (scopes.size() < maxPipelineMultibuffer) {
      stageScopesNaively(scopes);
      continue;
    }

    // Compute dependencies of each scope
    DenseMap<scope::ScopeOp, SetVector<scope::ScopeOp>> dependsOn;
    for (scope::ScopeOp scopeOp : scopes) {
      SetVector<scope::ScopeOp> dependencies;
      DenseSet<Value> visited;
      SmallVector<Value> worklist;

      scopeOp.walk([&](Operation *op) {
        worklist.append(op->operand_begin(), op->operand_end());
      });

      while (!worklist.empty()) {
        Value value = worklist.pop_back_val();
        if (!visited.insert(value).second)
          continue;

        Operation *defOp = value.getDefiningOp();
        if (!defOp)
          continue;

        if (auto dependency = dyn_cast<scope::ScopeOp>(defOp)) {
          if (!scopeOp->isProperAncestor(dependency)) {
            dependencies.insert(dependency);
            continue;
          }
        }
        if (defOp->getBlock() == scopeOp->getBlock() &&
            defOp->isBeforeInBlock(scopeOp))
          worklist.append(defOp->operand_begin(), defOp->operand_end());
      }

      dependsOn[scopeOp] = std::move(dependencies);
    }

    auto dependsTransitivelyOn = [&](scope::ScopeOp scopeOp,
                                     scope::ScopeOp dependency) {
      SmallVector<scope::ScopeOp, 8> worklist(dependsOn[scopeOp].begin(),
                                              dependsOn[scopeOp].end());
      llvm::SetVector<scope::ScopeOp> visited;
      while (!worklist.empty()) {
        scope::ScopeOp current = worklist.pop_back_val();
        if (current == dependency)
          return true;
        if (!visited.insert(current))
          continue;
        auto depIt = dependsOn.find(current);
        if (depIt != dependsOn.end())
          worklist.append(depIt->second.begin(), depIt->second.end());
      }
      return false;
    };

    using Stage = SmallVector<scope::ScopeOp, 4>;
    SmallVector<Stage, 8> stages; // earliest stage is 0
    DenseMap<scope::ScopeOp, int32_t> scopeToStage;
    SmallVector<scope::ScopeOp, 8> scheduledScopes;

    auto moveScopeToStage = [&](scope::ScopeOp scopeOp, int32_t targetStage) {
      int32_t oldStage = scopeToStage.lookup(scopeOp);
      if (oldStage == targetStage)
        return false;

      Stage &from = stages[oldStage];
      auto *fromIt = llvm::find(from, scopeOp);
      assert(fromIt != from.end() && "scope missing from assigned stage");
      from.erase(fromIt);

      stages[targetStage].push_back(scopeOp);
      scopeToStage[scopeOp] = targetStage;
      return true;
    };

    auto moveDependencyGroupToStage = [&](scope::ScopeOp root,
                                          int32_t targetStage) {
      bool changed = false;
      for (scope::ScopeOp candidate : scheduledScopes) {
        int32_t candidateStage = scopeToStage.lookup(candidate);
        if (candidateStage > targetStage)
          continue;
        if (candidate == root || dependsTransitivelyOn(candidate, root))
          changed |= moveScopeToStage(candidate, targetStage);
      }
      return changed;
    };

    for (scope::ScopeOp scopeOp : scopes) {
      int32_t currentStage = stages.size();
      stages.push_back({scopeOp});
      scopeToStage[scopeOp] = currentStage;
      scheduledScopes.push_back(scopeOp);

      bool changed = true;
      while (changed) {
        changed = false;

        for (scope::ScopeOp consumer : scheduledScopes) {
          int32_t consumerStage = scopeToStage.lookup(consumer);
          for (scope::ScopeOp producer : dependsOn[consumer]) {
            auto producerStageIt = scopeToStage.find(producer);
            assert(producerStageIt != scopeToStage.end());

            int32_t producerStage = producerStageIt->second;
            int32_t distance = consumerStage - producerStage + 1;
            if (distance <= maxPipelineMultibuffer)
              continue;

            int32_t targetStage = consumerStage - maxPipelineMultibuffer + 1;
            changed |= moveDependencyGroupToStage(producer, targetStage);
          }
        }
      }
    }

    SmallVector<Stage, 8> compactStages;
    for (Stage &stage : stages) {
      if (!stage.empty())
        compactStages.push_back(std::move(stage));
    }

    int32_t maxPreloadNum = compactStages.size();
    int32_t preloadNum = maxPreloadNum - 1;
    IntegerType i32Ty = IntegerType::get(loopOp->getContext(), 32);
    for (Stage &stage : compactStages) {
      for (scope::ScopeOp scopeOp : stage) {
        scopeOp.setNoInline(true);
        scopeOp->setAttr(hivm::PreloadNumAttr::name,
                         IntegerAttr::get(i32Ty, preloadNum));
        scopeOp->setAttr(hivm::MaxPreloadNumAttr::name,
                         IntegerAttr::get(i32Ty, maxPreloadNum));
      }
      --preloadNum;
    }
  }
}

static bool isFixpipeToUB(Operation *op) {
  auto fixpipe = dyn_cast<FixpipeOp>(op);
  if (!fixpipe)
    return false;
  Value dst = fixpipe.getDst();
  Operation *def = traceValueDef(dst).getDefiningOp();
  if (!isa_and_nonnull<memref::AllocOp>(def))
    return false;

  auto memSpaceAttr = dyn_cast_or_null<AddressSpaceAttr>(
      cast<MemRefType>(def->getResult(0).getType()).getMemorySpace());
  if (!memSpaceAttr)
    return false;

  return memSpaceAttr.getAddressSpace() == AddressSpace::UB;
}

static bool hasWriteEffectOnAlias(Operation *op, Value root) {
  auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectInterface)
    return false;

  SmallVector<MemoryEffects::EffectInstance> effects;
  effectInterface.getEffects(effects);
  return llvm::any_of(effects, [&](MemoryEffects::EffectInstance &effect) {
    Value effectValue = effect.getValue();
    return isa<MemoryEffects::Write>(effect.getEffect()) && effectValue &&
           traceValueDef(effectValue) == root;
  });
}

static void getAllWritesToAlloc(Value root, SmallVector<Operation *> &writes) {
  SmallVector<Value> valuesToVisit{root};

  DenseSet<Value> visitedValues;
  DenseSet<Operation *> visitedOps;
  while (!valuesToVisit.empty()) {
    Value value = valuesToVisit.pop_back_val();
    if (!visitedValues.insert(value).second)
      continue;

    for (Operation *user : value.getUsers()) {
      if (!visitedOps.insert(user).second)
        continue;

      // Marks describe the buffer and default to a write effect, but do not
      // constitute the cross-core transfer being counted here.
      if (!isa<annotation::MarkOp>(user) && hasWriteEffectOnAlias(user, root)) {
        writes.push_back(user);
      }

      for (Value result : user->getResults()) {
        if (isa<MemRefType>(result.getType()) && traceValueDef(result) == root)
          valuesToVisit.push_back(result);
      }

      if (auto loop = dyn_cast<LoopLikeOpInterface>(user)) {
        for (BlockArgument iterArg : loop.getRegionIterArgs()) {
          if (traceValueDef(iterArg) == root)
            valuesToVisit.push_back(iterArg);
        }
      }
    }
  }
}

static void sortScopes(LoopLikeOpInterface &loopOp) {
  for (Region &region : loopOp->getRegions()) {
    assert(llvm::all_of(region.getOps(),
                        [](Operation &op) {
                          return isa<scope::ScopeOp>(op) ||
                                 op.hasTrait<OpTrait::IsTerminator>();
                        }) &&
           "expected only scopes in the CV-pipelined loop body");

    SmallVector<scope::ScopeOp> scopes;
    for (scope::ScopeOp scopeOp : region.getOps<scope::ScopeOp>())
      scopes.push_back(scopeOp);
    if (scopes.size() < 2)
      continue;

    llvm::stable_sort(scopes, [](scope::ScopeOp lhs, scope::ScopeOp rhs) {
      auto lhsPreloadNum =
          lhs->getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name);
      auto rhsPreloadNum =
          rhs->getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name);
      return lhsPreloadNum.getInt() > rhsPreloadNum.getInt();
    });

    for (size_t i = scopes.size() - 1; i > 0; --i) {
      if (!scopes[i - 1]->isBeforeInBlock(scopes[i]))
        scopes[i - 1]->moveBefore(scopes[i]);
    }
  }
}

static void hoistAllocAndToTensor(LoopLikeOpInterface &loopOp) {
  for (Region &region : loopOp->getRegions()) {
    SmallVector<scope::ScopeOp> scopes;
    for (scope::ScopeOp scopeOp : region.getOps<scope::ScopeOp>())
      scopes.push_back(scopeOp);
    if (scopes.empty())
      continue;

    for (scope::ScopeOp scopeOp : llvm::make_early_inc_range(scopes)) {
      if (scopeOp.getNumResults() == 0)
        continue;

      Block &body = scopeOp.getRegion().front();
      auto returnOp = cast<scope::ReturnOp>(body.getTerminator());

      auto onlyWrittenToOnceByCrossCore = [&](Value allocVal) -> Operation * {
        SmallVector<Operation *> writes;
        getAllWritesToAlloc(allocVal, writes);

        if (writes.size() != 1)
          return nullptr;

        Operation *write = writes[0];
        if (isFixpipeToUB(write) || isCrossCoreCopy(write))
          return write;
        return nullptr;
      };

      DenseMap<unsigned, Value> replacementResults;
      DenseSet<Operation *> moveBeforeSet;
      DenseSet<Operation *> moveAfterSet;
      for (auto [index, returnOperand] :
           llvm::enumerate(returnOp->getOperands())) {
        auto toTensorOp =
            returnOperand.getDefiningOp<bufferization::ToTensorOp>();
        if (!toTensorOp || toTensorOp->getBlock() != &body ||
            !returnOperand.hasOneUse())
          continue;

        Value operand = toTensorOp.getOperand();
        auto castOp = operand.getDefiningOp<memref::MemorySpaceCastOp>();
        auto allocOp = castOp
                           ? castOp.getSource().getDefiningOp<memref::AllocOp>()
                           : operand.getDefiningOp<memref::AllocOp>();
        if (!allocOp || allocOp->getBlock() != &body ||
            (castOp && castOp->getBlock() != &body))
          continue;

        Operation *writeOp = onlyWrittenToOnceByCrossCore(allocOp);
        Operation *writeInBody =
            writeOp ? body.findAncestorOpInBlock(*writeOp) : nullptr;
        if (!writeInBody || !allocOp->isBeforeInBlock(writeInBody))
          continue;

        bool castUsedByWrite =
            castOp && llvm::is_contained(writeOp->getOperands(), operand);
        if (castUsedByWrite && !castOp->isBeforeInBlock(writeInBody))
          continue;

        SmallVector<Operation *> moveBefore{allocOp};
        SmallVector<Operation *> moveAfter{toTensorOp};
        if (castOp)
          (castUsedByWrite ? moveBefore : moveAfter).push_back(castOp);

        auto collectUsers = [&](Value value, bool valueInMoveBefore) {
          for (Operation *user : value.getUsers()) {
            if (user == writeOp || user == toTensorOp || user == castOp)
              continue;
            if (user->getBlock() != &body || !isa<annotation::MarkOp>(user))
              return false;
            (valueInMoveBefore ? moveBefore : moveAfter).push_back(user);
          }
          return true;
        };
        if (!collectUsers(allocOp, /*valueInMoveBefore=*/true) ||
            (castOp && !collectUsers(castOp, castUsedByWrite)))
          continue;

        DenseSet<Operation *> movingOps;
        movingOps.insert(moveBefore.begin(), moveBefore.end());
        movingOps.insert(moveAfter.begin(), moveAfter.end());
        auto operandsAvailable = [&](Operation *op, bool beforeScope) {
          for (Value value : op->getOperands()) {
            Operation *def = value.getDefiningOp();
            if (def && scopeOp->isAncestor(def) &&
                (!movingOps.contains(def) ||
                 (beforeScope && !llvm::is_contained(moveBefore, def))))
              return false;
          }
          return true;
        };
        if (!llvm::all_of(
                moveBefore,
                [&](Operation *op) { return operandsAvailable(op, true); }) ||
            !llvm::all_of(moveAfter, [&](Operation *op) {
              return operandsAvailable(op, false);
            }))
          continue;

        replacementResults[index] = returnOperand;
        moveBeforeSet.insert(moveBefore.begin(), moveBefore.end());
        moveAfterSet.insert(moveAfter.begin(), moveAfter.end());
      }

      if (replacementResults.empty())
        continue;

      SmallVector<Operation *> moveBefore;
      SmallVector<Operation *> moveAfter;
      for (Operation &op : body) {
        if (moveBeforeSet.contains(&op))
          moveBefore.push_back(&op);
        if (moveAfterSet.contains(&op))
          moveAfter.push_back(&op);
      }

      SmallVector<Type> resultTypes;
      SmallVector<Value> returnValues;
      for (auto [index, value] : llvm::enumerate(returnOp->getOperands())) {
        if (replacementResults.count(index))
          continue;
        resultTypes.push_back(scopeOp.getResult(index).getType());
        returnValues.push_back(value);
      }

      OpBuilder builder(scopeOp);
      auto newScopeOp =
          builder.create<scope::ScopeOp>(scopeOp.getLoc(), resultTypes);
      newScopeOp->setAttrs(scopeOp->getAttrs());
      newScopeOp.getRegion().takeBody(scopeOp.getRegion());
      returnOp->setOperands(returnValues);

      for (Operation *op : moveBefore)
        op->moveBefore(newScopeOp);
      for (Operation *op : moveAfter)
        op->moveBefore(scopeOp);

      unsigned nextResult = 0;
      for (auto [index, result] : llvm::enumerate(scopeOp.getResults())) {
        auto replacement = replacementResults.find(index);
        result.replaceAllUsesWith(replacement != replacementResults.end()
                                      ? replacement->second
                                      : newScopeOp.getResult(nextResult++));
      }
      scopeOp.erase();
    }
  }
}

void PipelineStagingPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (maxPipelineMultibuffer <= 0) {
    getOperation()->emitWarning("[pipeline-staging] max pipeline multibuffer "
                                "should be greater than 0, defaulting to 3.");
    maxPipelineMultibuffer = 3;
  }
  SmallVector<LoopLikeOpInterface> cvPipelinedLoops;
  collectCVPipelinedLoops(funcOp, cvPipelinedLoops);

  for (LoopLikeOpInterface loopOp : cvPipelinedLoops) {
    stageScopes(loopOp, maxPipelineMultibuffer);
    sortScopes(loopOp);
    hoistAllocAndToTensor(loopOp);
  }
}

std::unique_ptr<Pass>
hivm::createPipelineStagingPass(const PipelineStagingOptions &options) {
  return std::make_unique<PipelineStagingPass>(options);
}
} // namespace mlir
