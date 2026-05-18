//===---- DelayedCrossCoreGSS.cpp ---- Delayed Cross-Core GSS -------------===//
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverCodeGen.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIRTranslator.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

#define DEBUG_TYPE "hivm-delayed-cross-core-gss"

namespace mlir {
#define GEN_PASS_DEF_DELAYEDCROSSCOREGSS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm::syncsolver;

namespace mlir {
struct DelayedCrossCoreGSSPass
    : public impl::DelayedCrossCoreGSSBase<DelayedCrossCoreGSSPass> {
  void runOnOperation() override;

private:
  SmallVector<CVTripletKernels> findTriplets(ModuleOp mod) const;

  void crossCoreGssRunOnOperation(ModuleOp moduleOp, const CVTripletKernels &t);
};
} // namespace mlir

SmallVector<CVTripletKernels>
DelayedCrossCoreGSSPass::findTriplets(ModuleOp mod) const {
  SmallVector<func::FuncOp> backupFuncOps;
  mod.walk([&](func::FuncOp funcOp) {
    if (auto coreType = hivm::queryFuncCoreType(funcOp)) {
      if (coreType.value() == hivm::TFuncCoreType::MIX) {
        if (funcOp->hasAttr(hivm::BackupFunctionAttr::name)) {
          backupFuncOps.push_back(funcOp);
        }
      }
    }
  });
  SymbolTable symTable(mod);
  SmallVector<CVTripletKernels> triplets;
  for (func::FuncOp backupFuncOp : backupFuncOps) {
    StringRef funcName = backupFuncOp.getSymName();
    assert(funcName.ends_with(hivm::kFuncBackupSuffix));
    StringRef ogFuncName = funcName.drop_back(hivm::kFuncBackupSuffix.size());

    auto cubeFunc = symTable.lookup<func::FuncOp>(
        (ogFuncName + hivm::kMixFuncAicSuffix).str());
    auto vecFunc = symTable.lookup<func::FuncOp>(
        (ogFuncName + hivm::kMixFuncAivSuffix).str());

    if (!vecFunc || !cubeFunc) {
      backupFuncOp.emitWarning(
          "delayed-cross-core-gss: split kernels not found.");
      continue;
    }

    triplets.push_back(CVTripletKernels(backupFuncOp, vecFunc, cubeFunc));
  }
  return triplets;
}

static OperationBase *getNextOperation(OperationBase *op, bool &goingIn) {
  assert(op != nullptr);
  if (op->parentOp == nullptr) {
    return nullptr;
  }

  if (goingIn) {
    if (auto scopeOp = dyn_cast<Scope>(op)) {
      if (!scopeOp->body.empty()) {
        return scopeOp->body.front().get();
      }
    }
  }

  auto *parentScopeOp = dyn_cast<Scope>(op->parentOp);
  assert(parentScopeOp != nullptr);

  auto &parentBody = parentScopeOp->body;
  if (parentBody.back().get() == op) {
    return getNextOperation(parentScopeOp, goingIn = false);
  }

  auto it = std::find_if(parentBody.begin(), parentBody.end(),
                         [op](const auto &item) { return item.get() == op; });
  assert(it != parentBody.end());
  assert(std::next(it) != parentBody.end());
  return std::next(it)->get();
}

static OperationBase *getNextOperation(OperationBase *op) {
  bool goingIn = true;
  return getNextOperation(op, goingIn);
}

static OperationBase *getPrevOperation(OperationBase *op, bool &goingIn) {
  assert(op != nullptr);
  if (op->parentOp == nullptr) {
    return nullptr;
  }

  if (goingIn) {
    if (auto scopeOp = dyn_cast<Scope>(op)) {
      if (!scopeOp->body.empty()) {
        return scopeOp->body.back().get();
      }
    }
  }

  auto *parentScopeOp = dyn_cast<Scope>(op->parentOp);
  assert(parentScopeOp != nullptr);

  auto &parentBody = parentScopeOp->body;
  if (parentBody.front().get() == op) {
    return getPrevOperation(parentScopeOp, goingIn = false);
  }

  auto it = std::find_if(parentBody.begin(), parentBody.end(),
                         [op](const auto &item) { return item.get() == op; });
  assert(it != parentBody.end());
  assert(it != parentBody.begin());
  return std::prev(it)->get();
}

static OperationBase *getPrevOperation(OperationBase *op) {
  bool goingIn = true;
  return getPrevOperation(op, goingIn);
}

static AnchorInfo getAnchorInfo(IRTranslator *irTranslator, int64_t anchorId1,
                                int64_t anchorId2) {
  auto anchorIt1 = irTranslator->anchorOpMap.find(anchorId1);
  auto anchorIt2 = irTranslator->anchorOpMap.find(anchorId2);
  assert(anchorIt1 != irTranslator->anchorOpMap.end());
  assert(anchorIt2 != irTranslator->anchorOpMap.end());
  Anchor *anchor1 = anchorIt1->second;
  Anchor *anchor2 = anchorIt2->second;
  assert(anchor1 != nullptr && anchor2 != nullptr);
  assert(anchor1->anchorId < anchor2->anchorId);
  return AnchorInfo(anchor1, anchor2);
}

static llvm::SmallVector<RWOperation *>
getAllRWOperationsBetweenAnchors(AnchorInfo anchorInfo) {
  llvm::SmallVector<RWOperation *> collectedOps;
  bool goingIn = true;
  OperationBase *curOp = anchorInfo.anchorBefore;
  while (curOp != anchorInfo.anchorAfter) {
    assert(curOp != nullptr);
    if (auto rwOp = dyn_cast<RWOperation>(curOp)) {
      collectedOps.push_back(rwOp);
    }
    curOp = getNextOperation(curOp, goingIn);
  }
  return collectedOps;
}

static std::unique_ptr<RWOperation>
createMergedRWOperation(OperationBase *parentOp, hivm::TCoreType coreType,
                        const llvm::SmallVector<RWOperation *> &rwOps) {
  std::optional<hivm::PIPE> pipeRead;
  std::optional<hivm::PIPE> pipeWrite;
  llvm::SmallVector<Value> readMemVals;
  llvm::SmallVector<Value> writeMemVals;
  for (auto *rwOp : rwOps) {
    assert(rwOp != nullptr);
    if (!pipeRead.has_value()) {
      pipeRead = rwOp->pipeRead;
    }
    if (!pipeWrite.has_value()) {
      pipeWrite = rwOp->pipeWrite;
    }
    assert(pipeRead.has_value() && pipeWrite.has_value());
    if (pipeRead.value() != rwOp->pipeRead) {
      pipeRead = hivm::PIPE::PIPE_S;
    }
    if (pipeWrite.value() != rwOp->pipeWrite) {
      pipeWrite = hivm::PIPE::PIPE_S;
    }
    llvm::append_range(readMemVals, rwOp->readMemVals);
    llvm::append_range(writeMemVals, rwOp->writeMemVals);
  }
  assert(pipeRead.has_value() && pipeWrite.has_value());
  return std::make_unique<RWOperation>(nullptr, parentOp, coreType,
                                       pipeRead.value(), pipeWrite.value(),
                                       readMemVals, writeMemVals);
}

void DelayedCrossCoreIRTranslator::initIRTranslators() {
  cubeIRTranslator =
      std::make_unique<IRTranslator>(tripletKernels.cubeFuncOp, options);
  vectorIRTranslator =
      std::make_unique<IRTranslator>(tripletKernels.vectorFuncOp, options);
}

std::unique_ptr<OperationBase>
DelayedCrossCoreIRTranslator::buildDelayedFuncIr() {
  auto mixIRTranslatorOptions = options;
  mixIRTranslatorOptions.ignoreNonAnchorOps = true;
  mixIRTranslatorOptions.buildUnrolledSyncIR = false;
  auto mixIRTranslator = std::make_unique<IRTranslator>(
      tripletKernels.mixFuncOp, mixIRTranslatorOptions);

  int64_t anchorIdStart = mixIRTranslator->anchorOpMap.begin()->first;
  int64_t anchorIdEnd = mixIRTranslator->anchorOpMap.rbegin()->first;
  assert(anchorIdEnd - anchorIdStart + 1 ==
         static_cast<int64_t>(mixIRTranslator->anchorOpMap.size()));
  for (int64_t anchorId = anchorIdStart; anchorId < anchorIdEnd; anchorId++) {
    auto mixAnchorInfo =
        getAnchorInfo(mixIRTranslator.get(), anchorId, anchorId + 1);
    auto cubeAnchorInfo =
        getAnchorInfo(cubeIRTranslator.get(), anchorId, anchorId + 1);
    auto vectorAnchorInfo =
        getAnchorInfo(vectorIRTranslator.get(), anchorId, anchorId + 1);

    if (mixAnchorInfo.anchorBefore->parentOp !=
        mixAnchorInfo.anchorAfter->parentOp) {
      int64_t depthBefore = mixAnchorInfo.anchorBefore->getDepth();
      int64_t depthAfter = mixAnchorInfo.anchorAfter->getDepth();
      if (depthBefore < depthAfter) {
        auto *parentOp =
            mixAnchorInfo.anchorAfter->getNthParent(depthAfter - depthBefore);
        assert(parentOp != nullptr);
        if (!parentOp->cubeAnchorInfo.has_value()) {
          parentOp->cubeAnchorInfo = AnchorInfo();
        }
        if (!parentOp->vectorAnchorInfo.has_value()) {
          parentOp->vectorAnchorInfo = AnchorInfo();
        }
        parentOp->cubeAnchorInfo->anchorBefore = cubeAnchorInfo.anchorBefore;
        parentOp->vectorAnchorInfo->anchorBefore =
            vectorAnchorInfo.anchorBefore;
        if (auto mixParentLoopOp = dyn_cast<Loop>(parentOp)) {
          auto cubeParentOp = cubeAnchorInfo.anchorAfter->getNthParent(
              depthAfter - depthBefore);
          assert(cubeParentOp != nullptr);
          auto cubeParentLoopOp = dyn_cast<Loop>(cubeParentOp);
          assert(cubeParentLoopOp != nullptr);
          parentOp->cubeAnchorInfo->loopOp = cubeParentLoopOp;
          loopMap[{mixParentLoopOp, TCoreType::CUBE}] = cubeParentLoopOp;

          auto vectorParentOp = vectorAnchorInfo.anchorAfter->getNthParent(
              depthAfter - depthBefore);
          assert(vectorParentOp != nullptr);
          auto vectorParentLoopOp = dyn_cast<Loop>(vectorParentOp);
          assert(vectorParentLoopOp != nullptr);
          parentOp->vectorAnchorInfo->loopOp = vectorParentLoopOp;
          loopMap[{mixParentLoopOp, TCoreType::VECTOR}] = vectorParentLoopOp;
        }
      }
      if (depthBefore > depthAfter) {
        auto *parentOp =
            mixAnchorInfo.anchorBefore->getNthParent(depthBefore - depthAfter);
        assert(parentOp != nullptr);
        if (!parentOp->cubeAnchorInfo.has_value()) {
          parentOp->cubeAnchorInfo = AnchorInfo();
        }
        if (!parentOp->vectorAnchorInfo.has_value()) {
          parentOp->vectorAnchorInfo = AnchorInfo();
        }
        parentOp->cubeAnchorInfo->anchorAfter = cubeAnchorInfo.anchorAfter;
        parentOp->vectorAnchorInfo->anchorAfter = vectorAnchorInfo.anchorAfter;
      }
      if (depthBefore != depthAfter) {
        auto beforeAnchorNextOp = getNextOperation(mixAnchorInfo.anchorBefore);
        assert(beforeAnchorNextOp != nullptr);
        auto afterAnchorPrevOp = getPrevOperation(mixAnchorInfo.anchorAfter);
        assert(afterAnchorPrevOp != nullptr);
        assert(beforeAnchorNextOp != afterAnchorPrevOp);
        if (auto placeHolderOp = dyn_cast<PlaceHolder>(beforeAnchorNextOp)) {
          placeHolderOp->cubeAnchorInfo =
              AnchorInfo(cubeAnchorInfo.anchorBefore);
          placeHolderOp->vectorAnchorInfo =
              AnchorInfo(vectorAnchorInfo.anchorBefore);
        }
        if (auto placeHolderOp = dyn_cast<PlaceHolder>(afterAnchorPrevOp)) {
          placeHolderOp->cubeAnchorInfo =
              AnchorInfo(cubeAnchorInfo.anchorAfter);
          placeHolderOp->vectorAnchorInfo =
              AnchorInfo(vectorAnchorInfo.anchorAfter);
        }
      }
      continue;
    }

    auto cubeRWOps = getAllRWOperationsBetweenAnchors(cubeAnchorInfo);
    auto vectorRWOps = getAllRWOperationsBetweenAnchors(vectorAnchorInfo);
    if (cubeRWOps.empty() && vectorRWOps.empty()) {
      continue;
    }

    TCoreType coreType;
    if (!cubeRWOps.empty()) {
      coreType = TCoreType::CUBE;
    } else if (!vectorRWOps.empty()) {
      coreType = TCoreType::VECTOR;
    } else {
      coreType = TCoreType::CUBE_AND_VECTOR;
    }

    auto anchorBeforeOp = mixAnchorInfo.anchorBefore;
    auto parentScopeOp = dyn_cast<Scope>(anchorBeforeOp->parentOp);
    assert(parentScopeOp != nullptr);

    SmallVector<RWOperation *> allRWOps;
    llvm::append_range(allRWOps, cubeRWOps);
    llvm::append_range(allRWOps, vectorRWOps);
    auto mergedRWOperation =
        createMergedRWOperation(parentScopeOp, coreType, allRWOps);
    mergedRWOperation->mixAnchorInfo = mixAnchorInfo;
    mergedRWOperation->cubeAnchorInfo = cubeAnchorInfo;
    mergedRWOperation->vectorAnchorInfo = vectorAnchorInfo;

    auto &body = parentScopeOp->body;
    auto it = std::find_if(body.begin(), body.end(),
                           [anchorBeforeOp](const auto &item) {
                             return item.get() == anchorBeforeOp;
                           });
    assert(it != body.end());
    body.insert(it + 1, std::move(mergedRWOperation));
  }

  return std::move(mixIRTranslator->funcIr);
}

void DelayedCrossCoreGSSPass::crossCoreGssRunOnOperation(
    ModuleOp moduleOp, const CVTripletKernels &t) {
  bool isMemBasedArch = hacc::utils::isMemBasedArch(moduleOp);
  bool isRegBasedArch = hacc::utils::isRegBasedArch(moduleOp);
  assert(isMemBasedArch != isRegBasedArch);

  if (this->forceIsRegBased) {
    isMemBasedArch = false;
    isRegBasedArch = true;
  }
  if (this->forceIsMemBased) {
    isMemBasedArch = true;
    isRegBasedArch = false;
  }

  SyncSolverOptions options(SyncMode::CROSS_CORE_SYNC, isMemBasedArch,
                            isRegBasedArch);
  if (this->alwaysUsePipeSAsWaitingPipe) {
    options.alwaysUsePipeSAsWaitingPipe = true;
  }
  if (this->useDifferentMultiBufferFlagIds) {
    options.useDifferentMultiBufferFlagIds = true;
  }

  auto mixIRTranslator =
      std::make_unique<DelayedCrossCoreIRTranslator>(t, options);
  auto cubeIRTranslator = std::move(mixIRTranslator->cubeIRTranslator);
  auto vectorIRTranslator = std::move(mixIRTranslator->vectorIRTranslator);

  LLVM_DEBUG({
    llvm::dbgs() << "before:\n";
    llvm::dbgs() << mixIRTranslator->funcIr->str(0, true) << '\n';
    llvm::dbgs() << cubeIRTranslator->funcIr->str(0, true) << '\n';
    llvm::dbgs() << vectorIRTranslator->funcIr->str(0, true) << '\n';
  });

  auto loopMap = std::move(mixIRTranslator->loopMap);
  auto fixEventIdInfoMultiBufferLoops = [&loopMap](SetWaitOp *setWaitOp,
                                                   hivm::TCoreType coreType) {
    auto &eventIdInfo = setWaitOp->eventIdInfo;
    if (eventIdInfo.multibufferLoop) {
      auto it = loopMap.find({eventIdInfo.multibufferLoop, coreType});
      assert(it != loopMap.end());
      eventIdInfo.multibufferLoop = it->second;
    }
    if (eventIdInfo.multibufferUnrollLoop1) {
      auto it = loopMap.find({eventIdInfo.multibufferUnrollLoop1, coreType});
      assert(it != loopMap.end());
      eventIdInfo.multibufferUnrollLoop1 = it->second;
    }
    if (eventIdInfo.multibufferUnrollLoop2) {
      auto it = loopMap.find({eventIdInfo.multibufferUnrollLoop2, coreType});
      assert(it != loopMap.end());
      eventIdInfo.multibufferUnrollLoop2 = it->second;
    }
  };

  auto mixSolver = std::make_unique<Solver>(std::move(mixIRTranslator));

  DEBUG_WITH_TYPE("gss-print-unrolled-sync-ir", {
    for (auto &occ : mixSolver->syncIr) {
      llvm::dbgs() << std::string(occ->depth, ' ') << occ->op->id << ' '
                   << occ->op->preOrderIndex << ' ' << occ->syncIrIndex << ' '
                   << occ->startIndex << ' ' << occ->endIndex << '\n';
      llvm::dbgs() << occ->op->str(occ->depth, false) << '\n';
    }
  });

  mixSolver->solve();
  auto [mixSyncBeforeMap, mixSyncAfterMap] =
      mixSolver->getBeforeAfterSyncMaps();
  SyncBeforeAfterMap newMixSyncBeforeAfterMap;
  SyncBeforeAfterMap cubeSyncBeforeAfterMap;
  SyncBeforeAfterMap vectorSyncBeforeAfterMap;
  auto &[newMixBeforeMap, newMixAfterMap] = newMixSyncBeforeAfterMap;
  auto &[cubeBeforeMap, cubeAfterMap] = cubeSyncBeforeAfterMap;
  auto &[vectorBeforeMap, vectorAfterMap] = vectorSyncBeforeAfterMap;

  // clone sync ops before
  for (auto &[op, syncOps] : mixSyncBeforeMap) {
    if (syncOps.empty()) {
      continue;
    }
    assert(op->cubeAnchorInfo.has_value());
    assert(op->vectorAnchorInfo.has_value());
    auto cubeAnchorBefore = op->cubeAnchorInfo->anchorBefore;
    auto vectorAnchorBefore = op->vectorAnchorInfo->anchorBefore;
    assert(cubeAnchorBefore != nullptr);
    assert(vectorAnchorBefore != nullptr);
    auto &cubeBeforeAnchorSyncOpsMap = cubeBeforeMap[cubeAnchorBefore];
    auto &vectorBeforeAnchorSyncOpsMap = vectorBeforeMap[vectorAnchorBefore];
    for (auto &syncOp : syncOps) {
      assert(syncOp != nullptr);
      if (auto barrierOp = dyn_cast<BarrierOp>(syncOp.get())) {
        {
          std::unique_ptr<SyncOp> clonedSyncOp = barrierOp->clone(
              cubeAnchorBefore->op, cubeAnchorBefore->parentOp);
          assert(clonedSyncOp != nullptr);
          dyn_cast<BarrierOp>(clonedSyncOp.get())->coreType =
              hivm::TCoreType::CUBE;
          cubeBeforeAnchorSyncOpsMap.push_back(std::move(clonedSyncOp));
        }
        {
          std::unique_ptr<SyncOp> clonedSyncOp = barrierOp->clone(
              vectorAnchorBefore->op, vectorAnchorBefore->parentOp);
          assert(clonedSyncOp != nullptr);
          dyn_cast<BarrierOp>(clonedSyncOp.get())->coreType =
              hivm::TCoreType::VECTOR;
          vectorBeforeAnchorSyncOpsMap.push_back(std::move(clonedSyncOp));
        }
      } else {
        auto setWaitOp = dyn_cast<SetWaitOp>(syncOp.get());
        assert(setWaitOp != nullptr);
        if (setWaitOp->coreType == hivm::TCoreType::CUBE) {
          std::unique_ptr<SyncOp> clonedSyncOp = setWaitOp->clone(
              cubeAnchorBefore->op, cubeAnchorBefore->parentOp);
          assert(clonedSyncOp != nullptr);
          auto clonedSetWaitOp = dyn_cast<SetWaitOp>(clonedSyncOp.get());
          assert(clonedSetWaitOp != nullptr);
          fixEventIdInfoMultiBufferLoops(clonedSetWaitOp,
                                         hivm::TCoreType::CUBE);
          cubeBeforeAnchorSyncOpsMap.push_back(std::move(clonedSyncOp));
        }
        if (setWaitOp->coreType == hivm::TCoreType::VECTOR) {
          std::unique_ptr<SyncOp> clonedSyncOp = setWaitOp->clone(
              vectorAnchorBefore->op, vectorAnchorBefore->parentOp);
          assert(clonedSyncOp != nullptr);
          auto clonedSetWaitOp = dyn_cast<SetWaitOp>(clonedSyncOp.get());
          assert(clonedSetWaitOp != nullptr);
          fixEventIdInfoMultiBufferLoops(clonedSetWaitOp,
                                         hivm::TCoreType::VECTOR);
          vectorBeforeAnchorSyncOpsMap.push_back(std::move(clonedSyncOp));
        }
      }
    }
  }

  // clone sync ops after
  for (auto &[op, syncOps] : mixSyncAfterMap) {
    if (syncOps.empty()) {
      continue;
    }
    assert(op->cubeAnchorInfo.has_value());
    assert(op->vectorAnchorInfo.has_value());
    auto cubeAnchorAfter = op->cubeAnchorInfo->anchorAfter;
    auto vectorAnchorAfter = op->vectorAnchorInfo->anchorAfter;
    auto &cubeAfterAnchorSyncOpsMap = cubeAfterMap[cubeAnchorAfter];
    auto &vectorAfterAnchorSyncOpsMap = vectorAfterMap[vectorAnchorAfter];
    for (auto &syncOp : syncOps) {
      assert(syncOp != nullptr);
      if (auto barrierOp = dyn_cast<BarrierOp>(syncOp.get())) {
        {
          std::unique_ptr<SyncOp> clonedSyncOp =
              barrierOp->clone(cubeAnchorAfter->op, cubeAnchorAfter->parentOp);
          assert(clonedSyncOp != nullptr);
          dyn_cast<BarrierOp>(clonedSyncOp.get())->coreType =
              hivm::TCoreType::CUBE;
          cubeAfterAnchorSyncOpsMap.push_back(std::move(clonedSyncOp));
        }
        {
          std::unique_ptr<SyncOp> clonedSyncOp = barrierOp->clone(
              vectorAnchorAfter->op, vectorAnchorAfter->parentOp);
          assert(clonedSyncOp != nullptr);
          dyn_cast<BarrierOp>(clonedSyncOp.get())->coreType =
              hivm::TCoreType::VECTOR;
          vectorAfterAnchorSyncOpsMap.push_back(std::move(clonedSyncOp));
        }
      } else {
        auto setWaitOp = dyn_cast<SetWaitOp>(syncOp.get());
        assert(setWaitOp != nullptr);
        if (setWaitOp->coreType == hivm::TCoreType::CUBE) {
          std::unique_ptr<SyncOp> clonedSyncOp =
              setWaitOp->clone(cubeAnchorAfter->op, cubeAnchorAfter->parentOp);
          assert(clonedSyncOp != nullptr);
          auto clonedSetWaitOp = dyn_cast<SetWaitOp>(clonedSyncOp.get());
          assert(clonedSetWaitOp != nullptr);
          fixEventIdInfoMultiBufferLoops(clonedSetWaitOp,
                                         hivm::TCoreType::CUBE);
          cubeAfterAnchorSyncOpsMap.push_back(std::move(clonedSyncOp));
        }
        if (setWaitOp->coreType == hivm::TCoreType::VECTOR) {
          std::unique_ptr<SyncOp> clonedSyncOp = setWaitOp->clone(
              vectorAnchorAfter->op, vectorAnchorAfter->parentOp);
          assert(clonedSyncOp != nullptr);
          auto clonedSetWaitOp = dyn_cast<SetWaitOp>(clonedSyncOp.get());
          assert(clonedSetWaitOp != nullptr);
          fixEventIdInfoMultiBufferLoops(clonedSetWaitOp,
                                         hivm::TCoreType::VECTOR);
          vectorAfterAnchorSyncOpsMap.push_back(std::move(clonedSyncOp));
        }
      }
    }
  }

  // move sync ops to before/after anchors
  for (auto &[op, syncOps] : mixSyncBeforeMap) {
    if (syncOps.empty()) {
      continue;
    }
    if (!isa<RWOperation>(op)) {
      newMixBeforeMap[op] = std::move(syncOps);
      continue;
    }
    assert(op->mixAnchorInfo.has_value());
    auto mixAnchorBefore = op->mixAnchorInfo->anchorBefore;
    auto &newMixBeforeAnchorSyncOpsMap = newMixBeforeMap[mixAnchorBefore];
    for (auto &syncOp : syncOps) {
      assert(syncOp != nullptr);
      std::unique_ptr<SyncOp> clonedSyncOp =
          syncOp->clone(mixAnchorBefore->op, mixAnchorBefore->parentOp);
      assert(clonedSyncOp != nullptr);
      newMixBeforeAnchorSyncOpsMap.push_back(std::move(clonedSyncOp));
    }
  }
  for (auto &[op, syncOps] : mixSyncAfterMap) {
    if (syncOps.empty()) {
      continue;
    }
    if (!isa<RWOperation>(op)) {
      newMixAfterMap[op] = std::move(syncOps);
      continue;
    }
    assert(op->mixAnchorInfo.has_value());
    auto mixAnchorAfter = op->mixAnchorInfo->anchorAfter;
    auto &newMixAfterAnchorSyncOpsMap = newMixAfterMap[mixAnchorAfter];
    for (auto &syncOp : syncOps) {
      assert(syncOp != nullptr);
      std::unique_ptr<SyncOp> clonedSyncOp =
          syncOp->clone(mixAnchorAfter->op, mixAnchorAfter->parentOp);
      assert(clonedSyncOp != nullptr);
      newMixAfterAnchorSyncOpsMap.push_back(std::move(clonedSyncOp));
    }
  }

  CodeGenerator mixCodeGen(options);
  mixCodeGen.syncMapBefore = std::move(newMixSyncBeforeAfterMap.first);
  mixCodeGen.syncMapAfter = std::move(newMixSyncBeforeAfterMap.second);
  mixCodeGen.funcOp = mixSolver->funcOp;
  mixCodeGen.funcIr = std::move(mixSolver->funcIr);
  mixCodeGen.generateResultOps();
  LLVM_DEBUG({
    mixCodeGen.generateFuncIrResultOps();
    llvm::dbgs() << "after-mix:\n" << mixCodeGen.funcIr->str(0, true) << '\n';
  });

  CodeGenerator cubeCodeGen(options);
  cubeCodeGen.syncMapBefore = std::move(cubeSyncBeforeAfterMap.first);
  cubeCodeGen.syncMapAfter = std::move(cubeSyncBeforeAfterMap.second);
  cubeCodeGen.funcOp = cubeIRTranslator->funcOp;
  cubeCodeGen.funcIr = std::move(cubeIRTranslator->funcIr);

  CodeGenerator vectorCodeGen(options);
  vectorCodeGen.syncMapBefore = std::move(vectorSyncBeforeAfterMap.first);
  vectorCodeGen.syncMapAfter = std::move(vectorSyncBeforeAfterMap.second);
  vectorCodeGen.funcOp = vectorIRTranslator->funcOp;
  vectorCodeGen.funcIr = std::move(vectorIRTranslator->funcIr);
  cubeCodeGen.generateResultOps();
  vectorCodeGen.generateResultOps();
  LLVM_DEBUG({
    cubeCodeGen.generateFuncIrResultOps();
    vectorCodeGen.generateFuncIrResultOps();
    llvm::dbgs() << "after-cube:\n" << cubeCodeGen.funcIr->str(0, true) << '\n';
    llvm::dbgs() << "after-vector:\n"
                 << vectorCodeGen.funcIr->str(0, true) << '\n';
  });
}

template <typename SyncOp>
struct EraseSyncOpPattern : public OpRewritePattern<SyncOp> {
  using OpRewritePattern<SyncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SyncOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

void DelayedCrossCoreGSSPass::runOnOperation() {
  ModuleOp mod = getOperation();
  bool isMemBasedArch = hacc::utils::isMemBasedArch(mod);
  bool isRegBasedArch = hacc::utils::isRegBasedArch(mod);
  assert(isMemBasedArch != isRegBasedArch);

  auto triplets = findTriplets(mod);
  for (CVTripletKernels &t : triplets) {

    for (auto funcOp : {t.mixFuncOp, t.cubeFuncOp, t.vectorFuncOp}) {
      auto *ctx = mod->getContext();
      RewritePatternSet patterns(ctx);
      patterns.add<EraseSyncOpPattern<hivm::SyncBlockSetOp>,
                   EraseSyncOpPattern<hivm::SyncBlockWaitOp>,
                   EraseSyncOpPattern<hivm::PipeBarrierOp>>(ctx);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // run cross-core gss
    crossCoreGssRunOnOperation(mod, t);
  }
}

std::unique_ptr<Pass> mlir::hivm::createDelayedCrossCoreGSSPass() {
  return std::make_unique<DelayedCrossCoreGSSPass>();
}
