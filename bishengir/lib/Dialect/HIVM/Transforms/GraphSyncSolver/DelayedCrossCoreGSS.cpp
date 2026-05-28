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

// Delayed cross-core auto-sync.
//
// Sister pass to the in-place GraphSyncSolver cross-core flow. The in-place
// flow inserts sync ops on the mixed kernel before split-mix-kernel, which
// means later memory rewrites (notably plan-memory) can invalidate or
// understate the hazards the solver saw. The delayed flow defers solving
// until *after* those rewrites, but still uses the mixed kernel as the
// analysis source so the solver retains its high-quality hazard reasoning.
//
// The bridge is a backup of the mixed function plus an anchor model
// (InsertAnchorsAndBackup pass): each consecutive anchor pair (k, k+1)
// defines an interval whose contents on the live cube and vector functions
// are merged into a single synthetic RW operation on the backup. Solving
// this synthesized IR yields sync ops that are then cloned into the live
// cube, vector, and backup functions at the matching anchors.
namespace mlir {
struct DelayedCrossCoreGSSPass
    : public impl::DelayedCrossCoreGSSBase<DelayedCrossCoreGSSPass> {
  void runOnOperation() override;

private:
  // Discover (mix-backup, vector, cube) triplets. The pass operates one
  // triplet at a time.
  SmallVector<CVTripletKernels> findTriplets(ModuleOp mod) const;

  // Run cross-core sync solving for a single triplet and write the resulting
  // sync ops back into the live IR.
  void crossCoreGssRunOnOperation(ModuleOp moduleOp, const CVTripletKernels &t);
};
} // namespace mlir

SmallVector<CVTripletKernels>
DelayedCrossCoreGSSPass::findTriplets(ModuleOp mod) const {
  // Backups are mix functions tagged by InsertAnchorsAndBackup; their original
  // counterparts have been replaced by the cube and vector split kernels by
  // the time this pass runs.
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
    // The backup name is the original mix-function name plus the backup
    // suffix; the split kernels keep that original name with the cube/vector
    // suffix appended by SplitMixKernel.
    StringRef ogFuncName = funcName.drop_back(hivm::kFuncBackupSuffix.size());

    auto cubeFunc = symTable.lookup<func::FuncOp>(
        (ogFuncName + hivm::kMixFuncAicSuffix).str());
    auto vecFunc = symTable.lookup<func::FuncOp>(
        (ogFuncName + hivm::kMixFuncAivSuffix).str());

    if (!vecFunc || !cubeFunc) {
      // A backup without a complete split pair indicates an upstream pipeline
      // mistake; warn and skip rather than fail the pipeline.
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
    goingIn = false;
    return parentScopeOp;
  }
  goingIn = true;
  auto it = std::find_if(parentBody.begin(), parentBody.end(),
                         [op](const auto &item) { return item.get() == op; });
  assert(it != parentBody.end());
  assert(std::next(it) != parentBody.end());
  return std::next(it)->get();
}

static OperationBase *getNextOperation(OperationBase *op) {
  bool goingIn = false;
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
    goingIn = false;
    return parentScopeOp;
  }
  goingIn = true;
  auto it = std::find_if(parentBody.begin(), parentBody.end(),
                         [op](const auto &item) { return item.get() == op; });
  assert(it != parentBody.end());
  assert(it != parentBody.begin());
  return std::prev(it)->get();
}

static OperationBase *getPrevOperation(OperationBase *op) {
  bool goingIn = false;
  return getPrevOperation(op, goingIn);
}

// Look up the SyncSolver IR ops carrying the requested anchor ids on a single
// kernel side. Anchors are registered while building the SyncSolver IR
// (anchorOpMap), so for any pair of valid ids both must resolve.
static AnchorInfo getAnchorInfo(IRTranslator *irTranslator, int64_t anchorId1,
                                int64_t anchorId2) {
  auto anchorIt1 = irTranslator->anchorOpMap.find(anchorId1);
  auto anchorIt2 = irTranslator->anchorOpMap.find(anchorId2);
  assert(anchorIt1 != irTranslator->anchorOpMap.end());
  assert(anchorIt2 != irTranslator->anchorOpMap.end());
  auto *anchor1 = anchorIt1->second;
  auto *anchor2 = anchorIt2->second;
  assert(anchor1 != nullptr && anchor2 != nullptr);
  return AnchorInfo(anchor1, anchor2);
}

// Walk the open interval (anchorBefore, anchorAfter) and collect every
// RWOperation reachable in pre-order, recursing into nested scopes.
static llvm::SmallVector<RWOperation *>
getAllRWOperationsBetweenAnchors(AnchorInfo anchorInfo) {
  llvm::SmallVector<RWOperation *> collectedOps;
  OperationBase *curOp = anchorInfo.anchorBefore;
  curOp = getNextOperation(curOp);
  bool goingIn = true;
  while (curOp != anchorInfo.anchorAfter) {
    assert(curOp != nullptr);
    if (auto rwOp = dyn_cast<RWOperation>(curOp)) {
      collectedOps.push_back(rwOp);
    }
    curOp = getNextOperation(curOp, goingIn);
  }
  return collectedOps;
}

// Collapse all RW ops collected from one interval (across both cube and
// vector sides) into a single synthetic RW op that the solver can process as
// one node. Conservative merge policy:
//   - read/write memory values are concatenated.
//   - if the input ops disagree on a pipe, fall back to the scalar pipe so
//     the solver inserts the most general waiting pipe.
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
    LLVM_DEBUG({
      if (pipeRead.value() != rwOp->pipeRead ||
          pipeWrite.value() != rwOp->pipeWrite) {
        llvm::dbgs() << "createMergedRWOperation: unexpected rw ops with "
                        "different read/write pipes, check sync-block-ops with "
                        "src/dst pipe_s.\n";
      }
    });
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
  // The cube and vector translators are full SyncSolver IR translators on the
  // live split kernels. We hold them so the merged interval RW ops can carry
  // pointers back to the actual anchor ops on each side, which is what
  // codegen later uses to place sync ops on the live IR.
  cubeIRTranslator =
      std::make_unique<IRTranslator>(tripletKernels.cubeFuncOp, options);
  vectorIRTranslator =
      std::make_unique<IRTranslator>(tripletKernels.vectorFuncOp, options);
}

// Build the synthetic mix-side IR consumed by the solver.
//
// Strategy: walk anchor ids in order on the mix side; for each consecutive
// pair (k, k+1) match it against the same pair on the cube and vector sides.
// Three cases:
//   1. The mix interval lies entirely within a single scope on every side -
//      collect RW ops from the cube and vector sides between the anchors and
//      synthesize one merged RW op that records anchor pointers for all three
//      sides. Insert it right after the mix `anchorBefore`.
//   2. The interval crosses a scope boundary on the mix side - record the
//      partial anchor information on the *parent* scope/loop op so codegen
//      can place sync ops at the correct nesting level on each kernel.
//   3. Both sides empty - skip; no hazard to model.
std::unique_ptr<OperationBase>
DelayedCrossCoreIRTranslator::buildDelayedFuncIr() {
  // The mix translator only needs anchor positions, not the full RW IR; the
  // RW data comes from the cube/vector translators. Build it cheaply.
  auto mixIRTranslatorOptions = options;
  mixIRTranslatorOptions.ignoreNonAnchorOps = true;
  mixIRTranslatorOptions.buildUnrolledSyncIR = false;
  auto mixIRTranslator = std::make_unique<IRTranslator>(
      tripletKernels.mixFuncOp, mixIRTranslatorOptions);

  // Anchor ids are dense within each mix function, so a simple range walk
  // visits every interval.
  int64_t anchorIdStart = mixIRTranslator->anchorOpMap.begin()->first;
  int64_t anchorIdEnd = mixIRTranslator->anchorOpMap.rbegin()->first;
  assert(anchorIdEnd - anchorIdStart + 1 ==
         static_cast<int64_t>(mixIRTranslator->anchorOpMap.size()));

  auto createRWOperation = [&](int64_t anchorId, AnchorInfo mixAnchorInfo,
                               AnchorInfo cubeAnchorInfo,
                               AnchorInfo vectorAnchorInfo) {
    auto cubeRWOps = getAllRWOperationsBetweenAnchors(cubeAnchorInfo);
    auto vectorRWOps = getAllRWOperationsBetweenAnchors(vectorAnchorInfo);
    if (cubeRWOps.empty() && vectorRWOps.empty()) {
      return;
    }

    // Synthetic core type:
    //   - cube-only ops -> CUBE
    //   - vector-only ops -> VECTOR
    //   - both sides non-empty -> CUBE_AND_VECTOR (handled by the else-branch
    //     defensively; in practice the third case is the common cross-core
    //     hazard).
    TCoreType coreType;
    if (!cubeRWOps.empty()) {
      coreType = TCoreType::CUBE;
    } else if (!vectorRWOps.empty()) {
      coreType = TCoreType::VECTOR;
    } else {
      coreType = TCoreType::CUBE_AND_VECTOR;
    }

    LLVM_DEBUG({
      if (coreType == TCoreType::CUBE_AND_VECTOR) {
        llvm::dbgs() << "createRWOperation: unexpected for both cube and "
                        "vector kernels to have rw ops between given anchors, "
                        "check anchor-id="
                     << anchorId << "\n";
      }
    });

    auto anchorBeforeOp = mixAnchorInfo.anchorBefore;
    auto anchorAfterOp = mixAnchorInfo.anchorAfter;
    assert(anchorBeforeOp->parentOp == anchorAfterOp->parentOp);
    auto parentScopeOp = dyn_cast<Scope>(anchorBeforeOp->parentOp);
    assert(parentScopeOp != nullptr);

    SmallVector<RWOperation *> allRWOps;
    llvm::append_range(allRWOps, cubeRWOps);
    llvm::append_range(allRWOps, vectorRWOps);
    auto mergedRWOperation =
        createMergedRWOperation(parentScopeOp, coreType, allRWOps);
    // Carry back the live anchors on every side so codegen can splice the
    // generated sync ops into all three kernels at the right point.
    mergedRWOperation->mixAnchorInfo = mixAnchorInfo;
    mergedRWOperation->cubeAnchorInfo = cubeAnchorInfo;
    mergedRWOperation->vectorAnchorInfo = vectorAnchorInfo;

    // Insert the synthetic op directly after `anchorBefore` so the solver
    // sees it in the position the original interval would occupy.
    auto &body = parentScopeOp->body;
    auto it = std::find_if(body.begin(), body.end(),
                           [anchorBeforeOp](const auto &item) {
                             return item.get() == anchorBeforeOp;
                           });
    assert(it != body.end());
    body.insert(it + 1, std::move(mergedRWOperation));
  };

  auto createRWOperationBlockBefore = [&](int64_t anchorId,
                                          AnchorInfo mixAnchorInfo,
                                          AnchorInfo cubeAnchorInfo,
                                          AnchorInfo vectorAnchorInfo) {
    assert(isa<Anchor>(mixAnchorInfo.anchorBefore));
    int64_t depthBefore = mixAnchorInfo.anchorBefore->getDepth();
    int64_t depthAfter = mixAnchorInfo.anchorAfter->getDepth();
    assert(depthBefore < depthAfter);
    mixAnchorInfo.anchorAfter =
        mixAnchorInfo.anchorAfter->getNthParent(depthAfter - depthBefore);
    cubeAnchorInfo.anchorAfter =
        cubeAnchorInfo.anchorAfter->getNthParent(depthAfter - depthBefore);
    vectorAnchorInfo.anchorAfter =
        vectorAnchorInfo.anchorAfter->getNthParent(depthAfter - depthBefore);
    if (auto placeHolderOp = dyn_cast_if_present<PlaceHolder>(
            getPrevOperation(mixAnchorInfo.anchorAfter))) {
      mixAnchorInfo.anchorAfter = placeHolderOp;
    }
    if (auto placeHolderOp = dyn_cast_if_present<PlaceHolder>(
            getPrevOperation(cubeAnchorInfo.anchorAfter))) {
      cubeAnchorInfo.anchorAfter = placeHolderOp;
    }
    if (auto placeHolderOp = dyn_cast_if_present<PlaceHolder>(
            getPrevOperation(vectorAnchorInfo.anchorAfter))) {
      vectorAnchorInfo.anchorAfter = placeHolderOp;
    }
    createRWOperation(anchorId, mixAnchorInfo, cubeAnchorInfo,
                      vectorAnchorInfo);
  };

  auto createRWOperationBlockAfter = [&](int64_t anchorId,
                                         AnchorInfo mixAnchorInfo,
                                         AnchorInfo cubeAnchorInfo,
                                         AnchorInfo vectorAnchorInfo) {
    assert(isa<Anchor>(mixAnchorInfo.anchorAfter));
    int64_t depthBefore = mixAnchorInfo.anchorBefore->getDepth();
    int64_t depthAfter = mixAnchorInfo.anchorAfter->getDepth();
    assert(depthBefore > depthAfter);
    mixAnchorInfo.anchorBefore =
        mixAnchorInfo.anchorBefore->getNthParent(depthBefore - depthAfter);
    cubeAnchorInfo.anchorBefore =
        cubeAnchorInfo.anchorBefore->getNthParent(depthBefore - depthAfter);
    vectorAnchorInfo.anchorBefore =
        vectorAnchorInfo.anchorBefore->getNthParent(depthBefore - depthAfter);
    if (auto placeHolderOp = dyn_cast_if_present<PlaceHolder>(
            getNextOperation(mixAnchorInfo.anchorBefore))) {
      mixAnchorInfo.anchorBefore = placeHolderOp;
    }
    if (auto placeHolderOp = dyn_cast_if_present<PlaceHolder>(
            getNextOperation(cubeAnchorInfo.anchorBefore))) {
      cubeAnchorInfo.anchorBefore = placeHolderOp;
    }
    if (auto placeHolderOp = dyn_cast_if_present<PlaceHolder>(
            getNextOperation(vectorAnchorInfo.anchorBefore))) {
      vectorAnchorInfo.anchorBefore = placeHolderOp;
    }
    createRWOperation(anchorId, mixAnchorInfo, cubeAnchorInfo,
                      vectorAnchorInfo);
  };

  auto createRWOperationBlockBegin =
      [&](int64_t anchorId, AnchorInfo mixAnchorInfo, AnchorInfo cubeAnchorInfo,
          AnchorInfo vectorAnchorInfo) {
        assert(isa<Anchor>(mixAnchorInfo.anchorAfter));
        int64_t depthBefore = mixAnchorInfo.anchorBefore->getDepth();
        int64_t depthAfter = mixAnchorInfo.anchorAfter->getDepth();
        assert(depthBefore < depthAfter);
        mixAnchorInfo.anchorBefore =
            dyn_cast<Scope>(mixAnchorInfo.anchorAfter->parentOp)
                ->body.front()
                .get();
        cubeAnchorInfo.anchorBefore =
            dyn_cast<Scope>(cubeAnchorInfo.anchorAfter->parentOp)
                ->body.front()
                .get();
        vectorAnchorInfo.anchorBefore =
            dyn_cast<Scope>(vectorAnchorInfo.anchorAfter->parentOp)
                ->body.front()
                .get();
        createRWOperation(anchorId, mixAnchorInfo, cubeAnchorInfo,
                          vectorAnchorInfo);
      };

  auto createRWOperationBlockEnd =
      [&](int64_t anchorId, AnchorInfo mixAnchorInfo, AnchorInfo cubeAnchorInfo,
          AnchorInfo vectorAnchorInfo) {
        assert(isa<Anchor>(mixAnchorInfo.anchorBefore));
        int64_t depthBefore = mixAnchorInfo.anchorBefore->getDepth();
        int64_t depthAfter = mixAnchorInfo.anchorAfter->getDepth();
        assert(depthBefore > depthAfter);
        mixAnchorInfo.anchorAfter =
            dyn_cast<Scope>(mixAnchorInfo.anchorBefore->parentOp)
                ->body.back()
                .get();
        cubeAnchorInfo.anchorAfter =
            dyn_cast<Scope>(cubeAnchorInfo.anchorBefore->parentOp)
                ->body.back()
                .get();
        vectorAnchorInfo.anchorAfter =
            dyn_cast<Scope>(vectorAnchorInfo.anchorBefore->parentOp)
                ->body.back()
                .get();
        createRWOperation(anchorId, mixAnchorInfo, cubeAnchorInfo,
                          vectorAnchorInfo);
      };

  for (int64_t anchorId = anchorIdStart; anchorId < anchorIdEnd; anchorId++) {
    auto mixAnchorInfo =
        getAnchorInfo(mixIRTranslator.get(), anchorId, anchorId + 1);
    auto cubeAnchorInfo =
        getAnchorInfo(cubeIRTranslator.get(), anchorId, anchorId + 1);
    auto vectorAnchorInfo =
        getAnchorInfo(vectorIRTranslator.get(), anchorId, anchorId + 1);

    if (mixAnchorInfo.anchorBefore->parentOp ==
        mixAnchorInfo.anchorAfter->parentOp) {
      createRWOperation(anchorId, mixAnchorInfo, cubeAnchorInfo,
                        vectorAnchorInfo);
      continue;
    }

    int64_t depthBefore = mixAnchorInfo.anchorBefore->getDepth();
    int64_t depthAfter = mixAnchorInfo.anchorAfter->getDepth();
    if (depthBefore < depthAfter) {
      auto *mixParentOp =
          mixAnchorInfo.anchorAfter->getNthParent(depthAfter - depthBefore);
      auto *cubeParentOp =
          cubeAnchorInfo.anchorAfter->getNthParent(depthAfter - depthBefore);
      auto *vectorParentOp =
          vectorAnchorInfo.anchorAfter->getNthParent(depthAfter - depthBefore);
      assert(mixParentOp && cubeParentOp && vectorParentOp);

      mixParentOp->cubeAnchorInfo = AnchorInfo(cubeParentOp);
      mixParentOp->vectorAnchorInfo = AnchorInfo(vectorParentOp);

      if (isa<Anchor>(mixAnchorInfo.anchorBefore)) {
        createRWOperationBlockBefore(anchorId, mixAnchorInfo, cubeAnchorInfo,
                                     vectorAnchorInfo);
        if (auto mixPlaceHolderOp = dyn_cast_if_present<PlaceHolder>(
                getPrevOperation(mixParentOp))) {
          auto *cubePlaceHolderOp = getPrevOperation(cubeParentOp);
          auto *vectorPlaceHolderOp = getPrevOperation(vectorParentOp);
          assert(isa<PlaceHolder>(cubePlaceHolderOp));
          assert(isa<PlaceHolder>(vectorPlaceHolderOp));
          mixPlaceHolderOp->cubeAnchorInfo = AnchorInfo(cubePlaceHolderOp);
          mixPlaceHolderOp->vectorAnchorInfo = AnchorInfo(vectorPlaceHolderOp);
        }
      }

      if (isa<Anchor>(mixAnchorInfo.anchorAfter)) {
        createRWOperationBlockBegin(anchorId, mixAnchorInfo, cubeAnchorInfo,
                                    vectorAnchorInfo);
        auto *mixBlockFrontOp =
            dyn_cast<Scope>(mixAnchorInfo.anchorAfter->parentOp)
                ->body.front()
                .get();
        if (auto mixPlaceHolderOp = dyn_cast<PlaceHolder>(mixBlockFrontOp)) {
          auto *cubePlaceHolderOp =
              dyn_cast<Scope>(vectorAnchorInfo.anchorAfter->parentOp)
                  ->body.front()
                  .get();
          auto *vectorPlaceHolderOp =
              dyn_cast<Scope>(cubeAnchorInfo.anchorAfter->parentOp)
                  ->body.front()
                  .get();
          assert(isa<PlaceHolder>(cubePlaceHolderOp));
          assert(isa<PlaceHolder>(vectorPlaceHolderOp));
          mixPlaceHolderOp->cubeAnchorInfo = AnchorInfo(cubePlaceHolderOp);
          mixPlaceHolderOp->vectorAnchorInfo = AnchorInfo(vectorPlaceHolderOp);
        }
      }
    }
    if (depthBefore > depthAfter) {
      auto *mixParentOp =
          mixAnchorInfo.anchorBefore->getNthParent(depthBefore - depthAfter);
      auto *cubeParentOp =
          cubeAnchorInfo.anchorBefore->getNthParent(depthBefore - depthAfter);
      auto *vectorParentOp =
          vectorAnchorInfo.anchorBefore->getNthParent(depthBefore - depthAfter);
      assert(mixParentOp && cubeParentOp && vectorParentOp);

      mixParentOp->cubeAnchorInfo = AnchorInfo(cubeParentOp);
      mixParentOp->vectorAnchorInfo = AnchorInfo(vectorParentOp);

      if (isa<Anchor>(mixAnchorInfo.anchorAfter)) {
        createRWOperationBlockAfter(anchorId, mixAnchorInfo, cubeAnchorInfo,
                                    vectorAnchorInfo);
        if (auto mixPlaceHolderOp = dyn_cast_if_present<PlaceHolder>(
                getNextOperation(mixParentOp))) {
          auto *cubePlaceHolderOp = getNextOperation(cubeParentOp);
          auto *vectorPlaceHolderOp = getNextOperation(vectorParentOp);
          assert(isa<PlaceHolder>(cubePlaceHolderOp));
          assert(isa<PlaceHolder>(vectorPlaceHolderOp));
          mixPlaceHolderOp->cubeAnchorInfo = AnchorInfo(cubePlaceHolderOp);
          mixPlaceHolderOp->vectorAnchorInfo = AnchorInfo(vectorPlaceHolderOp);
        }
      }

      if (isa<Anchor>(mixAnchorInfo.anchorBefore)) {
        createRWOperationBlockEnd(anchorId, mixAnchorInfo, cubeAnchorInfo,
                                  vectorAnchorInfo);
        auto *mixBlockBackOp =
            dyn_cast<Scope>(mixAnchorInfo.anchorBefore->parentOp)
                ->body.back()
                .get();
        if (auto mixPlaceHolderOp = dyn_cast<PlaceHolder>(mixBlockBackOp)) {
          auto *cubePlaceHolderOp =
              dyn_cast<Scope>(vectorAnchorInfo.anchorBefore->parentOp)
                  ->body.back()
                  .get();
          auto *vectorPlaceHolderOp =
              dyn_cast<Scope>(cubeAnchorInfo.anchorBefore->parentOp)
                  ->body.back()
                  .get();
          assert(isa<PlaceHolder>(cubePlaceHolderOp));
          assert(isa<PlaceHolder>(vectorPlaceHolderOp));
          mixPlaceHolderOp->cubeAnchorInfo = AnchorInfo(cubePlaceHolderOp);
          mixPlaceHolderOp->vectorAnchorInfo = AnchorInfo(vectorPlaceHolderOp);
        }
      }
    }
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

  // Build the synthetic mix IR (consuming cube and vector translators in the
  // process) and hand its translators off so we can talk to live IR later.
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

  // The solver decides set/wait pairs in terms of the mix-side loops; codegen
  // needs the cube/vector-side loop counterparts when materializing them.
  auto fixEventIdInfoMultiBufferLoops = [](SetWaitOp *setWaitOp,
                                           hivm::TCoreType coreType) {
    auto &eventIdInfo = setWaitOp->eventIdInfo;
    auto fixLoop = [coreType](Loop *loopOp) -> Loop * {
      if (!loopOp) {
        return nullptr;
      }
      if (coreType == hivm::TCoreType::CUBE) {
        assert(loopOp->cubeAnchorInfo.has_value());
        auto *fixedLoopOp =
            dyn_cast<Loop>(loopOp->cubeAnchorInfo->anchorBefore);
        assert(fixedLoopOp != nullptr);
        return fixedLoopOp;
      } else if (coreType == hivm::TCoreType::VECTOR) {
        assert(loopOp->vectorAnchorInfo.has_value());
        auto *fixedLoopOp =
            dyn_cast<Loop>(loopOp->vectorAnchorInfo->anchorBefore);
        assert(fixedLoopOp != nullptr);
        return fixedLoopOp;
      }
      return loopOp;
    };
    if (eventIdInfo.multibufferLoop) {
      eventIdInfo.multibufferLoop = fixLoop(eventIdInfo.multibufferLoop);
    }
    if (eventIdInfo.multibufferUnrollLoop1) {
      eventIdInfo.multibufferUnrollLoop1 =
          fixLoop(eventIdInfo.multibufferUnrollLoop1);
    }
    if (eventIdInfo.multibufferUnrollLoop2) {
      eventIdInfo.multibufferUnrollLoop2 =
          fixLoop(eventIdInfo.multibufferUnrollLoop2);
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

  // Solve once on the synthetic mix IR. The result is keyed on synthetic
  // ops whose AnchorInfos point back to the live anchors on each side; the
  // remainder of this function fans those decisions out to all three
  // kernels.
  mixSolver->solve();
  auto [mixSyncBeforeMap, mixSyncAfterMap] =
      mixSolver->getBeforeAfterSyncMaps();
  SyncBeforeAfterMap newMixSyncBeforeAfterMap;
  SyncBeforeAfterMap cubeSyncBeforeAfterMap;
  SyncBeforeAfterMap vectorSyncBeforeAfterMap;
  auto &[newMixBeforeMap, newMixAfterMap] = newMixSyncBeforeAfterMap;
  auto &[cubeBeforeMap, cubeAfterMap] = cubeSyncBeforeAfterMap;
  auto &[vectorBeforeMap, vectorAfterMap] = vectorSyncBeforeAfterMap;

  // Stage 1: clone "sync before" decisions onto the cube and vector kernels.
  // Barriers fan out to both sides; set/wait ops route to the side matching
  // their core type. Multibuffer event-id loop references are translated
  // from mix-side loops to live-side loops via fixEventIdInfoMultiBufferLoops.
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

  // Stage 2: clone "sync after" decisions onto the cube and vector kernels.
  // Symmetric to stage 1 but routes to anchorAfter instead of anchorBefore.
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

  // Stage 3: relocate the mix-side decisions onto the mix anchors.
  //
  // The solver attached sync ops to the synthetic merged RW operations, but
  // those synthetic ops will not exist in the final IR. For RWOperation
  // entries we re-anchor to the mix-side anchor; for non-RW entries (e.g.
  // sync ops attached to scopes/loops directly) we keep the original key.
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

    // Erase legacy sync ops on the mix backup and both split kernels so the
    // solver below sees a clean slate.
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

    crossCoreGssRunOnOperation(mod, t);
  }
}

std::unique_ptr<Pass> mlir::hivm::createDelayedCrossCoreGSSPass() {
  return std::make_unique<DelayedCrossCoreGSSPass>();
}
