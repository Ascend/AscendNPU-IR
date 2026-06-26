//===--- InsertAnchorsAndBackup.cpp ----------------------------*- C++ -*--===//
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "hivm-insert-anchors-and-backup"

namespace mlir {
#define GEN_PASS_DEF_DELAYEDCROSSCOREGSS
#define GEN_PASS_DEF_INSERTANCHORSANDBACKUP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

// Pre-split instrumentation pass for the delayed cross-core GSS flow.
//
// Two responsibilities:
//   1. Stamp each mix function with monotonically-numbered anchors so the
//      delayed pass can recover interval boundaries after split-mix-kernel
//      and plan-memory rewrite the IR.
//   2. Clone each mix function into a backup that the delayed pass consumes
//      as the analysis source. The backup is marked so other passes opt out
//      of running on it via `annotation.filter_passes`.
//
// `cleanup` is a self-inverse mode that strips both artifacts before lowering.
struct InsertAnchorsAndBackupPass
    : public impl::InsertAnchorsAndBackupBase<InsertAnchorsAndBackupPass> {
  explicit InsertAnchorsAndBackupPass(
      const InsertAnchorsAndBackupOptions &options)
      : InsertAnchorsAndBackupBase(options) {}

  ~InsertAnchorsAndBackupPass() override = default;

  void runOnOperation() override;

private:
  // Insert a real `hivm.anchor` op adjacent to `op`. Used for non
  // region-bearing ops where it is safe to splice an additional sibling op into
  // the block.
  void insertAnchor(Operation *op, OpBuilder &builder, int64_t &nextAnchorId,
                    bool insertBefore = false);

  void insertAnchorsInBlock(Block &block, OpBuilder &builder,
                            int64_t &nextAnchorId,
                            bool onlyInsertFrontBack = false);

  func::FuncOp backupFunc(func::FuncOp funcOp);
  func::FuncOp
  getOrCreateBackupFunc(func::FuncOp funcOp,
                        llvm::DenseMap<Operation *, func::FuncOp> &backupFuncs);
  // Make the backup a closed graph: any `func.call` inside the backup is
  // redirected to the backup of its callee, recursively cloning callees on
  // demand. Without this the backup would still reach into the live functions
  // and pick up later rewrites through symbol references.
  void retargetCallsToBackupFuncs(
      func::FuncOp backupFuncOp,
      llvm::DenseMap<Operation *, func::FuncOp> &backupFuncs);

  void eraseAllAnchors(func::FuncOp funcOp);

  void eraseBackupFuncOps(ModuleOp mod);

  bool ignoreOp(Operation *op) const {
    return isa<hivm::SyncBlockOp, hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp,
               hivm::SetFlagOp, hivm::WaitFlagOp, hivm::PipeBarrierOp>(op);
  }

  bool isBlockFrontOp(Operation *op) const {
    return op == &(op->getBlock()->front());
  }

  bool isBlockBackOp(Operation *op) const {
    return op == &(op->getBlock()->back());
  }

  bool isCodeStructureOp(Operation *op) const {
    if (op->getNumRegions() > 0) {
      return true;
    }
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      return true;
    }
    return false;
  }

  bool isOpOfCoreType(Operation *op, TCoreType coreType) const {
    auto tryGetCoreType = getCoreType(op);
    return (succeeded(tryGetCoreType) && tryGetCoreType.value() == coreType);
  }

  bool mightHaveCrossCoreMemoryEffect(Operation *op) const {
    auto tryGetCoreType = getCoreType(op);
    if (succeeded(tryGetCoreType) &&
        (tryGetCoreType.value() != TCoreType::CUBE_OR_VECTOR)) {
      return true;
    }
    if (isa<DestinationStyleOpInterface>(op)) {
      return true;
    }
    if (isa<func::CallOp, hivm::BitcastOp, memref::LoadOp, memref::StoreOp,
            affine::AffineLoadOp, affine::AffineStoreOp, tensor::ExtractOp,
            tensor::InsertOp, tensor::InsertSliceOp, tensor::ExtractSliceOp,
            hivm::CustomOp, hivm::CustomMacroOp>(op)) {
      return true;
    }
    if (auto mei = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      mei.getEffects(effects);
      if (!effects.empty()) {
        return true;
      }
    }
    return false;
  }

  bool isOpToBeAnchored(Operation *op, bool isBefore) const {
    if (ignoreOp(op)) {
      return false;
    }
    if (this->insertAnchorOpsBeforeAll) {
      return isBefore;
    }
    if (this->insertAnchorOpsBeforeMemEffectOps) {
      if (isBefore) {
        if (mightHaveCrossCoreMemoryEffect(op)) {
          return true;
        }
      }
    }
    if (this->insertAnchorOnlyBeforeCubeOps) {
      return isOpOfCoreType(op, TCoreType::CUBE) ||
             isOpOfCoreType(op, TCoreType::CUBE_AND_VECTOR);
    }
    if (this->insertAnchorOnlyBeforeVectorOps) {
      return isOpOfCoreType(op, TCoreType::VECTOR) ||
             isOpOfCoreType(op, TCoreType::CUBE_AND_VECTOR);
    }
    if (this->insertAnchorBeforeCubeAndVectorOps) {
      return isOpOfCoreType(op, TCoreType::CUBE) ||
             isOpOfCoreType(op, TCoreType::VECTOR) ||
             isOpOfCoreType(op, TCoreType::CUBE_AND_VECTOR);
    }
    return false;
  }

  bool needAnyAnchor(Operation *op) const {
    if (isOpToBeAnchored(op, /*isBefore=*/true)) {
      return true;
    }
    if (isOpToBeAnchored(op, /*isBefore=*/false)) {
      return true;
    }
    for (Region &region : op->getRegions()) {
      for (Block &nestedBlock : region) {
        for (Operation &childOp : nestedBlock) {
          if (needAnyAnchor(&childOp)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  bool isCVPipeliningLoop(Operation *op) const {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      return forOp->hasAttr(kCVUnrolledLoopName) ||
             forOp->hasAttr(kMultibufferUnrollAttrName);
    }
    return false;
  }

  bool isSIMTScope(Operation *op) const {
    // Skip anchor-insert in simt scope
    if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
      if (auto vectorType = scopeOp->getAttrOfType<StringAttr>("vector_type")) {
        if (vectorType.getValue() == "simt") {
          return true;
        }
      }
    }
    return false;
  }

  bool isRegionsToBeAnchored(Operation *op) const {
    if (isCVPipeliningLoop(op)) {
      return true;
    }
    if (isSIMTScope(op)) {
      return false;
    }
    if (isOpToBeAnchored(op, /*insertBefore=*/true)) {
      return false;
    }
    for (Region &region : op->getRegions()) {
      for (Block &nestedBlock : region) {
        for (Operation &childOp : nestedBlock) {
          if (needAnyAnchor(&childOp)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  bool isBlockToBeAnchored(Operation *op, Block &block) const {
    if (isCVPipeliningLoop(op)) {
      return true;
    }
    if (isSIMTScope(op)) {
      return false;
    }
    if (isOpToBeAnchored(op, /*insertBefore=*/true)) {
      return false;
    }
    for (Operation &childOp : block) {
      if (needAnyAnchor(&childOp)) {
        return true;
      }
    }
    return false;
  }

  bool isOpToNeverBeAnchored(Operation *op) const {
    if (ignoreOp(op)) {
      return true;
    }
    if (this->insertAnchorOpsBeforeAll) {
      return false;
    }
    if (isBlockFrontOp(op) || isBlockBackOp(op)) {
      return false;
    }
    if (isCodeStructureOp(op)) {
      return false;
    }
    if (mightHaveCrossCoreMemoryEffect(op)) {
      return false;
    }
    return true;
  }
};

void InsertAnchorsAndBackupPass::insertAnchor(Operation *op, OpBuilder &builder,
                                              int64_t &nextAnchorId,
                                              bool insertBefore) {
  OpBuilder::InsertionGuard guard(builder);
  if (insertBefore) {
    builder.setInsertionPoint(op);
  } else {
    builder.setInsertionPointAfter(op);
  }
  builder.create<AnchorOp>(op->getLoc(),
                           builder.getI64IntegerAttr(nextAnchorId++), nullptr);
}

void InsertAnchorsAndBackupPass::insertAnchorsInBlock(
    Block &block, OpBuilder &builder, int64_t &nextAnchorId,
    bool onlyInsertFrontBack) {
  // Snapshot the block ops first so the iteration is not perturbed by the
  // anchors we are about to splice in.
  SmallVector<Operation *> blockOps;
  for (Operation &op : block) {
    blockOps.push_back(&op);
  }
  bool anchorWasInsertedBeforeLastOp = false;
  for (Operation *op : blockOps) {
    if (isOpToNeverBeAnchored(op)) {
      continue;
    }

    if (onlyInsertFrontBack) {
      if (!anchorWasInsertedBeforeLastOp) {
        if (isBlockFrontOp(op) || isBlockBackOp(op)) {
          insertAnchor(op, builder, nextAnchorId, /*insertBefore=*/true);
        }
      }
      continue;
    }

    if (!anchorWasInsertedBeforeLastOp) {
      if (isBlockFrontOp(op) || isBlockBackOp(op) ||
          isOpToBeAnchored(op, /*isBefore=*/true) ||
          isRegionsToBeAnchored(op)) {
        insertAnchor(op, builder, nextAnchorId, /*insertBefore=*/true);
      }
    }

    if (isRegionsToBeAnchored(op)) {
      for (Region &region : op->getRegions()) {
        for (Block &nestedBlock : region) {
          insertAnchorsInBlock(
              nestedBlock, builder, nextAnchorId,
              /*onlyInsertFrontBack=*/!isBlockToBeAnchored(op, nestedBlock));
        }
      }
    }

    if (!isBlockBackOp(op) && isOpToBeAnchored(op, /*isBefore=*/false)) {
      insertAnchor(op, builder, nextAnchorId, /*insertBefore=*/false);
      anchorWasInsertedBeforeLastOp = true;
    } else {
      anchorWasInsertedBeforeLastOp = false;
    }
  }
}

void InsertAnchorsAndBackupPass::eraseAllAnchors(func::FuncOp funcOp) {
  SmallVector<Operation *> toBeErased;
  funcOp.walk([&](Operation *op) {
    if (isa<hivm::AnchorOp>(op)) {
      toBeErased.push_back(op);
    }
  });
  for (Operation *op : toBeErased) {
    op->erase();
  }
}

void InsertAnchorsAndBackupPass::eraseBackupFuncOps(ModuleOp mod) {
  SmallVector<func::FuncOp> backupFuncOps;
  mod.walk([&](func::FuncOp funcOp) {
    if (funcOp->hasAttr(hivm::BackupFunctionAttr::name)) {
      backupFuncOps.push_back(funcOp);
    }
  });
  for (func::FuncOp funcOp : backupFuncOps) {
    funcOp.erase();
  }
}

func::FuncOp InsertAnchorsAndBackupPass::backupFunc(func::FuncOp src) {
  OpBuilder builder(src);
  auto *ctx = src->getContext();

  auto backup = cast<func::FuncOp>(builder.clone(*src.getOperation()));
  backup.setSymName((src.getSymName() + hivm::kFuncBackupSuffix).str());

  // Mark the function so the delayed pass and the BiShengIR pass manager can
  // recognize it as a backup.
  backup->setAttr(hivm::BackupFunctionAttr::name, builder.getUnitAttr());

  // Restrict the backup to passes whose argument names appear in
  // `annotation.filter_passes`. Only this pass and the delayed cross-core GSS
  // pass should rewrite the backup; every other pass in the pipeline must skip
  // it via BiShengIRPassManager's filtering action handler.
  std::string insertAnchorsAndBackupPassName = this->getArgument().str();
  auto delayedCrossCoreGSSPass = createDelayedCrossCoreGSSPass();
  std::string delayedCrossCoreGSSPassName =
      delayedCrossCoreGSSPass->getArgument().str();
  auto splitSimtModulePass = createSplitSimtModulePass();
  std::string splitSimtModulePassName =
      splitSimtModulePass->getArgument().str();
  std::string allPassesNames = insertAnchorsAndBackupPassName + "," +
                               delayedCrossCoreGSSPassName + "," +
                               splitSimtModulePassName;

  auto attr = mlir::annotation::FilterPassesAttr::get(
      ctx, StringAttr::get(ctx, allPassesNames));
  backup->setAttr(mlir::annotation::FilterPassesAttr::name, attr);

  // Keep backup public so the Inliner/DCE-of-private-funcs in later passes
  // (e.g. inline-scope -> upstream Inliner) does not reclaim it. It will be
  // erased wholesale at the end of DelayedCrossCoreGSS.
  backup.setPublic();
  return backup;
}

func::FuncOp InsertAnchorsAndBackupPass::getOrCreateBackupFunc(
    func::FuncOp funcOp,
    llvm::DenseMap<Operation *, func::FuncOp> &backupFuncs) {
  if (!funcOp)
    return nullptr;
  if (funcOp->hasAttr(hivm::BackupFunctionAttr::name))
    return funcOp;
  if (auto it = backupFuncs.find(funcOp.getOperation());
      it != backupFuncs.end())
    return it->second;

  func::FuncOp backupFuncOp = backupFunc(funcOp);
  backupFuncs.try_emplace(funcOp.getOperation(), backupFuncOp);
  retargetCallsToBackupFuncs(backupFuncOp, backupFuncs);
  return backupFuncOp;
}

void InsertAnchorsAndBackupPass::retargetCallsToBackupFuncs(
    func::FuncOp backupFuncOp,
    llvm::DenseMap<Operation *, func::FuncOp> &backupFuncs) {
  SmallVector<func::CallOp> callOps;
  backupFuncOp.walk([&](func::CallOp callOp) { callOps.push_back(callOp); });

  for (func::CallOp callOp : callOps) {
    auto calleeFunc = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        callOp, callOp.getCalleeAttr());
    if (!calleeFunc)
      continue;

    func::FuncOp backupCalleeFunc =
        getOrCreateBackupFunc(calleeFunc, backupFuncs);
    if (!backupCalleeFunc)
      continue;

    callOp.setCallee(backupCalleeFunc.getSymName());
  }
}

void InsertAnchorsAndBackupPass::runOnOperation() {
  ModuleOp mod = getOperation();

  // Cleanup mode is the inverse of normal mode: strip every artifact this
  // pass produces. Run it after the delayed solver has consumed the backups
  // and before lowering, so downstream passes never see anchors.
  if (this->cleanup) {
    eraseBackupFuncOps(mod);
    mod.walk([&](func::FuncOp funcOp) { eraseAllAnchors(funcOp); });
    return;
  }

  // Only mix functions need anchors and backups; cube-only and vector-only
  // functions are not subject to cross-core hazards by construction.
  SmallVector<func::FuncOp> mixFuncOps;
  mod.walk([&](func::FuncOp funcOp) {
    if (auto coreType = queryFuncCoreType(funcOp)) {
      if (coreType.value() == TFuncCoreType::MIX) {
        mixFuncOps.push_back(funcOp);
      }
    }
  });

  OpBuilder builder(&getContext());
  llvm::DenseMap<Operation *, func::FuncOp> backupFuncs;
  for (func::FuncOp funcOp : mixFuncOps) {
    // Anchor ids restart at zero per function: ids only need to be unique
    // within a single (mix, cube, vector) triplet's translator scope.
    int64_t nextAnchorId = 0;
    for (Region &region : funcOp->getRegions()) {
      for (Block &block : region) {
        insertAnchorsInBlock(block, builder, nextAnchorId);
      }
    }

    // Anchors must already be in place when we clone, so the backup carries
    // the same anchor map the delayed pass will look up later.
    func::FuncOp backupFuncOp = getOrCreateBackupFunc(funcOp, backupFuncs);
    retargetCallsToBackupFuncs(backupFuncOp, backupFuncs);
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createInsertAnchorsAndBackupPass(
    const InsertAnchorsAndBackupOptions &options) {
  return std::make_unique<InsertAnchorsAndBackupPass>(options);
}
