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
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"

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

  void insertAnchorBlockOp(Operation *op, Block &block, OpBuilder &builder,
                           int64_t id_start, int64_t id_end);

  void insertAnchorsInBlock(Block &block, OpBuilder &builder,
                            int64_t &nextAnchorId);

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

  // Decide whether `op` should get a leading anchor.
  //
  // The default ("relevant ops only") case anchors before ops that can take
  // part in cross-core memory hazards or that act as structural boundaries:
  // memory/tensor accesses, custom ops, loop-likes/ifs/scopes, terminators,
  // ops that infer a core type, and any op declaring non-empty memory effects.
  // Sync ops are explicitly skipped because they are inserted/erased by the
  // GSS flow itself; anchoring them would couple anchor ids to sync placement.
  //
  // `insertAnchorOpsBeforeAll` flips the filter back to "every op" for
  // debugging and for parity with the original proposal.
  bool isOpTypeToBeAnchored(Operation *op) const {
    if (this->insertAnchorOpsBeforeAll) {
      return true;
    }
    if (isa<hivm::PipeBarrierOp, hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp,
            hivm::SyncBlockOp, hivm::SetFlagOp, hivm::WaitFlagOp>(op)) {
      return false;
    }
    if (isa<hivm::BitcastOp>(op)) {
      return true;
    }
    if (isa<memref::LoadOp, memref::StoreOp, affine::AffineLoadOp,
            affine::AffineStoreOp, tensor::ExtractOp, tensor::InsertOp,
            tensor::InsertSliceOp, tensor::ExtractSliceOp>(op)) {
      return true;
    }
    if (isa<hivm::CustomOp, hivm::CustomMacroOp>(op)) {
      return true;
    }
    if (isa<LoopLikeOpInterface, scf::IfOp, scope::ScopeOp, func::CallOp>(op)) {
      return true;
    }
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      return true;
    }
    if (isa<hivm::InferCoreTypeInterface, DestinationStyleOpInterface>(op)) {
      return true;
    }
    if (op->hasTrait<OpTrait::CoreTypeTrait<TCoreType::CUBE>::Impl>()) {
      return true;
    }
    if (op->hasTrait<OpTrait::CoreTypeTrait<TCoreType::VECTOR>::Impl>()) {
      return true;
    }
    if (op->hasTrait<
            OpTrait::CoreTypeTrait<TCoreType::CUBE_OR_VECTOR>::Impl>()) {
      return true;
    }
    if (op->hasTrait<
            OpTrait::CoreTypeTrait<TCoreType::CUBE_AND_VECTOR>::Impl>()) {
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
                           builder.getI64IntegerAttr(nextAnchorId++));
}

void InsertAnchorsAndBackupPass::insertAnchorBlockOp(Operation *op,
                                                     Block &block,
                                                     OpBuilder &builder,
                                                     int64_t id_start,
                                                     int64_t id_end) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&block);
  builder.create<AnchorBlockOp>(op->getLoc(), id_start, id_end);
}

void InsertAnchorsAndBackupPass::insertAnchorsInBlock(Block &block,
                                                      OpBuilder &builder,
                                                      int64_t &nextAnchorId) {
  // Snapshot the block ops first so the iteration is not perturbed by the
  // anchors we are about to splice in.
  SmallVector<Operation *> blockOps;
  for (Operation &op : block) {
    blockOps.push_back(&op);
  }
  for (Operation *op : blockOps) {
    // Lead the block with one anchor and add a leading anchor before each
    // anchorable op. Combined with the trailing anchor emitted for
    // region-bearing ops below, this produces the N+1 anchor model from the
    // proposal at every nesting level.
    if (op == blockOps.front() || isOpTypeToBeAnchored(op)) {
      insertAnchor(op, builder, nextAnchorId, /*insertBefore=*/true);
    }
    if (op->getNumRegions() > 0) {
     // Skip anchor-insert in simt scope
      if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
        if (auto vectorType = scopeOp->getAttrOfType<StringAttr>("vector_type")) {
          if (vectorType.getValue() == "simt") {
            continue;
          }
        }
      }
      for (Region &region : op->getRegions()) {
        for (Block &nestedBlock : region) {
          int64_t block_start_id = nextAnchorId++;
          insertAnchorsInBlock(nestedBlock, builder, nextAnchorId);
          int64_t block_end_id = nextAnchorId++;
          insertAnchorBlockOp(op, nestedBlock, builder, block_start_id,
                              block_end_id);
        }
      }
    }
  }
}

void InsertAnchorsAndBackupPass::eraseAllAnchors(func::FuncOp funcOp) {
  SmallVector<Operation *> toBeErased;
  funcOp.walk([&](Operation *op) {
    if (isa<hivm::AnchorOp, hivm::AnchorBlockOp>(op)) {
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
  std::string allPassesNames =
      insertAnchorsAndBackupPassName + "," + delayedCrossCoreGSSPassName
      + ",split-simt-module";

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
