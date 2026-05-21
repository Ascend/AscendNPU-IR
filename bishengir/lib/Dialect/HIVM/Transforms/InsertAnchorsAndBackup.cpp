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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
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

struct InsertAnchorsAndBackupPass
    : public impl::InsertAnchorsAndBackupBase<InsertAnchorsAndBackupPass> {
  explicit InsertAnchorsAndBackupPass(
      const InsertAnchorsAndBackupOptions &options)
      : InsertAnchorsAndBackupBase(options) {}

  ~InsertAnchorsAndBackupPass() override = default;

  void runOnOperation() override;

private:
  void insertAnchor(Operation *op, OpBuilder &builder, int64_t &nextAnchorId,
                    bool insertBefore = false);

  void insertAnchorAttr(Operation *op, OpBuilder &builder,
                        int64_t &nextAnchorId, bool insertBefore = false);

  void insertAnchorsInBlock(Block &block, OpBuilder &builder,
                            int64_t &nextAnchorId);

  func::FuncOp backupFunc(func::FuncOp funcOp);
  func::FuncOp
  getOrCreateBackupFunc(func::FuncOp funcOp,
                        llvm::DenseMap<Operation *, func::FuncOp> &backupFuncs);
  void retargetCallsToBackupFuncs(
      func::FuncOp backupFuncOp,
      llvm::DenseMap<Operation *, func::FuncOp> &backupFuncs);

  void eraseAllAnchors(func::FuncOp funcOp);

  void eraseBackupFuncOps(ModuleOp mod);
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

void InsertAnchorsAndBackupPass::insertAnchorAttr(Operation *op,
                                                  OpBuilder &builder,
                                                  int64_t &nextAnchorId,
                                                  bool insertBefore) {
  if (insertBefore) {
    op->setAttr(hivm::AnchorIdBeforeAttr::name,
                builder.getI64IntegerAttr(nextAnchorId++));
  } else {
    op->setAttr(hivm::AnchorIdAfterAttr::name,
                builder.getI64IntegerAttr(nextAnchorId++));
  }
}

void InsertAnchorsAndBackupPass::insertAnchorsInBlock(Block &block,
                                                      OpBuilder &builder,
                                                      int64_t &nextAnchorId) {
  SmallVector<Operation *> blockOps;
  for (Operation &op : block) {
    blockOps.push_back(&op);
  }
  for (Operation *op : blockOps) {
    insertAnchor(op, builder, nextAnchorId, /*insertBefore=*/true);
    if (op->getNumRegions() > 0) {
      if (isa<scf::ForOp>(op)) {
        insertAnchorAttr(op, builder, nextAnchorId, /*insertBefore=*/true);
      }
      for (Region &region : op->getRegions()) {
        for (Block &nestedBlock : region) {
          insertAnchorsInBlock(nestedBlock, builder, nextAnchorId);
        }
      }
      if (isa<scf::ForOp>(op)) {
        insertAnchorAttr(op, builder, nextAnchorId);
      }
    }
  }
}

void InsertAnchorsAndBackupPass::eraseAllAnchors(func::FuncOp funcOp) {
  SmallVector<hivm::AnchorOp> anchorOps;
  funcOp.walk([&](Operation *op) {
    if (auto anchorOp = dyn_cast<hivm::AnchorOp>(op)) {
      anchorOps.push_back(anchorOp);
      return;
    }
    if (op->hasAttr(hivm::AnchorIdAttr::name)) {
      op->removeAttr(hivm::AnchorIdAttr::name);
    }
    if (op->hasAttr(hivm::AnchorIdBeforeAttr::name)) {
      op->removeAttr(hivm::AnchorIdBeforeAttr::name);
    }
    if (op->hasAttr(hivm::AnchorIdAfterAttr::name)) {
      op->removeAttr(hivm::AnchorIdAfterAttr::name);
    }
  });
  for (hivm::AnchorOp anchorOp : anchorOps) {
    anchorOp->erase();
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

  // Set the gssbackup function attribute
  backup->setAttr(hivm::BackupFunctionAttr::name, builder.getUnitAttr());

  // Set the filter passes attribute
  std::string insertAnchorsAndBackupPassName = this->getArgument().str();
  auto delayedCrossCoreGSSPass = createDelayedCrossCoreGSSPass();
  std::string delayedCrossCoreGSSPassName =
      delayedCrossCoreGSSPass->getArgument().str();
  std::string allPassesNames =
      insertAnchorsAndBackupPassName + "," + delayedCrossCoreGSSPassName;

  auto attr = mlir::annotation::FilterPassesAttr::get(
      ctx, StringAttr::get(ctx, allPassesNames));
  backup->setAttr(mlir::annotation::FilterPassesAttr::name, attr);

  // Keep backup public so the Inliner/DCE-of-private-funcs in later passes
  // (e.g. inline-scope → upstream Inliner) does not reclaim it. It will be
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

  if (this->cleanup) {
    eraseBackupFuncOps(mod);
    mod.walk([&](func::FuncOp funcOp) { eraseAllAnchors(funcOp); });
    return;
  }

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
    int64_t nextAnchorId = 0;
    for (Region &region : funcOp->getRegions()) {
      for (Block &block : region) {
        insertAnchorsInBlock(block, builder, nextAnchorId);
      }
    }

    func::FuncOp backupFuncOp = getOrCreateBackupFunc(funcOp, backupFuncs);
    retargetCallsToBackupFuncs(backupFuncOp, backupFuncs);
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createInsertAnchorsAndBackupPass(
    const InsertAnchorsAndBackupOptions &options) {
  return std::make_unique<InsertAnchorsAndBackupPass>(options);
}
