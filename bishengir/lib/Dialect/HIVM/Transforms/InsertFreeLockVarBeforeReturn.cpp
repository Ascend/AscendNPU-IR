//===--------- InsertFreeLockVarBeforeReturn.cpp ----------------*- C++ -*-===//
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
//
// When SyncBlockLock/Unlock are in conditional branches (e.g. else block),
// some blocks may not execute them due to control flow, causing lock_var count
// mismatch and deadlock. This pass inserts FreeLockVarOp before every
// return. Each FreeLockVarOp runs one sync_block_lock / sync_block_unlock pair
// in the template, preventing count mismatch when some blocks skipped the
// guarded region.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "hivm-insert-free-lock-var-before-return"

namespace mlir {
#define GEN_PASS_DEF_INSERTFREELOCKVARBEFORERETURN
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct LockVarInfo {
  Value lockVar;
  bool withSubblock = false;
};

void collectLockVarsAndReturnOps(func::FuncOp funcOp,
                                 SmallVector<LockVarInfo> &lockVars,
                                 SmallVector<func::ReturnOp> &returnOps) {
  lockVars.clear();
  returnOps.clear();

  llvm::SmallDenseMap<Value, bool> lockVarToWithSubblock;

  // assuming that each SyncBlockLock is paired with a SyncBlockUnlock on the
  // same lock_var
  auto recordLockVar = [&](Value lockVar, bool withSubblock) {
    lockVarToWithSubblock.try_emplace(lockVar, false).first->second =
        withSubblock;
  };

  // Each SyncBlockLock is assumed paired with a SyncBlockUnlock on the same
  // lock_var; collecting lock ops alone yields the same distinct lock_vars.
  funcOp.walk([&](Operation *op) {
    if (auto lockOp = dyn_cast<SyncBlockLockOp>(op)) {
      recordLockVar(lockOp.getLockVar(),
                    lockOp->hasAttr(SyncBlockLockWithSubblockAttr::name));
    } else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      returnOps.push_back(returnOp);
    }
  });

  for (const auto &[lockVar, withSubblock] : lockVarToWithSubblock)
    lockVars.push_back({lockVar, withSubblock});
}

class InsertFreeLockVarBeforeReturnPass
    : public impl::InsertFreeLockVarBeforeReturnBase<
          InsertFreeLockVarBeforeReturnPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      if (!hacc::utils::isDeviceEntry(funcOp)) {
        return;
      }

      SmallVector<LockVarInfo> lockVars;
      SmallVector<func::ReturnOp> returnOps;

      collectLockVarsAndReturnOps(funcOp, lockVars, returnOps);

      if (lockVars.empty())
        return;

      OpBuilder builder(funcOp.getContext());

      for (func::ReturnOp returnOp : returnOps) {
        builder.setInsertionPoint(returnOp);

        for (const LockVarInfo &info : lockVars) {
          auto freeLockOp =
              builder.create<FreeLockVarOp>(returnOp.getLoc(), info.lockVar);

          if (info.withSubblock)
            freeLockOp->setAttr(SyncBlockLockWithSubblockAttr::name,
                                builder.getUnitAttr());
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createInsertFreeLockVarBeforeReturnPass() {
  return std::make_unique<InsertFreeLockVarBeforeReturnPass>();
}
