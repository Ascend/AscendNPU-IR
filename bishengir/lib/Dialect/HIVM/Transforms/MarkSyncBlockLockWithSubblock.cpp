//===--------- MarkSyncBlockLockWithSubblock.cpp ----------------*- C++ -*-===//
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
// This pass marks sync_block_lock and sync_block_unlock ops with
// sync_block_lock_with_subblock tag when they are not inside an scf.if op with
// limit_sub_block_id0 attribute, in mix type modules.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"

namespace mlir {
#define GEN_PASS_DEF_MARKSYNCBLOCKLOCKWITHSUBBLOCK
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-mark-sync-block-lock-with-subblock"

namespace {
static constexpr llvm::StringLiteral kLimitSubBlockId0Attr =
    "limit_sub_block_id0";

/// Check if the module contains any mix type function.
static bool isMixModule(ModuleOp module) {
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func->getAttrOfType<UnitAttr>(TPartOfMixAttr::name))
      return true;
    auto mixMode = func->getAttrOfType<StringAttr>("mix_mode");
    if (mixMode && mixMode.getValue() == "mix")
      return true;
  }
  return false;
}

/// Check if the given operation is inside an scf.if op that has
/// limit_sub_block_id0 attribute.
static bool isInsideLimitSubBlockId0If(Operation *op) {
  Operation *current = op->getParentOp();
  while (current) {
    if (auto ifOp = dyn_cast<scf::IfOp>(current)) {
      return ifOp->hasAttr(kLimitSubBlockId0Attr);
    }
    current = current->getParentOp();
  }
  return false;
}

struct MarkSyncBlockLockWithSubblockPass
    : public impl::MarkSyncBlockLockWithSubblockBase<
          MarkSyncBlockLockWithSubblockPass> {
  using Base::Base;
  void runOnOperation() override;
};

void MarkSyncBlockLockWithSubblockPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (!isMixModule(module))
    return;

  module->walk([&](Operation *op) {
    if (!isa<SyncBlockLockOp, SyncBlockUnlockOp>(op))
      return WalkResult::advance();

    if (op->hasAttr(SyncBlockLockWithSubblockAttr::name))
      return WalkResult::advance();

    if (!isInsideLimitSubBlockId0If(op)) {
      op->setAttr(SyncBlockLockWithSubblockAttr::name,
                  UnitAttr::get(op->getContext()));
    }
    return WalkResult::advance();
  });
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createMarkSyncBlockLockWithSubblockPass() {
  return std::make_unique<MarkSyncBlockLockWithSubblockPass>();
}
