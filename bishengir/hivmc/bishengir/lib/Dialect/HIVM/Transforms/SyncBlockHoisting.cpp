//===- SyncBlockHoisting.cpp ---------------------------------------*- C++
//-*-===//
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
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DECL_SYNCBLOCKHOISTING
#define GEN_PASS_DEF_SYNCBLOCKHOISTING
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

#define DEBUG_TYPE "hivm-sync-block-hoisting"

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct SyncBlockHoistingPass
    : public mlir::impl::SyncBlockHoistingBase<SyncBlockHoistingPass> {

public:
  void runOnOperation() override;
};

bool eraseSyncBlockOp(Operation *opWithBlock) {
  SmallVector<Operation *> toBeErasedOps = {};
  opWithBlock->walk([&toBeErasedOps](Operation *op) {
    if (isa<SyncBlockLockOp, SyncBlockUnlockOp>(op))
      toBeErasedOps.emplace_back(op);
  });
  if (toBeErasedOps.empty())
    return false;
  llvm::for_each(toBeErasedOps, [](Operation *op) { op->erase(); });
  return true;
}

} // namespace

void SyncBlockHoistingPass::runOnOperation() {
  auto wrapInLock = [builder =
                         OpBuilder(&getContext())](Operation &op) mutable {
    builder.setInsertionPoint(&op);
    auto lockVar = createSyncBlockLockVar(builder, op.getLoc());
    builder.create<hivm::SyncBlockLockOp>(op.getLoc(), lockVar);
    builder.setInsertionPointAfter(&op);
    builder.create<hivm::SyncBlockUnlockOp>(op.getLoc(), lockVar);
  };

  func::FuncOp funcOp = getOperation();
  auto&& toBeLockedOps =
      llvm::make_filter_range(funcOp.getBody().getOps(), [](Operation &op) {
        return op.getNumRegions() > 0 && eraseSyncBlockOp(&op);
      });

  llvm::for_each(toBeLockedOps, wrapInLock);
}

std::unique_ptr<Pass> mlir::hivm::createSyncBlockHoistingPass() {
  return std::make_unique<SyncBlockHoistingPass>();
}
