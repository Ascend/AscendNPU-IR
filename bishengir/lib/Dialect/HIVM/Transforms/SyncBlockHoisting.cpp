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
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

struct HoistingSyncBlockPattern
    : public OpInterfaceRewritePattern<LoopLikeOpInterface> {
  using OpInterfaceRewritePattern<
      LoopLikeOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LoopLikeOpInterface op,
                                PatternRewriter &rewriter) const override {
    // Step 1: Get the list of lock and unlock operation
    SmallVector<hivm::SyncBlockLockOp> lockVec = {};
    SmallVector<hivm::SyncBlockUnlockOp> unlockVec = {};
    for (auto &region : op->getRegions()) {
      region.walk([&](hivm::SyncBlockLockOp op) { lockVec.push_back(op); });
      region.walk([&](hivm::SyncBlockUnlockOp op) { unlockVec.push_back(op); });
    }
    // Step 2: Return if no lock and unlock op is found
    assert(lockVec.size() == unlockVec.size() &&
           "The number of lock and unlock should be the same in one region.");
    if (lockVec.empty())
      return failure();
    // Step 3: Erase all the lock and unlock op and create them outside
    for (auto op : lockVec)
      rewriter.eraseOp(op);
    for (auto op : unlockVec)
      rewriter.eraseOp(op);
    auto lockVar = createSyncBlockLockVar(rewriter, op->getLoc());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    rewriter.create<hivm::SyncBlockLockOp>(op->getLoc(), lockVar);
    rewriter.setInsertionPointAfter(op);
    rewriter.create<hivm::SyncBlockUnlockOp>(op->getLoc(), lockVar);
    return success();
  }
};

} // namespace

void SyncBlockHoistingPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<HoistingSyncBlockPattern>(patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createSyncBlockHoistingPass() {
  return std::make_unique<SyncBlockHoistingPass>();
}
