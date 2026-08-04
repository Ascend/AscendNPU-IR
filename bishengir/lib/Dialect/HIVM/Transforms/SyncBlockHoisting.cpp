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
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"
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

static constexpr llvm::StringLiteral kPostLoopAllVectorSyncAttr =
    "hivm.unordered_lock_post_loop_all_vector_sync";

static void insertPostLoopAllVectorSync(PatternRewriter &rewriter,
                                        Operation *loopOp) {
  auto *ctx = loopOp->getContext();
  auto syncMode =
      hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL_VECTOR);
  auto vectorPipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE3);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(loopOp);
  rewriter.create<hivm::SyncBlockOp>(loopOp->getLoc(), syncMode, IntegerAttr{},
                                     Value{}, hivm::PipeAttr{}, vectorPipe);
}

struct SyncBlockHoistingPass
    : public mlir::impl::SyncBlockHoistingBase<SyncBlockHoistingPass> {

public:
  void runOnOperation() override;
};

/// Hoist every create_sync_block_lock in scf.if regions to the start of the
/// enclosing func. Does not move sync_block_lock / sync_block_unlock.
struct HoistCreateSyncBlockLockInIfPattern
    : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<CreateSyncBlockLockOp, 4> createOps;
    for (Region &region : ifOp->getRegions()) {
      region.walk([&](CreateSyncBlockLockOp createOp) {
        createOps.push_back(createOp);
      });
    }
    if (createOps.empty())
      return failure();

    auto funcOp = ifOp->getParentOfType<func::FuncOp>();
    assert(funcOp && "create hoisting expects scf.if inside func.func");
    Block &entry = funcOp.getBody().front();

    for (CreateSyncBlockLockOp createOp : createOps)
      rewriter.moveOpBefore(createOp, &entry.front());
    return success();
  }
};

struct HoistingSyncBlockPattern
    : public OpInterfaceRewritePattern<LoopLikeOpInterface> {
  using OpInterfaceRewritePattern<
      LoopLikeOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LoopLikeOpInterface op,
                                PatternRewriter &rewriter) const override {
    Operation *loopOp = op.getOperation();
    // Step 1: Collect all lock/unlock in the loop (including under scf.if).
    SmallVector<hivm::SyncBlockLockOp> lockVec = {};
    SmallVector<hivm::SyncBlockUnlockOp> unlockVec = {};
    for (auto &region : op->getRegions()) {
      region.walk(
          [&](hivm::SyncBlockLockOp lockOp) { lockVec.push_back(lockOp); });
      region.walk([&](hivm::SyncBlockUnlockOp unlockOp) {
        unlockVec.push_back(unlockOp);
      });
    }
    assert(lockVec.size() == unlockVec.size() &&
           "The number of lock and unlock should be the same in one region.");
    if (lockVec.empty())
      return failure();

    bool hasUnorderedLock = false;
    for (auto lockOp : lockVec)
      hasUnorderedLock |=
          lockOp->hasAttr(hivm::SyncBlockLockUnorderedAttr::name);

    // Do NOT hoist unordered lock/unlock out of loops. The unordered bakery
    // lock supports skipped participants, so it does not need the old ordered
    // lock's "everyone reaches the lock" transformation. Hoisting would widen
    // the critical section from the guarded RMW to the whole autoblockify loop,
    // serializing pure-vector kernels and pulling any later cross-block sync
    // into the critical section.
    bool hasCrossBlockSync = false;
    for (auto &region : op->getRegions())
      region.walk([&](Operation *inner) {
        if (isa<hivm::SyncBlockOp, hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp>(
                inner))
          hasCrossBlockSync = true;
      });

    SmallVector<Operation *, 4> createOps;
    llvm::SmallPtrSet<Operation *, 4> createOpSet;
    for (auto lockOp : lockVec) {
      auto *defOp = lockOp.getLockVar().getDefiningOp();
      if (!isa<CreateSyncBlockLockOp>(defOp)) {
        op->emitOpError(
            "expected lock memref defined by hivm.hir.create_sync_block_lock");
        return failure();
      }
      if (createOpSet.insert(defOp).second)
        createOps.push_back(defOp);
    }

    auto funcOp = op->getParentOfType<func::FuncOp>();
    assert(funcOp && "sync block hoisting expects loop inside func.func");
    Block &entry = funcOp.getBody().front();

    // move create_sync_block_lock out of the loop if it is nested in the loop.
    // keep the lock/unlock in the loop to avoid widening the critical section.
    auto hoistCreateOpsNestedInLoop = [&]() -> bool {
      bool changed = false;
      for (auto it = createOps.rbegin(); it != createOps.rend(); ++it) {
        Operation *createOp = *it;
        bool nestedInLoop = false;
        for (Operation *parent = createOp->getParentOp(); parent;
             parent = parent->getParentOp()) {
          if (parent == loopOp) {
            nestedInLoop = true;
            break;
          }
        }
        if (!nestedInLoop)
          continue;
        rewriter.moveOpBefore(createOp, &entry.front());
        changed = true;
      }
      return changed;
    };

    if (hasUnorderedLock) {
      bool changed = hoistCreateOpsNestedInLoop();
      if (hasCrossBlockSync && !loopOp->hasAttr(kPostLoopAllVectorSyncAttr)) {
        rewriter.modifyOpInPlace(loopOp, [&] {
          loopOp->setAttr(kPostLoopAllVectorSyncAttr,
                          UnitAttr::get(loopOp->getContext()));
        });
        insertPostLoopAllVectorSync(rewriter, loopOp);
        return success();
      }
      return changed ? success() : failure();
    }

    Value lockMemref = lockVec.front().getLockVar();

    auto primaryCreate =
        cast<CreateSyncBlockLockOp>(lockMemref.getDefiningOp());
    rewriter.moveOpBefore(primaryCreate, &entry.front());

    for (auto lockOp : lockVec)
      rewriter.eraseOp(lockOp);
    for (auto unlockOp : unlockVec)
      rewriter.eraseOp(unlockOp);

    for (Operation *createOp : createOpSet) {
      if (createOp != primaryCreate && createOp->use_empty())
        rewriter.eraseOp(createOp);
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(loopOp);
    rewriter.create<hivm::SyncBlockLockOp>(op->getLoc(), lockMemref);
    rewriter.setInsertionPointAfter(loopOp);
    rewriter.create<hivm::SyncBlockUnlockOp>(op->getLoc(), lockMemref);
    return success();
  }
};

} // namespace

void SyncBlockHoistingPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<HoistCreateSyncBlockLockInIfPattern>(patterns.getContext());
  patterns.add<HoistingSyncBlockPattern>(patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createSyncBlockHoistingPass() {
  return std::make_unique<SyncBlockHoistingPass>();
}
