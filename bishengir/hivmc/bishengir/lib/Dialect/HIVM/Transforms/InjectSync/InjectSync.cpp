//===------------- InjectSync.cpp ----Auto Inject Sync --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/InjectSync.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncDebug.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "hivm-inject-sync"

namespace mlir {
#define GEN_PASS_DEF_INJECTSYNC
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace mlir {
struct InjectSyncPass : public impl::InjectSyncBase<InjectSyncPass> {
  explicit InjectSyncPass(const InjectSyncOptions &options)
      : InjectSyncBase(options) {}

public:
  void runOnOperation() override;
};
} // namespace mlir

void InjectSyncAnalysis::InjectSyncAll() {
  auto checkInsertBarrierAllBeforeOp = [](Operation *op) {
    if (op->getDialect()->getNamespace() ==
        HIVMDialect::getDialectNamespace()) {
      return true;
    }
    if (isa<memref::LoadOp, memref::StoreOp, affine::AffineLoadOp,
            affine::AffineStoreOp, tensor::ExtractOp, tensor::InsertOp>(op)) {
      return true;
    }
    if (isa<func::ReturnOp, func::CallOp>(op)) {
      return true;
    }
    return false;
  };
  MLIRContext *ctx = func_->getContext();
  IRRewriter rewriter(ctx);
  func_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (checkInsertBarrierAllBeforeOp(op)) {
      Location loc = op->getLoc();
      rewriter.setInsertionPoint(op);
      auto pipeAll = PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
      rewriter.create<hivm::PipeBarrierOp>(loc, pipeAll);
    }
  });
  bool isRegBasedArch =
      hacc::utils::isRegBasedArch(func_->getParentOfType<ModuleOp>());
  if (isRegBasedArch) {
    InjectSetWaitPipeMPipeMTE1ForAllMmadL1();
  }
}

void InjectSyncAnalysis::InjectSetWaitPipeMPipeMTE1ForAllMmadL1() {
  MLIRContext *ctx = func_->getContext();
  IRRewriter rewriter(ctx);
  func_->walk<WalkOrder::PreOrder>([&](hivm::MmadL1Op op) {
    auto loc = op->getLoc();
    auto ctx = op->getContext();
    rewriter.setInsertionPoint(op.getOperation());
    auto eventId0Attr = EventAttr::get(ctx, hivm::EVENT::EVENT_ID0);
    auto eventId1Attr = EventAttr::get(ctx, hivm::EVENT::EVENT_ID1);
    auto setPipe = PipeAttr::get(ctx, hivm::PIPE::PIPE_M);
    auto waitPipe = PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE1);
    rewriter.create<hivm::SetFlagOp>(loc, setPipe, waitPipe, eventId0Attr,
                                     Value{});
    rewriter.create<hivm::SetFlagOp>(loc, setPipe, waitPipe, eventId1Attr,
                                     Value{});
    rewriter.setInsertionPointAfter(op.getOperation());
    rewriter.create<hivm::WaitFlagOp>(loc, setPipe, waitPipe, eventId0Attr,
                                      Value{});
    rewriter.create<hivm::WaitFlagOp>(loc, setPipe, waitPipe, eventId1Attr,
                                      Value{});
  });
}

void InjectSyncAnalysis::AutoInjectSync(bool enableUnitFlag,
                                        bool assumeAliveLoops) {
  MemoryDependentAnalyzer memAnalyzer;
  SyncIRs syncIR;
  SyncOperations syncOperations;
  Buffer2MemInfoMap buffer2MemInfoMap;

  IRTranslator trans(syncIR, memAnalyzer, buffer2MemInfoMap, func_,
                     SyncAnalysisMode::NORMALSYNC);
  trans.Build();
  LLVM_DEBUG(llvm::dbgs() << "IRTranslator\n");
  LLVM_DEBUG(SyncDebug(syncIR).PrintSyncIr());

  // Single instruction or no instruction, no need to insert synchronization.
  if (syncIR.size() <= 1) {
    return;
  }

  SyncAnalyzer syncAnalyzer(syncIR, memAnalyzer, syncOperations, func_,
                            SyncAnalysisMode::NORMALSYNC, enableUnitFlag,
                            assumeAliveLoops);
  syncAnalyzer.SetBuffer2ParentAliasBuffer(trans.GetBuffer2ParentAliasBuffer());
  syncAnalyzer.Plan();
  LLVM_DEBUG(llvm::dbgs() << "SyncAnalyzer\n");
  LLVM_DEBUG(SyncDebug(syncIR).PrintSyncIr());

  MoveSyncState syncMove(syncIR, syncOperations);
  syncMove.StateOptimize();
  LLVM_DEBUG(llvm::dbgs() << "MoveSyncState\n");
  LLVM_DEBUG(SyncDebug(syncIR).PrintSyncIr());

  RemoveRedundantSync removeRedundantSync(syncIR, syncOperations);
  removeRedundantSync.Plan();
  LLVM_DEBUG(llvm::dbgs() << "RemoveRedundantSync\n");
  LLVM_DEBUG(SyncDebug(syncIR).PrintSyncIr());

  SyncEventIdAllocation eventIdAllocation(syncIR, syncOperations);
  eventIdAllocation.Allocate();
  LLVM_DEBUG(llvm::dbgs() << "SyncEventIdAllocation\n");
  LLVM_DEBUG(SyncDebug(syncIR).PrintSyncIr());

  SyncCodegen syncCodegen(syncIR, func_, SyncAnalysisMode::NORMALSYNC);
  syncCodegen.Build();
  LLVM_DEBUG(llvm::dbgs() << "SyncCodegen\n");
  LLVM_DEBUG(SyncDebug(syncIR).PrintSyncIr());
}

void InjectSyncPass::runOnOperation() {
  auto func = getOperation();
  if (hacc::utils::isHost(func)) {
    return;
  }
  if (func->hasAttr(hivm::VectorFunctionAttr::name)) {
    return;
  }
  InjectSyncAnalysis injectsyncAnalysis(func);
  if (syncMode == SyncMode::BARRIERALL) {
    injectsyncAnalysis.InjectSyncAll();
  } else if (syncMode == SyncMode::NORMAL) {
    injectsyncAnalysis.AutoInjectSync(enableUnitFlag, assumeAliveLoops);
  } else {
    llvm_unreachable("Illegal synchronization mode! ");
  }
}

std::unique_ptr<Pass>
mlir::hivm::createInjectSyncPass(const InjectSyncOptions &options) {
  return std::make_unique<InjectSyncPass>(options);
}
