//===------------- InjectSync.cpp ----Auto Inject Sync --------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/InjectSync.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncDebug.h"
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
  MLIRContext *ctx = func_->getContext();
  IRRewriter rewriter(ctx);
  func_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getDialect()->getNamespace() ==
            HIVMDialect::getDialectNamespace() ||
        mlir::isa<func::ReturnOp>(op)) {
      Location loc = op->getLoc();
      rewriter.setInsertionPoint(op);
      auto pipeAll = PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
      rewriter.create<hivm::PipeBarrierOp>(loc, pipeAll);
    }
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

  // Single instruction or no instruction, no need to insert synchronization.
  if (syncIR.size() <= 1) {
    return;
  }

  SyncAnalyzer syncAnalyzer(syncIR, memAnalyzer, syncOperations, func_,
                            SyncAnalysisMode::NORMALSYNC, enableUnitFlag,
                            assumeAliveLoops);
  syncAnalyzer.SetBuffer2ParentAliasBuffer(trans.GetBuffer2ParentAliasBuffer());
  syncAnalyzer.Plan();

  MoveSyncState syncMove(syncIR, syncOperations);
  syncMove.StateOptimize();

  RemoveRedundantSync removeRedundantSync(syncIR, syncOperations);
  removeRedundantSync.Plan();

  SyncEventIdAllocation eventIdAllocation(syncIR, syncOperations);
  eventIdAllocation.Allocate();

  SyncCodegen syncCodegen(syncIR, func_, SyncAnalysisMode::NORMALSYNC);
  syncCodegen.Build();
}

void InjectSyncPass::runOnOperation() {
  auto func = getOperation();
  if (hacc::utils::isHost(func))
    return;
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
