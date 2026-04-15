//===---------- SyncSolverCodeGen.h ---- Graph Sync Solver ----------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERCODEGEN_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERCODEGEN_H

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <utility>

namespace mlir::hivm::syncsolver {

class CodeGenerator {

public:
  // Configuration options.
  const SyncSolverOptions options;

  // Original MLIR function being processed (may be null for test-only Solver).
  func::FuncOp funcOp;

  // In-memory hierarchical IR (Function -> Scopes -> Ops) used by the solver.
  std::unique_ptr<OperationBase> funcIr;

  // Set of RW operations that expose unit-flag feature and need special
  // handling.
  llvm::DenseSet<RWOperation *> unitFlagFeaturedOps;

private:
  bool resultFuncIrWasGenerated{false};

  // Per-multibuffer loop cached helper: nested index modular counters created
  // during codegen and reused to select between multi-buffer event ids.
  llvm::DenseMap<std::tuple<LoopLikeOpInterface, int64_t, int64_t>, Value>
      nestedIndexModularMem;

  // Cache mapping a loop + (eventIdA,eventIdB) pair to the created select Value
  // that chooses which buffer/event id to use at runtime.
  llvm::DenseMap<std::pair<LoopLikeOpInterface, int64_t>,
                 std::map<llvm::SmallVector<int64_t>, Value>>
      bufferSelectedMem;

  // Per-MMAD L1 op arguments collected during sync codegen insertion.
  llvm::DenseMap<hivm::MmadL1Op, MmadL1SyncArgs> mmadl1SyncArgsMap;

  // Mapping to cache loop DB conditions used during codegen insertion.
  llvm::DenseMap<LoopLikeOpInterface, Value> loopDBCondMap;

  SyncMap syncMapBefore, syncMapAfter;

public:
  CodeGenerator() = delete;

  CodeGenerator(std::unique_ptr<Solver> solver) : options(solver->options) {
    init(std::move(solver));
  }

  void init(std::unique_ptr<Solver> solver) {
    auto [syncBefore, syncAfter] = solver->getBeforeAfterSyncMaps();
    syncMapBefore = std::move(syncBefore);
    syncMapAfter = std::move(syncAfter);
    funcOp = solver->funcOp;
    funcIr = std::move(solver->funcIr);
    unitFlagFeaturedOps = std::move(solver->unitFlagFeaturedOps);
  }

  // Insert sync ops into func-ir.
  void generateFuncIrResultOps();

  // Insert sync ops into actual MLIR IR using rewriter.
  void generateResultOps();

private:
  // Location/IR insertion helpers and event id value creation.
  Location getProperLoc(OperationBase *opBase);

  void setProperInsertionPoint(IRRewriter &rewriter, OperationBase *opBase,
                               bool insertAfterOp);

  void insertBlockOp(IRRewriter &rewriter, OperationBase *opBase,
                     BarrierOp *barrierOp, bool insertAfterOp);

  void insertBarrierOp(IRRewriter &rewriter, OperationBase *opBase,
                       BarrierOp *barrierOp, bool insertAfterOp);

  void insertSetFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                       SetFlagOp *setFlagOp, bool insertAfterOp);

  void insertWaitFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                        WaitFlagOp *waitFlagOp, bool insertAfterOp);

  void insertSetBlockFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                            SetFlagOp *setFlagOp, bool insertAfterOp);

  void insertWaitBlockFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                             WaitFlagOp *waitFlagOp, bool insertAfterOp);

  Value getEventIdValue(IRRewriter &rewriter, SetWaitOp *setWaitOp,
                        Location loc);

  llvm::LogicalResult handleMmadL1SyncOps(IRRewriter &rewriter,
                                          OperationBase *opBase,
                                          SyncOp *syncOp);

  Value getNestedIndexModular(IRRewriter &rewriter,
                              LoopLikeOpInterface multibufferLoop,
                              int64_t eventIdNum, int64_t preloadOffset);

  Value getMultiBufferSelectOp(IRRewriter &rewriter, SetWaitOp *syncOp);

  Value getMultiBufferSelectOpConsecutive(IRRewriter &rewriter,
                                          SetWaitOp *syncOp);

  Value getCVMultiBufferSelectOp(IRRewriter &rewriter, SetWaitOp *syncOp);

  Value getCVMultiBufferSelectOpConsecutive(IRRewriter &rewriter,
                                            SetWaitOp *syncOp);

  Value getMultiBufferBlockSelectOp(IRRewriter &rewriter, SetWaitOp *syncOp);

  Value getLoopDBCond(IRRewriter &rewriter, Operation *op);

  void insertMmadL1SyncArgs(IRRewriter &rewriter);

  void handleUnitFlagEnabledOps(IRRewriter &rewriter);

  // Ensure a barrier-all exists before function return.
  void insertBarrierAllBeforeReturn(IRRewriter &rewriter);
};

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERCODEGEN_H
