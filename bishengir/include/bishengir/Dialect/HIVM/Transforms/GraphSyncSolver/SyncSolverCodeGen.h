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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/CustomMacroSync.h"
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
#include "llvm/ADT/MapVector.h"
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

  // Linearized occurrence sequence (syncIr): each Occurrence is one appearance
  // of an operation in analysis order.
  std::vector<std::unique_ptr<Occurrence>> syncIr;

  // Set of RW operations that expose unit-flag feature and need special
  // handling.
  llvm::SetVector<RWOperation *> unitFlagFeaturedOps;

  // Map sync ops to before/after operations.
  SyncMap syncMapBefore, syncMapAfter;

private:
  bool resultFuncIrWasGenerated{false};

  // Per-multibuffer loop cached helper: modular indices created during codegen
  // and reused to select between multi-buffer event ids.
  // Key -> value: (loop, mod, offset, counterBlock) -> modular slot index.
  //   loop         := the multi-buffer loop that rotates
  //   mod          := number of event ids, i.e. the modulo
  //   offset       := preload shift; 0 is the plain `counter mod mod` index
  //   counterBlock := the block that owns the counter
  // counterBlock is part of the key because two id sets of one loop may live in
  // different blocks and therefore own different counters.
  llvm::DenseMap<std::tuple<LoopLikeOpInterface, int64_t, int64_t, Block *>,
                 Value>
      nestedIndexModularMem;

  // Counter block of an id set, from collectCounterAnchors.
  // counterBlock := the block that owns the counter.
  // vetoed := no block may own it, and the set keeps the loop counter:
  //   consumers span regions => no block holds the whole handshake
  //   a flag lands outside it => its id could not read that counter
  // => ids on two cadences never match their wait.
  struct IdAnchor {
    Block *counterBlock = nullptr;
    bool vetoed = false;
  };
  // Key := `(candidate block, id set)`.
  // candidate := the block the handshake's own buffers resolve to.
  // => two handshakes sharing one id set keep separate entries.
  // Their vetos and block choices cannot reach each other.
  // `std::map`, not DenseMap: LLVM has no DenseMapInfo for these keys.
  llvm::DenseMap<LoopLikeOpInterface,
                 std::map<std::pair<Block *, llvm::SmallVector<int64_t>>,
                          IdAnchor>>
      multiBufferIdAnchors;

  // Select value per `(loop, preload offset, counter block, id set)`.
  // It chooses which buffer/event id to use at runtime.
  llvm::DenseMap<std::tuple<LoopLikeOpInterface, int64_t>,
                 std::map<std::pair<Block *, llvm::SmallVector<int64_t>>,
                          Value>>
      bufferSelectedMem;

  llvm::DenseMap<Value, std::map<llvm::SmallVector<int64_t>, Value>>
      explicitBufferSelectedMem;

  // Per-MMAD L1 op arguments collected during sync codegen insertion.
  llvm::MapVector<Operation *, MmadL1SyncArgs> mmadl1SyncArgsMap;

  // Per-MMAD MxL1 op arguments collected during sync codegen insertion.
  llvm::MapVector<hivm::MmadMxL1Op, MmadMxL1SyncArgs> mmadMxL1SyncArgsMap;

  // Mapping to cache loop DB conditions used during codegen insertion.
  llvm::DenseMap<LoopLikeOpInterface, Value> loopDBCondMap;

  CustomMacroSyncCodegenState customMacroCodegen;
  BufferPoolSync bufferPoolSync;

public:
  CodeGenerator(const SyncSolverOptions &options) : options(options) {}

  CodeGenerator(std::unique_ptr<SyncSolverBase> solver)
      : options(solver->options) {
    init(std::move(solver));
  }

  void init(std::unique_ptr<SyncSolverBase> solver) {
    auto [syncBefore, syncAfter] = solver->getBeforeAfterSyncMaps();
    syncMapBefore = std::move(syncBefore);
    syncMapAfter = std::move(syncAfter);
    bufferPoolSync = std::move(solver->bufferPoolSync);
    bufferPoolSync.clearEventIdNodes();
    funcOp = solver->funcOp;
    funcIr = std::move(solver->funcIr);
    syncIr = std::move(solver->syncIr);
    unitFlagFeaturedOps = std::move(solver->unitFlagFeaturedOps);
    customMacroCodegen.setResolvedSlotEventIds(
        std::move(solver->customMacroSync.resolvedSlotEventIds()));
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

  void insertBlockAllOp(IRRewriter &rewriter, OperationBase *opBase,
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

  llvm::SmallVector<int64_t>
  getEventIdsWithOffset(const llvm::SmallVector<int64_t> &eventIds,
                        int64_t offset);

  Value getNestedIndexModular(IRRewriter &rewriter,
                              LoopLikeOpInterface multibufferLoop, int64_t mod,
                              int64_t offset = 0,
                              Block *counterBlock = nullptr);

  void collectCounterAnchors();

  /// Block the flag of \p opBase lands in; mirrors setProperInsertionPoint
  /// without creating the blocks it creates on demand.
  /// Post: null when that block cannot be known yet.
  Block *getFlagLandingBlock(OperationBase *opBase);

  /// Counter block of \p ids in \p loop, for the handshake whose consumer block
  /// is \p candidate.
  /// Post: null when the set keeps the loop counter.
  Block *getMultiBufferIdAnchor(LoopLikeOpInterface loop, Block *candidate,
                                llvm::ArrayRef<int64_t> ids);

  Value getPoolSlotEventId(IRRewriter &rewriter, SetWaitOp *syncOp);

  Value getMultiBufferSelectOp(IRRewriter &rewriter, SetWaitOp *syncOp);

  Value getMultiBufferSelectOpConsecutive(IRRewriter &rewriter,
                                          SetWaitOp *syncOp);

  Value getCVPipeliningSelectOp(IRRewriter &rewriter, SetWaitOp *syncOp);

  Value getCVPipeliningSelectOpConsecutive(IRRewriter &rewriter,
                                           SetWaitOp *syncOp);

  Value getMultiBufferBlockSelectOp(IRRewriter &rewriter, SetWaitOp *syncOp);

  Value getLoopDBCond(IRRewriter &rewriter, Operation *op);

  void insertPipeMPipeMte1OuterBwdPairs(IRRewriter &rewriter);

  void insertMmadL1SyncArgs(IRRewriter &rewriter);

  llvm::LogicalResult handleMmadMxL1SyncOps(IRRewriter &rewriter,
                                            OperationBase *opBase,
                                            SyncOp *syncOp);

  void insertMmadMxL1SyncArgs(IRRewriter &rewriter);

  // Create the function-scoped L0 ping-pong signal and attach it to MMAD ops.
  void insertPingPongSignal(IRRewriter &rewriter);

  void handleUnitFlagEnabledOps(IRRewriter &rewriter);

  // Ensure a barrier-all exists before function return.
  void insertBarrierAllBeforeReturn(IRRewriter &rewriter);
};

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVERCODEGEN_H
