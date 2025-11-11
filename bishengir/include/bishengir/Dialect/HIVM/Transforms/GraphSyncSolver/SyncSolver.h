//===------------- SyncSolver.h ---- Graph Sync Solver --------------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVER_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVER_H

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <deque>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

using SyncMap =
    std::map<mlir::hivm::syncsolver::OperationBase *,
             std::deque<std::unique_ptr<mlir::hivm::syncsolver::SyncOp>>>;
using SyncBeforeAfterMap = std::pair<SyncMap, SyncMap>;

namespace mlir::hivm::syncsolver {

struct ProcessingOrder {
  Occurrence *occ{nullptr};
  int start{-1};
  int end{-1};
  bool reverseOrder{false};
  bool isUseless{false};
  bool skip{false};
  ProcessingOrder(Occurrence *occ, int start, int end, bool reverseOrder,
                  bool isUseless, bool skip = false)
      : occ(occ), start(start), end(end), reverseOrder(reverseOrder),
        isUseless(isUseless), skip(skip) {}
};

class Solver {
public:
  uint64_t globalIndex{0};
  uint64_t globalCodeGenIndex{0};
  bool resultFuncIrWasGenerated{false};
  bool considerMergedBackwardSyncEventIds{true};
  bool disableMultiEventIdForBarrierAllPairs{true};
  bool reuseSyncPairToSaveEventIds{false};
  bool enableUnitFlagFeature{false};
  bool decomposeMmadl1Op{true};

  // Original MLIR function being processed (may be null for test-only Solver).
  func::FuncOp func;

  // In-memory hierarchical IR (Function -> Scopes -> Ops) used by the solver.
  std::unique_ptr<OperationBase> funcIr;

  // Linearized occurrence sequence (sync IR) built from funcIr, each Occurrence
  // represents one appearance of an operation in the sync-analysis order.
  std::vector<std::unique_ptr<Occurrence>> syncIr;

  // Map op -> list of occurrences in syncIr (quick lookup for an op's
  // occurrences).
  llvm::DenseMap<OperationBase *, std::vector<Occurrence *>> opAllOccurrences;

  // Collected conflict pairs chosen by the algorithm for insertion (and
  // persistent ones that survive multiple passes).
  std::vector<std::unique_ptr<ConflictPair>> chosenConflictedPairs,
      persistentChosenConflictedPairs;

  // Bookkeeping map used to record that a pair (scopeOp, op1, op2, setPipe,
  // waitPipe) has already been synchronized and which ConflictPair performed
  // it.
  llvm::DenseMap<std::tuple<OperationBase *, OperationBase *, OperationBase *,
                            hivm::PIPE, hivm::PIPE>,
                 ConflictPair *>
      syncedPairs, replacedWithReusableSyncedPairs;

  // For fast lookup of chosen conflicts relevant to a particular scope op.
  llvm::DenseMap<OperationBase *, llvm::DenseSet<ConflictPair *>>
      scopeOpChosenConflicts;

  // Chosen conflicts keyed by occurrence (scope occurrence) to allow retrieving
  // conflicts that affect a particular occurrence subtree.
  llvm::DenseMap<Occurrence *, llvm::DenseSet<ConflictPair *>>
      scopeOccChosenConflicts, persistentScopeOccChosenConflicts;

  // Chosen conflicts keyed by a pair of scope-occurrences, used when conflicts
  // are associated with a pair of sibling blocks (e.g., condition branches).
  llvm::DenseMap<std::pair<Occurrence *, Occurrence *>,
                 llvm::DenseSet<ConflictPair *>>
      scopeOccPairChosenConflicts, persistentScopeOccPairChosenConflicts;

  // Chosen conflicts keyed by a pair of operations (useful for reuse search).
  llvm::DenseMap<std::pair<OperationBase *, OperationBase *>,
                 llvm::DenseSet<ConflictPair *>>
      scopeOpPairChosenConflicts;

  // Processing order list created from syncIr that drives pairwise conflict
  // checks.
  std::vector<ProcessingOrder> processingOrders;

  // Set of processed occurrence pairs to avoid re-processing the same pair.
  llvm::DenseSet<std::pair<Occurrence *, Occurrence *>> processedOccPairs;

  // Occurrences marked skippable (exclusion set used during processing).
  llvm::DenseSet<Occurrence *> skipOcc;

  // Per-multibuffer loop cached helper: nested index modular counters created
  // during codegen and reused to select between multi-buffer event ids.
  llvm::DenseMap<LoopLikeOpInterface, Value> nestedIndexModularMem;

  // Cache mapping a loop + (eventIdA,eventIdB) pair to the created select Value
  // that chooses which buffer/event id to use at runtime.
  llvm::DenseMap<LoopLikeOpInterface,
                 llvm::DenseMap<std::pair<hivm::EVENT, hivm::EVENT>, Value>>
      bufferSelectedMem;

  // For a parent occurrence, list of its child occurrences.
  llvm::DenseMap<Occurrence *, llvm::SmallVector<Occurrence *>> occChildrenMem;

  // Accumulated backward-sync events for each operation (recorded instead of
  // inserting explicit conflict pairs). The outer map key is the scope op; the
  // inner map key is (setPipe, waitPipe) and value is the set of event ids
  // used.
  std::map<OperationBase *, llvm::DenseMap<std::pair<hivm::PIPE, hivm::PIPE>,
                                           llvm::DenseSet<hivm::EVENT>>>
      backwardSyncEvents, backwardSyncEventsAfterMerge;

  // Indices allocated during codegen walk: start/end and inclusive variants
  // used to evaluate ordering relationships between ops during merging checks.
  llvm::DenseMap<OperationBase *, uint64_t> codeGenStartIndex, codeGenEndIndex;
  llvm::DenseMap<OperationBase *, uint64_t> codeGenInclusiveStartIndex,
      codeGenInclusiveEndIndex;

  // Index of set/wait ops: key=(setPipe,waitPipe,eventId) -> ordered set of
  // (codegen-index, SetWaitOp*) for quick queries.
  llvm::DenseMap<std::tuple<hivm::PIPE, hivm::PIPE, hivm::EVENT>,
                 std::set<std::pair<uint64_t, SetWaitOp *>>>
      setWaitFlagOpsIndex;

  // Memoization of memory-conflict discovery between specific RWOperation
  // pairs.
  llvm::DenseMap<
      std::pair<syncsolver::RWOperation *, syncsolver::RWOperation *>,
      std::vector<std::pair<hivm::PIPE, hivm::PIPE>>>
      checkMemoryConflictsMem, checkTestMemoryConflictsMem;

  // Set of pipe pairs that were forced to barrier-all (no event ids available).
  llvm::DenseSet<std::pair<hivm::PIPE, hivm::PIPE>> barrierAllPairs;

  // Count-per-pipe-pair used to limit reuse of conflict pairs (reuse budget).
  llvm::DenseMap<std::pair<hivm::PIPE, hivm::PIPE>, int> reusePairs;

  // Tracks inserted barrier-all markers before occurrences: op -> set of (occ,
  // isUseless).
  llvm::DenseMap<OperationBase *,
                 llvm::DenseSet<std::pair<Occurrence *, int32_t>>>
      insertedBarrierAllBefore;

  // Per-MMAD L1 op arguments collected during sync codegen insertion.
  llvm::DenseMap<hivm::MmadL1Op, MmadL1SyncArgs> mmadl1SyncArgsMap;

  // Set of RW operations that expose unit-flag feature and need special
  // handling.
  llvm::DenseSet<RWOperation *> unitFlagFeaturedOps;

  // Mapping to cache loop DB conditions used during codegen insertion.
  llvm::DenseMap<LoopLikeOpInterface, Value> loopDBCondMap;

public:
  Solver(func::FuncOp func) : func(func) {
    syncsolver::OperationBase::resetLCAMem();
    syncsolver::Occurrence::resetLCAMem();
    auto funcOp = std::make_unique<syncsolver::Function>(func.getOperation());
    auto scopeOp = funcIrBuilder(func.getRegion(), funcOp.get());
    funcOp->body.push_back(std::move(scopeOp));
    funcIr = std::move(funcOp);
    syncIrBuilder(funcIr.get());
  }
  Solver(std::unique_ptr<OperationBase> funcIr) : funcIr(std::move(funcIr)) {
    syncsolver::OperationBase::resetLCAMem();
    syncsolver::Occurrence::resetLCAMem();
    syncIrBuilder(this->funcIr.get());
  }

  // Orchestrate the solving process (entry point).
  void solve(int runNum = 0);

  // Solve function variant used in tests (keeps internal behaviour compatible).
  void solveTest(int runNum = 0);

  // Insert sync ops into func-ir.
  void generateFuncIrResultOps();

  // Insert sync ops into actual MLIR IR using rewriter.
  void generateResultOps();

  // Build before/after maps of sync ops computed from chosen conflicts.
  SyncBeforeAfterMap getBeforeAfterSyncMaps();

private:
  // Reset solver internal bookkeeping prior to another pass.
  void reset();

  // Walk and process the generated processingOrders to choose conflicts.
  void processOrders();

  // Alternative processing used for tests.
  void processOrdersTest();

  // Convert MLIR Region into the in-memory funcIr Scope representation.
  std::unique_ptr<Scope> funcIrBuilder(Region &region, OperationBase *parentOp);

  // Create a decomposed representation for certain MMAD L1 ops if enabled.
  std::unique_ptr<OperationBase> getDecomposedMmadl1(hivm::MmadL1Op mmadl1Op,
                                                     OperationBase *parentOp);

  // Generate processing orders (various flavors) used by the main algorithm.
  void generateProcessingOrders(Occurrence *scopeOcc, int l, int r,
                                bool isUseless);

  void generateProcessingOrders(int l, int r, bool isUseless);

  void generateProcessingOrders(int l1, int r1, int l2, int r2, bool isUseless);

  // Build sync IR occurrences from the operation tree.
  void syncIrBuilder(OperationBase *op, Occurrence *parentOcc = nullptr,
                     int depth = 0, bool isUseless = false);

  // Collect pointer-like operands reachable from a Value.
  llvm::SmallVector<Value> collectPointerOps(Value val);

  // Extract memory-related Values from a list of pointer values.
  llvm::SmallVector<Value> getMemOps(const SmallVector<Value> &vals);

  // Return read and write memory operand lists for an MLIR operation.
  std::pair<llvm::SmallVector<Value>, llvm::SmallVector<Value>>
  getReadWriteMemOps(Operation *op);

  // Return a wrapped Load/Store RWOperation when encountering affine/memref
  // load/store ops.
  template <typename OP>
  typename std::enable_if<std::is_same_v<OP, memref::LoadOp> ||
                              std::is_same_v<OP, affine::AffineLoadOp> ||
                              std::is_same_v<OP, affine::AffineStoreOp> ||
                              std::is_same_v<OP, memref::StoreOp>,
                          std::unique_ptr<OperationBase>>::type
  getLoadStoreOp(OP *op, OperationBase *parentOp);

  // Check if given eventIdNum can be used without RW conflicts.
  bool
  checkEventIdNum(const llvm::SmallVector<llvm::SmallVector<int>> &memValsList1,
                  const llvm::SmallVector<llvm::SmallVector<int>> &memValsList2,
                  int lcmLen, int eventIdNum);

  // Multi-buffer/event-related helpers that determine if double event id can be
  // used.
  std::optional<LoopLikeOpInterface>
  checkDoubleMultiBufferEventId(const llvm::SmallVector<Value> &memValsList1,
                                const llvm::SmallVector<Value> &memValsList2);

  std::optional<LoopLikeOpInterface>
  checkDoubleMultiBufferEventId(hivm::PointerCastOp pointerCastOp1,
                                hivm::PointerCastOp pointerCastOp2);

  std::optional<LoopLikeOpInterface>
  checkDoubleMultiBufferEventId(RWOperation *rwOp1, RWOperation *rwOp2);

  // Determine how many event ids are needed for a particular occurrence pair.
  std::pair<uint32_t, LoopLikeOpInterface> getEventIdNum(Occurrence *occ1,
                                                         Occurrence *occ2,
                                                         hivm::PIPE setPipe,
                                                         hivm::PIPE waitPipe);

  // Helpers for test-mode event id num estimation.
  uint32_t getTestEventIdNum(RWOperation *rwOp1, RWOperation *rwOp2);

  uint32_t getTestEventIdNum(Occurrence *occ1, Occurrence *occ2,
                             hivm::PIPE setPipe, hivm::PIPE waitPipe);

  // Graph-based conflict checking and memory conflict detection helpers.
  bool checkGraphConflict(Occurrence *occ1, Occurrence *occ2,
                          hivm::PIPE startPipe, hivm::PIPE endPipe,
                          uint32_t eventIdNum);

  std::vector<std::pair<hivm::PIPE, hivm::PIPE>>
  checkMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2);

  std::vector<std::pair<hivm::PIPE, hivm::PIPE>>
  checkTestMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2);

  bool checkRWMemoryConflicts(const llvm::SmallVector<Value> &memValsList1,
                              const llvm::SmallVector<Value> &memValsList2);

  bool checkTestRWMemoryConflicts(
      const llvm::SmallVector<llvm::SmallVector<int>> &memValsList1,
      const llvm::SmallVector<llvm::SmallVector<int>> &memValsList2);

  bool checkPointerCastMemConflict(hivm::PointerCastOp pointerCastOp1,
                                   hivm::PointerCastOp pointerCastOp2);

  // Feasibility checks and bookkeeping accessors used by the solver loop.
  bool checkImpossibleOpPair(OperationBase *op1, OperationBase *op2);

  bool checkImpossibleOccPair(Occurrence *occ1, Occurrence *occ2);

  bool checkAlreadySynced(Occurrence *occ1, Occurrence *occ2);

  bool checkAlreadySyncedWithUnitFlag(RWOperation *rwOp1, RWOperation *rwOp2);

  bool skipMMad1DecomposedLoopOpt(Occurrence *occ1, Occurrence *occ2);

  // Event-id allocation and reuse helpers.
  llvm::SmallVector<hivm::EVENT>
  getAvailableEventIds(ConflictPair *conflictPair);

  llvm::SmallVector<hivm::EVENT>
  getAnyAvailableEventId(ConflictPair *conflictPair, uint32_t count,
                         bool reversedPriority);

  llvm::SmallVector<hivm::EVENT>
  getAnyAvailableMultiBufferEventIds(ConflictPair *conflictPair, uint32_t count,
                                     bool reversedPriority);

  // Visit tracking helpers for occurrence pairs.
  bool checkVisited(Occurrence *occ1, Occurrence *occ2);

  bool checkSkippable(Occurrence *occ);

  // Bookkeeping for previously synchronized pairs within a scope to reuse their
  // event-ids.
  std::optional<llvm::SmallVector<hivm::EVENT>>
  getOldEventIdIfExists(OperationBase *scopeOp, Occurrence *occ1,
                        Occurrence *occ2, ConflictPair *conflictPair);

  void memorizeSyncedPair(OperationBase *scopeOp, ConflictPair *conflictPair);

  void memorizeReusedSyncedPair(OperationBase *scopeOp,
                                ConflictPair *conflictPair,
                                ConflictPair *reusedConflictPair);

  void forgetSyncedPair(OperationBase *scopeOp, ConflictPair *conflictPair);

  // Utilities to map an occurrence pair to their set/wait occurrences.
  std::pair<Occurrence *, Occurrence *> getSetWaitOcc(Occurrence *occ1,
                                                      Occurrence *occ2);

  // Convenience to insert barrier-all before a given occurrence.
  void insertBarrierAllBefore(Occurrence *occ, bool isUseless,
                              bool isPersistent = false);

  // Determine the direction (backward) of a synchronization candidate.
  bool isBackwardSync(Occurrence *occ1, Occurrence *occ2);

  // Reuse existing conflict pairs where possible to save event ids.
  ConflictPair *getReusableConflictPair(
      ConflictPair *conflictPair,
      const llvm::DenseSet<ConflictPair *> &conflictPairsSet);

  bool reuseConflictPair(ConflictPair *conflictPair, Occurrence *scopeOcc1,
                         Occurrence *scopeOcc2);

  // Primary handler invoked to register/record a found conflict.
  void handleConflict(Occurrence *occ1, Occurrence *occ2, hivm::PIPE setPipe,
                      hivm::PIPE waitPipe, bool isUseless, uint32_t eventIdNum,
                      LoopLikeOpInterface multibufferLoopPar);

  // Location/IR insertion helpers and event id value creation.
  Location getProperLoc(OperationBase *opBase);

  void setProperInsertionPoint(IRRewriter &rewriter, OperationBase *opBase,
                               bool insertAfterOp);

  void insertBarrierOp(IRRewriter &rewriter, OperationBase *opBase,
                       BarrierOp *barrierOp, bool insertAfterOp);

  void insertSetFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                       SetFlagOp *setFlagOp, bool insertAfterOp);

  void insertWaitFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                        WaitFlagOp *waitFlagOp, bool insertAfterOp);

  Value getEventIdValue(IRRewriter &rewriter, SetWaitOp *setWaitOp,
                        Location loc);

  llvm::LogicalResult handleMmadL1SyncOps(IRRewriter &rewriter,
                                          OperationBase *opBase,
                                          SyncOp *syncOp);

  Value getMultiBufferSelectOp(IRRewriter &rewriter, SetWaitOp *syncOp);

  void insertMultiBufferSetFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                                  SetFlagOp *setFlagOp, bool insertAfterOp);

  void insertMultiBufferWaitFlagOp(IRRewriter &rewriter, OperationBase *opBase,
                                   WaitFlagOp *waitFlagOp, bool insertAfterOp);

  Value getLoopDBCond(IRRewriter &rewriter, Operation *op);

  void insertMmadL1SyncArgs(IRRewriter &rewriter);

  // Unit-flag helpers (detection and applying modes to ops).
  hivm::UNIT_FLAG getUnitFlagMode(RWOperation *rwOp);

  Value getIsNotDeadLoopValue(scf::ForOp forOp, Location loc,
                              IRRewriter &rewriter);

  std::optional<mlir::Value> getUnitFlagCond(IRRewriter &rewriter,
                                             RWOperation *rwOp);

  void handleUnitFlagEnabledOps(IRRewriter &rewriter);

  // Ensure a barrier-all exists before function return.
  void insertBarrierAllBeforeReturn(IRRewriter &rewriter);

  // Helper utilities for iter/loop occurrence finding and L0 optimizations.
  Occurrence *getFirstIterOcc(Occurrence *occ, Occurrence *parOcc);

  Occurrence *getLastIterOcc(Occurrence *occ, Occurrence *parOcc);

  std::pair<Occurrence *, Occurrence *>
  checkAndApplyMmadl0LoopOpt(ConflictPair *conflictPair, Occurrence *occ1,
                             Occurrence *occ2, Occurrence *parOcc1,
                             Occurrence *parOcc2);

  // Unit-flag pattern checks used to transform sync into unit-flag modes.
  std::optional<std::pair<UNIT_FLAG, UNIT_FLAG>>
  checkUnitFlagPatterns(ConflictPair *conflictPair, Occurrence *occ1,
                        Occurrence *occ2, Occurrence *parentLCALoopOcc);

  std::optional<std::pair<UNIT_FLAG, UNIT_FLAG>>
  checkMmadl1FixpipeUnitFlagPattern(RWOperation *rwOp1, RWOperation *rwOp2,
                                    hivm::PIPE pipe1, hivm::PIPE pipe2);

  std::optional<std::pair<hivm::UNIT_FLAG, hivm::UNIT_FLAG>>
  checkMmadl1FixpipeSingleForLoopUnitFlagPattern(RWOperation *rwOp1,
                                                 RWOperation *rwOp2,
                                                 hivm::PIPE pipe1,
                                                 hivm::PIPE pipe2,
                                                 bool rw1IsFrontOcc);

  std::pair<bool, Occurrence *> checkMmadl0BackwardSyncOpt(Occurrence *loopOcc);

  void resetAndBuildSetWaitOpIndex(SyncMap &syncMapBefore,
                                   SyncMap &syncMapAfter);

  void collectSetWaitOpsIndexes(OperationBase *op, SyncMap &syncMapBefore,
                                SyncMap &syncMapAfter);

  // Merge-related helpers for backward sync events and scope-level
  // optimizations.
  bool checkMergeable(Scope *scopeOp, hivm::PIPE pipeSrc, hivm::PIPE pipeDst,
                      hivm::EVENT eventId, bool shouldBeUsedAtleastOnce);

  void mergeBackwardSyncEventIds(OperationBase *op);

  void mergeRootBackwardSyncEventIds(OperationBase *op);

  void pickAndInsertABarrierAll();
};

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVER_H
