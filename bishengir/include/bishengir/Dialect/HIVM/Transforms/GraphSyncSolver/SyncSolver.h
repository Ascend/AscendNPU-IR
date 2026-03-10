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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/EventIdSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIRTranslator.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::hivm::syncsolver {

class Solver {
public:
  // Configuration options.
  const SyncSolverOptions options;

  // Original MLIR function being processed (may be null for test-only Solver).
  func::FuncOp funcOp;

  // In-memory hierarchical IR (Function -> Scopes -> Ops) used by the solver.
  std::unique_ptr<OperationBase> funcIr;

  // Linearized occurrence sequence (sync IR) built from funcIr, each Occurrence
  // represents one appearance of an operation in the sync-analysis order.
  std::vector<std::unique_ptr<Occurrence>> syncIr;

  // Set of RW operations that expose unit-flag feature and need special
  // handling.
  llvm::DenseSet<RWOperation *> unitFlagFeaturedOps;

  // Collected conflict pairs chosen by the algorithm for insertion (and
  // persistent ones that survive multiple passes).
  std::vector<std::unique_ptr<ConflictPair>> chosenConflictedPairs,
      persistentChosenConflictedPairs;

protected:
  int64_t globalSetWaitIndex{0};
  int64_t maxReuseNum{20};
  int64_t maxRunNum{99};
  bool moveBackwardSyncPairsToOutmostLoop{false};
  bool dontMoveBackwardSyncPairsToOutmostLoop{false};

  llvm::DenseMap<std::tuple<hivm::PIPE, hivm::PIPE>,
                 std::unique_ptr<EventIdSolver>>
      eventIdSolver;

  // Map op -> list of occurrences in syncIr (quick lookup for an op's
  // occurrences).
  llvm::DenseMap<OperationBase *, std::vector<Occurrence *>> opAllOccurrences;

  // Bookkeeping map used to record that a pair (scopeOp, op1, op2, setPipe,
  // waitPipe) has already been synchronized and which ConflictPair performed
  // it.
  llvm::DenseMap<std::tuple<OperationBase *, OperationBase *, OperationBase *,
                            CorePipeInfo, CorePipeInfo>,
                 llvm::DenseSet<ConflictPair *>>
      syncedPairs;

  llvm::DenseMap<std::tuple<OperationBase *, OperationBase *, OperationBase *,
                            CorePipeInfo, CorePipeInfo>,
                 ConflictPair *>
      replacedWithReusableSyncedPairs;

  // Chosen conflicts keyed by occurrence (scope occurrence) to allow retrieving
  // conflicts that affect a particular occurrence subtree.
  llvm::DenseMap<Occurrence *, llvm::DenseSet<ConflictPair *>>
      scopeOccChosenConflicts, persistentScopeOccChosenConflicts;

  // Chosen conflicts keyed by a pair of scope-occurrences, used when conflicts
  // are associated with a pair of sibling blocks (e.g., condition branches).
  llvm::DenseMap<std::pair<Occurrence *, Occurrence *>,
                 llvm::DenseSet<ConflictPair *>>
      scopeOccPairChosenConflicts, persistentScopeOccPairChosenConflicts;

  // Processing order list created from syncIr that drives pairwise conflict
  // checks.
  std::vector<ProcessingOrder> processingOrders;

  // Set of processed occurrence pairs to avoid re-processing the same pair.
  llvm::DenseSet<std::pair<Occurrence *, Occurrence *>> processedOccPairs;

  // Occurrences marked skippable (exclusion set used during processing).
  llvm::DenseMap<uint8_t, llvm::DenseSet<Occurrence *>> skipOcc;

  // Accumulated backward-sync events for each operation (recorded instead of
  // inserting explicit conflict pairs). The outer map key is the scope op; the
  // inner map key is (setPipe, waitPipe) and value is the set of event ids
  // used.
  llvm::MapVector<OperationBase *,
                  llvm::DenseMap<std::tuple<CorePipeInfo, CorePipeInfo>,
                                 llvm::DenseMap<int64_t, int64_t>>>
      backwardSyncEvents;

  llvm::MapVector<OperationBase *,
                  llvm::DenseSet<std::tuple<CorePipeInfo, CorePipeInfo>>>
      backwardSyncEventsAfterMerge;

  // Memoization of memory-conflict discovery between specific RWOperation
  // pairs.
  llvm::DenseMap<
      std::pair<syncsolver::RWOperation *, syncsolver::RWOperation *>,
      llvm::SmallVector<std::tuple<CorePipeInfo, CorePipeInfo>>>
      checkMemoryConflictsMem;

  // Set of pipe pairs that were forced to barrier-all (no event ids available).
  llvm::DenseSet<std::tuple<CorePipeInfo, CorePipeInfo>> barrierAllPairs;

  // Set of pipe pairs for which multi-event-id usage is disabled.
  llvm::DenseSet<std::tuple<CorePipeInfo, CorePipeInfo>>
      disabledMultiEventIdPairs;

  // Count-per-pipe-pair used to limit reuse of conflict pairs (reuse budget).
  llvm::DenseMap<std::tuple<CorePipeInfo, CorePipeInfo>, int> reusePairs,
      reusedPairs;

  // Tracks inserted barrier-all markers before occurrences: op -> set of (occ,
  // isUseless).
  llvm::DenseMap<OperationBase *,
                 llvm::DenseSet<std::pair<Occurrence *, int32_t>>>
      insertedBarrierAllBefore;

  // Indices allocated during codegen walk: start/end and inclusive variants
  // used to evaluate ordering relationships between ops during merging checks.
  llvm::DenseMap<OperationBase *, int64_t> setWaitStartIndex, setWaitEndIndex,
      setWaitStartIndexInclusive, setWaitEndIndexInclusive;

  // Index of set/wait ops: key=(setPipe,waitPipe,eventId) -> ordered set of
  // (codegen-index, SetWaitOp*) for quick queries.
  llvm::DenseMap<std::tuple<hivm::PIPE, hivm::PIPE, int64_t>,
                 std::set<std::pair<int64_t, SetWaitOp *>>>
      setWaitFlagOpsIndex;

public:
  Solver() = delete;
  virtual ~Solver() = default;

  Solver(std::unique_ptr<IRTranslator> irTranslator)
      : options(irTranslator->options) {
    init(std::move(irTranslator));
  }

  // Orchestrate the solving process (entry point).
  void solve();

  // Build before/after maps of sync ops computed from chosen conflicts.
  SyncBeforeAfterMap getBeforeAfterSyncMaps();

protected:
  void init(std::unique_ptr<IRTranslator> irTranslator) {
    funcOp = irTranslator->funcOp;
    funcIr = std::move(irTranslator->funcIr);
    syncIr = std::move(irTranslator->syncIr);
    unitFlagFeaturedOps = std::move(irTranslator->unitFlagFeaturedOps);
    opAllOccurrences = std::move(irTranslator->opAllOccurrences);
    processingOrders = std::move(irTranslator->processingOrders);
  }

  // Reset solver internal bookkeeping prior to another pass.
  void reset(bool resetEventIdRanOutOpts = false);

  llvm::LogicalResult runSolver(bool enableOpts1 = true,
                                bool enableOpts2 = true);

  // Reset unit-flag related bookkeeping prior to another pass.
  void resetUnitFlag();

  // Walk and process the generated processingOrders to choose conflicts.
  void processOrders();

  virtual void processConflict(Occurrence *occ1, Occurrence *occ2,
                               RWOperation *rwOp1, RWOperation *rwOp2,
                               bool isUseless);

  std::optional<LoopLikeOpInterface>
  getMultiBufferLoop(const llvm::SmallVector<MemInfo> &memInfoList1,
                     const llvm::SmallVector<MemInfo> &memInfoList2);
  std::optional<LoopLikeOpInterface> getMultiBufferLoop(RWOperation *rwOp1,
                                                        RWOperation *rwOp2);
  std::optional<EventIdInfo> getMultiBufferEventIdInfo(Occurrence *occ1,
                                                       Occurrence *occ2,
                                                       RWOperation *rwOp1,
                                                       RWOperation *rwOp2);

  // Determine how many event ids are needed for a particular occurrence pair.
  EventIdInfo getEventIdInfo(Occurrence *occ1, Occurrence *occ2,
                             RWOperation *rwOp1, RWOperation *rwOp2,
                             CorePipeInfo corePipeSrc,
                             CorePipeInfo corePipeDst);

  std::optional<EventIdInfo> checkCVMultiBufferEventIdInfo(RWOperation *rwOp1,
                                                           RWOperation *rwOp2);
  std::optional<EventIdInfo> checkMultiBufferEventIdInfo(Occurrence *occ1,
                                                         Occurrence *occ2,
                                                         RWOperation *rwOp1,
                                                         RWOperation *rwOp2);

  // Graph-based conflict checking and memory conflict detection helpers.
  bool checkGraphConflict(
      Occurrence *occ1, Occurrence *occ2, CorePipeInfo corePipeSrc,
      CorePipeInfo corePipeDst, EventIdInfo eventIdInfo,
      std::optional<int> startIndex = {}, std::optional<int> endIndex = {},
      const llvm::SmallVector<ConflictPair *> &extraConflictPairs = {},
      const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs = {});

  llvm::SmallVector<std::tuple<CorePipeInfo, CorePipeInfo>>
  checkMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2);

  bool checkMemoryConflictBetweenOccExclusive(Occurrence *occ1,
                                              Occurrence *occ2);

  // Feasibility checks and bookkeeping accessors used by the solver loop.
  bool checkImpossibleOccPair(Occurrence *occ1, Occurrence *occ2);

  bool checkSkipCrossCorePair(Occurrence *occ1, Occurrence *occ2);

  bool checkSkipParallelLoop(Occurrence *occ1, Occurrence *occ2);

  bool checkAlreadySynced(Occurrence *occ1, Occurrence *occ2);

  bool checkAlreadySyncedWithUnitFlag(Occurrence *occ1, Occurrence *occ2);

  bool skipMMad1DecomposedLoopOpt(Occurrence *occ1, Occurrence *occ2);

  bool checkSyncOpsConflicts(ConflictPair *conflictPair1,
                             ConflictPair *conflictPair2);

  // Check whether two ConflictPair ranges/event mapping intersect (same
  // pipes/events).
  bool checkIntersect(ConflictPair *conflictPair1, ConflictPair *conflictPair2);

  // Event-id allocation and reuse helpers.
  std::vector<ConflictPair *>
  getIntersectingConflictPairs(ConflictPair *conflictPair);

  // Visit tracking helpers for occurrence pairs.
  bool checkVisited(Occurrence *occ1, Occurrence *occ2);

  bool checkSkippable(bool reverseOrder, Occurrence *occ);

  // Bookkeeping for previously synchronized pairs within a scope to reuse their
  // event-ids.
  EventIdNode *getOldEventIdNodeIfExists(ConflictPair *conflictPair);

  void memorizeSyncedPair(ConflictPair *conflictPair);

  llvm::DenseSet<ConflictPair *>
  getMemorizedSyncedPairs(ConflictPair *conflictPair);

  void memorizeReusedSyncedPair(ConflictPair *conflictPair,
                                ConflictPair *reusedConflictPair);

  void forgetSyncedPair(ConflictPair *conflictPair);

  // Utilities to map an occurrence pair to their set/wait occurrences.
  std::pair<Occurrence *, Occurrence *> getSetWaitLCAPairOcc(Occurrence *occ1,
                                                             Occurrence *occ2);
  std::pair<Occurrence *, Occurrence *> getSetWaitOcc(Occurrence *occ1,
                                                      Occurrence *occ2);
  std::pair<Occurrence *, Occurrence *> getFixedSetWaitOcc(Occurrence *occ1,
                                                           Occurrence *occ2);

  Occurrence *getBarrierWaitOcc(Occurrence *occ1, Occurrence *occ2);

  std::optional<std::pair<Occurrence *, Occurrence *>>
  getFunctionBlockSetWaitOcc(Occurrence *occ1, Occurrence *occ2);

  std::optional<std::pair<Occurrence *, Occurrence *>>
  getUnlikelyCondSetWaitOcc(Occurrence *occ1, Occurrence *occ2);

  // Convenience to insert barrier-all before a given occurrence/op.
  void insertBarrierAllBeforeOcc(Occurrence *occ, bool isUseless,
                                 bool isPersistent = false);

  void insertBarrierAllBeforeOp(OperationBase *op, bool isUseless,
                                bool isPersistent);

  // Determine the direction (backward) of a synchronization candidate.
  bool isBackwardSync(Occurrence *occ1, Occurrence *occ2);

  bool reuseCmp(ConflictPair *conflictPair1, ConflictPair *conflictPair2);

  // Reuse existing conflict pairs where possible to save event ids.
  ConflictPair *getReusableConflictPair(
      ConflictPair *conflictPair,
      const llvm::DenseSet<ConflictPair *> &conflictPairsSet);

  bool reuseConflictPair(ConflictPair *conflictPair, Occurrence *scopeOcc1,
                         Occurrence *scopeOcc2);

  std::unique_ptr<EventIdSolver> &getEventIdSolverRef(hivm::PIPE pipeSrc,
                                                      hivm::PIPE pipeDst);

  bool checkReuseMultiBufferFlagId(ConflictPair *conflictPair);

  // Primary handler invoked to register/record a found conflict.
  void handleConflict(Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1,
                      RWOperation *rwOp2, CorePipeInfo corePipeSrc,
                      CorePipeInfo corePipeDst, EventIdInfo eventIdInfo,
                      bool isUseless);

  void handleBarrierConflict(Occurrence *occ1, Occurrence *occ2,
                             CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                             bool isUseless);

  void handleSetWaitConflict(Occurrence *occ1, Occurrence *occ2,
                             CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                             EventIdInfo eventIdInfo, bool isUseless);

  void handleUnitFlagConflict(Occurrence *occ1, Occurrence *occ2,
                              CorePipeInfo corePipeSrc,
                              CorePipeInfo corePipeDst,
                              UnitFlagInfo unitFlagInfo, bool isUseless);

  Occurrence *getFirstIterOcc(Occurrence *occ, Occurrence *parOcc);

  Occurrence *getLastIterOcc(Occurrence *occ, Occurrence *parOcc);

  std::optional<std::pair<Occurrence *, Occurrence *>>
  checkAndApplyMmadl0LoopOpt(ConflictPair *conflictPair, Occurrence *occ1,
                             Occurrence *occ2, Occurrence *parOcc1,
                             Occurrence *parOcc2);

  // Unit-flag pattern checks used to transform sync into unit-flag modes.
  std::optional<UnitFlagInfo> checkUnitFlagPatterns(Occurrence *occ1,
                                                    Occurrence *occ2);

  void pickAndInsertABarrierAll();

  void calcAllEventIds();

  void collectBackwardSyncEventIds();

  void resetAndBuildSetWaitOpIndex(const SyncMap &syncMapBefore,
                                   const SyncMap &syncMapAfter);

  std::set<std::pair<int64_t, SetWaitOp *>> &
  getSetWaitOpsIndexRef(hivm::PIPE pipeSrc, hivm::PIPE pipeDst,
                        int64_t eventId);

  void collectSetWaitOpsIndexes(OperationBase *op, const SyncMap &syncMapBefore,
                                const SyncMap &syncMapAfter);

  bool checkBackwardSyncEventsContains(OperationBase *op,
                                       CorePipeInfo corePipeSrc,
                                       CorePipeInfo corePipeDst,
                                       int64_t eventId);

  bool checkBackwardSyncEventsContainsAfterMerge(OperationBase *op,
                                                 CorePipeInfo corePipeSrc,
                                                 CorePipeInfo corePipeDst);

  // Merge-related helpers for backward sync events and scope-level
  // optimizations.
  bool checkMergeable(Scope *scopeOp, CorePipeInfo corePipeSrc,
                      CorePipeInfo corePipeDst, int64_t eventId,
                      bool shouldBeUsedAtleastOnce = true);

  void mergeBackwardSyncEventIds(OperationBase *op);

  void mergeBackwardSyncPairs(SyncMap &syncMapBefore, SyncMap &syncMapAfter);

  void insertMergedBackwardSyncPairs();

  llvm::LogicalResult considerOuterBackwardSyncPairs();

  llvm::LogicalResult reuseSyncPairToSaveEventIds();

  llvm::LogicalResult disableMultiEventIdForBarrierAllPairs();

  llvm::LogicalResult tryMovingOutBackwardSyncPairsToOuterLoops();

  Occurrence *getBeforePlaceHolderOcc(Occurrence *occ);
  Occurrence *getAfterPlaceHolderOcc(Occurrence *occ);
  Occurrence *getScopeBeginPlaceHolderOcc(Occurrence *occ);
  Occurrence *getScopeEndPlaceHolderOcc(Occurrence *occ);
};

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVER_H
