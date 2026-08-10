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

#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/CustomMacroSync.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/EventIdSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/GraphSolver.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverIRTranslator.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::hivm::syncsolver {

// Shared SyncSolver infrastructure used by V1 and V2: analyzes memory hazards
// between RW occurrences, chooses sync operations (set/wait, barrier, or
// unit-flag), allocates EventIdNodes via EventIdSolver, and applies post-pass
// optimizations.
class SyncSolverBase {
public:
  // Pass/options configuration from the IR translator.
  const SyncSolverOptions options;

  // Original MLIR function (may be null in test-only solvers).
  func::FuncOp funcOp;

  // Hierarchical IR (Function -> Scopes -> Ops) used for analysis.
  std::unique_ptr<OperationBase> funcIr;

  // Linearized occurrence sequence (syncIr): each Occurrence is one appearance
  // of an operation in analysis order.
  std::vector<std::unique_ptr<Occurrence>> syncIr;

  // RW ops that can use unit-flag sync operations.
  llvm::SetVector<RWOperation *> unitFlagFeaturedOps;

  // Chosen ConflictPairs (and persistent / erased variants kept across passes).
  std::vector<std::unique_ptr<ConflictPair>> chosenConflictedPairs,
      persistentChosenConflictedPairs, erasedChosenConflictedPairs,
      erasedPersistentChosenConflictedPairs;

  // Custom-macro reserved event-id numbers and any reservation conflicts.
  CustomMacroSyncState customMacroSync;

  // Counters for debugging / performance tracing of the solve loop.
  struct PerfInfo {
    int64_t ordersCheckedNum{0};
    int64_t failedInitialChecksNum{0};
    int64_t conflictsProcessedNum{0};
    int64_t memoryConflictsFoundNum{0};
    int64_t handledConflictsNum{0};
    int64_t graphConflictPairsCheckedNum{0};
    int64_t solverSkipNum{0};
    int64_t checkGraphConflictSkipDijNum{0};

    void print() {
      llvm::dbgs() << "processing orders checked: " << ordersCheckedNum << '\n';
      llvm::dbgs() << "failed initial checks: " << failedInitialChecksNum
                   << '\n';
      llvm::dbgs() << "conflicts processed: " << conflictsProcessedNum << '\n';
      llvm::dbgs() << "memory conflicts found: " << memoryConflictsFoundNum
                   << '\n';
      llvm::dbgs() << "handled conflicts: " << handledConflictsNum << '\n';
      llvm::dbgs() << "graph conflict pairs checked: "
                   << graphConflictPairsCheckedNum << '\n';
      llvm::dbgs() << "graph conflict pairs skipped Dijkstra: "
                   << checkGraphConflictSkipDijNum << '\n';
      llvm::dbgs() << "solver skipped: " << solverSkipNum << '\n';
    }
  } perfInfo;

protected:
  // Codegen walk counter used when indexing set/wait ops.
  int64_t globalSetWaitIndex{0};
  // Caps for ConflictPair EventIdNode reuse and multi-pass solve retries.
  int64_t maxReuseNum{20};
  int64_t maxRunNum{99};
  // Flags toggled by post-pass opts that hoist or preserve backward pairs.
  bool moveBackwardSyncPairsToOutmostLoop{false};
  bool dontMoveBackwardSyncPairsToOutmostLoop{false};
  bool enableSaveCVPreloadingEventIdsOpt{false};

  // Per (pipeSrc, pipeDst) EventIdSolver used to allocate EventIdNodes.
  llvm::DenseMap<std::tuple<hivm::PIPE, hivm::PIPE>,
                 std::unique_ptr<EventIdSolver>>
      eventIdSolver;

  // Op -> all of its occurrences in syncIr.
  llvm::DenseMap<OperationBase *, std::vector<Occurrence *>> opAllOccurrences;

  // Already-covered (scope, op1, op2, setPipe, waitPipe) keys and the
  // ConflictPairs that covered them.
  llvm::DenseMap<std::tuple<OperationBase *, OperationBase *, OperationBase *,
                            CorePipeInfo, CorePipeInfo>,
                 llvm::DenseSet<ConflictPair *>>
      syncedPairs;

  // Useless pairs that were replaced by reusing another synced ConflictPair.
  llvm::DenseMap<std::tuple<OperationBase *, OperationBase *, OperationBase *,
                            CorePipeInfo, CorePipeInfo>,
                 ConflictPair *>
      replacedWithReusableSyncedPairs;

  // Chosen ConflictPairs keyed by a single parent/scope occurrence.
  llvm::DenseMap<Occurrence *, llvm::DenseSet<ConflictPair *>>
      scopeOccChosenConflicts, persistentScopeOccChosenConflicts;

  // Chosen ConflictPairs keyed by a pair of parent/scope occurrences (e.g.
  // sibling blocks).
  llvm::DenseMap<std::pair<Occurrence *, Occurrence *>,
                 llvm::DenseSet<ConflictPair *>>
      scopeOccPairChosenConflicts, persistentScopeOccPairChosenConflicts;

  // Tentative ConflictPairs considered during coverage checks but not yet
  // committed.
  llvm::SmallVector<std::tuple<Occurrence *, Occurrence *, ConflictPair *>>
      tempInsertedConflictPairs;

  // Backward ConflictPairs summarized as
  // scopeOp -> (setPipe, waitPipe) -> (eventId -> repeatCount).
  llvm::MapVector<OperationBase *,
                  llvm::DenseMap<std::tuple<CorePipeInfo, CorePipeInfo>,
                                 llvm::DenseMap<int64_t, int64_t>>>
      backwardSyncEvents;

  // Pipe pairs that remain after merging backward sync operations at a scope.
  llvm::MapVector<OperationBase *,
                  llvm::DenseSet<std::tuple<CorePipeInfo, CorePipeInfo>>>
      backwardSyncEventsAfterMerge;

  // Cache of getMemoryConflicts results for RWOperation pairs.
  llvm::DenseMap<
      std::pair<syncsolver::RWOperation *, syncsolver::RWOperation *>,
      llvm::SmallVector<std::pair<CorePipeInfo, CorePipeInfo>>>
      checkMemoryConflictsMem;

  // Pipe pairs that fell back to barrier-all (event ids exhausted).
  llvm::DenseSet<std::tuple<CorePipeInfo, CorePipeInfo>> barrierAllPairs;

  // Pipe pairs for which EventIdInfo.eventIdNum > 1 has been disabled.
  llvm::DenseSet<std::tuple<CorePipeInfo, CorePipeInfo>>
      disabledMultiEventIdPairs;

  // Per-pipe-pair budgets / counts for reusing existing ConflictPairs.
  llvm::DenseMap<std::tuple<CorePipeInfo, CorePipeInfo>, int> reusePairs,
      reusedPairs;

  // Barrier-all sync operations already inserted before an op's occurrences.
  llvm::DenseMap<OperationBase *,
                 llvm::DenseSet<std::pair<Occurrence *, int32_t>>>
      insertedBarrierAllBefore;

  // Codegen indexes of set/wait ops (exclusive and inclusive variants).
  llvm::DenseMap<OperationBase *, int64_t> setWaitStartIndex, setWaitEndIndex,
      setWaitStartIndexInclusive, setWaitEndIndexInclusive;

  // (setPipe, waitPipe, eventId) -> ordered (codegen-index, SetWaitOp*) for
  // merge / outer-backward queries.
  llvm::DenseMap<std::tuple<hivm::PIPE, hivm::PIPE, int64_t>,
                 std::set<std::pair<int64_t, SetWaitOp *>>>
      setWaitFlagOpsIndex;

public:
  SyncSolverBase() = delete;
  virtual ~SyncSolverBase() = default;

  SyncSolverBase(std::unique_ptr<IRTranslator> irTranslator)
      : options(irTranslator->options) {
    init(std::move(irTranslator));
  }

  // Entry point: run the solver (or barrier-all mode) for the function.
  void solve();

  // Whether custom macros reserved conflicting event-id numbers.
  bool hasCustomMacroEventIdConflict() const {
    return customMacroSync.hasConflict();
  }

  // Diagnostic message when customMacroSync reports a reservation conflict.
  StringRef getCustomMacroEventIdConflictMsg() const {
    return customMacroSync.conflictMessage();
  }

  // Build SyncBeforeAfterMap (SyncMap before/after) from chosen ConflictPairs.
  SyncBeforeAfterMap getBeforeAfterSyncMaps();

protected:
  // Clear per-pass bookkeeping (optionally also state used after EventIdSolver
  // runs out of ids).
  virtual void reset(bool resetEventIdRanOutOpts = false);

  // Record a chosen ConflictPair in the solver's bookkeeping.
  virtual bool insertConflictPair(std::unique_ptr<ConflictPair> conflictPair,
                                  Occurrence *parOcc = nullptr) = 0;
  // Remove a previously chosen ConflictPair from bookkeeping.
  virtual bool eraseConflictPair(ConflictPair *conflictPair) = 0;
  // Tentatively record a ConflictPair for a coverage check (not committed).
  virtual bool insertTempConflictPair(ConflictPair *conflictPair,
                                      Occurrence *parOcc = nullptr) = 0;

  // Whether two cross-core ConflictPairs interfere with each other.
  virtual bool checkCrossCoreIntersect(ConflictPair *conflictPair1,
                                       ConflictPair *conflictPair2) = 0;

  // EventIdNodes whose intervals overlap the given ConflictPair.
  virtual llvm::SmallVector<EventIdNode *>
  getIntersectingEventIdNodes(ConflictPair *conflictPair) = 0;

  // Version-specific discovery/analysis of producer/consumer hazards.
  virtual void processOrders() = 0;

  // Take ownership of the translated IR and related lookup tables from the
  // IRTranslator, then record event-id numbers reserved by custom macros.
  void init(std::unique_ptr<IRTranslator> irTranslator) {
    funcOp = irTranslator->funcOp;
    funcIr = std::move(irTranslator->funcIr);
    syncIr = std::move(irTranslator->syncIr);
    unitFlagFeaturedOps = std::move(irTranslator->unitFlagFeaturedOps);
    opAllOccurrences = std::move(irTranslator->opAllOccurrences);
    customMacroSync.collectReservedEventIds(funcOp, options);
  }

  // Clear unit-flag bookkeeping on RW ops / occurrences.
  void resetUnitFlag();

  // Whether the hazard is a backward (cross-iteration) sync operation.
  bool isBackwardSync(Occurrence *occ1, Occurrence *occ2);

  // Placeholder occurrences used when placing sync operations around a region.
  Occurrence *getBeforePlaceHolderOcc(Occurrence *occ);
  Occurrence *getAfterPlaceHolderOcc(Occurrence *occ);
  Occurrence *getScopeBeginPlaceHolderOcc(Occurrence *occ);
  Occurrence *getScopeEndPlaceHolderOcc(Occurrence *occ);

  // Return the LCA occurrence pair used when placing set/wait for a hazard.
  std::pair<Occurrence *, Occurrence *> getSetWaitLCAPairOcc(Occurrence *occ1,
                                                             Occurrence *occ2);

  // Map an occurrence to the first/last loop-iteration copy under parOcc.
  Occurrence *getFirstIterOcc(Occurrence *occ, Occurrence *parOcc);
  Occurrence *getLastIterOcc(Occurrence *occ, Occurrence *parOcc);

  // Whether a cross-core pipe / occurrence pair should be skipped.
  bool checkSkipCrossCorePair(hivm::TCoreType coreTypeSrc,
                              hivm::TCoreType coreTypeDst);
  bool checkSkipCrossCorePair(Occurrence *occ1, Occurrence *occ2);

  // Whether the pair is inside a parallel loop that needs no sync operation.
  bool checkSkipParallelLoop(Occurrence *occ1, Occurrence *occ2);

  // Whether (occ1, occ2) cannot form a valid sync-operation candidate.
  bool checkImpossibleOccPair(Occurrence *occ1, Occurrence *occ2);

  // Whether syncedPairs already covers this occurrence pair on matching pipes.
  bool checkAlreadySynced(Occurrence *occ1, Occurrence *occ2);

  // Whether unit-flag bookkeeping already covers this occurrence pair.
  bool checkAlreadySyncedWithUnitFlag(Occurrence *occ1, Occurrence *occ2);

  // Whether two intra-core ConflictPairs' index ranges / EventIdNodes overlap.
  bool checkIntraCoreIntersect(ConflictPair *conflictPair1,
                               ConflictPair *conflictPair2);
  // Whether two ConflictPairs intersect (dispatches to intra- or cross-core).
  bool checkIntersect(ConflictPair *conflictPair1, ConflictPair *conflictPair2);

  // Whether mmadl1 decomposed-loop handling should skip this pair.
  bool skipMMad1DecomposedLoopOpt(Occurrence *occ1, Occurrence *occ2);

  // Construct and return UnitFlagInfo when occ1/occ2 match a unit-flag pattern;
  // otherwise nullopt.
  std::optional<UnitFlagInfo> checkUnitFlagPatterns(Occurrence *occ1,
                                                    Occurrence *occ2);

  // Whether two MemInfos (or MemInfo lists) conflict under the given
  // constraints.
  bool checkMemInfoConflict(
      RWOperation *rwOp1, RWOperation *rwOp2, const MemInfo &memInfo1,
      const MemInfo &memInfo2, std::optional<int64_t> lcmLen = {},
      std::optional<int64_t> eventIdNum = {},
      std::optional<std::pair<int64_t, int64_t>> offsetPair = {});

  bool checkMemInfoConflict(
      RWOperation *rwOp1, RWOperation *rwOp2,
      const llvm::SmallVector<MemInfo> &memInfoList1,
      const llvm::SmallVector<MemInfo> &memInfoList2,
      std::optional<int64_t> lcmLen = {},
      std::optional<int64_t> eventIdNum = {},
      std::optional<std::pair<int64_t, int64_t>> offsetPair = {});

  // Whether two RW ops have any memory conflict (cached yes/no).
  bool checkMemoryConflicts(
      RWOperation *rwOp1, RWOperation *rwOp2,
      std::optional<int64_t> lcmLen = {},
      std::optional<int64_t> eventIdNum = {},
      std::optional<std::pair<int64_t, int64_t>> offsetPair = {});

  // Return conflicting MemInfo pairs between two RW ops (or explicit MemInfo
  // lists).
  llvm::SmallVector<std::pair<const MemInfo *, const MemInfo *>>
  getMemInfoConflict(
      RWOperation *rwOp1, RWOperation *rwOp2,
      const llvm::SmallVector<MemInfo> &memInfoList1,
      const llvm::SmallVector<MemInfo> &memInfoList2,
      std::optional<int64_t> lcmLen = {},
      std::optional<int64_t> eventIdNum = {},
      std::optional<std::pair<int64_t, int64_t>> offsetPair = {});

  llvm::SmallVector<std::pair<const MemInfo *, const MemInfo *>>
  getMemInfoConflict(
      RWOperation *rwOp1, RWOperation *rwOp2,
      std::optional<int64_t> lcmLen = {},
      std::optional<int64_t> eventIdNum = {},
      std::optional<std::pair<int64_t, int64_t>> offsetPair = {});

  // Whether two RW ops conflict specifically via CV-pipelining buffers.
  bool checkCVPipeliningMemConflict(RWOperation *rwOp1, RWOperation *rwOp2);
  // Whether two RW ops conflict specifically via CV-preloading buffers.
  bool checkCVPreloadingMemConflict(RWOperation *rwOp1, RWOperation *rwOp2,
                                    int64_t eventIdNum);

  // Return the (corePipeSrc, corePipeDst) edges that need a ConflictPair
  // between two RW ops.
  llvm::SmallVector<std::pair<CorePipeInfo, CorePipeInfo>>
  getMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2);

  // Whether any RW under occ1 conflicts with any RW under occ2 (optional
  // filter).
  bool checkMemoryConflictBetweenOccExclusive(
      Occurrence *occ1, Occurrence *occ2,
      std::function<bool(RWOperation *)> filter = [](RWOperation *) {
        return true;
      });

  // Innermost multibuffer scope shared by two RW ops (from explicit MemInfos).
  std::optional<Scope *>
  getMultiBufferScope(RWOperation *rwOp1, RWOperation *rwOp2,
                      const llvm::SmallVector<MemInfo> &memInfoList1,
                      const llvm::SmallVector<MemInfo> &memInfoList2);
  std::optional<Scope *> getMultiBufferScope(RWOperation *rwOp1,
                                             RWOperation *rwOp2);
  // Return the shared multibuffer depth as the number of event ids required.
  std::optional<int64_t> getMultiBufferEventIdNum(
      RWOperation *rwOp1, RWOperation *rwOp2,
      std::optional<std::pair<int64_t, int64_t>> offsetPair = {});
  // Construct and return EventIdInfo for a multibuffer producer/consumer pair.
  std::optional<EventIdInfo> getMultiBufferEventIdInfo(
      Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1,
      RWOperation *rwOp2,
      std::optional<std::pair<int64_t, int64_t>> offsetPair = {});

  // Construct and return EventIdInfo if the hazard is generic multibuffer
  // reuse; otherwise nullopt.
  std::optional<EventIdInfo> checkMultiBufferEventIdInfo(Occurrence *occ1,
                                                         Occurrence *occ2,
                                                         RWOperation *rwOp1,
                                                         RWOperation *rwOp2);
  // Construct and return EventIdInfo if the hazard is CV-pipelining buffer
  // reuse; otherwise nullopt.
  std::optional<EventIdInfo> checkCVPipeliningEventIdInfo(Occurrence *occ1,
                                                          Occurrence *occ2,
                                                          RWOperation *rwOp1,
                                                          RWOperation *rwOp2);
  // Construct and return EventIdInfo if the hazard is CV-preloading buffer
  // reuse; otherwise nullopt.
  std::optional<EventIdInfo> checkCVPreloadingEventIdInfo(Occurrence *occ1,
                                                          Occurrence *occ2,
                                                          RWOperation *rwOp1,
                                                          RWOperation *rwOp2);

  // Construct and return the EventIdInfo required for a hazard.
  std::tuple<EventIdInfo, SetWaitPairInfo>
  getEventIdSetWaitPairInfo(Occurrence *occ1, Occurrence *occ2,
                            RWOperation *rwOp1, RWOperation *rwOp2,
                            CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst);

  // Existing EventIdNode for a ConflictPair coverage key, if already allocated.
  EventIdNode *getOldEventIdNodeIfExists(ConflictPair *conflictPair);

  // All ConflictPairs previously recorded for the same coverage key.
  llvm::DenseSet<ConflictPair *>
  getMemorizedSyncedPairs(ConflictPair *conflictPair);

  // Record that a ConflictPair now covers its (scope, ops, pipes) key.
  void memorizeSyncedPair(ConflictPair *conflictPair);

  // Record that a useless ConflictPair was replaced by reusing another pair.
  void memorizeReusedSyncedPair(ConflictPair *conflictPair,
                                ConflictPair *reusedConflictPair);

  // Construct and return SetWaitPairInfo with fixed set/wait placement (no
  // further sinking/hoisting).
  SetWaitPairInfo
  getFixedSetWaitOcc(Occurrence *occ1, Occurrence *occ2,
                     std::optional<EventIdInfo> eventIdInfo = {});

  // Construct and return set/wait Occurrences at function-block boundaries, if
  // applicable.
  std::optional<std::pair<Occurrence *, Occurrence *>>
  getFunctionBlockSetWaitOcc(Occurrence *occ1, Occurrence *occ2);

  // Construct and return set/wait Occurrences around an unlikely conditional,
  // if applicable.
  std::optional<std::pair<Occurrence *, Occurrence *>>
  getUnlikelyCondSetWaitOcc(Occurrence *occ1, Occurrence *occ2);

  // Construct and return SetWaitPairInfo with default set/wait placement.
  SetWaitPairInfo getSetWaitOcc(Occurrence *occ1, Occurrence *occ2,
                                std::optional<EventIdInfo> eventIdInfo = {});

  // Return the Occurrence where a barrier wait should be placed.
  Occurrence *getBarrierWaitOcc(Occurrence *occ1, Occurrence *occ2,
                                std::optional<EventIdInfo> eventIdInfo = {});

  // Insert a barrier-all sync operation before an occurrence / op.
  void insertBarrierAllBeforeOcc(Occurrence *occ, bool isUseless,
                                 bool isPersistent = false);
  void insertBarrierAllBeforeOp(OperationBase *op, bool isUseless,
                                bool isPersistent);

  // When EventIdSolver cannot allocate more ids, insert one barrier-all and
  // retry.
  void pickAndInsertABarrierAll();

  // Return the EventIdSolver for (pipeSrc, pipeDst), creating it on first use.
  std::unique_ptr<EventIdSolver> &getEventIdSolverRef(hivm::PIPE pipeSrc,
                                                      hivm::PIPE pipeDst);

  // Optional mmadl0 last-iteration set placement rewrite.
  std::optional<std::pair<Occurrence *, Occurrence *>>
  checkAndApplyMmadl0LoopOpt(ConflictPair *conflictPair, Occurrence *occ1,
                             Occurrence *occ2, Occurrence *parOcc1,
                             Occurrence *parOcc2);

  // Ordering used when choosing which existing ConflictPair to reuse.
  bool reuseCmp(ConflictPair *conflictPair1, ConflictPair *conflictPair2);

  // Best existing ConflictPair whose EventIdNode can be reused, if any.
  ConflictPair *getReusableConflictPair(
      ConflictPair *conflictPair,
      const llvm::DenseSet<ConflictPair *> &conflictPairsSet);

  // Try to reuse another ConflictPair's EventIdNode for this hazard.
  bool reuseConflictPair(ConflictPair *conflictPair, Occurrence *scopeOcc1,
                         Occurrence *scopeOcc2);

  // Whether conflictPair should raise eventIdRepeatNum so the same flag id is
  // set/waited repeatedly (multibuffer backward case).
  bool checkRepeatMultiBufferFlagId(ConflictPair *conflictPair);

  // Whether a CV-preloading occurrence pair should skip ConflictPair insertion.
  bool checkSkipCVPreloadingPair(Occurrence *occ1, Occurrence *occ2);

  // Create and record a set/wait ConflictPair for a cross-pipe hazard.
  ConflictPair *handleSetWaitConflict(Occurrence *occ1, Occurrence *occ2,
                                      CorePipeInfo corePipeSrc,
                                      CorePipeInfo corePipeDst,
                                      EventIdInfo eventIdInfo,
                                      SetWaitPairInfo setWaitPairInfo,
                                      bool isUseless);

  // Create and record a barrier ConflictPair for a same-pipe hazard.
  ConflictPair *handleBarrierConflict(Occurrence *occ1, Occurrence *occ2,
                                      CorePipeInfo corePipeSrc,
                                      CorePipeInfo corePipeDst,
                                      EventIdInfo eventIdInfo, bool isUseless);

  // Create and record a unit-flag ConflictPair when that pattern applies.
  ConflictPair *handleUnitFlagConflict(Occurrence *occ1, Occurrence *occ2,
                                       CorePipeInfo corePipeSrc,
                                       CorePipeInfo corePipeDst,
                                       UnitFlagInfo unitFlagInfo,
                                       bool isUseless);

  // Fill backwardSyncEvents from chosen inner-backward ConflictPairs
  // (eventId -> repeatCount).
  void collectBackwardSyncEventIds();

  // Return a mutable reference to the ordered set/wait index for
  // (pipeSrc, pipeDst, eventId).
  std::set<std::pair<int64_t, SetWaitOp *>> &
  getSetWaitOpsIndexRef(hivm::PIPE pipeSrc, hivm::PIPE pipeDst,
                        int64_t eventId);

  // Walk an op subtree and populate setWaitFlagOpsIndex from SyncMaps.
  void collectSetWaitOpsIndexes(OperationBase *op, const SyncMap &syncMapBefore,
                                const SyncMap &syncMapAfter);

  // Whether backwardSyncEvents already records (corePipeSrc, corePipeDst,
  // eventId) at op.
  bool checkBackwardSyncEventsContains(OperationBase *op,
                                       CorePipeInfo corePipeSrc,
                                       CorePipeInfo corePipeDst,
                                       int64_t eventId);

  // Whether the post-merge backward pipe set still contains this pipe pair.
  bool checkBackwardSyncEventsContainsAfterMerge(OperationBase *op,
                                                 CorePipeInfo corePipeSrc,
                                                 CorePipeInfo corePipeDst);

  // Whether redundant backward sync operations at a scope can be merged.
  bool checkMergeable(Scope *scopeOp, CorePipeInfo corePipeSrc,
                      CorePipeInfo corePipeDst, int64_t eventId,
                      bool shouldBeUsedAtleastOnce = true);

  // Rebuild setWaitFlagOpsIndex from before/after SyncMaps.
  void resetAndBuildSetWaitOpIndex(const SyncMap &syncMapBefore,
                                   const SyncMap &syncMapAfter);

  // Merge eventId -> repeatCount maps recorded under op.
  void mergeBackwardSyncEventIds(OperationBase *op);

  // Rewrite SyncMaps after merging backward ConflictPairs.
  void mergeBackwardSyncPairs(SyncMap &syncMapBefore, SyncMap &syncMapAfter);

  // Finalize EventIdSolver allocations (shrink maxima) and validate
  // custom-macro pinned assignments.
  void calcAllEventIds();

  // Re-insert ConflictPairs for previously merged backward sync operations.
  void insertMergedBackwardSyncPairs();

  // Hoist eligible backward sync operations to an outer scope.
  llvm::LogicalResult considerOuterBackwardSyncPairs();

  // Reuse ConflictPair EventIdNodes to free ids when barrier-all pressure is
  // high.
  llvm::LogicalResult reuseSyncPairToSaveEventIds();

  // Prefer preserving EventIdNodes used by CV-preloading ConflictPairs.
  llvm::LogicalResult saveCVPreloadingEventIdsOpt();

  // Disable EventIdInfo.eventIdNum > 1 for pipe pairs that fell back to
  // barrier-all.
  llvm::LogicalResult disableMultiEventIdForBarrierAllPairs();

  // Move backward sync operations out to enclosing loops when profitable.
  llvm::LogicalResult tryMovingOutBackwardSyncPairsToOuterLoops();

  // Multi-pass solve loop: processOrders plus optional hoist / reuse / barrier
  // fallback retries.
  llvm::LogicalResult runSolver(bool enableOpts1 = true,
                                bool enableOpts2 = true);

  // Insert barrier-all before every RW op (no fine-grained solving).
  void solveBlockAllMode();
};

// SyncSolver algorithm version 1: resolves memory hazards by deciding which
// sync operations to insert between producer/consumer occurrences.
class SyncSolverV1 : public SyncSolverBase {
public:
  SyncSolverV1() = delete;
  explicit SyncSolverV1(std::unique_ptr<IRTranslator> irTranslator)
      : SyncSolverBase(std::move(irTranslator)) {}

protected:
  // A producer/consumer occurrence pair scheduled for conflict analysis.
  // isUseless marks pairs that only matter for later-iteration covering.
  struct ProcessingOrderV1 {
    Occurrence *occ1{nullptr};
    Occurrence *occ2{nullptr};
    RWOperation *rwOp1{nullptr};
    RWOperation *rwOp2{nullptr};
    bool isUseless{false};
    ProcessingOrderV1(Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1,
                      RWOperation *rwOp2, bool isUseless)
        : occ1(occ1), occ2(occ2), rwOp1(rwOp1), rwOp2(rwOp2),
          isUseless(isUseless) {}
  };

  // Ordered worklist of pairs to analyze.
  std::vector<ProcessingOrderV1> processingOrders;

  // Occurrence pairs already considered (debug-only duplicate-visit check).
  llvm::DenseSet<std::pair<Occurrence *, Occurrence *>> processedOccPairs;

  void reset(bool resetEventIdRanOutOpts = false) override;

  // Record a chosen ConflictPair in the solver's bookkeeping.
  bool insertConflictPair(std::unique_ptr<ConflictPair> conflictPair,
                          Occurrence *parOcc = nullptr) override;
  // Remove a previously chosen ConflictPair from bookkeeping.
  bool eraseConflictPair(ConflictPair *conflictPair) override;
  // Tentatively record a ConflictPair for a coverage check (not committed).
  bool insertTempConflictPair(ConflictPair *conflictPair,
                              Occurrence *parOcc = nullptr) override;

  // Return true if GraphSolver says a new ConflictPair is still required
  // between occ1 and occ2 on the given pipes.
  bool checkGraphConflict(
      Occurrence *occ1, Occurrence *occ2, CorePipeInfo corePipeSrc,
      CorePipeInfo corePipeDst, EventIdInfo eventIdInfo,
      std::optional<int64_t> startIndex = {},
      std::optional<int64_t> endIndex = {},
      const llvm::SmallVector<ConflictPair *> &extraConflictPairs = {},
      const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs = {});

  // Whether two cross-core ConflictPairs interfere with each other.
  bool checkCrossCoreIntersect(ConflictPair *conflictPair1,
                               ConflictPair *conflictPair2) override;

  // EventIdNodes whose intervals overlap the given ConflictPair.
  llvm::SmallVector<EventIdNode *>
  getIntersectingEventIdNodes(ConflictPair *conflictPair) override;

  // Whether this pair should be ignored because it is only a later-iteration
  // variant of a hazard handled elsewhere.
  bool skipLaterIterations(Occurrence *occ1, Occurrence *occ2);

  // Expand hierarchical occurrence pairs into leaf RW ProcessingOrderV1
  // entries (or recurse into child lists / loop iterations).
  void generateProcessingOrders(Occurrence *occ1, Occurrence *occ2,
                                bool isUseless);
  void generateProcessingOrders(const llvm::SmallVector<Occurrence *> &occs,
                                bool isUseless);
  void generateProcessingOrders(const llvm::SmallVector<Occurrence *> &occs1,
                                const llvm::SmallVector<Occurrence *> &occs2,
                                bool isUseless);
  void generateProcessingOrders(Scope *scopeOp, Occurrence *occ,
                                bool isUseless);
  void generateProcessingOrders(Loop *loopOp, Occurrence *occ, bool isUseless);
  void generateProcessingOrders(RWOperation *rwOp1, RWOperation *rwOp2,
                                Occurrence *occ1, Occurrence *occ2,
                                bool isUseless);

  // DFS over syncIr: recurse into children, then generateProcessingOrders for
  // the current scope/loop (second loop iterations marked isUseless).
  void buildOfflineProcessingOrders(Occurrence *occ, bool isUseless = false);

  // Choose barrier / unit-flag / set-wait handling for a confirmed pipe-level
  // hazard and record the resulting ConflictPair.
  void handleConflict(Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1,
                      RWOperation *rwOp2, CorePipeInfo corePipeSrc,
                      CorePipeInfo corePipeDst, bool isUseless);

  // Insert (occ1, occ2) into processedOccPairs; return true if already present.
  bool checkVisited(Occurrence *occ1, Occurrence *occ2);

  // For each pipe edge from getMemoryConflicts, run handleConflict if still
  // needed.
  void processConflict(Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1,
                       RWOperation *rwOp2, bool isUseless);

  // Run skip checks then processConflict for one ProcessingOrderV1 entry.
  void processOrder(ProcessingOrderV1 processingOrder);

  // Build processingOrders from syncIr, then process each entry.
  void processOrders() override;
};

// SyncSolver algorithm version 2: same goal as V1 (insert sync operations for
// memory hazards), with a different strategy for discovering and pruning work.
class SyncSolverV2 : public SyncSolverBase {
public:
  SyncSolverV2() = delete;
  explicit SyncSolverV2(std::unique_ptr<IRTranslator> irTranslator)
      : SyncSolverBase(std::move(irTranslator)) {}

protected:
  // A producer/consumer pair under a common analysis scope (LCA pair), with
  // the pipe edge already identified. isUseless as in SyncSolverV1.
  struct ProcessingOrderV2 {
    Occurrence *lcaOcc1{nullptr};
    Occurrence *lcaOcc2{nullptr};
    Occurrence *occ1{nullptr};
    Occurrence *occ2{nullptr};
    RWOperation *rwOp1{nullptr};
    RWOperation *rwOp2{nullptr};
    CorePipeInfo corePipeSrc;
    CorePipeInfo corePipeDst;
    bool isUseless{false};

    ProcessingOrderV2(Occurrence *lcaOcc1, Occurrence *lcaOcc2,
                      Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1,
                      RWOperation *rwOp2, CorePipeInfo corePipeSrc,
                      CorePipeInfo corePipeDst, bool isUseless)
        : lcaOcc1(lcaOcc1), lcaOcc2(lcaOcc2), occ1(occ1), occ2(occ2),
          rwOp1(rwOp1), rwOp2(rwOp2), corePipeSrc(corePipeSrc),
          corePipeDst(corePipeDst), isUseless(isUseless) {}
  };

  // Cached GraphSolver plus watermarks into the ConflictPair insert/erase logs.
  struct GraphSolverInfo {
    std::unique_ptr<GraphSolverBase> graphSolver;
    size_t insertedConflictPairsIndex{0};
    size_t insertedPersistentConflictPairsIndex{0};
    size_t erasedConflictPairsIndex{0};
    size_t erasedPersistentConflictPairsIndex{0};
  };

  // GraphSolverInfo keyed by (parentOcc1, parentOcc2, eventIdNum,
  // isCVPreloading).
  llvm::DenseMap<std::tuple<Occurrence *, Occurrence *, int64_t, int16_t>,
                 GraphSolverInfo>
      graphSolverMap;

  // Recent ConflictPair insertions/erasures that must be applied to cached
  // GraphSolver instances before further coverage checks.
  llvm::SmallVector<std::tuple<Occurrence *, Occurrence *, ConflictPair *>>
      insertedConflictPairs, insertedPersistentConflictPairs,
      erasedConflictPairs, erasedPersistentConflictPairs;

  void reset(bool resetEventIdRanOutOpts = false) override;

  // Record a chosen ConflictPair in the solver's bookkeeping.
  bool insertConflictPair(std::unique_ptr<ConflictPair> conflictPair,
                          Occurrence *parOcc = nullptr) override;
  // Remove a previously chosen ConflictPair from bookkeeping.
  bool eraseConflictPair(ConflictPair *conflictPair) override;
  // Tentatively record a ConflictPair for a coverage check (not committed).
  bool insertTempConflictPair(ConflictPair *conflictPair,
                              Occurrence *parOcc = nullptr) override;

  // Return the GraphSolverBase for (occ1, occ2, eventIdInfo), creating it if
  // needed and applying pending ConflictPair insert/erase logs.
  std::unique_ptr<GraphSolverBase> &
  getGraphSolverRef(Occurrence *occ1, Occurrence *occ2,
                    const EventIdInfo &eventIdInfo);

  // Return true if GraphSolver says a new ConflictPair is still required
  // between occ1 and occ2 on the given pipes.
  bool checkGraphConflict(
      Occurrence *occ1, Occurrence *occ2, CorePipeInfo corePipeSrc,
      CorePipeInfo corePipeDst, EventIdInfo eventIdInfo,
      std::optional<int64_t> startIndex = {},
      std::optional<int64_t> endIndex = {},
      const llvm::SmallVector<ConflictPair *> &extraConflictPairs = {},
      const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs = {});

  // Whether two cross-core ConflictPairs interfere with each other.
  bool checkCrossCoreIntersect(ConflictPair *conflictPair1,
                               ConflictPair *conflictPair2) override;

  // EventIdNodes whose intervals overlap the given ConflictPair.
  llvm::SmallVector<EventIdNode *>
  getIntersectingEventIdNodes(ConflictPair *conflictPair) override;

  // Choose barrier / unit-flag / set-wait handling for a confirmed pipe-level
  // hazard. Returns the created ConflictPair, or nullptr if none was inserted.
  ConflictPair *handleConflict(Occurrence *occ1, Occurrence *occ2,
                               RWOperation *rwOp1, RWOperation *rwOp2,
                               CorePipeInfo corePipeSrc,
                               CorePipeInfo corePipeDst,
                               EventIdInfo eventIdInfo,
                               SetWaitPairInfo setWaitPairInfo, bool isUseless);

  // Run skip checks, then for each getMemoryConflicts pipe edge call
  // checkGraphConflict / handleConflict as needed.
  void processOrder(Occurrence *occ1, Occurrence *occ2, RWOperation *rwOp1,
                    RWOperation *rwOp2, bool isUseless);

  // Process one ProcessingOrderV2 candidate. Returns true when further queue
  // candidates for the same (corePipeSrc, corePipeDst) can be skipped.
  bool processOrder(ProcessingOrderV2 processingOrder);

  // Walk MemInfo trees under (lcaOcc1, lcaOcc2) and process conflicting
  // occurrence pairs in [occs1] x [occs2].
  void generateProcessingOrders(Occurrence *lcaOcc1, Occurrence *lcaOcc2,
                                const llvm::ArrayRef<Occurrence *> &occs1,
                                const llvm::ArrayRef<Occurrence *> &occs2,
                                bool isUseless, bool occs1IsLcaOcc1 = false,
                                bool occs2IsLcaOcc2 = false);
  void generateProcessingOrders(Occurrence *occ1, Occurrence *occ2,
                                bool isUseless);
  void generateProcessingOrders(Occurrence *lcaOcc,
                                const llvm::ArrayRef<Occurrence *> &occs,
                                bool isUseless);
  void generateProcessingOrders(Scope *scopeOp, Occurrence *occ,
                                bool isUseless);
  void generateProcessingOrders(Loop *loopOp, Occurrence *occ, bool isUseless);

  // DFS over syncIr: recurse into children, then generateProcessingOrders for
  // the current scope/loop (second loop iterations marked isUseless).
  void collectProcessingOrders(Occurrence *occ, bool isUseless = false);

  // Entry: collectProcessingOrders on the syncIr root.
  void processOrders() override;
};

// Map a pass option string ("v1"/"v2", case-insensitive; empty → V1) to
// SyncSolverVersion.
SyncSolverVersion parseSyncSolverVersion(llvm::StringRef value);

// Construct SyncSolverV1 or SyncSolverV2 from
// irTranslator->options.solverVersion.
std::unique_ptr<SyncSolverBase>
createSolver(std::unique_ptr<IRTranslator> irTranslator);

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVER_H
