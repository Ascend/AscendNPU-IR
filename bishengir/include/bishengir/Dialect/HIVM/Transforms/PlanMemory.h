//===- PlanMemory.h ----Plan Buffer Memory Address ------------------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_PLAN_MEMORY_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_PLAN_MEMORY_H

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/OptMemPlanForPipeline.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/SmallSet.h"

#include <list>

namespace mlir {
namespace hivm {

/// Various states when collecting gen-kill.
enum BufferStatus { UNDEFFINED = 0, DEFFINED, GENED, KILLED };

/// Pair of inplace Value.
using ValuePair = std::pair<Value, Value>;

/// Result status after plan memory.
enum PlanStatus {
  PLAN_SUCCESS = 0,
  RESTART_NEW_PLAN,
  CONTINUE_PLAN,
  PLAN_FAILED
};

/// Memory reuse plan mode can be achieved without conflicting life
/// intervals, offset = 0.
constexpr const int SPEC_LEVEL_0 = 0;

/// By increasing the lifespan by 1 without conflict,
/// memory reuse plan mode can be implemented to avoid dependency on
/// continuous instructions caused by plan, offset = 1.
constexpr const int SPEC_LEVEL_1 = 1;

/// pipe conflict opt.
constexpr const int SPEC_LEVEL_2 = 2;

/// plan information of alloc buffer.
struct BufferInfo {
  /// Alloc operation of buffer.
  Operation *operation{nullptr};
  /// Space corresponding to buffer.
  hivm::AddressSpace bufferScope;
  /// The size required for the buffer.
  int64_t constBits{0};
  /// The type of element in the buffer.
  Type bufferType;
  /// Alias buffer does not participate in inplace.
  /// e.g :
  ///  alloc A
  ///  for(arg = A) :
  ///    alloc B
  ///    ...
  ///    alloc C
  ///    vadd ins(B, D), outs(C)
  ///    scf.yield C
  /// Put (A, C) in inplacePairList and inplace them together in next plan
  /// memory. Because here do not union the lifetime of A and C, just set the
  /// ignoreInplace of A and C to be true so that A will not be inplaced with
  /// other buffer due to wrong lifetime.
  /// TODO: Modify the lifetime of A and C and allow them to be inplaced further
  bool ignoreInplace{false};
};

/// linear operation info.
struct OpInfo {
  OpInfo(Operation *operation, int index)
      : operation(operation), index(index) {}
  Operation *operation{nullptr};
  int index{0};
};

struct GenKillEntry {
  /// record the gen operands, namely the operand buffer that is firstly written
  /// by operation.
  SmallVector<Value> gen;

  /// record the kill operands, namely the operand buffer that is last read by
  /// operation.
  SmallVector<Value> kill;
};

/// Record buffer life interval information.
struct BufferLife {
  BufferLife(Value buffer, int64_t start, int64_t end)
      : buffer(buffer), allocTime(start), freeTime(end) {}
  BufferLife(Value buffer) : buffer(buffer) {}
  /// buffer value.
  Value buffer;
  /// the buffer allocate time.
  int64_t allocTime{-1};
  /// the buffer free time.
  int64_t freeTime{-1};
};

/// a list of buffer life for a given storage entry
using BufferLifeVec = SmallVector<std::shared_ptr<BufferLife>>;

struct StorageEntry {
  /// The the buffer plan info.
  BufferInfo *bufInfo{nullptr};

  /// The lifespan of a buffer.
  BufferLifeVec bufferLifeVec;

  /// The children of this entry, not including itself.
  SmallVector<StorageEntry *> mergedChildren;

  /// The current entry needs to be planed an aligned size.
  uint64_t alignedConstBits{0};

  /// The current entry's child index.
  int childIdx;

  /// The starting address after the current entry allocation.
  uint64_t bitsOffset{0};

  /// Allocs that inplace buffer this entry.
  SmallVector<Value> inplaceBuffers;

  /// multiBuffer relation StorageEntry.
  StorageEntry *relationPongEntry{nullptr};

  /// The number of multibuffer optimization.
  /// note: default 1 which means single buffer and does not do multibuffer
  /// optimization.
  uint32_t multiBufferNum{1};

  /// Get Bufferlife by vaule
  std::shared_ptr<BufferLife> GetBufferLifeByValue(const Value v) const;
};

struct MemoryBound {
  MemoryBound(BufferLifeVec life, uint64_t o, uint64_t e, const StorageEntry *s)
      : bufferLifeVec(std::move(life)), offset(o), extent(e),
        lastStorageEntry(s) {}
  /// collection of buffer plan and free time which use this Memory
  BufferLifeVec bufferLifeVec;
  /// offset of tagged memory
  uint64_t offset;
  /// extent of this bound
  uint64_t extent;
  /// always record storage entry of last plan
  const StorageEntry *lastStorageEntry;
};
using MemBoundList = std::list<std::shared_ptr<MemoryBound>>;
using MemBoundListConstIter = MemBoundList::const_iterator;

/// record of buffer plan. used for speculative rollback
struct PlanRecord {
  /// speculative level of this plan
  int specLevel;
  /// child index
  int childIdx;
  /// if this plan split last memory bound
  bool tailed;
  /// if this plan has bank offset memory bound
  bool headed;
  /// split number of entry
  size_t splitNums;
  /// record the entry for bank info
  StorageEntry *entry;
  /// the whole extent,add all the split e together
  uint64_t allExtent;
  /// inserted memory bound node
  std::shared_ptr<MemoryBound> firstMemBound;
  /// replaced memory bound node
  MemBoundList replaced;
  /// When the current PlanRecord is rolled back, it must be rolled back
  /// directly.
  bool isDirectlyRollback;
};

using PlanRecHis = SmallVector<PlanRecord>;

struct SpecInfo {
  int maxLevel = SPEC_LEVEL_2;
  int minLevel = SPEC_LEVEL_0;
  int specLevel = SPEC_LEVEL_2;
  int childIdx = -1;
  int specStartIdx = 0;
  int rollbackIdx = -1;
  uint64_t rollbackAddr = UINT64_MAX;
};

struct OutlineSectionInfo {
  OutlineSectionInfo() = default;
  OutlineSectionInfo(MemBoundListConstIter &start, MemBoundListConstIter &end,
                     uint64_t s, bool isDirectlyRollback)
      : mem_start(start), mem_end(end), size(s),
        isDirectlyRollback(isDirectlyRollback) {}
  /// The start of memory plan
  MemBoundListConstIter mem_start;
  /// The end of memory plan
  MemBoundListConstIter mem_end;
  /// The size of memory plan
  uint64_t size{0};
  /// When the current PlanRecord is rolled back, it must be rolled back
  /// directly.
  bool isDirectlyRollback;
};

/// comparator of buffer life
struct CompareBufferLife {
  bool operator()(const std::shared_ptr<BufferLife> &lhs,
                  const std::shared_ptr<BufferLife> &rhs) const {
    if (lhs->allocTime == rhs->allocTime) {
      return lhs->freeTime < rhs->freeTime;
    }
    return lhs->allocTime < rhs->allocTime;
  }
};

struct StatusWrapper {
  /// Is it enough to roll back
  bool hasEnoughRollBackSize;
  /// The size required for the buffer
  uint64_t alignedConstBits;
  /// spec info
  SpecInfo *si;
  /// current outline info
  MemBoundList &outline;
  /// current history plan info
  PlanRecHis &history;
  /// for origin e StorageEntry
  StorageEntry *RootE;
};

/// Pair of alias buffer and whether the alias buffer is conditional
using BufferCondPair = std::pair<Value, bool>;

class MemLivenessAnalysis {
public:
  MemLivenessAnalysis(func::FuncOp func, MemPlanMode planMode)
      : func_(func), planMode(planMode) {}

  void build();

  /// linear operation info.
  SmallVector<std::unique_ptr<OpInfo>> linearOperation;

  /// map from buffer value to its buffer information.
  std::map<Value, BufferInfo, utils::ValueComparator> bufferInfos;

  /// map from buffer to its lifetime.
  DenseMap<Value, std::shared_ptr<BufferLife>> buffer2Life;

  /// map from operation to its gen and kill buffer.
  DenseMap<OpInfo *, GenKillEntry> genKillMap;

  /// record the map from the buffer to its number of buffer if it does
  /// multibuffer optimization.
  /// note: the map only record the buffer which do multi buffer
  /// optimization and ignore single buffer.
  DenseMap<Value, uint32_t> buffer2MultiNum;

  /// record inplace pair list.
  SmallVector<ValuePair> inplacePairList;

  /// now plan mode is LOCAL_MEM_PLAN.
  bool isLocalMemPlan() const;

  /// now plan mode is GLOBAL_WORKSPACE_PLAN.
  bool isGlobalWorkSpaceMemPlan() const;

private:
  void RecursionIR(Region *region, Liveness live);

  /// Get the buffer used within the loop and defined outside the loop.
  SmallVector<Value> GetLiveBuffersInLoop(LoopLikeOpInterface loopOp,
                                          Liveness live);

  /// Update for Op tensor init args and tensor result args alias info.
  void UpdateInitAndResAlias(DestinationStyleOpInterface dstStyleOp);

  /// Recursive operation while.
  void RecursiveWhileOp(scf::WhileOp whileOp, Liveness live);

  /// Update while Op init args region iter args alias info.
  void UpdateWhileOpInitArgsAlias(scf::WhileOp whileOp);

  /// Update whileOp result buffer/region iter arg/yielded buffer args alias
  /// info.
  void UpdateWhileOpBufferAlias(scf::WhileOp whileOp);

  /// Update ConditionOp operands buffer/whileOp after args alias info.
  void UpdateConditionOpBufferAlias(scf::ConditionOp conditionOp);

  /// Recursive operation for.
  void RecursiveForOp(scf::ForOp forOp, Liveness live);

  /// Update for Op init args region iter args alias info.
  void UpdateForOpInitArgsAlias(scf::ForOp forOp);

  /// Update forOp result buffer/region iter arg/yielded buffer args alias info.
  void UpdateForOpBufferAlias(scf::ForOp forOp);

  /// Recursive operation if.
  void RecursiveIfOp(scf::IfOp ifOp, Liveness live);

  /// Update branch dest arg and operand buffer alias info.
  void UpdateBranchOpAlias(Block *brBlock, OperandRange destOperands);

  /// Update buffer alias information for ifop.
  void UpdateIfOpBufferAlias(scf::IfOp ifOp, scf::YieldOp yieldOp);

  /// Update and obtain op info information.
  OpInfo *UpdateLinearOperation(Operation *op);

  /// Obtain all information about the buffer.
  void UpdateOpBufferInfo(Operation *op, const ValueRange &results);

  /// Generate buffer info.
  BufferInfo GenerateBufferInfo(Operation *op, Value operand);

  /// Obtain the buffer info of plan operation.
  BufferInfo GetBufferInfo(Operation *op, Value operand,
                           hivm::AddressSpace bufferScope);

  /// Process gen buffer based on the result value of op.
  void UpdateOpGenInfo(OpInfo *opInfo, const ValueRange &results);

  /// Update normal operand gen information on buffer.
  void UpdateOperandGenInfo(OpInfo *opInfo, Value operand);

  /// Update temp buffer for DestinationStyleOpInterface op.
  void UpdateOpTempGenInfo(OpInfo *opInfo);

  /// Update temp buffer for ignoring inplace.
  void UpdateExtraBufferIgnoreInplace(const ValueRange &results);

  /// Update alias buffer and its condition.
  void UpdateBuffer2AliasVec(const SetVector<Value> &buffers,
                             const SetVector<Value> &aliasBuffers,
                             bool hasCond);

  /// Update the relationship of buffer aliases.
  void UpdateBufferAlias(Value buffer, Value aliasBuffer, bool hasCond = false,
                         bool isIgnoreInplace = false);

  /// Find buffer cond pair from aliasVec.
  std::optional<BufferCondPair *> FindBufferCondPair(Value buffer,
                                                     Value aliasValue);

  /// Get alias buffer information.
  SmallVector<BufferCondPair> GetAliasBufferCondPairs(Value aliasBuffer);

  /// Get alias buffers.
  SetVector<Value> GetAliasBuffers(Value aliasBuffer);

  /// Check whether there is an unknown operation with buffer
  /// information.
  LogicalResult CheckIfUnknownOpTouchBuffer(Operation *op) const;

  /// Determine whether the current operation can be skipped.
  bool isSkippableOp(Operation *op) const;

  /// Update multi buffer information.
  void UpdateMultiBufferInfo(annotation::MarkOp markOp);

  /// Update store op information.
  void UpdateStoreOpInfo(OpInfo *opInfo, const Value storeValue, Liveness live);

  /// Check if it is local buffer with memory space
  LogicalResult CheckLocalBufferAllocOp(Operation *op) const;

  /// kill buffer handle.
  void OpKillHandle(OpInfo *opInfo, Liveness live, Block *block);

  /// Process kill buffer based on the result live of op.
  void UpdateOpKillInfo(OpInfo *opInfo, Value operand, Liveness live);

  /// Whether afterBlock is after beforeBlock.
  bool IsBlockAfter(Block *afterBlock,  Block *beforeBlock) const;

  /// Whether the value is dead after a certain block.
  bool IsDeadAfterBlock(Value value,  Block *block) const;

  /// Have all alias buffer been killed.
  bool AllDeadAfter(Operation *op, SetVector<Value> aliasVec,
                    Liveness live) const;

  /// Determine whether two operation are in the same block or op2 is the
  /// ancestor of op1.
  bool isParentOpDominate(Operation *op1, Operation *op2) const;

  /// Generate buffer's life time.
  void GenerateBufferLife();

  /// initialize the buffers that must be inplaced together
  /// namely, the alias buffers of memref.alloc,
  /// e.g. for iter arg and for yield.
  void InitializeInplacePairList();

  func::FuncOp func_;

  /// different mode for mem plan.
  MemPlanMode planMode;

  /// Gen-kill status corresponding to buffer.
  DenseMap<Value, BufferStatus> buffer2status;

  /// map on buffer alias, and whether the alias buffer is conditional.
  DenseMap<Value, SmallVector<BufferCondPair>> buffer2AliasVec;

  int seqIndex{0};
};

/// Pair of StorageEntry.
using StorageEntryPair = std::pair<const StorageEntry *, const StorageEntry *>;

class MemPlan {
public:
  MemPlan(MemPlanMode planMode, bool enableGlobalReuse,
          bool restrictInplaceAsISA)
      : planMode(planMode), enableGlobalReuse(enableGlobalReuse),
        restrictInplaceAsISA(restrictInplaceAsISA) {}

  LogicalResult plan();

  /// Get buffer2Offsets
  inline DenseMap<Value, SmallVector<uint64_t>> GetBuffer2Offsets() {
    return buffer2Offsets;
  }

  inline void
  SetLinearOperation(SmallVector<std::unique_ptr<OpInfo>> &linearOp) {
    linearOperation = std::move(linearOp);
  };

  inline void
  SetBufferInfos(std::map<Value, BufferInfo, utils::ValueComparator> bufsInfo) {
    bufferInfos = bufsInfo;
  }

  inline void
  SetBuffer2Life(DenseMap<Value, std::shared_ptr<BufferLife>> buf2Life) {
    buffer2Life = buf2Life;
  }

  inline void SetGenKillMap(DenseMap<OpInfo *, GenKillEntry> gkMap) {
    genKillMap = gkMap;
  }

  inline void SetBuffer2MultiNum(DenseMap<Value, uint32_t> buf2MulBufNum) {
    buffer2MultiNum = buf2MulBufNum;
  }

  inline void SetInplacePairList(SmallVector<ValuePair> inplaceList) {
    inplacePairList = inplaceList;
  }

  func::FuncOp func_;

private:
  /// different mode for mem plan.
  MemPlanMode planMode;

  /// Enable global workspace reuse.
  bool enableGlobalReuse;

  /// enable HIVM op plan memory inplace
  bool restrictInplaceAsISA;

  /// StorageEntry generate.
  void GenerateStorageEntry();

  /// Prepare the memref.alloc plan.
  PlanStatus PlanLocalMemAddress();

  /// Prepare the memrefExt.alloc_workspace plan.
  PlanStatus PlanWorkSpaceMemAddress();

  /// merge all storage entry to the first storage entry for WorkSpaceArg.
  void MergeSameWorkSpaceArgSE();

  /// Start plan for same work space arg offset.
  PlanStatus PlanMemOffsetOfWholeWorkSpace();

  /// Enable global workspace no reuse.
  void GlobalWorkspaceNoReuse(StorageEntry *rootStorageEntry);

  /// Verify that constBits is legal.
  void ValidateParameters(std::unique_ptr<StorageEntry> &e) const;

  /// Expanding the Storage Entry due to the addition of MultiBuffer.
  void ExpandMultiBufferStorageEntry();

  /// merge all storage entry to the first storage entry.
  void MergeSameScopeSE();

  /// merge all storage entry which can be inplaced.
  void MergeInplaceSE();

  /// Start plan.
  PlanStatus PlanMemAddressOfWholeLocalBuffer();

  /// Plan memory only by level0 to report failure info.
  void PlanMemAddressForLevel0(StorageEntry *rootStorageEntry);

  /// Determine if the current space is enough to allocate all buffers.
  bool IsEnoughForBuffersNoReuse(StorageEntry *rootStorageEntry,
                                 size_t restBufferSize, size_t alignUnit);

  /// Adjust the allocation order of rootStoreEntry to prioritize the allocation
  /// of buffers corresponding to DMA.
  StorageEntry *GetReorderRootStorageEntry(StorageEntry *rootStorageEntry);

  /// Assign addresses without reuse.
  void PlanBuffersWithoutReuse(StorageEntry *rootStorageEntry,
                               size_t alignUnit);

  /// Obtain buffer space size and alignment information.
  std::pair<size_t, size_t> GetBufferSpaceInfo(hivm::AddressSpace &space) const;

  /// Emit buffer applied failure message.
  void EmitPlanMemoryFailureInfo();

  /// Multi level plan strategy.
  LogicalResult MultiSpecPlan(SpecInfo &si, MemBoundList &outline,
                              PlanRecHis &history, StorageEntry *entry);

  /// plan buffer in speculative ways.
  LogicalResult SpecAlloc(MemBoundList &outline, PlanRecHis &his,
                          StorageEntry *e, const SpecInfo &si, int localLevel);

  /// spec_level == SPEC_LEVEL_2, mte2/3 do not reuse with vector.
  bool VerifyConflictStage2(PlanRecHis &his, const StorageEntry *e,
                            int specLevel, MemBoundListConstIter &start,
                            const MemBoundList &outline);

  /// spec_level == SPEC_LEVEL_1, pure single can reuse with db.
  bool VerifyConflictStage1(MemBoundList &outline, PlanRecHis &his,
                            StorageEntry *e,
                            const OutlineSectionInfo &outlineInfo,
                            uint64_t &pongOffset);

  /// check if e1 and e2 has pipe conflict.
  bool PipeConflict(const StorageEntry *e1, const StorageEntry *e2,
                    DenseMap<StorageEntryPair, bool> &conflictMap);

  /// spec_level == SPEC_LEVEL_2, MTE2/MTE3 is pipe conflict with all existing
  /// allocation. check if current entry has OptDmaPipe-conflict with buffers
  /// already allocate at current position. if conflict exists, continue loop
  /// until first not-conflict iter is found. Then update start as the first
  /// bound right before the not-conflict one.
  bool VerifyDmaPipeConflict(const StorageEntry *e, int specLevel,
                             MemBoundListConstIter &start,
                             MemBoundListConstIter &end);

  /// Check if it matches the previous rollback result.
  bool IsSamePlanAsLastRollBack(uint64_t allocOffset, int curChildIdx,
                                const SpecInfo &si) const;

  /// spec_level == SPEC_LEVEL_0, life time reuse.
  inline bool VerifyConflictStage0(StorageEntry *e,
                                   const std::shared_ptr<MemoryBound> &last);

  /// Update the outline information and record history
  void UpdateOutline(MemBoundList &outline, PlanRecHis &his, StorageEntry *e,
                     const OutlineSectionInfo &outlineInfo,
                     int localLevel) const;

  /// plan strategy is achieved through split method.
  void AddMemBoundInSectionalWay(
      StorageEntry *e, MemBoundListConstIter start, MemBoundListConstIter end,
      SmallVector<std::shared_ptr<MemoryBound>> &splitBound) const;

  /// merge the buffer life between start and end.
  inline void MergeBufferLife(MemBoundList::const_iterator start,
                              MemBoundList::const_iterator end,
                              BufferLifeVec &newLife) const;

  /// merge buffers in a vector.
  void MergeBufferVec(BufferLifeVec &bufferLife) const;

  /// Judge if need to restart plan memory with other strategy after
  /// plan failed.
  PlanStatus ApplyFailStrategy(StatusWrapper &statusWrapper,
                               const size_t maxBits);

  void RollBackForAllocFail(StatusWrapper &statusWrapper, const size_t maxBits);

  /// Check if memory plan can be rolled back.
  bool ContinueRollBack(const StatusWrapper &statusWrapper) const;

  /// Memory plan fallback information processing.
  void RollBackForAllocFailInner(StatusWrapper &statusWrapper,
                                 const size_t maxBits);

  /// Fallback outline plan.
  PlanRecord RollbackOutline(PlanRecHis &history, MemBoundList &outline) const;

  /// Update the plan memory address corresponding to mem buffer.
  void UpdateBuffer2Offsets();

  /// Update extra addresses offset caused by multi buffer reuse.
  void UpdateMultiBufferReuseExtraOffset();

  /// generate inplace list by some rules
  SmallVector<ValuePair> GenerateInplaceList();

  /// the hivmop that can reuse dst address and src address in limited situation
  bool IsReuseHIVMOp(Operation *op, const Value &genBuffer,
                     const Value &killBuffer) const;

  /// Get overlap buffer life.
  DenseMap<ValuePair, BufferLife>
  GetOverlapBufferLife(const BufferLifeVec &b1, const BufferLifeVec &b2) const;

  /// Reorder and make the storage entries of ping and pong continuous.
  void
  ReorderContinuousPingPongEntry(SmallVector<StorageEntry *> &storageEntryVec);

  /// Determine if the current buffer life of the Storage Entry conflicts with
  /// the memory that has already been allocated in history.
  bool IsBufferLifeVecConflict(PlanRecord &r, uint64_t offset,
                               const StorageEntry *e) const;

  /// Assign pong storage entry's address.
  void PlanRelationPongEntryAddress(uint64_t offset, StorageEntry *e);

  /// Processing Pong Storage Entry Information.
  void SpecAllocRelationPongEntry(MemBoundList &outline, PlanRecHis &his,
                                  StorageEntry *e, uint64_t offset);

  /// Get relative pong storage entry when the current reuse bound storage entry
  /// is of type db.
  StorageEntry *
  GetMultiRelationPongEntry(const StorageEntry *reuseBoundStorageEntry);

  /// Get the innermost for loop of buffer definition.
  LoopLikeOpInterface GetBufferParentLoop(const SmallVector<Value> &buffers);

  /// Report all tensors life time info.
  void ReportMemLifeDebugInfo(const StorageEntry *rootStorageEntry);

  /// Report tensor life time for debug.
  void MemLifeDebugInfo(const StorageEntry *storageEntry) const;

  /// Record successfully planed memories.
  void RecordAllocatedEntry(const StorageEntry *e);

  /// Report tensor which is defined by memref allco.
  void ReportCurEntryDebugInfo(const StorageEntry *curEntry) const;

  /// Report tensor allocate info.
  void ReportAllocatedEntryDebugInfo(const StorageEntry *rootStorageEntry,
                                     bool isFail) const;

private:
  /// The buffer corresponding to each operation.
  SmallVector<std::unique_ptr<OpInfo>> linearOperation;

  /// map from buffer value to its buffer information.
  std::map<Value, BufferInfo, utils::ValueComparator> bufferInfos;

  /// map from buffer to its lifetime.
  DenseMap<Value, std::shared_ptr<BufferLife>> buffer2Life;

  /// record the map from the buffer to its number of buffer if it does
  /// multibuffer optimization.
  /// note: the map only record the buffer which do multi buffer optimization
  /// and ignore single buffer.
  DenseMap<Value, uint32_t> buffer2MultiNum;

  /// map from operation to its gen and kill buffer.
  DenseMap<OpInfo *, GenKillEntry> genKillMap;

  /// record all storage entry to be plan address.
  SmallVector<std::unique_ptr<StorageEntry>> StorageEntryVec;

  /// The current status of memory plan.
  PlanStatus planStatus{PlanStatus::PLAN_SUCCESS};

  /// Whether to adopt a split strategy.
  bool splitOutline{false};

  /// map from memref buffer to plan memory address.
  DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;

  /// map from each scope to its root StorageEntry.
  DenseMap<hivm::AddressSpace, StorageEntry *> memscope2rootStorageEntry;

  /// map from workspace arg to its root StorageEntry.
  DenseMap<Value, StorageEntry *> workSpaceArg2rootStorageEntry;

  /// map from buffer scope to its required size to plan rest memory without any
  /// reuse.
  DenseMap<hivm::AddressSpace, size_t> bufferScope2RequiredSize;

  /// map from buffer value to its storage entry info
  DenseMap<Value, StorageEntry *> buffer2storageEntry;

  /// Memory dma pipe first plan optimization.
  OptMemPlanForDma dmaFirstPipelineOpt;

  /// Map from the storage entry pair to its pipeDma conflict info.
  DenseMap<StorageEntryPair, bool> pipeDmaConflictMap;

  /// Ping storage entry corresponding to reused additional Pong entry.
  DenseMap<StorageEntry *, std::unique_ptr<StorageEntry>>
      pingEntry2RelationPongEntry;

  DenseMap<hivm::AddressSpace, SmallVector<const StorageEntry *>>
      memscope2allocatedEntry;

  /// record inplace pair list.
  SmallVector<ValuePair> inplacePairList;

  /// The scope of the buffer applied memory fail and the max bits it applied.
  std::map<hivm::AddressSpace, uint64_t> failApplyBufferInfo;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_PLAN_MEMORY_H
