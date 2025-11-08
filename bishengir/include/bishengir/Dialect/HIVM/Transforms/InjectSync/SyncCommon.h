//===- SyncCommon.h ----sync correlation common code ----------------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_SYNC_COMMON_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_SYNC_COMMON_H

#include <utility>

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include <deque>

#define MAX_MULTI_BUFFER_NUM 16

namespace mlir {
namespace hivm {

/// different mode for sync analysis.
enum SyncAnalysisMode { NORMALSYNC = 0, BLOCKSYNC };

/// Meminfo of the target buffer
struct BaseMemInfo {
  BaseMemInfo(
      Value baseBuffer, Value rootBuffer, hivm::AddressSpace scope,
      SmallVector<uint64_t> baseAddresses, uint64_t allocateSize,
      std::optional<bishengir::memref_ext::AllocWorkspaceOp> allocWorkspaceOp)
      : baseBuffer(baseBuffer), rootBuffer(rootBuffer), scope(scope),
        baseAddresses(std::move(baseAddresses)), allocateSize(allocateSize),
        allocWorkspaceOp(std::move(allocWorkspaceOp)) {}
  /// baseBuffer means the buffer used for the corresponding operation.
  Value baseBuffer;
  /// rootBuffer means the result operand of defining alloc op or function
  /// argument.
  Value rootBuffer;
  hivm::AddressSpace scope;
  SmallVector<uint64_t> baseAddresses;
  uint64_t allocateSize;
  /// when specified alloc_workspace op, it is workspace gm argument, otherwise,
  /// it is normal input or output gm argument.
  std::optional<bishengir::memref_ext::AllocWorkspaceOp> allocWorkspaceOp;

  bool areVectorEqual(SmallVector<uint64_t> vec1,
                      SmallVector<uint64_t> vec2) const {
    if (vec1.size() != vec2.size()) {
      return false;
    }
    for (size_t i = 0; i < vec1.size(); ++i) {
      if (vec1[i] != vec2[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator==(const BaseMemInfo &other) const {
    if (!areVectorEqual(baseAddresses, other.baseAddresses)) {
      return false;
    }
    if (rootBuffer != other.rootBuffer) {
      return false;
    }
    if (scope != other.scope) {
      return false;
    }
    if (allocateSize != other.allocateSize) {
      return false;
    }
    if (baseBuffer != other.baseBuffer) {
      return false;
    }
    return true;
  }

  std::unique_ptr<BaseMemInfo> clone() const {
    auto newMemInfo = std::make_unique<BaseMemInfo>(
        baseBuffer, rootBuffer, scope, baseAddresses, allocateSize,
        allocWorkspaceOp);
    return newMemInfo;
  }

  std::unique_ptr<BaseMemInfo> clone(Value cloneBaseBuffer) const {
    auto newMemInfo = std::make_unique<BaseMemInfo>(
        cloneBaseBuffer, rootBuffer, scope, baseAddresses, allocateSize,
        allocWorkspaceOp);
    return newMemInfo;
  }
};

using DepBaseMemInfoPairVec =
    SmallVector<std::pair<const BaseMemInfo *, const BaseMemInfo *>>;

class SyncOperation {
public:
  enum class TYPE {
    SET_EVENT,
    WAIT_EVENT,
    PIPE_BARRIER,
    PIPE_BARRIER_CUBE,
    PIPE_BARRIER_VECTOR,
    SYNC_BLOCK_SET,
    SYNC_BLOCK_WAIT,
    SYNC_BLOCK_ALL,
  };
  // event id = kNullEventId, when no event ID is allocated.
  static const int kNullEventId{-1};

public:
  // Indicates the event ID used by the hardware. Can use 0 - 4.
  SmallVector<int> eventIds;

  // Indicates whether to insert the synchronization when codegen sync.
  bool uselessSync{false};

  // Need to allocate the size of eventId.
  int eventIdNum{1};

  // the lowest common ancestor one of buffers that generate the sync operation
  // dependency
  Value lowestCommonAncestorBuffer{nullptr};

  // counter number of the sync operation that is reused by other sync through
  // widen when event id is not enough.
  int reuseCntForWiden{0};

  bool reallocatedLoopHeadTailSync{false};

  TCoreType syncCoreType{TCoreType::CUBE_OR_VECTOR};

  Value block_sync_event_value{nullptr};

public:
  SyncOperation(TYPE type, hivm::PIPE srcPipe, hivm::PIPE dstPipe,
                unsigned kSyncIndex, unsigned syncIRIndex,
                std::optional<int> forEndIndex)
      : eventIds({}), type_(type), srcPipe_(srcPipe), dstPipe_(dstPipe),
        kSyncIndex_(kSyncIndex), syncIRIndex_(syncIRIndex),
        forEndIndex_(forEndIndex) {};

  ~SyncOperation() = default;

  std::unique_ptr<SyncOperation> GetMatchSync(unsigned index) const;
  TYPE GetType() const { return type_; }
  hivm::PIPE GetSrcPipe() const { return srcPipe_; }
  hivm::PIPE GetDstPipe() const { return dstPipe_; }
  hivm::PIPE GetActualSrcPipe() const {
    return ((srcPipe_ == hivm::PIPE::VIRTUAL_PIPE_MTE2_L1A) ||
            (srcPipe_ == hivm::PIPE::VIRTUAL_PIPE_MTE2_L1B))
               ? hivm::PIPE::PIPE_MTE2
               : srcPipe_;
  }
  hivm::PIPE GetActualDstPipe() const {
    return ((dstPipe_ == hivm::PIPE::VIRTUAL_PIPE_MTE2_L1A) ||
            (dstPipe_ == hivm::PIPE::VIRTUAL_PIPE_MTE2_L1B))
               ? hivm::PIPE::PIPE_MTE2
               : dstPipe_;
  }
  SmallVector<int> GetEventIDs() const { return eventIds; }
  unsigned GetSyncIndex() const { return kSyncIndex_; }
  unsigned GetSyncIRIndex() const { return syncIRIndex_; }
  void SetSyncIRIndex(unsigned index) { syncIRIndex_ = index; }
  std::optional<int> GetForEndIndex() const { return forEndIndex_; }
  void SetDepSyncIRIndex(unsigned index) { depSyncIRIndex_ = index; }
  unsigned GetDepSyncIRIndex() const { return depSyncIRIndex_; }
  void SetType(TYPE syncType) { type_ = syncType; }
  bool operator==(const SyncOperation &other) const;

  bool isSyncSetType() const;
  bool isSyncWaitType() const;
  bool isBarrierType() const;

  // Returns the intrinsic function name for a Type
  static std::string TypeName(TYPE t);
  std::string GetCoreTypeName(TCoreType t) const;

  using SyncOperations =
      SmallVector<SmallVector<std::unique_ptr<SyncOperation>>>;

  void SetPipeAll();

private:
  // Sync type
  TYPE type_;

  // Type of the Src intrinsic that generates the dependency relationship.
  hivm::PIPE srcPipe_;

  // Type of the Dst intrinsic that generates the dependency relationship.
  hivm::PIPE dstPipe_;

  // Primary key of the sync ID that increases from 0.
  const unsigned kSyncIndex_;

  // Record corresponding syncIR index
  unsigned syncIRIndex_;

  // record the parent loop end index in syncIR
  std::optional<int> forEndIndex_{};

  // the index of ops with dependency in syncIR
  unsigned depSyncIRIndex_{0};
};

// used to store references of sync operations before/after an InstanceElement
using SyncOps = std::deque<SyncOperation *>;

class InstanceElement {
public:
  Operation *elementOp = nullptr;

  // sync is inserted above the element when codegen is executed.
  SyncOps pipeBefore;

  // sync is inserted under the element when codegen is executed.
  SyncOps pipeAfter;

  enum class KindTy { COMPOUND, LOOP, BRANCH };

public:
  virtual ~InstanceElement() = default;

  unsigned GetIndex() const { return kIndex; }

  KindTy GetKind() const { return kKindTy; }

  // Delete one of the sync lines from a sync vector.
  static bool RemoveSync(SyncOps &syncVector, const SyncOperation *sync);

protected:
  // Each InstanceElement subclass has a unique index,
  // and the index increases from 0.
  const unsigned kIndex;

  // The parent class cannot directly initialize the instance.
  InstanceElement(KindTy kind, unsigned index)
      : kIndex(index), kKindTy(kind) {};

private:
  const KindTy kKindTy;
};

enum class KindOfLoop { LOOP_BEGIN, LOOP_END };

// Identifies one iteration of a for loop operation
class LoopInstanceElement : public InstanceElement {
public:
  unsigned beginId;

  unsigned endId;

public:
  LoopInstanceElement(unsigned index, unsigned beginId, unsigned endId,
                      KindOfLoop loopKind = KindOfLoop::LOOP_BEGIN)
      : InstanceElement(KindTy::LOOP, index), beginId(beginId), endId(endId),
        kLoopKind(loopKind) {}

  ~LoopInstanceElement() override = default;

  std::unique_ptr<InstanceElement> CloneFor(KindOfLoop loopKind) const;

  KindOfLoop getLoopKind() const { return kLoopKind; }

  static bool classof(const InstanceElement *e);

  bool ignore_block_sync_move_out{false};

private:
  const KindOfLoop kLoopKind;
};

enum class KindOfBranch { IF_BEGIN, ELSE_BEGIN, IF_END };

// Identifies one statement within a if operation
class BranchInstanceElement : public InstanceElement {
public:
  unsigned beginId;

  unsigned branchId;

  unsigned endId{0};

public:
  BranchInstanceElement(unsigned index, unsigned beginId,
                        KindOfBranch branchKind = KindOfBranch::IF_BEGIN)
      : InstanceElement(KindTy::BRANCH, index), beginId(beginId),
        branchId(beginId), kBranchKind(branchKind) {}

  ~BranchInstanceElement() override = default;
  std::unique_ptr<InstanceElement> CloneBranch(KindOfBranch branchKind) const;
  KindOfBranch getBranchKind() const { return kBranchKind; }

  static bool classof(const InstanceElement *e);

  BranchInstanceElement(unsigned index, unsigned beginId, unsigned branchId,
                        unsigned endId,
                        KindOfBranch branchKind = KindOfBranch::IF_BEGIN)
      : InstanceElement(KindTy::BRANCH, index), beginId(beginId),
        branchId(branchId), endId(endId), kBranchKind(branchKind) {}

private:
  const KindOfBranch kBranchKind;
};

// Identifies one statement within a compound operation
class CompoundInstanceElement : public InstanceElement {
public:
  // Write/Read tensor in the storage Coproc
  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;

  // Intrinsic pipeline type
  hivm::PIPE kPipeValue;

  // The string name of the Compound node
  OperationName opName;

  // The CoreType for compound node.
  TCoreType compoundCoreType{TCoreType::CUBE_OR_VECTOR};

  SyncOperation *PipeMTE1ToPipeMSync{nullptr};

  // One operation can be synchronized with 2 other operations, one before it
  // and one after it ex: fixpipe-mmadl1-fixpipe. Usually using set/wait flags
  // we would have this: (fixpipe/set) (wait/mmadl1/set) (wait/fixpipe)
  // unitFlagModeAsSet is used to handle when an operation is synchronized with
  // an operation after it, and the opposite is true for unitFlagModeAsWait.
  UNIT_FLAG unitFlagModeAsSet{UNIT_FLAG::DISABLED};
  UNIT_FLAG unitFlagModeAsWait{UNIT_FLAG::DISABLED};
  CompoundInstanceElement *linkedUnitFlagCompAsSet{nullptr};
  CompoundInstanceElement *linkedUnitFlagCompAsWait{nullptr};

public:
  CompoundInstanceElement(unsigned index,
                          SmallVector<const BaseMemInfo *> defVec,
                          SmallVector<const BaseMemInfo *> useVec,
                          const hivm::PIPE PipeValue, OperationName opName)
      : InstanceElement(KindTy::COMPOUND, index), defVec(std::move(defVec)),
        useVec(std::move(useVec)), kPipeValue(PipeValue), opName(opName) {}

  ~CompoundInstanceElement() override = default;

  static bool classof(const InstanceElement *e);

  // Check if unit-flag synchronization is enabled for current operation and
  // return ENABLED if so.
  UNIT_FLAG getUnitFlagMode() const;
  std::optional<mlir::Value> getUnitFlagCond(Location loc,
                                             IRRewriter &rewriter) const;
};

// Save all operation information on IR in order, with three main types of
// forOP, ifOP and instructionOP.
using SyncIRs = SmallVector<std::unique_ptr<InstanceElement>>;

// Save synchronization instructions in order of allocation, where size 1 is
// pipe_barrier and size 2 is paired set_flag and wait_flag.
using SyncOperations = SmallVector<SmallVector<std::unique_ptr<SyncOperation>>>;

// the map from the buffer to its mem info
using Buffer2MemInfoMap =
    llvm::DenseMap<Value, llvm::SmallVector<std::unique_ptr<BaseMemInfo>>>;

// check and assert that index is within the bounds of syncIR
void checkSyncIRIndex(const SyncIRs &syncIR, int index);

// check and assert given condition
void checkCondition(bool condition, const std::string &message);

} // namespace hivm
} // namespace mlir

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_SYNC_COMMON_H