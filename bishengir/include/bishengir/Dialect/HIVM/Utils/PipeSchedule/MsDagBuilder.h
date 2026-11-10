//===- MsDagBuilder.h - MS DAG from MLIR IR -------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
//
// Bridges MLIR HIVM IR to the pure-STL SMS scheduler (sms::DDG).
//
// Primary (DAG1/DAG2/DAG3) path — build from the authoritative Default
// WorkItem partition (WorklistBuildResult):
//
//   DAG1  one compute DDG node per Default WorkItem (ops, core, shape/dtype
//         metadata, WorkItem-granular dependencies);
//   DAG2  synthetic, independently schedulable transfer nodes derived from
//         the DAG1 data edges following the CVpie storage-transition rules
//         (mte2 GM→L1/UB, mte3 UB→L1/GM, mte1 L1→L0A/L0B, fixpipe L0C→…);
//   DAG3  non-schedulable data nodes (shape/dtype/bytes/storage/producer/
//         consumers) for validation and memory planning.
//
// The builder NEVER creates, splits, merges, or re-owns WorkItems — the
// Default partition is authoritative.  Same-WorkItem loop-carried
// recurrences are legal (handled by the existing iter_arg/yield codegen);
// cross-WorkItem recurrences reject the build so the caller falls back to
// the Default scheduler.
//
// A legacy per-op path (no WorkItems) remains for unit tests; it performs a
// READ-ONLY analysis mirroring WorklistBuilder's classification without
// mutating operation attributes.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_MSDAGBUILDER_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_MSDAGBUILDER_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/CostModel.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DDG.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/OpNode.h"
#include "bishengir/Dialect/HIVM/Utils/WorklistBuilder.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hivm {

// Import the pure-STL sms types into this namespace for concise references.
namespace sms = ::bishengir::hivm::sms;

//===----------------------------------------------------------------------===//
// MsNode — per-DDG-index metadata bridging Operation* to the scheduler
//===----------------------------------------------------------------------===//

/// Classification of an MS DAG node.
enum class MsNodeKind : uint8_t {
  /// A single cube-core compute op (e.g. hivm.hir.mmad_l1).
  CubeCompute,
  /// An indivisible group of vector-core compute ops.
  VectorGroup,
  /// A transfer/DMA op: MTE1, MTE2, MTE3, or fixpipe.
  Transfer,
};

/// One node in the MS DAG, mapping a DDG index to MLIR IR.
struct MsNode {
  /// DDG node index (stable ordinal assigned in block-walk order).
  int ddgIndex = -1;

  /// Classification.
  MsNodeKind kind = MsNodeKind::Transfer;

  /// The primary MLIR operation.  For CubeCompute and Transfer nodes this is
  /// the single op.  For VectorGroup nodes this is the first op in the group
  /// (the one that appears first in block order).
  Operation *primaryOp = nullptr;

  /// For VectorGroup: all ops in this indivisible group (in block order).
  /// For other kinds: contains only primaryOp.
  SmallVector<Operation *, 4> ops;

  /// Execution unit pipeline name (lowercase): "cube", "vector", "vector1",
  /// "vector2", "mte1", "mte2", "mte3", "fixpipe".
  std::string pipeline;

  /// True if this node's pipeline is a parallel (non-single-occupancy) unit
  /// — i.e. MTE1/MTE2/MTE3/fixpipe.
  bool isParallelUnit = false;

  // ===========================================================================
  // DAG1 WorkItem linkage — populated when building from Default WorkItems.
  // Null / -1 for transfer nodes that are not WorkItem-aligned (Phase 2–3
  // transition replaces IR-derived transfer nodes with CVpie-derived ones).
  // ===========================================================================

  /// Owning Default WorkItem.  Compute nodes always have a WorkItem; transfer
  /// nodes derived from IR ops are associated with a WorkItem (the one that
  /// owns the op), but synthetic DAG2 transfer nodes may have null.
  WorkItem *workItem = nullptr;

  /// Index of the WorkItem in the Default worklist (stable ordinal).
  int workItemIndex = -1;

  /// Shape and element-type metadata for an operand or result of a core op.
  struct OperandInfo {
    SmallVector<int64_t> shape;
    Type elementType;
    /// True for an input operand; false for an output / result.
    bool isInput = true;
    /// For CUBE nodes, the operand index (0 = A, 1 = B).  -1 for vector ops.
    int operandIndex = -1;
    /// Owning op.  For VectorGroup nodes the operandInfos of the whole group
    /// form one flat list; this field marks the op boundary so per-op cost
    /// inputs can be regrouped (buildVectorGroupCostInput).  Null for
    /// entries whose op could not be recorded.
    Operation *ownerOp = nullptr;
  };

  /// Collected shape/dtype info for the compute node's core ops.
  SmallVector<OperandInfo> operandInfos;

  // ===========================================================================
  // DAG2 synthetic transfer metadata — populated by buildTransferNodes().
  // Only meaningful when isSynthetic is true.
  // ===========================================================================

  /// True for synthetic transfer nodes created by DAG2 (no corresponding
  /// MLIR operation — not part of any WorkItem).
  bool isSynthetic = false;

  /// Storage transition tag (e.g., "GM_to_L1", "L1_to_L0A", "L0C_to_UB").
  std::string storageTag;

  /// DDG index of the source compute node (-1 for GM source or non-transfer).
  int sourceComputeIdx = -1;

  /// DDG index of the target compute node (-1 for GM sink or non-transfer).
  int targetComputeIdx = -1;

  /// For cube input transfers: 0 = A (L0A), 1 = B (L0B).  -1 otherwise.
  int cubeOperandSlot = -1;
};

//===----------------------------------------------------------------------===//
// MsDataNode — DAG3 non-schedulable data metadata
//===----------------------------------------------------------------------===//

/// Ascend NPU on-chip storage types used in DAG3 data nodes.
enum class MsStorage : uint8_t {
  GM,      ///< Global Memory (HBM/DDR)
  L1,      ///< Level-1 buffer
  L0A,     ///< Level-0 A buffer (cube input A)
  L0B,     ///< Level-0 B buffer (cube input B)
  L0C,     ///< Level-0 C accumulator (cube output)
  UB,      ///< Unified Buffer (vector I/O)
  Virtual, ///< Logical-only (loop/prefetch connections)
};

/// A data node recording the shape, type and storage of a data block flowing
/// between DAG1/DAG2 OpNodes.  Data nodes are NOT part of the DDG/SMS
/// resource graph; they exist for validation and memory planning.
struct MsDataNode {
  int id = -1;

  SmallVector<int64_t> shape;
  Type elementType;
  int64_t elementBytes = 0;
  int64_t totalBytes = 0;

  MsStorage storage = MsStorage::GM;

  /// DDG index of the producer OpNode (compute or transfer).
  int producerIdx = -1;
  /// DDG indices of consumer OpNodes.
  SmallVector<int, 2> consumerIdxs;

  bool isVirtual = false;
  bool noTransfer = false;
};

//===----------------------------------------------------------------------===//
// LatencyProvider — injectable cost model
//===----------------------------------------------------------------------===//

/// Abstract interface returning positive-integer latency/issue-time for an
/// operation.  Allows unit tests to inject deterministic costs without an
/// actual hardware model.
struct LatencyProvider {
  virtual ~LatencyProvider() = default;

  /// Return the execution latency (cycles) of `op`.  Must be > 0.
  virtual int getLatency(Operation *op) const = 0;

  /// Return the issue time (resource occupation in the MRT) of `op`.
  /// Must be > 0.  Defaults to `getLatency(op)`.
  virtual int getIssueTime(Operation *op) const { return getLatency(op); }
};

//===----------------------------------------------------------------------===//
// MsDagBuildResult — everything the downstream partitioner needs
//===----------------------------------------------------------------------===//

struct MsDagBuildResult {
  /// The pure-STL DDG, ready for smsSchedule().
  sms::DDG ddg;

  /// One MsNode per DDG index (indexed by DDG node ordinal).
  SmallVector<MsNode, 1> nodes;

  /// Forward map: Operation* → DDG index that owns it.
  DenseMap<Operation *, int> opToDdgIndex;

  /// For each DDG index, the set of DDG indices it depends on (data
  /// predecessors including separator/SSA/memory/transfer edges).
  /// Mirrors the DDG's own edge structure but keyed by DDG index for quick
  /// lookup.
  DenseMap<int, DenseSet<int>> dependenceMap;

  /// For each `bufferization::to_tensor`, the DPS writer op, mirroring
  /// WorklistBuilder::outputMemrefMap semantics (read-only trace, no attr
  /// mutation).
  DenseMap<bufferization::ToTensorOp, DestinationStyleOpInterface>
      outputMemrefMap;

  /// True if loop-carried dependencies were detected (MS rejects these in
  /// P0 — no RecMII support).
  bool hasLoopCarriedDeps = false;

  /// Diagnostic string when loop-carried deps are detected.
  std::string loopCarriedDiag;

  // ===========================================================================
  // DAG1 metadata — populated when building from Default WorkItems.
  // ===========================================================================

  /// Forward map: WorkItem → DDG index of its compute node.
  DenseMap<WorkItem *, int> workItemToDdgIndex;

  /// Authoritative Default worklist snapshot (shared_ptrs).
  SmallVector<std::shared_ptr<WorkItem>> defaultWorklist;

  /// Authoritative Default opToWorkItemMap snapshot.
  DenseMap<Operation *, SmallVector<WorkItem *>> defaultOpToWorkItemMap;

  /// Number of compute DDG nodes (indices [0, computeNodeCount) are compute;
  /// indices [computeNodeCount, ddg.size()) are DAG2 synthetic transfers).
  int computeNodeCount = 0;

  /// DAG3 data nodes — non-schedulable metadata for validation and memory
  /// planning.  Not part of the DDG.
  SmallVector<MsDataNode> dataNodes;
};

//===----------------------------------------------------------------------===//
// MsDagBuilder — the main builder
//===----------------------------------------------------------------------===//

class MsDagBuilder {
public:
  /// Construct a builder for the given pipeline loop using Default WorkItems.
  ///
  /// \param loop              The scf::ForOp whose body is being pipelined.
  /// \param latencyProvider   Cost model for operation latencies.
  /// \param counterClones     Counter-advance clone map (same as passed to
  ///                          WorklistBuilder::setCounterClones).
  /// \param defaultBuild      Authoritative WorkItem partition from the Default
  ///                          WorklistBuilder.  The builder creates one compute
  ///                          node per WorkItem and preserves the ownership and
  ///                          dependency structure.
  /// \param costModel         Optional MS cost model.  When non-null,
  ///                          assignCosts() replaces the LatencyProvider
  ///                          placeholder latencies with model-derived costs;
  ///                          when null the placeholders are kept (backwards
  ///                          compatible).
  MsDagBuilder(scf::ForOp loop, const LatencyProvider &latencyProvider,
                const DenseMap<Value, Operation *> &counterClones,
                const WorklistBuildResult &defaultBuild,
                const costmodel::CostModel *costModel = nullptr);

  /// Legacy constructor without Default WorkItems (deprecated — will be
  /// removed once all callers pass the build result).
  MsDagBuilder(scf::ForOp loop, const LatencyProvider &latencyProvider,
                const DenseMap<Value, Operation *> &counterClones =
                    DenseMap<Value, Operation *>());

  /// Build the DAG.  Returns failure if the loop body contains patterns that
  /// MS cannot handle (mixed-core region ops, unsupported units, zero
  /// latency, etc.).  On failure the IR is unmodified — no attributes set,
  /// no side effects.
  FailureOr<MsDagBuildResult> build();

  /// True after build() fails due to a cross-WorkItem loop-carried recurrence.
  bool failedDueToLoopCarriedDependency() const {
    return result.hasLoopCarriedDeps;
  }

private:
  //--- Controlled-copy classifiers (read-only, from WorklistBuilder) --------

  /// cf. WorklistBuilder.cpp: traceValueDef
  static Value traceValueDef(Value v);

  /// cf. WorklistBuilder.cpp: traceAlloc
  static memref::AllocOp traceAlloc(Value v);

  /// cf. WorklistBuilder.cpp: isCrossCoreCopy
  static bool isCrossCoreCopy(Operation *op);

  /// cf. WorklistBuilder.cpp: isSeparator
  static bool isSeparator(Operation *op);

  /// cf. WorklistBuilder.cpp: isCoreOp (READ-ONLY: checks existing attrs and
  /// HIVM DPS, never sets cubeonly/veconly)
  static bool isCoreOp(Operation &op);

  /// cf. WorklistBuilder.cpp: isLoadLikeOp
  static bool isLoadLikeOp(Operation *op);

  /// cf. WorklistBuilder.cpp: isMemrefSubnetWriter
  static bool isMemrefSubnetWriter(Operation *op);

  /// cf. WorklistBuilder.cpp: isExtractedScalarGather
  static bool isExtractedScalarGather(scf::ForOp forOp);

  /// cf. WorklistBuilder.cpp: getContainedParent
  static Operation *getContainedParent(Operation *containing, Operation &inner);
  static Operation *getContainedParent(Operation *containing, Value inner);

  /// cf. WorklistBuilder.cpp: memrefDFS
  static void memrefDFS(Value memrefVal, SmallVector<Operation *> &users);

  /// Read-only variant of illegalRegionedOp: classifies region ops into
  /// {cube, vector, mixed, none} WITHOUT setting attributes.  Returns the
  /// core type of the region body.
  enum class RegionCoreKind { Cube, Vector, Mixed, None };
  static RegionCoreKind classifyRegion(Operation &op);

  //--- DAG1 analysis phases (WorkItem-based) -------------------------------

  /// Phase 1-WI (DAG1): Create one compute DDG node per Default WorkItem.
  /// IR transfer ops stay inside their WorkItem's node; transfer timing is
  /// modeled by DAG2 synthetic nodes.  Populates result.nodes,
  /// result.opToDdgIndex, and the DDG node array.
  LogicalResult buildNodesFromWorkItems();

  /// Phase 2-WI (DAG1): Build inter-node SSA + separator dependency edges
  /// and reject cross-WorkItem loop-carried recurrences.
  LogicalResult buildWorkItemDependencies();

  /// Phase 3-WI: Extract operand shape/dtype metadata for compute nodes.
  void collectOperandMetadata();

  /// Phase DAG2: Insert synthetic transfer DDG nodes between compute nodes
  /// following CVpie storage-transition rules.  Adds transfer nodes and
  /// edges to the DDG on top of the existing compute-to-compute edges.
  LogicalResult buildTransferNodes();

  /// Phase DAG3: Derive non-schedulable data nodes for validation and
  /// memory planning.  Must be called after buildTransferNodes().
  LogicalResult buildDataNodes();

  /// Phase cost: Query the MS cost model for every compute and transfer
  /// node, backfill ddgNode.latency/issueTime, and refresh edge latencies.
  /// No-op when no cost model was injected.  Must be called after
  /// buildDataNodes() (transfer inputs come from the data nodes).
  LogicalResult assignCosts();

  /// Extract the cube query input (slots A/B) from a CubeCompute node's
  /// operandInfos.  Returns a failed result when the metadata is unusable
  /// (missing shaped slots).
  FailureOr<costmodel::CubeCostInput>
  buildCubeCostInput(const MsNode &sn) const;

  /// Group a VectorGroup node's flat operandInfos into per-op
  /// VectorOpDescs (op boundaries from OperandInfo::ownerOp).  Transfer ops
  /// are skipped — their cost is billed by the transfer module.
  costmodel::VectorGroupCostInput
  buildVectorGroupCostInput(const MsNode &sn) const;

  /// Extract the transfer query input from a synthetic transfer node's
  /// associated MsDataNode.  Returns a failed result when no data node
  /// carries a usable shape/dtype.
  FailureOr<costmodel::TransferCostInput>
  buildTransferCostInput(const MsNode &sn) const;

  //--- Legacy analysis phases (op-level, used without Default WorkItems) ---

  /// Phase 1: Walk the scope body in block order, classify all ops, identify
  /// separators, compute nodes (cube + vector groups), and transfer nodes.
  LogicalResult classifyOps();

  /// Phase 2: Build separator-forward SSA dependency edges.
  /// cf. WorklistBuilder.cpp: populateDependencies
  LogicalResult buildSsaDependencies();

  /// Phase 3: Detect loop-carried dependencies.
  /// cf. WorklistBuilder.cpp: populateLoopCarriedDependencies
  /// P0: any loop-carried dependency rejects the loop (sets
  /// result.hasLoopCarriedDeps).
  void detectLoopCarriedDeps();

  /// Phase 4: Trace memref subnets for output registration.
  /// cf. WorklistBuilder.cpp: traceMemrefSubnet
  LogicalResult traceMemrefSubnets();

  /// Phase 5: Assign latencies, fill the pure DDG, verify, return.
  LogicalResult constructDdg();

  /// Map a pipeline string to the canonical SMS pipeline name.
  static std::string canonicalizePipeline(TCoreType core, Operation *op);

  //--- State ---------------------------------------------------------------
  scf::ForOp pipelineLoop;
  Operation *scopeOp;
  const LatencyProvider &latencyProvider;
  DenseMap<Value, Operation *> counterClones;

  /// Authoritative Default WorkItem partition.  When non-null the DAG1
  /// (WorkItem-based) path is used; when null the legacy op-level path runs.
  const WorklistBuildResult *defaultBuild = nullptr;

  /// Optional MS cost model (see constructor).  Null → assignCosts() is a
  /// no-op and LatencyProvider placeholder latencies are kept.
  const costmodel::CostModel *costModel = nullptr;

  /// The result being built — moved out on success.
  MsDagBuildResult result;

  /// Intermediate classification state.
  SmallVector<Operation *> separators;

  /// Separator → { ops it reaches via shaped-result DFS }.
  DenseMap<Operation *, DenseSet<Operation *>> separatorDeps;

  /// Operations that are compute nodes (cube or vector groups).
  SmallVector<Operation *> computeOps;

  /// Operations that are transfer nodes.
  SmallVector<Operation *> transferOps;

  /// Vector-op groups: first op → all ops in the contiguous group.
  DenseMap<Operation *, SmallVector<Operation *, 4>> vectorGroups;

  /// Ops already assigned to a DDG node (to prevent double-counting).
  DenseSet<Operation *> assignedOps;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_MSDAGBUILDER_H
