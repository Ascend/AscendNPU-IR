//===- MsPartitioner.h - SMS schedule → WorkItem coordinates ----*- C++
//-*-===//
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
// Materializes an SMS schedule onto the authoritative Default WorkItems.
//
// MsPartitioner is a schedule MATERIALIZER, not a partitioner: the WorkItem
// partition comes from the Default WorklistBuilder (via MsDagBuildResult's
// defaultWorklist) and is never created, split, merged, or re-owned here.
//
// Flow:
//   1. Validates the schedule against the DAG (all nodes scheduled, II > 0).
//   2. Maps each DAG1 compute node's SMS stage back to its owning WorkItem.
//   3. Dense-reindexes raw SMS stages and assigns preload coordinates.
//   4. Records synthetic DAG2 transfer schedules as independent metadata
//      (transfers are NOT inserted into WorkItem::ops).
//
// Invariants:
//   - result.worklist and result.opToWorkItemMap are the SAME objects as the
//     Default build (shared_ptr identity preserved).
//   - Each WorkItem gets exactly one preload coordinate; multiple WorkItems
//     may share a coordinate.
//   - Multi-owner ops (Default lazy-load semantics) are preserved.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_MSPARTITIONER_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_MSPARTITIONER_H

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/MsDagBuilder.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/SmsScheduler.h"
#include "bishengir/Dialect/HIVM/Utils/WorkItem.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hivm {

//===----------------------------------------------------------------------===//
// MaterializedStage — one logical pipeline stage after dense renumbering
//===----------------------------------------------------------------------===//

struct MaterializedStage {
  int rawMsStage = -1;
  int stageOrdinal = -1;
  int32_t preloadCoordinate = -1;

  SmallVector<WorkItem *> workItems;
};

//===----------------------------------------------------------------------===//
// MsTransferSchedule — timing metadata for a synthetic DAG2 transfer node
//===----------------------------------------------------------------------===//

struct MsTransferSchedule {
  /// DDG index of the synthetic transfer node.
  int ddgIndex = -1;
  int cycle = -1;
  int stage = -1;
  /// Pipeline name: "mte1", "mte2", "mte3", "fixpipe".
  std::string pipeline;
  /// Storage transition tag, e.g. "GM_to_L1".
  std::string storageTag;
};

//===----------------------------------------------------------------------===//
// MsPartitionResult — everything the downstream installer needs
//===----------------------------------------------------------------------===//

struct MsPartitionResult {
  /// The authoritative Default worklist (same shared_ptrs — identity
  /// preserved).
  SmallVector<std::shared_ptr<WorkItem>> worklist;

  /// The authoritative Default ownership map (may contain multi-owner ops).
  DenseMap<Operation *, SmallVector<WorkItem *>> opToWorkItemMap;

  /// Dense-renumbered stages with preload coordinates.
  SmallVector<MaterializedStage> stages;

  /// Per-op SMS schedule metadata (cycle/stage of the owning compute node).
  /// Sanity-check data only — not an authoritative schedule source.  An op
  /// owned by several WorkItems lands in several compute nodes, so these
  /// single-valued tables record just one of its owners (whichever
  /// mapWorkItemStages writes last).  The authoritative schedule is
  /// WorkItem-level: `stages` plus the WorkItem's preload coordinate.
  DenseMap<Operation *, int> opToMsCycle;
  DenseMap<Operation *, int> opToMsStage;

  /// Schedules for synthetic DAG2 transfer nodes (no WorkItem membership).
  SmallVector<MsTransferSchedule> transferSchedules;

  int32_t pipelineDepth = 0;
  int64_t ii = 0;
};

//===----------------------------------------------------------------------===//
// MsPartitioner — the schedule materializer
//===----------------------------------------------------------------------===//

class MsPartitioner {
public:
  MsPartitioner(scf::ForOp loop, const MsDagBuildResult &dagResult,
                 const sms::ScheduleResult &schedule);

  FailureOr<MsPartitionResult> partition();

private:
  LogicalResult validateSchedule();

  /// Map each DAG1 compute node's SMS stage/cycle back to its WorkItem and
  /// populate opToMsCycle/opToMsStage for all member ops.
  LogicalResult mapWorkItemStages();

  /// Dense-reindex raw stages and assign preload coordinates
  /// (coordinate = depth - 1 - ordinal).
  LogicalResult assignCoordinates();

  /// Record synthetic DAG2 transfer node schedules as metadata.
  LogicalResult recordTransferSchedules();

  scf::ForOp pipelineLoop;
  const MsDagBuildResult &dagResult;
  const sms::ScheduleResult &schedule;

  MsPartitionResult result;

  /// WorkItem → raw SMS stage of its compute node.
  DenseMap<WorkItem *, int> workItemToRawStage;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_MSPARTITIONER_H
