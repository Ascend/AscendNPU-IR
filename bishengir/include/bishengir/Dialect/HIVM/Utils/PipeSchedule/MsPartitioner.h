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
// Checks an SMS schedule and maps it onto the authoritative Default WorkItems.
// WorkItems and their ownership are never changed. Stages are shifted to start
// at zero, preserving gaps; synthetic transfer schedules remain independent.

//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_MSPARTITIONER_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_MSPARTITIONER_H

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/SmsScheduler.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/MsDagBuilder.h"
#include "bishengir/Dialect/HIVM/Utils/WorkItem.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hivm {

//===----------------------------------------------------------------------===//
// MaterializedStage — one stage slot after shifting the minimum stage to zero
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

  /// All normalized stage slots, including empty slots, with preload
  /// coordinates.
  SmallVector<MaterializedStage> stages;

  /// Unique compute node and its original algorithm schedule.
  struct WorkItemSchedule {
    int ddgIndex = -1;
    int cycle = -1;
    int rawStage = -1;
  };
  DenseMap<WorkItem *, WorkItemSchedule> workItemSchedules;

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

  /// Map each WorkItem to exactly one compute node's original schedule.
  LogicalResult mapWorkItemSchedules();

  LogicalResult verifyWorkItemOrdering();

  /// Shift stages to start at zero without removing gaps, then assign
  /// coordinate = depth - 1 - normalized stage.
  LogicalResult assignCoordinates();

  /// Record synthetic DAG2 transfer node schedules as metadata.
  LogicalResult recordTransferSchedules();

  scf::ForOp pipelineLoop;
  const MsDagBuildResult &dagResult;
  const sms::ScheduleResult &schedule;

  MsPartitionResult result;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_MSPARTITIONER_H
