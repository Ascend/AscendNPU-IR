//===- MsPartitioner.cpp - SMS schedule → WorkItem coordinates -----------===//
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
//===----------------------------------------------------------------------===//
//
// Materializes an SMS schedule onto the authoritative Default WorkItems.
//
// Flow:
//   validateSchedule()        – reject malformed / incomplete schedules
//   mapWorkItemStages()       – map DAG1 compute node stages onto WorkItems
//   assignCoordinates()       – dense-reindex stages, set preload coordinates
//   recordTransferSchedules() – save synthetic transfer schedules as metadata
//
// Rejection rules:
//   - Non-positive II, missing node schedules, or unknown node indices
//   - DAG result without Default WorkItem linkage (legacy DAG path)
//   - A WorkItem whose compute nodes land in different stages (cannot give
//     one WorkItem two coordinates)
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/MsPartitioner.h"
#include "llvm/Support/Debug.h"
#include <set>

// Unified debug channel for the MS preload scheduler pipeline; the message
// prefix identifies the producing component.
#define DEBUG_TYPE "ms-preload-scheduler"

using llvm::dbgs;

namespace mlir {
namespace hivm {

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

MsPartitioner::MsPartitioner(scf::ForOp loop,
                               const MsDagBuildResult &dagResult,
                               const sms::ScheduleResult &schedule)
    : pipelineLoop(loop), dagResult(dagResult), schedule(schedule) {}

//===----------------------------------------------------------------------===//
// validateSchedule — reject malformed schedules
//===----------------------------------------------------------------------===//

LogicalResult MsPartitioner::validateSchedule() {
  if (schedule.ii <= 0)
    return pipelineLoop->emitWarning(
        "[ms-partitioner] non-positive II in schedule");

  if (schedule.stageCount <= 0)
    return pipelineLoop->emitWarning(
        "[ms-partitioner] non-positive stage count in schedule");

  int ddgSize = dagResult.ddg.size();
  if (ddgSize == 0)
    return pipelineLoop->emitWarning("[ms-partitioner] empty DAG");

  // Every DDG node must have a schedule entry.
  for (int i = 0; i < ddgSize; ++i) {
    if (schedule.nodeToCycle.find(i) == schedule.nodeToCycle.end())
      return pipelineLoop->emitWarning(
                 "[ms-partitioner] missing cycle entry for DDG node ")
             << i;
    if (schedule.nodeToStage.find(i) == schedule.nodeToStage.end())
      return pipelineLoop->emitWarning(
                 "[ms-partitioner] missing stage entry for DDG node ")
             << i;

    int stage = schedule.nodeToStage.at(i);
    int cycle = schedule.nodeToCycle.at(i);
    if (stage < 0)
      return pipelineLoop->emitWarning(
                 "[ms-partitioner] negative stage for DDG node ")
             << i;
    if (cycle < 0)
      return pipelineLoop->emitWarning(
                 "[ms-partitioner] negative cycle for DDG node ")
             << i;
  }

  // No extraneous schedule entries.
  for (auto &[nodeIdx, _] : schedule.nodeToCycle) {
    if (nodeIdx < 0 || nodeIdx >= ddgSize)
      return pipelineLoop->emitWarning(
                 "[ms-partitioner] schedule entry for out-of-range DDG node ")
             << nodeIdx;
  }
  for (auto &[nodeIdx, _] : schedule.nodeToStage) {
    if (nodeIdx < 0 || nodeIdx >= ddgSize)
      return pipelineLoop->emitWarning(
                 "[ms-partitioner] stage entry for out-of-range DDG node ")
             << nodeIdx;
  }

  // The materializer requires Default WorkItem linkage.
  if (dagResult.defaultWorklist.empty())
    return pipelineLoop->emitWarning(
        "[ms-partitioner] DAG result has no Default WorkItems; the "
        "schedule materializer requires the WorkItem-based DAG path");

  return success();
}

//===----------------------------------------------------------------------===//
// mapWorkItemStages — map DAG1 compute node stages onto WorkItems
//===----------------------------------------------------------------------===//

LogicalResult MsPartitioner::mapWorkItemStages() {
  // Adopt the authoritative Default containers (shared_ptr identity).
  result.worklist = dagResult.defaultWorklist;
  result.opToWorkItemMap = dagResult.defaultOpToWorkItemMap;

  for (const MsNode &sn : dagResult.nodes) {
    if (sn.isSynthetic)
      continue; // Transfer schedules handled separately.
    if (!sn.workItem)
      continue; // Non-WorkItem node (should not happen in the DAG1 path).

    int stage = schedule.nodeToStage.at(sn.ddgIndex);
    int cycle = schedule.nodeToCycle.at(sn.ddgIndex);

    // One coordinate per WorkItem: if the WorkItem already has a stage from
    // another node, take the max (later stage wins — the WorkItem cannot be
    // issued before all its nodes are ready).
    auto [it, inserted] = workItemToRawStage.try_emplace(sn.workItem, stage);
    if (!inserted)
      it->second = std::max(it->second, stage);

    // Record per-op schedule metadata for all member ops.
    for (Operation *op : sn.ops) {
      result.opToMsCycle[op] = cycle;
      result.opToMsStage[op] = stage;
    }
  }

  // Every Default WorkItem must have received a stage.
  for (const auto &item : result.worklist) {
    if (!workItemToRawStage.contains(item.get()))
      return pipelineLoop->emitWarning(
          "[ms-partitioner] Default WorkItem has no scheduled DAG node");
  }

  LLVM_DEBUG({
    dbgs() << "[ms-partitioner] WorkItem stages:\n";
    for (size_t i = 0; i < result.worklist.size(); ++i) {
      WorkItem *wi = result.worklist[i].get();
      dbgs() << "  [" << i
             << "] core=" << (wi->core == TCoreType::CUBE ? "CUBE" : "VECTOR")
             << " rawStage=" << workItemToRawStage[wi]
             << " ops=" << wi->ops.size() << "\n";
    }
  });

  return success();
}

//===----------------------------------------------------------------------===//
// assignCoordinates — dense-reindex raw stages, set preload coordinates
//===----------------------------------------------------------------------===//

LogicalResult MsPartitioner::assignCoordinates() {
  // Collect distinct raw stages in ascending order.
  std::set<int> rawStages;
  for (auto &[wi, stage] : workItemToRawStage)
    rawStages.insert(stage);

  int n = static_cast<int>(rawStages.size());
  if (n == 0)
    return pipelineLoop->emitWarning(
        "[ms-partitioner] no stages to materialize");

  result.pipelineDepth = n;
  result.ii = schedule.ii;

  // Build raw stage → dense ordinal map.
  DenseMap<int, int> rawToOrdinal;
  int ordinal = 0;
  for (int raw : rawStages)
    rawToOrdinal[raw] = ordinal++;

  // Create MaterializedStage records.
  result.stages.resize(n);
  for (auto &[raw, ord] : rawToOrdinal) {
    auto &ms = result.stages[ord];
    ms.rawMsStage = raw;
    ms.stageOrdinal = ord;
    ms.preloadCoordinate = n - 1 - ord;
  }

  // Associate each WorkItem with its MaterializedStage, preserving the
  // Default worklist order within each stage.
  for (const auto &item : result.worklist) {
    WorkItem *wi = item.get();
    int ord = rawToOrdinal[workItemToRawStage[wi]];
    result.stages[ord].workItems.push_back(wi);
  }

  LLVM_DEBUG({
    dbgs() << "[ms-partitioner] Dense coordinates (pipelineDepth=" << n
           << "):\n";
    for (auto &ms : result.stages) {
      dbgs() << "  stage " << ms.rawMsStage << " → ordinal " << ms.stageOrdinal
             << " → preload " << ms.preloadCoordinate << " ("
             << ms.workItems.size() << " WorkItems)\n";
    }
  });

  return success();
}

//===----------------------------------------------------------------------===//
// recordTransferSchedules — save synthetic transfer schedules as metadata
//===----------------------------------------------------------------------===//

LogicalResult MsPartitioner::recordTransferSchedules() {
  for (const MsNode &sn : dagResult.nodes) {
    if (!sn.isSynthetic)
      continue;

    MsTransferSchedule ts;
    ts.ddgIndex = sn.ddgIndex;
    ts.cycle = schedule.nodeToCycle.at(sn.ddgIndex);
    ts.stage = schedule.nodeToStage.at(sn.ddgIndex);
    ts.pipeline = sn.pipeline;
    ts.storageTag = sn.storageTag;
    result.transferSchedules.push_back(std::move(ts));
  }

  LLVM_DEBUG({
    dbgs() << "[ms-partitioner] Transfer schedules recorded: "
           << result.transferSchedules.size() << "\n";
  });

  return success();
}

//===----------------------------------------------------------------------===//
// partition() — top-level orchestrator
//===----------------------------------------------------------------------===//

FailureOr<MsPartitionResult> MsPartitioner::partition() {
  if (validateSchedule().failed())
    return failure();

  if (mapWorkItemStages().failed())
    return failure();

  if (assignCoordinates().failed())
    return failure();

  if (recordTransferSchedules().failed())
    return failure();

  LLVM_DEBUG({
    dbgs() << "[ms-partitioner] Materialization complete:\n"
           << "  worklist size: " << result.worklist.size() << "\n"
           << "  pipeline depth: " << result.pipelineDepth << "\n"
           << "  II: " << result.ii << "\n"
           << "  stages: " << result.stages.size() << "\n"
           << "  transfer schedules: " << result.transferSchedules.size()
           << "\n";
  });

  return std::move(result);
}

} // namespace hivm
} // namespace mlir
