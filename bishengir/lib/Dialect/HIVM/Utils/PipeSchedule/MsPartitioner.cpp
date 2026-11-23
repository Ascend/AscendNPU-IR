//===- MsPartitioner.cpp - SMS schedule to WorkItem coordinates -----------===//
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
// Checks node schedules and unique WorkItem linkage before assigning preload
// coordinates. Only the leading stage offset is removed; internal gaps remain.
// WorkItem contents and ownership are trusted and never changed.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/MsPartitioner.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <limits>

#define DEBUG_TYPE "ms-preload-scheduler"

using llvm::dbgs;

namespace mlir {
namespace hivm {

MsPartitioner::MsPartitioner(scf::ForOp loop, const MsDagBuildResult &dagResult,
                             const sms::ScheduleResult &schedule)
    : pipelineLoop(loop), dagResult(dagResult), schedule(schedule) {}

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

  for (int i = 0; i < ddgSize; ++i) {
    auto cycle = schedule.nodeToCycle.find(i);
    auto stage = schedule.nodeToStage.find(i);
    if (cycle == schedule.nodeToCycle.end() ||
        stage == schedule.nodeToStage.end())
      return pipelineLoop->emitWarning(
                 "[ms-partitioner] missing schedule entry for DDG node ")
             << i;
    if (cycle->second < 0 || stage->second < 0)
      return pipelineLoop->emitWarning(
                 "[ms-partitioner] negative cycle or stage for DDG node ")
             << i;
  }
  for (const auto &[nodeIdx, cycle] : schedule.nodeToCycle) {
    if (nodeIdx < 0 || nodeIdx >= ddgSize)
      return pipelineLoop->emitWarning(
          "[ms-partitioner] out-of-range cycle entry");
  }
  for (const auto &[nodeIdx, stage] : schedule.nodeToStage) {
    if (nodeIdx < 0 || nodeIdx >= ddgSize)
      return pipelineLoop->emitWarning(
          "[ms-partitioner] out-of-range stage entry");
  }
  if (dagResult.defaultWorklist.empty())
    return pipelineLoop->emitWarning(
        "[ms-partitioner] DAG result has no Default WorkItems");
  return success();
}

LogicalResult MsPartitioner::mapWorkItemSchedules() {
  result.worklist = dagResult.defaultWorklist;
  result.opToWorkItemMap = dagResult.defaultOpToWorkItemMap;

  DenseSet<WorkItem *> workItems;
  for (const auto &item : result.worklist) {
    if (!item || !workItems.insert(item.get()).second)
      return pipelineLoop->emitWarning(
          "[ms-partitioner] null or duplicate Default WorkItem");
  }

  DenseSet<int> nodeIndices;
  for (const MsNode &node : dagResult.nodes) {
    if (node.ddgIndex < 0 || node.ddgIndex >= dagResult.ddg.size() ||
        !nodeIndices.insert(node.ddgIndex).second)
      return pipelineLoop->emitWarning(
          "[ms-partitioner] invalid or duplicate DDG node index");
    if (node.isSynthetic)
      continue;
    if (!node.workItem || !workItems.contains(node.workItem))
      return pipelineLoop->emitWarning(
          "[ms-partitioner] compute node has no Default WorkItem owner");

    MsPartitionResult::WorkItemSchedule timing;
    timing.ddgIndex = node.ddgIndex;
    timing.cycle = schedule.nodeToCycle.at(node.ddgIndex);
    timing.rawStage = schedule.nodeToStage.at(node.ddgIndex);
    if (!result.workItemSchedules.try_emplace(node.workItem, timing).second)
      return pipelineLoop->emitWarning(
          "[ms-partitioner] WorkItem has multiple compute nodes");
  }
  if (nodeIndices.size() != static_cast<size_t>(dagResult.ddg.size()))
    return pipelineLoop->emitWarning(
        "[ms-partitioner] missing DDG node record");
  for (const auto &item : result.worklist) {
    if (!result.workItemSchedules.contains(item.get()))
      return pipelineLoop->emitWarning(
          "[ms-partitioner] Default WorkItem has no scheduled DAG node");
  }
  return success();
}

LogicalResult MsPartitioner::verifyWorkItemOrdering() {
  DenseMap<int, size_t> nodeToPosition;
  for (size_t i = 0; i < result.worklist.size(); ++i)
    nodeToPosition[result.workItemSchedules.lookup(result.worklist[i].get())
                       .ddgIndex] = i;

  for (const sms::DDGEdge &edge : dagResult.ddg.getEdges()) {
    auto src = nodeToPosition.find(edge.src);
    auto dst = nodeToPosition.find(edge.dst);
    // Synthetic transfer endpoints have no WorkItem position.
    if (src == nodeToPosition.end() || dst == nodeToPosition.end())
      continue;
    if (src->second > dst->second)
      return pipelineLoop->emitWarning(
          "[ms-partitioner] DDG edge violates WorkItem forward ordering");
  }
  return success();
}

LogicalResult MsPartitioner::assignCoordinates() {
  int minStage = std::numeric_limits<int>::max();
  int maxStage = 0;
  for (const auto &entry : result.workItemSchedules) {
    minStage = std::min(minStage, entry.second.rawStage);
    maxStage = std::max(maxStage, entry.second.rawStage);
  }
  int64_t depth = static_cast<int64_t>(maxStage) - minStage + 1;
  if (depth <= 0 || depth > std::numeric_limits<int32_t>::max())
    return pipelineLoop->emitWarning(
        "[ms-partitioner] pipeline depth is not representable");

  result.pipelineDepth = static_cast<int32_t>(depth);
  result.ii = schedule.ii;
  result.stages.resize(result.pipelineDepth);
  for (int32_t stage = 0; stage < result.pipelineDepth; ++stage) {
    auto &slot = result.stages[stage];
    slot.rawMsStage = minStage + stage;
    slot.stageOrdinal = stage;
    slot.preloadCoordinate = result.pipelineDepth - 1 - stage;
  }
  for (const auto &item : result.worklist) {
    int stage = result.workItemSchedules.lookup(item.get()).rawStage - minStage;
    result.stages[stage].workItems.push_back(item.get());
  }
  return success();
}

LogicalResult MsPartitioner::recordTransferSchedules() {
  for (const MsNode &node : dagResult.nodes) {
    if (!node.isSynthetic)
      continue;
    MsTransferSchedule timing;
    timing.ddgIndex = node.ddgIndex;
    timing.cycle = schedule.nodeToCycle.at(node.ddgIndex);
    timing.stage = schedule.nodeToStage.at(node.ddgIndex);
    timing.pipeline = node.pipeline;
    timing.storageTag = node.storageTag;
    result.transferSchedules.push_back(std::move(timing));
  }
  return success();
}

FailureOr<MsPartitionResult> MsPartitioner::partition() {
  if (failed(validateSchedule()) || failed(mapWorkItemSchedules()) ||
      failed(verifyWorkItemOrdering()) || failed(assignCoordinates()) ||
      failed(recordTransferSchedules()))
    return failure();

  LLVM_DEBUG(dbgs() << "[ms-partitioner] Materialization complete: "
                    << result.worklist.size() << " WorkItems, depth="
                    << result.pipelineDepth << ", II=" << result.ii << "\n");
  return std::move(result);
}

} // namespace hivm
} // namespace mlir
