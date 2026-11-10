//===- PreloadProtocolVerifier.cpp - MS preload protocol verification
//-----===//
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
// Read-only verification of an MS partition against the CV preload lowering
// protocol.  Four categories:
//
//   1. Coordinate & depth — structural stage/coordinate invariants
//   2. Ownership & ordering — single-owner, block order, DDG forward
//   3. Carrier & memory — cross-WorkItem values have derivable carriers
//   4. Lowering expressibility — P0 stage-level recurrence contract
//
// Diagnostics are emitted through the pipeline loop or the offending
// operation.  The verifier never mutates IR, creates scopes, or invokes
// WorklistBuilder.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/PreloadProtocolVerifier.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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

PreloadProtocolVerifier::PreloadProtocolVerifier(
    scf::ForOp loop, const MsDagBuildResult &dagResult,
    const MsPartitionResult &partition)
    : pipelineLoop(loop), dagResult(dagResult), partition(partition) {}

//===----------------------------------------------------------------------===//
// verifyCoordinatesAndDepth
//===----------------------------------------------------------------------===//

LogicalResult PreloadProtocolVerifier::verifyCoordinatesAndDepth() {
  int32_t depth = partition.pipelineDepth;

  // (1) pipelineDepth > 0
  if (depth <= 0)
    return pipelineLoop->emitWarning(
               "[preload-verifier] pipelineDepth must be positive, got ")
           << depth;

  // (2) stages.size() == pipelineDepth
  if (static_cast<int32_t>(partition.stages.size()) != depth)
    return pipelineLoop->emitWarning("[preload-verifier] stages.size()=")
           << partition.stages.size() << " != pipelineDepth=" << depth;

  // (3) ii > 0
  if (partition.ii <= 0)
    return pipelineLoop->emitWarning(
               "[preload-verifier] II must be positive, got ")
           << partition.ii;

  // (4) Stage records have unique raw stages and dense ordinals 0..N-1.
  DenseSet<int> seenRawStages;
  DenseSet<int> seenOrdinals;
  std::set<int32_t> seenCoordinates;

  for (const MaterializedStage &ms : partition.stages) {
    // Unique raw stage.
    if (!seenRawStages.insert(ms.rawMsStage).second)
      return pipelineLoop->emitWarning(
                 "[preload-verifier] duplicate raw stage ")
             << ms.rawMsStage;

    // Unique ordinal.
    if (!seenOrdinals.insert(ms.stageOrdinal).second)
      return pipelineLoop->emitWarning("[preload-verifier] duplicate ordinal ")
             << ms.stageOrdinal;

    // Ordinal in range [0, depth).
    if (ms.stageOrdinal < 0 || ms.stageOrdinal >= depth)
      return pipelineLoop->emitWarning("[preload-verifier] ordinal ")
             << ms.stageOrdinal << " out of range [0, " << depth << ")";

    // Coordinate matches formula.
    int32_t expected = depth - 1 - ms.stageOrdinal;
    if (ms.preloadCoordinate != expected)
      return pipelineLoop->emitWarning("[preload-verifier] stage ordinal=")
             << ms.stageOrdinal << " has coordinate=" << ms.preloadCoordinate
             << " but expected " << expected;

    // Coordinate in range.
    if (ms.preloadCoordinate < 0 || ms.preloadCoordinate >= depth)
      return pipelineLoop->emitWarning("[preload-verifier] coordinate ")
             << ms.preloadCoordinate << " out of range [0, " << depth << ")";

    seenCoordinates.insert(ms.preloadCoordinate);

    // Stage must have at least one WorkItem.
    if (ms.workItems.empty())
      return pipelineLoop->emitWarning("[preload-verifier] stage ordinal=")
             << ms.stageOrdinal << " has no WorkItems";
  }

  // (5) Coordinate set = [0, pipelineDepth).
  for (int32_t c = 0; c < depth; ++c) {
    if (seenCoordinates.find(c) == seenCoordinates.end())
      return pipelineLoop->emitWarning("[preload-verifier] coordinate ")
             << c << " missing from coordinate set";
  }

  // (6) Every WorkItem appears in exactly one stage.
  DenseSet<const WorkItem *> workItemsInStages;
  for (const MaterializedStage &ms : partition.stages) {
    for (const WorkItem *wi : ms.workItems) {
      if (!wi)
        return pipelineLoop->emitWarning(
                   "[preload-verifier] null WorkItem pointer in stage ordinal=")
               << ms.stageOrdinal;
      if (!workItemsInStages.insert(wi).second)
        return pipelineLoop->emitWarning(
            "[preload-verifier] WorkItem appears in multiple stages");
    }
  }

  // Every owned WorkItem must appear in some stage.
  for (const auto &item : partition.worklist) {
    if (!workItemsInStages.contains(item.get()))
      return pipelineLoop->emitWarning(
          "[preload-verifier] worklist WorkItem not found in any stage");
  }

  // (7) opToMsStage metadata consistency: every operation's stage metadata
  // must match some real stage.
  for (const auto &[op, stage] : partition.opToMsStage) {
    if (!seenRawStages.contains(stage))
      return op->emitWarning("[preload-verifier] operation has opToMsStage=")
             << stage << " which is not a valid stage";
  }

  LLVM_DEBUG(dbgs() << "[preload-verifier] Coordinate/depth checks passed\n");
  return success();
}

//===----------------------------------------------------------------------===//
// verifyOwnershipAndOrdering
//===----------------------------------------------------------------------===//

LogicalResult PreloadProtocolVerifier::verifyOwnershipAndOrdering() {
  // Build block-position table for loop body.
  DenseMap<Operation *, int> opToBlockPos;
  int pos = 0;
  for (Operation &op : *pipelineLoop.getBody())
    opToBlockPos[&op] = pos++;

  // (1) Every WorkItem op must be a direct child of the pipeline loop body
  //     and must appear in the DAG result.
  for (const auto &item : partition.worklist) {
    for (Operation *op : item->ops) {
      if (op->getParentOp() != pipelineLoop.getOperation())
        return op->emitWarning(
            "[preload-verifier] WorkItem op is not a direct child of "
            "the pipeline loop body");
      if (!dagResult.opToDdgIndex.contains(op))
        return op->emitWarning(
            "[preload-verifier] WorkItem op not found in DAG result");
    }
  }

  // (2) opToWorkItemMap consistency: forward check.
  //     Every (op, owners) in the map: each owner WorkItem must contain op.
  //     Multi-owner entries are legal (Default lazy-load / CUBE_OR_VECTOR
  //     dependency semantics).
  for (const auto &[op, owners] : partition.opToWorkItemMap) {
    if (owners.empty())
      return op->emitWarning(
          "[preload-verifier] opToWorkItemMap entry has empty owner list");
    for (const WorkItem *wi : owners) {
      if (!wi)
        return op->emitWarning(
            "[preload-verifier] null pointer in opToWorkItemMap owner list");
      if (!wi->ops.contains(op))
        return op->emitWarning(
            "[preload-verifier] opToWorkItemMap owner does not contain the op");
    }
  }

  // (3) Reverse check: every op in every WorkItem must be in the map.
  for (const auto &item : partition.worklist) {
    for (Operation *op : item->ops) {
      auto it = partition.opToWorkItemMap.find(op);
      if (it == partition.opToWorkItemMap.end())
        return op->emitWarning(
            "[preload-verifier] WorkItem op not in opToWorkItemMap");
      // Check that this WorkItem is listed as owner.
      bool found = false;
      for (const WorkItem *wi : it->second) {
        if (wi == item.get()) {
          found = true;
          break;
        }
      }
      if (!found)
        return op->emitWarning(
            "[preload-verifier] WorkItem op's map entry does not list "
            "this WorkItem as owner");
    }
  }

  // (4) Every DAG node op must have an owner.
  for (const MsNode &sn : dagResult.nodes) {
    for (Operation *op : sn.ops) {
      if (!partition.opToWorkItemMap.contains(op))
        return op->emitWarning(
            "[preload-verifier] DAG node op has no WorkItem owner");
    }
  }

  // (5) Each WorkItem's SetVector order must be nondecreasing in original
  //     block position.
  for (const auto &item : partition.worklist) {
    int lastPos = -1;
    for (Operation *op : item->ops) {
      auto it = opToBlockPos.find(op);
      if (it == opToBlockPos.end())
        return op->emitWarning(
            "[preload-verifier] op not found in loop body (block pos)");
      if (it->second < lastPos)
        return op->emitWarning(
            "[preload-verifier] WorkItem ops not in nondecreasing block order");
      lastPos = it->second;
    }
  }

  // (6) Vector group indivisibility: all ops in a VectorGroup node must be
  //     owned by the node's WorkItem, and the ops owned by it exclusively
  //     must share schedule metadata.
  for (const MsNode &sn : dagResult.nodes) {
    if (sn.kind != MsNodeKind::VectorGroup || sn.ops.size() <= 1)
      continue;
    if (sn.workItem) {
      // DAG1 path: every member op's owner list must contain the node's
      // WorkItem (multi-owner entries may list additional owners).
      for (Operation *op : sn.ops) {
        auto it = partition.opToWorkItemMap.find(op);
        if (it == partition.opToWorkItemMap.end())
          return op->emitWarning(
              "[preload-verifier] vector group op has no owner");
        if (!llvm::is_contained(it->second, sn.workItem))
          return op->emitWarning(
              "[preload-verifier] vector group op not owned by the node's "
              "WorkItem");
      }
    } else {
      // Legacy path: single-owner comparison.
      const WorkItem *groupOwner = nullptr;
      for (Operation *op : sn.ops) {
        auto it = partition.opToWorkItemMap.find(op);
        if (it == partition.opToWorkItemMap.end())
          return op->emitWarning(
              "[preload-verifier] vector group op has no owner");
        const WorkItem *owner = it->second[0];
        if (!groupOwner) {
          groupOwner = owner;
        } else if (owner != groupOwner) {
          return op->emitWarning(
              "[preload-verifier] vector group split across WorkItems");
        }
      }
    }
    // Schedule metadata is recorded per op, but an op can be owned by several
    // WorkItems (shared index computation, Default lazy-load), and
    // mapWorkItemStages stores a single value per op.  A shared op therefore
    // carries the stage of just one of its owners, which says nothing about
    // this group.  Group atomicity is a node-level property and is already
    // established above (every member op lists this node's WorkItem as an
    // owner), so compare metadata only across the ops this node owns
    // exclusively, where the recorded value is unambiguous.
    bool haveCommon = false;
    int commonCycle = 0;
    int commonStage = 0;
    for (Operation *op : sn.ops) {
      auto ownerIt = partition.opToWorkItemMap.find(op);
      if (ownerIt == partition.opToWorkItemMap.end() ||
          ownerIt->second.size() != 1)
        continue; // Shared op: single-valued metadata is ambiguous.
      auto cIt = partition.opToMsCycle.find(op);
      auto sIt = partition.opToMsStage.find(op);
      if (cIt == partition.opToMsCycle.end() ||
          sIt == partition.opToMsStage.end())
        continue; // Missing metadata caught elsewhere.
      if (!haveCommon) {
        haveCommon = true;
        commonCycle = cIt->second;
        commonStage = sIt->second;
        continue;
      }
      if (cIt->second != commonCycle || sIt->second != commonStage)
        return op->emitWarning(
            "[preload-verifier] vector group ops have inconsistent "
            "schedule metadata");
    }
  }

  // (7) DDG edge forward ordering: for every DDG edge whose endpoints
  //     resolve to different WorkItems, producer must precede consumer
  //     in worklist order.
  //
  // Build a WorkItem → worklist position map.
  DenseMap<const WorkItem *, int> wiToPos;
  for (size_t i = 0; i < partition.worklist.size(); ++i)
    wiToPos[partition.worklist[i].get()] = i;

  for (const sms::DDGEdge &edge : dagResult.ddg.getEdges()) {
    // Find representative ops for each endpoint.
    if (edge.src < 0 || edge.src >= static_cast<int>(dagResult.nodes.size()) ||
        edge.dst < 0 || edge.dst >= static_cast<int>(dagResult.nodes.size()))
      continue;

    const MsNode &srcNode = dagResult.nodes[edge.src];
    const MsNode &dstNode = dagResult.nodes[edge.dst];

    // Synthetic transfer nodes have no ops/WorkItem — their ordering is
    // covered by the transfer schedule metadata, not worklist positions.
    if (srcNode.isSynthetic || dstNode.isSynthetic)
      continue;

    // Prefer direct WorkItem linkage (DAG1 path); fall back to primaryOp
    // owner lookup (legacy path).
    const WorkItem *srcWI = srcNode.workItem;
    const WorkItem *dstWI = dstNode.workItem;
    if (!srcWI || !dstWI) {
      if (!srcNode.primaryOp || !dstNode.primaryOp)
        continue;
      auto srcMapIt = partition.opToWorkItemMap.find(srcNode.primaryOp);
      auto dstMapIt = partition.opToWorkItemMap.find(dstNode.primaryOp);
      if (srcMapIt == partition.opToWorkItemMap.end() ||
          dstMapIt == partition.opToWorkItemMap.end())
        continue; // Missing owner caught elsewhere.
      srcWI = srcMapIt->second[0];
      dstWI = dstMapIt->second[0];
    }

    // Same WorkItem → internal edge, skip.
    if (srcWI == dstWI)
      continue;

    auto srcPosIt = wiToPos.find(srcWI);
    auto dstPosIt = wiToPos.find(dstWI);
    if (srcPosIt == wiToPos.end() || dstPosIt == wiToPos.end())
      continue;

    if (srcPosIt->second > dstPosIt->second)
      return pipelineLoop->emitWarning("[preload-verifier] DDG edge from node ")
             << edge.src << " (worklist pos " << srcPosIt->second
             << ") to node " << edge.dst << " (worklist pos "
             << dstPosIt->second << ") violates forward ordering";
  }

  LLVM_DEBUG(dbgs() << "[preload-verifier] Ownership/ordering checks passed\n");
  return success();
}

//===----------------------------------------------------------------------===//
// verifyCarriersAndMemory
//===----------------------------------------------------------------------===//

LogicalResult PreloadProtocolVerifier::verifyCarriersAndMemory() {
  // Build a set of values produced by each WorkItem.
  DenseMap<const WorkItem *, DenseSet<Value>> producedByWI;
  for (const auto &item : partition.worklist) {
    auto &produced = producedByWI[item.get()];
    for (Operation *op : item->ops) {
      for (Value result : op->getResults())
        produced.insert(result);
    }
  }

  // Build a set of values that are tracked outputs (localOutputs,
  // yieldedOutputs, workspaceOutputs results).
  DenseSet<Value> trackedOutputs;
  for (const auto &item : partition.worklist) {
    for (const auto &lo : item->localOutputs)
      trackedOutputs.insert(lo.first);
    for (const auto &yo : item->yieldedOutputs)
      trackedOutputs.insert(yo.first);
    for (Operation *wsOp : item->workspaceOutputs)
      for (Value result : wsOp->getResults())
        trackedOutputs.insert(result);
  }

  // For each WorkItem, walk operands of each operation and check cross-WI
  // values.  A cross-WI value is one defined inside the loop by a different
  // WorkItem.  It must be tracked.
  for (const auto &item : partition.worklist) {
    for (Operation *op : item->ops) {
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp)
          continue; // Block argument — defined outside or by loop.
        if (!pipelineLoop->isAncestor(defOp))
          continue; // Defined outside the loop.

        // Check if the defining op is in a different WorkItem.
        bool sameItem = item->ops.contains(defOp);
        if (sameItem)
          continue;

        // Is the defining op nested under an op that is in this WorkItem?
        // Walk up to find the top-level parent within the loop.
        Operation *cur = defOp;
        while (cur && cur->getParentOp() != pipelineLoop.getOperation())
          cur = cur->getParentOp();
        if (cur && item->ops.contains(cur))
          continue; // Nested under a clone-root in this item.

        // The value crosses WorkItem boundaries.  It must be a tracked
        // output, a block argument of the loop, or resolvable by the
        // forwarding mechanism.
        if (trackedOutputs.contains(operand))
          continue;

        // Block arguments of the loop body are always available.
        if (isa<BlockArgument>(operand))
          continue;

        // Strict check: untracked cross-WorkItem value is an error.
        // But only flag it when carrier metadata is actually populated
        // (i.e. at least one WorkItem has any outputs set).
        bool hasAnyOutputMetadata = false;
        for (const auto &wi : partition.worklist) {
          if (!wi->localOutputs.empty() || !wi->yieldedOutputs.empty() ||
              !wi->workspaceOutputs.empty()) {
            hasAnyOutputMetadata = true;
            break;
          }
        }

        if (hasAnyOutputMetadata) {
          return op->emitWarning(
              "[preload-verifier] cross-WorkItem operand has no tracked "
              "carrier (localOutput, yieldedOutput, or workspaceOutput)");
        }
        // Pre-installation: output metadata not yet populated — defer
        // carrier check to Step ⑤ integration.
      }
    }
  }

  // Validate multibuffer depth when populated.
  for (const auto &item : partition.worklist) {
    if (item->multibuffer > 0 &&
        static_cast<int32_t>(item->multibuffer) < partition.pipelineDepth) {
      return pipelineLoop->emitWarning(
                 "[preload-verifier] WorkItem multibuffer=")
             << item->multibuffer
             << " is less than pipelineDepth=" << partition.pipelineDepth;
    }
  }

  // Cross-core tensor loop-carried dependency check.
  // Mirror the semantic rule from CVPipelineImpl::checkWorkItemDependencies:
  // a tensor iter_arg produced by one core and consumed by a different core
  // cannot be honored after pipelining.
  ArrayRef<BlockArgument> iterArgs = pipelineLoop.getRegionIterArgs();
  ValueRange yieldedVals = pipelineLoop.getYieldedValues();

  // Build producer core map from yieldedOutputs.
  DenseMap<unsigned, TCoreType> producerCoreByIterArg;
  for (const auto &item : partition.worklist) {
    for (const auto &yo : item->yieldedOutputs) {
      producerCoreByIterArg[yo.second] = item->core;
    }
  }

  for (unsigned argIdx = 0, e = iterArgs.size(); argIdx < e; ++argIdx) {
    BlockArgument iterArg = iterArgs[argIdx];
    if (!isa<TensorType>(iterArg.getType()))
      continue;
    auto prodIt = producerCoreByIterArg.find(argIdx);
    if (prodIt == producerCoreByIterArg.end())
      continue;
    TCoreType producerCore = prodIt->second;
    if (producerCore != TCoreType::CUBE && producerCore != TCoreType::VECTOR)
      continue;

    // Walk users of the iter_arg to find cross-core consumers.
    for (Operation *usr : iterArg.getUsers()) {
      if (isa<scf::YieldOp>(usr))
        continue;
      // Find the top-level parent within the loop body.
      Operation *top = usr;
      while (top && top->getParentOp() != pipelineLoop.getOperation())
        top = top->getParentOp();
      if (!top)
        continue;

      auto mapIt = partition.opToWorkItemMap.find(top);
      if (mapIt == partition.opToWorkItemMap.end())
        continue;
      TCoreType consumerCore = mapIt->second[0]->core;
      if (consumerCore != TCoreType::CUBE && consumerCore != TCoreType::VECTOR)
        continue;
      if (consumerCore == producerCore)
        continue;

      return pipelineLoop->emitWarning("[preload-verifier] cross-core tensor "
                                       "loop-carried dependency: "
                                       "iter_arg #")
             << argIdx << " produced on "
             << (producerCore == TCoreType::CUBE ? "CUBE" : "VECTOR")
             << " but consumed on "
             << (consumerCore == TCoreType::CUBE ? "CUBE" : "VECTOR");
    }
  }

  LLVM_DEBUG(dbgs() << "[preload-verifier] Carrier/memory checks passed\n");
  return success();
}

//===----------------------------------------------------------------------===//
// verifyLoweringExpressibility
//===----------------------------------------------------------------------===//

LogicalResult PreloadProtocolVerifier::verifyLoweringExpressibility() {
  // (1) All schedule metadata must be non-negative.
  for (const auto &[op, cycle] : partition.opToMsCycle) {
    if (cycle < 0)
      return op->emitWarning("[preload-verifier] negative schedule cycle ")
             << cycle;
  }
  for (const auto &[op, stage] : partition.opToMsStage) {
    if (stage < 0)
      return op->emitWarning("[preload-verifier] negative schedule stage ")
             << stage;
  }

  // (2) Every DAG node op must have schedule metadata.
  for (const MsNode &sn : dagResult.nodes) {
    for (Operation *op : sn.ops) {
      if (!partition.opToMsCycle.contains(op))
        return op->emitWarning(
            "[preload-verifier] DAG node op missing opToMsCycle entry");
      if (!partition.opToMsStage.contains(op))
        return op->emitWarning(
            "[preload-verifier] DAG node op missing opToMsStage entry");
    }
  }

  // (3) All final operations must be direct children of the loop body.
  for (const auto &item : partition.worklist) {
    for (Operation *op : item->ops) {
      if (op->getParentOp() != pipelineLoop.getOperation())
        return op->emitWarning(
            "[preload-verifier] WorkItem op is not a direct child of "
            "the pipeline loop body; nested scope materialization is "
            "not supported in P0");
    }
  }

  // (4) No standalone transfer WorkItem: every WorkItem must have a compute
  //     core (CUBE or VECTOR), not a transfer-only core.
  for (const auto &item : partition.worklist) {
    if (item->core != TCoreType::CUBE && item->core != TCoreType::VECTOR)
      return pipelineLoop->emitWarning(
                 "[preload-verifier] WorkItem has unsupported core type ")
             << static_cast<int>(item->core) << " (P0 requires CUBE or VECTOR)";
  }

  // (5) Verify no standalone transfer node escaped partition:
  //     every Transfer DDG node's ops must be owned by a compute WorkItem.
  for (const MsNode &sn : dagResult.nodes) {
    if (sn.kind != MsNodeKind::Transfer)
      continue;
    for (Operation *op : sn.ops) {
      auto mapIt = partition.opToWorkItemMap.find(op);
      if (mapIt == partition.opToWorkItemMap.end())
        return op->emitWarning(
            "[preload-verifier] transfer op has no WorkItem owner");
      const WorkItem *owner = mapIt->second[0];
      if (owner->core != TCoreType::CUBE && owner->core != TCoreType::VECTOR)
        return op->emitWarning(
            "[preload-verifier] transfer op owned by non-compute WorkItem");
    }
  }

  // (6) Reject empty partition.
  if (partition.worklist.empty())
    return pipelineLoop->emitWarning(
        "[preload-verifier] empty partition worklist");
  if (partition.stages.empty())
    return pipelineLoop->emitWarning("[preload-verifier] empty stages");

  LLVM_DEBUG(
      dbgs() << "[preload-verifier] Lowering expressibility checks passed\n");
  return success();
}

//===----------------------------------------------------------------------===//
// verify() — top-level orchestrator
//===----------------------------------------------------------------------===//

LogicalResult PreloadProtocolVerifier::verify() {
  LLVM_DEBUG(dbgs() << "[preload-verifier] Starting verification...\n");

  if (failed(verifyCoordinatesAndDepth()))
    return failure();

  if (failed(verifyOwnershipAndOrdering()))
    return failure();

  if (failed(verifyCarriersAndMemory()))
    return failure();

  if (failed(verifyLoweringExpressibility()))
    return failure();

  LLVM_DEBUG(dbgs() << "[preload-verifier] All checks passed\n");
  return success();
}

} // namespace hivm
} // namespace mlir
