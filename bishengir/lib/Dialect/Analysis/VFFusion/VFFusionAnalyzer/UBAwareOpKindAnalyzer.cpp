//===- UBAwareOpKindAnalyzer.cpp - UB-aware fusion analyzer ---------------===//
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

#include "bishengir/Dialect/Analysis/VFFusion/VFFusionAnalyzer.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vf-fusion-ub-aware-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::analysis {

/// Compute the byte size of a shaped type, rounded up to the given alignment.
/// This matches PlanMemory's buffer sizing which aligns all UB allocations to
/// UB_ALIGN_SIZE (from NPUTargetSpec.td, typically 256 bits = 32 bytes).
int64_t getShapedBytes(Type type, int64_t alignBytes) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape() ||
      !shaped.getElementType().isIntOrFloat())
    return 0;
  int64_t raw = (shaped.getNumElements() *
                     shaped.getElementType().getIntOrFloatBitWidth() +
                 7) /
                8;
  if (alignBytes <= 0)
    return raw;
  return ((raw + alignBytes - 1) / alignBytes) * alignBytes;
}

/// partition block ops into two DSU group sets for the given roots.
void collectGroupOps(const SmallVector<Operation *> &opsInBlock,
                            VFUnionFind &dsu, int xRoot, int yRoot,
                            SmallPtrSet<Operation *, 16> &groupXOps,
                            SmallPtrSet<Operation *, 16> &groupYOps) {
  for (size_t i = 0; i < opsInBlock.size(); ++i) {
    int root = dsu.find(i);
    if (root == xRoot)
      groupXOps.insert(opsInBlock[i]);
    else if (root == yRoot)
      groupYOps.insert(opsInBlock[i]);
  }
}

/// Check whether PlanMemory could reuse the input's UB slot for this output.
/// PlanMemory's GenerateInplaceList allows at most 1 output to reuse a dead
/// input per VF call, provided the output is write-only (DPS op with
/// tensor.empty init), the input is read-only (not used as a DPS init), and
/// the input buffer is at least as large as the output.
// TODO: The liveness check below is conservative -- it rejects reuse if the
// input has ANY user outside the merged group. PlanMemory is more nuanced
// (checks point-wise liveness). This may over-estimate by ~1 buffer in
// rare cases but never under-estimates, so it is safe.
bool canReuseInputForOutput(
    Value output, Value input, const SmallPtrSet<Operation *, 32> &mergedOps,
    const SmallPtrSet<Operation *, 8> &hoistedOps, int64_t alignBytes) {
  auto *outputOp = output.getDefiningOp();
  if (!outputOp)
    return false;
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(outputOp);
  if (!dpsOp)
    return false;
  auto opResult = dyn_cast<OpResult>(output);
  if (!opResult)
    return false;
  if (opResult.getResultNumber() >=
      static_cast<unsigned>(dpsOp.getNumDpsInits()))
    return false;
  auto *tiedInit = dpsOp.getTiedOpOperand(opResult);
  if (!tiedInit)
    return false;
  auto *initOp = tiedInit->get().getDefiningOp();
  if (!initOp || !isa<tensor::EmptyOp>(initOp))
    return false;

  for (Operation *user : input.getUsers()) {
    // If the input has consumers outside the merged group it will remain live
    // across VF calls at the caller level, so PlanMemory cannot reuse its slot.
    if (!mergedOps.contains(user) || hoistedOps.contains(user))
      return false;
    if (auto userDps = dyn_cast<DestinationStyleOpInterface>(user)) {
      for (OpOperand &initOperand : userDps.getDpsInitsMutable()) {
        if (initOperand.get() == input)
          return false;
      }
    }
  }

  if (getShapedBytes(input.getType(), alignBytes) <
      getShapedBytes(output.getType(), alignBytes))
    return false;

  return true;
}

/// Collect external shaped inputs for a set of ops. An input is "external" if
/// its defining op is outside \p allOps (or is hoisted). tensor.empty results
/// are excluded because they are output placeholders, not true data inputs.
void collectExternalInputs(
    const SmallPtrSetImpl<Operation *> &groupOps,
    const SmallPtrSetImpl<Operation *> &allOps,
    const SmallPtrSetImpl<Operation *> &hoistedOps,
    SetVector<Value> &inputs) {
  for (Operation *op : groupOps) {
    if (hoistedOps.contains(op))
      continue;
      // Walk handles ops with nested regions (e.g. scf.for, linalg.generic).
    op->walk<WalkOrder::PreOrder>([&](Operation *inner) {
      for (Value operand : inner->getOperands()) {
        if (!isa<ShapedType>(operand.getType()))
          continue;
        if (auto *defOp = operand.getDefiningOp()) {
          if (allOps.contains(defOp) && !hoistedOps.contains(defOp))
            continue;
          if (isa<tensor::EmptyOp>(defOp))
            continue;
        }
        inputs.insert(operand);
      }
    });
  }
  for (Operation *op : groupOps) {
    // Hoisted non-empty ops (e.g. memref.alloc) become caller-side arguments;
    // their results are external inputs to the outlined function.
    if (!hoistedOps.contains(op) || isa<tensor::EmptyOp>(op))
      continue;
    for (Value result : op->getResults())
      inputs.insert(result);
  }
}

/// Estimate the peak caller-side UB footprint if Union-Find groups X and Y
/// were merged into a single outlined function.
///
/// Algorithm:
///   1. Collect all ops in both groups and identify hoisted ops
///      (isSafeToExcludeOps: memref.alloc, tensor.empty, etc.)
///   2. Identify external inputs: shaped operands whose defining op is outside
///      the merged group (or is a hoisted op). tensor.empty results are skipped
///      since they are output placeholders, not true inputs.
///   3. Identify external outputs: results of group ops used by ops outside
///      the group (these become return values of the outlined function).
///   4. Model PlanMemory's in-place reuse: at most 1 output can reuse a dead
///      input's UB slot per VF call (matching GenerateInplaceList behavior).
///   5. Sum aligned byte sizes of all external inputs + unreused outputs.
int64_t UBAwareOpKindAnalyzer::estimateMergedGroupBytes(int xIndex,
                                                        int yIndex) {
  const int xRoot = dsu.find(xIndex);
  const int yRoot = dsu.find(yIndex);

  SmallPtrSet<Operation *, 32> mergedOps;
  for (size_t i = 0; i < opsInBlock.size(); ++i) {
    int root = dsu.find(i);
    if (root == xRoot || root == yRoot)
      mergedOps.insert(opsInBlock[i]);
  }

  SmallPtrSet<Operation *, 8> hoistedOps;
  for (Operation *op : mergedOps) {
    if (isSafeToExcludeOps(op))
      hoistedOps.insert(op);
  }

  SetVector<Value> externalInputs;
  collectExternalInputs(mergedOps, mergedOps, hoistedOps, externalInputs);

  // external outputs: results used outside the group.
  SetVector<Value> externalOutputs;
  for (Operation *op : mergedOps) {
    if (hoistedOps.contains(op))
      continue;
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!mergedOps.contains(user) || hoistedOps.contains(user)) {
          externalOutputs.insert(result);
          break;
        }
      }
    }
  }

  int64_t totalBytes = 0;
  for (Value v : externalInputs)
    totalBytes += getShapedBytes(v.getType(), ubAlignBytes_);

  Value reusedOutput;
  for (Value output : externalOutputs) {
    if (!reusedOutput) {
      for (Value input : externalInputs) {
        if (canReuseInputForOutput(output, input, mergedOps, hoistedOps,
                                   ubAlignBytes_)) {
          reusedOutput = output;
          break;
        }
      }
    }
    if (output != reusedOutput)
      totalBytes += getShapedBytes(output.getType(), ubAlignBytes_);
  }

  LDBG("estimateMergedGroupBytes: "
       << externalInputs.size() << " inputs, " << externalOutputs.size()
       << " outputs (reuse=" << (reusedOutput ? 1 : 0) << ") = " << totalBytes
       << " bytes");
  return totalBytes;
}

/// Sum aligned bytes of results from \p ops where at least one user satisfies
/// \p userPred. Used for both boundary-byte and external-output-byte queries.
template <typename Pred>
int64_t sumResultBytesIf(
    const SmallPtrSetImpl<Operation *> &ops,
    const SmallPtrSetImpl<Operation *> &hoistedOps, int64_t alignBytes,
    Pred userPred) {
  int64_t bytes = 0;
  for (Operation *op : ops) {
    if (hoistedOps.contains(op))
      continue;
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (userPred(user)) {
          bytes += getShapedBytes(result.getType(), alignBytes);
          break;
        }
      }
    }
  }
  return bytes;
}

/// Estimate the peak caller-side UB pressure if groups X and Y are kept as
/// separate VFs that run sequentially.
///
/// When split, VF_X runs first, then VF_Y. The caller's UB must hold:
///   - Shared inputs (used by both X and Y): live across both VF calls
///   - Boundary tensors (X->Y results): produced by VF_X, persist until VF_Y
///   - Exclusive inputs of whichever VF is currently running
///   - Outputs of whichever VF is currently running
///
///   splitPeak = max(peakDuringX, peakDuringY)
///
/// When merged, boundary tensors become internal (never in caller UB), but ALL
/// exclusive inputs are live simultaneously. Split wins when exclusive inputs
/// of the idle group offset boundary cost; merge wins when boundary tensors
/// are large relative to exclusive inputs.
int64_t UBAwareOpKindAnalyzer::estimateSplitCostBytes(int xIndex, int yIndex) {
  const int xRoot = dsu.find(xIndex);
  const int yRoot = dsu.find(yIndex);

  SmallPtrSet<Operation *, 16> groupXOps, groupYOps;
  collectGroupOps(opsInBlock, dsu, xRoot, yRoot, groupXOps, groupYOps);

  SmallPtrSet<Operation *, 32> allOps;
  allOps.insert(groupXOps.begin(), groupXOps.end());
  allOps.insert(groupYOps.begin(), groupYOps.end());
  SmallPtrSet<Operation *, 8> hoistedOps;
  for (Operation *op : allOps) {
    if (isSafeToExcludeOps(op))
      hoistedOps.insert(op);
  }

  SetVector<Value> inputsX, inputsY;
  collectExternalInputs(groupXOps, allOps, hoistedOps, inputsX);
  collectExternalInputs(groupYOps, allOps, hoistedOps, inputsY);

  int64_t exclusiveXBytes = 0, exclusiveYBytes = 0, sharedBytes = 0;
  SetVector<Value> allInputs;
  for (Value v : inputsX)
    allInputs.insert(v);
  for (Value v : inputsY)
    allInputs.insert(v);
  for (Value v : allInputs) {
    int64_t bytes = getShapedBytes(v.getType(), ubAlignBytes_);
    bool inX = inputsX.contains(v);
    bool inY = inputsY.contains(v);
    if (inX && inY)
      sharedBytes += bytes;
    else if (inX)
      exclusiveXBytes += bytes;
    else
      exclusiveYBytes += bytes;
  }

  int64_t totalBoundary =
      sumResultBytesIf(groupXOps, hoistedOps, ubAlignBytes_,
                       [&](Operation *u) { return groupYOps.contains(u); }) +
      sumResultBytesIf(groupYOps, hoistedOps, ubAlignBytes_,
                       [&](Operation *u) { return groupXOps.contains(u); });
  int64_t outputsXBytes = sumResultBytesIf(
      groupXOps, hoistedOps, ubAlignBytes_,
      [&](Operation *u) { return !allOps.contains(u) || hoistedOps.contains(u); });
  int64_t outputsYBytes = sumResultBytesIf(
      groupYOps, hoistedOps, ubAlignBytes_,
      [&](Operation *u) { return !allOps.contains(u) || hoistedOps.contains(u); });

  int64_t peakDuringX =
      exclusiveXBytes + sharedBytes + totalBoundary + outputsXBytes;
  int64_t peakDuringY =
      exclusiveYBytes + sharedBytes + totalBoundary + outputsYBytes;
  int64_t splitPeak = std::max(peakDuringX, peakDuringY);

  LDBG("estimateSplitCostBytes:"
       << " exclusiveX=" << exclusiveXBytes << " exclusiveY=" << exclusiveYBytes
       << " shared=" << sharedBytes << " boundary=" << totalBoundary
       << " outputsX=" << outputsXBytes << " outputsY=" << outputsYBytes
       << " peakX=" << peakDuringX << " peakY=" << peakDuringY
       << " splitPeak=" << splitPeak);
  return splitPeak;
}

/// Decide whether merging Union-Find groups containing xIndex and yIndex is
/// allowed under the UB budget constraint.
///
/// Decision logic:
///   1. If no budget is set, always allow (feature off).
///   2. If the merged estimate fits within budget, allow.
///   3. Consumer lookahead (up to maxLookaheadDepth_ iterations): if the
///      estimate exceeds budget, iteratively expand the tentative group by
///      including downstream consumers of external outputs. Each round
///      internalizes outputs-to-consumers, potentially reducing the estimate
///      below budget. Allow if any depth fits.
///   4. Split-cost guard: if keeping X and Y separate would require strictly
///      more UB than merging them, allow the merge as the lesser evil.
///   5. Otherwise reject.
bool UBAwareOpKindAnalyzer::isFusibleImpl(const int xIndex, const int yIndex) {
  if (ubBudgetBytes_ <= 0)
    return true;

  int64_t mergedBytes = estimateMergedGroupBytes(xIndex, yIndex);
  LDBG("isFusibleImpl x=" << xIndex << " y=" << yIndex << " merged="
                          << mergedBytes << " budget=" << ubBudgetBytes_
                          << " xOp=" << opsInBlock[xIndex]->getName()
                          << " yOp=" << opsInBlock[yIndex]->getName());
  if (mergedBytes <= ubBudgetBytes_)
    return true;

  // --- Consumer lookahead (iterative, up to maxLookaheadDepth_) ---
  const int xRoot = dsu.find(xIndex);
  const int yRoot = dsu.find(yIndex);
  SmallPtrSet<Operation *, 32> expandedOps;
  for (size_t i = 0; i < opsInBlock.size(); ++i) {
    int root = dsu.find(i);
    if (root == xRoot || root == yRoot)
      expandedOps.insert(opsInBlock[i]);
  }
  SmallPtrSet<Operation *, 8> expandedHoisted;
  for (Operation *op : expandedOps) {
    if (isSafeToExcludeOps(op))
      expandedHoisted.insert(op);
  }

  for (int depth = 1; depth <= maxLookaheadDepth_; ++depth) {
    // Find consumers of external outputs that are outside the expanded group.
    SmallPtrSet<Operation *, 16> newConsumers;
    for (Operation *op : expandedOps) {
      if (expandedHoisted.contains(op))
        continue;
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (!expandedOps.contains(user) && opToIndex.contains(user))
            newConsumers.insert(user);
        }
      }
    }

    if (newConsumers.empty())
      break;

    // Expand the group with new consumers.
    expandedOps.insert(newConsumers.begin(), newConsumers.end());
    for (Operation *op : newConsumers) {
      if (isSafeToExcludeOps(op))
        expandedHoisted.insert(op);
    }

    SetVector<Value> expandedInputs;
    collectExternalInputs(expandedOps, expandedOps, expandedHoisted,
                          expandedInputs);

    SetVector<Value> expandedOutputs;
    for (Operation *op : expandedOps) {
      if (expandedHoisted.contains(op))
        continue;
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (!expandedOps.contains(user) || expandedHoisted.contains(user)) {
            expandedOutputs.insert(result);
            break;
          }
        }
      }
    }

    int64_t expandedBytes = 0;
    for (Value v : expandedInputs)
      expandedBytes += getShapedBytes(v.getType(), ubAlignBytes_);
    for (Value v : expandedOutputs)
      expandedBytes += getShapedBytes(v.getType(), ubAlignBytes_);

    LDBG("consumer-lookahead depth=" << depth << ": +" << newConsumers.size()
         << " consumers -> " << expandedInputs.size() << " inputs, "
         << expandedOutputs.size() << " outputs = " << expandedBytes
         << " bytes (was " << mergedBytes << ")");

    if (expandedBytes <= ubBudgetBytes_) {
      LDBG("lookahead allows merge at depth " << depth << ": expanded="
           << expandedBytes << " <= budget=" << ubBudgetBytes_);
      return true;
    }
  }

  // Split-cost guard: if keeping X and Y separate would require strictly MORE
  // UB at the caller than merging them (due to boundary buffers between VFs),
  // merging is the lesser evil even though the merged estimate exceeds budget.
  // Use strict inequality: when costs are equal, prefer splitting because each
  // individual VF is smaller and more likely to fit in downstream PlanMemory.
  int64_t splitCost = estimateSplitCostBytes(xIndex, yIndex);
  if (splitCost > mergedBytes) {
    LDBG("split-cost guard: splitPeak=" << splitCost
         << " > merged=" << mergedBytes
         << " -> splitting is worse, allowing merge");
    return true;
  }

  LDBG("rejecting merge: " << mergedBytes << " > " << ubBudgetBytes_
       << " (lookahead exhausted at depth " << maxLookaheadDepth_
       << ", splitCost=" << splitCost
       << " <= merged=" << mergedBytes << ")");
  return false;
}

/// Walk all ops in program order, attempting to fuse each op with its operand
/// producers via the Union-Find. Each candidate pair is checked by the base
/// class (outlineability, dependency validity, reshape constraints) then by
/// isFusibleImpl (UB budget gate + consumer lookahead + split-cost guard).
/// Successfully fused pairs share a Union-Find group and will be outlined
/// into the same function.
LogicalResult UBAwareOpKindAnalyzer::fuseImpl(Block &block) {
  LDBG("UBAwareOpKind fusing with UB budget " << ubBudgetBytes_ << " bytes");
  initialize(block);
  for (Operation &op : block.getOperations()) {
    LDBG("Try to fuse " << op);
    const int yIndex = opToIndex.at(&op);
    for (auto opr : op.getOperands()) {
      auto *defOp = opr.getDefiningOp();
      if (!defOp)
        continue;
      // tensor.empty is a shape placeholder, not a real data producer.
      // Skip it to avoid creating false fusion edges between independent
      // chains that share a CSE-merged tensor.empty.
      if (isa<tensor::EmptyOp>(defOp))
        continue;

      LDBG("check if defOp is outside of the block " << defOp->getName());
      if (!opToIndex.contains(defOp))
        continue;

      const int xIndex = opToIndex.at(defOp);
      if (!VFFusionAnalyzerBase::isFusible(xIndex, yIndex))
        continue;

      LDBG("fusing index of " << xIndex << " with " << yIndex);
      if (!VFFusionAnalyzerBase::fuseIndexWith(xIndex, yIndex))
        continue;

      LDBG("Fused " << op.getName() << " " << defOp->getName());
    }
  }

  return success();
}

} // namespace mlir::analysis
