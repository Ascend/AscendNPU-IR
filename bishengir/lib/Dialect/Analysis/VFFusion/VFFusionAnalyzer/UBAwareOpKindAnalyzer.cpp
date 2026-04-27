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
static int64_t getShapedBytes(Type type, int64_t alignBytes) {
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
static void collectGroupOps(const SmallVector<Operation *> &opsInBlock,
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
static bool canReuseInputForOutput(
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

  // external inputs (tensor.empty is an output init, not an input).
  SetVector<Value> externalInputs;
  for (Operation *op : mergedOps) {
    if (hoistedOps.contains(op))
      continue;
    op->walk<WalkOrder::PreOrder>([&](Operation *inner) {
      for (Value operand : inner->getOperands()) {
        if (!isa<ShapedType>(operand.getType()))
          continue;
        if (auto *defOp = operand.getDefiningOp()) {
          if (mergedOps.contains(defOp) && !hoistedOps.contains(defOp))
            continue;
          if (isa<tensor::EmptyOp>(defOp))
            continue;
        }
        externalInputs.insert(operand);
      }
    });
  }
  for (Operation *op : hoistedOps) {
    if (isa<tensor::EmptyOp>(op))
      continue;
    for (Value result : op->getResults())
      externalInputs.insert(result);
  }

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

/// Compute the byte cost of shared intermediates that would be materialized at
/// the caller level if groups X and Y were kept separate.
///
/// When a producer op in group X has results consumed by both group X and
/// group Y (a fan-out), splitting the groups forces the intermediate result to
/// become a caller-side buffer. This would increase UB pressure rather than
/// decrease it, so the merge should be allowed.
///
/// Returns the byte size of the first such shared intermediate, or 0 if none.
/// DPS ops whose init is tensor.empty are skipped because their output is
/// hoisted to the caller regardless of merge/split.
int64_t UBAwareOpKindAnalyzer::sharedProducerBytes(int xIndex, int yIndex) {
  const int xRoot = dsu.find(xIndex);
  const int yRoot = dsu.find(yIndex);

  SmallPtrSet<Operation *, 16> groupXOps, groupYOps;
  collectGroupOps(opsInBlock, dsu, xRoot, yRoot, groupXOps, groupYOps);

  const int64_t alignBytes = ubAlignBytes_;
  auto fanoutBytes =
      [alignBytes](const SmallPtrSet<Operation *, 16> &srcGroup,
                   const SmallPtrSet<Operation *, 16> &dstGroup) -> int64_t {
    for (Operation *op : srcGroup) {
      if (isSafeToExcludeOps(op))
        continue;
      // DPS op with tensor.empty init: output is hoisted regardless of merge.
      if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
        bool hasEmptyInit = llvm::any_of(dpsOp.getDpsInits(), [](Value init) {
          auto *initOp = init.getDefiningOp();
          return initOp && isa<tensor::EmptyOp>(initOp);
        });
        if (hasEmptyInit)
          continue;
      }
      for (Value result : op->getResults()) {
        int64_t bytes = getShapedBytes(result.getType(), alignBytes);
        if (bytes == 0)
          continue;
        bool usedByDst = false;
        bool usedByExternal = false;
        unsigned totalConsumers = 0;
        for (Operation *user : result.getUsers()) {
          if (dstGroup.contains(user))
            usedByDst = true;
          else if (!srcGroup.contains(user))
            usedByExternal = true;
          ++totalConsumers;
        }
        if (usedByDst && !usedByExternal && totalConsumers >= 2)
          return bytes;
      }
    }
    return 0;
  };

  int64_t bytes = fanoutBytes(groupXOps, groupYOps);
  if (bytes == 0)
    bytes = fanoutBytes(groupYOps, groupXOps);
  return bytes;
}

/// Decide whether merging Union-Find groups containing xIndex and yIndex is
/// allowed under the UB budget constraint.
///
/// Decision logic:
///   1. If no budget is set (ubBudgetBytes_ <= 0), always allow (feature off).
///   2. Estimate the combined caller-side UB footprint via
///      estimateMergedGroupBytes (external inputs + unreused outputs, aligned).
///   3. If the estimate fits within budget, allow the merge.
///   4. If it exceeds budget, check sharedProducerBytes:
///      - If shared intermediates exist, splitting would materialize them as
///        additional caller-side buffers, making the overflow worse. Allow
///        merge.
///      - If no shared intermediates, the groups are independent and splitting
///        them into separate VFs reduces peak UB. Reject the merge.
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

  if (int64_t savedBytes = sharedProducerBytes(xIndex, yIndex)) {
    LDBG("shared producer guard: overflow without merge="
         << (mergedBytes + savedBytes)
         << ", overflow with merge=" << mergedBytes
         << " (budget=" << ubBudgetBytes_ << "), better to merge");
    return true;
  }

  LDBG("rejecting merge: " << mergedBytes << " > " << ubBudgetBytes_);
  return false;
}

/// Walk all ops in program order, attempting to fuse each op with its operand
/// producers via the Union-Find. Each candidate pair is checked by the base
/// class (outlineability, dependency validity, reshape constraints) then by
/// isFusibleImpl (UB budget gate + shared-producer guard). Successfully fused
/// pairs share a Union-Find group and will be outlined into the same function.
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
