//===- MaxParallelAnalyzer.cpp - Implementation of
// MaxParallelAnalyzer--------===//
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

#include "bishengir/Dialect/Analysis/VFFusion/CostModelInfo/CostModelInfoUtils.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFFusionAnalyzer.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Visitors.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#define DEBUG_TYPE "vf-fusion-max-parallel-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::analysis {
static constexpr unsigned kIssueQueueLen = 64;
static constexpr float kEpsilon = std::numeric_limits<float>::epsilon();

// === Common Helpers ===
// Iterate over the non-terminator inner operations of a linalgOp,
// invoking the callback for each. Returns false if linalgOp is invalid.
template <typename CallbackT>
static bool iterateLinalgInnerOps(linalg::LinalgOp linalgOp,
                                  CallbackT &&callback) {
  if (!linalgOp || linalgOp->getNumRegions() == 0 ||
      linalgOp->getRegion(0).empty()) {
    return false;
  }
  for (Operation &innerOp : linalgOp->getRegion(0).front()) {
    if (!innerOp.hasTrait<OpTrait::IsTerminator>()) {
      callback(&innerOp);
    }
  }
  return true;
}

static bool isValidLinalgOp(linalg::LinalgOp linalgOp) {
  if (!linalgOp || linalgOp.getOperation()->getNumRegions() == 0 ||
      linalgOp.getOperation()->getRegion(0).empty()) {
    return false;
  }
  return true;
}

bool isLinalgReductionOp(linalg::LinalgOp linalgOp) {
  auto iteratorTypes = linalgOp.getIteratorTypesArray();
  return llvm::any_of(iteratorTypes, [](auto attr) {
    return linalg::isReductionIterator(attr);
  });
}

static bool isReductionOp(Operation *innerOp, Operation *outterOp) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(outterOp);
  if (!linalgOp)
    return false;

  int64_t numInputs = linalgOp.getNumDpsInputs();

  Block *body = &linalgOp->getRegion(0).front();

  for (Value operand : innerOp->getOperands()) {
    auto blockArg = dyn_cast<BlockArgument>(operand);
    if (!blockArg || blockArg.getOwner() != body)
      continue;
    // if an op take the inits arg as its operand, it is a reduction op
    if (blockArg.getArgNumber() >= numInputs) {
      return true;
    }
  }
  return false;
}

static bool isInFusionWhiteList(Operation *op) {
  return isReshapeOp(op) || isa<tensor::ExtractSliceOp>(op) ||
         isValidLinalgOp(dyn_cast<linalg::LinalgOp>(op));
}

// === Cost Model Helpers ===
// Extract CostInfo for all inner ops of a linalgOp.
static void extractInstrSet(linalg::LinalgOp linalgOp,
                            SmallVector<CostInfo> &costInfoSet) {
  iterateLinalgInnerOps(linalgOp, [&](Operation *innerOp) {
    auto isReduction = isReductionOp(innerOp, linalgOp.getOperation());
    costInfoSet.push_back(
        CostModelInfoUtils::getOpCostInfo(innerOp, isReduction));
  });
}

// Extract CostInfo for ops
template <typename RangeT>
static void extractInstrSet(const RangeT &ops,
                            SmallVector<CostInfo> &costInfoSet) {
  for (auto *op : ops) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (linalgOp) {
      extractInstrSet(linalgOp, costInfoSet);
    }
  }
}

static std::pair<int, int> getExuCnt(const SmallVector<CostInfo> &costInfoSet) {
  int singleExuCnt = 0;
  int doubleExuCnt = 0;
  for (const auto &costInfo : costInfoSet) {
    if (costInfo.execUnit == 2) {
      doubleExuCnt++;
    } else {
      singleExuCnt++;
    }
  }
  return {singleExuCnt, doubleExuCnt};
}

template <typename RangeT>
static llvm::DenseMap<std::pair<int64_t, int64_t>, unsigned>
getSameGroupOpCnts(const RangeT &ops) {
  llvm::DenseMap<std::pair<int64_t, int64_t>, unsigned> innerOpCnts;
  SmallVector<CostInfo> costInfoSet;
  extractInstrSet(ops, costInfoSet);
  for (const auto &costInfo : costInfoSet) {
    auto key = std::make_pair(costInfo.execInterval, costInfo.execUnit);
    innerOpCnts[key]++;
  }
  return innerOpCnts;
}

template <typename RangeT>
static std::pair<int, int> getExecUnitCounts(const RangeT &ops) {
  SmallVector<CostInfo> costInfoSet;
  extractInstrSet(ops, costInfoSet);
  return getExuCnt(costInfoSet);
}

static float getComputeScores(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!isValidLinalgOp(linalgOp)) {
    return 0.0f;
  }
  float_t computeScores = 0.0f;
  for (Operation &innerOp : linalgOp.getOperation()->getRegion(0).front()) {
    if (!innerOp.hasTrait<OpTrait::IsTerminator>()) {
      auto isReduction = isReductionOp(&innerOp, op);
      auto costInfo = CostModelInfoUtils::getOpCostInfo(&innerOp, isReduction);
      computeScores += 1.0 * costInfo.execInterval / costInfo.execUnit;
    }
  }
  return computeScores;
}

static float_t getFusedOpsComputeScores(const DenseSet<Operation *> &fusedOps) {
  auto computeScores = 0.0f;
  for (auto op : fusedOps) {
    computeScores += getComputeScores(op);
  }
  return computeScores;
}

/// Count non-terminator inner ops of a single linalgOp.
static int getComputeOpCount(Operation *op) {
  int count = 0;
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  iterateLinalgInnerOps(linalgOp, [&](Operation *) { count++; });
  return count;
}

/// Count non-terminator inner ops across a set of operations.
static int getComputeOpCount(const DenseSet<Operation *> &fusedOps) {
  int count = 0;
  for (auto *op : fusedOps) {
    count += getComputeOpCount(op);
  }
  return count;
}
// === Exec Unit Utilization ===
static float getExecUnitUtilization(const SmallVector<Operation *> &ops) {
  if (ops.empty()) {
    return 0.0f;
  }
  const auto execUnitCounts = getExecUnitCounts(ops);
  float avgMaxCycle = 0.0f;
  const auto &groupInstMap = getSameGroupOpCnts(ops);
  for (const auto &[key, opCnt] : groupInstMap) {
    const auto [numerator, denominator] = key;
    if (denominator == 0) {
      continue;
    }
    const float cycle =
        1.0f * opCnt * (static_cast<float>(numerator) / denominator);
    avgMaxCycle = std::max(cycle, avgMaxCycle);
  }
  if (execUnitCounts.second < execUnitCounts.first) {
    avgMaxCycle = std::max(avgMaxCycle, 1.0f * execUnitCounts.first);
  }
  int totalExecUnits = execUnitCounts.second + execUnitCounts.first;
  float maxAllowedUnits = avgMaxCycle * 2.0f;
  float utilization = totalExecUnits / maxAllowedUnits;
  return std::min(utilization, 1.0f);
}

// === IO Score Helpers ===
static void collectValuesFromOps(const DenseSet<Operation *> &ops,
                                 llvm::SmallDenseSet<Value> &uniqueInputs,
                                 llvm::SmallDenseSet<Value> &uniqueOutputs,
                                 llvm::SmallDenseSet<Value> &uniqueAll) {
  for (auto op : ops) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      continue;

    for (auto value : linalgOp.getDpsInputs()) {
      if (!value.getDefiningOp<arith::ConstantOp>()) {
        uniqueInputs.insert(value);
        uniqueAll.insert(value);
      }
    }

    for (auto value : linalgOp.getOperation()->getResults()) {
      if (!value.getDefiningOp<arith::ConstantOp>()) {
        uniqueOutputs.insert(value);
        uniqueAll.insert(value);
      }
    }
  }
}

// IO Scores compute with eliminated inputs/outputs
static std::pair<size_t, size_t>
calculateIoCounts(const llvm::SmallDenseSet<Value> &uniqueInputs,
                  const llvm::SmallDenseSet<Value> &uniqueOutputs,
                  const llvm::SmallDenseSet<Value> &uniqueAll) {
  auto optIoNum =
      (uniqueInputs.size() + uniqueOutputs.size()) - uniqueAll.size();
  auto inputsNums = uniqueInputs.size() - optIoNum;
  auto outputsNums = uniqueOutputs.size() - optIoNum;
  return {inputsNums, outputsNums};
}

static float_t getFusedOpsIoScores(const DenseSet<Operation *> &fusedOps) {
  llvm::SmallDenseSet<Value> uniqueInputs;
  llvm::SmallDenseSet<Value> uniqueOutputs;
  llvm::SmallDenseSet<Value> uniqueAll;
  collectValuesFromOps(fusedOps, uniqueInputs, uniqueOutputs, uniqueAll);
  auto [inputsNums, outputsNums] =
      calculateIoCounts(uniqueInputs, uniqueOutputs, uniqueAll);
  float ioScores = 0.0f;
  if (inputsNums > outputsNums) {
    ioScores = (outputsNums + inputsNums) * 0.5f;
  } else {
    ioScores = outputsNums;
  }
  return ioScores;
}

static llvm::SmallVector<size_t>
getFusedIoCount(const DenseSet<Operation *> &producerGroup,
                const DenseSet<Operation *> &consumerGroup) {
  llvm::SmallVector<size_t> ioCounts = {0, 0};

  bool hasValidLinalg = llvm::any_of(producerGroup, [](Operation *op) {
    return isValidLinalgOp(dyn_cast<linalg::LinalgOp>(op));
  });
  if (!hasValidLinalg)
    return ioCounts;

  llvm::SmallDenseSet<Value> uniqueInputs;
  llvm::SmallDenseSet<Value> uniqueOutputs;
  llvm::SmallDenseSet<Value> uniqueAll;
  collectValuesFromOps(producerGroup, uniqueInputs, uniqueOutputs, uniqueAll);
  collectValuesFromOps(consumerGroup, uniqueInputs, uniqueOutputs, uniqueAll);
  auto [inputsNums, outputsNums] =
      calculateIoCounts(uniqueInputs, uniqueOutputs, uniqueAll);
  ioCounts[0] = inputsNums;
  ioCounts[1] = outputsNums;
  return ioCounts;
}

// === Parallelism ===
static float getParallelism(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!isValidLinalgOp(linalgOp)) {
    return 0.0f;
  }
  float_t parallelism = 0.0f;
  for (Operation &innerOp : linalgOp.getOperation()->getRegion(0).front()) {
    if (!innerOp.hasTrait<OpTrait::IsTerminator>()) {
      auto isReduction = isReductionOp(&innerOp, op);
      auto costInfo = CostModelInfoUtils::getOpCostInfo(&innerOp, isReduction);
      if (costInfo.execInterval == 0) {
        continue;
      }
      parallelism = std::max(1.0f * costInfo.execUnit * costInfo.execLatency /
                                 costInfo.execInterval,
                             parallelism);
    }
  }
  return parallelism;
}

static float getParallelism(const DenseSet<Operation *> &fusedOps) {
  float_t parallelism = 0.f;
  float_t currentOpParallelism = 0.f;
  for (auto op : fusedOps) {
    currentOpParallelism = getParallelism(op);
    parallelism = std::max(currentOpParallelism, parallelism);
  }
  return parallelism;
}

// === MaxParallelAnalyzer Member Functions ===
bool MaxParallelAnalyzer::hasReductionToConsumer(const int producerIndex,
                                                 const int consumerIndex) {
  if (!opToGroupIndex.count(opsInBlock[producerIndex]) ||
      !opToGroupIndex.count(opsInBlock[consumerIndex])) {
    return false;
  }

  auto &producerGroupNodes =
      AllFusedGroupBlocks[opToGroupIndex[opsInBlock[producerIndex]]];
  auto &consumerGroupNodes =
      AllFusedGroupBlocks[opToGroupIndex[opsInBlock[consumerIndex]]];

  for (auto *node : producerGroupNodes) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(node);
    if (!isValidLinalgOp(linalgOp) || !isLinalgReductionOp(linalgOp))
      continue;
    for (auto *user : linalgOp->getUsers()) {
      if (consumerGroupNodes.contains(user)) {
        return true;
      }
    }
  }
  return false;
}

bool MaxParallelAnalyzer::areFusibleOps(const int producerIndex,
                                        const int consumerIndex) {
  auto producerOp = opsInBlock[producerIndex];
  auto consumerOp = opsInBlock[consumerIndex];

  // If a SyncBlockSetOp is found, prohibit fusion to avoid data races
  for (auto it = producerOp->getNextNode(); it != nullptr && it != consumerOp;
       it = it->getNextNode()) {
    if (isa<hivm::SyncBlockSetOp>(it)) {
      return false;
    }
  }

  // Only producer ExtractSliceOps need to be fused to VF.
  int producerGroupId = static_cast<int>(opToGroupIndex[producerOp]);
  int consumerGroupId = static_cast<int>(opToGroupIndex[consumerOp]);
  auto &producerGroup = AllFusedGroupBlocks[producerGroupId];
  auto &consumerGroup = AllFusedGroupBlocks[consumerGroupId];
  DenseSet<Operation *> fusedGroupsSet;
  for (Operation* op : producerGroup)
    fusedGroupsSet.insert(op);
  for (Operation* op : consumerGroup)
    fusedGroupsSet.insert(op);
  for (Operation* op : fusedGroupsSet) {
    if (isa<tensor::ExtractSliceOp>(op)) {
      for (Operation* user : op->getUsers()) {
        if (!fusedGroupsSet.contains(user)) {
          return false;
        }
      }
    }
  }

  if (!isInFusionWhiteList(producerOp) || !isInFusionWhiteList(consumerOp))
    return false;

  auto producerLinalgOp = dyn_cast<linalg::LinalgOp>(producerOp);

  if (producerLinalgOp != nullptr && isLinalgReductionOp(producerLinalgOp))
    return false;

  // Prevent reduction op having a consumer in the fused group
  if (hasReductionToConsumer(producerIndex, consumerIndex))
    return false;

  return true;
}

std::vector<OpOperand *>
MaxParallelAnalyzer::getSortedConsumerOperands(Operation *producerOp) {
  std::vector<OpOperand *> validUses;
  for (OpOperand &use : producerOp->getUses()) {
    if (opToIndex.contains(use.getOwner())) {
      validUses.push_back(&use);
    }
  }

  // Sort the uses in ascending called order.
  std::sort(validUses.begin(), validUses.end(),
            [&](OpOperand *x, OpOperand *y) {
              int xIndex = opToIndex.at(x->getOwner());
              int pxMax = dsu.getMaxIndexUnion(xIndex);
              int yIndex = opToIndex.at(y->getOwner());
              int pyMax = dsu.getMaxIndexUnion(yIndex);
              return pxMax < pyMax;
            });

  return validUses;
}

void MaxParallelAnalyzer::initializeImpl(Block &block) {
  for (auto &op : block) {
    if (!opToIndex.contains(&op))
      continue;
    if (!isInFusionWhiteList(&op))
      continue;
    auto currentOpIndex = opToIndex.at(&op);
    if (!opToGroupIndex.count(opsInBlock[currentOpIndex])) {
      auto groupId = AllFusedGroupBlocks.size();
      DenseSet<Operation *> newGroup;
      newGroup.insert(opsInBlock[currentOpIndex]);
      AllFusedGroupBlocks[groupId] = newGroup;
      opToGroupIndex.insert({opsInBlock[currentOpIndex], groupId});
    }
  }
  LDBG("Initialized " << AllFusedGroupBlocks.size()
                      << " groups with one op each");
}

bool MaxParallelAnalyzer::canFuseGroups(int producerGroupId,
                                        int consumerGroupId,
                                        int producerIndex) {

  auto &consumerGroup = AllFusedGroupBlocks[consumerGroupId];

  // Calculate scores for individual groups
  Operation *const candidateOp = opsInBlock[producerIndex];
  if (isa<hfusion::CastOp>(candidateOp)) {
    LDBG("candidateOp is Cast");
    return true;
  }
  if (stage == 1) {
    DenseSet<Operation *> candidateOpGroup;
    candidateOpGroup.insert(candidateOp);
    float_t producerComputeScores = getFusedOpsComputeScores(candidateOpGroup);
    float_t producerIoScores = getFusedOpsIoScores(candidateOpGroup);
    float consumerComputeScores = getFusedOpsComputeScores(consumerGroup);
    float consumerIoScores = getFusedOpsIoScores(consumerGroup);

    // IO Scores compute with uneliminated inputs/outputs
    float_t fusedIoScores = producerIoScores + consumerIoScores;
    float_t fusedComputeScores = producerComputeScores + consumerComputeScores;

    auto paralLift = parallelismSubModel(candidateOpGroup, consumerGroup);
    auto execUtilLift =
      execUnitUtilizationSubModel(candidateOpGroup, consumerGroup);

    LDBG("candidate op producer: compute=" << producerComputeScores
                                         << " io=" << producerIoScores);
    LDBG("consumer(" << consumerGroup.size() << "): compute="
                   << consumerComputeScores << " io=" << consumerIoScores);

    if (producerIoScores + kEpsilon > producerComputeScores &&
        consumerIoScores + kEpsilon > consumerComputeScores) {
            LDBG("Both IO Bound -> Fuse");
            return true;
    }

    if (!(producerIoScores < producerComputeScores &&
        consumerIoScores < consumerComputeScores) &&
      fusedIoScores + kEpsilon > fusedComputeScores) {
    LDBG("Fused IO >= Compute -> Fuse");
    return true;
  }
  LDBG("execUtilLift=" << execUtilLift << "paralLift=" << paralLift);
  return (execUtilLift || paralLift);
  } else {
    return true;
  }
}

bool MaxParallelAnalyzer::parallelismSubModel(
    DenseSet<Operation *> &producerGroup,
    DenseSet<Operation *> &consumerGroup) const {
  llvm::SmallDenseSet<Value> uniqueInputs;
  llvm::SmallDenseSet<Value> uniqueOutputs;
  llvm::SmallDenseSet<Value> uniqueAll;
  auto fusedIoCount = getFusedIoCount(producerGroup, consumerGroup);
  auto fusedIoCountNum = fusedIoCount[0] + fusedIoCount[1];
  auto fusedComputeCount =
      getComputeOpCount(producerGroup) + getComputeOpCount(consumerGroup);
  int totalOps =
      static_cast<int>(fusedIoCountNum) + static_cast<int>(fusedComputeCount);
  float fusedLoopParallelism = (1.0f * kIssueQueueLen * 2 / totalOps) *
                               (1.0f * fusedComputeCount / totalOps);
  auto opMaxParallelism =
      std::max(getParallelism(producerGroup), getParallelism(consumerGroup));
  return fusedLoopParallelism + kEpsilon > opMaxParallelism;
}

bool MaxParallelAnalyzer::execUnitUtilizationSubModel(
    DenseSet<Operation *> &producerGroup,
    DenseSet<Operation *> &consumerGroup) const {
  llvm::SmallVector<Operation *> producerGroupOps(producerGroup.begin(),
                                                  producerGroup.end());
  llvm::SmallVector<Operation *> consumerGroupOps(consumerGroup.begin(),
                                                  consumerGroup.end());
  llvm::SmallVector<Operation *> fusableOps;
  fusableOps.append(consumerGroup.begin(), consumerGroup.end());
  fusableOps.append(producerGroup.begin(), producerGroup.end());
  float producerUtil = getExecUnitUtilization(producerGroupOps);
  float consumerUtil = getExecUnitUtilization(consumerGroupOps);
  LDBG("producerUtil=" << producerUtil << " consumerUtil=" << consumerUtil);
  float beforeUtil = std::min(consumerUtil, producerUtil);
  float mergedUtil = getExecUnitUtilization(fusableOps);
  LDBG("mergedUtil=" << mergedUtil);
  return beforeUtil + kEpsilon < 1.0f && beforeUtil < mergedUtil + kEpsilon;
}

bool MaxParallelAnalyzer::isIOBoundGroup(int groupId) {
  if (AllFusedGroupBlocks[groupId].empty()) {
    return false;
  }
  float computeScores = getFusedOpsComputeScores(AllFusedGroupBlocks[groupId]);
  float ioScores = getFusedOpsIoScores(AllFusedGroupBlocks[groupId]);
  return ioScores + kEpsilon > computeScores;
}

bool MaxParallelAnalyzer::mergeGroups(const int producerGroupId,
                                      const int consumerGroupId) {
  auto &producerGroup = AllFusedGroupBlocks[producerGroupId];
  auto &consumerGroup = AllFusedGroupBlocks[consumerGroupId];
  for (auto *op : producerGroup) {
    consumerGroup.insert(op);
    opToGroupIndex[op] = consumerGroupId;
  }
  producerGroup.clear();
  LDBG("merged group " << producerGroupId << " into group " << consumerGroupId);
  return true;
}

// Attempt to fuse two groups by: checking fusibility, checking index
// fusion, and merging. Returns true on success.
bool MaxParallelAnalyzer::tryFuseGroups(int producerIndex, int consumerIndex,
                                        int producerGroupId,
                                        int consumerGroupId) {
  if (!VFFusionAnalyzerBase::isFusible(producerIndex, consumerIndex)) {
    LDBG("isFusible failed");
    return false;
  }
  if (!VFFusionAnalyzerBase::fuseIndexWith(producerIndex, consumerIndex)) {
    LDBG("fuseIndexWith failed");
    return false;
  }
  if (!mergeGroups(producerGroupId, consumerGroupId)) {
    LDBG("mergeGroups failed");
    return false;
  }
  return true;
}

bool MaxParallelAnalyzer::isFusibleImpl(const int producerIndex,
                                        const int consumerIndex) {
  LDBG("checking group-based fusible");

  auto *producerOp = opsInBlock[producerIndex];
  auto *consumerOp = opsInBlock[consumerIndex];

  if (!opToGroupIndex.count(producerOp) || !opToGroupIndex.count(consumerOp)) {
    LDBG("producer or consumer has no group, cannot fuse");
    return false;
  }

  int producerGroupId = static_cast<int>(opToGroupIndex[producerOp]);
  int consumerGroupId = static_cast<int>(opToGroupIndex[consumerOp]);

  auto &producerGroup = AllFusedGroupBlocks[producerGroupId];
  auto &consumerGroup = AllFusedGroupBlocks[consumerGroupId];

  // Check if ALL ops in producer group have zero compute
  bool allZeroCompute = llvm::all_of(
      producerGroup, [](Operation *op) { return getComputeOpCount(op) == 0; });
  if (allZeroCompute && !producerGroup.empty()) {
    LDBG("producer group all zero compute op, fuse");
    return true;
  }

  // Check if producerGroup or consumerGroup are ALL reshape ops
  bool producerAllReshape = llvm::all_of(producerGroup, isReshapeOp);
  bool consumerAllReshape = llvm::all_of(consumerGroup, isReshapeOp);
  if ((producerAllReshape || consumerAllReshape) && !producerGroup.empty() &&
      !consumerGroup.empty()) {
    LDBG("producer or consumer group all reshape Op, fuse");
    return true;
  }

  if (canFuseGroups(producerGroupId, consumerGroupId, producerIndex)) {
    LDBG("groups can be fused based on cost model");
    return true;
  }

  return false;
}

bool MaxParallelAnalyzer::fuseProducerConsumerImpl(Block &block) {
  bool hasFused = false;
  // Iterate through block in reverse order (from last op to first)
  for (auto it = block.rbegin(); it != block.rend(); ++it) {
    Operation &producerOp = *it;

    if (!opToIndex.contains(&producerOp))
      continue;

    const int producerIndex = opToIndex.at(&producerOp);
    if (!opToGroupIndex.contains(&producerOp))
      continue;

    std::vector<OpOperand *> validUses = getSortedConsumerOperands(&producerOp);
    for (OpOperand *opOperandPtr : validUses) {
      Operation *consumerOp = opOperandPtr->getOwner();
      if (!opToGroupIndex.contains(consumerOp))
        continue;

      const int consumerIndex = opToIndex.at(consumerOp);
      int producerGroupId = static_cast<int>(opToGroupIndex[&producerOp]);
      int consumerGroupId = static_cast<int>(opToGroupIndex[consumerOp]);

      LDBG("Check: " << producerOp << " " << producerOp << " (group "
                     << producerGroupId << ", "
                     << AllFusedGroupBlocks[producerGroupId].size()
                     << " ops) -> " << consumerOp << " " << *consumerOp
                     << " (group " << consumerGroupId << ", "
                     << AllFusedGroupBlocks[consumerGroupId].size() << " ops)");

      if (producerGroupId == consumerGroupId) {
        LDBG("same group, skip");
        continue;
      }

      if (!areFusibleOps(producerIndex, consumerIndex)) {
        LDBG("areFusibleOps returned false");
        continue;
      }

      if (tryFuseGroups(producerIndex, consumerIndex, producerGroupId,
                        consumerGroupId)) {
        hasFused = true;
        LDBG("Successfully fused group " << producerGroupId << " with group "
                                         << consumerGroupId);
      }
    }
  }
  return hasFused;
}


bool MaxParallelAnalyzer::fuseIOBoundGroupsWithNearestConsumer() {
  bool hasFused = false;
  bool madeProgress = true;

  while (madeProgress) {
    madeProgress = false;

    // Find all IO-bound groups
    std::vector<int> ioBoundGroupIds;
    for (auto &[id, ops] : AllFusedGroupBlocks) {
      if (!ops.empty() && isIOBoundGroup(id)) {
        ioBoundGroupIds.push_back(id);
      }
    }

    for (int producerGroupId : ioBoundGroupIds) {
      LDBG("IO-bound group " << producerGroupId);
      auto &producerGroup = AllFusedGroupBlocks[producerGroupId];
      if (producerGroup.empty())
        continue;

      // Collect all consumer groups from every op in the producer group,
      // then find the truly nearest consumer group.
      int consumerGroupId = -1;
      int consumerGroupMinIndex = -1;
      for (auto *producerOp : producerGroup) {
        if (!opToIndex.contains(producerOp))
          continue;

        std::vector<OpOperand *> validUses =
            getSortedConsumerOperands(producerOp);

        for (auto *opOperandPtr : validUses) {
          auto *consumerOp = opOperandPtr->getOwner();
          if (!opToGroupIndex.contains(consumerOp))
            continue;
          int foundGroupId = static_cast<int>(opToGroupIndex[consumerOp]);
          if (foundGroupId == producerGroupId)
            continue;
          if (AllFusedGroupBlocks[foundGroupId].empty())
            continue;

          int foundGroupMaxIndex = dsu.getMaxIndexUnion(static_cast<int>(opToIndex[consumerOp]));
          if (consumerGroupId < 0 || foundGroupMaxIndex < consumerGroupMinIndex) {
            consumerGroupId = foundGroupId;
            consumerGroupMinIndex = foundGroupMaxIndex;
          }
        }
      }
      if (consumerGroupId < 0)
        continue;
      if (AllFusedGroupBlocks[consumerGroupId].empty())
        continue;

      LDBG("IO-bound group " << producerGroupId
                             << " -> consumer group " << consumerGroupId);

      auto &consumerGroup = AllFusedGroupBlocks[consumerGroupId];

      // Use representative ops from each group for areFusibleOps/tryFuseGroups
      auto *producerOp = *producerGroup.begin();
      auto *consumerOp = *consumerGroup.begin();
      if (!opToIndex.contains(producerOp) || !opToIndex.contains(consumerOp))
        continue;

      int producerIndex = static_cast<int>(opToIndex[producerOp]);
      int consumerIndex = static_cast<int>(opToIndex[consumerOp]);
      
      if (!areFusibleOps(producerIndex, consumerIndex)) {
          LDBG("areFusibleOps returned false");
          continue;
      }

      if (tryFuseGroups(producerIndex, consumerIndex,
                        producerGroupId, consumerGroupId)) {
        hasFused = true;
        madeProgress = true;
        LDBG("IO-bound group " << producerGroupId
                               << " fused with consumer group "
                               << consumerGroupId);
      }
    }
  }
  return hasFused;
}

void MaxParallelAnalyzer::printValidGroupCount() {
  int64_t count = 0;
  std::vector<int> validGroupIds;

  LDBG("=============================================");
  LDBG("        All Fusion Groups Info");
  LDBG("=============================================");

  for (auto &entry : AllFusedGroupBlocks) {
    int groupId = entry.first;
    auto &opSet = entry.second;

    if (opSet.empty())
      continue;

    validGroupIds.push_back(groupId);
    count++;
  }
  LDBG("=============================================");
  LDBG("Total valid groups: " << count);
  LDBG("All Valid Group IDs: ");
  for (int id : validGroupIds) {
    LDBG("  - " << id);
  }
  LDBG("=============================================\n");
  return;
}

LogicalResult MaxParallelAnalyzer::fuseImpl(Block &block) {
  LDBG("MaxParallel Fusing" << block << "\n");
  initialize(block);
  stage = 1;
  // Perform producer-consumer fusion until no more fusions occur.
  while (fuseProducerConsumerImpl(block)) {
    // Keep looping
  }
  stage = 2;
  if (fuseIOBoundGroupsWithNearestConsumer())
    LDBG("=== Phase 2: find IO bound group to be merged ===");
  printValidGroupCount();
  return success();
}
} // namespace mlir::analysis
