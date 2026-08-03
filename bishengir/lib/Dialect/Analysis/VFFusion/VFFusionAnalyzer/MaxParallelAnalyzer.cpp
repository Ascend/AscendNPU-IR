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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
         isa<tensor::ExtractOp>(op) ||
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

/// Count non-terminator inner ops of a single linalgOp.
static int getComputeOpCount(Operation *op) {
  int count = 0;
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  iterateLinalgInnerOps(linalgOp, [&](Operation *) { count++; });
  return count;
}

// === Exec Unit Utilization ===
static float getExecUnitUtilization(
    int singleCnt, int doubleCnt,
    const DenseMap<std::pair<int64_t, int64_t>, unsigned> &innerOpCnts) {
  float avgMaxCycle = 0.0f;
  for (const auto &[key, opCnt] : innerOpCnts) {
    const auto [numerator, denominator] = key;
    if (denominator == 0) {
      continue;
    }
    const float cycle =
        1.0f * opCnt * (static_cast<float>(numerator) / denominator);
    avgMaxCycle = std::max(cycle, avgMaxCycle);
  }
  if (doubleCnt < singleCnt) {
    avgMaxCycle = std::max(avgMaxCycle, 1.0f * singleCnt);
  }
  int totalExecUnits = doubleCnt + singleCnt;
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
static std::pair<size_t, size_t> calculateIoCounts(size_t input, size_t output,
                                                   size_t all) {
  auto optIoNum = (input + output) - all;
  auto inputsNums = input - optIoNum;
  auto outputsNums = output - optIoNum;
  return {inputsNums, outputsNums};
}

static float ioScoreFromCounts(size_t inputsNums, size_t outputsNums) {
  if (inputsNums > outputsNums)
    return (outputsNums + inputsNums) * 0.5f;
  return static_cast<float>(outputsNums);
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

  // Only producer ExtractSlice/Extract Ops need to be fused to VF.
  int producerGroupId = static_cast<int>(opToGroupIndex[producerOp]);
  int consumerGroupId = static_cast<int>(opToGroupIndex[consumerOp]);
  auto &producerGroup = AllFusedGroupBlocks[producerGroupId];
  auto &consumerGroup = AllFusedGroupBlocks[consumerGroupId];
  // Similar to hasInvalidDependencyIfFused, if the number of ops in a group is
  // greater than 1, then the extract ops in that group must satisfy the
  // constraint rules.
  SmallVector<Operation *, 4> extractVec;
  if (producerGroup.size() == 1)
    for (Operation *op : producerGroup)
      if (isa<tensor::ExtractSliceOp>(op) || isa<tensor::ExtractOp>(op))
        extractVec.push_back(op);
  if (consumerGroup.size() == 1)
    for (Operation *op : consumerGroup)
      if (isa<tensor::ExtractSliceOp>(op) || isa<tensor::ExtractOp>(op))
        extractVec.push_back(op);
  for (auto *extractOp : extractVec)
    for (auto *user : extractOp->getUsers())
      if (!consumerGroup.count(user) && !producerGroup.count(user))
        return false;

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
    Operation *groupOp = opsInBlock[currentOpIndex];
    if (!opToGroupIndex.count(groupOp)) {
      auto groupId = AllFusedGroupBlocks.size();
      DenseSet<Operation *> newGroup;
      newGroup.insert(groupOp);
      AllFusedGroupBlocks[groupId] = newGroup;
      opToGroupIndex.insert({groupOp, groupId});
      groupMetrics[groupId] = getOpMetrics(groupOp);
    }
  }
  LDBG("Initialized " << AllFusedGroupBlocks.size()
                      << " groups with one op each");
}

const MaxParallelAnalyzer::CostMetrics &
MaxParallelAnalyzer::getOpMetrics(Operation *op) {
  if (!isa<linalg::LinalgOp>(op))
    return nonLinalgOpMetrics;
  auto it = linalgOpMetrics.find(op);
  if (it != linalgOpMetrics.end())
    return it->second;
  CostMetrics curOpMetrics;
  curOpMetrics.isValidLinalg = isValidLinalgOp(dyn_cast<linalg::LinalgOp>(op));
  curOpMetrics.computeScore = getComputeScores(op);
  curOpMetrics.computeOpCount = getComputeOpCount(op);
  curOpMetrics.parallelism = getParallelism(op);
  SmallVector<Operation *, 1> single{op};
  auto [s, d] = getExecUnitCounts(single);
  curOpMetrics.singleExuCnt = s;
  curOpMetrics.doubleExuCnt = d;
  curOpMetrics.innerOpCnts = getSameGroupOpCnts(single);
  DenseSet<Operation *> singleSet;
  singleSet.insert(op);
  collectValuesFromOps(singleSet, curOpMetrics.inputs, curOpMetrics.outputs,
                       curOpMetrics.allValues);
  linalgOpMetrics[op] = curOpMetrics;
  return linalgOpMetrics[op];
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
    // TODO Perhaps we can cache the computation state of operation, but we
    // cannot estimate the memory overhead as each operation consumes 208 bytes
    // of memory.
    const CostMetrics &cand = getOpMetrics(candidateOp);
    float producerComputeScores = cand.computeScore;
    auto [candIn, candOut] = calculateIoCounts(
        cand.inputs.size(), cand.outputs.size(), cand.allValues.size());
    float producerIoScores = ioScoreFromCounts(candIn, candOut);
    const CostMetrics &consumer = groupMetrics[consumerGroupId];
    float consumerComputeScores = consumer.computeScore;
    auto [consIn, consOut] =
        calculateIoCounts(consumer.inputs.size(), consumer.outputs.size(),
                          consumer.allValues.size());
    float consumerIoScores = ioScoreFromCounts(consIn, consOut);

    // IO Scores compute with uneliminated inputs/outputs
    float_t fusedIoScores = producerIoScores + consumerIoScores;
    float_t fusedComputeScores = producerComputeScores + consumerComputeScores;

    auto paralLift = parallelismSubModel(cand, consumer);
    auto execUtilLift = execUnitUtilizationSubModel(cand, consumer);

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

static size_t countUnique(const llvm::SmallDenseSet<Value> &lhs,
                          const llvm::SmallDenseSet<Value> &rhs) {
  size_t cnt = 0;
  for (Value v : rhs) {
    if (!lhs.contains(v))
      ++cnt;
  }
  return cnt;
}

bool MaxParallelAnalyzer::parallelismSubModel(
    const MaxParallelAnalyzer::CostMetrics &candidateCM,
    const MaxParallelAnalyzer::CostMetrics &consumerCM) const {
  size_t fusedIoCountNum = 0;
  if (candidateCM.isValidLinalg) {
    size_t mergedIn = consumerCM.inputs.size() +
                      countUnique(consumerCM.inputs, candidateCM.inputs);
    size_t mergedOut = consumerCM.outputs.size() +
                       countUnique(consumerCM.outputs, candidateCM.outputs);
    size_t mergedAll = consumerCM.allValues.size() +
                       countUnique(consumerCM.allValues, candidateCM.allValues);
    auto [inNum, outNum] = calculateIoCounts(mergedIn, mergedOut, mergedAll);
    fusedIoCountNum = inNum + outNum;
  }
  auto fusedComputeCount =
      candidateCM.computeOpCount + consumerCM.computeOpCount;
  int totalOps =
      static_cast<int>(fusedIoCountNum) + static_cast<int>(fusedComputeCount);
  float fusedLoopParallelism = (1.0f * kIssueQueueLen * 2 / totalOps) *
                               (1.0f * fusedComputeCount / totalOps);
  auto opMaxParallelism =
      std::max(candidateCM.parallelism, consumerCM.parallelism);
  return fusedLoopParallelism + kEpsilon > opMaxParallelism;
}

bool MaxParallelAnalyzer::execUnitUtilizationSubModel(
    const CostMetrics &candidateCM, const CostMetrics &consumerCM) const {
  float producerUtil =
      getExecUnitUtilization(candidateCM.singleExuCnt, candidateCM.doubleExuCnt,
                             candidateCM.innerOpCnts);
  float consumerUtil = getExecUnitUtilization(
      consumerCM.singleExuCnt, consumerCM.doubleExuCnt, consumerCM.innerOpCnts);
  LDBG("producerUtil=" << producerUtil << " consumerUtil=" << consumerUtil);
  float beforeUtil = std::min(consumerUtil, producerUtil);
  DenseMap<std::pair<int64_t, int64_t>, unsigned> temp = consumerCM.innerOpCnts;
  for (const auto &[k, v] : candidateCM.innerOpCnts)
    temp[k] += v;
  float mergedUtil = getExecUnitUtilization(
      candidateCM.singleExuCnt + consumerCM.singleExuCnt,
      candidateCM.doubleExuCnt + consumerCM.doubleExuCnt, temp);
  LDBG("mergedUtil=" << mergedUtil);
  return beforeUtil + kEpsilon < 1.0f && beforeUtil < mergedUtil + kEpsilon;
}

bool MaxParallelAnalyzer::isIOBoundGroup(int groupId) {
  if (AllFusedGroupBlocks[groupId].empty()) {
    return false;
  }
  const CostMetrics &cm = groupMetrics[groupId];
  float computeScores = cm.computeScore;
  auto [in, out] = calculateIoCounts(cm.inputs.size(), cm.outputs.size(),
                                     cm.allValues.size());
  float ioScores = ioScoreFromCounts(in, out);
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
  auto it = groupMetrics.find(producerGroupId);
  if (it != groupMetrics.end()) {
    groupMetrics[consumerGroupId] += groupMetrics[producerGroupId];
    groupMetrics.erase(it);
  }
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

          int foundGroupMaxIndex =
              dsu.getMaxIndexUnion(static_cast<int>(opToIndex[consumerOp]));
          if (consumerGroupId < 0 ||
              foundGroupMaxIndex < consumerGroupMinIndex) {
            consumerGroupId = foundGroupId;
            consumerGroupMinIndex = foundGroupMaxIndex;
          }
        }
      }
      if (consumerGroupId < 0)
        continue;
      if (AllFusedGroupBlocks[consumerGroupId].empty())
        continue;

      LDBG("IO-bound group " << producerGroupId << " -> consumer group "
                             << consumerGroupId);

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

      if (tryFuseGroups(producerIndex, consumerIndex, producerGroupId,
                        consumerGroupId)) {
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
