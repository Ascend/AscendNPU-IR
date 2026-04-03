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
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Visitors.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#define DEBUG_TYPE "vf-fusion-max-parallel-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::analysis {
static constexpr unsigned issueQueueLens = 64;
// === Cost Model Helpers Begin ===
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

bool MaxParallelAnalyzer::hasReductionToConsumer(const int producerIndex,
                                                 const int consumerIndex) {
  if (opToGroupIndex.find(opsInBlock[producerIndex]) == opToGroupIndex.end() ||
      opToGroupIndex.find(opsInBlock[consumerIndex]) == opToGroupIndex.end()) {
    return false;
  }

  auto producerGroupNodes =
      AllFusedBlocks[opToGroupIndex[opsInBlock[producerIndex]]];
  auto consumerGroupNodes =
      AllFusedBlocks[opToGroupIndex[opsInBlock[consumerIndex]]];

  for (auto xNode : producerGroupNodes) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(xNode);
    if (!isValidLinalgOp(linalgOp)) {
      continue;
    }
    if (!isLinalgReductionOp(linalgOp)) {
      continue;
    }
    for (auto user : linalgOp->getUsers()) {
      if (consumerGroupNodes.contains(user)) {
        return true;
      }
    }
  }
  return false;
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

static bool isCubeScopeOp(Operation *op) {
  auto scopeOp = dyn_cast<scope::ScopeOp>(op);
  if (!scopeOp)
    return false;

  auto attr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  if (!attr)
    return false;

  return attr.getTcoretype() == mlir::hivm::TCoreType::CUBE;
}

static bool isInFusionWhiteList(Operation *op) {
  return isReshapeOp(op) || isa<tensor::ExtractSliceOp>(op) ||
         isValidLinalgOp(dyn_cast<linalg::LinalgOp>(op));
}

bool MaxParallelAnalyzer::areFusibleOps(const int producerIndex, 
                                        const int consumerIndex, 
                                        OpOperand *fusedOperand) {
  auto producerOp =  opsInBlock[producerIndex];
  auto consumerOp =  opsInBlock[consumerIndex];
  
  if (hfusion::isMatmulOps(producerOp) || hfusion::isMatmulOps(consumerOp) ||
      isCubeScopeOp(producerOp) || isCubeScopeOp(consumerOp)) 
    return false;


  if (!isInFusionWhiteList(producerOp) || !isInFusionWhiteList(consumerOp))
    return false;

  auto producerLinalgOp = dyn_cast<linalg::LinalgOp>(producerOp);
  auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(consumerOp);

  if (producerLinalgOp != nullptr && isLinalgReductionOp(producerLinalgOp))
    return false;
  
  // Prevent reduction op having a consumer in the fused group                                                 
  if (hasReductionToConsumer(producerIndex, consumerIndex))
    return false;

  if (consumerLinalgOp != nullptr && !consumerLinalgOp.isDpsInput(fusedOperand))
    return false;

  return true;
}

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

static int getComputeOpCount(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!isValidLinalgOp(linalgOp)) {
    return 0;
  }
  int count = 0;
  for (Operation &innerOp : linalgOp.getOperation()->getRegion(0).front()) {
    if (!innerOp.hasTrait<OpTrait::IsTerminator>()) {
      count++;
    }
  }
  return count;
}

static int getComputeOpCount(const DenseSet<Operation *> &fusedOps) {
  int count = 0;
  int opCount = 0;
  for (auto op : fusedOps) {
    opCount = getComputeOpCount(op);
    count += opCount;
  }
  return count;
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

void getSameGroupOpCnts(
    Operation *op,
    llvm::DenseMap<std::pair<int64_t, int64_t>, uint> &linalgInnnerOpCnts) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!isValidLinalgOp(linalgOp)) {
    return;
  }
  for (Operation &innerOp : linalgOp.getOperation()->getRegion(0).front()) {
    if (!innerOp.hasTrait<OpTrait::IsTerminator>()) {
      auto isReduction = isReductionOp(&innerOp, op);
      auto costInfo = CostModelInfoUtils::getOpCostInfo(&innerOp, isReduction);
      // Key is execInterval and exeUnit
      auto key = std::make_pair(costInfo.execInterval, costInfo.execUnit);
      if (linalgInnnerOpCnts.count(key)) {
        linalgInnnerOpCnts[key]++;
      } else {
        linalgInnnerOpCnts[key] = 1;
      }
    }
  }
}

llvm::DenseMap<std::pair<int64_t, int64_t>, uint>
getSameGroupOpCnts(const SmallVector<Operation *> &ops) {
  llvm::DenseMap<std::pair<int64_t, int64_t>, uint> linalgInnnerOpCnts;
  for (auto op : ops) {
    getSameGroupOpCnts(op, linalgInnnerOpCnts);
  }
  return linalgInnnerOpCnts;
}

static std::pair<int, int> getExecUnitCounts(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!isValidLinalgOp(linalgOp)) {
    return {0, 0};
  }
  int singleExecCnt = 0;
  int doubleExecCnt = 0;
  for (Operation &innerOp : linalgOp.getOperation()->getRegion(0).front()) {
    if (!innerOp.hasTrait<OpTrait::IsTerminator>()) {
      auto isReduction = isReductionOp(&innerOp, op);
      auto costInfo = CostModelInfoUtils::getOpCostInfo(&innerOp, isReduction);
      if (costInfo.execUnit == 1) {
        singleExecCnt++;
      } else {
        doubleExecCnt++;
      }
    }
  }
  return {singleExecCnt, doubleExecCnt};
}

static std::pair<int, int>
getExecUnitCounts(const SmallVector<Operation *> &ops) {
  std::pair<int, int> totalCnts = {0, 0};
  for (auto op : ops) {
    auto cnts = getExecUnitCounts(op);
    totalCnts = {totalCnts.first + cnts.first, totalCnts.second + cnts.second};
  }
  return totalCnts;
}

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
    return 1.0f * (execUnitCounts.second + execUnitCounts.first) /
           (avgMaxCycle * 2);
  }
  if (execUnitCounts.second + execUnitCounts.first > avgMaxCycle * 2) {
    return 1.0f;
  } else {
    return 1.0f * (execUnitCounts.second + execUnitCounts.first) /
           (avgMaxCycle * 2);
  }
}

static float getExecUnitUtilization(Operation *op) {
  if (op == nullptr) {
    return 0.0f;
  }
  llvm::SmallVector<Operation *> ops;
  ops.push_back(op);
  return getExecUnitUtilization(ops);
}

static float_t getIoScores(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!isValidLinalgOp(linalgOp)) {
    return 0.0f;
  }
  llvm::SmallDenseSet<Value> uniqueInputs;
  for (auto value : linalgOp.getDpsInputs()) {
    if (!value.getDefiningOp<arith::ConstantOp>()) {
      uniqueInputs.insert(value);
    }
  }

  llvm::SmallDenseSet<Value> uniqueOutputs;
  for (auto value : linalgOp.getOperation()->getResults()) {
    if (!value.getDefiningOp<arith::ConstantOp>()) {
      uniqueOutputs.insert(value);
    }
  }

  float ioScores = 0.0f;
  if (uniqueInputs.size() > uniqueOutputs.size()) {
    ioScores = (uniqueOutputs.size() + uniqueInputs.size()) * 0.5f;
  } else {
    ioScores = uniqueOutputs.size();
  }
  return ioScores;
}

// IO Scores compute with eliminated inputs/outputs
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
getFusedIoCount(Operation *candidateOp, const DenseSet<Operation *> &fusedOps) {
  llvm::SmallVector<size_t> ioCounts = {0, 0};
  auto candidateLinalgOp = dyn_cast<linalg::LinalgOp>(candidateOp);
  if (!candidateLinalgOp ||
      candidateLinalgOp.getOperation()->getNumRegions() == 0 ||
      candidateLinalgOp.getOperation()->getRegion(0).empty()) {
    return ioCounts;
  }
  llvm::SmallDenseSet<Value> uniqueInputs;
  llvm::SmallDenseSet<Value> uniqueOutputs;
  llvm::SmallDenseSet<Value> uniqueAll;

  DenseSet<Operation *> singleOpSet;
  singleOpSet.insert(candidateOp);
  collectValuesFromOps(singleOpSet, uniqueInputs, uniqueOutputs, uniqueAll);
  collectValuesFromOps(fusedOps, uniqueInputs, uniqueOutputs, uniqueAll);
  auto [inputsNums, outputsNums] =
      calculateIoCounts(uniqueInputs, uniqueOutputs, uniqueAll);
  ioCounts[0] = inputsNums;
  ioCounts[1] = outputsNums;
  return ioCounts;
}

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

static bool execUnitUtilizationCostModel(Operation *candidateOp,
                                         const DenseSet<Operation *> &fusedOps) {
  SmallVector<Operation *> ops;
  SmallVector<Operation *> fusableOps;
  if (candidateOp) {
    ops.push_back(candidateOp);
  }
  for (mlir::Operation *fusedOp : fusedOps) {
    if (fusedOp) {
      ops.push_back(fusedOp);
      fusableOps.push_back(fusedOp);
    }
  }
  float candidateOpExecUnitUtil = getExecUnitUtilization(candidateOp);
  float fusedOpsExecUnitUtil = getExecUnitUtilization(fusableOps);
  float beforeExecUnitUtil =
      std::min(fusedOpsExecUnitUtil, candidateOpExecUnitUtil);
  float mergeExecUnitUtil = getExecUnitUtilization(ops);
  return beforeExecUnitUtil + std::numeric_limits<float>::epsilon() < 1.0f &&
         beforeExecUnitUtil <
             mergeExecUnitUtil + std::numeric_limits<float>::epsilon();
}

static bool parallelismCostModel(Operation *candidateOp,
                                 const DenseSet<Operation *> &fusedOps) {
  auto fusedIoCount = getFusedIoCount(candidateOp, fusedOps);
  auto fusedIoCountNum = fusedIoCount[0] + fusedIoCount[1];
  auto fusedComputeCount =
      getComputeOpCount(candidateOp) + getComputeOpCount(fusedOps);
  auto fusedLoopParallelism =
      (1.0f * issueQueueLens * 2 / (fusedIoCountNum + fusedComputeCount)) *
      (1.0f * fusedComputeCount / (fusedIoCountNum + fusedComputeCount));
  auto opMaxParallelism =
      std::max(getParallelism(candidateOp), getParallelism(fusedOps));
  return fusedLoopParallelism + std::numeric_limits<float>::epsilon() >
         opMaxParallelism;
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

bool MaxParallelAnalyzer::fuseIntoGroup(const int producerIndex,
                                        const int consumerIndex) {
  if ((opToGroupIndex.find(opsInBlock[producerIndex]) ==
       opToGroupIndex.end()) &&
      (opToGroupIndex.find(opsInBlock[consumerIndex]) ==
       opToGroupIndex.end())) {
    auto groupId = AllFusedBlocks.size();
    DenseSet<Operation *> newBlocks;
    newBlocks.insert(opsInBlock[producerIndex]);
    newBlocks.insert(opsInBlock[consumerIndex]);
    AllFusedBlocks.push_back(newBlocks);
    opToGroupIndex.insert({opsInBlock[consumerIndex], groupId});
    opToGroupIndex.insert({opsInBlock[producerIndex], groupId});
  } else if ((opToGroupIndex.find(opsInBlock[producerIndex]) !=
              opToGroupIndex.end()) &&
             (opToGroupIndex.find(opsInBlock[consumerIndex]) !=
              opToGroupIndex.end())) {
    auto producerGroupId = opToGroupIndex[opsInBlock[producerIndex]];
    auto consumerGroupId = opToGroupIndex[opsInBlock[consumerIndex]];
    if (producerGroupId == consumerGroupId) {
      return false;
    }
    for (auto &op : AllFusedBlocks[consumerGroupId]) {
      AllFusedBlocks[producerGroupId].insert(op);
      opToGroupIndex[op] = producerGroupId;
    }
  } else {
    if (opToGroupIndex.find(opsInBlock[producerIndex]) !=
        opToGroupIndex.end()) {
      auto groupId = opToGroupIndex[opsInBlock[producerIndex]];
      AllFusedBlocks[groupId].insert(opsInBlock[consumerIndex]);
      opToGroupIndex.insert({opsInBlock[consumerIndex], groupId});
    } else {
      auto groupId = opToGroupIndex[opsInBlock[consumerIndex]];
      AllFusedBlocks[groupId].insert(opsInBlock[producerIndex]);
      opToGroupIndex.insert({opsInBlock[producerIndex], groupId});
    }
  }
  return true;
}

bool MaxParallelAnalyzer::isFusibleImpl(const int producerIndex,
                                        const int consumerIndex) {
  LDBG("checking fusible");
  Operation *const candidiateOp = opsInBlock[producerIndex];
  if (getComputeOpCount(candidiateOp) == 0) {
    LDBG("zero compute op, fuse");
    return true;
  }
  if (isReshapeOp(candidiateOp) || isReshapeOp(opsInBlock[consumerIndex])) {
    LDBG("has reshape Op, fuse");
    return true;
  }
  if (opToGroupIndex.find(opsInBlock[consumerIndex]) == opToGroupIndex.end()) {
    return true;
  }

  float_t candidateOpComputeScores = getComputeScores(candidiateOp);
  float_t candidateOpIoScores = getIoScores(candidiateOp);
  auto groupId = opToGroupIndex[opsInBlock[consumerIndex]];
  const auto& fusedOps = AllFusedBlocks[groupId];
  float_t fusedOpsIoScores = getFusedOpsIoScores(fusedOps);
  float_t fusedOpsComputeScores = getFusedOpsComputeScores(fusedOps);
  auto paralLift = parallelismCostModel(candidiateOp, fusedOps);
  auto execUtilLift = execUnitUtilizationCostModel(candidiateOp, fusedOps);
  // IO Scores compute with uneliminated inputs/outputs
  float_t afterFusedIoScores = candidateOpIoScores + fusedOpsIoScores;
  float_t afterFusedComputeScores =
      candidateOpComputeScores + fusedOpsComputeScores;
  LDBG("==================candidateOp============================");
  LDBG("Start candidiate Op cost model check computeScores:  "
       << candidateOpComputeScores << " IO Scores " << candidateOpIoScores);
  LDBG("==================fusedOps============================"
       << fusedOps.size());
  LDBG("Start Fused op cost model check computeScores:  "
       << fusedOpsComputeScores << " IO Scores " << fusedOpsIoScores);
  LDBG("Start Fused op cost model check paralLift:  "
       << paralLift << " execUtilLift " << execUtilLift);
  if (candidateOpIoScores + std::numeric_limits<float>::epsilon() >
          candidateOpComputeScores &&
      fusedOpsIoScores + std::numeric_limits<float>::epsilon() >
          fusedOpsComputeScores) {
    LDBG("Both IO Bound -> Fuse");
    return true;
  }
  if (!(candidateOpIoScores < candidateOpComputeScores &&
        fusedOpsIoScores < fusedOpsComputeScores) &&
      afterFusedIoScores + std::numeric_limits<float>::epsilon() >
          afterFusedComputeScores) {
    LDBG("Fused IO >= Compute -> Fuse");
    return true;
  }
  return paralLift || execUtilLift;
}

bool MaxParallelAnalyzer::fuseProducerConsumerImpl(Block &block) {
  bool hasFused = false;
  for (auto it = block.rbegin(); it != block.rend(); ++it) {
    Operation &producerOp = *it;
    if (!opToIndex.contains(&producerOp)) {
      continue;
    }
    const int producerIndex = opToIndex.at(&producerOp);

    std::vector<OpOperand *> validUses = getSortedConsumerOperands(&producerOp);

    for (OpOperand *opOperandPtr : validUses) {
      OpOperand &opOperand = *opOperandPtr;
      Operation *consumerOp = opOperand.getOwner();
      const int consumerIndex = opToIndex.at(consumerOp);

      LDBG("looped at " << producerOp.getName() << " -> "
                        << consumerOp->getName());

      if (!areFusibleOps(producerIndex, consumerIndex, &opOperand)) {
        LDBG("!areFusibleOps");
        continue;
      }

      if (!VFFusionAnalyzerBase::isFusible(producerIndex, consumerIndex))
        continue;

      if (!VFFusionAnalyzerBase::fuseIndexWith(producerIndex, consumerIndex))
        continue;

      if (!fuseIntoGroup(producerIndex, consumerIndex))
        continue;
      hasFused = true;
    }
  }
  return hasFused;
}

LogicalResult MaxParallelAnalyzer::fuseImpl(Block &block) {
  LDBG("MaxParallel Fusing " << block << "\n");
  initialize(block);
  // Perform producer-consumer fusion until no more fusions occur.
  while (fuseProducerConsumerImpl(block)) {
    // Keep looping
  }
  return success();
}

} // namespace mlir::analysis
