//===- AVECostModel.cpp - AVE profitability cost model ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMAVE/CostModelInfo/AVECostModel.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#define DEBUG_TYPE "ave-cost-model"

using namespace mlir;
using namespace mlir::hivmave;

namespace {

constexpr float kCostEpsilon = 1.0e-5f;
constexpr unsigned kIssueQueueLength = 64;
const analysis::CostInfo kDefaultExecuteCost{1, 4, 2};
const analysis::CostInfo kDefaultExpCost{4, 13, 2};
const analysis::CostInfo kDefaultDivSqrtCost{4, 14, 2};
const analysis::CostInfo kDefaultLogCost{4, 15, 2};
// Conservative fallback for non-A5 targets or entries absent from the table.
const analysis::CostInfo kInterleaveCost{2, 5, 2};
const analysis::CostInfo kDeInterleaveCost{2, 5, 2};

static Type getCostSourceType(const Operation &op) {
  // MLIR's Operation range accessors are not const-qualified.
  Operation &operation = const_cast<Operation &>(op);
  for (Value operand : operation.getOperands()) {
    auto vectorType = dyn_cast<VectorType>(operand.getType());
    if (vectorType && !vectorType.getElementType().isInteger(1))
      return vectorType;
  }
  return {};
}

static Type getCostResultType(const Operation &op) {
  Operation &operation = const_cast<Operation &>(op);
  for (Value result : operation.getResults())
    if (isa<VectorType>(result.getType()))
      return result.getType();
  return {};
}

static AVECostTypeKind getAVECostTypeKind(Type type) {
  auto vectorType = dyn_cast_or_null<VectorType>(type);
  if (!vectorType)
    return AVECostTypeKind::Unknown;

  Type elementType = vectorType.getElementType();
  if (isa<Float8E4M3FNType>(elementType))
    return AVECostTypeKind::F8E4M3FN;
  if (isa<Float8E5M2Type>(elementType))
    return AVECostTypeKind::F8E5M2;
  if (elementType.isF16())
    return AVECostTypeKind::F16;
  if (elementType.isBF16())
    return AVECostTypeKind::BF16;
  if (elementType.isF32())
    return AVECostTypeKind::F32;

  auto integerType = dyn_cast<IntegerType>(elementType);
  if (!integerType)
    return AVECostTypeKind::Unknown;
  if (integerType.getWidth() == 8)
    return AVECostTypeKind::I8;
  if (integerType.getWidth() == 16)
    return AVECostTypeKind::I16;
  if (integerType.getWidth() == 32)
    return AVECostTypeKind::I32;
  if (integerType.getWidth() == 64)
    return AVECostTypeKind::I64;
  return AVECostTypeKind::Unknown;
}

static analysis::CostInfo getAVEFallbackCost(const Operation &op) {
  if (isa<VFExpOp>(&op))
    return kDefaultExpCost;
  if (isa<VFLnOp>(&op))
    return kDefaultLogCost;
  if (isa<VFDivOp, VFDivfOp, VFDivFHPOp, VFSqrtOp, VFRsqrtOp>(&op))
    return kDefaultDivSqrtCost;
  return kDefaultExecuteCost;
}

static float getParallelismRatio(const AVEPipelineCost &cost) {
  float required = cost.requiredParallelism();
  if (required <= kCostEpsilon)
    return std::numeric_limits<float>::infinity();
  return cost.issueQueueParallelism() / required;
}

static void debugPipelineCost(StringRef label, const AVEPipelineCost &cost) {
  LLVM_DEBUG(
      llvm::dbgs() << "[" DEBUG_TYPE "] " << label << ": io=" << cost.ioScore()
                   << " (loads=" << cost.loadInstructionCount
                   << ", stores=" << cost.storeInstructionCount << ")"
                   << ", execute=" << cost.executeScore()
                   << ", instructions=" << cost.totalInstructionCount()
                   << ", required-parallelism=" << cost.requiredParallelism()
                   << ", issue-queue-parallelism="
                   << cost.issueQueueParallelism() << "\n");
}

} // namespace

void AVEPipelineCost::addLoad(int64_t count) { loadInstructionCount += count; }

void AVEPipelineCost::addStore(int64_t count) {
  storeInstructionCount += count;
}

void AVEPipelineCost::addExecution(const analysis::CostInfo &cost,
                                   int64_t count) {
  for (AVEExecutionCostEntry &entry : executionCosts) {
    if (entry.cost.execInterval == cost.execInterval &&
        entry.cost.execLatency == cost.execLatency &&
        entry.cost.execUnit == cost.execUnit) {
      entry.count += count;
      return;
    }
  }
  executionCosts.push_back({cost, count});
}

AVEPipelineCost &AVEPipelineCost::operator+=(const AVEPipelineCost &other) {
  loadInstructionCount += other.loadInstructionCount;
  storeInstructionCount += other.storeInstructionCount;
  for (const AVEExecutionCostEntry &entry : other.executionCosts)
    addExecution(entry.cost, entry.count);
  return *this;
}

AVEPipelineCost AVEPipelineCost::scaled(int64_t factor) const {
  AVEPipelineCost result;
  result.loadInstructionCount = loadInstructionCount * factor;
  result.storeInstructionCount = storeInstructionCount * factor;
  for (const AVEExecutionCostEntry &entry : executionCosts)
    result.addExecution(entry.cost, entry.count * factor);
  return result;
}

float AVEPipelineCost::ioScore() const {
  assert(loadInstructionCount >= 0 && storeInstructionCount >= 0 &&
         "negative IO count after applying delta");
  // HFusion's average-cycle model assumes no bank conflict, two load ports and
  // one store port. Paired load/store traffic overlaps; remaining loads dual
  // issue.
  if (storeInstructionCount >= loadInstructionCount)
    return static_cast<float>(storeInstructionCount);
  return static_cast<float>(storeInstructionCount) +
         static_cast<float>(loadInstructionCount - storeInstructionCount) *
             0.5f;
}

float AVEPipelineCost::executeScore() const {
  float score = 0.0f;
  for (const AVEExecutionCostEntry &entry : executionCosts) {
    assert(entry.count >= 0 && "negative execution cost after applying delta");
    if (entry.count == 0 || entry.cost.execUnit <= 0)
      continue;
    score += static_cast<float>(entry.count * entry.cost.execInterval) /
             static_cast<float>(entry.cost.execUnit);
  }
  return score;
}

float AVEPipelineCost::bottleneck() const {
  return std::max(ioScore(), executeScore());
}

float AVEPipelineCost::totalThroughput() const {
  return ioScore() + executeScore();
}

int64_t AVEPipelineCost::executeInstructionCount() const {
  int64_t count = 0;
  for (const AVEExecutionCostEntry &entry : executionCosts) {
    assert(entry.count >= 0 && "negative execution count after applying delta");
    count += entry.count;
  }
  return count;
}

int64_t AVEPipelineCost::totalInstructionCount() const {
  assert(loadInstructionCount >= 0 && storeInstructionCount >= 0 &&
         "negative IO count after applying delta");
  return loadInstructionCount + executeInstructionCount() +
         storeInstructionCount;
}

float AVEPipelineCost::requiredParallelism() const {
  float parallelism = 0.0f;
  for (const AVEExecutionCostEntry &entry : executionCosts) {
    assert(entry.count >= 0 && "negative execution cost after applying delta");
    if (entry.count == 0 || entry.cost.execInterval <= 0)
      continue;
    float instructionParallelism =
        static_cast<float>(entry.cost.execUnit * entry.cost.execLatency) /
        static_cast<float>(entry.cost.execInterval);
    parallelism = std::max(parallelism, instructionParallelism);
  }
  return parallelism;
}

float AVEPipelineCost::executionUnitUtilization() const {
  int64_t singleCount = 0;
  int64_t doubleCount = 0;
  float maximumGroupCycles = 0.0f;

  for (const AVEExecutionCostEntry &entry : executionCosts) {
    assert(entry.count >= 0 && "negative execution cost after applying delta");
    if (entry.count == 0 || entry.cost.execUnit <= 0)
      continue;
    if (entry.cost.execUnit == 2)
      doubleCount += entry.count;
    else
      singleCount += entry.count;
    float groupCycles =
        static_cast<float>(entry.count * entry.cost.execInterval) /
        static_cast<float>(entry.cost.execUnit);
    maximumGroupCycles = std::max(maximumGroupCycles, groupCycles);
  }

  if (doubleCount < singleCount)
    maximumGroupCycles =
        std::max(maximumGroupCycles, static_cast<float>(singleCount));
  if (maximumGroupCycles <= kCostEpsilon)
    return 0.0f;
  float utilization = static_cast<float>(singleCount + doubleCount) /
                      (maximumGroupCycles * 2.0f);
  return std::min(utilization, 1.0f);
}

float AVEPipelineCost::issueQueueParallelism() const {
  int64_t totalCount = totalInstructionCount();
  int64_t executeCount = executeInstructionCount();
  if (totalCount == 0 || executeCount == 0)
    return std::numeric_limits<float>::infinity();
  return (static_cast<float>(kIssueQueueLength * 2) /
          static_cast<float>(totalCount)) *
         (static_cast<float>(executeCount) / static_cast<float>(totalCount));
}

AVECostModel::AVECostModel(Operation &anchor) {
  ModuleOp module = utils::getTopLevelModuleOp(&anchor);
  std::optional<hacc::TargetDevice> target =
      hacc::utils::getTargetDevice(module);
  if (!target || !hacc::utils::isAscend950(*target))
    return;
  targetConfig = &AVECostModelInfo::getInstance().getConfigMap();
}

std::optional<analysis::CostInfo>
AVECostModel::lookupExecutionCost(const Operation &op) const {
  Type sourceType = getCostSourceType(op);
  Type resultType = getCostResultType(op);
  if (!sourceType && !resultType)
    return std::nullopt;
  if (!sourceType)
    sourceType = resultType;
  if (!resultType)
    resultType = sourceType;
  Operation &operation = const_cast<Operation &>(op);
  std::optional<RegisteredOperationName> registeredInfo =
      operation.getRegisteredInfo();
  if (!registeredInfo)
    return std::nullopt;
  return lookupExecutionCost(registeredInfo->getTypeID(), sourceType,
                             resultType);
}

std::optional<analysis::CostInfo>
AVECostModel::lookupExecutionCost(TypeID opTypeID, Type sourceType,
                                  Type resultType) const {
  if (!targetConfig || !sourceType || !resultType)
    return std::nullopt;

  auto opCost = targetConfig->find(opTypeID);
  if (opCost == targetConfig->end())
    return std::nullopt;
  AVECostTypeKind sourceKind = getAVECostTypeKind(sourceType);
  auto sourceCost = opCost->second.find(sourceKind);
  if (sourceCost == opCost->second.end())
    return std::nullopt;
  AVECostTypeKind resultKind = getAVECostTypeKind(resultType);
  auto resultCost = sourceCost->second.find(resultKind);
  if (resultCost != sourceCost->second.end())
    return resultCost->second;
  return std::nullopt;
}

AVEPipelineCost AVECostModel::estimateOperation(const Operation &op) const {
  AVEPipelineCost result;
  if (isa<VFLoadOp>(&op)) {
    result.addLoad();
    return result;
  }
  if (isa<VFMaskedStoreOp, VFStoreWithStrideOp, VFUnalignedMaskedStoreOp>(
          &op)) {
    result.addStore();
    return result;
  }
  if (isa<VFPgeOp>(&op))
    return result;
  if (isa<VFInterleaveOp>(&op)) {
    result.addExecution(getInterleaveCost(getCostResultType(op)));
    return result;
  }
  if (isa<VFDeInterleaveOp>(&op)) {
    result.addExecution(getDeInterleaveCost(getCostResultType(op)));
    return result;
  }
  // The interfaces define AVE operations known to execute on the vector
  // pipeline. Other AVE operations are unknown to this model and do not
  // participate in profitability decisions.
  if (!isa<AVEElementwiseOp>(&op) && !isa<AVEOpWithLibraryFunction>(&op))
    return result;

  std::optional<analysis::CostInfo> targetCost = lookupExecutionCost(op);
  result.addExecution(targetCost.value_or(getAVEFallbackCost(op)));
  return result;
}

AVEPipelineCost AVECostModel::estimateLoop(scf::ForOp loop) const {
  AVEPipelineCost result;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (op.getNumRegions() != 0)
      continue;
    Dialect *dialect = op.getDialect();
    if (!dialect || dialect->getNamespace() != "ave")
      continue;
    result += estimateOperation(op);
  }
  return result;
}

analysis::CostInfo AVECostModel::getInterleaveCost(Type dataType) const {
  return lookupExecutionCost(TypeID::get<VFInterleaveOp>(), dataType, dataType)
      .value_or(kInterleaveCost);
}

analysis::CostInfo AVECostModel::getDeInterleaveCost(Type dataType) const {
  return lookupExecutionCost(TypeID::get<VFDeInterleaveOp>(), dataType,
                             dataType)
      .value_or(kDeInterleaveCost);
}

bool AVECostModel::isProfitable(const AVEPipelineCost &originalIteration,
                                const AVEPipelineCost &beforeWindow,
                                const AVEPipelineCost &afterWindow) const {
  debugPipelineCost("original iteration", originalIteration);
  debugPipelineCost("before", beforeWindow);
  debugPipelineCost("after", afterWindow);

  float beforeBottleneck = beforeWindow.bottleneck();
  float afterBottleneck = afterWindow.bottleneck();
  if (afterBottleneck > beforeBottleneck + kCostEpsilon) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] reject: bottleneck increases\n");
    return false;
  }
  // A lower dominant pipeline cost is the strongest signal and must not be
  // vetoed merely because unrolling makes each loop body larger. VFFusion's
  // issue-queue model is used below only to break throughput ties.
  if (beforeBottleneck > afterBottleneck + kCostEpsilon) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] accept: bottleneck decreases\n");
    return true;
  }

  if (beforeWindow.totalThroughput() >
      afterWindow.totalThroughput() + kCostEpsilon) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] accept: total pipeline work decreases\n");
    return true;
  }
  if (beforeWindow.totalInstructionCount() >
      afterWindow.totalInstructionCount()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] accept: instruction count decreases\n");
    return true;
  }

  // Preserve enough independent loop bodies to hide the longest instruction
  // only when throughput and instruction work are otherwise tied. Applying
  // this as a hard unroll veto would hide known factor-4 store-side savings.
  float beforeRatio = getParallelismRatio(originalIteration);
  float afterRatio = getParallelismRatio(afterWindow);
  if (beforeRatio + kCostEpsilon >= 1.0f && afterRatio + kCostEpsilon < 1.0f) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE
               << "] reject: issue queue no longer hides latency\n");
    return false;
  }
  if (beforeRatio < 1.0f && afterRatio + kCostEpsilon < beforeRatio) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE
               << "] reject: insufficient parallelism becomes worse\n");
    return false;
  }

  bool improvesUtilization =
      afterWindow.executionUnitUtilization() >
      beforeWindow.executionUnitUtilization() + kCostEpsilon;
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] "
                          << (improvesUtilization ? "accept" : "reject")
                          << ": execution-unit utilization tie-break\n");
  return improvesUtilization;
}
