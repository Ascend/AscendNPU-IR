//===- BufferUtils.cpp ----------------------------------------------------===//
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

#include "bishengir/Dialect/HFusion/Utils/BufferUtils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/ExtraBuffer.h"
#include "bishengir/Dialect/Utils/UnionFind.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <set>

#define DEBUG_TYPE "bishengir-buffer-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace utils {

namespace {

inline bool isReshapingOp(Operation *op) {
  return isa<tensor::CollapseShapeOp, tensor::ReshapeOp, tensor::ExpandShapeOp>(
      op);
}

inline bool isSlicingOp(Operation *op) {
  return isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(op);
}

inline bool isTensorAliasingOp(Operation *op) {
  return isReshapingOp(op) || isSlicingOp(op);
}

inline Value getAliasSource(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case([](tensor::ExpandShapeOp expand) { return expand.getSrc(); })
      .Case([](tensor::CollapseShapeOp collapse) { return collapse.getSrc(); })
      .Case([](tensor::ExtractSliceOp extract) { return extract.getSource(); })
      .Case([](tensor::InsertSliceOp insert) { return insert.getSource(); })
      .Default([](Operation *op) {
        llvm_unreachable("Unsupported aliasing op");
        return Value();
      });
}

// Start, End, Weighted live range of operations
struct WeightedLiveRange {
  uint32_t start, end;
  int64_t weight;
  explicit WeightedLiveRange(uint32_t s = 0, uint32_t e = 0, int64_t w = 1)
      : start(s), end(e), weight(w) {}
  bool operator<(const WeightedLiveRange &other) const {
    return std::tie(start, end, weight) <
           std::tie(other.start, other.end, other.weight);
  }
};

using IdxToValMap = std::map<uint32_t, Value>;
using IdxToOpMap = std::map<uint32_t, Operation *>;
using LiveRanges = SmallVector<WeightedLiveRange>;
using WeightedEndPair = std::pair<uint32_t, int64_t>;
// using MultiBufferMap is defined in BufferUtils.h
using DataTypeWeightMap = DenseMap<Value, uint32_t>;
using ValToIdxMap = DenseMap<Value, uint32_t>;
using OpToIdxMap = DenseMap<Operation *, uint32_t>;

class ValOperationIndexer {
  uint32_t opCount = 0;

public:
  ValToIdxMap valToIdx;
  OpToIdxMap opToIdx;
  IdxToValMap idxToVal;
  IdxToOpMap idxToOp;

public:
  FailureOr<Value> getVal(uint32_t idx) const {
    if (idxToVal.count(idx))
      return idxToVal.at(idx);
    return failure();
  }
  FailureOr<Operation *> getOp(uint32_t idx) const {
    if (idxToOp.count(idx))
      return idxToOp.at(idx);
    return failure();
  }
  const uint32_t kOpNotFoundLiveRange = static_cast<uint32_t>(-1);
  uint32_t getClosestOpIdx(uint32_t idx) const {
    auto it = idxToOp.lower_bound(idx);
    if (it == idxToOp.end()) {
      // in case one live range is extended to the end for some case
      return kOpNotFoundLiveRange;
    }
    return it->first;
  }
  uint32_t getIndex(Value val) const { return valToIdx.at(val); }
  uint32_t getIndex(Operation *op) const { return opToIdx.at(op); }
  uint32_t getCurrentCount() { return opCount; }
  bool insert(Value val) {
    LDBG(val << " " << opCount);
    if (valToIdx.count(val))
      return 0;
    valToIdx[val] = opCount;
    idxToVal[opCount] = val;
    opCount++;
    return 1;
  }
  bool insert(Operation *val) {
    if (opToIdx.count(val))
      return 0;
    opToIdx[val] = opCount;
    idxToOp[opCount] = val;
    opCount++;
    return 1;
  }
};

class BufferAnalysis {
public:
  BufferAnalysis(Block &block, const BufferAnalysisOptions &options,
                 func::FuncOp op)
      : block(block), options(options), liveness(op) {}
  int64_t countMaxBuffer();

private:
  Block &block;
  BufferAnalysisOptions options;
  Liveness liveness;

  DataTypeWeightMap dataTypeWeightMap;
  DenseMap<Value, uint32_t> valToLiveRangeIdx;
  LiveRanges liveRanges;
  DenseMap<int64_t, DenseSet<uint32_t>> opToEndValIdx;
  DenseMap<uint32_t, int64_t> aliasFurthest;
  /// Alias information.
  UnionFindBase aliasSet;
  ValOperationIndexer indexer;

  // Check if a value is a buffer value
  static bool isUsingBuffer(const Value &value) {
    return isa<TypedValue<ShapedType>>(value);
  }

  // Skip operations that are ignorable
  static bool skippableOperation(Operation *op) {
    return isa<tensor::EmptyOp>(op);
  }

  void adjustInplaceReuseOp(Operation *op) {
    if (!isa<linalg::ElemwiseBinaryOp, hfusion::ElemwiseBinaryOp,
             linalg::ElemwiseUnaryOp, hfusion::ElemwiseUnaryOp>(op)) {
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "Adjusting inplace reuse op " << *op << "\n";);
    auto linalgDstOp = cast<DestinationStyleOpInterface>(op);
    auto inputsValRange = linalgDstOp.getDpsInputs();
    std::set<Value, ValueComparator> inputs(inputsValRange.begin(),
                                            inputsValRange.end());
    auto outputs = linalgDstOp.getDpsInits();
    if (outputs.size() != op->getNumResults()) {
      LDBG("Can't align result and output");
      return;
    }
    for (auto [idx, out] : llvm::enumerate(outputs)) {
      auto currentValOut = op->getResult(idx);
      auto rangesIndex = valToLiveRangeIdx.at(currentValOut);
      // magic ✿, looks for the stop of a live range if its over an op, it means
      // an inplace
      //
      // E.g:            a
      //                 |
      // empty buffer    |  e
      //                 |  |-
      //                 |  |
      // operation ends  |  e <-- inplace and ignore
      //                 |
      //                 |
      //                 a
      //
      if (indexer.getClosestOpIdx(liveRanges[rangesIndex].start) ==
          liveRanges[rangesIndex].end) {
        LDBG("Inplace reuse detected");
        liveRanges[rangesIndex].weight = 0;
      } else {
        LDBG("Still used:" << liveRanges[rangesIndex].start << " "
                           << liveRanges[rangesIndex].end);
      }
    }

    auto currentLiveIndex = indexer.opToIdx[op];
    LLVM_DEBUG(llvm::dbgs() << "Checking op " << *op << "\n";);
    LLVM_DEBUG(llvm::dbgs()
                   << "CurrentLiveIndex " << currentLiveIndex << "\n";);
    for (auto [idx, in] : llvm::enumerate(inputs)) {
      // E.g:            a
      //                 |
      // empty buffer    a - e <-- inplace and ignore the a - e transition
      // buffer
      //                     |
      //                     |
      // operation ends      e
      //
      if (!valToLiveRangeIdx.count(in))
        continue;
      auto rangesIndex = valToLiveRangeIdx.at(in);
      LLVM_DEBUG(llvm::dbgs()
                     << "Here --> rangesIndex " << rangesIndex << "\n";);
      LLVM_DEBUG(llvm::dbgs() << "Here --> rangesLast "
                              << liveRanges[rangesIndex].end << "\n";);
      if (indexer.getClosestOpIdx(liveRanges[rangesIndex].end) ==
          currentLiveIndex) {
        auto prePtr = indexer.idxToOp.lower_bound(currentLiveIndex);
        if (prePtr == indexer.idxToOp.begin())
          continue;
        prePtr--;
        liveRanges[rangesIndex].end = prePtr->first;
        LDBG("Inplace reuse detected");
      }
    }
  }

  void adjustCopyInCopyOut(Operation *op) {
    if (isa<hfusion::LoadOp>(op) || isa<hfusion::StoreOp>(op)) {
      auto copyResult = op->getResult(0);
      auto rangesIndex = valToLiveRangeIdx.at(copyResult);
      // extend load/store op result's live range to the fullest
      liveRanges[rangesIndex].end = indexer.getCurrentCount();
      LDBG("Extended live range of " << copyResult << " to "
                                     << indexer.getCurrentCount());
    }
  }

  uint32_t insertValue(const Value &value, uint32_t pos, uint32_t weight);
  /// Record the value's data type size in bits, and update the
  /// \p smallestTypeBits seen so far.
  void recordDataTypeWeight(const Value &value, uint32_t *smallestTypeBits);
  int64_t getExtraBufferSizeByFactor(Operation *op) const;
  SmallVector<Value> getOperands(Operation &op) const;
  uint32_t getValMultiBuffer(const Value &value, uint32_t def = 1) const;
  uint32_t getValDataTypeWeight(const Value &value, uint32_t def = 1) const;
  void gatherLiveRanges(const LivenessBlockInfo *blockInfo);
  uint32_t updateAliasIntoFurthest(const Value &value, Operation *endOp);
  void gatherDataTypeWeights();
  void gatherIndexingAndAlias();
  void printLiveRanges() const;
  void printAliasInfo();
  int64_t lineSweepRanges();
};

uint32_t BufferAnalysis::insertValue(const Value &value, uint32_t pos,
                                     uint32_t weight = 1) {
  LDBG("--- Inserting value " << value << " " << pos << " " << weight);
  assert(!valToLiveRangeIdx.count(value));
  liveRanges.emplace_back(pos, pos, weight);
  return valToLiveRangeIdx[value] = liveRanges.size() - 1;
}

// FIXME: add other ops that needs temp buffer (e.g. select)
// TODO: change the return type to double
int64_t BufferAnalysis::getExtraBufferSizeByFactor(Operation *op) const {
  // The extra buffer factor also needs to be weighted by the data type weight
  // of the factor's source.
  // TODO: Need to refactor this code later because we're using the knowledge
  // that which op's extra buffer is infered from which operand.
  if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op)) {
    if (auto res = hfusion::util::getExtraBufferSizeForReduceOp(
            reduceOp, hfusion::util::BufferSizeUnit::FACTOR)) {
      // For multi-reduce, we can multiply by the number of results because
      // the inputs/results have the same shape.
      // But we're also assuming that they have the same data type here.
      // This may not always hold.
      return *res * static_cast<int64_t>(reduceOp.getNumResults()) *
             static_cast<int64_t>(
                 getValDataTypeWeight(reduceOp.getDpsInputOperand(0)->get()));
    }
  }
  if (auto reduceWithIndexOp = dyn_cast<hfusion::ReduceWithIndexOp>(op)) {
    if (auto res = hfusion::util::getExtraBufferSizeForReduceOp(
            reduceWithIndexOp, hfusion::util::BufferSizeUnit::FACTOR)) {
      return *res * static_cast<int64_t>(getValDataTypeWeight(
                        reduceWithIndexOp.getDpsInputOperand(0)->get()));
    }
  }
  if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(op)) {
    if (auto res = hfusion::util::getExtraBufferSizeForBroadcastOp(
            broadcastOp, hfusion::util::BufferSizeUnit::FACTOR)) {
      return *res * static_cast<int64_t>(
                        getValDataTypeWeight(broadcastOp->getResult(0)));
    }
  }
  return 0;
}

SmallVector<Value> BufferAnalysis::getOperands(Operation &op) const {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    return linalgOp.getDpsInputs();
  }
  if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
    return SmallVector<Value>(returnOp.getOperands().begin(),
                              returnOp.getOperands().end());
  }
  return SmallVector<Value>(op.getOperands().begin(), op.getOperands().end());
}

uint32_t BufferAnalysis::getValMultiBuffer(const Value &value,
                                           uint32_t def) const {
  if (options.multiBufferCount.count(value))
    return options.multiBufferCount.at(value);
  return def;
}

uint32_t BufferAnalysis::getValDataTypeWeight(const Value &value,
                                              uint32_t def) const {
  if (dataTypeWeightMap.count(value))
    return dataTypeWeightMap.at(value);
  return def;
}

/// Get the final user of \p val considering aliasing information.
///
/// For example,
/// ```mlir
///   %ret0 = end_op(%val)
///   ...
///   %ret1 = aliasing_op(%val)
///   ...
///   real_end_op(%ret1)
/// ```
///
/// The liveness will return `end_op` as the end operation. However, since
/// `%val` has an aliasing op, the real end op is `real_end_op`.
uint32_t
mlir::utils::BufferAnalysis::updateAliasIntoFurthest(const Value &val,
                                                     Operation *endOp) {
  auto valIdx = indexer.getIndex(val);
  LDBG("found valIdx " << valIdx);
  // Finding parent, min Index of ufds parent
  auto aliasParent = aliasSet.minIndex[aliasSet.find(valIdx)];
  LDBG("found alias parent " << aliasParent);
  // Check if the alias is updated
  LDBG("Ok found endIdx " << *endOp);
  auto endIdx = indexer.getIndex(endOp);
  LDBG("Ok found endIdx " << endIdx);
  // why no overflow static checker?
  if (!aliasFurthest.count(aliasParent))
    aliasFurthest[aliasParent] = -1;
  auto &furthestPtr = aliasFurthest[aliasParent];
  LDBG(endIdx << " " << furthestPtr << " end -- " << *endOp);
  if (endIdx > furthestPtr) {
    LDBG("Updating furthest " << endIdx << " " << furthestPtr);
    opToEndValIdx[furthestPtr].erase(aliasParent);
    furthestPtr = endIdx;
    opToEndValIdx[endIdx].insert(aliasParent);
  }
  return aliasParent;
}

void BufferAnalysis::gatherLiveRanges(const LivenessBlockInfo *blockInfo) {
  LDBG("Gathering live range information...");
  assert(blockInfo != nullptr);
  for (const auto &arg : block.getArguments()) {
    LDBG("Processing arguments " << arg);
    // For the arguments, we assume that they are explicitly copied into
    // a local buffer. So the arguments themselves have a weight of zero.
    Operation *startOp = blockInfo->getStartOperation(arg);
    Operation *endOp = blockInfo->getEndOperation(arg, startOp);
    auto aliasParent = updateAliasIntoFurthest(arg, endOp);
    auto currentWeight = 0;
    LDBG("inserting " << arg);
    insertValue(arg, aliasParent, currentWeight);
  }
  for (auto &op : block) {
    if (skippableOperation(&op))
      continue;
    uint32_t currentOpIndex = indexer.getIndex(&op);
    auto destOp = dyn_cast<DestinationStyleOpInterface>(op);
    for (const auto &[idx, res] : llvm::enumerate(op.getResults())) {
      if (isUsingBuffer(res)) {
        Operation *endOp = blockInfo->getEndOperation(res, &op);
        auto aliasParent = updateAliasIntoFurthest(res, endOp);
        auto currentWeight =
            getValMultiBuffer(res) *
            (destOp ? getValMultiBuffer(destOp.getDpsInits()[idx], 1) : 1) *
            getValDataTypeWeight(res);
        LDBG("inserting " << res);
        if (aliasParent != indexer.getIndex(res))
          currentWeight = 0;
        insertValue(res, aliasParent, currentWeight);
      }
    }

    // opToEndValIdx have the list of operations which a certain op stops
    LDBG("Printing dead val at " << currentOpIndex << " " << op);
    for (auto deadVal : opToEndValIdx[currentOpIndex]) {
      LDBG("Here is " << deadVal);
      Value curVal = indexer.getVal(deadVal).value();
      auto indexPos = valToLiveRangeIdx[curVal];
      liveRanges[indexPos].end = currentOpIndex;
    }
    if (auto extraWeight = getExtraBufferSizeByFactor(&op)) {
      extraWeight *= std::max(
          (uint32_t)1, getValMultiBuffer(op.getResult(0), 0) +
                           getValMultiBuffer(
                               cast<linalg::LinalgOp>(op).getDpsInits()[0], 0));
      LDBG("Appending " << op << " with " << extraWeight);
      liveRanges.emplace_back(currentOpIndex, currentOpIndex, extraWeight);
    }
  }
  for (auto &op : block) {
    if (skippableOperation(&op))
      continue;

    adjustInplaceReuseOp(&op);

    if (options.enableDmaOpt)
      adjustCopyInCopyOut(&op);
  }
}

void BufferAnalysis::recordDataTypeWeight(const Value &value,
                                          uint32_t *smallestTypeBits) {
  auto maybeElementType = getElementTypeOrSelf(value);
  assert(maybeElementType.isIntOrFloat() &&
         "Can only handle int or float element type!");
  uint32_t currentTypeBits =
      static_cast<uint32_t>(maybeElementType.getIntOrFloatBitWidth());
  dataTypeWeightMap[value] = currentTypeBits;
  *smallestTypeBits = std::min<uint32_t>(*smallestTypeBits, currentTypeBits);
}

void BufferAnalysis::gatherDataTypeWeights() {
  LDBG("Gathering data type information...");
  uint32_t smallestTypeBits = std::numeric_limits<uint32_t>::max();
  for (const auto &arg : block.getArguments()) {
    if (!isUsingBuffer(arg))
      continue;

    recordDataTypeWeight(arg, &smallestTypeBits);
  }

  for (auto &op : block) {
    // Only consider operations involved in computation.
    if (!isa<linalg::LinalgOp>(op))
      continue;

    for (const auto &[idx, res] : llvm::enumerate(op.getResults())) {
      if (!isUsingBuffer(res))
        continue;

      recordDataTypeWeight(res, &smallestTypeBits);
    }

    for (const auto &[idx, operand] : llvm::enumerate(op.getOperands())) {
      if (!isUsingBuffer(operand))
        continue;

      recordDataTypeWeight(operand, &smallestTypeBits);
    }
  }
  LDBG("Smallest type bits is " << smallestTypeBits
                                << ", normalizing weights...");
  std::for_each(
      dataTypeWeightMap.begin(), dataTypeWeightMap.end(), [&](auto &pair) {
        auto normalizedTypeBits = pair.second / smallestTypeBits;
        if (pair.second % smallestTypeBits != 0) {
          LDBG(
              "WARN: Current type bits "
              << pair.second
              << " is not divisible by the smallest type bits! Rounding up...");
          normalizedTypeBits =
              (pair.second + smallestTypeBits - 1) / smallestTypeBits;
        }
        pair.second = normalizedTypeBits;
      });
}

void BufferAnalysis::printLiveRanges() const {
  llvm::outs() << "Considering " << valToLiveRangeIdx.size() << " and "
               << liveRanges.size() - valToLiveRangeIdx.size()
               << " extra Live Range:\n";

  for (size_t i = 0; i < liveRanges.size(); i++) {
    llvm::outs() << "Live Range #" << i << ": "
                 << "\n";
    if (i == 0 || liveRanges[i].start != liveRanges[i - 1].start) {
      auto currentVal = indexer.getVal(liveRanges[i].start);
      if (succeeded(currentVal)) {
        llvm::outs() << indexer.getVal(liveRanges[i].start).value() << ": \n";
      } else {
        llvm::outs() << *indexer.getOp(liveRanges[i].start).value() << ": \n";
      }
    }

    llvm::outs() << liveRanges[i].start << " " << liveRanges[i].end << " "
                 << liveRanges[i].weight << "\n";
    llvm::outs() << "Done Live Range\n";
  }
}

int64_t BufferAnalysis::lineSweepRanges() {
  llvm::PriorityQueue<WeightedEndPair, SmallVector<WeightedEndPair>,
                      std::greater<WeightedEndPair>>
      earlyDone;

  int64_t maxBuffer = 0;
  int64_t currentBuffer = 0;

  for (const auto &liveRange : liveRanges) {
    if (liveRange.start == liveRange.end) {
      LDBG("WARN: dead operation or temporary buffer exists at position "
           << liveRange.start);
    }
    while (!earlyDone.empty() && earlyDone.top().first < liveRange.start) {
      currentBuffer -= earlyDone.top().second;
      earlyDone.pop();
    }
    earlyDone.push({liveRange.end, liveRange.weight});
    currentBuffer += liveRange.weight;
    maxBuffer = std::max(maxBuffer, currentBuffer);
  }
  return maxBuffer;
}

void BufferAnalysis::printAliasInfo() {
  for (auto [idx, val] : indexer.idxToVal) {
    auto aliasParent = indexer.getVal(aliasSet.minIndex[aliasSet.find(idx)]);
    if (aliasParent != val)
      LDBG("value: " << val << " alias parent is: " << aliasParent);
  }
}

void BufferAnalysis::gatherIndexingAndAlias() {
  LDBG("Gathering alias information...");
  for (const auto &arg : block.getArguments()) {
    LDBG("Processing argument: " << arg);
    indexer.insert(arg);
  }
  for (auto &op : block) {
    LDBG("Processing op: " << op);
    for (auto res : op.getResults()) {
      if (!isUsingBuffer(res))
        continue;

      // Example:
      // %res = tensor.expand_shape %src
      // %src is an alias of %res, should be treated equally
      indexer.insert(res);
      if (isTensorAliasingOp(&op)) {
        auto src = getAliasSource(&op);
        auto aliasSrcPar = indexer.getIndex(src);
        aliasSet.join(aliasSrcPar, indexer.getIndex(res));
      }
    }
    LDBG("Inserting op " << op);
    indexer.insert(&op);
  }
  printAliasInfo();
}

int64_t BufferAnalysis::countMaxBuffer() {
  const LivenessBlockInfo *blockInfo = liveness.getLiveness(&block);
  gatherIndexingAndAlias();
  gatherDataTypeWeights();
  gatherLiveRanges(blockInfo);
  llvm::sort(liveRanges);
  if (options.printLiveRange) {
    printLiveRanges();
  }
  auto maxBuffer = lineSweepRanges();
  return maxBuffer;
}
} // namespace

int64_t countMaxBuffer(func::FuncOp func,
                       const BufferAnalysisOptions &options) {
  if (func.getBody().getBlocks().size() != 1)
    return -1;

  LDBG("=== Begin counting max buffer of func: \n" << *func);
  BufferAnalysis analysis(*func.getBody().begin(), options, func);
  return analysis.countMaxBuffer();
}

} // namespace utils
} // namespace mlir