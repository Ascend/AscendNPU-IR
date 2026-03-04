//===- DimensionAnalyzer.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/Utils/Util.h"

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;

#define DEBUG_TYPE "dimension-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hfusion {
namespace detail {

BitVector DimensionAnalyzer::getCommonAxis(Value v) {
  // Nothing to do if an anchor has not yet been computed.
  if (anchor_.empty() || !argumentsRefPointer_.contains(v))
    return BitVector();
  LDBG("Computing common axis");
#ifndef NDEBUG
  debugPrintAnchor();
#endif
  auto vRef = getArgumentRef(v);
  LDBG("v: " << v);
  LDBG("vRef: " << utils::debugger::to_string(vRef));
  IndexSet vIndexSet;
  computeIndexSet(vIndexSet, vRef);
  BitVector bitRet(anchor_.size(), false);
  for (const auto &[i, anchorIndex] : enumerate(anchorIndexSet_)) {
    if (vIndexSet.contains(anchorIndex))
      bitRet[i] = true;
  }
  LDBG("Result: " << utils::debugger::to_string(bitRet));
  return bitRet;
}

SmallVector<Value> DimensionAnalyzer::getAnchorCandidate() {
  SmallVector<Value> analyzedTensors;
  for (Block &block : op_->getRegion(0)) {
    block.walk([&](Operation *op) {
      // TODO: what is the definition of a candidate anchor
      bool isCandidateAnchor =
          opfusion::FusibleHelper::isImportantPattern(op) ||
          isa<DestinationStyleOpInterface>(op);
      if (!isCandidateAnchor)
        return;
      for (auto res : op->getResults()) {
        if (argumentsRefPointer_.contains(res)) {
          analyzedTensors.push_back(res);
        }
      }
    });
  }
  return analyzedTensors;
}

SmallVector<SmallVector<DimensionAnalyzer::AnchorDimension>>
DimensionAnalyzer::getAnchorShape() {
  SmallVector<SmallVector<AnchorDimension>> anchorMaxRank;
  LDBG("Get max rank dims");
  for (auto &[anchorIndex, maxStaticShape, dynamicShapeIndices] : anchor_) {
    SmallVector<AnchorDimension> anchorDim;
    LDBG("Dim " << anchorMaxRank.size() << ":");
    anchorDim.emplace_back(nullptr, maxStaticShape);
    for (auto dynamicShapeCurrentIndex : dynamicShapeIndices) {
      dynamicShapeCurrentIndex =
          solverShapeElem_->find(dynamicShapeCurrentIndex);
      auto [dynamicShapeValue, dynamicShapeIndex] =
          getDimension(dynamicShapeCurrentIndex);
      LDBG(anchorDim.size() << "th dynamicShapeValue is " << dynamicShapeValue);
      LDBG(anchorDim.size() << "th dynamicShapeIndex is " << dynamicShapeIndex);
      anchorDim.emplace_back(dynamicShapeValue, dynamicShapeIndex);
    }
    anchorMaxRank.emplace_back(std::move(anchorDim));
  }
  return anchorMaxRank;
}

SmallVector<int64_t> DimensionAnalyzer::getMaxRankDimShape() {
  SmallVector<int64_t> staticMaxDims;
  for (auto &anchorAxis : anchor_) {
    staticMaxDims.push_back(anchorAxis.maxStaticShape);
  }
  return staticMaxDims;
}

size_t DimensionAnalyzer::getAnchorRank() const { return anchor_.size(); }

void DimensionAnalyzer::computeIndexSet(IndexSet &indexSet,
                                        ArrayRef<int64_t> vRef) {
  DenseMap<int64_t, int64_t> indexCount;
  for (auto index : vRef) {
    index = solverShapeElem_->minIndex[solverCollapserElem_->find(index)];
    indexSet.insert({index, indexCount[index]++});
  }
}

void DimensionAnalyzer::computeAnchorElement(
    DenseMap<int64_t, AnchorElement> &indexAncherElemMap,
    Value anchorCandidate) {
  auto anchorRef = getArgumentRef(anchorCandidate);
  computeIndexSet(anchorIndexSet_, anchorRef);
  for (auto anchorAxis : anchorRef) {
    auto currentAxisMinIndex =
        solverShapeElem_->minIndex[solverCollapserElem_->find(anchorAxis)];
    auto anchorAxisShape =
        solverShapeElem_->getMinParentAndShapePair(anchorAxis).second;

    auto &[axisIndex, maxStaticShape, dynamicShapeIndices] =
        indexAncherElemMap[currentAxisMinIndex];
    axisIndex = currentAxisMinIndex;
    if (ShapedType::isDynamic(anchorAxisShape)) {
      dynamicShapeIndices.push_back(
          solverShapeElem_->minIndex[solverShapeElem_->find(anchorAxis)]);
    } else {
      maxStaticShape = std::max(maxStaticShape, anchorAxisShape);
    }
  }
}

void DimensionAnalyzer::computeAnchor() {
  if (!anchor_.empty())
    return;

  if (reverseShapeElem_.empty())
    computeReverseElementMap();

  LDBG("Computing anchor");
  LDBG("Input IR: " << *op_);

  // Strategy: Find all distinct axis
  auto analyzedTensors = getAnchorCandidate();

  linalg::ReduceOp reduceOp = nullptr;
  for (auto res : analyzedTensors)
    if ((reduceOp = res.getDefiningOp<linalg::ReduceOp>()))
      break;

  DenseMap<int64_t, AnchorElement> indexAncherElemMap;
  if (reduceOp) {
    // WARN: Temporary solution to support tileReductionUsingFor
    // Should be removed after supporting interchange
    // Not working if there is more than one reduceOp
    // For example:
    // 1x2x3x4 -> 1x2x4x3 (transpose)
    // 1x2x3x4 -> 2x3x4 (reduce)
    // 1x2x4x3 -> 2x4x3 (reduce)
    computeAnchorElement(indexAncherElemMap,
                         reduceOp.getDpsInputOperand(0)->get());
  }

  for (auto res : analyzedTensors) {
    if (res.getDefiningOp<hfusion::StoreOp>()) {
      LDBG("Found candidate anchor " << res);
      computeAnchorElement(indexAncherElemMap, res);
    }
  }

  for (auto res : analyzedTensors) {
    if (!res.getDefiningOp<hfusion::StoreOp>()) {
      LDBG("Found candidate anchor " << res);
      computeAnchorElement(indexAncherElemMap, res);
    }
  }

  for (auto [anchorAxisIndex, _] : anchorIndexSet_) {
    auto &anchorElement = indexAncherElemMap[anchorAxisIndex];
    llvm::sort(anchorElement.dynamicShapeIndices);
    anchorElement.dynamicShapeIndices.erase(
        std::unique(anchorElement.dynamicShapeIndices.begin(),
                    anchorElement.dynamicShapeIndices.end()),
        anchorElement.dynamicShapeIndices.end());
    anchor_.push_back(std::move(anchorElement));
  }

#ifndef NDEBUG
  debugPrintAnchor();
#endif
}

SmallVector<int64_t> DimensionAnalyzer::getInterchange(Value v) {
  SmallVector<int64_t> interchange(utils::getShapeRank(v.getType()).value_or(0),
                                   -1);
  if (anchor_.empty() || !argumentsRefPointer_.contains(v))
    return interchange;
  DenseMap<int64_t, SmallVector<int64_t>> anchorPos;
  for (size_t i = 0; i < anchor_.size(); ++i) {
    auto currentAxis = solverCollapserElem_->find(anchor_[i].axisIndex);
    if (anchorPos.contains(currentAxis)) {
      LDBG("Two same aligned collapse in solver anchor "
           << i << " " << anchorPos[currentAxis].front());
    }
    anchorPos[currentAxis].push_back(i);
  }
  auto vRef = getArgumentRef(v);
  assert(anchor_.size() >= vRef.size() &&
         "Anchor should represent all possible values");
  for (auto &val : anchorPos) {
    std::reverse(val.second.begin(), val.second.end());
  }
  for (size_t i = 0; i < vRef.size(); ++i) {
    auto currentAxis = solverCollapserElem_->find(vRef[i]);
    if (!anchorPos.contains(currentAxis))
      continue;
    interchange[i] = anchorPos[currentAxis].back();
    assert(!anchorPos[currentAxis].empty() &&
           "Anchor should represent all possible values");
    anchorPos[currentAxis].pop_back();
  }

  LDBG("Interchanged " << v << " to "
                       << utils::debugger::to_string(interchange));
  return interchange;
}

SmallVector<int64_t> DimensionAnalyzer::getNormalizedInterchange(Value v) {
  auto res = getInterchange(v);
  //  Normalize the negative one
  LDBG("Computing normalized interchange: " << utils::debugger::to_string(res));
  BitVector availableOnes(res.size());
  int expectReplace = 0;
  for (auto dim : res) {
    if (dim != -1) {
      // If the current dimension exceed the predicted
      if (dim >= availableOnes.size())
        availableOnes.resize(dim + 1);
      availableOnes[dim] = true;
    } else
      expectReplace++;
  }
  SmallVector<int> available;
  int last = static_cast<int>(availableOnes.size());
  for (int i = 0; i < last; ++i) {
    if (!availableOnes[i])
      available.push_back(i);
  }
  int numbersToAdd = expectReplace - static_cast<int64_t>(available.size());
  for (int i = 0; i < numbersToAdd; i++) {
    available.push_back(last++);
  }
  size_t ptr = 0;
  for (auto &dim : res) {
    if (dim == -1) {
      assert(ptr != available.size());
      dim = available[ptr++];
    }
  }
  LDBG(utils::debugger::to_string(res));
  return res;
}

#ifndef NDEBUG
void DimensionAnalyzer::debugPrintAnchor() {
  LDBG("Anchor is: ");
  for (const auto &anchorAxis : anchor_) {
    LDBG("index: " << anchorAxis.axisIndex
                   << ", max static shape: " << anchorAxis.maxStaticShape);
    LDBG(utils::debugger::to_string(anchorAxis.dynamicShapeIndices));
    for (auto dynIndex : anchorAxis.dynamicShapeIndices) {
      auto [anchorValue, anchorIndex] = getDimension(dynIndex);
      LDBG(anchorValue << " of index " << anchorIndex);
    }
  }
}
#endif

bool DimensionAnalyzer::isDimensionEqualToAnchor(int64_t anchorDimIdx,
                                                 Dimension other,
                                                 bool isStrict) {
  return areDimensionsEqual(getDimension(anchor_[anchorDimIdx].axisIndex),
                            other, isStrict);
}

} // namespace detail
} // namespace hfusion
} // namespace mlir