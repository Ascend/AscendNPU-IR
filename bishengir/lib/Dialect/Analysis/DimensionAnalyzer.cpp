//===- DimensionAnalyzer.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/Utils/Util.h"

using namespace mlir;
using namespace mlir::utils::debugger;

#define DEBUG_TYPE "dimension-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace detail {

void DimensionAnalyzerBase::computeReverseElementMap() {
  LDBG("Computing Reverse Element Map");
  auto markShapes = [this](Value val) -> void {
    if (!argumentsRefPointer_.contains(val))
      return;
    auto vRef = getArgumentRef(val);
    for (auto el : vRef) {
      auto currentElIdx = solverShapeElem_->find(el);
      if (!reverseShapeElem_.contains(currentElIdx)) {
        LDBG("Found new elem: " << currentElIdx << ' ' << val);
        reverseShapeElem_[currentElIdx] = val;
      }
    }
  };
  for (auto arg : argumentList_) {
    markShapes(arg);
  }
  for (Block &block : op_->getRegion(0)) {
    block.walk([&markShapes](Operation *op) {
      for (auto res : op->getResults()) {
        markShapes(res);
      }
    });
  }
}

std::optional<OpFoldResult>
DimensionAnalyzerBase::getElementShape(int elemIndex) {
  if (reverseShapeElem_.empty())
    computeReverseElementMap();
  elemIndex = solverShapeElem_->find(elemIndex);
  if (!reverseShapeElem_.contains(elemIndex))
    return std::nullopt;
  Value val = reverseShapeElem_.at(elemIndex);
  auto vRef = getArgumentRef(val);
  OpBuilder builder(val.getContext());
  builder.setInsertionPointAfterValue(val);

  // if the mixed size is found, construct a tensor.dim out of it
  // reify propagation will be done outside of this
  for (size_t i = 0; i < vRef.size(); ++i) {
    auto currentElIdx = solverShapeElem_->find(vRef[i]);
    if (currentElIdx == elemIndex) {
      return tensor::getMixedSize(builder, val.getLoc(), val, i);
    }
  }
  llvm_unreachable("Element shape found but cannot be inferred");
}

std::optional<SmallVector<int64_t>>
DimensionAnalyzerBase::getParentShapeRef(Value v) {
  if (!argumentsRefPointer_.contains(v))
    return std::nullopt;

  const auto &vRef = getArgumentRef(v);
  SmallVector<int64_t> ret(vRef.size());
  for (size_t i = 0; i < ret.size(); ++i) {
    ret[i] = solverShapeElem_->find(vRef[i]);
  }
  return ret;
}

SmallVector<int64_t> DimensionAnalyzerBase::getDimShape(Value v) {
  assert(argumentsRefPointer_.count(v));
  auto vRef = getArgumentRef(v);
  SmallVector<int64_t> ret(vRef.size());
  for (size_t i = 0; i < ret.size(); ++i) {
    ret[i] = solverShapeElem_->getMinParentAndShapePair(vRef[i]).second;
  }
  return ret;
}

DimensionAnalyzerBase::Dimension
DimensionAnalyzerBase::getEarliestDimension(Dimension dim) {
  auto firstParentIndex = solverShapeElem_->minIndex[solverShapeElem_->find(
      getArgumentRef(dim.first)[dim.second])];

  return getDimension(firstParentIndex);
}

DimensionAnalyzerBase::Dimension
DimensionAnalyzerBase::getDimension(int64_t parentIndex) {
  if (reverseShapeElem_.empty())
    computeReverseElementMap();
  parentIndex = solverShapeElem_->find(parentIndex);
  auto value = reverseShapeElem_.at(parentIndex);
  auto vRef = getArgumentRef(value);
  for (size_t i = 0; i < vRef.size(); ++i) {
    auto currentElIdx = solverShapeElem_->find(vRef[i]);
    if (currentElIdx == parentIndex) {
      LDBG(parentIndex << " is mapped to \n" << value << "\n" << i);
      return Dimension(value, i);
    }
  }
  llvm_unreachable("Element shape index cannot be inferred");
}

SmallVector<int64_t> DimensionAnalyzerBase::getArgumentRef(Value v) const {
  auto it = argumentsRefPointer_.find(v);
  if (it == argumentsRefPointer_.end()) {
    LDBG("Warning: Argument Value not found in argumentsRefPointer_: " << v << "\n");
    LDBG("Address of v: " << &v << "\n");
    return SmallVector<int64_t>();
  }
  return argumentsRef_[it->second];
}

SmallVector<int64_t>
DimensionAnalyzerBase::getArgumentRefOrCreateDummy(Value v) {
  createDummyRefIfNotExist({v});
  return getArgumentRef(v);
}

bool DimensionAnalyzerBase::areDimensionsEqual(Dimension lhs, Dimension rhs,
                                               bool isStrict) {
  LDBG("Checking common axis between "
       << lhs.first << " axis " << lhs.second << "and " << rhs.first << " axis "
       << lhs.second << (isStrict ? " are strictly " : " are structurally ")
       << "equal");
  auto lhsRef = getArgumentRefOrCreateDummy(lhs.first);
  auto rhsRef = getArgumentRefOrCreateDummy(rhs.first);
  if (isStrict)
    return solverShapeElem_->find(lhsRef[lhs.second]) ==
           solverShapeElem_->find(rhsRef[rhs.second]);

  return solverCollapserElem_->find(lhsRef[lhs.second]) ==
         solverCollapserElem_->find(rhsRef[rhs.second]);
}

} // namespace detail
} // namespace mlir