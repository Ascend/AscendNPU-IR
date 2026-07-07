//===- DimensionAnalyzer.cpp ----------------------------------------------===//
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
    if (!valueToDimIndicesIndex_.contains(val))
      return;
    auto vRef = getValueDimIndices(val);
    for (auto el : vRef) {
      auto currentElIdx = equivalentDsu_->find(el);
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
  elemIndex = equivalentDsu_->find(elemIndex);
  if (!reverseShapeElem_.contains(elemIndex))
    return std::nullopt;
  Value val = reverseShapeElem_.at(elemIndex);
  auto vRef = getValueDimIndices(val);
  OpBuilder builder(val.getContext());
  builder.setInsertionPointAfterValue(val);

  // if the mixed size is found, construct a tensor.dim out of it
  // reify propagation will be done outside of this
  for (size_t i = 0; i < vRef.size(); ++i) {
    auto currentElIdx = equivalentDsu_->find(vRef[i]);
    if (currentElIdx == elemIndex) {
      return tensor::getMixedSize(builder, val.getLoc(), val, i);
    }
  }
  llvm_unreachable("Element shape found but cannot be inferred");
}

std::optional<SmallVector<int64_t>>
DimensionAnalyzerBase::getParentShapeRef(Value v) {
  if (!valueToDimIndicesIndex_.contains(v))
    return std::nullopt;

  const auto &vRef = getValueDimIndices(v);
  SmallVector<int64_t> ret(vRef.size());
  for (size_t i = 0; i < ret.size(); ++i) {
    ret[i] = equivalentDsu_->find(vRef[i]);
  }
  return ret;
}

SmallVector<int64_t> DimensionAnalyzerBase::getDimShape(Value v) {
  assert(valueToDimIndicesIndex_.count(v));
  auto vRef = getValueDimIndices(v);
  SmallVector<int64_t> ret(vRef.size());
  for (size_t i = 0; i < ret.size(); ++i) {
    ret[i] = equivalentDsu_->getMinParentAndShapePair(vRef[i]).second;
  }
  return ret;
}

DimensionAnalyzerBase::Dimension
DimensionAnalyzerBase::getEarliestDimension(Dimension dim) {
  auto firstParentIndex = equivalentDsu_->minIndex[equivalentDsu_->find(
      getValueDimIndices(dim.first)[dim.second])];

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
  parentIndex = equivalentDsu_->find(parentIndex);
  auto value = reverseShapeElem_.at(parentIndex);
  auto vRef = getValueDimIndices(value);
  for (size_t i = 0; i < vRef.size(); ++i) {
    auto currentElIdx = equivalentDsu_->find(vRef[i]);
    if (currentElIdx == parentIndex) {
      LDBG(parentIndex << " is mapped to \n" << value << "\n" << i);
      return Dimension(value, i);
    }
  }
  llvm::report_fatal_error("Element shape index cannot be inferred");
}

SmallVector<int64_t> DimensionAnalyzerBase::getArgumentRef(Value v) const {
  return argumentsRef_[argumentsRefPointer_.at(v)];
}

SmallVector<int64_t>
DimensionAnalyzerBase::getArgumentRefOrCreateDummy(Value v) {
  createDummyRefIfNotExist({v});
  return getArgumentRef(v);
  llvm_unreachable("Element shape index cannot be inferred");
}

SmallVector<int64_t> DimensionAnalyzerBase::getValueDimIndices(Value v) const {
  auto it = valueToDimIndicesIndex_.find(v);
  if (it == valueToDimIndicesIndex_.end()) {
    LDBG("Warning: Value not found in valueToDimIndicesIndex_: " << v << "\n");
    LDBG("Address of v: " << &v << "\n");
    return SmallVector<int64_t>();
  }
  return dimIndices_[it->second];
}

SmallVector<int64_t>
DimensionAnalyzerBase::getValueDimIndicesOrCreateDummy(Value v) {
  createDummyRefIfNotExist({v});
  return getValueDimIndices(v);
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
  createDummyRefIfNotExist({lhs.first, rhs.first});
  auto lhsRef = getValueDimIndices(lhs.first);
  auto rhsRef = getValueDimIndices(rhs.first);
  if (isStrict)
    return equivalentDsu_->find(lhsRef[lhs.second]) ==
           equivalentDsu_->find(rhsRef[rhs.second]);

  return structuralDsu_->find(lhsRef[lhs.second]) ==
         structuralDsu_->find(rhsRef[rhs.second]);
}

bool DimensionAnalyzerBase::hasValueRef(Value v) const {
  return valueToDimIndicesIndex_.contains(v);
}

void DimensionAnalyzerBase::joinValueGroup(Value a, Value b) {
  if (!isAllowedType(a.getType()) || !isAllowedType(b.getType()))
    return;
  createDummyRefIfNotExist({a, b});
  assert(valueGroupDSU_ && "valueGroupDSU_ must be initialized");
  valueGroupDSU_->join(valueToDimIndicesIndex_.at(a),
                       valueToDimIndicesIndex_.at(b));
}

int64_t DimensionAnalyzerBase::getValueGroupIndex(Value v) {
  if (!isAllowedType(v.getType()))
    return -1;
  createDummyRefIfNotExist({v});
  assert(valueGroupDSU_ && "valueGroupDSU_ must be initialized");
  return valueGroupDSU_->find(valueToDimIndicesIndex_.at(v));
}

} // namespace detail
} // namespace mlir