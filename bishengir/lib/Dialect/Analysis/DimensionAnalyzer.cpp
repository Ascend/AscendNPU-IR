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
  llvm::report_fatal_error("Element shape index cannot be inferred");
}

SmallVector<int64_t> DimensionAnalyzerBase::getArgumentRef(Value v) const {
  return argumentsRef_[argumentsRefPointer_.at(v)];
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