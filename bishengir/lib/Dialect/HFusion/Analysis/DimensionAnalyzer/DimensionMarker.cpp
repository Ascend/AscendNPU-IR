//===- DimensionMarker.cpp ------------------------------------------------===//
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include <numeric>

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;
using namespace mlir::tensor::reshape_utils;

#define DEBUG_TYPE "dimension-analyzer-marker"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hfusion {
namespace detail {

bool DimensionAnalyzer::processOperation(Operation *op, Value current) {
  LLVM_DEBUG(llvm::dbgs() << "Processing operation: " << *op << "\nwith value "
                          << current << "\n");
  if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(op)) {
    processBroadcastOp(broadcastOp);
  } else if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op)) {
    processReduceLikeOp(reduceOp);
  } else if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op)) {
    processTransposeOp(transposeOp);
  } else if (bool isMatmul = isa<linalg::MatmulOp>(op),
             isTransposeA = isa<linalg::MatmulTransposeAOp>(op),
             isTransposeB = isa<linalg::MatmulTransposeBOp>(op);
             isMatmul || isTransposeA || isTransposeB) {
    processMatmulOp(op, isTransposeA, isTransposeB);
  } else if (auto reduceWithIndexOp =
                 dyn_cast<hfusion::ReduceWithIndexOp>(op)) {
    processReduceLikeOp(reduceWithIndexOp);
  } else if (auto gatherOp = dyn_cast<hfusion::GatherOp>(op)) {
    processGatherOp(gatherOp);
  } else if (auto interleaveOp = dyn_cast<hfusion::InterleaveOp>(op)) {
    processInterleaveOp(interleaveOp);
  } else if (auto deinterleaveOp = dyn_cast<hfusion::DeinterleaveOp>(op)) {
    processDeinterleaveOp(deinterleaveOp);
  } else if (auto cumsumOp = dyn_cast<hfusion::CumsumOp>(op)) {
    processCumOp(cumsumOp);
  } else if (auto cumprodOp = dyn_cast<hfusion::CumprodOp>(op)) {
    processCumOp(cumprodOp);
  } else {
    if (isAllParallelOp(op)) {
      processParallelOp(op, current);
    } else {
      if (!DimensionAnalyzerBase::processOperation(op, current)) {
        LLVM_DEBUG(llvm::dbgs()
                       << "Warning: operation is unchecked " << *op << "\n";);
        return false;
      }
    }
  }
  return true;
}

void DimensionAnalyzer::processBroadcastOp(linalg::BroadcastOp op) {
  Value input = op.getInput();
  Value output = op.getInit();

  createDummyRefIfNotExist({output});
  assert(argumentsRefPointer_.contains(output));
  const auto &outputArgs = getArgumentRef(output);
  auto newValRef =
      processDecreasingDimensions(outputArgs, op.getDimensions(), input);
  initCollapseOrVerify(input, newValRef);

  for (Value result : op->getResults()) {
    processValue(result, output);
  }
}

template <class T, typename>
void DimensionAnalyzer::processReduceLikeOp(T reduceOp) {
  ArrayRef<int64_t> reduceDims = reduceOp.getDimensions();
  auto dpsOp = cast<DestinationStyleOpInterface>(reduceOp.getOperation());
  SmallVector<Value> inputs = dpsOp.getDpsInputs();
  SmallVector<Value> outputs = dpsOp.getDpsInits();
  auto &pivotInput = inputs[0];
  auto &pivotOutput = outputs[0];
  createDummyRefIfNotExist({pivotInput});
  assert(argumentsRefPointer_.contains(pivotInput));
  const auto refPtr = argumentsRefPointer_.at(pivotInput);
  for (Value input : inputs) {
    initCollapseOrVerify(input, refPtr);
  }

  // Connect input and output
  assert(argumentsRefPointer_.contains(pivotInput));
  auto inputArgs = getArgumentRefOrCreateDummy(pivotInput);
  auto newValRef =
      processDecreasingDimensions(inputArgs, reduceDims, pivotOutput);
  initCollapseOrVerify(pivotOutput, newValRef);

  const auto refPtrOut = argumentsRefPointer_.at(pivotOutput);
  for (Value output : outputs) {
    initCollapseOrVerify(output, refPtrOut);
  }
  for (Value result : reduceOp.getResults()) {
    processValue(result, pivotOutput);
  }
  reduceOp.walk([&](linalg::IndexOp indexOp) {
    const auto accessedIdx = indexOp.getDim();
    LDBG(accessedIdx);
    const auto &inputArgs = getArgumentRef(pivotInput);
    LDBG(inputArgs.size());
    if (accessedIdx - 1 >= 0)
      disconnect(inputArgs[accessedIdx - 1], inputArgs[accessedIdx]);
    LDBG("Disconnect with left");
    if (accessedIdx + 1 < inputArgs.size())
      disconnect(inputArgs[accessedIdx], inputArgs[accessedIdx + 1]);
    LDBG("Disconnect with right");
  });
}

void DimensionAnalyzer::processTransposeOp(linalg::TransposeOp op) {
  Value input = op.getInput();
  Value output = op.getInit();
  auto perm = op.getPermutation();
  const auto &inputArgs = getArgumentRefOrCreateDummy(input);
  auto newValRef = processPermutation(inputArgs, perm, output);
  initCollapseOrVerify(output, newValRef);
  for (Value result : op->getResults()) {
    processValue(result, output);
  }
}

template <class T>
void DimensionAnalyzer::processCumOp(T cumOp) {
  auto input = cumOp.getInput();
  auto inputRef = getArgumentRefOrCreateDummy(input);
  auto res = cumOp.getResult();
  auto resRef = getArgumentRefOrCreateDummy(res);
  auto resultShape = utils::getShape(res.getType());
  auto rank = resultShape.size();
  BitVector cumMask(rank);
  for (auto &cumDim : cumOp.getCumDims())
    cumMask.set(cumDim);
  for (unsigned i = 0; i < rank; ++i) {
    if (!cumMask[i]) {
      isConnected_[resRef[i]].elementKind =
          (resultShape[i] == 1 ? ElementKind::Unit : ElementKind::NoMutation);
      joinShape(resRef[i], inputRef[i]);
    } else {
      isConnected_[resRef[i]].elementKind = ElementKind::HasMutation;
      joinCollapser(resRef[i], inputRef[i]);
    }
  }
}

void DimensionAnalyzer::processGatherOp(hfusion::GatherOp gatherOp) {
  auto axis = gatherOp.getAxis();
  auto resVariadic = gatherOp.getResult();
  Value res;
  if (resVariadic.empty())
    res = gatherOp.getDpsInitOperand(0)->get();
  else
    res = gatherOp.getResult().front();

  auto resultShape = utils::getShape(res.getType());
  auto rank = resultShape.size();

  for (auto opr : gatherOp.getOperands()) {
    if (utils::getShapeRank(opr.getType()).value_or(0) != resultShape.size())
      continue;
    createDummyRefIfNotExist({opr, res});
    auto oprArgRef = getArgumentRef(opr);
    auto gatherResRef = getArgumentRef(res);
    for (unsigned i = 0; i < rank; i++) {
      if (i != axis) {
        isConnected_[gatherResRef[i]].elementKind =
            (resultShape[i] == 1 ? ElementKind::Unit : ElementKind::NoMutation);
        joinShape(gatherResRef[i], oprArgRef[i]);
      } else {
        isConnected_[gatherResRef[i]].elementKind = ElementKind::HasMutation;
        joinCollapser(gatherResRef[i], oprArgRef[i]);
      }
    }
  }
}

void DimensionAnalyzer::processInterleaveOp(
    hfusion::InterleaveOp interleaveOp) {
  auto res = interleaveOp.getResult();
  auto resultShape = utils::getShape(res.getType());
  const auto rank = static_cast<int>(resultShape.size());
  auto firstOperand = interleaveOp.getOperand(0);
  createDummyRefIfNotExist(SmallVector<Value>(interleaveOp->getOperands()));
  if (rank < 1)
    return;
  const int lastIdx = rank - 1;
  for (auto opr : interleaveOp.getOperands()) {
    if (utils::getShapeRank(opr.getType()).value_or(0) != resultShape.size())
      continue;
    auto oprRef = getArgumentRefOrCreateDummy(opr);
    auto resRef = getArgumentRefOrCreateDummy(res);
    for (int i = 0; i < lastIdx; ++i) {
      isConnected_[resRef[i]].elementKind =
          (resultShape[i] == 1 ? ElementKind::Unit : ElementKind::NoMutation);
      // Shape elem has type inference, can only be joined if the shape is
      // exactly the same
      joinShape(resRef[i], oprRef[i]);
    }
    // Last element is a mutation kind
    auto firstOperandRef = getArgumentRef(firstOperand);
    isConnected_[resRef[lastIdx]].elementKind = ElementKind::NoMutation;
    // Bind shape with each other
    joinShape(firstOperandRef[lastIdx], oprRef[lastIdx]);
    // Bind structure with res
    joinCollapser(resRef[lastIdx], oprRef[lastIdx]);
  }
  // No need to disconnect anything, handled by elementKind resolver
}

void DimensionAnalyzer::processDeinterleaveOp(
    hfusion::DeinterleaveOp deinterleaveOp) {
  SmallVector<Value> results = deinterleaveOp.getResults();
  createDummyRefIfNotExist(results);
  auto resultShape = utils::getShape(results[0].getType());
  const auto rank = static_cast<int>(resultShape.size());
  auto src = deinterleaveOp.getInput();
  auto oprRef = getArgumentRefOrCreateDummy(src);
  if (rank < 1)
    return;
  const int lastIdx = rank - 1;
  for (auto res : results) {
    auto resRef = getArgumentRefOrCreateDummy(res);
    for (int i = 0; i < lastIdx; ++i) {
      isConnected_[resRef[i]].elementKind =
          (resultShape[i] == 1 ? ElementKind::Unit : ElementKind::NoMutation);
      // Shape elem has type inference, can only be joined if the shape is
      // exactly the same
      joinShape(resRef[i], oprRef[i]);
    }
    // Last element is a mutation kind
    auto firstResRef = getArgumentRef(results[0]);
    isConnected_[resRef[lastIdx]].elementKind = ElementKind::NoMutation;
    // Bind shape with each other
    joinShape(firstResRef[lastIdx], resRef[lastIdx]);
    // Bind structure with opr, but not shape
    joinCollapser(resRef[lastIdx], oprRef[lastIdx]);
  }
  // No need to disconnect anything, handled by elementKind resolver
}

} // namespace detail
} // namespace hfusion
} // namespace mlir