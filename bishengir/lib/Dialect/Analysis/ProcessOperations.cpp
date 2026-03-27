//===- ProcessOperations.cpp ----------------------------------------------===//
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
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <numeric>
#include <type_traits>

using namespace mlir;
using namespace mlir::utils::debugger;
using namespace mlir::reshape_utils;
using namespace mlir::tensor::reshape_utils;

#define DEBUG_TYPE "dimension-analyzer-hfusion-process-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace detail {

// Step 2: BFS argument list and creating segments
void DimensionAnalyzerBase::processBFS() {
  std::queue<Value> bfsQueue;
  for (const auto &arg : argumentList_) {
    updatePreviousType(arg);
    bfsQueue.push(arg);
  }
  DenseSet<Value> visited(argumentList_.begin(), argumentList_.end());
  combineInferable();

  while (!bfsQueue.empty()) {
    Value current = bfsQueue.front();
    bfsQueue.pop();

    for (Operation *user : current.getUsers()) {
      processOperation(user, current);

      for (Value result : user->getResults()) {
        updatePreviousType(result);
        if (visited.insert(result).second) {
          bfsQueue.push(result);
        }
      }
    }
  }
}

bool DimensionAnalyzerBase::processOperation(Operation *op, Value current) {
  if (auto concatOp = dyn_cast<tensor::ConcatOp>(op)) {
    processConcatOp(concatOp);
  } else if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
    processPadOp(padOp);
  } else if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    processExtractSliceOp(extractSliceOp);
  } else if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
    processInsertSliceOp(insertSliceOp);
  } else {
    if (isContainerAllocator(op)) {
      processParallelOp(op, current);
    } else {
      return false;
    }
  }
  return true;
}

void DimensionAnalyzerBase::processParallelOp(Operation *op, Value current) {
  LDBG("Processing parellel op " << *op);
  createDummyRefIfNotExist({current});
  for (Value v : op->getOperands()) {
    processValue(v, current);
  }
  for (Value v : op->getResults()) {
    processValue(v, current);
  }
}

void DimensionAnalyzerBase::processValue(Value v, Value current) {
  if (v == current)
    return;
  LLVM_DEBUG(llvm::dbgs() << "Trying to bind two values " << v << " " << current
                          << "\n";);
  size_t vRank = utils::getShapeRank(v).value_or(0);
  size_t currentRank = utils::getShapeRank(current).value_or(0);
  // Can actually assert the shape too if the rank is the same
  if (vRank != currentRank)
    return;

  collapsePropagateOrVerify(v, current);
}

size_t
DimensionAnalyzerBase::processDecreasingDimensions(ArrayRef<int64_t> inputArgs,
                                                   ArrayRef<int64_t> dimensions,
                                                   const Value &output) {
  size_t outputRank = utils::getShapeRank(output).value_or(0);
  LLVM_DEBUG(llvm::dbgs() << "\nProcess decreasing dims, output: " << outputRank
                          << "\n");
  assert(inputArgs.size() == outputRank + dimensions.size());
  DimensionIndex outputArgs;
  outputArgs.reserve(outputRank);
  SmallVector<int64_t> sortedDimensions = {dimensions.begin(),
                                           dimensions.end()};
  llvm::sort(sortedDimensions);
  const auto *dimPtr = sortedDimensions.begin();
  for (int64_t i = 0; i < static_cast<int64_t>(inputArgs.size()); ++i) {
    if (dimPtr != sortedDimensions.end() && *dimPtr == i) {
      // Skip this dimension as it's being reduced
      ++dimPtr;
    } else {
      outputArgs.push_back(inputArgs[i]);
      LLVM_DEBUG(llvm::dbgs() << outputArgs.back() << " ";);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "\n";);

  assert(outputArgs.size() == outputRank);
  argumentsRef_.push_back(std::move(outputArgs));
  LLVM_DEBUG(llvm::dbgs() << "New argumentsRef_ " << argumentsRef_.size() - 1
                          << "\n";);
  return argumentsRef_.size() - 1;
}

size_t DimensionAnalyzerBase::processPermutation(ArrayRef<int64_t> inputArgs,
                                                 ArrayRef<int64_t> perm,
                                                 const Value &output) {
  auto [outputRank, shapeOut] = utils::getValueShapeInfo(output).value_or(
      std::make_pair(0, DimensionIndex{}));
  DimensionIndex outputArgs(outputRank, -1);
  for (int i = 0; i < static_cast<int>(inputArgs.size()); ++i) {
    outputArgs[i] = inputArgs[perm[i]];
  }
  argumentsRef_.push_back(std::move(outputArgs));
  return argumentsRef_.size() - 1;
}

void DimensionAnalyzerBase::processMatmulOp(Operation *op, bool isTransposeA,
                                            bool isTransposeB) {
  auto matmulOp = dyn_cast<DestinationStyleOpInterface>(op);

  auto inputs = matmulOp.getDpsInputs();
  assert(matmulOp.getDpsInits().size() == 1);
  Value output = matmulOp.getDpsInits()[0];

  auto arg0 = getArgumentRefOrCreateDummy(inputs[0]);
  auto arg1 = getArgumentRefOrCreateDummy(inputs[1]);

  // When isTransposeA = false, isTransposeB = false:
  // A = [MxK], B = [KxN], C = [MxN]
  // When isTransposeA = true, isTransposeB = false:
  // A = [KxM], B = [KxN], C = [MxN]
  // When isTransposeA = false, isTransposeB = true:
  // A = [MxK], B = [NxK], C = [MxN]
  int mDimIdx = isTransposeA ? 1 : 0;
  int nDimIdx = isTransposeB ? 0 : 1;

  argumentsRef_.push_back({arg0[mDimIdx] /* this refers to the M */,
                           arg1[nDimIdx] /* this refers to the N */});

  initCollapseOrVerify(output, static_cast<int64_t>(argumentsRef_.size() - 1));
  for (Value result : op->getResults()) {
    processValue(result, output);
  }
}

void DimensionAnalyzerBase::processConcatOp(tensor::ConcatOp concatOp) {
  LDBG("Processing concat " << concatOp);
  auto dim = concatOp.getDim();
  auto res = concatOp.getResult();
  auto resultShape = utils::getShape(res.getType());
  auto rank = resultShape.size();
  LDBG(res);
  for (auto opr : concatOp.getOperands()) {
    if (utils::getShapeRank(opr.getType()).value_or(0) != resultShape.size())
      continue;
    auto oprArgRef = getArgumentRefOrCreateDummy(opr);
    auto argConcat = getArgumentRefOrCreateDummy(res);
    LDBG(opr << ": " << utils::debugger::to_string(oprArgRef));
    for (unsigned i = 0; i < rank; ++i) {
      if (i != dim) {
        isConnected_[argConcat[i]].elementKind =
            (resultShape[i] == 1 ? ElementKind::Unit : ElementKind::NoMutation);
        // Shape elem has type inference, can only be joined if the shape is
        // exactly the same
        joinShape(argConcat[i], oprArgRef[i]);
      } else {
        isConnected_[argConcat[i]].elementKind = ElementKind::HasMutation;
        // Can be binded as far as the structure is the same
        joinCollapser(argConcat[i], oprArgRef[i]);
      }
    }
  }
  // No need to disconnect anything, handled by elementKind resolver
}

void DimensionAnalyzerBase::processPadOp(tensor::PadOp padOp) {
  auto padSrc = padOp.getSource();
  auto padHigh = padOp.getStaticHigh();
  auto padLow = padOp.getStaticLow();
  auto padResult = padOp.getResult();
  auto rank = padOp.getType().getRank();
  auto argPadSrc = getArgumentRefOrCreateDummy(padSrc);
  auto argPadResult = getArgumentRefOrCreateDummy(padResult);
  auto srcShape = utils::getShape(padSrc.getType());
  for (int i = 0; i < rank; i++) {
    if (padHigh[i] == 0 && padLow[i] == 0) {
      // Connect;
      joinShape(argPadSrc[i], argPadResult[i]);
      isConnected_[argPadSrc[i]].elementKind =
          srcShape[i] == 1 ? ElementKind::Unit : ElementKind::NoMutation;
      LDBG("From the pad: "
           << static_cast<int64_t>(isConnected_[argPadSrc[i]].elementKind));
    } else {
      joinCollapser(argPadSrc[i], argPadResult[i]);
      isConnected_[argPadSrc[i]].elementKind = ElementKind::HasMutation;
    }
  }
}

template <class T, typename>
void DimensionAnalyzerBase::processSlicingOp(T slicingOp) {
  Value src = slicingOp.getSource();
  Value res = slicingOp.getResult();
  if (std::is_same_v<T, tensor::InsertSliceOp>)
    std::swap(src, res);
  SmallVector<OpFoldResult> srcShape;
  if (auto expandOp = src.template getDefiningOp<tensor::ExpandShapeOp>()) {
    srcShape = expandOp.getMixedOutputShape();
  } else {
    srcShape = llvm::map_to_vector(
        utils::getShape(src.getType()),
        [&slicingOp](int64_t elementShape) -> OpFoldResult {
          return getAsIndexOpFoldResult(slicingOp.getContext(), elementShape);
        });
  }
  auto resShape = slicingOp.getMixedSizes();
  auto droppedDims = slicingOp.getDroppedDims();
  auto rank = srcShape.size();
  // TODO: Process for dynamic
  auto srcRefPtr = getArgumentRefOrCreateDummy(src);
  auto resRefPtr = getArgumentRefOrCreateDummy(res);
  int resPtr = 0;
  for (unsigned i = 0; i < rank; i++) {
    if (droppedDims[i]) {
      // Dropped unit dimensions
      isConnected_[srcRefPtr[i]].elementKind = ElementKind::Unit;
      continue;
    }
    auto staticSrc =
        getConstantIntValue(srcShape[i]).value_or(ShapedType::kDynamic);
    auto staticRes =
        getConstantIntValue(resShape[resPtr]).value_or(ShapedType::kDynamic);
    bool isStaticAndSame =
        staticSrc == staticRes && !ShapedType::isDynamic(staticSrc);
    bool isDynamicAndSame = srcShape[i] == resShape[resPtr];
    if (isDynamicAndSame || isStaticAndSame) {
      LDBG("No Mutation " << i << ' ' << resPtr);
      // No mutation on this one
      isConnected_[srcRefPtr[i]].elementKind =
          (getConstantIntValue(srcShape[i]).value_or(ShapedType::kDynamic) == 1)
              ? ElementKind::Unit
              : ElementKind::NoMutation;
      LDBG("From the extract slice: "
           << static_cast<int64_t>(isConnected_[srcRefPtr[i]].elementKind));
      joinShape(srcRefPtr[i], resRefPtr[resPtr]);
    } else {
      LDBG("Has Mutation " << i << ' ' << resPtr);
      isConnected_[srcRefPtr[i]].elementKind = ElementKind::HasMutation;
      joinCollapser(srcRefPtr[i], resRefPtr[resPtr]);
    }
    // Used the resPtr
    resPtr++;
  }
}

void DimensionAnalyzerBase::processInsertSliceOp(
    tensor::InsertSliceOp insertSliceOp) {
  auto dest = insertSliceOp.getDest();
  auto res = insertSliceOp.getResult();
  createDummyRefIfNotExist({res, dest});
  assert(res.getType().getRank() == dest.getType().getRank());
  collapsePropagateOrVerify(res, dest);
  LDBG("\nProcessing insert slice ----");
  processSlicingOp(insertSliceOp);
}

void DimensionAnalyzerBase::processExtractSliceOp(
    tensor::ExtractSliceOp extractSliceOp) {
  auto res = extractSliceOp.getResult();
  createDummyRefIfNotExist({res});
  LDBG("\nProcessing extract slice ----");
  processSlicingOp(extractSliceOp);
}
} // namespace detail
} // namespace mlir
