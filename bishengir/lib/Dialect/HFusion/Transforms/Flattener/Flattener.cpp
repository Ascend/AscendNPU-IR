//===- Flattener.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Flattener/Flattener.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::tensor::reshape_utils;
using namespace mlir::utils::debugger;

#define DEBUG_TYPE "flattener"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hfusion {
namespace detail {

// Flattener constructor
Flattener::Flattener(Operation *op,
                     mlir::detail::DimensionAnalyzerOptions options)
    : DimensionAnalyzer(op, options) {
  // Step 1: Initialize invariant, each operations has its connection with
  // arguments Thus we know which one to flatten, each dimension in the
  // argument is assigned With a certain number
  //
  // Each existing value, including the the ins (inputs), the outs (outputs),
  // the results can be represented as a tensor shape
  //
  // Tensor shapes of values can be inferred from the arguments,
  // thus we need to bfs for each arguments, its relationships with all values
  //
  // For example args:
  //
  //
  // argname: arg0      arg1     arg2
  // shape:   [7x6xf32]_[5xf32 ]_[8x7x6x9xf32]_
  // index:    0 1     2 3      4 5 6 7 8     9
  //
  // arg0 -> [0, 1]
  // arg1 -> [3]
  // arg2 -> [5, 6, 7, 8]
  // op1  -> [6, 7]
  // op2  -> [0, 1]
  //
  // if there is
  // op1 -> reduce arg2 -> [6, 7]
  // op2 -> add(arg0, op1);
  //
  // thus [0, 1] == [6, 7] will be a gang
  //
  // thus we're representing each shape element with a union find
  // solverShapeElem_, solverShapeElem_.join(0, 6) later will be run
  //
  // solverShapeElem_ however represents the segments of joined segments
  // in this case, there are 4, op2 can inherit either from arg0 or op1
  // every reduce and broadcast will create new segments representation
  //
  // for large shapes can actually use set data structures, to store
  // continuous segments but might be overkill

  // Disable tensor.dim binding to avoid strict flattening inter region
  // e.g: Region A has <?xf32>
  // Region B has <20x?x32xf32>
  // Binding both question mark would cause region B to not bind at all
  bindUsingTensorDim = false;
}

LogicalResult Flattener::flatten(bool multiDynamicShape) {
  // Check if tensor::ExpandShapeOp exists in the function
  auto result = initialize();
  if (failed(result))
    return success();
  for (const auto &ref : argumentsRef_) {
    markBroken(ref);
  }
  dumpIsConnected();
  propagateBroken();
  dumpIsConnected();
  if (!multiDynamicShape) {
    breakDynamicShapes();
    dumpIsConnected();
    propagateBroken();
    dumpIsConnected();
  }
  adjustOperations();
  return success();
}

/// This is marking broken based on shape inference
void Flattener::markBroken(const DimensionIndex &args) {
  int argSize = static_cast<int64_t>(args.size());
  for (int i = 0; i < argSize; ++i) {
    if ((i == 0 && !options.registerBased) ||
        (i > 0 && args[i - 1] != args[i] - 1)) {
      LLVM_DEBUG(llvm::dbgs() << "left of " << args[i] << " is disconnected\n");
      isConnected_[args[i]].leftConnected = false;
      if (i >= 1)
        isConnected_[args[i - 1]].rightConnected = false;
      if (args[i] >= 1)
        isConnected_[args[i] - 1].rightConnected = false;
    }
    if ((i + 1 == static_cast<int>(args.size()) && !options.registerBased) ||
        (i + 1 < static_cast<int>(args.size()) && args[i + 1] != args[i] + 1)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "right of " << args[i] << " is disconnected\n");
      isConnected_[args[i]].rightConnected = false;
      if (i + 1 < static_cast<int>(args.size()))
        isConnected_[args[i + 1]].leftConnected = false;
      if (args[i] + 1 < static_cast<int>(isConnected_.size()))
        isConnected_[args[i] + 1].leftConnected = false;
    }
  }
}

bool Flattener::computeMutation(int pos, int dir) const {
  LDBG("Computing mutation " << pos << " " << dir);
  bool canConnect = true;
  while (pos + dir >= 0 && pos + dir < ssize_t(isConnected_.size())) {
    // H U | U .. U U N
    // It checks for barrier stop or changing stop, if changing stop, it
    // will be merged with other mutation If all unit until end of barrier,
    // then its safe
    bool canProceed = isConnected_[std::min(pos + dir, pos)].rightConnected &&
                      isConnected_[std::max(pos + dir, pos)].leftConnected;
    LDBG("Can proceed to " << pos << " " << pos + dir << " ?" << canProceed << "Can proceed");
    if (!canProceed)
      break;
    if (isConnected_[pos + dir].elementKind != ElementKind::Unit) {
      LDBG(pos + dir << " is not unit "
                     << static_cast<int64_t>(
                            isConnected_[pos + dir].elementKind));
      canConnect = false;
      break;
    }
    pos += dir;
  }
  LDBG("Ok can connect ? " << canConnect);
  return canConnect;
}

// Step 4: Propagate broken nodes
void Flattener::propagateBroken() {
  propagateConnection();
  spreadConnection();
  // Attack all left and right
  for (int i = 0; i < argumentTotalLength_; ++i) {
    int parent = solverCollapserElem_->find(i);
    if (parent != i)
      continue;
    if (isConnected_[i].elementKind != ElementKind::HasMutation)
      continue;
    if (!computeMutation(i, -1)) {
      disconnect(i - 1, i);
    }
    if (!computeMutation(i, 1)) {
      disconnect(i, i + 1);
    }
  }
  propagateConnection();
  spreadConnection();
}

// Step 4.5: Special case to handle dynamics
void Flattener::breakDynamicShapes() {
  BitVector computed(argumentTotalLength_);
  auto markComputed = [&computed, this](int pos) -> void {
    computed[solverCollapserElem_->find(pos)] = true;
  };

  int rightBoundary;
  for (int leftBoundary = 0; leftBoundary < argumentTotalLength_;
       leftBoundary = rightBoundary + 1) {
    markComputed(leftBoundary);
    LDBG("Found left boundary here " << leftBoundary);
    assert(leftBoundary == 0 || !isConnected(leftBoundary - 1, leftBoundary));
    SmallVector<int> dynamicPosition;
    auto getAndAddIfDynamic = [&dynamicPosition, this](int pos) -> void {
      auto currentParShapePair =
          solverShapeElem_->getMinParentAndShapePair(pos);
      if (ShapedType::isDynamic(currentParShapePair.second)) {
        dynamicPosition.push_back(pos);
      }
    };
    rightBoundary = leftBoundary;
    getAndAddIfDynamic(rightBoundary);
    while (rightBoundary + 1 < argumentTotalLength_ &&
           isConnected(rightBoundary, rightBoundary + 1)) {
      rightBoundary++;
      // Check rightBoundary, if you assert isComputed, it can be false
      // Consecutive segments might be part of another subsegment
      markComputed(rightBoundary);
      getAndAddIfDynamic(rightBoundary);
    }
    // Trying to flatten more than one dynamic dimension
    if (dynamicPosition.size() > 1) {
      LDBG("Trying to flatten more than one dynamic dimension");
      for (size_t i = 1; i < dynamicPosition.size(); ++i) {
        disconnect(dynamicPosition[i] - 1, dynamicPosition[i]);
      }
    }
  }
}

} // namespace detail
} // namespace hfusion
} // namespace mlir