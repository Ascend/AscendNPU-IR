//===- Initializer.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include <numeric>

using namespace mlir;
using namespace mlir::utils::debugger;

#define DEBUG_TYPE "dimension-analyzer-initialize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
namespace mlir {
namespace detail {

DimensionAnalyzerBase::DimensionAnalyzerBase(Operation *op,
                                             DimensionAnalyzerOptions options)
    : op_(op), options(options) {}

LogicalResult DimensionAnalyzerBase::initialize() {
  // Check if tensor::ExpandShapeOp exists in the function
  bool hasReshaping = false;
  bool hasFunctionCall = false;
  op_->walk([&](Operation *op) {
    if (isa<func::CallOp>(op)) {
      hasFunctionCall = true;
    }
    if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::ReshapeOp>(
            op)) {
      hasReshaping = true;
    }
    return WalkResult::advance();
  });
  if (hasReshaping) {
    LLVM_DEBUG(llvm::dbgs() << "Will try to optimize scoped reshape\n";);
  }
  if (hasFunctionCall) {
    LLVM_DEBUG(llvm::dbgs()
                   << "Skipping function with function call inside\n";);
    return failure();
  }
  initializeStructures();
  processBFS();
  unifyGroups();
  // markAttributes() can be called here to check the connections
  return success();
}

int64_t
DimensionAnalyzerBase::allocateArguments(int rank,
                                         ArrayRef<int64_t> dimensionRef) {
  LLVM_DEBUG(
    llvm::dbgs() << "Allocating new arguments with rank = " << rank << "\n";
  );
  auto startingIdx = argumentTotalLength_;
  argumentTotalLength_ += rank + 1;
  isConnected_.resize(argumentTotalLength_);
  solverShapeElem_->allocateMinimum(argumentTotalLength_);
  solverCollapserElem_->allocateMinimum(argumentTotalLength_);
  assert(rank == ssize_t(dimensionRef.size()));
  LLVM_DEBUG(
    llvm::dbgs() << "dimensionAllocation_ = " << dimensionAllocation_ << "\n";
  );
  for (int64_t i = 0; i < rank; ++i) {
    LLVM_DEBUG(
      llvm::dbgs() << "Allocating axis(" << i << ") with dimSiz = "
                   << dimensionRef[i] << "\n";
    );
    int64_t currentIndex = startingIdx + i;
    solverShapeElem_->minParentIndex_[currentIndex] = {dimensionAllocation_, i};
    solverShapeElem_->shape_[currentIndex] = dimensionRef[i];
    isConnected_[currentIndex].elementKind =
        dimensionRef[i] == 1 ? tensor::reshape_utils::ElementKind::Unit
                             : tensor::reshape_utils::ElementKind::NoMutation;
    if (i > 0)
      isConnected_[currentIndex].leftConnected = true;
    if (i + 1 < rank)
      isConnected_[currentIndex].rightConnected = true;
  }
  dimensionAllocation_++;

  return startingIdx;
}

bool DimensionAnalyzerBase::isAllowedType(Type type) {
  if (options.registerBased) {
    return isa_and_present<ShapedType>(type);
  }
  return isa_and_present<RankedTensorType>(type);
}

bool DimensionAnalyzerBase::isHeadOperation(Operation *op) {
  if (options.registerBased) {
    return reshape_utils::isReshapingOp(op) || reshape_utils::isInitOp(op) ||
           isa_and_present<memref::AllocaOp, memref::AllocOp,
                           memref::ReinterpretCastOp, memref::CastOp,
                           arith::ConstantOp, memref::ExpandShapeOp,
                           memref::CollapseShapeOp>(op);
  }
  return reshape_utils::isArgOp(op);
}

bool DimensionAnalyzerBase::isTailOperation(Operation *op) {
  if (options.registerBased) {
    return reshape_utils::isReshapingOp(op) ||
           isa_and_present<memref::ExpandShapeOp, memref::CollapseShapeOp,
                           func::ReturnOp>(op);
  }
  return reshape_utils::isOutOp(op);
}

// Step 1: Initializing arguments segments
void DimensionAnalyzerBase::initializeStructures() {
  solverShapeElem_ = std::make_unique<ExtendedUnionFind>();
  solverCollapserElem_ = std::make_unique<SimpleUnionFind>();
  solverSegments_ = std::make_unique<SimpleUnionFind>();

  size_t sizeCount = 0;
  for (Block &block : op_->getRegion(0)) {
    LLVM_DEBUG(llvm::dbgs() << "Processing Block\n");
    sizeCount += block.getOperations().size();

    // FLATTEN-IN
    // Process block arguments
    for (BlockArgument arg : block.getArguments()) {
      if (DimensionAnalyzerBase::isAllowedType(arg.getType())) {
        processArgument(arg);
      }
    }

    // Process args of some knowing operations as an opener
    // operations
    block.walk([&](Operation *op) {
      if (DimensionAnalyzerBase::isHeadOperation(op)) {
        for (auto result : op->getResults()) {
          if (DimensionAnalyzerBase::isAllowedType(result.getType())) {
            LLVM_DEBUG(llvm::dbgs() << "Putting " << result << " in arguments "
                                    << "\n";);
            processArgument(result);
          }
        }
      }
    });
    block.walk([&](Operation *op) {
      if (DimensionAnalyzerBase::isTailOperation(op)) {
        outList_.push_back(op);
      }
    });
  }

  LLVM_DEBUG(llvm::dbgs() << "Initializing structures sizeCount: " << sizeCount
                          << "\n");
  solverSegments_->allocateMinimum(sizeCount);
  assert(dimensionAllocation_ == ssize_t(argumentList_.size()) &&
         "Inconsistency in argumentList_");
  LLVM_DEBUG(
    llvm::dbgs() << DEBUG_LINE_BEG("Flatten-After-initializeStructures");
    llvm::dbgs() << "solverSegments_:\n";
    solverSegments_->dump();
    dumpArgumentsRefPointer();
    dumpArgumentsRef();
    dumpIsConnected();
    llvm::dbgs() << DEBUG_LINE_END("Flatten-After-initializeStructures");
  );
}

void DimensionAnalyzerBase::processArgument(Value arg) {
  argumentList_.push_back(arg);

  auto [rank, shape] = utils::getValueShapeInfo(arg).value_or(
      std::make_pair(0, DimensionShape{}));
  // Add size for space as well
  LLVM_DEBUG(llvm::dbgs() << "Found args: " << arg << ' ' << rank << "\n");
  auto startingIdx = allocateArguments(rank, shape);
  initCollapseOrVerify(arg, argumentsRef_.size());
  argumentsRef_.push_back(DimensionShape(shape));
  std::iota(argumentsRef_.back().begin(), argumentsRef_.back().end(),
            startingIdx);
#ifndef NDEBUG
  for (auto val : argumentsRef_.back()) {
    LDBG(val);
  }
#endif
  LLVM_DEBUG(llvm::dbgs() << utils::debugger::to_string(argumentsRef_.back())
                          << '\n');

  // args:
  // [2x4xf32]_[5xf32 ]_[8x7x6xf32]_
  //  0 1     2 3      4 5 6 7     8
  //
  //  2, 4 and 8 are spacing,
  //  each argument shape is assigned with an index
  //
  //  argumentsRefPointer_ : {arg0 : 0, arg1 : 1, arg2 : 2}
  //  argumentsRef_ : {{0,1}, {3}, {5,6,7}}
  //
  //  Broadcasting new elements will also increase the arguments ref,
  //  and create a new arguments ref pointer index
}

void DimensionAnalyzerBase::markAttributes() {
  LLVM_DEBUG(llvm::dbgs() << "Marking attributes\n");

  op_->walk([&](Operation *op) {
    // Process each result of the operation
    for (auto result : op->getResults()) {
      // Check if this result has an argument reference
      if (!argumentsRefPointer_.count(result))
        continue;
      auto argRef = getArgumentRef(result);

      // Build the attribute array with parent indices from solverCollapserElem
      SmallVector<int64_t> parentIndices;
      for (int64_t idx : argRef) {
        int64_t parentIdx = solverCollapserElem_->find(idx);
        parentIndices.push_back(parentIdx);
      }

      // Set the attribute on the operation result
      if (!parentIndices.empty()) {
        auto attrName = ("result_" + Twine(result.getResultNumber())).str();
        op->setAttr(attrName,
                    Builder(op->getContext()).getI64ArrayAttr(parentIndices));
      }
    }

    return WalkResult::advance();
  });

  LLVM_DEBUG(llvm::dbgs() << "Finished marking attributes\n");
}
} // namespace detail
} // namespace mlir
