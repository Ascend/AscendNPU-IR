//===- Initializer.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::utils::debugger;

#define DEBUG_TYPE "dimension-analyzer-initialize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hivm {
namespace detail {

DimensionAnalyzer::DimensionAnalyzer(Operation *op, int64_t tilingSize)
    : DimensionAnalyzerBase(op), tilingSize(tilingSize) {
  assert(tilingSize > 0 && "Tiling Size must be positive integer");
}

LogicalResult DimensionAnalyzer::initialize() {
  if (failed(DimensionAnalyzerBase::initialize()))
    return failure();
  propagateConnection();
  spreadConnection();
  markDimensions();
  transferDimMark();
  for (auto &[rep, indices] : invalidUpdates) {
    rep = structuralDsu_->find(rep);
    for (auto &idx : indices)
      idx = structuralDsu_->find(idx);
    llvm::sort(indices);
  }
  llvm::sort(invalidUpdates);
  invalidUpdates.erase(llvm::unique(invalidUpdates), invalidUpdates.end());
  LDBG("InvalidUpdates: ");
  for (auto &[rep, indices] : invalidUpdates) {
    LLVM_DEBUG({
      llvm::dbgs() << rep << ' ' << utils::debugger::to_string(indices) << '\n';
    });
  }
  return success();
}

void DimensionAnalyzer::initializeStructures() {
  DimensionAnalyzerBase::initializeStructures();
  for (Block &block : op_->getRegion(0)) {
    LLVM_DEBUG(llvm::dbgs() << "Processing Block\n");

    // FLATTEN-IN
    // Process block arguments
    for (BlockArgument arg : block.getArguments()) {
      if (isa<MemRefType>(arg.getType())) {
        processArgument(arg);
      }
    }

    // Process args of some knowing operations as an opener
    // operations
    block.walk([&](Operation *op) {
      if (isa<memref::AllocOp, memref::AllocaOp>(op)) {
        Value result = op->getResult(0);
        if (isa<MemRefType>(result.getType())) {
          LLVM_DEBUG(llvm::dbgs() << "Putting " << result << " in arguments "
                                  << "\n";);
          processArgument(result);
        }
      }
    });
    block.walk([&](Operation *op) {
      if (isa<hivm::StoreOp>(op)) {
        outList_.push_back(op);
      }
    });
  }

  assert(dimensionAllocation_ == ssize_t(argumentList_.size()) &&
         "Inconsistency in argumentList_");
}

int64_t DimensionAnalyzer::allocateArguments(int rank,
                                             ArrayRef<int64_t> dimensionRef) {
  auto startingIdx = argumentTotalLength_;
  DimensionAnalyzerBase::allocateArguments(rank, dimensionRef);
  exclusiveDimIdx.resize(argumentTotalLength_);
  for (int64_t i = 0; i < rank; ++i) {
    int64_t currentIndex = startingIdx + i;
    for (int64_t j = 0; j < rank; ++j) {
      if (i == j)
        continue;
      int64_t exclusiveIndex = startingIdx + j;
      exclusiveDimIdx[currentIndex].insert(exclusiveIndex);
    }
  }
  return startingIdx;
}

} // namespace detail
} // namespace hivm
} // namespace mlir