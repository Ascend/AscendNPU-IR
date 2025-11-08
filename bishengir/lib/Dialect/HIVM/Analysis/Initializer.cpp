//===- Initializer.cpp ----------------------------------------------------===//
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

DimensionAnalyzer::DimensionAnalyzer(Operation *op)
    : DimensionAnalyzerBase(op) {}

LogicalResult DimensionAnalyzer::initialize() {
  auto result = DimensionAnalyzerBase::initialize();
  propagateConnection();
  spreadConnection();
  markDimensionKind();
  return result;
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

  assert(dimensionAllocation_ == argumentList_.size() &&
         "Inconsistency in argumentList_");
}

} // namespace detail
} // namespace hivm
} // namespace mlir