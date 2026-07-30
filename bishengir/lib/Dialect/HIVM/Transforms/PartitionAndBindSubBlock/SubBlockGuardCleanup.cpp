//--------------------------SubBlockGuardCleanup.cpp--------------------------//
//
// Cleanup pass: forwards redundant guard copies, then makes guards result-free.
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

#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/SubBlockGuardCleanup.h"
#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/PartitionTypes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_SUBBLOCKGUARDCLEANUP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivm::partition_and_bind;

namespace {

struct SubBlockGuardCleanupPass
    : public impl::SubBlockGuardCleanupBase<SubBlockGuardCleanupPass> {
  void runOnOperation() override {

    llvm::SmallVector<scf::IfOp, 8> guards;
    getOperation().walk([&](scf::IfOp ifOp) {
      if (ifOp.getNumResults() > 0 && isOperandParallelSubBlockGuard(ifOp))
        guards.push_back(ifOp);
    });
    for (scf::IfOp ifOp : guards) {
      // Make the guard result-free.
      makeSubBlockGuardResultFree(ifOp);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createSubBlockGuardCleanupPass() {
  return std::make_unique<SubBlockGuardCleanupPass>();
}
