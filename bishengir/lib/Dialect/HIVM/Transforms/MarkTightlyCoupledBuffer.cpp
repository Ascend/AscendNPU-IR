//===- MarkTightlyCoupledBuffer.cpp ---------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025~2026. All rights reserved.
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
//
// Marks every L1/UB `memref.alloc` of a MIX function with a
// `hivm.tightly_coupled_buffer` id. This runs before SplitMixKernel clones the
// MIX function into its AIC/AIV copies so that both clones inherit identical
// ids; PlanMemory later relies on those ids to pair the AIC/AIV buffers at
// consistent offsets.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/TightlyCoupledBufferUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_MARKTIGHTLYCOUPLEDBUFFER
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-mark-tightly-coupled-buffer"

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct MarkTightlyCoupledBufferPass
    : public impl::MarkTightlyCoupledBufferBase<MarkTightlyCoupledBufferPass> {
  void runOnOperation() override;
};

void MarkTightlyCoupledBufferPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (hacc::utils::isHost(func))
    return;

  // Only RegBase (Ascend950) targets use CV tightly-coupled buffers.
  auto module = func->getParentOfType<ModuleOp>();
  if (!module || !hacc::utils::isAscend950(module))
    return;

  SmallVector<memref::AllocOp> candidateAllocs;
  func.walk([&](memref::AllocOp allocOp) {
    if (isL1OrUBAlloc(allocOp))
      candidateAllocs.push_back(allocOp);
  });

  llvm::DenseSet<int64_t> usedIds =
      collectUsedTightlyCoupledIds(candidateAllocs);
  OpBuilder builder(func.getContext());
  for (memref::AllocOp allocOp : candidateAllocs) {
    if (getTightlyCoupledMark(allocOp.getMemref()).has_value())
      continue;
    int64_t newId = allocateNextTightlyCoupledId(usedIds);
    createTightlyCoupledBufferMark(builder, allocOp, newId);
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createMarkTightlyCoupledBufferPass() {
  return std::make_unique<MarkTightlyCoupledBufferPass>();
}
