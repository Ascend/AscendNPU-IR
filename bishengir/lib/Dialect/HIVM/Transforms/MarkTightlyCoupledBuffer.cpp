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
// Marks tightly-coupled L1/UB `memref.alloc` buffers in a MIX function with a
// `hivm.tightly_coupled_buffer` id. Candidates are discovered from
// `hivm.fixpipe` dst (UB alloc) and `hivm.copy` dst (L1/cbuf alloc) via
// `traceDefOps`. MIX function into its AIC/AIV copies so that both clones
// inherit identical ids; PlanMemory later relies on those ids to pair the
// AIC/AIV buffers at consistent offsets.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/TightlyCoupledBufferUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
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

static void tryAddCandidateAlloc(memref::AllocOp allocOp,
                                 SmallVectorImpl<memref::AllocOp> &candidates) {
  if (!allocOp || llvm::is_contained(candidates, allocOp))
    return;
  candidates.push_back(allocOp);
}

static void collectAllocsFromDst(Value dst, AddressSpace requiredSpace,
                                 SmallVectorImpl<memref::AllocOp> &candidates) {
  for (Operation *op : traceDefOps<memref::AllocOp>(dst)) {
    auto allocOp = dyn_cast_or_null<memref::AllocOp>(op);
    if (!allocOp)
      continue;
    auto maybeMemrefAddressSpace =
        getOptionalHIVMAddressSpace(allocOp.getMemref().getType());
    if (maybeMemrefAddressSpace != requiredSpace)
      continue;
    tryAddCandidateAlloc(allocOp, candidates);
  }
}

static void markTightlyCoupledBufferOnFunc(func::FuncOp func) {
  if (hacc::utils::isHost(func))
    return;

  // Collect already-used ids from every L1/UB alloc so newly assigned ids skip
  // reserved values even when those allocs are not fresh candidates.
  llvm::DenseSet<int64_t> usedIds;
  func.walk([&](memref::AllocOp allocOp) {
    if (!isL1OrUBAlloc(allocOp))
      return;
    if (auto id = getTightlyCoupledBufferId(allocOp))
      usedIds.insert(*id);
  });

  SmallVector<memref::AllocOp> candidateAllocs;
  func.walk([&](hivm::FixpipeOp fixpipeOp) {
    collectAllocsFromDst(fixpipeOp.getDst(), AddressSpace::UB, candidateAllocs);
  });
  func.walk([&](hivm::CopyOp copyOp) {
    collectAllocsFromDst(copyOp.getDst(), AddressSpace::L1, candidateAllocs);
  });

  OpBuilder builder(func.getContext());
  for (memref::AllocOp allocOp : candidateAllocs) {
    if (getTightlyCoupledMark(allocOp.getMemref()).has_value())
      continue;
    int64_t newId = allocateNextTightlyCoupledId(usedIds);
    createTightlyCoupledBufferMark(builder, allocOp, newId);
  }
}

void MarkTightlyCoupledBufferPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Mirror the original SplitMixKernel behavior: only RegBase (Ascend950)
  // targets use CV tightly-coupled buffers.
  ModuleOp moduleOp = func->getParentOfType<ModuleOp>();
  if (!moduleOp || !hacc::utils::isAscend950(moduleOp))
    return;

  markTightlyCoupledBufferOnFunc(func);
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createMarkTightlyCoupledBufferPass() {
  return std::make_unique<MarkTightlyCoupledBufferPass>();
}
