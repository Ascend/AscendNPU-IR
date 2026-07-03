//===- MarkTightlyCoupledBuffer.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Marks every L1/UB `memref.alloc` of a MIX function with a
// `hivm.tightly_coupled_buffer` id. This runs before SplitMixKernel clones the
// MIX function into its AIC/AIV copies so that both clones inherit identical
// ids; PlanMemory later relies on those ids to pair the AIC/AIV buffers at
// consistent offsets.
//
// This was previously done inline inside SplitMixKernel
// (`annotateTightlyCoupledBuffer`). It is split into its own pass so that the
// companion HoistTightlyCoupledAlloc pass can consume the ids before the split.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"

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
  using MarkTightlyCoupledBufferBase<
      MarkTightlyCoupledBufferPass>::MarkTightlyCoupledBufferBase;

  explicit MarkTightlyCoupledBufferPass(
      const MarkTightlyCoupledBufferOptions &options)
      : MarkTightlyCoupledBufferBase(options) {}
  void runOnOperation() override;
};

void MarkTightlyCoupledBufferPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (hacc::utils::isHost(func))
    return;

  // Mirror the original SplitMixKernel behavior: only RegBase (Ascend950)
  // targets use CV tightly-coupled buffers.
  auto module = func->getParentOfType<ModuleOp>();
  if (!module || !hacc::utils::isAscend950(module))
    return;

  OpBuilder builder(func.getContext());

  SmallVector<memref::AllocOp> candidateAllocs;
  llvm::DenseSet<int64_t> usedIds;

  func.walk([&](memref::AllocOp allocOp) {
    auto maybeMemrefAddressSpace =
        mlir::hivm::getOptionalHIVMAddressSpace(allocOp.getMemref().getType());
    if (maybeMemrefAddressSpace != AddressSpace::L1 &&
        maybeMemrefAddressSpace != AddressSpace::UB) {
      return;
    }

    candidateAllocs.push_back(allocOp);

    auto maybeMarked = utils::getAnnotateOpWithAttr(
        allocOp.getMemref(), hivm::HIVMTightlyCoupledBufferAttr::name);
    if (!maybeMarked.has_value()) {
      return;
    }

    auto markOp = dyn_cast<annotation::MarkOp>(*maybeMarked);
    if (!markOp) {
      return;
    }

    auto tcbAttr = dyn_cast_if_present<hivm::HIVMTightlyCoupledBufferAttr>(
        markOp->getAttr(hivm::HIVMTightlyCoupledBufferAttr::name));
    if (!tcbAttr || !tcbAttr.getId().has_value()) {
      return;
    }

    usedIds.insert(tcbAttr.getId().value());
  });

  int64_t nextId = 0;
  auto getNextAvailableId = [&]() -> int64_t {
    while (usedIds.contains(nextId)) {
      ++nextId;
    }
    int64_t assignedId = nextId;
    usedIds.insert(assignedId);
    ++nextId;
    return assignedId;
  };

  for (memref::AllocOp allocOp : candidateAllocs) {
    auto maybeMarked = utils::getAnnotateOpWithAttr(
        allocOp.getMemref(), hivm::HIVMTightlyCoupledBufferAttr::name);
    if (maybeMarked.has_value()) {
      continue;
    }

    int64_t newId = getNextAvailableId();

    builder.setInsertionPointAfter(allocOp);
    auto mark = builder.create<annotation::MarkOp>(
        allocOp.getLoc(), allocOp.getMemref(),
        builder.getStrArrayAttr(llvm::ArrayRef<StringRef>{
            stringifyEffectMode(mlir::annotation::EffectMode::Write),
            stringifyEffectMode(mlir::annotation::EffectMode::Read)}),
        /*values=*/ValueRange{},
        /*keys=*/nullptr);

    mark->setAttr(hivm::HIVMTightlyCoupledBufferAttr::name,
                  hivm::HIVMTightlyCoupledBufferAttr::get(
                      allocOp->getContext(), static_cast<int32_t>(newId)));
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createMarkTightlyCoupledBufferPass() {
  return std::make_unique<MarkTightlyCoupledBufferPass>();
}

std::unique_ptr<Pass> mlir::hivm::createMarkTightlyCoupledBufferPass(
 	  const MarkTightlyCoupledBufferOptions &options) {
  return std::make_unique<MarkTightlyCoupledBufferPass>(options);
}
