//===- MarkPreloadBuffers.cpp - Mark preload buffer slots ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
#define GEN_PASS_DEF_MARKPRELOADBUFFERS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct BufferMark {
  memref::AllocOp alloc;
  int64_t slots;
};

static LogicalResult collectBufferMarks(scf::ForOp loop,
                                        SmallVectorImpl<BufferMark> &plan) {
  auto allocations = loop.getBody()->getOps<memref::AllocOp>();
  if (allocations.empty()) {
    return success();
  }

  DenseMap<Value, std::pair<int64_t, int64_t>> spans;
  for (auto task : loop.getBody()->getOps<scope::ScopeOp>()) {
    if (!task->hasAttr(PreloadNumAttr::name) &&
        !task->hasAttr(MaxPreloadNumAttr::name)) {
      continue;
    }
    auto preload = task->getAttrOfType<IntegerAttr>(PreloadNumAttr::name);
    auto depth = task->getAttrOfType<IntegerAttr>(MaxPreloadNumAttr::name);
    if (!preload || !depth || !preload.getType().isInteger(32) ||
        !depth.getType().isInteger(32) || preload.getInt() < 0 ||
        preload.getInt() >= depth.getInt()) {
      return task.emitOpError("expected i32 preload attributes with "
                              "0 <= preload_num < max_preload_num");
    }

    int64_t preloadNum = preload.getInt();
    auto effects = getEffectsRecursively(task);
    if (!effects) {
      return task.emitOpError("cannot infer buffer versions without "
                              "explicit memory effects");
    }
    for (const auto &effect : *effects) {
      if (!isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) {
        continue;
      }
      Value root = effect.getValue();
      if (!root) {
        return task.emitOpError("cannot infer buffer versions from an "
                                "unbound memory effect");
      }
      while (auto view = root.getDefiningOp<ViewLikeOpInterface>())
        root = view.getViewSource();
      auto alloc = root.getDefiningOp<memref::AllocOp>();
      if (!alloc || alloc->getParentOp() != loop.getOperation()) {
        continue;
      }

      // The reversal from stage to preload_num preserves the lifetime span.
      auto &span =
          spans.try_emplace(root, preloadNum, preloadNum).first->second;
      span.first = std::min(span.first, preloadNum);
      span.second = std::max(span.second, preloadNum);
    }
  }

  for (auto alloc : allocations) {
    auto it = spans.find(alloc.getResult());
    if (it == spans.end() || it->second.first == it->second.second) {
      continue;
    }
    plan.push_back({alloc, it->second.second - it->second.first + 1});
  }
  return success();
}

// TODO: Use MarkMultiBuffer after better support for staged preload with
// memref alloc.
struct MarkPreloadBuffersPass
    : impl::MarkPreloadBuffersBase<MarkPreloadBuffersPass> {
  void runOnOperation() override {
    // Collect marks before changing IR so a failed analysis leaves it intact.
    SmallVector<BufferMark> plan;
    auto result = getOperation().walk([&](scf::ForOp loop) -> WalkResult {
      if (loop->hasAttrOfType<UnitAttr>(CVPipelinedLoopAttr::name) &&
          failed(collectBufferMarks(loop, plan))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return signalPassFailure();
    }

    // Extend an existing allocation mark to preserve its other attributes.
    for (auto [alloc, slots] : plan) {
      annotation::MarkOp mark;
      for (auto *user : alloc->getUsers())
        if (auto candidate = dyn_cast<annotation::MarkOp>(user)) {
          mark = candidate;
          break;
        }
      OpBuilder builder(alloc);
      if (!mark) {
        builder.setInsertionPointAfter(alloc);
        mark = builder.create<annotation::MarkOp>(alloc.getLoc(), alloc);
      }
      mark->setAttr(MultiBufferAttr::name, builder.getI32IntegerAttr(slots));
      mark->setAttr(PreloadLocalBufferAttr::name, builder.getUnitAttr());
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createMarkPreloadBuffersPass() {
  return std::make_unique<MarkPreloadBuffersPass>();
}
