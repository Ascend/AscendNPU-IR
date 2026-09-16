//===- SetBufferSize.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir {
#define GEN_PASS_DEF_SETBUFFERSIZE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-set-buffer-size"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
inline int64_t getBufferSizeFromAnnotation(annotation::MarkOp markOp) {
  return markOp->getAttrOfType<IntegerAttr>(kBufferSizeInByteAttr).getInt();
}

inline bool hasBufferSizeInfoInAnnotation(annotation::MarkOp markOp) {
  return markOp->hasAttrOfType<IntegerAttr>(kBufferSizeInByteAttr);
}

struct SetBufferSizePass : public impl::SetBufferSizeBase<SetBufferSizePass> {
  void runOnOperation() override;
};

} // namespace

void SetBufferSizePass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  DenseMap<Operation *, int64_t> alloc2BufferSize;
  auto walkResult = funcOp->walk([&](annotation::MarkOp markOp) {
    if (!hasBufferSizeInfoInAnnotation(markOp))
      return WalkResult::advance();
    Value markedValue = markOp.getSrc();
    auto maybeAlloc = utils::tracebackMemRef(markedValue);
    if (!utils::isAllocLikeOp(maybeAlloc)) {
      markOp->emitWarning(
          "Cannot find root memref alloc/alloca to set buffer size!");
      return WalkResult::advance();
    }
    Operation *definingOp = maybeAlloc.getDefiningOp();
    assert(definingOp);
    // Defining op should be a memref alloc-like op, so it should have one
    // result that has memref type.
    auto maybeMemRefType =
        cast<MemRefType>(definingOp->getOpResult(0).getType());
    if (maybeMemRefType.hasStaticShape()) {
      // If the memref alloc has static shape, only remove buffer size attr
      removeMarkOpAttr(markOp, kBufferSizeInByteAttr);
      return WalkResult::advance();
    }
    int64_t currentBufferSize = getBufferSizeFromAnnotation(markOp);
    auto it = alloc2BufferSize.find(definingOp);
    if (it != alloc2BufferSize.end() && it->second != currentBufferSize) {
      markOp->emitError(
          "Found conflicting buffer size annotation on the same alloc!");
      return WalkResult::interrupt();
    }
    alloc2BufferSize.insert({definingOp, currentBufferSize});
    removeMarkOpAttr(markOp, kBufferSizeInByteAttr);
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return signalPassFailure();

  IRRewriter opBuilder(&getContext());
  for (auto [allocOp, size] : alloc2BufferSize) {
    memref::ViewOp viewOp =
        utils::createAllocWithSettingBufferSize(allocOp, size, opBuilder);
    opBuilder.replaceOp(allocOp, viewOp);
  }
}

std::unique_ptr<Pass> mlir::hivm::createSetBufferSizePass() {
  return std::make_unique<SetBufferSizePass>();
}
