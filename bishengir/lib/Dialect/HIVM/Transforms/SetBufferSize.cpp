//===- SetBufferSize.cpp ----------------------------------------*- C++ -*-===//
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
