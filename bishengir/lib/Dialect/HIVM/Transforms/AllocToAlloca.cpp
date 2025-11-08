//===- AllocToAlloca.cpp - Code to convert AllocOp to AllocaOp ------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ALLOCTOALLOCA
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct AllocToAllocaPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    const auto &currentMemRefType = cast<BaseMemRefType>(op.getType());
    auto memorySpace = currentMemRefType.getMemorySpace();
    if (!memorySpace) {
      return failure();
    }
    auto hivmAddressSpace = dyn_cast<AddressSpaceAttr>(memorySpace);
    if (!hivmAddressSpace) {
      return failure();
    }
    if (hivmAddressSpace.getAddressSpace() == AddressSpace::GM) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<memref::AllocaOp>(
        op, currentMemRefType, op.getDynamicSizes(), op.getSymbolOperands(),
        op.getAlignmentAttr());
    return success();
  }
};

struct AllocToAllocaPass : public impl::AllocToAllocaBase<AllocToAllocaPass> {
  void runOnOperation() override;
};

} // namespace

void populateAllocToAllocaPatterns(RewritePatternSet &patterns) {
  patterns.insert<AllocToAllocaPattern>(patterns.getContext());
}

void AllocToAllocaPass::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(op->getContext());
  populateAllocToAllocaPatterns(patterns);
  if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createAllocToAllocaPass() {
  return std::make_unique<AllocToAllocaPass>();
}
