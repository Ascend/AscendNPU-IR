//===- AllocToAlloca.cpp - Code to convert AllocOp to AllocaOp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
    auto parentGpuOp = dyn_cast<gpu::GPUFuncOp>(op->getParentOp());
    if (parentGpuOp) {
      rewriter.replaceOpWithNewOp<memref::AllocaOp>(
          op, currentMemRefType, op.getDynamicSizes(), op.getSymbolOperands(),
          op.getAlignmentAttr());
      return success();
    }
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
