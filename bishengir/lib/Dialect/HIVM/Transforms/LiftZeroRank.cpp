//===- LiftZeroRank.cpp ----------------------------------------*- C++ -*-===//
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
//===---------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
#define GEN_PASS_DEF_LIFTZERORANK
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-lift-zero-rank"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
struct HIVMLiftZeroRankPattern
    : public OpInterfaceRewritePattern<hivm::HIVMStructuredOp> {
  using OpInterfaceRewritePattern<
      hivm::HIVMStructuredOp>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(hivm::HIVMStructuredOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         " op should have buffer semantics.");
    }

    DenseSet<OpOperand *> zeroRankOperands;
    for (OpOperand *operand :
         op.getHIVMOperands(/*includeExtraBuffer=*/false)) {
      Value value = operand->get();
      auto memrefType = dyn_cast_or_null<MemRefType>(value.getType());
      if (memrefType && memrefType.getRank() == 0) {
        zeroRankOperands.insert(operand);
      }
    }
    if (zeroRankOperands.empty()) {
      return failure();
    }
    for (OpOperand *operand : zeroRankOperands) {
      expandZeroRankOperands(op, operand, rewriter);
    }
    return success();
  }

private:
  void expandZeroRankOperands(hivm::HIVMStructuredOp op, OpOperand *operand,
                              PatternRewriter &rewriter) const {
    Value value = operand->get();
#ifndef NDEBUG
    auto memrefType = cast<MemRefType>(value.getType());
    assert(memrefType.getShape().empty() &&
           "shape must be empty before extend");
#endif
    SmallVector<int64_t> resultShape = {1};
    SmallVector<ReassociationIndices, 4> reassociation;
    Value expandOp = rewriter.create<memref::ExpandShapeOp>(
        op->getLoc(), resultShape, value, reassociation);
    operand->set(expandOp);
  }
};

struct LiftZeroRankPass : public impl::LiftZeroRankBase<LiftZeroRankPass> {
  void runOnOperation() override;
};
} // namespace

void LiftZeroRankPass::runOnOperation() {
  auto funcOp = getOperation();
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<HIVMLiftZeroRankPattern>(ctx);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createLiftZeroRankPass() {
  return std::make_unique<LiftZeroRankPass>();
}
