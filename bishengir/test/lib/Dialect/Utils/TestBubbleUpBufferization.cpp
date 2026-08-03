//===- TestBubbleUpBufferization.cpp - Test bufferization bubble up -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Test/TestPasses.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/BufferizationBubbleUp.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/Pattern.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "test-bubble-up-bufferization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace bishengir_test {

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivm::detail;
struct TestBubbleUpBufferizationPass
    : public PassWrapper<TestBubbleUpBufferizationPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBubbleUpBufferizationPass)

  StringRef getArgument() const final { return DEBUG_TYPE; }
  StringRef getDescription() const final {
    return "Test bubble up bufferization";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto *ctx = &getContext();

    RewritePatternSet patterns(funcOp.getContext());
    SmallVector<std::shared_ptr<BubbleUpStrategy>> strategies;
    strategies.push_back(std::make_shared<BufferizationBubbleUpStrategy>());
    patterns.add<BubbleUpPattern>(ctx, std::move(strategies));
    patterns
        .add<BufferizationPropagateDownPattern, BufferizationPropagateUpPattern,
             BufferizationPropagatePostProcessPattern>(ctx);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }

    auto walkResult = funcOp->walk(
        [](UnrealizedConversionCastOp op) { return WalkResult::interrupt(); });
    if (walkResult.wasInterrupted() || failed(funcOp.verify()))
      return signalPassFailure();

    llvm::outs() << "Successfully bubble up bufferization\n";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<annotation::AnnotationDialect, func::FuncDialect,
                    hivm::HIVMDialect>();
  }
};

void registerTestBubbleUpBufferization() {
  PassRegistration<TestBubbleUpBufferizationPass>();
}

} // namespace bishengir_test