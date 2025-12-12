//===----- TestHIVMTransforms.cpp - Test HIVM transformation patterns -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing HIVM transformations.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/HoistAffine.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


namespace bishengir_test {
using namespace mlir;
struct TestHIVMTransformsPass
    : public PassWrapper<TestHIVMTransformsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHIVMTransformsPass)

  TestHIVMTransformsPass() = default;
  TestHIVMTransformsPass(const TestHIVMTransformsPass &pass)
      : PassWrapper(pass) {}
  TestHIVMTransformsPass &operator=(const TestHIVMTransformsPass &other) {
    this->testHoistAffine = other.testHoistAffine;
    return *this;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect, affine::AffineDialect>();
  }

  StringRef getArgument() const final { return "test-hivm-transform-patterns"; }
  StringRef getDescription() const final {
    return "Test HIVM transformation patterns by applying them greedily.";
  }

  void runOnOperation() override;

  Option<bool> testHoistAffine{*this, "test-hoist-affine",
                               llvm::cl::desc("Test hoist affine ops"),
                               llvm::cl::init(false)};
};

static LogicalResult applyHoistAffinePatterns(Operation *rootOp) {
  RewritePatternSet patterns(rootOp->getContext());
  hivm::detail::populateHoistAffinePattern(patterns);
  return applyPatternsAndFoldGreedily(rootOp, std::move(patterns));
}

void TestHIVMTransformsPass::runOnOperation() {
  Operation *rootOp = getOperation();
  if (testHoistAffine && failed(applyHoistAffinePatterns(rootOp))) {
    signalPassFailure();
    return;
  }
}

void registerTestHIVMTransformsPass() {
  PassRegistration<TestHIVMTransformsPass>();
}
} // namespace bishengir_test