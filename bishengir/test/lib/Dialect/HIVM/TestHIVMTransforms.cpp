//===----- TestHIVMTransforms.cpp - Test HIVM transformation patterns -----===//
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