//===- TestCanFuse.cpp - Test `canFuse` API. ------------------------------===//
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
#include "Test/TestPasses.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"

namespace bishengir_test {
using namespace mlir;

struct TestCanFusePass
    : public PassWrapper<TestCanFusePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCanFusePass)

  StringRef getArgument() const final { return "test-can-fuse"; }
  StringRef getDescription() const final { return "Test `canFuse` API"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (succeeded(hfusion::canFuse(func))) {
      func.emitRemark("This function is fusible!");
    } else {
      func.emitRemark("This function is not fusible!");
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hfusion::HFusionDialect>();
  }
};

void registerTestCanFusePass() { PassRegistration<TestCanFusePass>(); }

} // namespace bishengir_test
