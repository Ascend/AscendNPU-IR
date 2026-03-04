//===- TestCanFuse.cpp - Test `canFuse` API. ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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