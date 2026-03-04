//===- TestFunctionCallPass.cpp - Test AutoSchedule Tiling Func ------- ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test the functionality of assigning fusion
// kind to functions.
//
//===----------------------------------------------------------------------===//
#include "Test/TestPasses.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/TilingUtils.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include "mlir/Pass/Pass.h"

#include "llvm/Support/CommandLine.h"

namespace bishengir_test {
using namespace mlir;

struct TestFunctionCallPass
    : public PassWrapper<TestFunctionCallPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFunctionCallPass)

  StringRef getArgument() const final { return "test-function-call"; }
  StringRef getDescription() const final { return "Test `function_call` op."; }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *ctx = moduleOp.getContext();

    mlir::hfusion::StmtExprBuilder stmtExpr(moduleOp, ctx);
    mlir::FlatSymbolRefAttr externfuncNameAttr =
        mlir::FlatSymbolRefAttr::get(ctx, "extern_callee");
    mlir::StringAttr externfuncPathAttr =
        mlir::StringAttr::get(ctx, "./extern_callee.so");

    moduleOp.walk([&](func::FuncOp funcOp) {
      if (funcOp.getName() == "test") {
        mlir::Block &entryBlock = funcOp.getBody().front();

        if (entryBlock.empty() ||
            !mlir::isa<func::ReturnOp>(entryBlock.back())) {
          funcOp.emitError("Function must end with return operation");
          signalPassFailure();
          return;
        }

        auto returnOp = mlir::dyn_cast<func::ReturnOp>(entryBlock.back());
        mlir::hfusion::StmtExprBuilder stmtExpr(moduleOp, ctx);
        stmtExpr.setInsertionPoint(returnOp);

        SmallVector<mlir::Value> operands;
        for (mlir::BlockArgument &arg : entryBlock.getArguments()) {
          operands.push_back(arg);
        }

        auto externCallStmt = stmtExpr.createExternCallStmt(
            externfuncNameAttr, operands, externfuncPathAttr);
      }
    });
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hfusion::HFusionDialect>();
  }
};

void registerTestFunctionCallPass() {
  PassRegistration<TestFunctionCallPass>();
}

} // namespace bishengir_test