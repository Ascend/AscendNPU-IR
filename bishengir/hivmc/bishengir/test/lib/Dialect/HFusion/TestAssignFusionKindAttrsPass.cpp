//===- TestAssignFusionKindAttrsPass.cpp - Test assign fusion kind --------===//
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
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include "Test/TestPasses.h"

#include "mlir/Pass/Pass.h"

#include "llvm/Support/CommandLine.h"

namespace bishengir_test {
using namespace mlir;

// Define a category for our options
static llvm::cl::OptionCategory
    TestAssignFusionKindCategory("Test Assign Fusion Kind Options");

// Define the command-line option for fusion kind
static llvm::cl::opt<std::string>
    FusionKindOption("fusion-kind", llvm::cl::desc("Specify the fusion kind"),
                     llvm::cl::value_desc("kind"),
                     llvm::cl::cat(TestAssignFusionKindCategory));

struct TestAssignFusionKindAttrsPass
    : public PassWrapper<TestAssignFusionKindAttrsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAssignFusionKindAttrsPass)

  StringRef getArgument() const final { return "test-assign-fusion-kind"; }
  StringRef getDescription() const final { return "Test assign fusion kind"; }

  void runOnOperation() override {
    // Get the current operation being operated on.
    mlir::ModuleOp moduleOp = getOperation();
    // Walk over the functions in the module
    auto fusionKindOptional = hfusion::symbolizeFusionKind(FusionKindOption);
    if (!fusionKindOptional) {
      emitError(moduleOp.getLoc(), "Invalid fusion kind specified");
      return signalPassFailure();
    }
    moduleOp.walk([&](func::FuncOp funcOp) {
      if (hacc::utils::isHost(funcOp)) {
        trySetFusionKind(funcOp, fusionKindOptional.value());
      }
    });
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hfusion::HFusionDialect>();
  }
};

void registerTestAssignFusionKindAttrs() {
  PassRegistration<TestAssignFusionKindAttrsPass>();
}

} // namespace bishengir_test
