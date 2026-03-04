//===- PassManagerOptions.cpp - PassManager Command Line Options ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Pass/PassManager.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

namespace mlir {
#define GEN_PASS_DEF_INJECTSYNC
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Populate dummy passes to get the argument name.
//===----------------------------------------------------------------------===//

struct DummyInjectSyncPass : public impl::InjectSyncBase<DummyInjectSyncPass> {
};

/// WhiListed passes that are allowed to print IR before/after execution.
static std::set<std::string> kAllowedPassList = {
    DummyInjectSyncPass::getArgumentName().str()};

namespace {
struct PassManagerOptions {
  //===--------------------------------------------------------------------===//
  // IR Printing
  //===--------------------------------------------------------------------===//
  PassNameCLParser printBefore{"bishengir-print-ir-before",
                               "Print IR before specified passes"};
  PassNameCLParser printAfter{"bishengir-print-ir-after",
                              "Print IR after specified passes"};

  /// Add an IR printing instrumentation if enabled by any 'print-ir' flags.
  void addPrinterInstrumentation(PassManager &pm);
};
} // namespace

static llvm::ManagedStatic<PassManagerOptions> options;

/// Add an IR printing instrumentation if enabled by any 'print-ir' flags.
void PassManagerOptions::addPrinterInstrumentation(PassManager &pm) {
  std::function<bool(Pass *, Operation *)> shouldPrintBeforePass;
  std::function<bool(Pass *, Operation *)> shouldPrintAfterPass;

  auto isAllowed = [](StringRef passName) {
    auto it = llvm::find(kAllowedPassList, passName);
    return (bool)(it != kAllowedPassList.end());
  };

  // Handle print-before.
  if (printBefore.hasAnyOccurrences()) {
    // Otherwise if there are specific passes to print before, then check to see
    // if the pass info for the current pass is included in the list.
    shouldPrintBeforePass = [&](Pass *pass, Operation *) {
      auto *passInfo = pass->lookupPassInfo();
      return passInfo && printBefore.contains(passInfo) &&
             isAllowed(pass->getArgument());
    };
  }

  if (printAfter.hasAnyOccurrences()) {
    // Otherwise if there are specific passes to print after, then check to see
    // if the pass info for the current pass is included in the list.
    shouldPrintAfterPass = [&](Pass *pass, Operation *) {
      auto *passInfo = pass->lookupPassInfo();
      return passInfo && printAfter.contains(passInfo) &&
             isAllowed(pass->getArgument());
    };
  }

  // If there are no valid printing filters, then just return.
  if (!shouldPrintBeforePass && !shouldPrintAfterPass)
    return;

  pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                      /*printModuleScope=*/false);
}

void bishengir::registerPassManagerCLOptions() {
  // Make sure that the options struct has been constructed.
  *options;
}

LogicalResult bishengir::applyPassManagerCLOptions(PassManager &pm) {
  if (!options.isConstructed())
    return failure();

  // Add the IR printing instrumentation.
  options->addPrinterInstrumentation(pm);
  return success();
}