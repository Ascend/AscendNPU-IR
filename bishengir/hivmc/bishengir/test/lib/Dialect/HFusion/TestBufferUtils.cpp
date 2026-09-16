//===- TestBufferUtils.cpp - Pass to test buffer utils --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Test/TestPasses.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/BufferUtils.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"

#include <set>

// Define the command-line option for fusion kind
static llvm::cl::opt<std::string> BufferUtilsTestVar(
    "test-buffer-utils-var",
    llvm::cl::desc("Specify the kind for testing modification"),
    llvm::cl::value_desc("modification"), llvm::cl::init(""));

namespace bishengir_test {
using namespace mlir;
using namespace mlir::utils;
using ArgsSet = std::set<std::string>;
namespace {
ArgsSet parseBufferUtilsTestVar() {
  ArgsSet result;
  llvm::StringRef input(BufferUtilsTestVar);
  SmallVector<llvm::StringRef> splitted;
  input.split(splitted, ',', -1, false);
  for (auto split : splitted)
    result.insert(split.str());
  return result;
}
} // namespace
struct TestBufferUtilsPass
    : public PassWrapper<TestBufferUtilsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBufferUtilsPass)

  StringRef getArgument() const final { return "test-buffer-utils"; }
  StringRef getDescription() const final { return "Test buffer utils"; }

  void runOnOperation() override {
    // Get the current operation being operated on.
    ArgsSet parsedVars = parseBufferUtilsTestVar();

    mlir::ModuleOp moduleOp = getOperation();
    // Walk over the functions in the module
    moduleOp.walk([&](func::FuncOp funcOp) {
      BufferAnalysisOptions options;
      if (parsedVars.count("pass-double-to-all-args")) {
        for (auto arg : funcOp.getArguments()) {
          auto copyOpUsers =
              llvm::make_filter_range(arg.getUsers(), [](Operation *user) {
                return (isa<hfusion::LoadOp>(user) ||
                        isa<hfusion::StoreOp>(user));
              });
          if (llvm::hasSingleElement(copyOpUsers)) {
            Operation *copyOp = *(copyOpUsers.begin());
            options.multiBufferCount[copyOp->getResult(0)] =
                2; /* 2 multi buffers */
          }
        }
      }

      if (parsedVars.count("enable-dma-opt"))
        options.enableDmaOpt = true;

      options.printLiveRange = true;
      auto maxBufferOut = mlir::utils::countMaxBuffer(funcOp, options);
      llvm::outs() << funcOp.getName() << ": " << maxBufferOut << "\n";
    });
  }
};
void registerTestBufferUtilsPass() { PassRegistration<TestBufferUtilsPass>(); }
} // namespace bishengir_test
