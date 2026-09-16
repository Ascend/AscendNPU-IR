//===- TestHIVMDimensionAnalyzer.cpp - Test HIVM dimension analyzer -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Test/TestPasses.h"

#include "bishengir/Dialect/HIVM/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "test-hivm-dimension-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir::utils::debugger;

namespace bishengir_test {
using namespace mlir;
struct TestHIVMDimensionAnalyzerPass
    : public PassWrapper<TestHIVMDimensionAnalyzerPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHIVMDimensionAnalyzerPass)

  TestHIVMDimensionAnalyzerPass() = default;
  TestHIVMDimensionAnalyzerPass(const TestHIVMDimensionAnalyzerPass &) {}
  TestHIVMDimensionAnalyzerPass &
  operator=(const TestHIVMDimensionAnalyzerPass &) {
    return *this;
  }

  StringRef getArgument() const final { return DEBUG_TYPE; }
  StringRef getDescription() const final {
    return "Test HIVM dimension analyzer";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    int64_t succeedFunc = 0;
    moduleOp.walk([&](func::FuncOp funcOp) {
      LDBG(funcOp);
      hivm::detail::DimensionAnalyzer analyzer(funcOp);
      auto res = analyzer.initialize();
      if (failed(res)) {
        LDBG("Failed initializing res");
        llvm::report_fatal_error("Analyzer failed");
      }
      analyzer.computeTilingDim();
      succeedFunc++;
    });

    llvm::outs() << succeedFunc << " succeedFunc - Function analyzed count\n";

    moduleOp.walk([&](hivm::FixpipeOp fixpipeOp) {
      auto forOp = cast<scf::ForOp>(fixpipeOp->getParentOp());
      hivm::detail::DimensionAnalyzer analyzer(forOp);
      auto res = analyzer.initialize();
      if (failed(res)) {
        LDBG("Failed initializing res");
        llvm::report_fatal_error("Analyzer failed");
      }
      analyzer.computeTilingDim(/*isVectorOp=*/false);
      llvm::outs() << "Tiling dim for " << fixpipeOp << " is "
                   << analyzer.getTilingDim(fixpipeOp.getSrc()) << '\n';
      forOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
        for (auto res : op->getResults()) {
          LDBG(res << '\n' << analyzer.getTilingDim(res));
        }
      });
    });
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect>();
  }
};

/// @see bishengir/tools/bishengir-opt/bishengir-opt.cpp for usage
void registerTestHIVMDimensionAnalyzer() {
  PassRegistration<TestHIVMDimensionAnalyzerPass>();
}

} // namespace bishengir_test
