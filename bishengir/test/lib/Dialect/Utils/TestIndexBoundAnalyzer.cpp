//===- TestIndexBoundAnalyzer.cpp - Test index bound analyzer ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Test/TestPasses.h"
#include "bishengir/Dialect/Utils/IndexBoundAnalyzer.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace bishengir_test {
using namespace mlir;

struct TestIndexBoundAnalyzerPass
    : public PassWrapper<TestIndexBoundAnalyzerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestIndexBoundAnalyzerPass)

  StringRef getArgument() const final { return "test-index-bound-analyzer"; }
  StringRef getDescription() const final { return "Test index bound analyzer"; }

  void runOnOperation() override {
    utils::IndexBoundAnalyzer analyzer;
    getOperation().walk([&](Operation *op) {
      auto label = op->getAttrOfType<StringAttr>("test.index_bound_label");
      if (!label)
        return;

      llvm::outs() << label.getValue() << ": " << analyzer.get(op->getResult(0))
                   << "\n";
    });
  }
};

void registerTestIndexBoundAnalyzer() {
  PassRegistration<TestIndexBoundAnalyzerPass>();
}

} // namespace bishengir_test
