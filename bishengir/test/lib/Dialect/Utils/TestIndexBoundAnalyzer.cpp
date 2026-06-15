//===- TestIndexBoundAnalyzer.cpp - Test index bound analyzer ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Test/TestPasses.h"
#include "bishengir/Dialect/Arith/Transforms/ValueBoundsOpInterfaceImpl.h"
#include "bishengir/Dialect/Utils/IndexBoundAnalyzer.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

namespace bishengir_test {
using namespace mlir;

struct TestIndexBoundAnalyzerPass
    : public PassWrapper<TestIndexBoundAnalyzerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestIndexBoundAnalyzerPass)

  StringRef getArgument() const final { return "test-index-bound-analyzer"; }
  StringRef getDescription() const final { return "Test index bound analyzer"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    arith::registerBiShengIRValueBoundsOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    utils::IndexBoundAnalyzer analyzer;
    llvm::StringMap<Value> labeledValues;
    getOperation().walk([&](Operation *op) {
      auto label = op->getAttrOfType<StringAttr>("test.index_bound_label");
      if (!label) {
        return;
      }
      labeledValues[label.getValue()] = op->getResult(0);
    });

    getOperation().walk([&](Operation *op) {
      auto label = op->getAttrOfType<StringAttr>("test.index_bound_label");
      if (!label) {
        return;
      }

      llvm::outs() << label.getValue() << ": " << analyzer.get(op->getResult(0))
                   << "\n";

      auto compareValue =
          op->getAttrOfType<IntegerAttr>("test.index_bound_compare_le");
      if (compareValue) {
        llvm::outs()
            << label.getValue() << " <= " << compareValue.getInt() << ": "
            << analyzer.compare(
                   op->getResult(0), utils::BoundComparisonPredicate::LE,
                   IntegerAttr::get(IndexType::get(op->getContext()),
                                    compareValue.getInt()))
            << "\n";
      }

      auto compareLtLabel =
          op->getAttrOfType<StringAttr>("test.index_bound_compare_lt_label");
      if (!compareLtLabel) {
        return;
      }

      auto rhs = labeledValues.find(compareLtLabel.getValue());
      if (rhs == labeledValues.end()) {
        op->emitOpError("unknown index bound label ")
            << compareLtLabel.getValue();
        signalPassFailure();
        return;
      }

      llvm::outs() << label.getValue() << " < " << compareLtLabel.getValue()
                   << ": " << (analyzer.get(op->getResult(0)) <
                                analyzer.get(rhs->second))
                   << "\n";
    });
  }
};

void registerTestIndexBoundAnalyzer() {
  PassRegistration<TestIndexBoundAnalyzerPass>();
}

} // namespace bishengir_test
