//===- TestDimensionAnalyzer.cpp - Test dimension analyzer ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Test/TestPasses.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Tensor/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "test-dimension-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir::utils::debugger;

namespace bishengir_test {
using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::hfusion::detail;
struct TestDimensionAnalyzerPass
    : public PassWrapper<TestDimensionAnalyzerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDimensionAnalyzerPass)

  StringRef getArgument() const final { return DEBUG_TYPE; }
  StringRef getDescription() const final { return "Test dimension analyzer"; }

  void runOnOperation() override {
    // Get the current operation being operated on.
    ModuleOp moduleOp = getOperation();
    // Walk over the functions in the module
    moduleOp.walk([&](func::FuncOp funcOp) {
      DimensionAnalyzer analyzer(funcOp);
      auto res = analyzer.initialize();
      if (failed(res)) {
        LDBG("Failed initializing res");
        return;
      }
      analyzer.computeAnchor();
      auto maxRank = analyzer.getMaxRankDimShape();
      LDBG("Max Rank is " << to_string(maxRank));
      auto maxValues = analyzer.getAnchorShape();

      DenseMap<Value, Value> settled;
      OpBuilder builder(&getContext());
      for (auto [idx, elDyn] : llvm::enumerate(maxValues)) {
        Value endVal = tensor::reifyShapeToArg(elDyn[0].first, std::nullopt,
                                               builder, settled);
        endVal.getDefiningOp()->setAttr("EndValue",
                                        builder.getI64IntegerAttr(idx));
        LDBG("Dynamic dim: " << elDyn[0].first);
        builder.setInsertionPointAfterValue(endVal);
        builder.create<annotation::MarkOp>(endVal.getLoc(), endVal);
      }
      funcOp.walk([&](Operation *op) {
        for (auto res : op->getResults()) {
          auto commonAxis = analyzer.getCommonAxis(res);
          if (commonAxis.empty())
            continue;
          LDBG(res << " Common axis");
          for (size_t i = 0; i < commonAxis.size(); ++i) {
            LDBG("bit: " << commonAxis[i]);
          }
          LDBG("Interchange: " << to_string(analyzer.getInterchange(res)));
        }
      });
    });
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hfusion::HFusionDialect>();
  }
};

void registerTestDimensionAnalyzer() {
  PassRegistration<TestDimensionAnalyzerPass>();
}

} // namespace bishengir_test
