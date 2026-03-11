//===-------------------- PropagateConvertLayout.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "hivm-propagate-convert-layout"

namespace mlir {
#define GEN_PASS_DEF_PROPAGATECONVERTLAYOUT
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct PropagateConvertLayoutPass
    : public impl::PropagateConvertLayoutBase<PropagateConvertLayoutPass> {
  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    // PropagateConvertLayoutInternalOptions class is defined in `ConvertLayoutUtils.h`
    // and is different with PropagateConvertLayoutOptions from the `HIVM/Transforms/Passes.td`
    PropagateConvertLayoutInternalOptions options;
    options.allowAgnosticOps = allowAgnosticOps;
    populateConvertLayoutExtractSlice(patterns, context);
    // Enable this only for vcast
    populateConvertLayoutElementwise(patterns, context, options);
    populateConvertLayoutScfIf(patterns, context);
    populateConvertLayoutScfFor(patterns, context);
    populateConvertLayoutScfWhile(patterns, context);
    ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createPropagateConvertLayoutPass() {
  return std::make_unique<PropagateConvertLayoutPass>();
}