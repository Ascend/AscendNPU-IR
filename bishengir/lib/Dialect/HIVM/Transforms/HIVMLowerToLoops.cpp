//===- HIVMLowerToLoops.cpp - hivm op decompose ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/RWMutex.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMLOWERTOLOOPS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-lower-to-loops"

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct HIVMLowerToLoopsPass
    : public impl::HIVMLowerToLoopsBase<HIVMLowerToLoopsPass> {
  void runOnOperation() override;
};

struct HIVMLowerToLoopsPattern
    : public OpInterfaceRewritePattern<mlir::hivm::ImplByScalarOpInterface> {
  using OpInterfaceRewritePattern<
      mlir::hivm::ImplByScalarOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(mlir::hivm::ImplByScalarOpInterface op,
                                PatternRewriter &rewriter) const override {
    if (!op.shouldLowerToScalarLoops()) {
      return failure();
    }

    FailureOr<SmallVector<Value>> maybeNewResults = op.lowerToLoops(rewriter);

    if (failed(maybeNewResults))
      return failure();

    if (maybeNewResults.value().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    rewriter.replaceOp(op, *maybeNewResults);
    return success();
  }
};

} // namespace

void HIVMLowerToLoopsPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;
  RewritePatternSet patterns(&getContext());
  patterns.add<HIVMLowerToLoopsPattern>(&getContext());
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createHIVMLowerToLoopsPass() {
  return std::make_unique<HIVMLowerToLoopsPass>();
}