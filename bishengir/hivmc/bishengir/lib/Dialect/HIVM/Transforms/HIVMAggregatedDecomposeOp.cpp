//===---------- HIVMAggregatedDecomposeOp.cpp - hivm op decompose----------===//
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
#include "bishengir/Interfaces/AggregatedOpInterface.h"

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
#define GEN_PASS_DEF_HIVMAGGREGATEDDECOMPOSEOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-aggregated-decompose-op"

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct HIVMAggregatedDecomposeOpPass
    : public impl::HIVMAggregatedDecomposeOpBase<
          HIVMAggregatedDecomposeOpPass> {
  explicit HIVMAggregatedDecomposeOpPass(
      const HIVMAggregatedDecomposeOpOptions &options)
      : HIVMAggregatedDecomposeOpBase(options) {}

  void runOnOperation() override;
};

struct HIVMDecomposePattern : public OpInterfaceRewritePattern<
                                  bishengir::BiShengIRAggregatedOpInterface> {
  using OpInterfaceRewritePattern<
      bishengir::BiShengIRAggregatedOpInterface>::OpInterfaceRewritePattern;

  explicit HIVMDecomposePattern(MLIRContext *context,
                                bishengir::DecomposePhase d)
      : OpInterfaceRewritePattern<bishengir::BiShengIRAggregatedOpInterface>(
            context) {
    decomposePhase = d;
  }

  LogicalResult matchAndRewrite(bishengir::BiShengIRAggregatedOpInterface op,
                                PatternRewriter &rewriter) const override {
    bishengir::DecomposePhase phase = op.getDecomposePhase();
    if (phase != decomposePhase &&
        phase != bishengir::DecomposePhase::NO_CONSTRAINT) {
      return rewriter.notifyMatchFailure(op, "Not current phase");
    }

    FailureOr<SmallVector<Value>> maybeNewResults =
        op.decomposeOperation(rewriter);

    if (failed(maybeNewResults))
      return failure();

    if (maybeNewResults.value().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    rewriter.replaceOp(op, *maybeNewResults);
    return success();
  }

private:
  bishengir::DecomposePhase decomposePhase;
};

} // namespace

void HIVMAggregatedDecomposeOpPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;
  RewritePatternSet patterns(&getContext());
  patterns.add<HIVMDecomposePattern>(&getContext(), decomposePhase);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createHIVMAggregatedDecomposeOpPass(
    const HIVMAggregatedDecomposeOpOptions &options) {
  return std::make_unique<HIVMAggregatedDecomposeOpPass>(options);
}
