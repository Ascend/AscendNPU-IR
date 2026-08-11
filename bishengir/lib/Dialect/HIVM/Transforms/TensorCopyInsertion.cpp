//===- TensorCopyInsertion.cpp - Insert tensor copies for bufferization ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass runs One-Shot Bufferize analysis (without bufferizing) and inserts
// `bufferization.alloc_tensor` copies wherever a conflict (e.g., RaW) was
// detected. It is equivalent to the analyze + insertTensorCopies phase of
// `one-shot-bufferize`, but without the final bufferization step.
//
// Running this pass before `one-shot-bufferize` allows a second round of
// analysis to catch conflicts that were exposed by the first round's copy
// insertion.
//
// The inserted `bufferization.alloc_tensor`(copy...) is converted into a
// `tensor.empty` + `hivm.hir.copy` pair afterwards: `alloc_tensor` has no
// in-place write, so a subsequent One-Shot Bufferization analysis cannot
// see/deal with the write it represents, which blocks the conflict analysis.
// In contrast, `hivm.hir.copy` carries an in-place write, so rewriting the
// alloc_tensor into `tensor.empty` + `hivm.hir.copy` keeps the write visible
// to the following conflict analysis.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_TENSORCOPYINSERTION
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;

static OneShotBufferizationOptions::AnalysisHeuristic
parseHeuristicOption(const std::string &s) {
  if (s == "bottom-up")
    return OneShotBufferizationOptions::AnalysisHeuristic::BottomUp;
  if (s == "top-down")
    return OneShotBufferizationOptions::AnalysisHeuristic::TopDown;
  if (s == "bottom-up-from-terminators")
    return OneShotBufferizationOptions::AnalysisHeuristic::
        BottomUpFromTerminators;
  if (s == "fuzzer")
    return OneShotBufferizationOptions::AnalysisHeuristic::Fuzzer;
  llvm_unreachable("invalid analysis heuristic option");
}

namespace {
/// Rewrite `bufferization.alloc_tensor`(copy=...) into a `tensor.empty`
/// followed by an `hivm.hir.copy` from the copy source into the empty tensor.
struct ConvertAllocTensorCopyToHIVMCopy
    : public OpRewritePattern<bufferization::AllocTensorOp> {
  using OpRewritePattern<bufferization::AllocTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::AllocTensorOp allocTensorOp,
                                PatternRewriter &rewriter) const override {
    Value copySrc = allocTensorOp.getCopy();
    if (!copySrc)
      return failure();
    rewriter.setInsertionPoint(allocTensorOp);
    Location loc = allocTensorOp.getLoc();
    auto emptyValue = utils::createEmptyOp(rewriter, loc, copySrc);
    auto copyOp = rewriter.create<hivm::CopyOp>(loc, allocTensorOp.getType(),
                                                copySrc, emptyValue);
    rewriter.replaceOp(allocTensorOp, copyOp.getResult(0));
    return success();
  }
};

struct TensorCopyInsertionPass
    : public impl::TensorCopyInsertionBase<TensorCopyInsertionPass> {
  TensorCopyInsertionPass() = default;

  explicit TensorCopyInsertionPass(
      const bufferization::OneShotBufferizationOptions &options)
      : options(options) {}

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Use externally provided options if any; otherwise fall back to the
    // default options used by the OneShotBufferize pass in the HIVM pipeline.
    OneShotBufferizationOptions opt;
    if (options) {
      opt = *options;
    } else {
      opt.allowReturnAllocsFromLoops = allowReturnAllocsFromLoops;
      opt.analysisHeuristic = parseHeuristicOption(analysisHeuristic);
      opt.allowUnknownOps = allowUnknownOps;
      opt.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;
    }

    if (failed(insertTensorCopies(moduleOp, opt)))
      return signalPassFailure();

    // Convert every `bufferization.alloc_tensor`(copy...) inserted by
    // insertTensorCopies into a `tensor.empty` + `hivm.hir.copy` pair.
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertAllocTensorCopyToHIVMCopy>(&getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns))))
      return signalPassFailure();
  }

private:
  std::optional<bufferization::OneShotBufferizationOptions> options;
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createTensorCopyInsertionPass() {
  return std::make_unique<TensorCopyInsertionPass>();
}

std::unique_ptr<Pass> mlir::hivm::createTensorCopyInsertionPass(
    const bufferization::OneShotBufferizationOptions &options) {
  return std::make_unique<TensorCopyInsertionPass>(options);
}
