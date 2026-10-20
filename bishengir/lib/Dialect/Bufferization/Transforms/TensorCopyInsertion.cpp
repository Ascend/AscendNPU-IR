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
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace bishengir::bufferization {
using namespace mlir;
#define GEN_PASS_DEF_TENSORCOPYINSERTION
#include "bishengir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bishengir::bufferization

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
struct TensorCopyInsertionPass
    : public bishengir::bufferization::impl::TensorCopyInsertionBase<
          TensorCopyInsertionPass> {
  TensorCopyInsertionPass() = default;

  explicit TensorCopyInsertionPass(
      const mlir::bufferization::OneShotBufferizationOptions &options)
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
  }

private:
  std::optional<mlir::bufferization::OneShotBufferizationOptions> options;
};
} // namespace

std::unique_ptr<Pass>
bishengir::bufferization::createTensorCopyInsertionPass() {
  return std::make_unique<TensorCopyInsertionPass>();
}

std::unique_ptr<Pass> bishengir::bufferization::createTensorCopyInsertionPass(
    const mlir::bufferization::OneShotBufferizationOptions &options) {
  return std::make_unique<TensorCopyInsertionPass>(options);
}
