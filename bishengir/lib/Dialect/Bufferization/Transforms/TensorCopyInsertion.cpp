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

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
    // options specified on the command line.
    OneShotBufferizationOptions opt;
    if (options) {
      opt = *options;
    } else {
      // Make new bufferization options if none were provided when creating
      // the pass.
      opt.allowReturnAllocsFromLoops = allowReturnAllocsFromLoops;
      opt.allowUnknownOps = allowUnknownOps;
      opt.analysisFuzzerSeed = analysisFuzzerSeed;
      opt.analysisHeuristic = parseHeuristicOption(analysisHeuristic);
      opt.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;
      opt.checkParallelRegions = checkParallelRegions;
      opt.copyBeforeWrite = copyBeforeWrite;
      opt.dumpAliasSets = dumpAliasSets;
      if (mustInferMemorySpace) {
        opt.defaultMemorySpaceFn =
            [](TensorType t) -> std::optional<Attribute> {
          return std::nullopt;
        };
      }
      opt.printConflicts = printConflicts;
      opt.testAnalysisOnly = testAnalysisOnly;
      opt.noAnalysisFuncFilter = noAnalysisFuncFilter;

      // Configure op filter.
      OpFilter::Entry::FilterFn filterFn = [&](Operation *op) {
        // Filter may be specified via options.
        if (this->dialectFilter.hasValue())
          return llvm::is_contained(this->dialectFilter,
                                    op->getDialect()->getNamespace());
        // No filter specified: All other ops are allowed.
        return true;
      };
      opt.opFilter.allowOperation(filterFn);
    }

    if (opt.copyBeforeWrite && opt.testAnalysisOnly) {
      // These two flags do not make sense together: "copy-before-write"
      // indicates that copies should be inserted before every memory write,
      // but "test-analysis-only" indicates that only the analysis should be
      // tested. (I.e., no IR is bufferized.)
      emitError(UnknownLoc::get(&getContext()),
                "Invalid option: 'copy-before-write' cannot be used with "
                "'test-analysis-only'");
      return signalPassFailure();
    }

    if (opt.printConflicts && !opt.testAnalysisOnly) {
      emitError(
          UnknownLoc::get(&getContext()),
          "Invalid option: 'print-conflicts' requires 'test-analysis-only'");
      return signalPassFailure();
    }

    if (opt.dumpAliasSets && !opt.testAnalysisOnly) {
      emitError(
          UnknownLoc::get(&getContext()),
          "Invalid option: 'dump-alias-sets' requires 'test-analysis-only'");
      return signalPassFailure();
    }

    if (!opt.bufferizeFunctionBoundaries && !opt.noAnalysisFuncFilter.empty()) {
      emitError(UnknownLoc::get(&getContext()),
                "Invalid option: 'no-analysis-func-filter' requires "
                "'bufferize-function-boundaries'");
      return signalPassFailure();
    }

    BufferizationStatistics statistics;
    if (opt.copyBeforeWrite) {
      // Skip the analysis: every buffer write is preceded by a copy. This
      // mirrors the copy-before-write path of one-shot-bufferize, where
      // copies are inserted without any analysis.
      AnalysisState state(opt);
      if (failed(insertTensorCopies(moduleOp, state)))
        return signalPassFailure();
    } else if (!opt.noAnalysisFuncFilter.empty()) {
      // FuncOps whose names are specified in options.noAnalysisFuncFilter
      // will not be analyzed. Ops in these FuncOps will not be analyzed as
      // well.
      OpFilter::Entry::FilterFn analysisFilterFn = [=](Operation *op) {
        auto func = dyn_cast<func::FuncOp>(op);
        if (!func)
          func = op->getParentOfType<func::FuncOp>();
        if (func)
          return llvm::is_contained(opt.noAnalysisFuncFilter,
                                    func.getSymName());
        return false;
      };
      OneShotBufferizationOptions updatedOptions(opt);
      updatedOptions.opFilter.denyOperation(analysisFilterFn);
      if (failed(insertTensorCopies(moduleOp, updatedOptions, &statistics)))
        return signalPassFailure();
    } else {
      if (failed(insertTensorCopies(moduleOp, opt, &statistics)))
        return signalPassFailure();
    }

    // Set pass statistics.
    this->numTensorInPlace = statistics.numTensorInPlace;
    this->numTensorOutOfPlace = statistics.numTensorOutOfPlace;
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
