//===- TensorCopyInsertion.cpp - Insert tensor copies for bufferization ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// tensor-copy-insertion runs One-Shot analysis and inserts
// `bufferization.alloc_tensor` copies for the conflicts it finds. It does not
// bufferize. Running it twice, with hivm-decompose-alloc-tensor-copy between
// the two runs, copies a swapped iter arg after the first copies alias their
// sources.
//
// insert-loop-init-copies is a separate pass. A loop yield aliases its init,
// so copying the init in the same analysis as the yield does not separate
// them: the yield is decided before the copy is written. This pass copies
// scf.for and scf.while inits first. The copy does not alias its source, and
// the following tensor-copy-insertion ties the yield to it. It runs once.
// The two passes share option parsing and each records only its own analysis.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
// This file contains code from the LLVM Project.
// Original License: Apache License v2.0 with LLVM Exceptions
// Original Copyright: NA
// Original Source:
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Bufferization/Transforms/OneShotAnalysis.cpp
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace bishengir::bufferization {
using namespace mlir;
#define GEN_PASS_DEF_INSERTLOOPINITCOPIES
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

// "copy-before-write" inserts a copy before every memory write, while
// "test-analysis-only" runs the analysis and leaves the IR unchanged.
static LogicalResult
validateCopyInsertionOptions(MLIRContext *ctx,
                             const OneShotBufferizationOptions &opt) {
  if (opt.copyBeforeWrite && opt.testAnalysisOnly) {
    emitError(UnknownLoc::get(ctx),
              "Invalid option: 'copy-before-write' cannot be used with "
              "'test-analysis-only'");
    return failure();
  }
  if (opt.printConflicts && !opt.testAnalysisOnly) {
    emitError(
        UnknownLoc::get(ctx),
        "Invalid option: 'print-conflicts' requires 'test-analysis-only'");
    return failure();
  }
  if (opt.dumpAliasSets && !opt.testAnalysisOnly) {
    emitError(
        UnknownLoc::get(ctx),
        "Invalid option: 'dump-alias-sets' requires 'test-analysis-only'");
    return failure();
  }
  if (!opt.bufferizeFunctionBoundaries && !opt.noAnalysisFuncFilter.empty()) {
    emitError(UnknownLoc::get(ctx),
              "Invalid option: 'no-analysis-func-filter' requires "
              "'bufferize-function-boundaries'");
    return failure();
  }
  return success();
}

// FuncOps named in noAnalysisFuncFilter, and ops nested in them, are not
// analyzed. Each pass applies this to its own options.
static OneShotBufferizationOptions
denyUnanalyzedFuncs(const OneShotBufferizationOptions &opt) {
  OpFilter::Entry::FilterFn analysisFilterFn = [=](Operation *op) {
    auto func = dyn_cast<func::FuncOp>(op);
    if (!func)
      func = op->getParentOfType<func::FuncOp>();
    if (func)
      return llvm::is_contained(opt.noAnalysisFuncFilter, func.getSymName());
    return false;
  };
  OneShotBufferizationOptions updated(opt);
  updated.opFilter.denyOperation(analysisFilterFn);
  return updated;
}

// Copy loop inits that must not share a buffer with a tensor read inside the
// body, then drop this analysis. `alloc_tensor() copy` does not alias its
// source, so the following tensor-copy-insertion ties a yield to the copy
// instead of the original init.
//
// resolveTensorOpOperandConflicts only copies operands the analysis marked
// out-of-place. ForOp::isNotConflicting hides a read of the init inside the
// body, so that operand stays in-place and would not be copied here.
// ForOp::resolveConflicts then snapshots the init. WhileOp::resolveConflicts
// is resolveTensorOpOperandConflicts. one-shot-bufferize keeps its own
// single post-order insertTensorCopies and does not run this pass.
static LogicalResult
insertLoopInitCopies(Operation *op,
                     const OneShotBufferizationOptions &options) {
  OneShotAnalysisState state(op, options);
  if (options.bufferizeFunctionBoundaries) {
    if (failed(
            analyzeModuleOp(cast<ModuleOp>(op), state, /*statistics=*/nullptr)))
      return failure();
  } else if (failed(analyzeOp(op, state, /*statistics=*/nullptr))) {
    return failure();
  }

  // Snapshot the loops first. Copy insertion adds ops in front of them.
  SmallVector<Operation *> loops;
  op->walk([&](Operation *nested) {
    if (isa<scf::ForOp, scf::WhileOp>(nested))
      loops.push_back(nested);
  });

  IRRewriter rewriter(op->getContext());
  for (Operation *loop : loops) {
    auto bufferizableOp = state.getOptions().dynCastBufferizableOp(loop);
    if (!bufferizableOp)
      continue;
    rewriter.setInsertionPoint(loop);
    if (failed(bufferizableOp.resolveConflicts(rewriter, state)))
      return failure();
  }
  return success();
}

namespace {
template <typename PassT>
void fillCopyInsertionOptions(PassT &pass, OneShotBufferizationOptions &opt);

struct InsertLoopInitCopiesPass
    : public bishengir::bufferization::impl::InsertLoopInitCopiesBase<
          InsertLoopInitCopiesPass> {
  friend void fillCopyInsertionOptions<>(InsertLoopInitCopiesPass &pass,
                                         OneShotBufferizationOptions &opt);

  InsertLoopInitCopiesPass() = default;

  explicit InsertLoopInitCopiesPass(const OneShotBufferizationOptions &options)
      : options(options) {}

  void runOnOperation() override {
    OneShotBufferizationOptions opt;
    if (options)
      opt = *options;
    else
      fillCopyInsertionOptions(*this, opt);
    if (failed(validateCopyInsertionOptions(&getContext(), opt)))
      return signalPassFailure();
    // These modes do not rewrite loop inits. tensor-copy-insertion owns them.
    if (opt.copyBeforeWrite || opt.testAnalysisOnly)
      return;
    if (!opt.noAnalysisFuncFilter.empty())
      opt = denyUnanalyzedFuncs(opt);
    if (failed(insertLoopInitCopies(getOperation(), opt)))
      return signalPassFailure();
  }

private:
  std::optional<OneShotBufferizationOptions> options;
};

struct TensorCopyInsertionPass
    : public bishengir::bufferization::impl::TensorCopyInsertionBase<
          TensorCopyInsertionPass> {
  friend void fillCopyInsertionOptions<>(TensorCopyInsertionPass &pass,
                                         OneShotBufferizationOptions &opt);

  TensorCopyInsertionPass() = default;

  explicit TensorCopyInsertionPass(const OneShotBufferizationOptions &options)
      : options(options) {}

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OneShotBufferizationOptions opt;
    if (options)
      opt = *options;
    else
      fillCopyInsertionOptions(*this, opt);
    if (failed(validateCopyInsertionOptions(&getContext(), opt)))
      return signalPassFailure();

    // Statistics describe this pass's analysis only.
    BufferizationStatistics statistics;
    if (opt.copyBeforeWrite) {
      // Skip the analysis: every buffer write is preceded by a copy. This
      // mirrors the copy-before-write path of one-shot-bufferize.
      AnalysisState state(opt);
      if (failed(insertTensorCopies(moduleOp, state)))
        return signalPassFailure();
    } else {
      if (!opt.noAnalysisFuncFilter.empty())
        opt = denyUnanalyzedFuncs(opt);
      if (failed(insertTensorCopies(moduleOp, opt, &statistics)))
        return signalPassFailure();
    }
    this->numTensorInPlace = statistics.numTensorInPlace;
    this->numTensorOutOfPlace = statistics.numTensorOutOfPlace;
  }

private:
  std::optional<OneShotBufferizationOptions> options;
};

// Pass options are protected. Each pass friends this function.
template <typename PassT>
void fillCopyInsertionOptions(PassT &pass, OneShotBufferizationOptions &opt) {
  opt.allowReturnAllocsFromLoops = pass.allowReturnAllocsFromLoops;
  opt.allowUnknownOps = pass.allowUnknownOps;
  opt.analysisFuzzerSeed = pass.analysisFuzzerSeed;
  opt.analysisHeuristic = parseHeuristicOption(pass.analysisHeuristic);
  opt.bufferizeFunctionBoundaries = pass.bufferizeFunctionBoundaries;
  opt.checkParallelRegions = pass.checkParallelRegions;
  opt.copyBeforeWrite = pass.copyBeforeWrite;
  opt.dumpAliasSets = pass.dumpAliasSets;
  if (pass.mustInferMemorySpace) {
    opt.defaultMemorySpaceFn = [](TensorType) -> std::optional<Attribute> {
      return std::nullopt;
    };
  }
  opt.printConflicts = pass.printConflicts;
  opt.testAnalysisOnly = pass.testAnalysisOnly;
  opt.noAnalysisFuncFilter = pass.noAnalysisFuncFilter;
  // The filter outlives this function and calls back into the pass options.
  opt.opFilter.allowOperation([&pass](Operation *op) {
    if (pass.dialectFilter.hasValue())
      return llvm::is_contained(pass.dialectFilter,
                                op->getDialect()->getNamespace());
    return true;
  });
}
} // namespace

std::unique_ptr<Pass>
bishengir::bufferization::createInsertLoopInitCopiesPass() {
  return std::make_unique<InsertLoopInitCopiesPass>();
}

std::unique_ptr<Pass> bishengir::bufferization::createInsertLoopInitCopiesPass(
    const mlir::bufferization::OneShotBufferizationOptions &options) {
  return std::make_unique<InsertLoopInitCopiesPass>(options);
}

std::unique_ptr<Pass>
bishengir::bufferization::createTensorCopyInsertionPass() {
  return std::make_unique<TensorCopyInsertionPass>();
}

std::unique_ptr<Pass> bishengir::bufferization::createTensorCopyInsertionPass(
    const mlir::bufferization::OneShotBufferizationOptions &options) {
  return std::make_unique<TensorCopyInsertionPass>(options);
}
