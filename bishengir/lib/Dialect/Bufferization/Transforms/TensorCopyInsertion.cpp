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
// Loop yields alias their init. A read of that init inside the body is then a
// read of the yielded tensor, so tensor-copy-insertion copies the yield even
// when copying the init would already separate the two buffers. Copying the
// init in the same analysis does not help: the yield decision is made before
// any copy is written into the IR.
//
// tensor-copy-insertion is the pipeline of those two stages. Options are
// parsed once on the outer pass and forwarded to both. insert-loop-init-copies
// copies scf.for / scf.while inits and drops its analysis. It is also
// registered on its own. alloc_tensor() copy does not alias its source.
// resolve-tensor-copy-conflicts, the former tensor-copy-insertion pass, then
// analyzes that IR. The stages are nested in this pass so nothing can run
// between them and re-alias the copy with its source. After
// hivm-decompose-alloc-tensor-copy the whole pipeline runs again, because
// that decompose may turn the copy into an aliasing op.
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
#include "mlir/Pass/PassManager.h"

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

// Copy loop inits that must not share a buffer with a tensor read inside the
// body, then drop this analysis. `alloc_tensor() copy` does not alias its
// source, so resolve-tensor-copy-conflicts ties a yield to the copy instead
// of the original init.
//
// resolveTensorOpOperandConflicts only copies operands the analysis marked
// out-of-place. ForOp::isNotConflicting hides a read of the init inside the
// body, so that operand stays in-place and would not be copied here.
// ForOp::resolveConflicts then snapshots the init; that is the copy
// resolve-tensor-copy-conflicts was inserting after the yield had already
// been decided. Call resolveConflicts so the snapshot lands in this stage.
// WhileOp::resolveConflicts is resolveTensorOpOperandConflicts.
// This stays in tensor-copy-insertion. one-shot-bufferize keeps its own
// single post-order insertTensorCopies.
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

// CLI options live only on tensor-copy-insertion. The two stages receive the
// parsed OneShotBufferizationOptions and have no flags of their own.
#define FILL_COPY_INSERTION_OPTIONS(opt)                                       \
  do {                                                                         \
    (opt).allowReturnAllocsFromLoops = allowReturnAllocsFromLoops;             \
    (opt).allowUnknownOps = allowUnknownOps;                                   \
    (opt).analysisFuzzerSeed = analysisFuzzerSeed;                             \
    (opt).analysisHeuristic = parseHeuristicOption(analysisHeuristic);         \
    (opt).bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;           \
    (opt).checkParallelRegions = checkParallelRegions;                         \
    (opt).copyBeforeWrite = copyBeforeWrite;                                   \
    (opt).dumpAliasSets = dumpAliasSets;                                       \
    if (mustInferMemorySpace) {                                                \
      (opt).defaultMemorySpaceFn =                                             \
          [](TensorType) -> std::optional<Attribute> { return std::nullopt; }; \
    }                                                                          \
    (opt).printConflicts = printConflicts;                                     \
    (opt).testAnalysisOnly = testAnalysisOnly;                                 \
    (opt).noAnalysisFuncFilter = noAnalysisFuncFilter;                         \
    OpFilter::Entry::FilterFn filterFn = [&](Operation *op) {                  \
      if (dialectFilter.hasValue())                                            \
        return llvm::is_contained(dialectFilter,                               \
                                  op->getDialect()->getNamespace());           \
      return true;                                                             \
    };                                                                         \
    (opt).opFilter.allowOperation(filterFn);                                   \
  } while (false)

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
// analyzed. Both stages share this filter: copying an init inside a skipped
// function would change IR that resolve-tensor-copy-conflicts leaves alone.
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

namespace {
void addCopyInsertionDialects(DialectRegistry &registry) {
  registry
      .insert<mlir::bufferization::BufferizationDialect, memref::MemRefDialect,
              tensor::TensorDialect, func::FuncDialect, scf::SCFDialect>();
}

/// First stage of tensor-copy-insertion. Registered so a pipeline can run it
/// on its own, for example ahead of one-shot-bufferize. The outer pass still
/// forwards an already-parsed options object and does not re-parse CLI flags.
struct InsertLoopInitCopiesPass
    : public PassWrapper<InsertLoopInitCopiesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertLoopInitCopiesPass)

  InsertLoopInitCopiesPass() = default;
  explicit InsertLoopInitCopiesPass(const OneShotBufferizationOptions &options)
      : injected(options) {}
  InsertLoopInitCopiesPass(const InsertLoopInitCopiesPass &other)
      : PassWrapper(other), injected(other.injected) {}

  StringRef getArgument() const override { return "insert-loop-init-copies"; }

  StringRef getDescription() const override {
    return "Copy scf.for and scf.while inits before resolving the remaining "
           "tensor copy conflicts";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    addCopyInsertionDialects(registry);
  }

  void runOnOperation() override {
    OneShotBufferizationOptions opt;
    if (injected) {
      opt = *injected;
    } else {
      opt.allowReturnAllocsFromLoops = allowReturnAllocsFromLoops;
      opt.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;
      opt.analysisHeuristic = parseHeuristicOption(analysisHeuristic);
    }
    if (failed(insertLoopInitCopies(getOperation(), opt)))
      return signalPassFailure();
  }

  // Same analysis flags as tensor-copy-insertion. Absent when this pass is
  // constructed with an already-parsed OneShotBufferizationOptions.
  Pass::Option<bool> allowReturnAllocsFromLoops{
      *this, "allow-return-allocs-from-loops",
      llvm::cl::desc("Allows returning/yielding new allocations from a loop."),
      llvm::cl::init(false)};
  Pass::Option<bool> bufferizeFunctionBoundaries{
      *this, "bufferize-function-boundaries",
      llvm::cl::desc("Bufferize function boundaries (experimental)."),
      llvm::cl::init(false)};
  Pass::Option<std::string> analysisHeuristic{
      *this, "analysis-heuristic",
      llvm::cl::desc(
          "Heuristic that controls the IR traversal during analysis"),
      llvm::cl::init("bottom-up")};

private:
  std::optional<OneShotBufferizationOptions> injected;
};

static PassRegistration<InsertLoopInitCopiesPass> insertLoopInitCopiesPass;

/// Second stage of tensor-copy-insertion. This is the former
/// tensor-copy-insertion pass, renamed so the pipeline can keep that name.
struct ResolveTensorCopyConflictsPass
    : public PassWrapper<ResolveTensorCopyConflictsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveTensorCopyConflictsPass)

  ResolveTensorCopyConflictsPass(const OneShotBufferizationOptions &options,
                                 BufferizationStatistics *statistics)
      : options(options), statistics(statistics) {}

  StringRef getArgument() const override {
    return "resolve-tensor-copy-conflicts";
  }

  StringRef getDescription() const override {
    return "Analyze tensor IR and insert copies for the remaining "
           "bufferization conflicts";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    addCopyInsertionDialects(registry);
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    BufferizationStatistics local;
    if (options.copyBeforeWrite) {
      // Skip the analysis: every buffer write is preceded by a copy. This
      // mirrors the copy-before-write path of one-shot-bufferize.
      AnalysisState state(options);
      if (failed(insertTensorCopies(moduleOp, state)))
        return signalPassFailure();
    } else if (failed(insertTensorCopies(moduleOp, options, &local))) {
      return signalPassFailure();
    }
    if (statistics)
      *statistics = local;
  }

private:
  OneShotBufferizationOptions options;
  BufferizationStatistics *statistics = nullptr;
};

struct TensorCopyInsertionPass
    : public bishengir::bufferization::impl::TensorCopyInsertionBase<
          TensorCopyInsertionPass> {
  TensorCopyInsertionPass() = default;

  explicit TensorCopyInsertionPass(const OneShotBufferizationOptions &options)
      : options(options) {}

  void runOnOperation() override {
    OneShotBufferizationOptions opt;
    if (options)
      opt = *options;
    else
      FILL_COPY_INSERTION_OPTIONS(opt);
    if (failed(validateCopyInsertionOptions(&getContext(), opt)))
      return signalPassFailure();
    // The filter is applied once. Both stages see the same op filter.
    // copy-before-write skips analysis, so the filter does not apply.
    if (!opt.copyBeforeWrite && !opt.noAnalysisFuncFilter.empty())
      opt = denyUnanalyzedFuncs(opt);

    // copy-before-write and test-analysis-only do not rewrite loop inits.
    // The first stage exists so the second stage can reanalyze copied inits.
    BufferizationStatistics statistics;
    OpPassManager pipeline(ModuleOp::getOperationName());
    if (!opt.copyBeforeWrite && !opt.testAnalysisOnly)
      pipeline.addPass(std::make_unique<InsertLoopInitCopiesPass>(opt));
    pipeline.addPass(
        std::make_unique<ResolveTensorCopyConflictsPass>(opt, &statistics));
    if (failed(runPipeline(pipeline, getOperation())))
      return signalPassFailure();

    this->numTensorInPlace = statistics.numTensorInPlace;
    this->numTensorOutOfPlace = statistics.numTensorOutOfPlace;
  }

private:
  std::optional<OneShotBufferizationOptions> options;
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
