//===- CopyInsertionUtils.h - Shared tensor-copy option helpers -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Option parsing shared by insert-loop-init-copies and tensor-copy-insertion.
// Each pass records its own analysis; these helpers do not.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_COPYINSERTIONUTILS_H
#define BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_COPYINSERTIONUTILS_H

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace bishengir {
namespace bufferization {

inline mlir::bufferization::OneShotBufferizationOptions::AnalysisHeuristic
parseHeuristicOption(const std::string &s) {
  using Heuristic =
      mlir::bufferization::OneShotBufferizationOptions::AnalysisHeuristic;
  if (s == "bottom-up")
    return Heuristic::BottomUp;
  if (s == "top-down")
    return Heuristic::TopDown;
  if (s == "bottom-up-from-terminators")
    return Heuristic::BottomUpFromTerminators;
  if (s == "fuzzer")
    return Heuristic::Fuzzer;
  llvm_unreachable("invalid analysis heuristic option");
}

// "copy-before-write" inserts a copy before every memory write, while
// "test-analysis-only" runs the analysis and leaves the IR unchanged.
inline mlir::LogicalResult validateCopyInsertionOptions(
    mlir::MLIRContext *ctx,
    const mlir::bufferization::OneShotBufferizationOptions &opt) {
  if (opt.copyBeforeWrite && opt.testAnalysisOnly) {
    mlir::emitError(mlir::UnknownLoc::get(ctx),
                    "Invalid option: 'copy-before-write' cannot be used with "
                    "'test-analysis-only'");
    return mlir::failure();
  }
  if (opt.printConflicts && !opt.testAnalysisOnly) {
    mlir::emitError(
        mlir::UnknownLoc::get(ctx),
        "Invalid option: 'print-conflicts' requires 'test-analysis-only'");
    return mlir::failure();
  }
  if (opt.dumpAliasSets && !opt.testAnalysisOnly) {
    mlir::emitError(
        mlir::UnknownLoc::get(ctx),
        "Invalid option: 'dump-alias-sets' requires 'test-analysis-only'");
    return mlir::failure();
  }
  if (!opt.bufferizeFunctionBoundaries && !opt.noAnalysisFuncFilter.empty()) {
    mlir::emitError(mlir::UnknownLoc::get(ctx),
                    "Invalid option: 'no-analysis-func-filter' requires "
                    "'bufferize-function-boundaries'");
    return mlir::failure();
  }
  return mlir::success();
}

// FuncOps named in noAnalysisFuncFilter, and ops nested in them, are not
// analyzed. Each pass applies this to its own options.
inline mlir::bufferization::OneShotBufferizationOptions denyUnanalyzedFuncs(
    const mlir::bufferization::OneShotBufferizationOptions &opt) {
  mlir::bufferization::OpFilter::Entry::FilterFn analysisFilterFn =
      [=](mlir::Operation *op) {
        auto func = mlir::dyn_cast<mlir::func::FuncOp>(op);
        if (!func)
          func = op->getParentOfType<mlir::func::FuncOp>();
        if (func)
          return llvm::is_contained(opt.noAnalysisFuncFilter,
                                    func.getSymName());
        return false;
      };
  mlir::bufferization::OneShotBufferizationOptions updated(opt);
  updated.opFilter.denyOperation(analysisFilterFn);
  return updated;
}

// Pass options are protected. Each pass friends this function.
template <typename PassT>
void fillCopyInsertionOptions(
    PassT &pass, mlir::bufferization::OneShotBufferizationOptions &opt) {
  opt.allowReturnAllocsFromLoops = pass.allowReturnAllocsFromLoops;
  opt.allowUnknownOps = pass.allowUnknownOps;
  opt.analysisFuzzerSeed = pass.analysisFuzzerSeed;
  opt.analysisHeuristic = parseHeuristicOption(pass.analysisHeuristic);
  opt.bufferizeFunctionBoundaries = pass.bufferizeFunctionBoundaries;
  opt.checkParallelRegions = pass.checkParallelRegions;
  opt.copyBeforeWrite = pass.copyBeforeWrite;
  opt.dumpAliasSets = pass.dumpAliasSets;
  if (pass.mustInferMemorySpace) {
    opt.defaultMemorySpaceFn =
        [](mlir::TensorType) -> std::optional<mlir::Attribute> {
      return std::nullopt;
    };
  }
  opt.printConflicts = pass.printConflicts;
  opt.testAnalysisOnly = pass.testAnalysisOnly;
  opt.noAnalysisFuncFilter = pass.noAnalysisFuncFilter;
  // The filter outlives this function and calls back into the pass options.
  opt.opFilter.allowOperation([&pass](mlir::Operation *op) {
    if (pass.dialectFilter.hasValue())
      return llvm::is_contained(pass.dialectFilter,
                                op->getDialect()->getNamespace());
    return true;
  });
}

} // namespace bufferization
} // namespace bishengir

#endif // BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_COPYINSERTIONUTILS_H
