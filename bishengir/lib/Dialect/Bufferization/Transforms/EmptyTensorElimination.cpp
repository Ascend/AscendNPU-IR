//===- EmptyTensorElimination.cpp - Eliminate empty tensors ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass runs One-Shot Bufferize analysis and replaces eligible tensor.empty
// ops with destination subsets from tensor.insert_slice or
// bufferization.materialize_in_destination. This allows producers to bufferize
// directly into destination storage, avoiding temporary allocations and copies.
// Function boundary analysis can track these relationships across VF calls.
// Memref materialization destinations must be restricted and writable.
//
//===----------------------------------------------------------------------===//
// This file contains code from the LLVM Project.
// Original License: Apache License v2.0 with LLVM Exceptions
// Original Copyright: NA
// Original Source:
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Bufferization/Transforms/EmptyTensorElimination.cpp
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/StringSwitch.h"

namespace bishengir::bufferization {
using namespace mlir;
#define GEN_PASS_DEF_EMPTYTENSORELIMINATION
#include "bishengir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bishengir::bufferization

using namespace mlir;
using namespace mlir::bufferization;

namespace {
/// Keep empty tensor elimination on the same module analysis configuration as
/// tensor copy insertion. The upstream pass does not accept these options.
struct EliminateEmptyTensorsPass
    : public bishengir::bufferization::impl::EmptyTensorEliminationBase<
          EliminateEmptyTensorsPass> {
  EliminateEmptyTensorsPass() = default;

  explicit EliminateEmptyTensorsPass(const OneShotBufferizationOptions &options)
      : options(options) {}

  void runOnOperation() override {
    // Prefer pipeline-provided options; otherwise use the command-line options.
    OneShotBufferizationOptions opt;
    if (options) {
      opt = *options;
    } else {
      opt.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;
      opt.allowReturnAllocsFromLoops = allowReturnAllocsFromLoops;
      opt.allowUnknownOps = allowUnknownOps;
      opt.checkParallelRegions = checkParallelRegions;
      opt.analysisFuzzerSeed = analysisFuzzerSeed;
      using Heuristic = OneShotBufferizationOptions::AnalysisHeuristic;
      auto heuristic =
          llvm::StringSwitch<std::optional<Heuristic>>(analysisHeuristic)
              .Case("bottom-up", Heuristic::BottomUp)
              .Case("top-down", Heuristic::TopDown)
              .Case("bottom-up-from-terminators",
                    Heuristic::BottomUpFromTerminators)
              .Case("fuzzer", Heuristic::Fuzzer)
              .Default(std::nullopt);
      if (!heuristic) {
        getOperation().emitError("invalid analysis-heuristic: ")
            << analysisHeuristic;
        return signalPassFailure();
      }
      opt.analysisHeuristic = *heuristic;
    }
    OneShotAnalysisState state(getOperation(), opt);
    LogicalResult result = opt.bufferizeFunctionBoundaries
                               ? analyzeModuleOp(getOperation(), state)
                               : analyzeOp(getOperation(), state);
    if (failed(result))
      return signalPassFailure();
    IRRewriter rewriter(&getContext());
    if (failed(eliminateEmptyTensors(rewriter, getOperation(), state)))
      signalPassFailure();
  }

private:
  std::optional<OneShotBufferizationOptions> options;
};
} // namespace

std::unique_ptr<Pass>
bishengir::bufferization::createEmptyTensorEliminationPass() {
  return std::make_unique<EliminateEmptyTensorsPass>();
}

std::unique_ptr<Pass>
bishengir::bufferization::createEmptyTensorEliminationPass(
    const OneShotBufferizationOptions &options) {
  return std::make_unique<EliminateEmptyTensorsPass>(options);
}
