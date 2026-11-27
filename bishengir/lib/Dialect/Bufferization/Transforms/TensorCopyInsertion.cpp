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
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
// This file contains code from the LLVM Project.
// Original License: Apache License v2.0 with LLVM Exceptions
// Original Copyright: NA
// Original Source:
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Bufferization/Transforms/OneShotAnalysis.cpp
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Bufferization/Transforms/CopyInsertionUtils.h"
#include "bishengir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace bishengir::bufferization {
using namespace mlir;
#define GEN_PASS_DEF_TENSORCOPYINSERTION
#include "bishengir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bishengir::bufferization

using namespace mlir;
using namespace mlir::bufferization;

namespace {
struct TensorCopyInsertionPass
    : public bishengir::bufferization::impl::TensorCopyInsertionBase<
          TensorCopyInsertionPass> {
  template <typename PassT>
  friend void bishengir::bufferization::fillCopyInsertionOptions(
      PassT &pass, OneShotBufferizationOptions &opt);

  TensorCopyInsertionPass() = default;

  explicit TensorCopyInsertionPass(const OneShotBufferizationOptions &options)
      : options(options) {}

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OneShotBufferizationOptions opt;
    if (options)
      opt = *options;
    else
      bishengir::bufferization::fillCopyInsertionOptions(*this, opt);
    if (failed(bishengir::bufferization::validateCopyInsertionOptions(
            &getContext(), opt)))
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
        opt = bishengir::bufferization::denyUnanalyzedFuncs(opt);
      if (failed(insertTensorCopies(moduleOp, opt, &statistics)))
        return signalPassFailure();
    }
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
