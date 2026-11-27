//===- InsertLoopInitCopies.cpp - Copy loop inits before yields -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A loop yield aliases its init, so copying the init in the same analysis as
// the yield does not separate them: the yield is decided before the copy is
// written. This pass copies scf.for and scf.while inits first. The copy does
// not alias its source, and the following tensor-copy-insertion ties the yield
// to it. It runs once.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Bufferization/Transforms/CopyInsertionUtils.h"
#include "bishengir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

namespace bishengir::bufferization {
using namespace mlir;
#define GEN_PASS_DEF_INSERTLOOPINITCOPIES
#include "bishengir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bishengir::bufferization

using namespace mlir;
using namespace mlir::bufferization;

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
struct InsertLoopInitCopiesPass
    : public bishengir::bufferization::impl::InsertLoopInitCopiesBase<
          InsertLoopInitCopiesPass> {
  template <typename PassT>
  friend void bishengir::bufferization::fillCopyInsertionOptions(
      PassT &pass, OneShotBufferizationOptions &opt);

  InsertLoopInitCopiesPass() = default;

  explicit InsertLoopInitCopiesPass(const OneShotBufferizationOptions &options)
      : options(options) {}

  void runOnOperation() override {
    OneShotBufferizationOptions opt;
    if (options)
      opt = *options;
    else
      bishengir::bufferization::fillCopyInsertionOptions(*this, opt);
    if (failed(bishengir::bufferization::validateCopyInsertionOptions(
            &getContext(), opt)))
      return signalPassFailure();
    // These modes do not rewrite loop inits. tensor-copy-insertion owns them.
    if (opt.copyBeforeWrite || opt.testAnalysisOnly)
      return;
    if (!opt.noAnalysisFuncFilter.empty())
      opt = bishengir::bufferization::denyUnanalyzedFuncs(opt);
    if (failed(insertLoopInitCopies(getOperation(), opt)))
      return signalPassFailure();
  }

private:
  std::optional<OneShotBufferizationOptions> options;
};
} // namespace

std::unique_ptr<Pass>
bishengir::bufferization::createInsertLoopInitCopiesPass() {
  return std::make_unique<InsertLoopInitCopiesPass>();
}

std::unique_ptr<Pass> bishengir::bufferization::createInsertLoopInitCopiesPass(
    const mlir::bufferization::OneShotBufferizationOptions &options) {
  return std::make_unique<InsertLoopInitCopiesPass>(options);
}
