//===- ExtendedCanonicalizer.cpp --------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/Linalg/IR/LinalgCanonicalizations.h"
#include "bishengir/Dialect/MemRef/IR/MemRefImpl.h"
#include "bishengir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "bishengir-canonicalize-ext"

namespace mlir {
#define GEN_PASS_DEF_CANONICALIZER
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

namespace {

using namespace mlir;

/// Fold transpose with transpose.
struct FoldTransposeWithTranspose : OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    auto defTransposeOp =
        transposeOp.getInput().getDefiningOp<linalg::TransposeOp>();
    if (!defTransposeOp)
      return failure();
    ArrayRef<int64_t> defPerms = defTransposeOp.getPermutation();
    ArrayRef<int64_t> perms = transposeOp.getPermutation();
    SmallVector<int64_t> foldedPerms;
    foldedPerms.reserve(perms.size());
    for (int64_t perm : perms)
      foldedPerms.push_back(defPerms[perm]);

    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        transposeOp, defTransposeOp.getInput(), transposeOp.getInit(),
        foldedPerms);
    return success();
  }
};

struct ExtendedCanonicalizer
    : public mlir::impl::CanonicalizerBase<ExtendedCanonicalizer> {

  using mlir::impl::CanonicalizerBase<ExtendedCanonicalizer>::CanonicalizerBase;

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("canonicalize-ext");
  }
  ::llvm::StringRef getArgument() const final { return "canonicalize-ext"; }

  ::llvm::StringRef getDescription() const final {
    return "Canonicalize operations";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ExtendedCanonicalizer");
  }
  ::llvm::StringRef getName() const final { return "ExtendedCanonicalizer"; }

  void runOnOperation() final {
    auto *context = getOperation()->getContext();
    RewritePatternSet patterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, context);

    mlir::memref::getExtendedCanonicalizationPatterns(patterns);
    linalg::getExtendedCanonicalizationPatterns(patterns);

    // The `FoldTransposeWithTranspose` pattern in our LLVM repo was modified
    // for A5 to avoid folding transposes that should be decomposed.
    // See the `needDecompose` method in the LLVM pattern implementation.
    //
    // However, for A2/A3 this hardcode isn't suitable and we need to fold
    // transposes unconditionally.
    //
    // It is difficult to propagate the `isAscend950` flag into the LLVM
    // implementation. Therefore, the original version of the pattern
    // (without the A5-specific logic) is copied here and used for A2/A3.
    //
    // TODO: After A5 merge is complete, try to remove the hardcoded logic from
    // LLVM and eliminate this workaround.
    auto moduleOp = dyn_cast<ModuleOp>(getOperation());
    if (moduleOp && !hacc::utils::isAscend950(moduleOp))
      patterns.add<FoldTransposeWithTranspose>(context);

    // Filter patterns according to the pass options.
    FrozenRewritePatternSet filteredPatterns{std::move(patterns),
                                             disabledPatterns, enabledPatterns};

    GreedyRewriteConfig config;
    config.useTopDownTraversal = topDownProcessingEnabled;
    config.enableRegionSimplification = enableRegionSimplification;
    config.maxIterations = maxIterations;
    config.maxNumRewrites = maxNumRewrites;

    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    if (auto converged =
            applyPatternsGreedily(getOperation(), filteredPatterns, config);
        testConvergence && failed(converged))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> bishengir::createExtendedCanonicalizerPass(
    const CanonicalizerOptions &options) {
  return std::make_unique<ExtendedCanonicalizer>(options);
}
