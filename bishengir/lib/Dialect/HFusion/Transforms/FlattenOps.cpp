//===- FlattenOps.cpp ---- Flatten Linalg/HFusion Ops ---------------------===//
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
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Flattener/Flattener.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Transforms.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Utils/Util.h"

#include <numeric>

#define DEBUG_TYPE "hfusion-flatten-ops"

namespace mlir {
#define GEN_PASS_DEF_FLATTENOPS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;

namespace {
struct FlattenOpsPass : public impl::FlattenOpsBase<FlattenOpsPass> {
  explicit FlattenOpsPass(const FlattenOpsOptions &options)
      : FlattenOpsBase(options) {}

public:
  void runOnOperation() override;
};

/// Pattern for flattening linalg and hfusion elemwise ops.
template <typename LinalgOpTy>
struct FlattenElemwiseOpPattern : public OpRewritePattern<LinalgOpTy> {
public:
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(LinalgOpTy linalgOp,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(linalgOp);
    if (linalgOp.getNumLoops() <= 1)
      return failure();
    ReassociationIndices reassociation(linalgOp.getNumLoops());
    std::iota(reassociation.begin(), reassociation.end(), 0);
    // NEXT : rewrite isElementwise by judging thera is at least one
    // computational elementwise op(e.g add).
    auto maybeFlattened =
        (isElementwise(linalgOp) && !llvm::isa<linalg::BroadcastOp>(linalgOp))
            ? collapseOpIterationDims(linalgOp, reassociation, rewriter)
            : FailureOr<linalg::CollapseResult>(rewriter.notifyMatchFailure(
                  linalgOp, "only elementwise flattening is supported"));
    if (failed(maybeFlattened))
      return failure();
    rewriter.replaceOp(linalgOp, maybeFlattened->results);
    return success();
  }
};

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  patterns.add<FlattenElemwiseOpPattern<OpType>>(patterns.getContext());
}

/// Variadic helper function.
template <typename... OpTypes>
static void registerAll(RewritePatternSet &patterns) {
  (registerOne<OpTypes>(patterns), ...);
}

} // namespace

void hfusion::populateFlattenOpsPattern(RewritePatternSet &patterns) {
  registerAll<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >(patterns);
  registerAll<
#define GET_OP_LIST
#include "bishengir/Dialect/HFusion/IR/HFusionStructuredOps.cpp.inc"
      >(patterns);
  registerOne<linalg::GenericOp>(patterns);
}

void FlattenOpsPass::runOnOperation() {
  auto funcOp = getOperation();

  std::optional<mlir::hivm::TFuncCoreType> funcCoreType =
      mlir::hivm::queryFuncCoreType(funcOp);
  if (funcCoreType.has_value()) {
    if (funcCoreType.value() == mlir::hivm::TFuncCoreType::AIC) {
      return;
    } else if (funcCoreType.value() == mlir::hivm::TFuncCoreType::AIV) {
      return;
    }
  }
  if (this->flattenMode == FlattenMode::Greedy) {
    RewritePatternSet patterns(&getContext());
    populateFlattenOpsPattern(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();

    return;
  }

  // Tidy flatten
  if (hacc::utils::isHost(funcOp) && this->skipHost)
    return;

  hfusion::detail::Flattener flattener(funcOp);
  // Experimental Flatten, can remove this if
  if (this->registerBased &&
      util::hasUnpropagateableCase(funcOp, this->skipScope)) {
    return;
  }
  mlir::detail::DimensionAnalyzerOptions options;
  options.registerBased = this->registerBased;
  options.usePreOrderWalkTraversal = this->usePreOrderWalkTraversal;
  hfusion::detail::Flattener flattener(funcOp, options);
  if (failed(flattener.flatten(multiDynamicShape)))
    signalPassFailure();
}

std::unique_ptr<Pass>
mlir::hfusion::createFlattenOpsPass(const FlattenOpsOptions &options) {
  return std::make_unique<FlattenOpsPass>(options);
}
