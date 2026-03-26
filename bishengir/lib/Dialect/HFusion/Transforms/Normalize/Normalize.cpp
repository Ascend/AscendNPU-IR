//===- Normalize.cpp --------------------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/HFusion/Transforms/NormalizePatterns.h"

namespace mlir {
#define GEN_PASS_DEF_NORMALIZE
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hfusion-normalize-ops"

namespace mlir::hfusion {

thread_local bool archIsRegbased{false};
thread_local bool archisAscend950{false};
thread_local bool archisAscend310B{false};
thread_local bool archisMembased{false};

static void populateNormalizeHFusionPatterns(RewritePatternSet &patterns,
                                             bool enableHighPrecision) {
  populateNormalizeF16ToF32Patterns(patterns);
  populateNormalizeTrigPatterns(patterns, enableHighPrecision);
  populateNormalizeI8I32CmpPatterns(patterns);
  populateNormalizeMulRecPatterns(patterns);
  populateNormalizeModPatterns(patterns);
  populateNormalizeCmpToCastPatterns(patterns);
  populateNormalizeArithmeticPatterns(patterns);
  populateNormalizePrimaryMathPatterns(patterns);
  populateNormalizeCastingPatterns(patterns);
  populateNormalizeComparisonCleanupPatterns(patterns);
  populateNormalizeBitwiseComparisonPatterns(patterns);
  populateNormalizePreReductionPatterns(patterns);
  populateNormalizeShiftI8ToI16(patterns);
  populateNormalizeI8ToTargetPatterns(patterns);
  populateNormalizeLateMathPatterns(patterns);
  populateNormalizeReductionPatterns(patterns);
  populateNormalizePreScalarCastingPatterns(patterns);
  populateNormalizeScalarLikeHFusionPatterns(patterns);
  populateNormalizeI1ToTargetPatterns(patterns);
  populateNormalizePreFinalArithmeticPatterns(patterns);
  populateNormalizeFinalCastingPatterns(patterns);
  populateNormalizeFinalReductionPatterns(patterns);
  populateNormalizeFinalScalarPatterns(patterns);
  populateNormalizeFinalArithmeticPatterns(patterns);
  populateNormalizeCmpVnePatterns(patterns);
  populateNormalizeAtomicAndSortPatterns(patterns);
}

namespace {
struct NormalizeHFusionPass : public impl::NormalizeBase<NormalizeHFusionPass> {
  explicit NormalizeHFusionPass(const NormalizeOptions &options)
      : NormalizeBase(options) {}

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation()->getParentOfType<ModuleOp>();
    archIsRegbased = hacc::utils::isRegBasedArch(moduleOp);
    archisAscend950 = hacc::utils::isAscend950(moduleOp);
    archisAscend310B = hacc::utils::isAscend310B(moduleOp);
    archisMembased = hacc::utils::isMemBasedArch(moduleOp);
    RewritePatternSet patterns(&getContext());
    populateNormalizeHFusionPatterns(patterns, enableHighPrecision);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
createHFusionNormalizeOpsPass(const NormalizeOptions &options) {
  return std::make_unique<NormalizeHFusionPass>(options);
}

} // namespace mlir::hfusion
