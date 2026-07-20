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

#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/RegBaseArchUtils.h"

#define DEBUG_TYPE "hfusion-normalize-ops"

namespace mlir::hfusion {

thread_local bool archIsRegbased{false};
thread_local bool archisAscend950{false};
thread_local bool archisAscend310B{false};
thread_local bool archisMembased{false};

void populateNormalizeHFusionRegBasePatterns(RewritePatternSet &patterns,
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
  populateNormalizeGatherIndexPatterns(patterns);
  populateNormalizeF8ToF16Patterns(patterns);
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
  populateNormalizeNonDenseScalarLikeBroadcastPatterns(patterns);
  populateNormalizeFinalArithmeticPatterns(patterns);
  populateNormalizeCmpVnePatterns(patterns);
  populateNormalizeAtomicPatterns(patterns);
  populateNormalizeSortPatterns(patterns);
}

LogicalResult runNormalizeRegBase(Operation *op, bool enableHighPrecision) {
  ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
  archIsRegbased = hacc::utils::isRegBasedArch(moduleOp);
  archisAscend950 = hacc::utils::isAscend950(moduleOp);
  archisAscend310B = hacc::utils::isAscend310B(moduleOp);
  archisMembased = hacc::utils::isMemBasedArch(moduleOp);
  RewritePatternSet patterns(op->getContext());
  populateNormalizeHFusionRegBasePatterns(patterns, enableHighPrecision);
  return applyPatternsGreedily(op, std::move(patterns));
}

} // namespace mlir::hfusion
