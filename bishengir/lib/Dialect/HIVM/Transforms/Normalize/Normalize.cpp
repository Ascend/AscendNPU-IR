//===-------- Normalize.cpp -------------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_HIVMNORMALIZE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

namespace mlir::hivm {

thread_local bool archIsRegbased{false};
thread_local bool archisAscend950{false};
thread_local bool archisAscend310B{false};
thread_local bool archisMembased{false};

struct NormalizeHIVMPass
    : public impl::HIVMNormalizeBase<NormalizeHIVMPass> {
  using impl::HIVMNormalizeBase<NormalizeHIVMPass>::HIVMNormalizeBase;
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation()->getParentOfType<ModuleOp>();
    archIsRegbased = hacc::utils::isRegBasedArch(moduleOp);
    archisAscend950 = hacc::utils::isAscend950(moduleOp);
    archisAscend310B = hacc::utils::isAscend310B(moduleOp);
    archisMembased = hacc::utils::isMemBasedArch(moduleOp);
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    populateNormalizeF16ToF32Patterns(patterns);
    populateNormalizeTrigPatterns(patterns, enableHighPrecision);
    populateNormalizeI8I32CmpPatterns(patterns);
    populateNormalizeModPatterns(patterns);
    populateNormalizeArithmeticPatterns(patterns);
    populateNormalizePrimaryMathPatterns(patterns);
    populateNormalizeCastingPatterns(patterns);
    populateNormalizeGatherIndexPatterns(patterns);
    populateNormalizeComparisonCleanupPatterns(patterns);
    populateNormalizeBitwiseComparisonPatterns(patterns);
    populateNormalizePreReductionPatterns(patterns);
    populateNormalizeShiftI8ToI16(patterns);
    populateNormalizeI8ToTargetPatterns(patterns);
    populateNormalizeLateMathPatterns(patterns);
    populateNormalizeReductionPatterns(patterns);
    populateNormalizeScalarLikeHIVMPatterns(patterns);
    populateNormalizeI1ToTargetPatterns(patterns);
    populateNormalizeFinalReductionPatterns(patterns);
    // "NonDense" means the broadcast source is a scalar-like shaped value,
    // but not an arith.constant dense tensor. Dense constants are handled by
    // populateNormalizeScalarLikeHIVMPatterns above; this late regbased-only
    // pass extracts the single runtime value and rebuilds the broadcast.
    populateNormalizeNonDenseScalarLikeBroadcastPatterns(patterns,
                                                         archIsRegbased);
    populateNormalizeFinalArithmeticPatterns(patterns);
    populateNormalizeCmpVnePatterns(patterns);
    populateNormalizeAtomicPatterns(patterns);
    populateNormalizeSortPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createHIVMNormalizeOpsPass() {
  return std::make_unique<NormalizeHIVMPass>();
}

} // namespace mlir::hivm
