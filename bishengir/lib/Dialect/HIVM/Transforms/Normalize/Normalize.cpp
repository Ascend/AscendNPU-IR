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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_NORMALIZE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

namespace mlir::hivm {

struct NormalizeHIVMPass
    : public impl::NormalizeBase<NormalizeHIVMPass> {
  using impl::NormalizeBase<NormalizeHIVMPass>::NormalizeBase;
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation()->getParentOfType<ModuleOp>();
    bool isRegbased = hacc::utils::isRegBasedArch(moduleOp);
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    populateNormalizeArithmeticPatterns(patterns);
    if (!isRegbased)
      populateNormalizeCmpVnePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::hivm::createHIVMNormalizeOpsPass() {
  return std::make_unique<NormalizeHIVMPass>();
}

} // namespace mlir::hivm
