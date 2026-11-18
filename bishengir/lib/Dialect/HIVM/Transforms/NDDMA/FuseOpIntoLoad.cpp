//===- FuseOpIntoLoad.cpp - fuse supported ops into loads ----------===//
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

#include "FuseOpIntoLoadPatterns.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEOPINTOLOAD
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct FuseOpIntoLoadPass
    : public impl::FuseOpIntoLoadBase<FuseOpIntoLoadPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!hacc::utils::hasTargetAttr(module) ||
        !hacc::utils::isAscend960(module))
      return;

    RewritePatternSet patterns(&getContext());
    hivm::detail::populateBroadcastLoadPatterns(patterns);

    // Patterns check their enclosing module's target. Keep operation folding
    // off; the region driver may still remove dead ops and simplify control
    // flow.
    GreedyRewriteConfig config;
    config.fold = false;
    if (failed(applyPatternsGreedily(module, std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createFuseOpIntoLoadPass() {
  return std::make_unique<FuseOpIntoLoadPass>();
}
