//===------------------ ConvertHIVMToHFusion.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements converting the vector side of HIVM operations back to
// upstream dialects for the delayed RegBase re-vectorization pipeline.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/ExecutionEngine/ConvertHIVMToUpstream.h"
#include "bishengir/ExecutionEngine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "convert-hivm-to-hfusion"

namespace mlir {
#define GEN_PASS_DEF_CONVERTHIVMTOHFUSION
#include "bishengir/ExecutionEngine/Passes.h.inc"
} // namespace mlir

namespace {
using namespace mlir;

struct ConvertHIVMToHFusion
    : public impl::ConvertHIVMToHFusionBase<ConvertHIVMToHFusion> {
  using Base::Base;

  void runOnOperation() override {
    auto &ctx = getContext();

    // TODO: The fa and hstu compilation issues are temporarily resolved, and a
    // formal solution will be provided after further investigation. For now,
    // these issues are commented out instead of being deleted. This branch will
    // only be used by native CV and will not affect other functions.

    auto moduleOp = getOperation();
    SmallVector<func::FuncOp> functions;
    moduleOp->walk([&functions](func::FuncOp funcOp) {
      if (hacc::utils::isHost(funcOp))
        return;
      std::optional<mlir::hivm::TFuncCoreType> funcCoreType =
          mlir::hivm::queryFuncCoreType(funcOp);
      if (funcCoreType.has_value() &&
          funcCoreType.value() == mlir::hivm::TFuncCoreType::AIC)
        return;
      functions.push_back(funcOp);
    });

    RewritePatternSet patterns(&ctx);
    mlir::execution_engine::populateConvertHIVMToHFusionPatterns(
        patterns, convertToNamedOp);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    for (func::FuncOp func : functions) {
      if (func.getBody().empty())
        continue;
      if (failed(applyPatternsGreedily(func, frozenPatterns))) {
        signalPassFailure();
        break;
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::execution_engine::createConvertHIVMToHFusionPass(
    const ConvertHIVMToHFusionOptions &options) {
  return std::make_unique<ConvertHIVMToHFusion>(options);
}
