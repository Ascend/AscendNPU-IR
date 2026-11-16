//===-----Decompose.cpp ----------------------------------------*- C++ -*-===//
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
//===---------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Interfaces/AggregatedOpInterface.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_DECOMPOSE
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

#define DEBUG_TYPE "hfusion-decompose"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
struct HFusionDecomposePattern
    : public OpInterfaceRewritePattern<
          bishengir::BiShengIRAggregatedOpInterface> {
  using OpInterfaceRewritePattern<
      bishengir::BiShengIRAggregatedOpInterface>::OpInterfaceRewritePattern;

  explicit HFusionDecomposePattern(MLIRContext *context,
                                   bishengir::DecomposePhase d)
      : OpInterfaceRewritePattern<bishengir::BiShengIRAggregatedOpInterface>(
            context) {
    decomposePhase = d;
  }

  LogicalResult matchAndRewrite(bishengir::BiShengIRAggregatedOpInterface op,
                                PatternRewriter &rewriter) const override {
    bishengir::DecomposePhase phase = op.getDecomposePhase();
    if (phase != decomposePhase &&
        phase != bishengir::DecomposePhase::NO_CONSTRAINT) {
      return rewriter.notifyMatchFailure(op, "Not current phase");
    }

    FailureOr<SmallVector<Value>> maybeNewResults =
        op.decomposeOperation(rewriter);

    if (failed(maybeNewResults))
      return failure();

    rewriter.replaceOp(op, *maybeNewResults);
    return success();
  }

private:
  bishengir::DecomposePhase decomposePhase;
};

struct DecomposePass : public impl::DecomposeBase<DecomposePass> {
  explicit DecomposePass(const DecomposeOptions &options)
      : DecomposeBase(options) {}
  void runOnOperation() override;
};
} // namespace

void DecomposePass::runOnOperation() {
  auto funcOp = getOperation();
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<HFusionDecomposePattern>(ctx, hfusionDecomposePhase);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass>
mlir::hfusion::createDecomposePass(const DecomposeOptions &options) {
  return std::make_unique<DecomposePass>(options);
}
