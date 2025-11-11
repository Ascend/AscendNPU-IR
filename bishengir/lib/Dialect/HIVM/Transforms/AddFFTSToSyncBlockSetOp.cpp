//===--------------------- AddFFTSToSyncBlockSetOp.cpp --------------------===//
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
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hivm-add-ffts-to-syncblocksetop"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace mlir {
#define GEN_PASS_DEF_ADDFFTSTOSYNCBLOCKSETOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// AddFFTSToSyncBlockSetOpPass
//===----------------------------------------------------------------------===//
namespace {

struct AddFFTSToSyncBlockSetOpPass
    : public impl::AddFFTSToSyncBlockSetOpBase<AddFFTSToSyncBlockSetOpPass> {
  using AddFFTSToSyncBlockSetOpBase<
      AddFFTSToSyncBlockSetOpPass>::AddFFTSToSyncBlockSetOpBase;

public:
  void runOnOperation() override;
};

std::optional<Value> getFFTSBaseAddrFromFunc(func::FuncOp funcOp) {
  auto funcParamSize = funcOp.getNumArguments();
  for (size_t i = 0; i < funcParamSize; i++) {
    if (hacc::utils::isKernelArg(funcOp, i, hacc::KernelArgType::kFFTSBaseAddr))
      return funcOp.getArgument(i);
  }
  return std::nullopt;
}

} // end anonymous namespace

struct AddFFTSPattern : public OpRewritePattern<hivm::SyncBlockSetOp> {
  using OpRewritePattern<hivm::SyncBlockSetOp>::OpRewritePattern;

  explicit AddFFTSPattern(MLIRContext *ctx)
      : OpRewritePattern<hivm::SyncBlockSetOp>(ctx) {}

  LogicalResult matchAndRewrite(hivm::SyncBlockSetOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getFftsBaseAddr()) {
      return failure();
    }
    func::FuncOp enclosingFuncOp =
        op.getOperation()->getParentOfType<func::FuncOp>();
    if (!enclosingFuncOp) {
      llvm::report_fatal_error(
          "didn't find enclosing function of hivm::SyncBlockSetOp");
    }
    std::optional<Value> baseAddr = getFFTSBaseAddrFromFunc(enclosingFuncOp);
    if (!baseAddr.has_value()) {
      llvm::report_fatal_error("didn't find ffts base addr arg in the "
                               "enclosing function of hivm::SyncBlockSetOp");
    }
    op.getFftsBaseAddrMutable().assign(baseAddr.value());
    return success();
  }
};

void AddFFTSToSyncBlockSetOpPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  RewritePatternSet patterns(&getContext());
  patterns.insert<AddFFTSPattern>(patterns.getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createAddFFTSToSyncBlockSetOpPass() {
  return std::make_unique<AddFFTSToSyncBlockSetOpPass>();
}