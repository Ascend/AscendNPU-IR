//===------------- HIVMMapForallToBlocks.cpp ----forall to block-----------===//
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
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hivm-map-forall-to-blocks"

namespace mlir {
#define GEN_PASS_DEF_HIVMMAPFORALLTOBLOCKS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct ForallToBlocksPattern : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<scf::ForallOp>())
      return failure();
    [[maybe_unused]] ForallRewriteResult rewriteResult;
    auto diag = mapForallToBlocksImpl(rewriter, op, rewriteResult);
    if (!diag.succeeded())
      return failure();
    return success();
  }
};

struct HIVMMapForallToBlocksPass
    : public impl::HIVMMapForallToBlocksBase<HIVMMapForallToBlocksPass> {
public:
  void runOnOperation() override;
};

} // namespace

void populateForallToBlocksPatterns(RewritePatternSet &patterns) {
  patterns.insert<ForallToBlocksPattern>(patterns.getContext());
}

void HIVMMapForallToBlocksPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  RewritePatternSet patterns(funcOp->getContext());
  populateForallToBlocksPatterns(patterns);
  // To expand delinearizeIndexOps
  affine::populateAffineExpandIndexOpsPatterns(patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createHIVMMapForallToBlocksPass() {
  return std::make_unique<HIVMMapForallToBlocksPass>();
}
