//===------------- LowerCreateSyncBlockLock.cpp -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cassert>
#include <cstdint>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_LOWERCREATESYNCBLOCKLOCK
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-lower-create-sync-block-lock"

namespace {
class LowerCreateSyncBlockLock
    : public OpRewritePattern<hivm::CreateSyncBlockLockOp> {
public:
  explicit LowerCreateSyncBlockLock(MLIRContext *context)
      : OpRewritePattern(context) {}

  // offset of current CreateSyncBlockLockOp in arg
  inline static size_t localOffset = 0;
  LogicalResult matchAndRewrite(hivm::CreateSyncBlockLockOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getLockArg()) {
      return op->emitOpError("failed to bind sync block lock argument");
    }

    auto loc = op.getLoc();
    // create viewOp
    auto constantOffset =
        rewriter.create<arith::ConstantIndexOp>(loc, localOffset);
    auto viewOp = rewriter.create<memref::ViewOp>(
        loc, op.getType(), op.getLockArg(),
        /*byte_shift*/ constantOffset, /*dynamic_sizes*/ ValueRange{});

    // calculate offset of the next CreateSyncBlockLockOp
    auto bindArgTypeWith =
        getElementTypeOrSelf(op.getLockArg()).getIntOrFloatBitWidth();
    auto lockResTypeWith =
        getElementTypeOrSelf(op.getMemref().getType()).getIntOrFloatBitWidth();
    auto perOffset = CEIL_DIV(lockResTypeWith, bindArgTypeWith);
    localOffset += perOffset;

    rewriter.replaceOp(op, viewOp);
    return success();
  }
};

struct LowerCreateSyncBlockLockPass
    : public impl::LowerCreateSyncBlockLockBase<LowerCreateSyncBlockLockPass> {
  void runOnOperation() override;
};
} // namespace

void LowerCreateSyncBlockLockPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  RewritePatternSet patterns(&getContext());

  patterns.add<LowerCreateSyncBlockLock>(&getContext());
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createSyncBlockLockLoweringPass() {
  return std::make_unique<LowerCreateSyncBlockLockPass>();
}
