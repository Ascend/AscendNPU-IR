//===------------- LowerCreateSyncBlockLock.cpp -----------------*- C++ -*-===//
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

  // Use two static counters to record the number of ordered and unordered
  // locks already processed. On each match, the offset of the current op
  // is computed based on these counters (which represent the locks preceding it).
  inline static size_t orderedCount = 0;
  inline static size_t unorderedCount = 0;

  LogicalResult matchAndRewrite(hivm::CreateSyncBlockLockOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getLockArg()) {
      return op->emitOpError("failed to bind sync block lock argument");
    }

    auto loc = op.getLoc();

    // Calculate the basic stride per lock (excluding the block_num multiplier)
    auto bindArgTypeWith =
        getElementTypeOrSelf(op.getLockArg()).getIntOrFloatBitWidth();
    auto lockResTypeWith =
        getElementTypeOrSelf(op.getMemref().getType()).getIntOrFloatBitWidth();
    int64_t perOffset = CEIL_DIV(lockResTypeWith, bindArgTypeWith);
    // Ordered lock stride in bytes = perOffset * 8
    int64_t orderedStep = perOffset * 8;
    // Unordered lock stride in bytes (without block_num) = perOffset * 8 * cacheLines
    bool isUnordered = op->hasAttr(SyncBlockLockUnorderedAttr::name);
    int64_t unorderedCacheLines = isUnordered ? hivm::kUnorderedSyncBlockLockCacheLines : 1;
    int64_t unorderedStep = perOffset * 8 * unorderedCacheLines;

    // ----- Compute the start offset for the current lock -----
    // offset = orderedCount * orderedStep + unorderedCount * unorderedStep * block_num
    // Note: unorderedStep here is the fixed stride per unordered lock (without block_num),
    // unorderedCount * unorderedStep is the total constant stride for all preceding
    // unordered locks (compile-time constant).

    // 1. Ordered part offset in bytes (i64)
    Value orderedPart = rewriter.create<arith::ConstantIntOp>(
        loc, (int64_t)(orderedCount * orderedStep), 64);
    // 2. Unordered part constant product (i64)
    Value unorderedConst = rewriter.create<arith::ConstantIntOp>(
        loc, (int64_t)(unorderedCount * unorderedStep), 64);
    // 3. Get block_num
    Value blockNum = rewriter.create<hivm::GetBlockNumOp>(loc)->getResult(0);
    // 4. Unordered part total bytes = unorderedConst * block_num (i64)
    Value unorderedPart = rewriter.create<arith::MulIOp>(loc, unorderedConst, blockNum);
    // 5. Total bytes = orderedPart + unorderedPart
    Value totalByte = rewriter.create<arith::AddIOp>(loc, orderedPart, unorderedPart);
    // 6. Cast to index
    Value offsetIndex = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), totalByte);

    // Create the view with dynamic byte offset
    auto viewOp = rewriter.create<memref::ViewOp>(
        loc, op.getType(), op.getLockArg(),
        /*byte_shift*/ offsetIndex, /*dynamic_sizes*/ ValueRange{});

    // Update counters: the current lock has been processed, increment the corresponding counter
    if (isUnordered) {
      unorderedCount++;
    } else {
      orderedCount++;
    }

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

  // Reset static counters
  LowerCreateSyncBlockLock::orderedCount = 0;
  LowerCreateSyncBlockLock::unorderedCount = 0;

  patterns.add<LowerCreateSyncBlockLock>(&getContext());
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createSyncBlockLockLoweringPass() {
  return std::make_unique<LowerCreateSyncBlockLockPass>();
}