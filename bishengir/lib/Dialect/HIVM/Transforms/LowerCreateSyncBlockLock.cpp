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
  inline static size_t totalOrderedCount = 0;

  LogicalResult matchAndRewrite(hivm::CreateSyncBlockLockOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getLockArg()) {
      return op->emitOpError("failed to bind sync block lock argument");
    }

    auto loc = op.getLoc();

    constexpr int64_t cacheLineBytes = 64;
    bool isUnordered = op->hasAttr(SyncBlockLockUnorderedAttr::name);
    Value totalByte;
    if (isUnordered) {
      Value blockNum =
          rewriter.create<hivm::GetBlockNumOp>(loc)->getResult(0);
      if (hivm::isMixModule(op->getParentOfType<ModuleOp>())) {
        Value subBlockNum = rewriter.create<hivm::GetSubBlockNumOp>(
            loc, rewriter.getI64Type());
        blockNum = rewriter.create<arith::MulIOp>(loc, blockNum, subBlockNum);
      }
      Value two = rewriter.create<arith::ConstantIntOp>(loc, 2, 64);
      Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
      Value cacheLineBytesVal = rewriter.create<arith::ConstantIntOp>(
          loc, cacheLineBytes, 64);
      Value participantCacheLines = rewriter.create<arith::AddIOp>(
          loc, rewriter.create<arith::MulIOp>(loc, blockNum, two), one);
      Value unorderedStride = rewriter.create<arith::MulIOp>(
          loc, participantCacheLines, cacheLineBytesVal);
      Value unorderedIndex = rewriter.create<arith::ConstantIntOp>(
          loc, static_cast<int64_t>(unorderedCount), 64);
      Value unorderedPart = rewriter.create<arith::MulIOp>(
          loc, unorderedStride, unorderedIndex);
      Value orderedRegion = rewriter.create<arith::ConstantIntOp>(
          loc, static_cast<int64_t>(totalOrderedCount) * cacheLineBytes, 64);
      totalByte = rewriter.create<arith::AddIOp>(
          loc, orderedRegion, unorderedPart);
      ++unorderedCount;
    } else {
      totalByte = rewriter.create<arith::ConstantIntOp>(
          loc, static_cast<int64_t>(orderedCount) * cacheLineBytes, 64);
      ++orderedCount;
    }

    Value offsetIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), totalByte);

    // Create the view with dynamic byte offset
    auto viewOp = rewriter.create<memref::ViewOp>(
        loc, op.getType(), op.getLockArg(),
        /*byte_shift*/ offsetIndex, /*dynamic_sizes*/ ValueRange{});

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
  LowerCreateSyncBlockLock::totalOrderedCount = 0;
  funcOp.walk([](hivm::CreateSyncBlockLockOp op) {
    if (!op->hasAttr(SyncBlockLockUnorderedAttr::name))
      ++LowerCreateSyncBlockLock::totalOrderedCount;
  });

  patterns.add<LowerCreateSyncBlockLock>(&getContext());
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createSyncBlockLockLoweringPass() {
  return std::make_unique<LowerCreateSyncBlockLockPass>();
}
