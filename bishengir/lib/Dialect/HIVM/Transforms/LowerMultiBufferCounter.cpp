//===- LowerMultiBufferCounter.cpp - Lower multi-buffer counters ----------===//
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

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
#define GEN_PASS_DEF_LOWERMULTIBUFFERCOUNTER
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct LowerMultiBufferCounterPass
    : public impl::LowerMultiBufferCounterBase<LowerMultiBufferCounterPass> {
  void runOnOperation() override;
};

} // namespace

void LowerMultiBufferCounterPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  SmallVector<hivm::MultiBufferCounterOp> counterOps;
  funcOp.walk([&](hivm::MultiBufferCounterOp counterOp) {
    counterOps.push_back(counterOp);
  });
  if (counterOps.empty())
    return;

  IRRewriter rewriter(funcOp.getContext());
  Type i64Ty = rewriter.getI64Type();
  auto memTy = MemRefType::get(/*shape=*/{1}, i64Ty);

  // Dedup by the block that owns the counter. For scf.while this means before
  // and after each keep an independent load/increment (SSA values do not
  // dominate across those regions). Sibling counters in the *same* block still
  // collapse onto one alloca/load/increment triplet.
  llvm::DenseMap<Block *, Value> loweredCounter;
  for (hivm::MultiBufferCounterOp counterOp : counterOps) {
    LoopLikeOpInterface loop =
        counterOp->getParentOfType<LoopLikeOpInterface>();
    if (!isa_and_nonnull<scf::ForOp, scf::WhileOp>(
            loop ? loop.getOperation() : nullptr)) {
      counterOp.emitError("multi-buffer counter must be inside scf.for or "
                          "scf.while");
      signalPassFailure();
      return;
    }
    // Use the counter's own block — not a hardcoded while-after — so a
    // before-region counter increments before scf.condition and an
    // after-region counter increments before scf.yield.
    Block *body = counterOp->getBlock();
    if (!body || !body->getTerminator()) {
      counterOp.emitError("multi-buffer counter loop body has no terminator");
      signalPassFailure();
      return;
    }

    // Reuse path: a sibling counter op in this block is already lowered.
    if (auto it = loweredCounter.find(body); it != loweredCounter.end()) {
      rewriter.replaceOp(counterOp, it->second);
      continue;
    }

    // Fresh body block: function-scoped alloca + zero-init at the entry block.
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    auto alloca = rewriter.create<memref::AllocaOp>(counterOp.getLoc(), memTy);
#ifndef BSPUB_DAVINCI_BISHENGIR_A5
    Value zero = rewriter.create<arith::ConstantIntOp>(
        counterOp.getLoc(), /*value=*/0, i64Ty);
#else
    Value zero = rewriter.create<arith::ConstantIntOp>(
        counterOp.getLoc(), i64Ty, /*value=*/0);
#endif
    Value initIdx =
        rewriter.create<arith::ConstantIndexOp>(counterOp.getLoc(), 0);
    rewriter.create<memref::StoreOp>(counterOp.getLoc(), zero,
                                     alloca.getResult(), ValueRange{initIdx});

    // Body-head load replaces the anchor op.
    rewriter.setInsertionPoint(counterOp);
    Value loadIdx =
        rewriter.create<arith::ConstantIndexOp>(counterOp.getLoc(), 0);
    auto load = rewriter.create<memref::LoadOp>(
        counterOp.getLoc(), alloca.getResult(), ValueRange{loadIdx});
    Value counterVal = load.getResult();

    // Body-tail increment/store keeps the counter monotonically increasing.
    Operation *terminator = body->getTerminator();
    rewriter.setInsertionPoint(terminator);
#ifndef BSPUB_DAVINCI_BISHENGIR_A5
    Value one = rewriter.create<arith::ConstantIntOp>(
        terminator->getLoc(), /*value=*/1, i64Ty);
#else
    Value one = rewriter.create<arith::ConstantIntOp>(
        terminator->getLoc(), i64Ty, /*value=*/1);
#endif
    Value next =
        rewriter.create<arith::AddIOp>(terminator->getLoc(), counterVal, one);
    Value storeIdx =
        rewriter.create<arith::ConstantIndexOp>(terminator->getLoc(), 0);
    rewriter.create<memref::StoreOp>(terminator->getLoc(), next,
                                     alloca.getResult(), ValueRange{storeIdx});

    loweredCounter.try_emplace(body, counterVal);
    rewriter.replaceOp(counterOp, counterVal);
  }
}

std::unique_ptr<Pass> mlir::hivm::createLowerMultiBufferCounterPass() {
  return std::make_unique<LowerMultiBufferCounterPass>();
}
