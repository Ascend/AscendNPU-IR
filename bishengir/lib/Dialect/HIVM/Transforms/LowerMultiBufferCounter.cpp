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

Block *getLoopBodyBlock(LoopLikeOpInterface loop) {
  if (!loop)
    return nullptr;
  Operation *op = loop.getOperation();
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return forOp.getBody();
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return &whileOp.getAfter().front();
  return nullptr;
}

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

  // A loop owns at most one lowered counter. Dedup all counter ops of the same
  // loop onto the first one so the alloca/init/increment triplet is emitted
  // exactly once per loop; later clones just reuse that load value.
  llvm::DenseMap<Operation *, Value> loweredCounter;
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
    Block *body = getLoopBodyBlock(loop);
    if (!body || !body->getTerminator()) {
      counterOp.emitError("multi-buffer counter loop body has no terminator");
      signalPassFailure();
      return;
    }
    Operation *loopOp = loop.getOperation();

    // Reuse path: a sibling counter op for this loop is already lowered.
    if (auto it = loweredCounter.find(loopOp); it != loweredCounter.end()) {
      rewriter.replaceOp(counterOp, it->second);
      continue;
    }

    // Fresh loop: function-scoped alloca + zero-init at the entry block.
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    auto alloca = rewriter.create<memref::AllocaOp>(counterOp.getLoc(), memTy);
    Value zero = rewriter.create<arith::ConstantIntOp>(
        counterOp.getLoc(), /*value=*/0, i64Ty);
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
    Value one = rewriter.create<arith::ConstantIntOp>(
        terminator->getLoc(), /*value=*/1, i64Ty);
    Value next =
        rewriter.create<arith::AddIOp>(terminator->getLoc(), counterVal, one);
    Value storeIdx =
        rewriter.create<arith::ConstantIndexOp>(terminator->getLoc(), 0);
    rewriter.create<memref::StoreOp>(terminator->getLoc(), next,
                                     alloca.getResult(), ValueRange{storeIdx});

    loweredCounter.try_emplace(loopOp, counterVal);
    rewriter.replaceOp(counterOp, counterVal);
  }
}

std::unique_ptr<Pass> mlir::hivm::createLowerMultiBufferCounterPass() {
  return std::make_unique<LowerMultiBufferCounterPass>();
}
