//===- LowerMultiBufferCounter.cpp - Lower multi-buffer counters ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

memref::AllocaOp findExistingCounterAlloca(FunctionOpInterface funcOp,
                                           IntegerAttr loopId) {
  if (!funcOp || funcOp.getFunctionBody().empty())
    return {};

  Block &entry = funcOp.getFunctionBody().front();
  for (auto &op : entry) {
    auto alloca = dyn_cast<memref::AllocaOp>(&op);
    if (!alloca)
      continue;
    auto counterAttr =
        alloca->getAttrOfType<IntegerAttr>(kMultiBufferCounterAttr);
    if (counterAttr == loopId)
      return alloca;
  }
  return {};
}

bool hasExistingCounterStore(LoopLikeOpInterface loop,
                             memref::AllocaOp alloca) {
  Block *body = getLoopBodyBlock(loop);
  if (!body)
    return false;
  for (auto &op : *body) {
    auto store = dyn_cast<memref::StoreOp>(&op);
    if (store && store.getMemref() == alloca.getResult())
      return true;
  }
  return false;
}

struct CounterLoweringState {
  LoopLikeOpInterface loop;
  IntegerAttr loopId;
  memref::AllocaOp alloca;
  Value firstLoad;
};

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

  llvm::DenseMap<Operation *, CounterLoweringState> states;
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

    IntegerAttr loopId = counterOp.getLoopIdAttr();
    Operation *loopOp = loop.getOperation();
    if (auto existingLoopId =
            loopOp->getAttrOfType<IntegerAttr>(kMultiBufferLoopIdAttr)) {
      if (existingLoopId != loopId) {
        counterOp.emitError("multi-buffer counter loop_id does not match "
                            "owning loop id");
        signalPassFailure();
        return;
      }
    } else {
      loopOp->setAttr(kMultiBufferLoopIdAttr, loopId);
    }

    CounterLoweringState &state = states[loopOp];
    if (!state.loop) {
      state.loop = loop;
      state.loopId = loopId;
      state.alloca = findExistingCounterAlloca(funcOp, loopId);
      if (!state.alloca) {
        rewriter.setInsertionPointToStart(&funcOp.getBody().front());
        state.alloca = rewriter.create<memref::AllocaOp>(counterOp.getLoc(),
                                                         memTy);
        state.alloca->setAttr(kMultiBufferCounterAttr, loopId);

        Value zero =
            rewriter.create<arith::ConstantIntOp>(counterOp.getLoc(), i64Ty,
                                                  /*value=*/0);
        Value zeroIdx =
            rewriter.create<arith::ConstantIndexOp>(counterOp.getLoc(), 0);
        rewriter.create<memref::StoreOp>(counterOp.getLoc(), zero,
                                         state.alloca.getResult(),
                                         ValueRange{zeroIdx});
      }
    } else if (state.loopId != loopId) {
      counterOp.emitError("multiple multi-buffer counter ids found for one "
                          "loop");
      signalPassFailure();
      return;
    }

    rewriter.setInsertionPoint(counterOp);
    Value zeroIdx =
        rewriter.create<arith::ConstantIndexOp>(counterOp.getLoc(), 0);
    auto load = rewriter.create<memref::LoadOp>(counterOp.getLoc(),
                                                state.alloca.getResult(),
                                                ValueRange{zeroIdx});
    if (!state.firstLoad)
      state.firstLoad = load.getResult();
    rewriter.replaceOp(counterOp, load.getResult());
  }

  for (auto &it : states) {
    CounterLoweringState &state = it.second;
    if (hasExistingCounterStore(state.loop, state.alloca))
      continue;

    Block *body = getLoopBodyBlock(state.loop);
    Operation *terminator = body->getTerminator();
    rewriter.setInsertionPoint(terminator);
    Value one = rewriter.create<arith::ConstantIntOp>(
        terminator->getLoc(), i64Ty, /*value=*/1);
    Value next =
        rewriter.create<arith::AddIOp>(terminator->getLoc(), state.firstLoad,
                                       one);
    Value zeroIdx =
        rewriter.create<arith::ConstantIndexOp>(terminator->getLoc(), 0);
    rewriter.create<memref::StoreOp>(terminator->getLoc(), next,
                                     state.alloca.getResult(),
                                     ValueRange{zeroIdx});
  }
}

std::unique_ptr<Pass> mlir::hivm::createLowerMultiBufferCounterPass() {
  return std::make_unique<LowerMultiBufferCounterPass>();
}
