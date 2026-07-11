//===------------- LoopInvariantPromotion.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <memory>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_LOOPINVARIANTPROMOTION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace {
struct SubsetInfo {
  SmallVector<Value> indices;
  VectorType vecTy;
  AffineMap permMap;
  Value mask, pad;
  ArrayAttr inBounds;
  bool operator!=(const SubsetInfo &rhs) const {
    // pad only compare when owner is read.
    return !llvm::equal(indices, rhs.indices) || vecTy != rhs.vecTy ||
           permMap != rhs.permMap || mask != rhs.mask ||
           inBounds != rhs.inBounds;
  }
  bool operator==(const SubsetInfo &rhs) const { return !(*this != rhs); }
};

struct Candidate {
  unsigned idx;
  SmallVector<vector::TransferReadOp> reads;
  SmallVector<vector::TransferWriteOp> writes;
  Value yield;
  bool endedInWrite;
  SubsetInfo subset;
};

struct LoopInvariantPromotion
    : public impl::LoopInvariantPromotionBase<LoopInvariantPromotion> {
  void runOnOperation() override;
};

static bool isDefinedOutsideLoop(LoopLikeOpInterface loop, Value v) {
  return !v || loop.isDefinedOutsideOfLoop(v);
}

static std::optional<Candidate> analyze(scf::ForOp forOp, unsigned idx) {
  BlockArgument arg = forOp.getRegionIterArg(idx);
  if (!isa<RankedTensorType>(arg.getType())) // Only solve tensor type
    return std::nullopt;
  Candidate cand;
  cand.idx = idx;
  SubsetInfo &s = cand.subset;
  bool haveSubsset = false;
  auto match = [&](ValueRange indices, VectorType vecTy, AffineMap map,
                   Value mask, ArrayAttr inBoudns, bool isRead,
                   Value pad) -> bool {
    for (Value i : indices)
      if (!isDefinedOutsideLoop(forOp, i))
        return false;
    if (!isDefinedOutsideLoop(forOp, mask) ||
        // For transfer_read we need to check padding.
        (isRead && !isDefinedOutsideLoop(forOp, pad)))
      return false;
    SubsetInfo temp = {indices, vecTy, map, mask, pad, inBoudns};
    if (!haveSubsset)
      s = temp, haveSubsset = true;
    else if (s != temp)
      return false;

    if (isRead) {
      if (s.pad == nullptr)
        s.pad = pad;
      else if (s.pad != pad)
        return false;
    }
    return true;
  };
  // 1. Check the legality of promotion and collect optimization instructions.
  SmallVector<Value> worklist{arg};
  while (!worklist.empty()) {
    Value t = worklist.pop_back_val();
    for (OpOperand &use : t.getUses()) {
      Operation *owner = use.getOwner();
      assert(owner->getBlock() == forOp.getBody());
      // TODO only support same index yield.
      if (owner == forOp.getBody()->getTerminator()) {
        if (use.getOperandNumber() != idx)
          return std::nullopt;
        continue;
      }
      if (auto rd = dyn_cast<vector::TransferReadOp>(owner)) {
        assert(rd.getSource() == t);
        if (!match(rd.getIndices(), rd.getVectorType(), rd.getPermutationMap(),
                   rd.getMask(), rd.getInBoundsAttr(), true, rd.getPadding()))
          return std::nullopt;
        cand.reads.push_back(rd);
        continue;
      }
      if (auto wr = dyn_cast<vector::TransferWriteOp>(owner)) {
        assert(wr.getSource() == t && wr.getResult() != nullptr);
        if (!match(wr.getIndices(), wr.getVectorType(), wr.getPermutationMap(),
                   wr.getMask(), wr.getInBoundsAttr(), false, {}))
          return std::nullopt;
        cand.writes.push_back(wr);
        worklist.push_back(wr.getResult());
        continue;
      }
      // Default
      return std::nullopt;
    }
  }
  // illiagle instruction arith.select(mask: vector<4x8xi1>, value:
  // vector<8x4xf32>, ...)
  if (cand.reads.empty() ||
      (s.mask &&
       cast<VectorType>(s.mask.getType()).getShape() != s.vecTy.getShape()))
    return std::nullopt;

  // 2. Classify the yield value.
  Value yield = forOp.getYieldedValues()[idx];
  cand.yield = yield;
  if (yield == arg)
    cand.endedInWrite = false;
  else if (auto *def = yield.getDefiningOp();
           isa_and_nonnull<vector::TransferWriteOp>(def) &&
           llvm::is_contained(cand.writes, cast<vector::TransferWriteOp>(def)))
    cand.endedInWrite = true;
  else
    return std::nullopt;

  return cand;
}

static LogicalResult promote(IRRewriter &rewriter, scf::ForOp forOp,
                             Candidate cand) {
  auto &s = cand.subset;
  auto loc = forOp.getLoc();

  // 1. Create hoist transfer_read.
  assert(!cand.reads.empty());
  rewriter.setInsertionPoint(forOp);
  auto hoistedRead = cast<vector::TransferReadOp>(
      rewriter.clone(*cand.reads.front().getOperation()));
  hoistedRead.getSourceMutable().assign(forOp.getInitArgs()[cand.idx]);
  Value padBrc;
  if (s.mask)
    padBrc = rewriter.create<vector::BroadcastOp>(loc, s.vecTy, s.pad);

  // For reads with padding, we need to legalize them.
  auto legalize = [&](Value r) -> Value {
    if (!s.mask)
      return r;
    return rewriter.create<arith::SelectOp>(loc, s.mask, r, padBrc);
  };
  auto removeDeadWrites =
      [&rewriter](ArrayRef<vector::TransferWriteOp> writes) {
        for (auto w : reverse(writes))
          if (w->use_empty())
            rewriter.eraseOp(w);
      };
  auto mem2reg = [&](DenseMap<Value, Value> &avail) {
    for (auto w : cand.writes)
      avail[w.getResult()] = w.getVector();
    for (auto r : cand.reads) {
      Value v = avail.lookup(r.getSource());
      assert(v != nullptr);
      rewriter.setInsertionPoint(r);
      rewriter.replaceAllUsesWith(r.getResult(), legalize(v));
    }
    for (auto r : cand.reads)
      rewriter.eraseOp(r);
  };

  // 2. Rewrite if there is no write on the def-use chain.
  if (!cand.endedInWrite) {
    BlockArgument arg = forOp.getRegionIterArg(cand.idx);
    DenseMap<Value, Value> avail;
    avail[arg] = hoistedRead.getResult();
    mem2reg(avail);
    removeDeadWrites(cand.writes);
    // No modifications have been made.
    rewriter.replaceAllUsesWith(forOp.getResult(cand.idx),
                                forOp.getInitArgs()[cand.idx]);
    return success();
  }

  // 3. Rewrite if there have write on the def-use chain.
  auto maybe = forOp.replaceWithAdditionalYields(
      rewriter, {hoistedRead.getResult()}, false,
      [&](OpBuilder &, Location,
          ArrayRef<BlockArgument>) -> SmallVector<Value> {
        return {cast<vector::TransferWriteOp>(cand.yield.getDefiningOp())
                    .getVector()};
      });
  if (failed(maybe)) {
    rewriter.eraseOp(hoistedRead);
    if (padBrc)
      rewriter.eraseOp(padBrc.getDefiningOp());
    return failure();
  }
  auto newFor = cast<scf::ForOp>(maybe->getOperation());
  DenseMap<Value, Value> avail;
  BlockArgument newArg = newFor.getRegionIterArgs().back(),
                idxArg = newFor.getRegionIterArg(cand.idx);
  avail[idxArg] = newArg;
  mem2reg(avail);
  Operation *yieldOp = cast<scf::YieldOp>(newFor.getBody()->getTerminator());
  // canonicalize will remove it.
  rewriter.modifyOpInPlace(yieldOp,
                           [&] { yieldOp->setOperand(cand.idx, idxArg); });

  // Reconstruct the tensor outside the loop: write the final register value
  // (the added vector result) back into the init, and redirect the loop's
  // tensor result to it.
  rewriter.setInsertionPointAfter(newFor);
  Value vecRes = newFor->getResults().back();
  auto newWrite = cast<vector::TransferWriteOp>(
      rewriter.clone(*cand.yield.getDefiningOp()));
  newWrite.getVectorMutable().assign(vecRes);
  newWrite.getSourceMutable().assign(newFor.getInitArgs()[cand.idx]);
  rewriter.replaceAllUsesWith(newFor.getResult(cand.idx), newWrite.getResult());
  removeDeadWrites(cand.writes);
  return success();
}

} // namespace

void LoopInvariantPromotion::runOnOperation() {
  func::FuncOp func = getOperation();
  // Only optimize the vf function.
  if (!func->hasAttr("hivm.vector_function"))
    return;

  IRRewriter rewriter(&getContext());
  bool changed = true;
  while (changed) {
    func.walk([](LoopLikeOpInterface loop) { moveLoopInvariantCode(loop); });
    WalkResult res = func.walk([&](scf::ForOp forOp) {
      for (unsigned i = 0, e = forOp.getNumRegionIterArgs(); i != e; ++i) {
        std::optional<Candidate> cand = analyze(forOp, i);
        if (!cand)
          continue;
        if (succeeded(promote(rewriter, forOp, *cand)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    changed = res.wasInterrupted();
  }
}

std::unique_ptr<Pass> mlir::hfusion::createLoopInvariantPromotionPass() {
  return std::make_unique<LoopInvariantPromotion>();
}