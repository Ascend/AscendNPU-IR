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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <functional>
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
  SmallVector<SmallVector<OpFoldResult>> offsets, sizes, strides;
  SmallVector<Value> indices;
  VectorType vecTy;
  AffineMap permMap;
  Value mask, pad;
  ArrayAttr inBounds;

  bool operator!=(const SubsetInfo &rhs) {
    if (offsets.size() != rhs.offsets.size())
      return true;
    assert(sizes.size() == rhs.sizes.size() &&
           strides.size() == rhs.strides.size());
    auto isSameOFRs = [](ArrayRef<OpFoldResult> lhs,
                         ArrayRef<OpFoldResult> rhs) {
      return lhs.size() == rhs.size() && llvm::equal(lhs, rhs);
    };
    for (unsigned i = 0, e = offsets.size(); i != e; ++i) {
      if (!isSameOFRs(offsets[i], rhs.offsets[i]) ||
          !isSameOFRs(sizes[i], rhs.sizes[i]) ||
          !isSameOFRs(strides[i], rhs.strides[i]))
        return true;
    }
    // pad is not check here.
    return !llvm::equal(indices, rhs.indices) || vecTy != rhs.vecTy ||
           permMap != rhs.permMap || mask != rhs.mask ||
           inBounds != rhs.inBounds;
  }
};

struct Candidate {
  unsigned idx;
  SmallVector<vector::TransferReadOp> reads;
  SmallVector<Operation *> threadOps;
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

static SmallVector<tensor::ExtractSliceOp> extractChainOf(Value v) {
  SmallVector<tensor::ExtractSliceOp> chain;
  while (auto es = v.getDefiningOp<tensor::ExtractSliceOp>()) {
    chain.push_back(es);
    v = es.getSource();
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

static std::optional<Candidate> analyze(scf::ForOp forOp, unsigned idx) {
  BlockArgument arg = forOp.getRegionIterArg(idx);
  if (!isa<RankedTensorType>(arg.getType())) // Only solve tensor type
    return std::nullopt;
  Candidate cand;
  cand.idx = idx;
  SubsetInfo &s = cand.subset;
  bool haveSubset = false;

  auto isInvariant = [&](ArrayRef<OpFoldResult> array) {
    for (auto ofr : array)
      if (auto v = dyn_cast<Value>(ofr))
        if (!isDefinedOutsideLoop(forOp, v))
          return false;
    return true;
  };
  auto match = [&](Value src, ValueRange indices, VectorType vecTy,
                   AffineMap map, Value mask, ArrayAttr inBounds, bool isRead,
                   Value pad) {
    for (Value i : indices)
      if (!isDefinedOutsideLoop(forOp, i))
        return false;
    if (!isDefinedOutsideLoop(forOp, mask) ||
        // For transfer_read we need to check padding.
        (isRead && !isDefinedOutsideLoop(forOp, pad)))
      return false;
    SmallVector<SmallVector<OpFoldResult>> offsets, sizes, strides;
    for (auto es : extractChainOf(src)) {
      auto offs = es.getMixedOffsets(), szs = es.getMixedSizes(),
           strs = es.getMixedStrides();
      if (!isInvariant(offs) || !isInvariant(szs) || !isInvariant(strs))
        return false;
      offsets.push_back(std::move(offs));
      sizes.push_back(std::move(szs));
      strides.push_back(std::move(strs));
    }
    SubsetInfo temp = {offsets, sizes, strides, indices, vecTy,
                       map,     mask,  pad,     inBounds};
    if (!haveSubset)
      s = temp, haveSubset = true;
    else if (s != temp)
      return false;
    if (isRead) {
      if (!s.pad)
        s.pad = pad;
      else if (s.pad != pad)
        return false;
    }
    return true;
  };

  DenseSet<Operation *> handled;
  DenseSet<Value> visited;
  SmallVector<Value> worklist{arg};
  auto record = [&](Operation *op) {
    if (handled.insert(op).second)
      cand.threadOps.push_back(op);
  };
  // 1. Check the legality of promotion and collect optimization instructions.
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    visited.insert(v);
    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();
      if (owner == forOp.getBody()->getTerminator()) {
        if (use.getOperandNumber() != idx)
          return std::nullopt;
        continue;
      }
      // A thread value used outside the loop's own body (e.g. captured into a
      // nested scf.for / scf.if) is not a straight-line access we can promote.
      if (owner->getBlock() != forOp.getBody())
        return std::nullopt;
      if (auto rd = dyn_cast<vector::TransferReadOp>(owner)) {
        assert(rd.getSource() == v);
        if (!match(v, rd.getIndices(), rd.getVectorType(),
                   rd.getPermutationMap(), rd.getMask(), rd.getInBoundsAttr(),
                   true, rd.getPadding()))
          return std::nullopt;
        cand.reads.push_back(rd);
        record(rd);
        // As the endpoint.
        continue;
      }
      if (auto wr = dyn_cast<vector::TransferWriteOp>(owner)) {
        assert(wr.getSource() == v && wr.getResult() != nullptr);
        if (!match(v, wr.getIndices(), wr.getVectorType(),
                   wr.getPermutationMap(), wr.getMask(), wr.getInBoundsAttr(),
                   false, {}))
          return std::nullopt;
        worklist.push_back(wr.getResult());
        record(wr);
        continue;
      }
      if (auto es = dyn_cast<tensor::ExtractSliceOp>(owner)) {
        assert(es.getSource() == v);
        worklist.push_back(es.getResult());
        record(es);
        continue;
      }
      if (auto is = dyn_cast<tensor::InsertSliceOp>(owner)) {
        if (is.getSource() == v) {
          if (!isInvariant(is.getMixedOffsets()) ||
              !isInvariant(is.getMixedSizes()) ||
              !isInvariant(is.getMixedStrides()))
            return std::nullopt;
          worklist.push_back(is.getResult());
          record(is);
          continue;
        }
        if (is.getDest() == v)
          continue;
        return std::nullopt; // ????
      }
      // Default.
      return std::nullopt;
    }
  }

  // 2. Verify the closure value.
  for (auto v : visited) {
    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();
      if (owner == forOp.getBody()->getTerminator() &&
          use.getOperandNumber() == idx)
        continue;
      // For example, %1 = insert_slice %2 into %3[0,0][1,8], and %2 is not in
      // the closure. We should conservatively exit optimization.
      if (!handled.count(owner))
        return std::nullopt;
    }
  }
  // Illiagle instruction arith.select(mask: vector<4x8xi1>, value:
  // vector<8x4xf32>, ...).
  if (cand.reads.empty() ||
      (s.mask &&
       cast<VectorType>(s.mask.getType()).getShape() != s.vecTy.getShape()))
    return std::nullopt;

  // 3. Classify the yield value.
  Value yield = forOp.getYieldedValues()[idx];
  cand.yield = yield;
  cand.endedInWrite = yield != arg;
  if (cand.endedInWrite) {
    if (!visited.count(yield))
      return std::nullopt;
    Value u = yield;
    while (auto ins = u.getDefiningOp<tensor::InsertSliceOp>())
      u = ins.getSource();
    if (!u.getDefiningOp<vector::TransferWriteOp>())
      return std::nullopt;
  }
  return cand;
}

static LogicalResult promote(IRRewriter &rewriter, scf::ForOp forOp,
                             Candidate cand) {
  auto &s = cand.subset;
  auto loc = forOp.getLoc();

  auto rootOf = [](Value v) {
    while (auto es = v.getDefiningOp<tensor::ExtractSliceOp>())
      v = es.getSource();
    return v;
  };
  auto cloneRead = [&](vector::TransferReadOp r, Value base) {
    IRMapping map;
    map.map(rootOf(r.getSource()), base);
    for (tensor::ExtractSliceOp es : extractChainOf(r.getSource()))
      rewriter.clone(*es.getOperation(), map);
    return rewriter.clone(*r.getOperation(), map);
  };

  // 1. Create hoist transfer_read.
  assert(!cand.reads.empty());
  rewriter.setInsertionPoint(forOp);
  auto hoistedRead = cast<vector::TransferReadOp>(
      cloneRead(cand.reads.front(), forOp.getInitArgs()[cand.idx]));
  Value padBrc;
  if (s.mask)
    padBrc = rewriter.create<vector::BroadcastOp>(loc, s.vecTy, s.pad);

  // For reads with padding, we need to legalize them.
  auto legalize = [&](Value r) -> Value {
    if (!s.mask)
      return r;
    return rewriter.create<arith::SelectOp>(loc, s.mask, r, padBrc);
  };
  auto removeDeadOp = [&rewriter](ArrayRef<Operation *> ops) {
    for (auto *op : reverse(ops))
      if (op->use_empty())
        rewriter.eraseOp(op);
  };
  auto mem2ssa = [&](Value regBase, Value regVal) {
    DenseMap<Value, Value> avail;
    avail[regBase] = regVal;
    std::function<Value(Value)> help = [&](Value v) -> Value {
      if (auto it = avail.find(v); it != avail.end())
        return it->second;
      Operation *d = v.getDefiningOp();
      Value r;
      if (auto es = dyn_cast_or_null<tensor::ExtractSliceOp>(d))
        r = help(es.getSource());
      else if (auto w = dyn_cast_or_null<vector::TransferWriteOp>(d))
        r = w.getVector();
      else if (auto ins = dyn_cast_or_null<tensor::InsertSliceOp>(d))
        r = help(ins.getSource());
      else
        r = nullptr;
      avail[v] = r;
      return r;
    };
    for (auto r : cand.reads) {
      Value v = help(r.getSource());
      assert(v != nullptr);
      rewriter.setInsertionPoint(r);
      rewriter.replaceAllUsesWith(r.getResult(), legalize(v));
    }
  };

  // 2. Rewrite if there is no write on the def-use chain.
  if (!cand.endedInWrite) {
    mem2ssa(forOp.getRegionIterArg(cand.idx), hoistedRead.getResult());
    removeDeadOp(cand.threadOps);
    rewriter.replaceAllUsesWith(forOp.getResult(cand.idx),
                                forOp.getInitArgs()[cand.idx]);
    return success();
  }

  // 3. Rewrite if there have write on the def-use chain.
  Value y = cand.yield;
  while (auto ins = y.getDefiningOp<tensor::InsertSliceOp>())
    y = ins.getSource();
  auto result = forOp.replaceWithAdditionalYields(
      rewriter, {hoistedRead.getResult()}, false,
      [&](OpBuilder &, Location,
          ArrayRef<BlockArgument>) -> SmallVector<Value> {
        return {cast<vector::TransferWriteOp>(y.getDefiningOp()).getVector()};
      });
  if (failed(result))
    return failure();
  auto newFor = cast<scf::ForOp>(result->getOperation());
  BlockArgument newArg = newFor.getRegionIterArgs().back(),
                idxArg = newFor.getRegionIterArg(cand.idx);
  mem2ssa(idxArg, newArg);
  Operation *yieldOp = cast<scf::YieldOp>(newFor.getBody()->getTerminator());
  // Canonicalize will remove it.
  rewriter.modifyOpInPlace(yieldOp,
                           [&] { yieldOp->setOperand(cand.idx, idxArg); });

  // Reconstruct the tensor outside the loop: write the final register value
  // (the added vector result) back into the init, and redirect the loop's
  // tensor result to it.
  auto cloneWrite = [&](Value yield, Value base, Value vec) {
    SmallVector<tensor::InsertSliceOp> inserts;
    Value u = yield;
    while (auto ins = u.getDefiningOp<tensor::InsertSliceOp>()) {
      inserts.push_back(ins);
      u = ins.getSource();
    }
    auto w = u.getDefiningOp<vector::TransferWriteOp>();
    assert(w != nullptr);
    IRMapping map;
    map.map(rootOf(w.getSource()), base);
    map.map(w.getVector(), vec);
    for (auto es : extractChainOf(w.getSource()))
      rewriter.clone(*es.getOperation(), map);
    Value res = rewriter.clone(*w.getOperation(), map)->getResult(0);
    for (auto ins : reverse(inserts))
      res = rewriter.clone(*ins.getOperation(), map)->getResult(0);
    return res;
  };
  rewriter.setInsertionPointAfter(newFor);
  Value vecRes = newFor->getResults().back();
  Value res = cloneWrite(cand.yield, newFor.getInitArgs()[cand.idx], vecRes);
  rewriter.replaceAllUsesWith(newFor.getResult(cand.idx), res);
  removeDeadOp(cand.threadOps);
  return success();
}

static void hoistLoopInvariant(LoopLikeOpInterface loop) {
  moveLoopInvariantCode(
      loop.getLoopRegions(),
      [&](Value v, Region *) { return loop.isDefinedOutsideOfLoop(v); },
      [&](Operation *op, Region *) {
        if (auto rd = dyn_cast<vector::TransferReadOp>(op))
          return isa<RankedTensorType>(rd.getShapedType());
        return isMemoryEffectFree(op) && isSpeculatable(op);
      },
      [&](Operation *op, Region *) { loop.moveOutOfLoop(op); });
}

} // namespace

/// TODO
/// 1. The current equivalent dependency on multiple extract_Slice/insert_stice
///    is that the extract_Slice/insert_stice chains are completely equivalent,
///    which can be optimized for offset and strips calculations in the future.
/// 2. If there are only invariant writes in the loop, a sink optimization may
///    be needed.
/// 3. Currently relying on canonicalize to remove dead iter_args, if
///    compilation time is long, consider rewriting scf.for internally in Pass.
/// 4. There is still room for optimization in scenarios where iter_arg index
///    and yield index are different.
void LoopInvariantPromotion::runOnOperation() {
  func::FuncOp func = getOperation();
  // Only optimize the vf function.
  if (!func->hasAttr("hivm.vector_function"))
    return;

  FrozenRewritePatternSet cleanup = [&] {
    RewritePatternSet patterns(&getContext());
    scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
    return FrozenRewritePatternSet(std::move(patterns));
  }();

  IRRewriter rewriter(&getContext());
  bool changed = true;
  while (changed) {
    func.walk([](LoopLikeOpInterface loop) { hoistLoopInvariant(loop); });
    bool nested = false;
    WalkResult res = func.walk([&](scf::ForOp forOp) {
      for (unsigned i = 0, e = forOp.getNumRegionIterArgs(); i != e; ++i) {
        std::optional<Candidate> cand = analyze(forOp, i);
        if (!cand)
          continue;
        bool isNested = forOp->getParentOfType<scf::ForOp>() != nullptr;
        if (succeeded(promote(rewriter, forOp, *cand))) {
          nested = isNested;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    changed = res.wasInterrupted();
    if (changed && nested)
      (void)applyPatternsGreedily(func, cleanup);
  }
}

std::unique_ptr<Pass> mlir::hfusion::createLoopInvariantPromotionPass() {
  return std::make_unique<LoopInvariantPromotion>();
}