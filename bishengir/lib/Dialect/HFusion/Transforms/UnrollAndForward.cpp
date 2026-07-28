//===--------------- UnrollAndForward.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pattern-based loop unrolling for VF intermediate-tensor relay.
//
// Instead of unrolling every small trip-count scf.for, this pass:
//   (1) identifies chains of innermost sibling scf.for that share constant
//       bounds and form a producer→consumer chain on a tensor value T whose
//       UB materialization can be eliminated after unrolling;
//   (2) fully unrolls the entire chain;
//   (3) forwards transfer_write vectors directly to transfer_read consumers,
//       eliminating the intermediate-tensor round-trip.
//
// The forward step is much simpler than the old generic pattern because the
// chain guarantees: all loops share the same bounds and are unrolled together
// → the body is straight-line SSA, write and read operate on the same slice
// value → single-hop forward, no def-chain walk or same-source fallback.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_UNROLLANDFORWARD
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hfusion-unroll-and-forward"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::hfusion;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Check that every constant-having mixed offset resolves to an int64_t.
static std::optional<SmallVector<int64_t>>
getConstOffsets(ArrayRef<OpFoldResult> mixed) {
  SmallVector<int64_t> result;
  for (OpFoldResult ofr : mixed) {
    auto c = getConstantIntValue(ofr);
    if (!c)
      return std::nullopt;
    result.push_back(*c);
  }
  return result;
}

/// Check that all strides in `mixed` are unit (1). Non-unit strides mean the
/// slice is sparse (e.g. strided transpose) — forwarding a write vector to a
/// read on such a slice would misread the data layout, so we require dense
/// (unit-stride) slices.
static bool hasUnitStrides(ArrayRef<OpFoldResult> mixed) {
  for (OpFoldResult ofr : mixed) {
    auto c = getConstantIntValue(ofr);
    if (!c || *c != 1)
      return false;
  }
  return true;
}

/// True when `forOp` contains no nested scf.for / scf.while inside its body.
static bool isInnermost(scf::ForOp forOp) {
  auto result = forOp.getBody()->walk([](scf::ForOp) { return WalkResult::interrupt(); });
  return !result.wasInterrupted();
}

/// Return the constant trip count of forOp, or std::nullopt.
static std::optional<int64_t> getConstantTripCount(scf::ForOp forOp) {
  auto lb = getConstantIntValue(forOp.getLowerBound());
  auto ub = getConstantIntValue(forOp.getUpperBound());
  auto step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub <= *lb)
    return std::nullopt;
  return (*ub - *lb + *step - 1) / *step; // ceil
}

/// A (producer, consumer) loop pair plus the transfer_write→transfer_read
/// edge through an intermediate tensor slice.
struct TransferWriteReadPair {
  vector::TransferWriteOp write;
  vector::TransferReadOp read;
};

/// A group of sibling scf.for loops connected by a producer→consumer
/// intermediate tensor T that can be eliminated after unrolling.
struct UnrollableChain {
  SmallVector<scf::ForOp> loops;        // all loops in the chain (sorted by pos)
  Value intermedTensor;                 // the tensor T to eliminate
  scf::ForOp producer;                  // loop that produces T
  SmallVector<scf::ForOp> consumers;    // loops that read T
};

/// Check that the loops in a group are contiguous in their parent block —
/// no other scf.for is interleaved between consecutive group members. Non-loop
/// ops (constants, affine.apply) between them are fine; only other scf.for
/// would break the relay chain (a different loop in between could add register
/// pressure or side effects that make cross-loop relay unsafe).
static bool isContiguousGroup(ArrayRef<scf::ForOp> group) {
  if (group.size() < 2)
    return true;
  for (size_t i = 0; i + 1 < group.size(); ++i) {
    // Walk forward from group[i] to group[i+1]; reject if another scf.for
    // appears in between.
    for (Operation *op = group[i]->getNextNode(); op && op != group[i + 1];
         op = op->getNextNode()) {
      if (isa<scf::ForOp>(op))
        return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Intermediate-tensor eliminability checks
//===----------------------------------------------------------------------===//

/// Check that every transfer_write in the producer body flows through a
/// straight transfer_write → insert_slice → yield chain with no calls or
/// conditionals, and the written vector is a pure SSA def.
static bool hasSimpleWritePath(scf::ForOp producer, Value T) {
  for (Operation &op : producer.getBody()->getOperations()) {
    auto write = dyn_cast<vector::TransferWriteOp>(op);
    if (!write)
      continue;
    Value written = write.getResult();
    bool toYield = false;
    for (Operation *user : written.getUsers()) {
      if (auto ins = dyn_cast<tensor::InsertSliceOp>(user)) {
        // Reject sparse inserts (non-unit strides).
        if (!hasUnitStrides(ins.getMixedStrides()))
          return false;
        for (Operation *u2 : user->getResults()[0].getUsers())
          if (isa<scf::YieldOp>(u2)) {
            toYield = true;
            break;
          }
      }
    }
    if (!toYield)
      return false;
    Value vec = write.getVector();
    if (!vec.getDefiningOp() || isa<memref::LoadOp>(vec.getDefiningOp()))
      return false;
    for (Operation *user : written.getUsers()) {
      if (isa<func::CallOp, scf::IfOp>(user))
        return false;
    }
  }
  return true;
}

/// Check that every consumer reads T via tensor.extract_slice %T →
/// vector.transfer_read, with no calls or conditionals. Offsets may be
/// loop induction variables (they become constants after unrolling).
/// Non-extract_slice uses of T (e.g. escapes, debug) are not checked here —
/// isPureIntermediate guards against escape separately.
static bool hasSimpleReadPath(Value T,
                              ArrayRef<scf::ForOp> consumers) {
  for (Operation *user : T.getUsers()) {
    auto extract = dyn_cast<tensor::ExtractSliceOp>(user);
    if (!extract)
      continue; // non-read use (escape) — let isPureIntermediate decide
    for (OpFoldResult ofr : extract.getMixedOffsets()) {
      if (auto val = dyn_cast<Value>(ofr)) {
        if (!getConstantIntValue(val) && !isa<BlockArgument>(val))
          return false;
      }
    }
    // Reject sparse slices (non-unit strides) — forwarding would misread
    // the strided layout.
    if (!hasUnitStrides(extract.getMixedStrides()))
      return false;
    for (Operation *u2 : extract.getResult().getUsers()) {
      if (!isa<vector::TransferReadOp>(u2))
        return false;
      for (Operation *u3 : u2->getUsers()) {
        if (auto sel = dyn_cast<arith::SelectOp>(u3)) {
          if (!isa<arith::ConstantOp>(sel.getFalseValue().getDefiningOp()))
            return false;
          continue;
        }
        if (!isa<arith::ArithDialect>(u3->getDialect()) &&
            !isa<vector::VectorDialect>(u3->getDialect()))
          return false;
      }
    }
  }
  return true;
}

/// Check that the intermediate tensor T does not escape the chain: all uses
/// stay inside one of chainLoops, and T is not returned from the VF.
static bool isPureIntermediate(Value T, ArrayRef<scf::ForOp> chainLoops) {
  for (Operation *user : T.getUsers()) {
    Operation *ancestor = user->getParentOp();
    while (ancestor && !isa<scf::ForOp>(ancestor))
      ancestor = ancestor->getParentOp();
    if (!ancestor || !llvm::is_contained(chainLoops, cast<scf::ForOp>(ancestor)))
      return false;
  }
  auto funcOp = (*chainLoops.begin())->getParentOfType<func::FuncOp>();
  if (funcOp) {
    for (Block &block : funcOp.getBody()) {
      for (auto ret : block.getOps<func::ReturnOp>()) {
        for (Value retVal : ret.getOperands())
          if (retVal == T)
            return false;
      }
    }
  }
  return true;
}

/// Check that writes within the chain are disjoint — each column offset is
/// written by at most one loop, so the forward pass never has to choose
/// between conflicting writes.
static bool hasDisjointWrites(ArrayRef<scf::ForOp> chainLoops) {
  DenseMap<int64_t, scf::ForOp> sliceWriters;
  bool overlap = false;
  for (auto loop : chainLoops) {
    loop.getBody()->walk([&](tensor::InsertSliceOp ins) {
      auto offsets = getConstOffsets(ins.getMixedOffsets());
      if (!offsets || offsets->empty())
        return;
      int64_t key = offsets->back();
      auto it = sliceWriters.find(key);
      if (it != sliceWriters.end() && it->second != loop) {
        overlap = true;
        return;
      }
      sliceWriters[key] = loop;
    });
  }
  return !overlap;
}

/// Check that forwarding a transfer_write vector to a transfer_read is
/// mask-safe: either the masks are identical, both sides have no mask
/// (trip divides VL), or the read has a wrapping arith.select(mask, read, 0)
/// that zeros OOB lanes.
/// Check if two mask values are semantically equivalent — either the same SSA
/// value, or both are `vector.create_mask` with the same constant operands.
/// After unrolling, each iteration's mask is a distinct SSA op even though
/// the mask values are identical (e.g. `create_mask 1, 64` appears 3 times).
static bool masksAreEqual(Value a, Value b) {
  if (a == b)
    return true;
  if (!a || !b)
    return false;
  auto aOp = a.getDefiningOp<vector::CreateMaskOp>();
  auto bOp = b.getDefiningOp<vector::CreateMaskOp>();
  if (!aOp || !bOp)
    return false;
  if (aOp->getNumOperands() != bOp->getNumOperands())
    return false;
  for (auto [aArg, bArg] : llvm::zip(aOp->getOperands(), bOp->getOperands())) {
    auto aConst = getConstantIntValue(aArg);
    auto bConst = getConstantIntValue(bArg);
    if (!aConst || !bConst || *aConst != *bConst)
      return false;
  }
  return true;
}

static bool isMaskSafe(TransferWriteReadPair pair) {
  Value wMask = pair.write.getMask();
  Value rMask = pair.read.getMask();
  // Same SSA mask, or both absent → safe.
  if (masksAreEqual(wMask, rMask))
    return true;
  if (!wMask && !rMask)
    return true;
  // Check whether the read has a wrapping arith.select(mask, read, 0) that
  // zeros OOB lanes regardless of mask equality.
  for (Operation *user : pair.read->getUsers()) {
    auto sel = dyn_cast<arith::SelectOp>(user);
    if (!sel)
      continue;
    if (sel.getTrueValue() != pair.read.getResult())
      continue;
    if (sel.getCondition() != rMask)
      continue;
    auto falseCst = sel.getFalseValue().getDefiningOp<arith::ConstantOp>();
    if (!falseCst)
      continue;
    if (auto dense = dyn_cast<DenseFPElementsAttr>(falseCst.getValue()))
      if (dense.isSplat() && dense.getSplatValue<APFloat>().isZero())
        return true;
    if (auto dense = dyn_cast<DenseIntElementsAttr>(falseCst.getValue()))
      if (dense.isSplat() && dense.getSplatValue<APInt>().isZero())
        return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Chain matching
//===----------------------------------------------------------------------===//

/// Count the number of operations in a loop body (excluding the loop itself
/// and its yield terminator).
static unsigned countBodyOps(scf::ForOp forOp) {
  unsigned count = 0;
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (!isa<scf::YieldOp>(op))
      count++;
  }
  return count;
}

/// Match unrollable producer→consumer chains in a vector_function.
static SmallVector<UnrollableChain>
matchUnrollableChains(func::FuncOp vfFunc, unsigned maxUnroll,
                      unsigned maxBodyOps) {
  SmallVector<UnrollableChain> chains;

  // Step 1: collect innermost scf.for with constant bounds (walk recursively,
  //  since the loops may be nested inside an outer row loop in the VF).
  SmallVector<scf::ForOp> innermost;
  vfFunc.getBody().walk([&](scf::ForOp forOp) {
    if (!isInnermost(forOp))
      return;
    if (!getConstantTripCount(forOp))
      return;
    // Reject large bodies — unrolling would explode code size even if the
    // chain pattern matches. Only unroll small, tight loop bodies.
    if (countBodyOps(forOp) > maxBodyOps)
      return;
    innermost.push_back(forOp);
  });
  if (innermost.size() < 2)
    return chains; // need at least producer + consumer

  // Step 2: group by (lb, ub, step).
  struct BoundsKey {
    int64_t lb, ub, step;
    bool operator==(const BoundsKey &o) const {
      return lb == o.lb && ub == o.ub && step == o.step;
    }
    bool operator<(const BoundsKey &o) const {
      if (lb != o.lb) return lb < o.lb;
      if (ub != o.ub) return ub < o.ub;
      return step < o.step;
    }
  };
  std::map<BoundsKey, SmallVector<scf::ForOp>> groups;
  for (auto forOp : innermost) {
    auto trip = *getConstantTripCount(forOp);
    auto lb = *getConstantIntValue(forOp.getLowerBound());
    auto ub = *getConstantIntValue(forOp.getUpperBound());
    auto step = *getConstantIntValue(forOp.getStep());
    if (static_cast<uint64_t>(trip) < 2 || static_cast<uint64_t>(trip) > maxUnroll)
      continue;
    groups[{lb, ub, step}].push_back(forOp);
  }

  // Step 3: find producer→consumer chains within each group.
  for (auto &[bounds, group] : groups) {
    if (group.size() < 2)
      continue;
    // Conservative: require group loops to be contiguous in the parent block
    // (no interleaved scf.for). A non-group loop in between could add register
    // pressure or side effects that make cross-loop relay unsafe.
    if (!isContiguousGroup(group))
      continue;
    // hasDisjointWrites depends only on the group (not on T), so check once
    // per group instead of once per producer.
    if (!hasDisjointWrites(group))
      continue;
    for (auto P : group) {
      for (Value T : P.getResults()) {
        // Find consumers: other loops in the group that read T via
        // extract_slice → transfer_read. The matching transfer_write
        // lives in the producer loop and writes into a *different* slice
        // (of the iter_arg), so we cannot pair them until after unrolling.
        SmallVector<scf::ForOp> consumers;
        for (auto C : group) {
          if (C == P)
            continue;
          // Producer must precede consumer in block order (T is defined by
          // P's yield; SSA dominance already guarantees this, but check
          // explicitly to avoid pairing a consumer that appears before the
          // producer in a multi-chain group). P and C must share the same
          // parent block — isBeforeInBlock asserts this.
          if (P->getBlock() != C->getBlock() || !P->isBeforeInBlock(C))
            continue;
          bool readsT = false;
          C.getBody()->walk([&](vector::TransferReadOp read) {
            auto extract =
                read.getSource().getDefiningOp<tensor::ExtractSliceOp>();
            if (!extract || extract.getSource() != T)
              return;
            readsT = true;
          });
          if (readsT)
            consumers.push_back(C);
        }

        if (!consumers.empty()) {
          // Verify eliminability of the intermediate tensor T.
          // Mask safety (isMaskSafe) is verified per-pair inside
          // ForwardTransferReadThroughSliceChain during the forward step.
          if (!hasSimpleWritePath(P, T))
            continue;
          if (!hasSimpleReadPath(T, consumers))
            continue;
          if (!isPureIntermediate(T, group))
            continue;
          // Pairs will be collected after unrolling when write and read
          // slices collapse to the same value.
          // chain.loops = entire group (all unrolled together);
          // P/consumers identify the relay pair.
          chains.push_back({group, T, P, consumers});
        }
      }
    }
  }

  return chains;
}

//===----------------------------------------------------------------------===//
// Multi-hop forward
//===----------------------------------------------------------------------===//

/// Pattern that walks the extract_slice(insert_slice(...transfer_write...))
/// chain to forward a transfer_read's source to the original transfer_write
/// vector. Handles multi-level chains and same-source fallback.
struct ForwardTransferReadThroughSliceChain
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    if (readOp.hasOutOfBoundsDim() ||
        !isa<RankedTensorType>(readOp.getShapedType()))
      return failure();

    // Read indices must all be constant (guaranteed after unroll).
    SmallVector<int64_t> readIdx;
    for (Value idx : readOp.getIndices()) {
      auto c = getConstantIntValue(idx);
      if (!c)
        return failure();
      readIdx.push_back(*c);
    }

    Value curSource = readOp.getSource();
    SmallVector<int64_t> absOffset = readIdx;

    // Walk extract_slice → insert_slice chain to reach the transfer_write.
    while (true) {
      if (auto extractOp =
              curSource.getDefiningOp<tensor::ExtractSliceOp>()) {
        auto offsets = getConstOffsets(extractOp.getMixedOffsets());
        if (!offsets)
          break;
        // Reject sparse slices — forwarding would misread strided layout.
        if (!hasUnitStrides(extractOp.getMixedStrides()))
          return failure();
        if (offsets->size() != absOffset.size())
          return failure();
        for (size_t i = 0; i < absOffset.size(); ++i)
          absOffset[i] += (*offsets)[i];
        curSource = extractOp.getSource();
        continue;
      }
      if (auto insertOp =
              curSource.getDefiningOp<tensor::InsertSliceOp>()) {
        auto offsets = getConstOffsets(insertOp.getMixedOffsets());
        auto sizes = getConstOffsets(insertOp.getMixedSizes());
        if (!offsets || !sizes)
          break;
        if (!hasUnitStrides(insertOp.getMixedStrides()))
          return failure();
        if (offsets->size() != absOffset.size())
          return failure();
        bool covers = true;
        for (size_t i = 0; i < absOffset.size(); ++i) {
          int64_t lo = (*offsets)[i];
          int64_t hi = lo + (*sizes)[i];
          if (absOffset[i] < lo || absOffset[i] >= hi) {
            covers = false;
            break;
          }
        }
        if (covers) {
          for (size_t i = 0; i < absOffset.size(); ++i)
            absOffset[i] -= (*offsets)[i];
          curSource = insertOp.getSource();
          continue;
        }
        curSource = insertOp.getDest();
        continue;
      }
      break;
    }

    auto writeOp = curSource.getDefiningOp<vector::TransferWriteOp>();
    if (!writeOp) {
      Value readSource = readOp.getSource();
      for (Operation *user : readSource.getUsers()) {
        auto w = dyn_cast<vector::TransferWriteOp>(user);
        if (!w || w == readOp)
          continue;
        if (w.getSource() == readSource && w->getBlock() == readOp->getBlock() && w->isBeforeInBlock(readOp)) {
          writeOp = w;
          break;
        }
      }
      if (!writeOp)
        return failure();
      absOffset = readIdx;
    }

    if (writeOp.getIndices().size() != absOffset.size())
      return failure();
    for (auto [idx, want] : llvm::zip(writeOp.getIndices(), absOffset)) {
      auto c = getConstantIntValue(idx);
      if (!c || *c != want)
        return failure();
    }

    if (writeOp->getBlock() != readOp->getBlock() ||
        !writeOp->isBeforeInBlock(readOp))
      return failure();

    // Permutation maps must match — a different map (e.g. a transpose read)
    // would reorder lanes, so forwarding the write vector would misread.
    if (writeOp.getPermutationMap() != readOp.getPermutationMap())
      return failure();

    if (!isMaskSafe({writeOp, readOp}))
      return failure();

    if (readOp.getVectorType() != writeOp.getVectorType()) {
      if (readOp.getVectorType().getNumElements() !=
          writeOp.getVectorType().getNumElements())
        return failure();
      Location loc = readOp->getLoc();
      auto shapeCast = rewriter.create<vector::ShapeCastOp>(
          loc, readOp.getVectorType(), writeOp.getVector());
      rewriter.replaceOp(readOp, shapeCast);
    } else {
      rewriter.replaceOp(readOp, writeOp.getVector());
    }
    return success();
  }
};

struct UnrollAndForwardPass
    : public impl::UnrollAndForwardBase<UnrollAndForwardPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func->hasAttr(hivm::VectorFunctionAttr::name))
      return;

    auto chains = matchUnrollableChains(func, maxUnroll, maxBodyOps);
    LLVM_DEBUG(DBGS() << "chains=" << chains.size() << " for " << func.getName() << "\n");
    if (chains.empty())
      return;

    IRRewriter rewriter(func.getContext());
    // Track loops already unrolled by a prior chain in this pass — multiple
    // chains may share the same loop group, so skip re-unrolling them (the
    // loop is already gone, loopUnrollByFactor would fail anyway).
    DenseSet<Operation *> unrolledLoops;
    for (auto &chain : chains) {
      // Skip a chain if any of its loops was already unrolled by an earlier
      // chain in the same group.
      if (llvm::any_of(chain.loops, [&](scf::ForOp l) {
            return unrolledLoops.contains(l.getOperation());
          }))
        continue;
      // Determine trip count from the first loop (all share the same bounds).
      auto trip = *getConstantTripCount(chain.loops[0]);
      uint64_t factor = static_cast<uint64_t>(trip);

      // Unroll every loop in the chain.
      bool unrollFailed = false;
      for (auto loop : chain.loops) {
        if (failed(loopUnrollByFactor(loop, factor, nullptr))) {
          unrollFailed = true;
          break;
        }
        unrolledLoops.insert(loop.getOperation());
      }
      if (unrollFailed)
        continue;

      // After unrolling, the extract_slice/insert_slice chain of the
      // intermediate tensor is straight-line. Run a dedicated forward pass
      // that walks the multi-hop chain to replace transfer_read ops with
      // the source transfer_write vector. This is the same mechanism as
      // RemoveRedundantWriteAndReadPair but handles multi-level chains.
      {
        RewritePatternSet forwardPatterns(func.getContext());
        forwardPatterns.add<ForwardTransferReadThroughSliceChain>(func.getContext());
        (void)applyPatternsGreedily(func, std::move(forwardPatterns));
      }
      // Dead transfer_writes and insert_slices are cleaned up by the
      // downstream canonicalize / CSE / RemoveRedundantWriteAndReadPair.
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createUnrollAndForwardPass() {
  return std::make_unique<UnrollAndForwardPass>();
}
