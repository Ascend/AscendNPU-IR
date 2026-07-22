//===- HoistAndFuseDotChains.cpp - Trans hoist + dot-loop fusion ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One pass, two complementary transformations on K-tile chain loops:
//
//   Pattern 1 — hoistLoopInvariantTrans
//     LICM specialized for tt.trans.  In K-tile loops created by TileDot,
//     a tt.trans of a loop-invariant value sometimes lands inside the chain
//     body.  Walks every scf.for and moves any tt.trans whose source is
//     defined OUTSIDE the loop body to a position immediately before the
//     loop.  Generic triton-licm handles this in many cases, but a
//     targeted step is cheap, explicit, and keeps trans placement
//     intentional even if upstream pass ordering changes.
//
//   Pattern 2 — fuseAdjacentDotChains
//     Fuses consecutive scf.for ops that share identical lower-bound,
//     upper-bound, and step.  Required additional condition: the second
//     loop's init args must NOT depend on the first loop's results (i.e.,
//     the chains are independent).  The fused loop carries both chains'
//     iter args, runs both bodies' work side-by-side per iteration, and
//     yields the combined result.  This puts independent dot chains in
//     the same loop so the scheduler can interleave their issue at the
//     instruction level — equivalent to "unroll all chains, interleave
//     the chained dot pairs, reroll into one loop" expressed as a single
//     fusion step.
//
// Pattern 1 runs first so the trans placements settle before fusion
// decides what's inside each loop body.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hoist-and-fuse-dot-chains"

namespace bishengir {
namespace triton {
#define GEN_PASS_DEF_HOISTANDFUSEDOTCHAINS
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {
using namespace mlir;

//===----------------------------------------------------------------------===//
// Pattern 1 — hoistLoopInvariantTrans
//===----------------------------------------------------------------------===//

static void hoistLoopInvariantTrans(mlir::triton::FuncOp fn) {
  SmallVector<mlir::triton::TransOp> work;
  fn.walk([&](mlir::triton::TransOp tr) { work.push_back(tr); });

  for (auto tr : work) {
    auto forOp = tr->getParentOfType<scf::ForOp>();
    if (!forOp)
      continue;

    // Bail if tr's source is loop-variant.
    Value src = tr.getSrc();
    bool srcInLoop = false;
    if (Operation *def = src.getDefiningOp()) {
      srcInLoop = forOp->isProperAncestor(def);
    } else if (auto barg = dyn_cast<BlockArgument>(src)) {
      if (barg.getOwner() == forOp.getBody())
        srcInLoop = true;
    }
    if (srcInLoop)
      continue;

    tr->moveBefore(forOp);
  }
}

//===----------------------------------------------------------------------===//
// Pattern 2 — fuseAdjacentDotChains
//===----------------------------------------------------------------------===//

// True iff `nextFor`'s init args don't reach `curFor`'s results via a
// bounded back-walk.
static bool areIndependent(scf::ForOp curFor, scf::ForOp nextFor) {
  DenseSet<Operation *> visited;
  SmallVector<Operation *> wl;
  for (Value v : nextFor.getInitArgs()) {
    if (Operation *def = v.getDefiningOp()) {
      if (def == curFor.getOperation())
        return false;
      if (visited.insert(def).second)
        wl.push_back(def);
    }
  }
  // Bounded BFS; on overflow, conservatively report NOT independent.
  unsigned budget = 64;
  while (!wl.empty() && budget-- > 0) {
    Operation *op = wl.pop_back_val();
    if (op == curFor.getOperation())
      return false;
    for (Value opd : op->getOperands()) {
      if (Operation *def = opd.getDefiningOp()) {
        if (def == curFor.getOperation())
          return false;
        if (visited.insert(def).second)
          wl.push_back(def);
      }
    }
  }
  return true;
}

// SSA-identity comparison of bounds + step. Sufficient for post-TileDot
// K-tile loops which are emitted with the same constants.
static bool sameBounds(scf::ForOp a, scf::ForOp b) {
  return a.getLowerBound() == b.getLowerBound() &&
         a.getUpperBound() == b.getUpperBound() &&
         a.getStep() == b.getStep();
}

// Sum envelope bytes of every `bishengir.scratch_shm` func arg referenced
// inside `forOp`. Returns 0 when no SMEM-staged args are touched.
static uint64_t computeForLoopSmemBytes(scf::ForOp forOp) {
  auto func = forOp->getParentOfType<mlir::triton::FuncOp>();
  if (!func)
    return 0;

  // Step 1: collect scratch_shm arg indices reached from the loop body.
  llvm::DenseSet<unsigned> touchedArgs;
  forOp.getBody()->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Value v = operand;
      while (true) {
        if (auto addptr = v.getDefiningOp<mlir::triton::AddPtrOp>()) {
          v = addptr.getPtr();
          continue;
        }
        if (auto splat = v.getDefiningOp<mlir::triton::SplatOp>()) {
          v = splat.getSrc();
          continue;
        }
        break;
      }
      if (auto ba = dyn_cast<BlockArgument>(v)) {
        if (ba.getOwner() == &func.front()) {
          unsigned idx = ba.getArgNumber();
          if (func.getArgAttr(idx, "bishengir.scratch_shm"))
            touchedArgs.insert(idx);
        }
      }
    }
  });

  if (touchedArgs.empty())
    return 0;

  // Step 2: for each touched arg sum max bytes over its envelope tt.store(s).
  uint64_t totalBytes = 0;
  for (unsigned idx : touchedArgs) {
    Value arg = func.getArgument(idx);
    uint64_t maxBytes = 0;
    SmallVector<Value> wl{arg};
    llvm::DenseSet<Value> seen;
    while (!wl.empty()) {
      Value v = wl.pop_back_val();
      for (Operation *user : v.getUsers()) {
        if (auto store = dyn_cast<mlir::triton::StoreOp>(user)) {
          if (auto vt = dyn_cast<RankedTensorType>(
                  store.getValue().getType())) {
            uint64_t bytes = 1;
            for (int64_t d : vt.getShape())
              bytes *= static_cast<uint64_t>(std::max<int64_t>(1, d));
            uint64_t elemBits =
                static_cast<uint64_t>(vt.getElementType().getIntOrFloatBitWidth());
            bytes *= (elemBits + 7) / 8;
            maxBytes = std::max(maxBytes, bytes);
          }
        }
        if (isa<mlir::triton::SplatOp, mlir::triton::AddPtrOp>(user)) {
          for (Value r : user->getResults())
            if (seen.insert(r).second)
              wl.push_back(r);
        }
      }
    }
    totalBytes += maxBytes;
  }
  return totalBytes;
}

// True iff `nextFor` does not reference any SSA value defined strictly
// between `curFor` and `nextFor` (post-fusion those defs no longer dominate
// the merged for at `curFor`'s position). Lets us fuse non-adjacent loops.
//
// ALSO refuses fusion when any op with memory-write effects sits between
// the two loops.  After fusion, the new loop sits at curFor's position
// and the in-between op (e.g., the envelope `tt.store` to scratch_shm
// that `StageNonLoadOperandPattern` emits before its inner K-tile loop)
// would move to AFTER the fused loop.  That breaks correctness silently
// because the SSA-independence check above does not see the memory
// dependency — the loop's body reads from memory the in-between op
// writes, but neither produces an SSA edge between them.  Conservative:
// any side-effecting op between blocks the fuse.
static bool nextDoesNotUseBetweenDefs(scf::ForOp curFor, scf::ForOp nextFor) {
  DenseSet<Value> betweenDefs;
  bool anyWriteBetween = false;
  Block::iterator it(curFor);
  ++it;
  Block::iterator end(nextFor);
  for (; it != end; ++it) {
    Operation *op = &*it;
    for (Value r : op->getResults())
      betweenDefs.insert(r);
    // Anything explicitly write-effecting, or any op without a known
    // pure dialect, blocks the fuse.  This catches `tt.store` to
    // scratch_shm `ptr<6>` args (the case `StageNonLoadOperandPattern`
    // emits between its envelope STORE and per-tile LOAD loops) before
    // CSPtoMD turns them into `ttg.local_store`.
    if (auto memOp = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      memOp.getEffects(effects);
      for (auto &e : effects)
        if (isa<MemoryEffects::Write>(e.getEffect())) {
          anyWriteBetween = true;
          break;
        }
    } else if (isa<mlir::triton::StoreOp>(op)) {
      anyWriteBetween = true;
    }
  }
  if (anyWriteBetween)
    return false;
  if (betweenDefs.empty())
    return true;
  for (Value v : nextFor.getInitArgs())
    if (betweenDefs.count(v))
      return false;
  bool clean = true;
  nextFor.getBody()->walk([&](Operation *op) {
    for (Value opd : op->getOperands())
      if (betweenDefs.count(opd))
        clean = false;
  });
  return clean;
}

// Fuse two adjacent scf.for ops.  Returns the new fused scf.for, or
// nullptr on failure.
static scf::ForOp fuseTwo(scf::ForOp curFor, scf::ForOp nextFor) {
  OpBuilder builder(curFor);
  Location loc = curFor.getLoc();
  unsigned nA = curFor.getInitArgs().size();
  unsigned nB = nextFor.getInitArgs().size();

  SmallVector<Value> newInits;
  for (Value v : curFor.getInitArgs())
    newInits.push_back(v);
  for (Value v : nextFor.getInitArgs())
    newInits.push_back(v);

  auto newFor =
      builder.create<scf::ForOp>(loc, curFor.getLowerBound(),
                                  curFor.getUpperBound(), curFor.getStep(),
                                  newInits);
  // Drop the auto-generated yield; we'll emit our own merged yield.
  if (!newFor.getBody()->empty())
    newFor.getBody()->getTerminator()->erase();

  IRMapping mA, mB;
  mA.map(curFor.getInductionVar(), newFor.getInductionVar());
  mB.map(nextFor.getInductionVar(), newFor.getInductionVar());
  for (unsigned i = 0; i < nA; ++i)
    mA.map(curFor.getRegionIterArgs()[i], newFor.getRegionIterArgs()[i]);
  for (unsigned i = 0; i < nB; ++i)
    mB.map(nextFor.getRegionIterArgs()[i], newFor.getRegionIterArgs()[nA + i]);

  builder.setInsertionPointToStart(newFor.getBody());
  for (Operation &op : curFor.getBody()->without_terminator())
    builder.clone(op, mA);
  for (Operation &op : nextFor.getBody()->without_terminator())
    builder.clone(op, mB);

  auto curYield = cast<scf::YieldOp>(curFor.getBody()->getTerminator());
  auto nextYield = cast<scf::YieldOp>(nextFor.getBody()->getTerminator());
  SmallVector<Value> newYieldOps;
  for (Value v : curYield.getOperands())
    newYieldOps.push_back(mA.lookupOrDefault(v));
  for (Value v : nextYield.getOperands())
    newYieldOps.push_back(mB.lookupOrDefault(v));
  builder.create<scf::YieldOp>(loc, newYieldOps);

  // Replace each old for's results with the matching new-for slot.
  for (unsigned i = 0; i < curFor.getNumResults(); ++i)
    curFor.getResult(i).replaceAllUsesWith(newFor.getResult(i));
  for (unsigned i = 0; i < nextFor.getNumResults(); ++i)
    nextFor.getResult(i).replaceAllUsesWith(newFor.getResult(nA + i));

  nextFor.erase();
  curFor.erase();
  return newFor;
}

static void fuseAdjacentDotChains(mlir::triton::FuncOp fn,
                                   int64_t smemBudgetBytes) {
  // SMEM budget gate: peak SMEM rises from max(A,B) to A+B once two
  // loops are fused, so cap combined footprint at 70% of the budget.
  uint64_t threshold = 0;
  if (smemBudgetBytes > 0)
    threshold = static_cast<uint64_t>(smemBudgetBytes) * 7 / 10;

  // Fixed point: each iteration merges one pair, exposing the next.
  bool changed = true;
  unsigned iters = 0;
  while (changed && iters++ < 32) {
    changed = false;

    SmallVector<std::pair<scf::ForOp, scf::ForOp>> candidates;
    fn.walk([&](Block *block) {
      // For each scf.for, take the earliest later for that's fuse-eligible.
      for (auto outerIt = block->begin(); outerIt != block->end(); ++outerIt) {
        auto cur = dyn_cast<scf::ForOp>(&*outerIt);
        if (!cur) continue;
        for (auto innerIt = std::next(outerIt); innerIt != block->end();
             ++innerIt) {
          auto nxt = dyn_cast<scf::ForOp>(&*innerIt);
          if (!nxt) continue;
          if (!sameBounds(cur, nxt)) continue;
          if (!areIndependent(cur, nxt)) continue;
          if (!nextDoesNotUseBetweenDefs(cur, nxt)) continue;
          // SMEM budget gate: skip the fuse when the combined footprint
          // exceeds 70% of the budget.  Computing on candidates only —
          // the cost is bounded by the number of candidates per func.
          if (threshold > 0) {
            uint64_t curBytes = computeForLoopSmemBytes(cur);
            uint64_t nextBytes = computeForLoopSmemBytes(nxt);
            uint64_t combined = curBytes + nextBytes;
            if (combined > threshold) {
              LLVM_DEBUG(llvm::dbgs()
                  << "[HoistAndFuse] skipping fuse: combined SMEM "
                  << combined << " B > 70% budget ("
                  << threshold << " B of " << smemBudgetBytes << " B)\n");
              continue;
            }
          }
          candidates.emplace_back(cur, nxt);
          break; // one match per cur
        }
        if (!candidates.empty()) break; // one fuse per outer walk
      }
    });

    for (auto [cur, nxt] : candidates) {
      if (!fuseTwo(cur, nxt)) continue;
      changed = true;
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct HoistAndFuseDotChainsPass
    : public impl::HoistAndFuseDotChainsBase<HoistAndFuseDotChainsPass> {
  using impl::HoistAndFuseDotChainsBase<
      HoistAndFuseDotChainsPass>::HoistAndFuseDotChainsBase;
  void runOnOperation() override {
    auto fn = getOperation();
    // Hoist trans first so loop bodies settle before fusion examines them.
    hoistLoopInvariantTrans(fn);
    fuseAdjacentDotChains(fn, this->smemBudgetBytes);
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
createHoistAndFuseDotChainsPass(const HoistAndFuseDotChainsOptions &options) {
  return std::make_unique<HoistAndFuseDotChainsPass>(options);
}

} // namespace triton
} // namespace bishengir
