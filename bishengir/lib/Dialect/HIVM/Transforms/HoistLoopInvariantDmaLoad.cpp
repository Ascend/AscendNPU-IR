//===- HoistLoopInvariantDmaLoad.cpp --------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025~2026. All rights reserved.
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
//
// Hoist loop-invariant DMA-load clusters out of scf.for loops.
//
// A masked tl.load materializes as a cluster: a memref.alloc for the dst
// buffer, a guarded zero-fill scf.if (a fill vector function or an inline
// linalg.fill), the DMA itself (hivm.hir.load or memref.copy) and a
// to_tensor view. Every op in that cluster carries memory effects, so the
// generic per-op LICM (isMemoryEffectFree) can never hoist it. This pass
// treats the cluster as one unit and moves it before the loop when:
//
//   * no cluster operand is defined inside the loop (in particular no
//     induction-variable or iter_arg dependence);
//   * the dst buffers have no user inside the loop outside the cluster;
//   * no op inside the loop may write the loaded GM source;
//   * the trip count rules the hoist form: a positive compile-time
//     constant hoists unguarded; a runtime-determined one hoists inside a
//     guard on the loop's entry test, with the loop itself moved into the
//     guard and the else region yielding the init args (exactly what a
//     zero-trip scf.for returns), so the cluster never runs unless the
//     loop does; a provably zero-trip loop, a zero or non-constant step,
//     and an unsigned comparison keep the cluster inside;
//   * no cluster result flows to the loop's yield into an iter_arg whose
//     init comes from a different buffer/cluster. Hoisting such a cluster
//     leaves the iter_arg chain non-aliasing with the yielded value, and
//     later passes then stitch the chain back together with per-iteration
//     buffer copies (and ad-hoc synchronization that the sync-solving
//     passes, which ran earlier, never derived). A loop whose only body
//     is the cluster has no DMA left to save either way, so the cluster
//     stays put.
//
// Functions carrying hivm.hir.anchor ops are left untouched entirely.
// Anchors partition the bodies of the mix-split kernels (cube / vector /
// backup) into the intervals that the delayed cross-core GSS later walks
// pairwise across the three copies. Hoisting moves a cluster across those
// anchors on one side only, so the load lands in a different interval on
// that side and the solver's sibling-only walk over a now cross-scope
// anchor pair breaks (DelayedCrossCoreGSS.cpp, buildDelayedFuncIr).
//
// The cluster performs identical work on every iteration, so removing the
// redundant copies preserves semantics. The compute tail follows its
// cluster out of the loop; a guarded cluster stops one nesting level up,
// because the next round scans only top-level body ops.
//
// Loops whose clusters were hoisted also get their invariant tensor compute
// hoisted: the cast/elemwise/broadcast chain in destination-passing form
// that the loaded tensors feed. Left in the loop it would be stamped back
// into the outlined function by the later VF outlining and recomputed on
// every iteration. The compute hoist is anchored to load-hoisted loops
// only, so the pass never moves compute on its own.
//
// Runs before VFFusionPass outlines loop bodies into separate functions;
// after that point the ops live in another function and are out of reach.
//
// Set BISHENGIR_DISABLE_HOIST_INVARIANT_DMA_LOAD=1 to turn the pass off.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_HOISTLOOPINVARIANTDMALOAD
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-hoist-loop-invariant-dma-load"

using namespace mlir;
using namespace mlir::hivm;

namespace {

static bool isDisabledByEnv() {
  return std::getenv("BISHENGIR_DISABLE_HOIST_INVARIANT_DMA_LOAD") != nullptr;
}

/// Follow view and bitcast ops up to the base buffer of `v`.
static Value getRootBuffer(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(def)) {
      v = view.getViewSource();
      continue;
    }
    // hivm.bitcast reinterprets a memref without changing the buffer.
    if (auto bitcast = dyn_cast<BitcastOp>(def)) {
      v = bitcast.getSrc();
      continue;
    }
    break;
  }
  return v;
}

static bool isMemref(Value v) { return isa<BaseMemRefType>(v.getType()); }

/// Whether `op` may write memory that a hoisted GM read must not cross.
static bool mayWriteMemref(Operation *op) {
  return isa<func::CallOp, LoopLikeOpInterface, StoreOp>(op) ||
         hasEffect<MemoryEffects::Write>(op);
}

/// A DMA-like op: hivm.hir.load or memref.copy. Both only write their own
/// dst buffer.
static bool isDmaOp(Operation *op) { return isa<LoadOp, memref::CopyOp>(op); }

static Value getDmaSrc(Operation *op) {
  if (auto load = dyn_cast<LoadOp>(op))
    return load.getSrc();
  return cast<memref::CopyOp>(op).getSource();
}

static Value getDmaDst(Operation *op) {
  if (auto load = dyn_cast<LoadOp>(op))
    return load.getDst();
  return cast<memref::CopyOp>(op).getTarget();
}

/// The DMA and every in-loop op that must move with it: the dst view chain
/// and its alloc, the guarded fill writing the dst, and to_tensor views of
/// the dst.
struct LoadCluster {
  scf::ForOp loop;
  Operation *seed;

  SetVector<Operation *> ops;
  SmallVector<Value> dstBuffers; // results of the cluster's allocs
  Value srcRoot;

  LoadCluster(scf::ForOp loop, Operation *seed) : loop(loop), seed(seed) {}

  bool definedInsideLoop(Value v) {
    return loop.getBodyRegion().isAncestor(v.getParentRegion());
  }

  /// Collect the in-loop def chain of the DMA's operands (src, dst, and for
  /// loads also pad_value and padding nums). Fails on any loop-carried
  /// dependence or on an op the pass cannot reason about.
  bool collectChain() {
    srcRoot = getRootBuffer(getDmaSrc(seed));
    ops.insert(seed);
    SmallVector<Value> work(seed->getOperands());
    while (!work.empty()) {
      Value v = work.pop_back_val();
      if (!definedInsideLoop(v))
        continue;
      Operation *def = v.getDefiningOp();
      // A block argument owned by the loop or a nested region is a
      // loop-carried dependence.
      if (!def || def->getBlock() != loop.getBody())
        return false;
      if (ops.contains(def))
        continue;
      // The chain may only contain buffer allocs, view ops and pure
      // region-free ops (index math); a second memory op in the chain is
      // not supported.
      if (!isa<memref::AllocOp, memref::AllocaOp, ViewLikeOpInterface>(def) &&
          !(isMemoryEffectFree(def) && def->getNumRegions() == 0))
        return false;
      ops.insert(def);
      llvm::append_range(work, def->getOperands());
    }
    for (Operation *op : ops)
      if (isa<memref::AllocOp, memref::AllocaOp>(op))
        dstBuffers.push_back(op->getResult(0));
    // The dst buffer must be created inside the loop; a dst allocated outside
    // is shared with the enclosing scope and is left alone.
    return !dstBuffers.empty();
  }

  /// Ops allowed inside the fill guard: plain pure region-free ops (scalar
  /// address math), or effect-carrying ops whose every declared effect sits
  /// on one of their own operands -- an inline linalg.fill writing its dst
  /// memref. The operand walk below pins those operands to the dst buffers.
  static bool mayFillDstBuffer(Operation *op) {
    if (isMemoryEffectFree(op) && op->getNumRegions() == 0)
      return true;
    auto mem = dyn_cast<MemoryEffectOpInterface>(op);
    if (!mem)
      return false;
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
    mem.getEffects(effects);
    return llvm::all_of(
        effects,
        [](const SideEffects::EffectInstance<MemoryEffects::Effect> &effect) {
          return effect.getValue() != nullptr;
        });
  }

  /// Add the in-loop pure defs of `values` (the fill guard's condition is
  /// computed inside the loop from cluster values, e.g. a mask cmpi) to the
  /// cluster. Returns false on a loop-carried dependence.
  bool collectConditionDefs(ValueRange values) {
    SmallVector<Value> work(values.begin(), values.end());
    while (!work.empty()) {
      Value v = work.pop_back_val();
      if (!definedInsideLoop(v))
        continue;
      Operation *def = v.getDefiningOp();
      if (!def || def->getBlock() != loop.getBody())
        return false;
      if (ops.contains(def))
        continue;
      if (!(isMemoryEffectFree(def) && def->getNumRegions() == 0))
        return false;
      ops.insert(def);
      llvm::append_range(work, def->getOperands());
    }
    return true;
  }

  /// If `ifOp` is a guarded fill of a dst buffer (no results, region only
  /// calls and fill ops that touch nothing but the dst buffers), add it to
  /// the cluster. An in-loop condition is absorbed with its pure defs.
  bool collectFill(scf::IfOp ifOp) {
    if (ifOp->getNumResults() != 0)
      return false;
    bool usesDst = false;
    for (Region &region : ifOp->getRegions())
      for (Operation &op : region.getOps()) {
        if (!isa<func::CallOp>(op) && !mayFillDstBuffer(&op))
          return false;
        // Every value the op touches, including nested-region captures:
        // memrefs must be dst buffers, scalars must be invariant.
        bool bad = false;
        op.walk([&](Operation *inner) {
          for (Value operand : inner->getOperands()) {
            if (op.isAncestor(operand.getParentRegion()->getParentOp()))
              continue; // block arg of one of the op's own regions
            if (isMemref(operand)) {
              if (!llvm::is_contained(dstBuffers, getRootBuffer(operand))) {
                bad = true;
                return WalkResult::interrupt();
              }
              usesDst = true;
            } else if (definedInsideLoop(operand)) {
              // Ops inside the if itself move with it.
              Operation *def = operand.getDefiningOp();
              if (!def || (!ops.contains(def) && !ifOp->isAncestor(def))) {
                bad = true;
                return WalkResult::interrupt();
              }
            }
          }
          return WalkResult::advance();
        });
        if (bad)
          return false;
      }
    if (!usesDst)
      return false;
    if (!collectConditionDefs(ifOp->getOperands()))
      return false;
    ops.insert(ifOp);
    return true;
  }

  /// Collect the in-loop guarded fills and to_tensor views of the dst
  /// buffers. Everything else that uses a dst buffer makes the hoist invalid
  /// (checked in verify()).
  void collectFillAndViews() {
    for (Operation &op : loop.getBody()->getOperations()) {
      if (ops.contains(&op))
        continue;
      if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(&op)) {
        if (llvm::is_contained(dstBuffers, getRootBuffer(toTensor.getMemref())))
          ops.insert(&op);
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(&op))
        (void)collectFill(ifOp);
    }
  }

  /// No op outside the cluster may use a dst buffer or write the GM source.
  bool verify() {
    for (Operation &op : loop.getBody()->getOperations()) {
      if (ops.contains(&op))
        continue;
      bool invalid = false;
      op.walk([&](Operation *inner) {
        bool writes = mayWriteMemref(inner);
        if (isDmaOp(inner)) {
          // A DMA only writes its own dst buffer; its src operand is a read.
          writes = false;
          if (getRootBuffer(getDmaDst(inner)) == srcRoot) {
            invalid = true;
            return WalkResult::interrupt();
          }
        }
        for (Value operand : inner->getOperands()) {
          if (!isMemref(operand))
            continue;
          Value root = getRootBuffer(operand);
          if (llvm::is_contained(dstBuffers, root) ||
              (writes && root == srcRoot)) {
            invalid = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (invalid)
        return false;
    }
    return true;
  }

  /// The root buffer a yielded tensor or memref value ultimately views: a
  /// to_tensor resolves to its memref, a memref to its root. Returns an
  /// empty value for anything that cannot be traced to a buffer.
  static Value traceToRootBuffer(Value v) {
    if (auto toTensor = v.getDefiningOp<bufferization::ToTensorOp>())
      v = toTensor.getMemref();
    return isMemref(v) ? getRootBuffer(v) : Value();
  }

  /// Bail-out: no cluster result may flow to the loop's yield into an
  /// iter_arg whose init comes from a different buffer/cluster. The
  /// cluster's dst allocs live inside the loop while iter_arg inits
  /// dominate it, so a cluster result reaching the yield means by
  /// construction that the chain starts from a foreign buffer. Hoisting
  /// would leave a degenerate loop whose iter_arg chain no longer aliases
  /// the yielded value, and later passes then materialize per-iteration
  /// buffer copies (with ad-hoc synchronization) to stitch the chain back
  /// together. Keep the cluster inside the loop instead.
  bool yieldsIntoNonAliasedIterArg() {
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield)
      return false;
    for (auto [index, value] : llvm::enumerate(yield.getOperands())) {
      Operation *def = value.getDefiningOp();
      if (!def || !ops.contains(def))
        continue;
      Value initRoot = traceToRootBuffer(loop.getInitArgs()[index]);
      if (!initRoot || !llvm::is_contained(dstBuffers, initRoot))
        return true;
    }
    return false;
  }

  void moveBeforeLoop() {
    for (Operation &op :
         llvm::make_early_inc_range(loop.getBody()->getOperations()))
      if (ops.contains(&op))
        op.moveBefore(loop.getOperation());
  }
};

/// How a loop's clusters may leave it, decided by the trip count.
enum class HoistKind {
  /// Positive compile-time constant trip count: hoist unguarded.
  Unguarded,
  /// Trip count known only at runtime: hoist inside a guard on the loop's
  /// entry test.
  Guarded,
  /// Provably zero trips, zero or non-constant step, or unsigned bounds
  /// comparison: keep the cluster inside.
  Skip
};

static HoistKind classifyHoist(scf::ForOp loop) {
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!step || *step == 0 || loop->hasAttr("unsignedCmp"))
    return HoistKind::Skip;
  std::optional<int64_t> trips = constantTripCount(
      loop.getLowerBound(), loop.getUpperBound(), loop.getStep());
  if (!trips)
    return HoistKind::Guarded;
  return *trips > 0 ? HoistKind::Unguarded : HoistKind::Skip;
}

/// Wrap `loop` in a guard on its own entry test: the loop moves, unchanged,
/// into the then region, the else region yields the init args (a zero-trip
/// scf.for's results), and downstream users of the loop's results read the
/// guard's results.
static scf::IfOp createEntryGuard(scf::ForOp loop) {
  OpBuilder builder(loop.getOperation());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  arith::CmpIPredicate pred =
      *step > 0 ? arith::CmpIPredicate::slt : arith::CmpIPredicate::sgt;
  Value cond = builder.create<arith::CmpIOp>(
      loop.getLoc(), pred, loop.getLowerBound(), loop.getUpperBound());
  auto guard = builder.create<scf::IfOp>(
      loop.getLoc(), TypeRange(loop->getResultTypes()), cond,
      /*addThenBlock=*/false, /*addElseBlock=*/false);
  Block *thenBlock = builder.createBlock(&guard.getThenRegion());
  loop->moveBefore(thenBlock, thenBlock->end());
  builder.create<scf::YieldOp>(loop.getLoc(), loop->getResults());
  builder.createBlock(&guard.getElseRegion());
  builder.create<scf::YieldOp>(loop.getLoc(), loop.getInitArgs());
  for (auto [result, guarded] :
       llvm::zip(loop->getResults(), guard.getResults()))
    result.replaceUsesWithIf(guarded, [&](OpOperand &use) {
      return !guard->isAncestor(use.getOwner());
    });
  return guard;
}

/// Try to hoist every DMA (hir.load or memref.copy) directly nested in
/// `loop`.
static bool hoistLoadsInLoop(scf::ForOp loop) {
  HoistKind kind = classifyHoist(loop);
  if (kind == HoistKind::Skip) {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: trip count is provably "
                               "non-positive, or the step is not a constant; "
                               "keeping all clusters\n");
    return false;
  }
  // Created on the first guarded hoist; later clusters of this loop land
  // inside it too.
  scf::IfOp entryGuard;
  bool changed = false;
  // Collect first: hoisting mutates the body.
  SmallVector<Operation *> dmas;
  for (Operation &op : loop.getBody()->getOperations())
    if (isDmaOp(&op))
      dmas.push_back(&op);
  for (Operation *dma : dmas) {
    LoadCluster cluster(loop, dma);
    if (!cluster.collectChain())
      continue;
    cluster.collectFillAndViews();
    if (!cluster.verify())
      continue;
    if (cluster.yieldsIntoNonAliasedIterArg()) {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                              << "]: cluster result yields into a "
                                 "non-aliased iter_arg; keeping it\n");
      continue;
    }
    if (kind == HoistKind::Guarded) {
      if (!entryGuard)
        entryGuard = createEntryGuard(loop);
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: hoisting load cluster "
                                 "inside the loop's entry-test guard\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: hoisting load cluster\n");
    }
    // Before the loop, wherever it lives by now.
    cluster.moveBeforeLoop();
    changed = true;
  }
  return changed;
}

/// Hoist the effect-free tensor compute the moved loads feed: plain pure
/// region-free ops and destination-passing structured ops on tensors (a
/// tensor-form structured op reports no memory effects -- its getEffects
/// only sees memref operands -- and cannot fault on the invariant values it
/// reads). Only loops whose load clusters were hoisted get this treatment,
/// so the pass never moves compute on its own.
static void hoistInvariantCompute(scf::ForOp loop) {
  moveLoopInvariantCode(
      loop.getLoopRegions(),
      [&](Value v, Region *) { return loop.isDefinedOutsideOfLoop(v); },
      [&](Operation *op, Region *) {
        auto structured = dyn_cast<linalg::LinalgOp>(op);
        return isMemoryEffectFree(op) &&
               (op->getNumRegions() == 0 ||
                (structured && structured.hasPureTensorSemantics()));
      },
      [&](Operation *op, Region *) { loop.moveOutOfLoop(op); });
}

struct HoistLoopInvariantDmaLoadPass
    : public impl::HoistLoopInvariantDmaLoadBase<
          HoistLoopInvariantDmaLoadPass> {
  void runOnOperation() override {
    if (isDisabledByEnv())
      return;
    func::FuncOp func = getOperation();
    if (hacc::utils::isHost(func))
      return;
    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module || !hacc::utils::supportsRegBaseVectorLowering(module))
      return;
    // Mix-split kernels carry hivm.hir.anchor ops partitioning their bodies
    // into the intervals the delayed cross-core GSS matches pairwise across
    // the cube / vector / backup copies. Hoisting moves a cluster across
    // those anchors on one side only, which desynchronizes the interval
    // structure and breaks the solver's sibling-only anchor walk. The pass
    // does not participate in that contract, so leave such kernels alone.
    if (func.walk([&](AnchorOp) { return WalkResult::interrupt(); })
            .wasInterrupted())
      return;
    // Repeat so a load invariant to several nested loops hoists one level
    // per round, taking its compute with it.
    bool changed = true;
    for (unsigned round = 0; changed && round < 8; ++round) {
      changed = false;
      SmallVector<scf::ForOp> loops;
      func.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
      for (scf::ForOp loop : loops)
        if (hoistLoadsInLoop(loop)) {
          hoistInvariantCompute(loop);
          changed = true;
        }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createHoistLoopInvariantDmaLoadPass() {
  return std::make_unique<HoistLoopInvariantDmaLoadPass>();
}
