//===- CVPipelining.cpp --- Pipelining pass for mix-cv ops ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/WorkItem.h"
#include "bishengir/Dialect/HIVM/Utils/WorklistBuilder.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "cv-pipelining"

using llvm::dbgs;
namespace mlir {
using namespace hivm;

#define GEN_PASS_DEF_CVPIPELINING
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

using hivm::detail::queryCoreTypeHelper;

namespace {

struct AtomicEffect {
  AtomicKind kind;
  TypeAttr type;
};

struct WorkspaceAllocParams {
  unsigned multibuffer;
  annotation::MarkOp marker;
  bufferization::ToTensorOp toTensor;
};

struct CVPipelineImpl {
  CVPipelineImpl(LoopLikeOpInterface loop, int multibuffer, bool skewMode,
                 bool enableLazyLoading)
      : pipelineLoop(loop), newLoop(nullptr), builder(loop->getContext()),
        numMultibuffer(multibuffer), enableSkewMode(skewMode),
        wlBuilder(cast<scf::ForOp>(loop.getOperation()), multibuffer,
                  enableLazyLoading),
        yieldedVals(loop.getYieldedValues().begin(),
                    loop.getYieldedValues().end()) {}

  LogicalResult run();

private:
  void collectAtomicEffects();

  /// Absorb non-core "merger" ops (e.g. `arith.select`, `arith.cmpi`) that
  /// sit between a work-item op's output and a `scf.yield` operand into the
  /// producing work item. Without this, those ops are never cloned into any
  /// work-item forOp nor into newLoop, so the trailing terminator clone in
  /// migrateOps copies the operand reference verbatim and ends up pointing
  /// at an op inside the soon-to-be-erased pipelineLoop.
  ///
  /// Returns failure when a merger chain cannot be cleanly attributed to a
  /// single work item (chain spans multiple work items, or has no work-item
  /// producer at all). In that case `run()` reverts and the pass becomes a
  /// no-op for this loop.
  LogicalResult absorbMergerOpsIntoWorkItems();

  LogicalResult markOutputs();

  /// Reject loops whose work-item partition has a cross-core data dependency
  /// that migrateOps cannot honor. migrateOps clones each work-item's ops into
  /// its own per-core loop, remapping operands through that work item's
  /// IRMapping. A value defined elsewhere in the loop body is only resolvable
  /// in a work item if either (a) its top-level producer is assigned to the
  /// same work item (so it is cloned alongside) or (b) it is tracked as a
  /// cross-work-item output (local/yielded) and therefore forwarded through the
  /// new loops' iter_args/results. A value that is neither -- e.g. a small
  /// reduction result feeding a scalar scale chain that is replicated onto both
  /// cores while the reduction itself stays on one core -- would be cloned with
  /// a dangling reference to the original op; erasing the pipeline loop then
  /// aborts with "operation destroyed but still has uses". Detect that here,
  /// before any IR mutation, so `run()` can bail and leave the loop
  /// un-pipelined instead of crashing.
  LogicalResult checkWorkItemDependencies();

  LogicalResult expandOutputInits(WorkItem &item);
  LogicalResult expandOutputInitsForPreload(WorkItem &item);

  LogicalResult createNewLoops();

  LogicalResult migrateOps();

  LogicalResult createNewLoopsForPreloadWithScopes();
  LogicalResult markScopesForPreload();

  // Undo partial IR changes by erasing newLoop if it was created.
  void revert();

  Value createSubview(OpBuilder &builder, Location loc, Value from, Type to,
                      Value iv);
  Value createToTensor(OpBuilder &builder, Location loc, Value src);
  // Returns failure() on malformed input. On success, the returned Value may
  // still be nullptr when there is no masking subview to update.
  FailureOr<Value> updateMaskingSubview(OpBuilder &builder, Location loc,
                                        Value expanded, OpOperand &initOperand,
                                        Value iv) const;

  // ===========================================================================
  // Data members
  // ===========================================================================

  // Loop being pipelined
  scf::ForOp pipelineLoop;

  // Unrolled pipelineLoop that will replace it once we're done
  scf::ForOp newLoop;

  OpBuilder builder;

  // Number of multibuffer/pipeline stages/unroll iterations
  int numMultibuffer;

  // Use skew-mode pipelining instead of the default unroll-mode
  bool enableSkewMode;

  // Worklist builder — owns dep-tracking machinery, separator/dependence
  // discovery, lazy-load hint surface, and outputMemrefMap. Held as a member
  // so post-build queries (e.g. wlBuilder.shouldLazyLoadFor in markOutputs)
  // remain valid for the duration of run().
  hivm::WorklistBuilder wlBuilder;

  // Mapping from the converted memref to the op that writes to it (i.e.
  // FixPipeOp). Copy populated from wlBuilder.build() so post-build
  // CV-specific code (markOutputs / expandOutputInits / migrateOps) can read
  // it via a stable local handle.
  DenseMap<bufferization::ToTensorOp, DestinationStyleOpInterface>
      outputMemrefMap;

  // Non-DPS ops could potentially be cloned to various different work items.
  // Copy populated from wlBuilder.build().
  DenseMap<Operation *, SmallVector<WorkItem *>> opToWorkItemMap;

  // Lookup for yielded values
  SetVector<Value> yieldedVals;

  // WorkItems are populated by wlBuilder.build() and copied here; we still
  // keep our own vector since markOutputs/expandOutputInits/migrateOps
  // mutate item->localOutputs / item->yieldedOutputs / item->forOp etc. on
  // the per-WorkItem state owned via shared_ptr.
  SmallVector<std::shared_ptr<WorkItem>> worklist;

  // Corresponding expanded tensors for each output of work items
  DenseMap<Value, Value> expandedTensorMap;

  // Mapping from each op under atomic effect to its atomic kind and data type
  DenseMap<Operation *, AtomicEffect> atomicEffectMap;

  // If the atomic effect is still active at the end of the loop body, this
  // holds that trailing state so it can be restored after the pipelined loops.
  std::optional<AtomicEffect> trailingAtomicEffect;

  // Mapping from the original pipelineLoop to the newLoop to guide the cloning
  // process
  IRMapping globalIRMap;

  DenseMap<Value, Value> localOuputsToRedurnRes;

  DenseSet<Operation *> toErase;
};

struct CVPipeliningPass
    : public ::mlir::impl::CVPipeliningBase<CVPipeliningPass> {
  using Base::Base;
  void runOnOperation() final;
};
} // namespace

static Value traceValueDef(Value v) {
  if (!v)
    return nullptr;
  if (auto result = dyn_cast<OpResult>(v)) {
    Operation *defining = result.getOwner();
    Value srcVal =
        TypeSwitch<Operation *, Value>(defining)
            .Case<CastOpInterface, ViewLikeOpInterface, tensor::CollapseShapeOp,
                  tensor::ExpandShapeOp, tensor::ExtractSliceOp,
                  tensor::ReshapeOp, bufferization::ToTensorOp,
                  bufferization::ToMemrefOp>(
                [](auto op) { return op->getOperand(0); })
            .Case([](tensor::InsertSliceOp insert) { return insert.getDest(); })
            .Case([result](LoopLikeOpInterface loop) {
              return loop.getTiedLoopInit(result)->get();
            })
            .Default([](Operation *op) { return nullptr; });
    if (srcVal)
      return traceValueDef(srcVal);
    return result;
  }

  // In case of Block Argument
  auto blkArg = dyn_cast<BlockArgument>(v);
  if (!blkArg) {
    LLVM_DEBUG(dbgs() << "[traceValueDef] expected block argument, got: " << v
                      << '\n');
    return nullptr;
  }
  Operation *parent = blkArg.getOwner()->getParentOp();
  auto loop = dyn_cast<LoopLikeOpInterface>(parent);
  if (!loop)
    return blkArg;
  return traceValueDef(loop.getTiedLoopInit(blkArg)->get());
}

static memref::AllocOp traceAlloc(Value v) {
  Value maybeAlloc = traceValueDef(v);
  return dyn_cast_if_present<memref::AllocOp>(maybeAlloc.getDefiningOp());
}

// Trace a memref/tensor value through memref & tensor casts / views and
// loop iter_arg inits to find the func-op block argument it ultimately
// aliases, or nullptr if it does not originate from a function argument.
static BlockArgument traceToFuncArg(Value v) {
  while (v) {
    if (auto blkArg = dyn_cast<BlockArgument>(v)) {
      Operation *parent = blkArg.getOwner()->getParentOp();
      if (isa<func::FuncOp>(parent))
        return blkArg;
      if (auto loop = dyn_cast<LoopLikeOpInterface>(parent)) {
        OpOperand *init = loop.getTiedLoopInit(blkArg);
        if (!init)
          return nullptr;
        v = init->get();
        continue;
      }
      return nullptr;
    }
    Operation *defining = v.getDefiningOp();
    if (!defining)
      return nullptr;
    if (isa<CastOpInterface, ViewLikeOpInterface, tensor::CollapseShapeOp,
            tensor::ExpandShapeOp, tensor::ExtractSliceOp, tensor::ReshapeOp,
            bufferization::ToTensorOp, bufferization::ToMemrefOp>(defining)) {
      v = defining->getOperand(0);
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

/// Get the highest level parent op that is not the containing op
static Operation *getContainedParent(Operation *containing, Operation *inner) {
  Operation *parent = inner->getParentOp();
  while (parent && parent != containing && containing->isAncestor(inner)) {
    inner = parent;
    parent = inner->getParentOp();
  }
  return inner;
}

static tensor::InsertSliceOp createInsertSlice(OpBuilder &builder, Location loc,
                                               Value src, Value into,
                                               Value iv) {
  auto const1 = builder.getIndexAttr(1);
  auto const0 = builder.getIndexAttr(0);
  auto originalType = cast<TensorType>(src.getType());
  SmallVector<OpFoldResult> offsets, sizes, strides;
  offsets.push_back(iv);
  offsets.append(originalType.getRank(), const0);

  // Set up the sizes
  sizes.push_back(const1);
  for (int i = 0; i < originalType.getRank(); ++i) {
    if (originalType.isDynamicDim(i))
      sizes.push_back(builder.createOrFold<tensor::DimOp>(loc, src, i));
    else
      sizes.push_back(builder.getIndexAttr(originalType.getDimSize(i)));
  }

  // And strides should be all ones
  strides.append(originalType.getRank() + 1, const1);

  return builder.create<tensor::InsertSliceOp>(loc, src, into, offsets, sizes,
                                               strides);
}

static Value createExtractSlice(OpBuilder &builder, Location loc, Value from,
                                Type to, Value iv) {
  auto const1 = builder.getIndexAttr(1);
  auto const0 = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> offsets, sizes, strides;
  auto newType = cast<TensorType>(from.getType());

  // Set up offsets
  offsets.push_back(iv);
  offsets.append(newType.getRank() - 1, const0);
  // Set up sizes
  sizes.push_back(const1);
  for (int i = 1; i < newType.getRank(); ++i) {
    if (newType.isDynamicDim(i))
      sizes.push_back(builder.createOrFold<tensor::DimOp>(loc, from, i));
    else
      sizes.push_back(builder.getIndexAttr(newType.getDimSize(i)));
  }

  // ... and strides
  strides.append(newType.getRank(), const1);
  auto finalTy = cast<RankedTensorType>(to);
  return builder.create<tensor::ExtractSliceOp>(loc, finalTy, from, offsets,
                                                sizes, strides);
}

Value CVPipelineImpl::createSubview(OpBuilder &builder, Location loc,
                                    Value from, Type to, Value iv) {
  auto const1 = builder.getIndexAttr(1);
  auto const0 = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> offsets, sizes, strides;
  auto targetTy = cast<MemRefType>(to);
  offsets.push_back(iv);
  offsets.append(targetTy.getRank(), const0);
  sizes.push_back(const1);
  for (int64_t dim : targetTy.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      pipelineLoop->emitWarning(
          "[cv-pipelining] unexpected dynamic dim in target memref");
      return nullptr;
    }
    sizes.push_back(builder.getIndexAttr(dim));
  }
  strides.append(targetTy.getRank() + 1, const1);
  int64_t offset;
  SmallVector<int64_t> layoutStrides;
  if (getStridesAndOffset(targetTy, layoutStrides, offset).failed()) {
    pipelineLoop->emitWarning("[cv-pipelining] unexpected memref layout");
    return nullptr;
  }
  auto layout = StridedLayoutAttr::get(builder.getContext(),
                                       ShapedType::kDynamic, layoutStrides);
  Attribute srcMemSpace = cast<MemRefType>(from.getType()).getMemorySpace();
  auto finalTy = MemRefType::Builder(targetTy).setLayout(layout).setMemorySpace(
      srcMemSpace);
  Value subview = builder.create<memref::SubViewOp>(loc, finalTy, from, offsets,
                                                    sizes, strides);
  if (srcMemSpace != targetTy.getMemorySpace())
    subview = builder.create<memref::MemorySpaceCastOp>(
        loc, MemRefType(MemRefType::Builder(finalTy).setMemorySpace(nullptr)),
        subview);
  return subview;
}

/// Walk the pipeline loop body and record which store-like ops (FixpipeOp,
/// StoreOp) are under an active atomic effect.
void CVPipelineImpl::collectAtomicEffects() {
  std::optional<AtomicEffect> current;
  for (Operation &op : *pipelineLoop.getBody()) {
    if (auto setAtomic = dyn_cast<SetAtomicOp>(&op)) {
      if (setAtomic.getKind() != AtomicKind::NONE)
        current = AtomicEffect{setAtomic.getKind(), setAtomic.getTypeAttr()};
      else
        current = std::nullopt;
      continue;
    }
    if (current && isa<FixpipeOp, StoreOp>(&op))
      atomicEffectMap[&op] = *current;
  }
  trailingAtomicEffect = current;
  LLVM_DEBUG({
    dbgs() << "[collectAtomicEffects] Ops under atomic effect:\n";
    for (auto &[op, effect] : atomicEffectMap) {
      dbgs() << "\t" << stringifyAtomicKind(effect.kind) << " ";
      if (effect.type)
        dbgs() << effect.type;
      dbgs() << ": ";
      op->dump();
    }
  });
}

/// Walk backward from every loop-yield operand through non-core ops that
/// are not yet claimed by any WorkItem. Every such "merger" op must end up
/// owned by a WorkItem so that:
///   - it is cloned into that work-item's forOp during `migrateOps`
///     (using the per-WorkItem `irMap`, which correctly remaps both the
///     reconstructed induction variable and work-item-produced values), and
///   - its result, when it equals a `yieldedVals` entry, can be picked up
///     by the existing `yieldedOutputs` mechanism in `markOutputs` /
///     `createNewLoops`.
///
/// A chain rooted at a yield operand is absorbed into work item `W` iff
/// every chain operand that is defined inside `pipelineLoop`'s body either
///   - belongs to `W` already (work-item op result), or
///   - belongs to another op in the same merger chain, or
///   - is a block argument of `pipelineLoop` (iter_arg or the IV).
///
/// If a chain references results from two or more distinct work items, or
/// has no work-item producer at all (purely iter_arg/IV-driven), we cannot
/// safely absorb it and return failure so `run()` reverts cleanly.
LogicalResult CVPipelineImpl::absorbMergerOpsIntoWorkItems() {
  Block *body = pipelineLoop.getBody();
  Operation *terminator = body->getTerminator();

  for (Value yieldOperand : terminator->getOperands()) {
    Operation *root = yieldOperand.getDefiningOp();
    if (!root || root->getBlock() != body)
      continue; // block arg or defined outside body — nothing to absorb
    if (opToWorkItemMap.contains(root))
      continue; // already a direct work-item output

    // Collect the chain of unclaimed non-core ops feeding `yieldOperand`
    // and the set of work items that ultimately produce its data.
    SetVector<Operation *> chain;
    SmallPtrSet<WorkItem *, 4> producers;
    SmallVector<Operation *> stack{root};
    while (!stack.empty()) {
      Operation *cur = stack.pop_back_val();
      if (!chain.insert(cur))
        continue;
      for (Value operand : cur->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue; // block arg (iter_arg / IV) — fine
        if (def->getBlock() != body)
          continue; // outside pipelineLoop body — fine
        auto it = opToWorkItemMap.find(def);
        if (it != opToWorkItemMap.end()) {
          for (WorkItem *wi : it->getSecond())
            producers.insert(wi);
          continue; // don't walk through work-item ops
        }
        stack.push_back(def);
      }
    }

    if (producers.size() != 1)
      return pipelineLoop->emitWarning(
          "[cv-pipelining] cannot absorb merger chain into a single work "
          "item: the yielded value depends on ")
             << producers.size()
             << " work-item producer(s); refusing to pipeline this loop";

    WorkItem *target = *producers.begin();
    for (Operation *m : chain) {
      target->ops.insert(m);
      opToWorkItemMap[m].push_back(target);
      LLVM_DEBUG(dbgs() << "[absorbMergerOps] absorbed into work item: ";
                 m->print(dbgs()); dbgs() << '\n');
    }
  }

  // Absorb counter-update ops (arith.addi + memref.store) that follow each
  // mmadL1 using an init_cond loaded from an alloca. These ops are not
  // reachable from scf.yield so the loop above misses them.
  //
  // Pattern: memref.load %alloca -> cmpi -> init_cond operand of mmadL1
  //          mmadL1 result ... (other ops) ...
  //          arith.addi %counter, %c1
  //          memref.store %incremented, %alloca
  //
  // For each workitem, find alloca values read as init_cond, then absorb
  // any memref.store to those allocas (and their arith.addi operands).
  for (const auto &item : worklist) {
    // Collect alloca values whose load feeds an init_cond in this workitem.
    SmallPtrSet<Value, 4> initCondAllocas;
    for (Operation *op : item->ops) {
      for (Value operand : op->getOperands()) {
        // init_cond is an i1; trace cmpi -> load -> alloca
        auto cmpi = dyn_cast_if_present<arith::CmpIOp>(operand.getDefiningOp());
        if (!cmpi)
          continue;
        for (Value cmpOperand : cmpi->getOperands()) {
          auto load = dyn_cast_if_present<memref::LoadOp>(
              cmpOperand.getDefiningOp());
          if (!load)
            continue;
          Value memref = load.getMemRef();
          if (isa_and_nonnull<memref::AllocaOp>(memref.getDefiningOp()))
            initCondAllocas.insert(memref);
        }
      }
    }

    // Absorb memref.store to those allocas and their addi operand.
    for (Operation &op : *body) {
      auto store = dyn_cast<memref::StoreOp>(op);
      if (!store || !initCondAllocas.contains(store.getMemRef()))
        continue;
      if (opToWorkItemMap.contains(&op))
        continue;
      // Also absorb the defining arith.addi of the stored value.
      if (Operation *addi = store.getValue().getDefiningOp()) {
        if (isa<arith::AddIOp>(addi) && !opToWorkItemMap.contains(addi)) {
          item->ops.insert(addi);
          opToWorkItemMap[addi].push_back(item.get());
          LLVM_DEBUG(dbgs() << "[absorbMergerOps] absorbed counter addi: ";
                     addi->print(dbgs()); dbgs() << '\n');
        }
      }
      item->ops.insert(&op);
      opToWorkItemMap[&op].push_back(item.get());
      LLVM_DEBUG(dbgs() << "[absorbMergerOps] absorbed counter store: ";
                 op.print(dbgs()); dbgs() << '\n');
    }
  }

  return success();
}

/// Split loop based on separator ops into individual work items
LogicalResult CVPipelineImpl::markOutputs() {
  for (const auto &item : worklist) {
    for (Operation *op : item->ops) {
      if (isa<tensor::EmptyOp>(op))
        continue;
      // With lazy loading (kernel-level switch, per-tensor compile hint,
      // or auto cross-core legality), skip to_tensor results backed by a
      // LoadOp since the load is cloned into each consuming work item
      // directly and therefore does not need a multi-buffered cross-stage
      // tensor.
      FailureOr<bool> shouldLazy = wlBuilder.shouldLazyLoadFor(op);
      if (failed(shouldLazy))
        return failure();
      if (*shouldLazy)
        continue;
      for (Value result : op->getResults()) {
        if (yieldedVals.contains(result)) {
          unsigned opNumber = static_cast<unsigned>(std::distance(
              yieldedVals.begin(), llvm::find(yieldedVals, result)));
          item->yieldedOutputs.push_back(std::make_pair(result, opNumber));
          continue;
        }
        // For local outputs, we only care about tensor values, since
        // others will be duplicated
        if (!isa<TensorType>(result.getType()))
          continue;

        for (Operation *usr : result.getUsers()) {
          // Only top-level ops in pipelineLoop's body are inserted into
          // opToWorkItemMap; a use nested inside an scf.for / scf.if (e.g.
          // a `tensor.extract` inside a parallel-loop scf.for) is invisible
          // if we check `usr` directly. Map it up to its top-level ancestor
          // within pipelineLoop, matching populateDependencies' convention.
          if (!pipelineLoop->isAncestor(usr))
            continue;
          Operation *usrTop = getContainedParent(pipelineLoop, usr);
          if (opToWorkItemMap.contains(usrTop) && !item->ops.contains(usrTop)) {
            item->localOutputs.push_back(std::make_pair(result, nullptr));
            break;
          } // End loop over result.users
        } // End loop over op->results
      } // End loop over item->ops
    } // End loop over worklist
  }

  // Detect cross-workitem aliasing on function arguments: if a Fixpipe/Store
  // writes to a func arg that another workitem loads from, pipelining would
  // reorder the store past the load and break correctness. Walk into nested
  // regions since item->ops may contain whole region ops (scf.if / inner
  // scf.for) whose nested Fixpipe/Store/Load would otherwise be missed.
  DenseMap<BlockArgument, WorkItem *> storedFuncArgs;
  for (const auto &item : worklist) {
    for (Operation *op : item->ops) {
      op->walk([&](Operation *nested) {
        Value dst;
        // TODO: Use StoreLikeOpInterface when available
        if (auto fixpipe = dyn_cast<FixpipeOp>(nested))
          dst = fixpipe.getDst();
        else if (auto store = dyn_cast<StoreOp>(nested))
          dst = store.getDst();
        else
          return;
        BlockArgument funcArg = traceToFuncArg(dst);
        if (!funcArg)
          return;
        storedFuncArgs[funcArg] = item.get();
      });
    }
  }
  for (const auto &item : worklist) {
    for (Operation *op : item->ops) {
      WalkResult result = op->walk([&](Operation *nested) {
        // TODO: replace this with a LoadLikeOpInterface
        if (!isa<LoadOp, ND2NZOp>(nested))
          return WalkResult::advance();
        BlockArgument funcArg;
        if (auto load = dyn_cast<LoadOp>(nested))
          funcArg = traceToFuncArg(load.getSrc());
        else if (auto nd2nz = dyn_cast<ND2NZOp>(nested))
          funcArg = traceToFuncArg(nd2nz.getSrc());
        else
          llvm_unreachable("Replace with LoadLikeOpInterface.getSrc()");
        if (!funcArg)
          return WalkResult::advance();
        auto it = storedFuncArgs.find(funcArg);
        if (it != storedFuncArgs.end() && it->second != item.get()) {
          nested->emitWarning(
              "[cv-pipelining] using GM as intermediate buffer is unsupported");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (result.wasInterrupted())
        return failure();
    }
  }
  return success();
}

LogicalResult CVPipelineImpl::checkWorkItemDependencies() {
  // Values migrateOps forwards across work items via the new loops'
  // iter_args / results: every work item's local and yielded outputs.
  DenseSet<Value> trackedOutputs;
  for (const auto &item : worklist) {
    for (auto &out : item->localOutputs)
      trackedOutputs.insert(out.first);
    for (auto &out : item->yieldedOutputs)
      trackedOutputs.insert(out.first);
  }

  // migrateOps clones each work item's ops into its own per-core loop. An
  // operand is resolvable in that clone only if it is produced inside the same
  // work item (cloned alongside) or forwarded as a tracked output. Anything
  // else leaves the clone referencing the original op, which crashes once the
  // pipeline loop is erased.
  for (const auto &item : worklist) {
    auto isResolvable = [this, &trackedOutputs, &item](Value v,
                                                       Operation *clonedRoot) {
      Operation *def = v.getDefiningOp();
      if (!def || !pipelineLoop->isAncestor(def))
        return true; // block arg, or defined outside the loop
      if (clonedRoot->isAncestor(def))
        return true; // nested in the op subtree cloned with it
      if (trackedOutputs.contains(v))
        return true; // forwarded through new loop iter_args/results
      return item->ops.contains(getContainedParent(pipelineLoop, def));
    };

    for (Operation *top : item->ops) {
      WalkResult res = top->walk([&isResolvable, top](Operation *op) {
        for (Value operand : op->getOperands())
          if (!isResolvable(operand, top))
            return WalkResult::interrupt();
        return WalkResult::advance();
      });
      if (res.wasInterrupted())
        return pipelineLoop->emitWarning(
            "[cv-pipelining] cannot pipeline loop: a value crosses work items "
            "without being a tracked output (e.g. a reduction feeding a scalar "
            "scale replicated onto both cores); skipping pipelining");
    }
  }
  return success();
}

Value CVPipelineImpl::createToTensor(OpBuilder &builder, Location loc,
                                     Value src) {
  auto memref = dyn_cast<MemRefType>(src.getType());
  if (!memref) {
    pipelineLoop->emitWarning("[cv-pipelining] expected MemRefType source");
    return nullptr;
  }
  if (memref.getMemorySpace()) {
    auto newMemRef = MemRefType::get(memref.getShape(), memref.getElementType(),
                                     memref.getLayout());
    src = builder.create<memref::MemorySpaceCastOp>(loc, newMemRef, src);
  }

  return builder.create<bufferization::ToTensorOp>(loc, src, /*restrict*/ true,
                                                   /*writable*/ true);
}

/// Expand the localOutputs of each work item by number of multibuffer/pipeline
/// stages.
LogicalResult CVPipelineImpl::expandOutputInits(WorkItem &item) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(newLoop.getBody());
  for (auto &[output, expanded] : item.localOutputs) {
    Operation *defining = output.getDefiningOp();
    if (!defining)
      return pipelineLoop->emitWarning(
          "[cv-pipelining] expected work item output to be result of op");
    Location loc = defining->getLoc();
    SmallVector<int64_t> newShape({numMultibuffer});
    bufferization::ToTensorOp toTensor = nullptr;
    // We take the init and expand it
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(defining)) {
      if (dps.getNumDpsInits() != 1)
        return dps->emitWarning(
            "[cv-pipelining] expected dps op with exactly one init");
      Value init = dps.getDpsInitOperand(0)->get();
      defining = init.getDefiningOp();
      if (!defining)
        return dps->emitWarning(
            "[cv-pipelining] expected dps init to be result of op");
      if (isa<tensor::EmptyOp>(defining)) {
        auto origTy = dyn_cast<TensorType>(init.getType());
        if (!origTy)
          return defining->emitWarning(
              "[cv-pipelining] expected output to be tensor type");
        auto shapeArr = origTy.getShape();
        newShape.append(shapeArr.begin(), shapeArr.end());
        // TODO: Add support for dynamic dims
        auto newType = RankedTensorType::get(newShape, origTy.getElementType());
        expanded = builder.create<tensor::EmptyOp>(loc, newType, ValueRange());
        continue;
      }
    }
    // scf.for yielding a tensor accumulated from a `tensor.empty` iter_init.
    // This is the cross-stage output shape for the scalar-gather loop
    // recognized by `isExtractedScalarGather` in WorklistBuilder.cpp.
    // Without this case, expandOutputInits would fall through to the
    // toTensor+alloc branch (since the defining op is neither a DPS op nor
    // a `bufferization.to_tensor`) and bail with "expected to_tensor for
    // non-tensor-empty output", taking the whole pipelining round down with
    // it. Treat the iter_init like a DPS init and expand the underlying
    // tensor.empty to multibuffer shape.
    if (auto forOp = dyn_cast<scf::ForOp>(defining)) {
      auto outRes = dyn_cast<OpResult>(output);
      if (!outRes)
        return forOp->emitWarning(
            "[cv-pipelining] expected scf.for output to be an OpResult");
      unsigned resultIdx = outRes.getResultNumber();
      if (resultIdx >= forOp.getNumRegionIterArgs())
        return forOp->emitWarning(
            "[cv-pipelining] scf.for output index out of range");
      Value init = forOp.getInits()[resultIdx];
      Operation *initOp = init.getDefiningOp();
      if (initOp && isa<tensor::EmptyOp>(initOp)) {
        auto origTy = dyn_cast<TensorType>(init.getType());
        if (!origTy)
          return initOp->emitWarning(
              "[cv-pipelining] expected scf.for init to be tensor type");
        auto shapeArr = origTy.getShape();
        newShape.append(shapeArr.begin(), shapeArr.end());
        auto newType = RankedTensorType::get(newShape, origTy.getElementType());
        expanded = builder.create<tensor::EmptyOp>(loc, newType, ValueRange());
        continue;
      }
    }
    toTensor = dyn_cast<bufferization::ToTensorOp>(defining);
    if (!toTensor)
      return defining->emitWarning(
          "[cv-pipelining] expected to_tensor for non-tensor-empty output");
    // Find the alloc
    auto alloc = traceAlloc(toTensor.getMemref());
    if (!alloc)
      return toTensor->emitWarning(
          "[cv-pipelining] expected alloc from toTensor");
    auto origTy = alloc.getMemref().getType();
    if (!origTy.hasStaticShape())
      return alloc->emitWarning(
          "[cv-pipelining] expected temporary buffer to be static");
    newShape.append(origTy.getShape().begin(), origTy.getShape().end());
    auto memspace = origTy.getMemorySpace();
    auto newType = MemRefType::get(newShape, origTy.getElementType(),
                                   MemRefLayoutAttrInterface(), memspace);
    auto expandedAlloc = builder.create<memref::AllocOp>(
        loc, newType, ValueRange(), alloc.getAlignmentAttr());
    // Mark the expanded alloc with an `annotation.mark` carrying
    // `hivm.cv_pipelined_multi_buffer` so downstream passes (notably
    // the ND2NZOp aggregated-decompose pad/vbrc path) know this
    // storage is sliced into per-stage slots — any pre-init must
    // target only the current slot, never the whole alloc.
    auto markOp =
        builder.create<annotation::MarkOp>(loc, expandedAlloc.getResult());
    markOp->setAttr(hivm::CVPipelinedMultiBufferAttr::name,
                    UnitAttr::get(builder.getContext()));
    expanded = expandedAlloc;
  }
  return success();
}

LogicalResult CVPipelineImpl::expandOutputInitsForPreload(WorkItem &item) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(pipelineLoop.getBody());
  for (auto &[output, expanded] : item.localOutputs) {
    Operation *defining = output.getDefiningOp();
    if (!defining)
      return pipelineLoop->emitWarning(
          "[cv-pipelining] expected work item output to be "
          "result of op");
    Location loc = defining->getLoc();
    bufferization::ToTensorOp toTensor = nullptr;
    // We take the init and expand it
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(defining)) {
      if (dps.getNumDpsInits() != 1)
        return dps->emitWarning(
            "[cv-pipelining] expected dps op with exactly one "
            "init");
      Value init = dps.getDpsInitOperand(0)->get();
      defining = init.getDefiningOp();
      if (!defining)
        return dps->emitWarning(
            "[cv-pipelining] expected dps init to be result of "
            "op");
      if (isa<tensor::EmptyOp>(defining)) {
        continue;
      }
    }
    toTensor = dyn_cast<bufferization::ToTensorOp>(defining);
    if (!toTensor)
      return defining->emitWarning("[cv-pipelining] expected to_tensor for "
                                   "non-tensor-empty output");
    // Find the alloc
    auto alloc = traceAlloc(toTensor.getMemref());
    if (!alloc)
      return toTensor->emitWarning(
          "[cv-pipelining] expected alloc from toTensor");
    auto origTy = alloc.getMemref().getType();
    if (!origTy.hasStaticShape())
      return alloc->emitWarning(
          "[cv-pipelining] expected temporary buffer to be "
          "static");
    auto memspace = origTy.getMemorySpace();
    auto newType = MemRefType::get(origTy.getShape(), origTy.getElementType(),
                                   MemRefLayoutAttrInterface(), memspace);
    expanded = builder.create<memref::AllocOp>(loc, newType, ValueRange(),
                                               alloc.getAlignmentAttr());
    alloc.replaceAllUsesWith(expanded);
    LLVM_DEBUG(dbgs() << "[Preload expand localOutputs] alloc: "; alloc.dump());
    LLVM_DEBUG(dbgs() << "[Preload expand localOutputs] expanded: ";
               expanded.dump());
    item.ops.remove(alloc.getOperation());
    alloc->erase();
  }
  return success();
}

/// Create the unrolled newLoop to replace the original pipelineLoop, as well as
/// a jam loop for each work item
LogicalResult CVPipelineImpl::createNewLoops() {
  builder.setInsertionPoint(pipelineLoop);
  Value lb = pipelineLoop.getLowerBound();
  Value ub = pipelineLoop.getUpperBound();
  Value originStep = pipelineLoop.getStep();
  Location loc = pipelineLoop->getLoc();
  Type origTy = originStep.getType();
  Value unrollVal = builder.create<arith::ConstantOp>(
      loc, origTy, builder.getIntegerAttr(origTy, numMultibuffer));
  Value newStep = builder.create<arith::MulIOp>(loc, originStep, unrollVal);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value pipelineIters =
      builder.create<arith::ConstantIndexOp>(loc, numMultibuffer);
  newLoop =
      builder.create<scf::ForOp>(loc, lb, ub, newStep, pipelineLoop.getInits());
  newLoop->setAttr(hivm::kCVUnrolledLoopName, builder.getUnitAttr());
  if (newLoop->getNumResults() == 0)
    newLoop.getBody()->getTerminator()->erase();

  globalIRMap.map(pipelineLoop.getRegionIterArgs(),
                  newLoop.getRegionIterArgs());

  // Common values needed to create inner loops
  builder.setInsertionPointToStart(newLoop.getBody());
  IndexType indexTy = builder.getIndexType();
  Value iv = newLoop.getInductionVar();
  Value origIV = pipelineLoop.getInductionVar();
  if (!ub.getType().isIndex()) {
    ub = builder.create<arith::IndexCastOp>(loc, indexTy, ub);
    iv = builder.create<arith::IndexCastOp>(loc, indexTy, iv);
    originStep = builder.create<arith::IndexCastOp>(loc, indexTy, originStep);
  }
  AffineExpr d0, d1, s0, s1;
  MLIRContext *ctx = builder.getContext();
  bindDims(ctx, d0, d1);
  bindSymbols(ctx, s0, s1);
  // Affine map for reconstructing IV, innerIV * originalStep + outerIV
  auto ivMap = AffineMap::get(1, 2, d0 * s0 + s1, ctx);
  // d0: ub, d1: iv, d2: oldStep
  AffineExpr cappedUBExpr = (d0 - d1).ceilDiv(s0);
  Value cappedUB = builder.create<affine::AffineApplyOp>(
      loc, cappedUBExpr, ValueRange({ub, iv, originStep}));
  Value actualUB = builder.create<arith::MinUIOp>(loc, cappedUB, pipelineIters);

  for (auto &item : worklist) {
    // Reset insertion point after we're done with this item
    OpBuilder::InsertionGuard g(builder);
    if (failed(expandOutputInits(*item.get())))
      return failure();

    // Create iter arg inits in order: yieldOutputs followed by localOutputs
    SmallVector<Value> inits;
    for (auto [output, opNumber] : item->yieldedOutputs) {
      BlockArgument iterArg = newLoop.getRegionIterArg(opNumber);
      inits.push_back(iterArg);
    }
    for (auto expandedOutputPair : item->localOutputs) {
      Value expandedInit = expandedOutputPair.second;
      if (isa<TensorType>(expandedInit.getType()))
        inits.push_back(expandedInit);
    }

    // Actually create the work item loop
    item->forOp = builder.create<scf::ForOp>(loc, c0, actualUB, c1, inits);
    item->forOp->setAttrs(
        {NamedAttribute(kPipelinedLoopCoreTypeAttrName,
                        TCoreTypeAttr::get(ctx, item->core)),
         NamedAttribute(kMultibufferUnrollAttrName,
                        builder.getI32IntegerAttr(numMultibuffer))});
    builder.setInsertionPointToStart(item->forOp.getBody());
    Value workItemIV = item->forOp.getInductionVar();
    item->reconstructedIV = builder.create<affine::AffineApplyOp>(
        loc, ivMap, ValueRange{workItemIV, originStep, iv});

    // Remap yield results
    unsigned numResult = 0;
    for (auto [output, opNumber] : item->yieldedOutputs) {
      globalIRMap.map(pipelineLoop.getYieldedValues()[opNumber],
                      item->forOp->getResult(numResult++));
    }

    item->irMap = globalIRMap;
    // Remap the induction variables
    if (origIV.getType() != indexTy) {
      Value ivCast = builder.create<arith::IndexCastOp>(loc, origIV.getType(),
                                                        item->reconstructedIV);
      item->irMap.map(origIV, ivCast);
    } else
      item->irMap.map(origIV, item->reconstructedIV);

    // Remap the yield results within the work item
    unsigned yieldArg = 0;
    for (auto [output, opNumber] : item->yieldedOutputs) {
      item->irMap.map(pipelineLoop.getRegionIterArg(opNumber),
                      item->forOp.getRegionIterArg(yieldArg++));
    }

    // If inits are empty, the default builder creates a yield by default, we
    // don't want that right now so we remove it
    if (inits.empty())
      item->forOp.getBody()->getTerminator()->erase();
  }
  return success();
}

FailureOr<Value> CVPipelineImpl::updateMaskingSubview(OpBuilder &builder,
                                                      Location loc,
                                                      Value expanded,
                                                      OpOperand &initOperand,
                                                      Value iv) const {
  auto subview =
      dyn_cast<memref::SubViewOp>(initOperand.get().getDefiningOp());
  if (!subview)
    return Value(nullptr);
  if (!isa<memref::AllocOp>(subview.getSource().getDefiningOp())) {
    subview->emitWarning("[cv-pipelining] expected subview to be from alloc");
    return failure();
  }
  SmallVector<OpFoldResult> offsets, sizes, strides;
  Attribute cst1Attr = builder.getI64IntegerAttr(1);
  offsets.push_back(iv);
  offsets.append(subview.getMixedOffsets());
  sizes.push_back(cst1Attr);
  sizes.append(subview.getMixedSizes());
  strides.push_back(cst1Attr);
  strides.append(subview.getMixedStrides());
  // Set up dynamic stride
  int64_t offset;
  auto targetTy = cast<MemRefType>(initOperand.get().getType());
  SmallVector<int64_t> layoutStrides;
  if (getStridesAndOffset(targetTy, layoutStrides, offset).failed()) {
    subview->emitWarning("[cv-pipelining] unexpected memref layout");
    return failure();
  }
  auto layout = StridedLayoutAttr::get(builder.getContext(),
                                       ShapedType::kDynamic, layoutStrides);
  auto finalTy = MemRefType::Builder(targetTy).setLayout(layout);
  auto newSubView = builder.create<memref::SubViewOp>(loc, finalTy, expanded,
                                                      offsets, sizes, strides);
  subview->replaceAllUsesWith(newSubView);
  return Value(newSubView);
}

/// Actually migrate/clone each op for each work item
LogicalResult CVPipelineImpl::migrateOps() {
  for (Operation &op : pipelineLoop.getBody()->getOperations()) {
    auto it = opToWorkItemMap.find(&op);
    if (it == opToWorkItemMap.end()) {
      LLVM_DEBUG(dbgs() << "[cv-pipelining] Skipping: "; op.dump());
      continue;
    }
    for (WorkItem *target : it->getSecond()) {
      builder.setInsertionPointToEnd(target->forOp.getBody());
      auto atomicIt = atomicEffectMap.find(&op);
      if (atomicIt != atomicEffectMap.end()) {
        auto &effect = atomicIt->getSecond();
        builder.create<SetAtomicOp>(op.getLoc(), effect.kind, effect.type);
      }
      builder.clone(op, target->irMap);
      if (atomicIt != atomicEffectMap.end()) {
        builder.create<SetAtomicOp>(op.getLoc(), AtomicKind::NONE,
                                    atomicIt->getSecond().type);
      }
    }
  }
  LLVM_DEBUG(dbgs() << "\n\n[migrateOps] After cloning:\n";
             newLoop->getParentOfType<func::FuncOp>()->dump());

  // Update outputs
  SmallVector<Value> yieldVals;
  for (auto &item : worklist) {
    // Yield outputs come before the local outputs in yield/iter args
    for (auto [orig, argNo] : item->yieldedOutputs) {
      Value newVal = item->irMap.lookup(orig);
      yieldVals.push_back(newVal);
    }

    auto argIt =
        item->forOp.getRegionIterArgs().begin() + item->yieldedOutputs.size();
    auto resIt = item->forOp.getResults().begin() + item->yieldedOutputs.size();
    Value iv = item->forOp.getInductionVar();
    for (auto [orig, expanded] : item->localOutputs) {
      Operation *defining = orig.getDefiningOp();
      LLVM_DEBUG(dbgs() << "orig: " << orig << "\n\texpanded: " << expanded
                        << '\n');
      if (auto toTensor =
              dyn_cast_if_present<bufferization::ToTensorOp>(defining)) {
        // Set `defining` to the op that writes to the tensor i.e. the actual
        // defining op for the tensor
        auto it = this->outputMemrefMap.find(toTensor);
        if (it == this->outputMemrefMap.end()) {
          LLVM_DEBUG(dbgs() << "[cv-pipelining] localOutput to_tensor has no "
                               "tracked writer (outputMemrefMap miss): ";
                     toTensor->dump());
          return failure();
        }
        defining = it->second;
      }
      defining = item->irMap.lookup(defining);

      // Migration counterpart of the scf.for case in expandOutputInits:
      // when the workitem's cross-stage output is a tensor yielded by an
      // scf.for over a `tensor.empty` iter_init, the loop has no DPS
      // interface to bind to and the existing `dyn_cast<DPS>` would fail.
      // Treat the iter_init operand like a DPS init: rewire it to an
      // extract_slice of the pipelined loop's iter_arg, then insert_slice
      // the cloned loop's result back at yield. Outside-loop users get
      // per-stage extract_slices, mirroring the post-dispatch logic below.
      if (auto clonedFor = dyn_cast<scf::ForOp>(defining)) {
        auto origRes = dyn_cast<OpResult>(orig);
        if (!origRes)
          return clonedFor->emitWarning(
              "[cv-pipelining] expected scf.for output to be an OpResult");
        unsigned resultIdx = origRes.getResultNumber();
        if (resultIdx >= clonedFor.getNumRegionIterArgs())
          return clonedFor->emitWarning(
              "[cv-pipelining] scf.for output index out of range");
        builder.setInsertionPoint(clonedFor);
        Location loc = clonedFor->getLoc();
        Value newResult = *resIt;
        Value extracted =
            createExtractSlice(builder, loc, *argIt, orig.getType(), iv);
        OpOperand &initOperand =
            clonedFor.getInitsMutable()[resultIdx];
        initOperand.set(extracted);
        // Also retype the matching iter_arg so the body uses the slice type.
        clonedFor.getRegionIterArg(resultIdx).setType(extracted.getType());
        Value newOutput = clonedFor->getResult(resultIdx);
        newOutput.setType(extracted.getType());
        builder.setInsertionPointAfterValue(newOutput);
        Value yieldVal = createInsertSlice(builder, loc, newOutput, *argIt, iv);
        orig.replaceUsesWithIf(newOutput, [&](OpOperand &use) {
          return item->forOp->isAncestor(use.getOwner());
        });
        resIt++;
        argIt++;
        yieldVals.push_back(yieldVal);
        // Update outside users (mirrors the post-dispatch logic below).
        SmallVector<OpOperand *> toReplaceFor;
        for (OpOperand &use : orig.getUses()) {
          Operation *owner = use.getOwner();
          if (pipelineLoop->isAncestor(owner) ||
              item->forOp->isAncestor(owner))
            continue;
          toReplaceFor.push_back(&use);
        }
        for (OpOperand *use : toReplaceFor) {
          Operation *owner = use->getOwner();
          Operation *ownerLoop = getContainedParent(newLoop, owner);
          Value userIV = cast<scf::ForOp>(ownerLoop).getInductionVar();
          builder.setInsertionPoint(owner);
          Value perStage =
              createExtractSlice(builder, loc, newResult, orig.getType(), userIV);
          use->set(perStage);
        }
        continue;
      }

      auto dps = dyn_cast_if_present<DestinationStyleOpInterface>(defining);
      if (!dps)
        return pipelineLoop->emitWarning(
            "[cv-pipelining] expected destination passing style op for output");
      builder.setInsertionPoint(dps);
      Location loc = dps->getLoc();

      if (dps.getNumDpsInits() != 1)
        return dps->emitWarning(
            "[cv-pipelining] expected dps op with exactly one init");
      OpOperand *initOperand = dps.getDpsInitOperand(0);
      Value newResult = *resIt;
      // Dispatch on `expanded`'s type, not on the DPS init's static type.
      // `expandOutputInits` chose `tensor.empty` vs `memref.alloc` based on
      // whether the init was a real `tensor.empty` or a `to_tensor` of an
      // alloc — so a tensor-typed init backed by a `to_tensor` (e.g. the
      // cross-core `hivm.hir.copy` writing into a pre-allocated L1 buffer)
      // gets a `memref.alloc` for `expanded` and must take the memref path
      // here. Driving off `initOperand->get().getType()` would mis-route those
      // to the tensor path and crash in `createExtractSlice` on a memref
      // iter_arg.
      if (isa<TensorType>(expanded.getType())) {
        Value extracted =
            createExtractSlice(builder, loc, *argIt, orig.getType(), iv);
        initOperand->set(extracted);
        Value newOutput = dps->getResult(0);
        builder.setInsertionPointAfterValue(newOutput);
        Value yieldVal = createInsertSlice(builder, loc, newOutput, *argIt, iv);
        orig.replaceUsesWithIf(newOutput, [&](OpOperand &use) {
          return item->forOp->isAncestor(use.getOwner());
        });
        resIt++;
        argIt++;
        yieldVals.push_back(yieldVal);
      } else if (auto targetTy = dyn_cast<MemRefType>(expanded.getType())) {
        // Find the inner `bufferization.to_tensor` that wraps the alloc-backed
        // buffer. Two shapes:
        //   - fixpipe-output flow: `orig` itself is the to_tensor (the DPS
        //     init is a plain memref alloc), so look up `orig` in irMap.
        //   - cross-core copy flow: `orig` is the DPS op's tensor result and
        //     the to_tensor is the DPS init operand itself.
        // The cloned init's defining op IS the cloned to_tensor in the
        // copy case, so try that first, then fall back.
        auto innerToTensor = dyn_cast_if_present<bufferization::ToTensorOp>(
            initOperand->get().getDefiningOp());
        if (!innerToTensor) {
          Value internalDef = item->irMap.lookup(orig);
          innerToTensor = dyn_cast_if_present<bufferization::ToTensorOp>(
              internalDef.getDefiningOp());
        }
        // If there are masking subviews, update those first
        FailureOr<Value> updatedSubviewOr =
            updateMaskingSubview(builder, loc, expanded, *initOperand, iv);
        if (failed(updatedSubviewOr))
          return failure();
        Value updatedSubview = *updatedSubviewOr;
        // Then replace the toTensor operand if it is not updated
        if (!innerToTensor)
          return dps->emitWarning("[cv-pipelining] expected memref outputs to "
                                  "be passed as tensors");
        OpOperand *memrefOperand = &innerToTensor.getMemrefMutable();
        if (memrefOperand->get() != updatedSubview) {
          // Always build the subview against a memref type — `initOperand`
          // may itself be a tensor when the writer's DPS init is a
          // `to_tensor` of an alloc (e.g. cross-core `hivm.hir.copy`, whose
          // L1 destination is presented as a tensor). Driving createSubview
          // off `initOperand`'s type would crash there.
          builder.setInsertionPointToStart(item->forOp.getBody());
          Value toTensorSubview = createSubview(
              builder, loc, expanded, memrefOperand->get().getType(), iv);
          if (!toTensorSubview)
            return failure();
          // If the DPS init is itself a memref (e.g. fixpipe writing
          // directly to an alloc), redirect it onto the multibuffered slot.
          // If it is a tensor backed by a `to_tensor` (e.g. the cross-core
          // copy case), leave it alone — rewriting the inner toTensor's
          // memref operand below is enough to redirect the writer.
          //
          // The fixpipe writes to a UB-typed memref but `toTensorSubview`
          // has been address-space-stripped to match the to_tensor's memref
          // operand. Recover the pre-cast UB-typed subview for the writer
          // so the fixpipe verifier and downstream codegen see the correct
          // address space; otherwise the writer is treated as an
          // unspecified-aspace store and the staged result is corrupted.
          if (!updatedSubview &&
              isa<MemRefType>(initOperand->get().getType())) {
            Value writerSubview = toTensorSubview;
            if (auto cast =
                    toTensorSubview.getDefiningOp<memref::MemorySpaceCastOp>())
              writerSubview = cast.getSource();
            initOperand->set(writerSubview);
          }
          memrefOperand->set(toTensorSubview);
        }
        builder.setInsertionPointAfter(item->forOp);
        newResult = createToTensor(builder, loc, expanded);
        if (!newResult)
          return failure();
      } else
        return dps->emitWarning("[cv-pipelining] unexpected output type that "
                                "is not tensor or memref");

      // Update outside users
      LLVM_DEBUG(dbgs() << "[cv-pipelining] Updating user of "; orig.dump());
      SmallVector<OpOperand *> toReplace;
      for (OpOperand &use : orig.getUses()) {
        Operation *owner = use.getOwner();
        LLVM_DEBUG(dbgs().indent(4) << *owner << '\n');
        if (pipelineLoop->isAncestor(owner) || item->forOp->isAncestor(owner)) {
          LLVM_DEBUG(dbgs().indent(8) << "Not in user loop, skipped\n");
          continue;
        }
        toReplace.push_back(&use);
      }
      for (OpOperand *use : toReplace) {
        Operation *owner = use->getOwner();
        // At this point the loop should only contain the pipeline loops we
        // created
        Operation *ownerLoop = getContainedParent(newLoop, owner);
        Value userIV = cast<scf::ForOp>(ownerLoop).getInductionVar();
        builder.setInsertionPoint(owner);
        Value newUse = createExtractSlice(builder, owner->getLoc(), newResult,
                                          use->get().getType(), userIV);
        use->set(newUse);
      }
    }

    builder.setInsertionPointToEnd(item->forOp.getBody());
    builder.create<scf::YieldOp>(item->forOp->getLoc(), yieldVals);
    yieldVals.clear();
  }
  builder.setInsertionPointToEnd(newLoop.getBody());
  if (trailingAtomicEffect) {
    builder.create<SetAtomicOp>(pipelineLoop->getLoc(),
                                trailingAtomicEffect->kind,
                                trailingAtomicEffect->type);
  }
  builder.clone(*pipelineLoop.getBody()->getTerminator(), globalIRMap);
  return success();
}

LogicalResult CVPipelineImpl::createNewLoopsForPreloadWithScopes() {
  int32_t preloadNum = static_cast<int32_t>(worklist.size()) - 1;
  for (auto &item : worklist) {
    // Reset insertion point after we're done with this item
    OpBuilder::InsertionGuard g(builder);
    if (failed(expandOutputInitsForPreload(*item.get())))
      return failure();

    LLVM_DEBUG(dbgs() << "Creating scope for work item #" << item->id << '\n');
    scf::ForOp parentFor = pipelineLoop;

    // collect return values
    SmallVector<Value> returnTensors{};
    for (auto &localOutput : item->localOutputs) {
      returnTensors.push_back(localOutput.first);
    }
    for (auto &yieldedOutput : item->yieldedOutputs) {
      returnTensors.push_back(yieldedOutput.first);
    }
    if (returnTensors.empty())
      return success();

    builder.setInsertionPoint(parentFor.getBody()->getTerminator());
    Location loc = pipelineLoop->getLoc();

    auto newScopeOp =
        builder.create<scope::ScopeOp>(loc, TypeRange(returnTensors));
    newScopeOp.setNoInline(true);
    newScopeOp->setAttr(kPipelinedLoopCoreTypeAttrName,
                        TCoreTypeAttr::get(builder.getContext(), item->core));
    newScopeOp->setAttr(
        "preload_num",
        IntegerAttr::get(IntegerType::get(newScopeOp->getContext(), 32),
                         preloadNum));

    Region &region = newScopeOp.getRegion();
    Block *bodyBlock = builder.createBlock(&region);
    builder.setInsertionPointToEnd(bodyBlock);
    IRMapping scopeMap(globalIRMap);

    Value origIV = pipelineLoop.getInductionVar();
    Value mappedIV = origIV;

    if (!origIV.getType().isIndex()) {
      mappedIV = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                                    origIV);
      scopeMap.map(origIV, mappedIV);
    } else {
      scopeMap.map(origIV, mappedIV);
    }

    LLVM_DEBUG(dbgs() << "Created scope for work item #" << item->id << " with "
                      << returnTensors.size() << " results\n");
    for (Operation &op : parentFor.getBody()->getOperations()) {
      if (!item->ops.contains(&op))
        continue;

      builder.clone(op, scopeMap);
      toErase.insert(&op);
    }

    builder.setInsertionPointToEnd(bodyBlock);
    SmallVector<Value> newReturnTensors;

    for (auto returnTensor : returnTensors) {
      if (scopeMap.contains(returnTensor)) {
        Value newReturnTensor = scopeMap.lookup(returnTensor);
        newReturnTensors.push_back(newReturnTensor);
      } else {
        newReturnTensors.push_back(returnTensor);
      }
    }
    builder.create<scope::ReturnOp>(loc, ValueRange(newReturnTensors));

    size_t resultIdx = 0;
    for (auto &localOutput : item->localOutputs) {
      Value returnTensor = localOutput.first;
      Value scopeResult = newScopeOp->getResult(resultIdx++);
      localOuputsToRedurnRes[returnTensor] = scopeResult;
      globalIRMap.map(returnTensor, scopeResult);
    }

    for (auto &yieldedOutput : item->yieldedOutputs) {
      Value returnTensor = yieldedOutput.first;
      Value scopeResult = newScopeOp->getResult(resultIdx++);

      pipelineLoop.getBody()->getTerminator()->replaceUsesOfWith(returnTensor,
                                                                 scopeResult);
      globalIRMap.map(returnTensor, scopeResult);
    }

    item->scopeOp = newScopeOp;
    preloadNum--;
  }

  return success();
}

LogicalResult CVPipelineImpl::markScopesForPreload() {
  toErase.clear();

  if (failed(createNewLoopsForPreloadWithScopes()))
    return failure();

  LLVM_DEBUG({
    for (auto item : worklist) {
      dbgs() << "after createNewLoopsForPreloadWithScopes WorkItem #"
             << item->id << ":---------------\n";
      item->scopeOp->dump();
      if (!item->localOutputs.empty())
        dbgs() << "\tLocal outputs:\n";
      for (auto p : item->localOutputs) {
        Value output = p.first;
        dbgs().indent(4) << output << '\n';
      }
      if (!item->yieldedOutputs.empty())
        dbgs() << "\tYield outputs:\n";
      for (auto [output, number] : item->yieldedOutputs)
        dbgs().indent(4) << output << " at " << number << '\n';
    }
  });

  LLVM_DEBUG(dbgs() << "\n\nAfter clone before erase:\n";
             pipelineLoop->getParentOfType<func::FuncOp>()->dump());

  LLVM_DEBUG(dbgs() << "toErase.size() scope for work item all"
                    << toErase.size() << " results\n");
  // Clean up
  Operation *eraseOp = *toErase.begin();
  while (!toErase.empty()) {
    if (eraseOp == nullptr)
      eraseOp = *toErase.begin();
    auto usrBegin = eraseOp->user_begin();
    if (usrBegin == eraseOp->user_end()) {
      LLVM_DEBUG({
        dbgs() << "eraseOp:";
        eraseOp->dump();
        dbgs() << ":---------------\n";
      });
      eraseOp->erase();
      toErase.erase(eraseOp);
      eraseOp = nullptr;
      continue;
    }
    Operation *usrOp = *usrBegin;
    Operation *usrParent = usrOp->getParentOp();
    while (!isa<func::FuncOp>(usrParent)) {
      if (toErase.contains(usrParent)) {
        usrOp = usrParent;
        break;
      }
      if (!usrParent)
        return eraseOp->emitWarning(
            "[cv-pipelining] reached null parent while tracing users");
      usrParent = usrParent->getParentOp();
    }

    if (!toErase.contains(usrOp)) {
      LLVM_DEBUG(dbgs() << "func" << "\n\nDef: " << *eraseOp
                        << "\nUser: " << *usrOp << '\n');
      return usrOp->emitWarning(
          "[cv-pipelining] cannot erase user of pipelined op, aborting "
          "pipelining pass");
    }
    eraseOp = usrOp;
  }
  return success();
}

void CVPipelineImpl::revert() {
  if (newLoop)
    newLoop->erase();
}

/// Main method of the pass
LogicalResult CVPipelineImpl::run() {
  collectAtomicEffects();
  auto buildResult = wlBuilder.build();
  if (failed(buildResult)) {
    revert();
    return failure();
  }
  worklist = buildResult->worklist;
  opToWorkItemMap = buildResult->opToWorkItemMap;
  outputMemrefMap = buildResult->outputMemrefMap;
  numMultibuffer = buildResult->resolvedMultibuffer;
  if (failed(absorbMergerOpsIntoWorkItems())) {
    revert();
    return failure();
  }
  if (failed(markOutputs()))
    return failure();
  LLVM_DEBUG({
    for (auto item : worklist) {
      dbgs() << "WorkItem #" << item->id << ":\n";
      for (Operation *op : item->ops)
        op->dump();
      if (!item->localOutputs.empty())
        dbgs() << "\tLocal outputs:\n";
      for (auto p : item->localOutputs) {
        Value output = p.first;
        dbgs().indent(4) << output << '\n';
      }
      if (!item->yieldedOutputs.empty())
        dbgs() << "\tYield outputs:\n";
      for (auto [output, number] : item->yieldedOutputs)
        dbgs().indent(4) << output << " at " << number << '\n';
    }
  });

  // No IR has been mutated yet; reject partitions migrateOps can't realize so
  // we bail cleanly instead of crashing in pipelineLoop->erase() later.
  if (failed(checkWorkItemDependencies()))
    return failure();

  // Preload pipeline reuse workitems with cvpipeline.
  if (enableSkewMode) {
    return markScopesForPreload();
  }

  if (failed(createNewLoops())) {
    revert();
    return failure();
  }
  if (failed(migrateOps())) {
    revert();
    return failure();
  }

  pipelineLoop.replaceAllUsesWith(newLoop.getResults());

  LLVM_DEBUG(dbgs() << "\n\nAfter everything:\n";
             newLoop->getParentOfType<func::FuncOp>()->dump());
  if (failed(newLoop.verify())) {
    revert();
    return failure();
  }
  pipelineLoop->erase();
  return success();
}

void CVPipeliningPass::runOnOperation() {
  // First find loop to operate on
  func::FuncOp func = getOperation();
  DenseSet<scf::ForOp> pipelinedLoops;

  // Disabled via options
  if (this->pipelineDepth == 1 || this->pipelineDepth == 0)
    return;

  // Check if we should skip the entire pass due to autoblockify with NormalizeMatmul counter
  bool hasAutoblockifyLoop = false;
  bool hasNormalizeMatmulCounter = false;
  static constexpr llvm::StringLiteral kAutoBlockifySubloopAttr = "autoblockify.subloop";
  
  // Check for autoblockify loop tag
  func->walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(kAutoBlockifySubloopAttr)) {
      hasAutoblockifyLoop = true;
    }
  });

  // Check for NormalizeMatmul counter (attribute on storeOp)
  func->walk([&](memref::StoreOp storeOp) {
    if (storeOp->hasAttr(hivm::TCoreTypeAttr::name)) {
      auto coreTypeAttr = storeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
      if (coreTypeAttr && coreTypeAttr.getTcoretype() == hivm::TCoreType::CUBE_AND_VECTOR) {
        hasNormalizeMatmulCounter = true;
      }
    }
  });

  // Skip entire CV Pipelining pass if there's autoblockify loop with NormalizeMatmul counter
  if (hasAutoblockifyLoop && hasNormalizeMatmulCounter) {
    LLVM_DEBUG(dbgs() << "[cv-pipelining] Skipping entire pass due to autoblockify loop with NormalizeMatmul counter\n");
    return;
  }

  func->walk<WalkOrder::PreOrder>([&pipelinedLoops, this](scf::ForOp loop) {
    auto parentLoop = loop->getParentOfType<scf::ForOp>();

    // Check if this is a part of pipelined loop already
    while (parentLoop) {
      if (pipelinedLoops.contains(parentLoop))
        return;
      parentLoop = parentLoop->getParentOfType<scf::ForOp>();
    }

    CVPipelineImpl impl(loop, this->pipelineDepth, this->enableSkewMode,
                        this->enableLazyLoading);
    if (impl.run().succeeded())
      pipelinedLoops.insert(loop);
  });
}

std::unique_ptr<Pass>
hivm::createCVPipeliningPass(const CVPipeliningOptions &options) {
  return std::make_unique<CVPipeliningPass>(options);
}
} // namespace mlir
