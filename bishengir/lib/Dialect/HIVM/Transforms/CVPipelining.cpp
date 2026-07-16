//===- CVPipelining.cpp --- Pipelining pass for mix-cv ops ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/TileUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Utils/WorkItem.h"
#include "bishengir/Dialect/HIVM/Utils/WorklistBuilder.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
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

using bishengir::memref_ext::AllocWorkspaceOp;
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
  CVPipelineImpl(LoopLikeOpInterface loop, int multibuffer,
                 CVPipelineMode pipelineMode, bool enableLazyLoading)
      : pipelineLoop(loop), newLoop(nullptr), builder(loop->getContext()),
        numMultibuffer(multibuffer), pipelineMode(pipelineMode),
        wlBuilder(cast<scf::ForOp>(loop.getOperation()), multibuffer,
                  enableLazyLoading),
        yieldedVals(loop.getYieldedValues().begin(),
                    loop.getYieldedValues().end()) {
    builder.setInsertionPoint(loop);
    checkpoint = builder.clone(*loop);
  }

  LogicalResult run();

private:
  void collectAtomicEffects();

  /// For each marked counter alloca whose value is incremented inside a
  /// regioned op (e.g. the scf.if after a matmul), clone that op and erase all
  /// CUBE ops inside the clone, so a vector-only stage can keep the counter
  /// advancing. Returns alloca -> vector-safe clone.
  LogicalResult preprocessCounterAllocas();

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

  // For each `annotation.mark` with
  // `DuplicateTensorExtractForCube::replacementLabel` inside `pipelineLoop`,
  // clone the scalar/control op chain that derives from the marked original
  // value and feeds into CUBE consumers, rewiring those consumers to read
  // from the cloned chain rooted at the replacement value (which lives on a
  // VECTOR side path produced by an `inserted-store`).
  LogicalResult duplicateExtractScalarForCube();

  LogicalResult markOutputs();

  LogicalResult collectWorkspaceAllocsForPreload();

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

  void expandWorkspace(OpBuilder &builder);
  LogicalResult migrateOpsForPreload(OpBuilder &builder);
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

  // Pipeline mode for CV-pipelining.
  CVPipelineMode pipelineMode;

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

  // Workspace allocations expanded for preload-mode CV pipelining.
  DenseMap<AllocWorkspaceOp, WorkspaceAllocParams> workspaceAllocs_;
  DenseMap<Value, Value> expandedWorkspaceMap_;

  // If the atomic effect is still active at the end of the loop body, this
  // holds that trailing state so it can be restored after the pipelined loops.
  std::optional<AtomicEffect> trailingAtomicEffect;

  // Mapping from the original pipelineLoop to the newLoop to guide the cloning
  // process
  IRMapping globalIRMap;

  DenseSet<Operation *> toErase;

  // Checkpoint for revert in case things go wrong
  Operation *checkpoint;

  // Marked counter alloca -> vector-safe clone of the op that increments it.
  DenseMap<Value, Operation *> counterCloneMap;
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

static AllocWorkspaceOp traceAllocWorkspace(Value v) {
  Value maybeAlloc = traceValueDef(v);
  return dyn_cast_if_present<AllocWorkspaceOp>(maybeAlloc.getDefiningOp());
}

static AllocWorkspaceOp getAllocWorkspace(Value v) {
  Operation *defining = v.getDefiningOp();
  if (!defining)
    return nullptr;

  if (auto alloc = dyn_cast<AllocWorkspaceOp>(defining))
    return alloc;

  if (isa<CastOpInterface, ViewLikeOpInterface, tensor::CollapseShapeOp,
          tensor::ExpandShapeOp, tensor::ExtractSliceOp, tensor::ReshapeOp,
          bufferization::ToTensorOp, bufferization::ToMemrefOp>(defining))
    return getAllocWorkspace(defining->getOperand(0));

  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(defining)) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return nullptr;
    OpOperand *tiedOperand = dpsOp.getTiedOpOperand(result);
    if (!tiedOperand)
      return nullptr;
    return getAllocWorkspace(tiedOperand->get());
  }
  return nullptr;
}

// Trace `v` back through casts/views/iter_arg inits to either a
// `memref::AllocOp` or a `memref_ext::AllocWorkspaceOp`. Returns nullptr if
// the value does not resolve to one of those.
static Operation *traceAllocLike(Value v) {
  Value maybeAlloc = traceValueDef(v);
  if (!maybeAlloc)
    return nullptr;
  Operation *defOp = maybeAlloc.getDefiningOp();
  if (isa_and_present<memref::AllocOp, bishengir::memref_ext::AllocWorkspaceOp>(
          defOp))
    return defOp;
  return nullptr;
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

static bool isTCBLocalOutput(const std::pair<Value, Value> &localOutput) {
  Value localBuffer = traceValueDef(localOutput.second);
  if (!localBuffer)
    return false;

  for (Operation *user : localBuffer.getUsers())
    if (utils::isAnnotationWithAttr(
            user, hivm::HIVMTightlyCoupledBufferAttr::name))
      return true;

  return false;
}

static bool isFreshOutputInit(Value init) {
  Operation *defining = init.getDefiningOp();
  if (isa_and_nonnull<tensor::EmptyOp, memref::AllocOp>(defining))
    return true;

  auto toTensor = dyn_cast_if_present<bufferization::ToTensorOp>(defining);
  if (!toTensor)
    return false;
  Value root = traceValueDef(toTensor.getMemref());
  return root && isa_and_nonnull<memref::AllocOp>(root.getDefiningOp());
}

static void createAttrForPreloadWS(OpBuilder &builder, Value markedVal) {
  Operation *markedOp = markedVal.getDefiningOp();
  if (markedOp)
    markedOp->setAttr(hivm::PreloadWorkspaceAttr::name, builder.getUnitAttr());
}

static Value createWorkspaceSubview(OpBuilder &builder, Location loc,
                                    Value from, Value iv,
                                    bool isPreload = false) {
  auto const1 = builder.getIndexAttr(1);
  auto const0 = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> offsets, sizes, strides;
  auto newType = cast<MemRefType>(from.getType());

  offsets.push_back(iv);
  offsets.append(newType.getRank() - 1, const0);
  sizes.push_back(const1);
  for (int i = 1; i < newType.getRank(); ++i) {
    if (newType.isDynamicDim(i))
      sizes.push_back(builder.createOrFold<memref::DimOp>(loc, from, i));
    else
      sizes.push_back(builder.getIndexAttr(newType.getDimSize(i)));
  }
  strides.append(newType.getRank(), const1);

  auto subview =
      builder.create<memref::SubViewOp>(loc, from, offsets, sizes, strides);
  if (isPreload)
    createAttrForPreloadWS(builder, subview);

  SmallVector<ReassociationIndices> reass{{0, 1}};
  for (unsigned i = 2; i < subview.getType().getRank(); ++i)
    reass.push_back({i});
  return builder.create<memref::CollapseShapeOp>(loc, subview, reass);
}

static Operation *cloneStoreLikeToWorkspace(OpBuilder &builder, Operation *op,
                                            Value newDst) {
  OperationState state(op->getLoc(), op->getName().getStringRef());
  state.addOperands(op->getOperands());
  state.operands[1] = newDst;
  state.addAttributes(op->getAttrs());
  return builder.create(state);
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
      return pipelineLoop->emitWarning("[cv-pipelining] cannot absorb merger "
                                       "chain into a single work "
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

  // Counter-advance absorption (normalize-matmul counter advanced directly in
  // the loop body). The `memref.store %inc` back to a counter alloca and the
  // `arith.addi` producing %inc have no SSA result and never reach scf.yield,
  // so the yield-chain walk above misses them and migrateOps would drop them,
  // leaving the post-loop load reading the initial 0. (The regioned-op case is
  // handled separately by preprocessCounterAllocas's vector-safe clone.)
  for (const auto &item : worklist) {
    SmallPtrSet<Value, 4> loadedCounters;
    for (Operation *op : item->ops)
      if (auto load = dyn_cast<memref::LoadOp>(op))
        if (auto a = load.getMemRef().getDefiningOp<memref::AllocaOp>())
          if (a->hasAttr(kNormalizeMatmulCounterAttr))
            loadedCounters.insert(load.getMemRef());

    for (Operation &op : *body) {
      auto store = dyn_cast<memref::StoreOp>(&op);
      if (!store || !loadedCounters.contains(store.getMemRef()) ||
          opToWorkItemMap.contains(&op))
        continue;
      if (Operation *inc = store.getValue().getDefiningOp()) {
        if (isa<arith::AddIOp>(inc) && !opToWorkItemMap.contains(inc)) {
          item->ops.insert(inc);
          opToWorkItemMap[inc].push_back(item.get());
          LLVM_DEBUG(dbgs() << "[absorbMergerOps] absorbed counter addi: ";
                     inc->print(dbgs()); dbgs() << '\n');
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

LogicalResult CVPipelineImpl::collectWorkspaceAllocsForPreload() {
  workspaceAllocs_.clear();
  if (worklist.empty())
    return success();

  unsigned preloadSlots = static_cast<unsigned>(worklist.size());

  for (Operation &op : pipelineLoop.getBody()->getOperations()) {
    if (auto mark = dyn_cast<annotation::MarkOp>(&op)) {
      if (auto alloc = traceAllocWorkspace(mark.getSrc())) {
        auto &info = workspaceAllocs_[alloc];
        info.multibuffer = preloadSlots;
        info.marker = mark;
      }
      continue;
    }

    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(&op)) {
      auto alloc = traceAllocWorkspace(toTensor.getMemref());
      if (!alloc)
        continue;
      auto &info = workspaceAllocs_[alloc];
      info.multibuffer = preloadSlots;
      info.toTensor = toTensor;
    }
  }

  SmallVector<AllocWorkspaceOp> incompleteAllocs;
  for (auto &[alloc, info] : workspaceAllocs_) {
    if (!info.marker || !info.toTensor) {
      incompleteAllocs.push_back(alloc);
      continue;
    }
    if (!info.toTensor.getResult().hasOneUse())
      return info.toTensor->emitWarning(
          "[cv-pipelining] expected preload workspace tensor to have one use");
  }
  for (AllocWorkspaceOp alloc : incompleteAllocs)
    workspaceAllocs_.erase(alloc);
  return success();
}

void CVPipelineImpl::expandWorkspace(OpBuilder &builder) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(pipelineLoop);
  for (auto &[alloc, info] : workspaceAllocs_) {
    Location loc = alloc.getLoc();
    MemRefType origType = alloc.getType();
    ArrayRef<int64_t> origShape = origType.getShape();

    SmallVector<int64_t> newShape = {static_cast<int64_t>(info.multibuffer)};
    newShape.append(origShape.begin(), origShape.end());
    auto newType = MemRefType::get(newShape, origType.getElementType());
    auto newAlloc = builder.create<AllocWorkspaceOp>(
        loc, newType, alloc.getWorkspaceArg(), alloc.getDynamicSize(),
        alloc.getOffset());

    expandedWorkspaceMap_[alloc] = newAlloc;

    info.marker.getSrcMutable().set(newAlloc);
    info.marker->removeAttr(MultiBufferAttr::name);
    info.marker->setAttr(hivm::PreloadWorkspaceAttr::name, builder.getUnitAttr());

    toErase.insert(alloc);
    toErase.insert(info.toTensor);
  }
}

/// Split loop based on separator ops into individual work items
LogicalResult CVPipelineImpl::markOutputs() {
  for (const auto &item : worklist) {
    for (Operation *op : item->ops) {
      if (pipelineMode == CVPipelineMode::Skew) {
        auto dps = dyn_cast<DestinationStyleOpInterface>(op);
        if (dps && isa<StoreOp, FixpipeOp>(op) && dps.getNumDpsInits() == 1) {
          auto alloc = getAllocWorkspace(dps.getDpsInitOperand(0)->get());
          if (alloc && workspaceAllocs_.contains(alloc)) {
            item->workspaceOutputs.push_back(op);
            continue;
          }
        }
      }
      if (isa<tensor::EmptyOp>(op))
        continue;
      // With lazy loading (kernel-level switch, per-tensor compile hint,
      // or auto cross-core legality), skip to_tensor results backed by a
      // load-like writer (LoadOp or ND2NZOp) since the writer is cloned into
      // each consuming work item directly and therefore does not need a
      // multi-buffered cross-stage tensor.
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
          if (opToWorkItemMap.contains(usrTop) &&
              llvm::any_of(opToWorkItemMap[usrTop],
                           [op](const WorkItem *usrWI) {
                             return !usrWI->ops.contains(op);
                           })) {
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
    for (Operation *workspaceOutput : item->workspaceOutputs)
      for (Value result : workspaceOutput->getResults())
        trackedOutputs.insert(result);
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

  // ---------------------------------------------------------------------------
  // Reject cross-core loop-carried dependencies.
  //
  // CV pipelining splits the loop body into a VECTOR stage and a CUBE stage,
  // each of which runs all of its iterations in its own loop before the other
  // stage's loop runs. A tensor carried across the iteration boundary (a loop
  // iter_arg) that is *produced* on one core but *consumed* on the other can
  // therefore never be honored: the consuming stage would read the loop-entry
  // value instead of the previous iteration's result computed by the other
  // core. The split happens silently, so the existing operand-resolvability
  // walk above does not catch it — an iter_arg is a BlockArgument and so is
  // always deemed "resolvable" there. Detect it here, before any IR mutation,
  // and leave the loop un-pipelined.
  //
  // Same-core carries (e.g. a CUBE matmul accumulator) are fine: that stage's
  // own loop runs sequentially. Non-tensor carries (loop-index / address
  // arithmetic) are replicated onto every stage and never form a cross-core
  // data hazard, so they are ignored.
  DenseMap<unsigned, const WorkItem *> producerByIterArg;
  for (const auto &item : worklist)
    for (auto &yielded : item->yieldedOutputs)
      producerByIterArg[yielded.second] = item.get();

  // Collect the cross-core consuming ops of an iter_arg, paired with the core
  // they run on. An op mapped in opToWorkItemMap is the consuming work item; an
  // unmapped glue op (view / cast / index math) is followed to its own users.
  auto consumerUses = [this](BlockArgument iterArg) {
    SmallVector<std::pair<TCoreType, Operation *>> uses;
    DenseSet<Operation *> visited;
    SmallVector<Operation *> stack(iterArg.getUsers().begin(),
                                   iterArg.getUsers().end());
    while (!stack.empty()) {
      Operation *top = getContainedParent(pipelineLoop, stack.pop_back_val());
      if (!top || isa<scf::YieldOp>(top) || !visited.insert(top).second)
        continue;
      auto it = opToWorkItemMap.find(top);
      if (it != opToWorkItemMap.end()) {
        for (const WorkItem *wi : it->second)
          uses.push_back({wi->core, top});
        continue; // stop at the consuming work item; don't walk past it
      }
      for (Operation *next : top->getUsers())
        stack.push_back(next);
    }
    return uses;
  };

  ArrayRef<BlockArgument> iterArgs = pipelineLoop.getRegionIterArgs();
  ValueRange yieldedValues = pipelineLoop.getYieldedValues();
  for (unsigned pos = 0, e = iterArgs.size(); pos < e; ++pos) {
    BlockArgument iterArg = iterArgs[pos];
    if (!isa<TensorType>(iterArg.getType()))
      continue; // only tensor data can form a cross-core hazard
    auto prodIt = producerByIterArg.find(pos);
    if (prodIt == producerByIterArg.end())
      continue; // not produced by a work item (e.g. forwarded unchanged)
    TCoreType producerCore = prodIt->second->core;
    if (producerCore != TCoreType::CUBE && producerCore != TCoreType::VECTOR)
      continue;
    for (auto [consumerCore, consumerOp] : consumerUses(iterArg)) {
      if (consumerCore != TCoreType::CUBE && consumerCore != TCoreType::VECTOR)
        continue;
      if (consumerCore == producerCore)
        continue;
      // Name the offending iter_arg and the two cores, and point at the ops
      // that produce and consume the carried value so the user can locate the
      // dependency in a larger kernel.
      InFlightDiagnostic diag =
          pipelineLoop->emitWarning()
          << "[cv-pipelining] cannot pipeline loop: loop-carried tensor "
             "iter_arg #"
          << pos << " is produced on the " << stringifyTCoreType(producerCore)
          << " core but consumed on the " << stringifyTCoreType(consumerCore)
          << " core across the iteration boundary; skipping pipelining";
      if (Operation *producerOp = yieldedValues[pos].getDefiningOp())
        diag.attachNote(producerOp->getLoc())
            << "loop-carried value produced here on the "
            << stringifyTCoreType(producerCore) << " core";
      diag.attachNote(consumerOp->getLoc())
          << "and consumed here on the " << stringifyTCoreType(consumerCore)
          << " core the next iteration";
      return diag;
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
      // All DMA ops should have the CopyOpInterface, and they can't be simply
      // inserted into an empty tensor due to addrspace handling
      if (isa<tensor::EmptyOp>(defining) ||
          !isa<CopyOpInterface>(dps.getOperation())) {
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
      if (isa_and_nonnull<tensor::EmptyOp>(initOp)) {
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
    // Scalar workspace stores (`hivm.hir.store` writing back to an
    // `alloc_workspace`-backed scalar tensor) surface the alloc_workspace's
    // memref result as the cross-stage value rather than the writer's tensor
    // result. Route through the to_tensor that wraps the alloc so the rest
    // of the flow sees a tensor-typed output it can expand uniformly.
    if (auto wsAlloc =
            dyn_cast<bishengir::memref_ext::AllocWorkspaceOp>(defining)) {
      for (Operation *u : wsAlloc->getUsers()) {
        if (auto tt = dyn_cast<bufferization::ToTensorOp>(u)) {
          toTensor = tt;
          defining = tt;
          break;
        }
      }
    }
    if (!toTensor)
      toTensor = dyn_cast<bufferization::ToTensorOp>(defining);
    if (!toTensor)
      return defining->emitWarning(
          "[cv-pipelining] expected to_tensor for non-tensor-empty output");
    // Find the alloc — either a plain `memref.alloc` or a workspace alloc
    // (`memref_ext.alloc_workspace`). Workspace allocs carve a slice out of
    // a function-arg workspace and are expanded the same way: we add a
    // leading multibuffer dim and preserve the workspace arg / dynamic-size /
    // offset operands so PlanMemory can re-place the enlarged region.
    Operation *allocOp = traceAllocLike(toTensor.getMemref());
    if (!allocOp)
      return toTensor->emitWarning(
          "[cv-pipelining] expected alloc from toTensor");
    auto origTy = cast<MemRefType>(allocOp->getResult(0).getType());
    if (!origTy.hasStaticShape())
      return allocOp->emitWarning(
          "[cv-pipelining] expected temporary buffer to be static");
    newShape.append(origTy.getShape().begin(), origTy.getShape().end());
    auto memspace = origTy.getMemorySpace();
    auto newType = MemRefType::get(newShape, origTy.getElementType(),
                                   MemRefLayoutAttrInterface(), memspace);
    Value expandedResult;
    if (auto plainAlloc = dyn_cast<memref::AllocOp>(allocOp)) {
      expandedResult =
          builder
              .create<memref::AllocOp>(loc, newType, ValueRange(),
                                       plainAlloc.getAlignmentAttr())
              .getResult();
    } else {
      auto wsAlloc = cast<bishengir::memref_ext::AllocWorkspaceOp>(allocOp);
      expandedResult = builder
                           .create<bishengir::memref_ext::AllocWorkspaceOp>(
                               loc, newType, wsAlloc.getWorkspaceArg(),
                               wsAlloc.getDynamicSize(), wsAlloc.getOffset())
                           .getResult();
    }
    // Mark the expanded alloc with an `annotation.mark` carrying
    // `hivm.cv_pipelined_multi_buffer` so downstream passes (notably
    // the ND2NZOp aggregated-decompose pad/vbrc path) know this
    // storage is sliced into per-stage slots — any pre-init must
    // target only the current slot, never the whole alloc.
    auto markOp = builder.create<annotation::MarkOp>(loc, expandedResult);
    markOp->setAttr(hivm::CVPipelinedMultiBufferAttr::name,
                    UnitAttr::get(builder.getContext()));
    expanded = expandedResult;
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
  auto subview = dyn_cast<memref::SubViewOp>(initOperand.get().getDefiningOp());
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
        OpOperand &initOperand = clonedFor.getInitsMutable()[resultIdx];
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
          if (pipelineLoop->isAncestor(owner) || item->forOp->isAncestor(owner))
            continue;
          toReplaceFor.push_back(&use);
        }
        for (OpOperand *use : toReplaceFor) {
          Operation *owner = use->getOwner();
          Operation *ownerLoop = getContainedParent(newLoop, owner);
          Value userIV = cast<scf::ForOp>(ownerLoop).getInductionVar();
          builder.setInsertionPoint(owner);
          Value perStage = createExtractSlice(builder, loc, newResult,
                                              orig.getType(), userIV);
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
        Value init = initOperand->get();
        if (isFreshOutputInit(init)) {
          Value extracted =
              createExtractSlice(builder, loc, *argIt, orig.getType(), iv);
          initOperand->set(extracted);
        }
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
        // Workspace scalar store: `hivm.hir.store` writing through a
        // `to_tensor` of an `alloc_workspace`. Rewire the cloned store to
        // write directly into the per-slot memref subview of `expanded` and
        // drop the intermediate `to_tensor` — the post-jam-loop `to_tensor`
        // below is the only one any consumer should read from.
        auto clonedStore = dyn_cast<hivm::StoreOp>(dps.getOperation());
        Operation *backingAlloc =
            innerToTensor ? traceAllocLike(innerToTensor.getMemref()) : nullptr;
        if (clonedStore && innerToTensor &&
            isa_and_present<bishengir::memref_ext::AllocWorkspaceOp>(
                backingAlloc)) {
          builder.setInsertionPointToStart(item->forOp.getBody());
          Value memrefSlot = createSubview(
              builder, loc, expanded, innerToTensor.getMemref().getType(), iv);
          if (!memrefSlot)
            return failure();
          builder.setInsertionPoint(clonedStore);
          builder.create<hivm::StoreOp>(clonedStore.getLoc(), TypeRange{},
                                        clonedStore.getSrc(), memrefSlot);
          // The cloned store carries a result tensor used by a cloned
          // `annotation.mark` (the VECTOR tcore_type marker inserted next to
          // the store). Drop those marker users so the store can be erased.
          SmallVector<Operation *> markUsers;
          for (Operation *u : clonedStore->getUsers())
            if (isa<annotation::MarkOp>(u))
              markUsers.push_back(u);
          for (Operation *u : markUsers)
            u->erase();
          clonedStore.erase();
          if (innerToTensor.use_empty())
            innerToTensor.erase();
          builder.setInsertionPointAfter(item->forOp);
          newResult = createToTensor(builder, loc, expanded);
          if (!newResult)
            return failure();
        } else {
          // If there are masking subviews, update those first
          FailureOr<Value> updatedSubviewOr =
              updateMaskingSubview(builder, loc, expanded, *initOperand, iv);
          if (failed(updatedSubviewOr))
            return failure();
          Value updatedSubview = *updatedSubviewOr;
          // Then replace the toTensor operand if it is not updated
          if (!innerToTensor)
            return dps->emitWarning(
                "[cv-pipelining] expected memref outputs to "
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
              if (auto cast = toTensorSubview
                                  .getDefiningOp<memref::MemorySpaceCastOp>())
                writerSubview = cast.getSource();
              initOperand->set(writerSubview);
            }
            memrefOperand->set(toTensorSubview);
          }
          builder.setInsertionPointAfter(item->forOp);
          newResult = createToTensor(builder, loc, expanded);
          if (!newResult)
            return failure();
        }
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

LogicalResult CVPipelineImpl::migrateOpsForPreload(OpBuilder &builder) {
  for (auto &item : worklist) {
    for (Operation *output : item->workspaceOutputs) {
      auto dpsOp = cast<DestinationStyleOpInterface>(output);
      Value wsAlloc = getAllocWorkspace(dpsOp.getDpsInitOperand(0)->get());
      if (!wsAlloc)
        continue;

      auto expandedIt = expandedWorkspaceMap_.find(wsAlloc);
      if (expandedIt == expandedWorkspaceMap_.end())
        return output->emitWarning(
            "[cv-pipelining] missing expanded preload workspace");

      Operation *storeLikeOp = item->irMap.lookupOrDefault(output);
      if (!storeLikeOp || storeLikeOp == output)
        return output->emitWarning(
            "[cv-pipelining] missing cloned preload workspace writer");

      Location loc = output->getLoc();
      builder.setInsertionPoint(storeLikeOp);
      Value sliceIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value newDst =
          createWorkspaceSubview(builder, loc, expandedIt->second, sliceIdx,
                                 /*isPreload=*/true);
      cloneStoreLikeToWorkspace(builder, storeLikeOp, newDst);

      builder.setInsertionPointAfter(item->scopeOp);
      auto workspaceTensor = builder.create<bufferization::ToTensorOp>(
          loc, expandedIt->second, /*restrict=*/true);

      for (OpOperand &operand :
           llvm::make_early_inc_range(storeLikeOp->getUses())) {
        Operation *userOp = operand.getOwner();
        if (!isa<LoadOp, ND2NZOp>(userOp))
          continue;
        builder.setInsertionPoint(userOp);
        Value loadSliceIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
        Value sliceOp =
            createExtractSlice(builder, loc, workspaceTensor,
                               operand.get().getType(), loadSliceIdx);
        createAttrForPreloadWS(builder, sliceOp);
        operand.set(sliceOp);
      }

      storeLikeOp->erase();
    }
  }
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

    // TCB local outputs are backed by local buffers and are rebuilt as
    // tensor views after the scope.
    bool hasTCBLocalOutput = false;
    SmallVector<Value> returnTensors{};
    for (auto &localOutput : item->localOutputs) {
      if (isTCBLocalOutput(localOutput)) {
        hasTCBLocalOutput = true;
      } else {
        returnTensors.push_back(localOutput.first);
      }
    }
    for (auto &yieldedOutput : item->yieldedOutputs) {
      returnTensors.push_back(yieldedOutput.first);
    }
    if (returnTensors.empty() && item->workspaceOutputs.empty() &&
        !hasTCBLocalOutput) {
      preloadNum--;
      continue;
    }

    builder.setInsertionPoint(parentFor.getBody()->getTerminator());
    Location loc = pipelineLoop->getLoc();

    auto newScopeOp =
        builder.create<scope::ScopeOp>(loc, TypeRange(returnTensors));
    newScopeOp.setNoInline(true);
    newScopeOp->setAttr(kPipelinedLoopCoreTypeAttrName,
                        TCoreTypeAttr::get(builder.getContext(), item->core));
    newScopeOp->setAttr(
        hivm::PreloadNumAttr::name,
        IntegerAttr::get(IntegerType::get(newScopeOp->getContext(), 32),
                         preloadNum));
    newScopeOp->setAttr(
        hivm::MaxPreloadNumAttr::name,
        IntegerAttr::get(IntegerType::get(newScopeOp->getContext(), 32),
                         static_cast<int32_t>(worklist.size())));

    Region &region = newScopeOp.getRegion();
    Block *bodyBlock = builder.createBlock(&region);
    builder.setInsertionPointToEnd(bodyBlock);
    IRMapping scopeMap(globalIRMap);

    Value origIV = pipelineLoop.getInductionVar();
    scopeMap.map(origIV, origIV);

    LLVM_DEBUG(dbgs() << "Created scope for work item #" << item->id << " with "
                      << returnTensors.size() << " results\n");

    // Skip original TCB to_tensor ops here. They are rebuilt after the
    // scope from the same local buffers.
    DenseSet<Operation *> tcbToTensorOps;
    for (auto &localOutput : item->localOutputs) {
      if (!isTCBLocalOutput(localOutput))
        continue;
      auto toTensorOp =
          localOutput.first.getDefiningOp<bufferization::ToTensorOp>();
      if (toTensorOp)
        tcbToTensorOps.insert(toTensorOp);
    }
    for (Operation &op : parentFor.getBody()->getOperations()) {
      if (!item->ops.contains(&op))
        continue;
      toErase.insert(&op);
      if (tcbToTensorOps.contains(&op))
        continue;
      builder.clone(op, scopeMap);
    }
    item->irMap = scopeMap;

    for (Operation *workspaceOutput : item->workspaceOutputs) {
      for (Value result : workspaceOutput->getResults()) {
        if (scopeMap.contains(result))
          globalIRMap.map(result, scopeMap.lookup(result));
      }
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

    builder.setInsertionPointAfter(newScopeOp);
    for (auto &localOutput : item->localOutputs) {
      if (!isTCBLocalOutput(localOutput))
        continue;
      // TODO: Avoid the need to generate bufferization.to_tensor.
      Value toTensor = createToTensor(builder, loc, localOutput.second);
      if (!toTensor)
        return failure();
      globalIRMap.map(localOutput.first, toTensor);
    }

    size_t resultIdx = 0;
    for (auto &localOutput : item->localOutputs) {
      if (isTCBLocalOutput(localOutput))
        continue;
      Value returnTensor = localOutput.first;
      Value scopeResult = newScopeOp->getResult(resultIdx++);
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

  if (failed(migrateOpsForPreload(builder))) {
    revert();
    return failure();
  }

  LLVM_DEBUG({
    for (auto item : worklist) {
      dbgs() << "after createNewLoopsForPreloadWithScopes WorkItem #"
             << item->id << ":---------------\n";
      if (item->scopeOp)
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
  if (toErase.empty())
    return success();
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
  checkpoint->erase();
  return success();
}

void CVPipelineImpl::revert() {
  if (newLoop)
    newLoop->erase();
  pipelineLoop->replaceAllUsesWith(checkpoint->getResults());
  pipelineLoop->erase();
}

LogicalResult CVPipelineImpl::preprocessCounterAllocas() {
  Block *body = pipelineLoop.getBody();
  // (1) Collect marked counter allocas. They live directly in the loop body,
  // so a plain iteration suffices. Any alloca here must be a known counter.
  SmallVector<memref::AllocaOp> counters;
  for (Operation &op : *body) {
    auto alloca = dyn_cast<memref::AllocaOp>(&op);
    if (!alloca)
      continue;
    if (!alloca->hasAttr(kNormalizeMatmulCounterAttr))
      return alloca->emitWarning(
          "[cv-pipelining] unexpected alloca without counter attribute");
    counters.push_back(alloca);
  }

  for (memref::AllocaOp alloca : counters) {
    // (2) Find the increment site: a store of (alloca + k) back to alloca,
    // nested inside a regioned op rather than the loop body itself.
    Operation *regioned = nullptr;
    for (Operation *user : alloca->getUsers()) {
      auto store = dyn_cast<memref::StoreOp>(user);
      if (!store || store.getMemRef() != alloca.getResult())
        continue;
      if (!isa_and_nonnull<arith::AddIOp>(store.getValue().getDefiningOp()))
        continue;
      Operation *top = getContainedParent(pipelineLoop, store);
      if (top && top != store && top->getNumRegions() > 0) {
        regioned = top;
        break;
      }
    }
    if (!regioned)
      continue;

    // Clone the regioned op and strip every CUBE op from the clone, leaving a
    // vector-safe skeleton that still advances the counter.
    builder.setInsertionPointAfter(regioned);
    Operation *clone = builder.clone(*regioned);
    SmallVector<Operation *> cubeOps;
    clone->walk([&](Operation *inner) {
      if (queryCoreTypeHelper(inner).value_or(TCoreType::CUBE_OR_VECTOR) ==
          TCoreType::CUBE)
        cubeOps.push_back(inner);
    });
    for (Operation *cube : llvm::reverse(cubeOps)) {
      // A surviving (vector) op may still read a cube result; feed it a fresh
      // tensor.empty of the same type so the skeleton stays well-formed.
      builder.setInsertionPoint(cube);
      for (Value res : cube->getResults()) {
        if (res.use_empty())
          continue;
        if (auto tensorTy = dyn_cast<RankedTensorType>(res.getType()))
          res.replaceAllUsesWith(builder.create<tensor::EmptyOp>(
              cube->getLoc(), tensorTy, ValueRange{}));
      }
      cube->erase();
    }
    counterCloneMap[alloca.getResult()] = clone;
  }
  return success();
}

namespace {
// Return the ancestor of `op` that lives directly inside `block`, or nullptr
// if `op` is not nested within `block`.
Operation *ancestorInBlock(Operation *op, const Block *block) {
  Operation *cur = op;
  while (cur) {
    if (cur->getBlock() == block)
      return cur;
    cur = cur->getParentOp();
  }
  return nullptr;
}
} // namespace

LogicalResult CVPipelineImpl::duplicateExtractScalarForCube() {
  static constexpr llvm::StringLiteral kReplacementLabel =
      "DuplicateTensorExtractForCube::replacementLabel";

  SmallVector<annotation::MarkOp> replacementMarks;
  pipelineLoop.walk([&](annotation::MarkOp markOp) {
    if (markOp->hasAttr(kReplacementLabel))
      replacementMarks.push_back(markOp);
  });

  for (annotation::MarkOp markOp : replacementMarks) {
    Value original = markOp.getSrc();
    if (markOp.getValues().empty())
      continue;
    Value replacement = markOp.getValues()[0];

    // Block defining `original` — the block we will clone into. Tainted ops
    // outside this block are not handled here.
    Block *defBlock = nullptr;
    if (Operation *defOp = original.getDefiningOp())
      defBlock = defOp->getBlock();
    else if (auto blkArg = dyn_cast<BlockArgument>(original))
      defBlock = blkArg.getOwner();
    if (!defBlock)
      continue;

    // Forward closure: scalar/control ops in `defBlock` that transitively
    // consume `original`. Cube/vector ops are recorded separately and are
    // not descended into.
    DenseSet<Operation *> tainted;
    DenseSet<Operation *> cubeConsumers;
    DenseSet<Value> visitedValues;
    SmallVector<Value> wl;
    wl.push_back(original);
    while (!wl.empty()) {
      Value v = wl.pop_back_val();
      if (!visitedValues.insert(v).second)
        continue;
      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        Operation *ancestor = ancestorInBlock(user, defBlock);
        if (!ancestor)
          continue;
        if (ancestor == markOp.getOperation())
          continue;
        TCoreType ancCore =
            getCoreType(ancestor).value_or(TCoreType::CUBE_OR_VECTOR);
        if (ancCore == TCoreType::CUBE) {
          cubeConsumers.insert(ancestor);
          continue;
        }
        if (ancCore == TCoreType::VECTOR)
          continue;
        if (tainted.insert(ancestor).second) {
          for (Value res : ancestor->getResults())
            wl.push_back(res);
        }
      }
    }

    // Reverse reachability: keep only tainted ops on some path to a cube
    // consumer. Seed with tainted ops whose results are directly consumed by
    // an op in `cubeConsumers`, then walk operand chains backward (including
    // inside nested regions, so e.g. an scf.if body's references to other
    // tainted results are picked up).
    DenseSet<Operation *> needed;
    SmallVector<Operation *> seed;
    for (Operation *op : tainted) {
      bool reachesCube = false;
      for (Value res : op->getResults()) {
        for (Operation *u : res.getUsers()) {
          Operation *ua = ancestorInBlock(u, defBlock);
          if (ua && cubeConsumers.contains(ua)) {
            reachesCube = true;
            break;
          }
        }
        if (reachesCube)
          break;
      }
      if (reachesCube)
        seed.push_back(op);
    }
    while (!seed.empty()) {
      Operation *op = seed.pop_back_val();
      if (!needed.insert(op).second)
        continue;
      op->walk([&](Operation *nested) {
        for (Value operand : nested->getOperands()) {
          Operation *def = operand.getDefiningOp();
          if (!def)
            continue;
          Operation *defAnc = ancestorInBlock(def, defBlock);
          if (defAnc && tainted.contains(defAnc) && !needed.contains(defAnc))
            seed.push_back(defAnc);
        }
      });
    }

    bool originalUsedDirectly = false;
    for (Operation *consumer : cubeConsumers) {
      consumer->walk([&](Operation *opIn) {
        for (Value operand : opIn->getOperands()) {
          if (operand == original) {
            originalUsedDirectly = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (originalUsedDirectly)
        break;
    }
    if (needed.empty() && !originalUsedDirectly)
      continue;

    // Clone needed ops in IR order, mapping `original → replacement`.
    // `builder.clone` updates the mapping with each cloned result, so
    // subsequent clones and the consumer rewires below resolve their
    // operands via `irMap`.
    IRMapping irMap;
    irMap.map(original, replacement);

    builder.setInsertionPointAfter(markOp);
    for (Operation &op : *defBlock) {
      if (!needed.contains(&op))
        continue;
      Operation *cloned = builder.clone(op, irMap);
      builder.setInsertionPointAfter(cloned);
    }

    // Rewire cube consumers to consume the cloned chain. Walk the entire
    // consumer subtree so nested uses are rerouted as well.
    for (Operation *consumer : cubeConsumers) {
      consumer->walk([&](Operation *opIn) {
        for (OpOperand &operand : opIn->getOpOperands()) {
          Value v = operand.get();
          if (v == original) {
            operand.set(replacement);
            continue;
          }
          if (Value mapped = irMap.lookupOrNull(v))
            operand.set(mapped);
        }
      });
    }

    // Drop the replacement-label `annotation.mark`. It paired the original
    // value with its cube replacement so the later split-mix-kernel pass
    // could rewrite cube uses; we have already done that rewrite by cloning
    // the chain above, so leaving the mark around just produces dangling
    // cross-workitem operands once cv-pipelining clones the mark into
    // every WI that mentions either side of the pair.
    markOp->erase();
  }

  return success();
}

/// Main method of the pass
LogicalResult CVPipelineImpl::run() {
  if (failed(duplicateExtractScalarForCube()))
    return failure();
  collectAtomicEffects();
  if (failed(preprocessCounterAllocas())) {
    revert();
    return failure();
  }
  wlBuilder.setCounterClones(counterCloneMap);
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
  if (pipelineMode == CVPipelineMode::Skew &&
      failed(collectWorkspaceAllocsForPreload()))
    return failure();
  if (failed(markOutputs())) {
    revert();
    return failure();
  }
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
  if (failed(checkWorkItemDependencies())) {
    revert();
    return failure();
  }

  // Preload pipeline reuse workitems with cvpipeline.
  if (pipelineMode == CVPipelineMode::Skew) {
    expandWorkspace(builder);
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
  checkpoint->erase();
  return success();
}

void CVPipeliningPass::runOnOperation() {
  // First find loop to operate on
  func::FuncOp func = getOperation();
  SmallVector<scf::ForOp> pipelineCandidates;

  // Disabled via options
  if (this->pipelineDepth == 1 || this->pipelineDepth == 0)
    return;

  // Disable CVP once batchmatmul is found
  SmallVector<func::FuncOp> funcOps{func};
  if (hasBatchMatmulLoopInAicFuncs(funcOps))
    return;

  // We want to work on the innermost loop first, so post order walk
  func->walk<WalkOrder::PostOrder>([&pipelineCandidates](scf::ForOp loop) {
    pipelineCandidates.push_back(loop);
  });

  DenseSet<scf::ForOp> pipelinedNest;
  for (auto loop : pipelineCandidates) {
    // Don't attempt pipeline if nested loop succeeded already
    if (pipelinedNest.contains(loop))
      continue;

    auto parentLoop = loop->getParentOfType<scf::ForOp>();
    CVPipelineImpl impl(loop, this->pipelineDepth, this->pipelineMode,
                        this->enableLazyLoading);

    // Mark all parent loops to not attempt pipelining to save compile time
    if (impl.run().succeeded())
      while (parentLoop) {
        pipelinedNest.insert(parentLoop);
        parentLoop = parentLoop->getParentOfType<scf::ForOp>();
      }
  }
}

std::unique_ptr<Pass>
hivm::createCVPipeliningPass(const CVPipeliningOptions &options) {
  return std::make_unique<CVPipeliningPass>(options);
}
} // namespace mlir
