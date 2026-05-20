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
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

static constexpr llvm::StringLiteral CubeOnlyAttrName = "pipeline.cubeonly";
static constexpr llvm::StringLiteral VecOnlyAttrName = "pipeline.veconly";

// Per-tensor compile hint. Attached to a bufferization::ToTensorOp result via
// `annotation.mark %t {cv_pipeline_lazy_load = true}` (or `false`).
//   * `true`  : opts this tensor into the lazy-load path even when the
//               kernel-level enable-lazy-loading switch is off.
//   * `false` : opts this tensor out of the lazy-load path even when the
//               kernel-level switch is on; a warning is emitted to flag the
//               override.
static constexpr llvm::StringLiteral CVPipelineLazyLoadAttrName =
    "cv_pipeline_lazy_load";
namespace {

// Tristate result of reading a `cv_pipeline_lazy_load` hint off of a value.
enum class LazyLoadHint {
  // No `annotation.mark` op carrying `cv_pipeline_lazy_load` on this value.
  None,
  // `cv_pipeline_lazy_load = true` -> opt-in.
  Enable,
  // `cv_pipeline_lazy_load = false` -> opt-out.
  Disable,
};

struct AtomicEffect {
  AtomicKind kind;
  TypeAttr type;
};

struct WorkspaceAllocParams {
  unsigned multibuffer;
  annotation::MarkOp marker;
  bufferization::ToTensorOp toTensor;
};

struct WorkItem {
  // Values that are referred by other work items later will be stored in this
  // list. Everything here requires the tensor types to be expanded by
  // Multibuffer times, e.g. <16xf16> into <2x16xf16>
  SmallVector<std::pair<Value, Value>> localOutputs;

  DenseSet<Operation *> ops;

  // Values that are yielded in the parent for loop
  SmallVector<std::pair<Value, unsigned>> yieldedOutputs;

  // Vector or Cube, other types shouldn't end up in here
  TCoreType core;

  // After unrolling the parent for loop, the upper bound for "reroll"ed loops
  // are computed and inserted here. Created in "unrollOuterLoop" Value
  // upperBound; The for op corresponding to the multibuffering, constructed in
  // "constructPipelineLoop"
  scf::ForOp forOp;

  IRMapping irMap;

  // Reconstructed original induction variable
  Value reconstructedIV;

  // ScopeOp for single cube or vector
  scope::ScopeOp scopeOp;
#ifndef NDEBUG
  int id;
#endif
};

struct CVPipelineImpl {
  CVPipelineImpl(LoopLikeOpInterface loop, int multibuffer, bool skewMode,
                 bool enableLazyLoading)
      : pipelineLoop(loop), newLoop(nullptr), builder(loop->getContext()),
        numMultibuffer(multibuffer), enableSkewMode(skewMode),
        enableLazyLoading(enableLazyLoading),
        yieldedVals(loop.getYieldedValues().begin(),
                    loop.getYieldedValues().end()) {}

  LogicalResult run();

private:
  LogicalResult createWorkItems();

  LogicalResult populateDependencies(Operation *separator);

  void populateLoopCarriedDependencies();

  LogicalResult extractAvailableOps(SmallVector<Operation *> &extractedOps,
                                    TCoreType &core);

  LogicalResult populateWorkItem(SmallVector<Operation *> &availableOps,
                                 TCoreType core);

  LogicalResult traceDependentOps(WorkItem *item);

  LogicalResult traceMemrefSubnet(Operation *start,
                                  SmallVector<Operation *> &workingStack);

  LogicalResult traceOperands(Value operand, WorkItem *item,
                              SmallVector<Operation *> &workingStack);

  // Returns the tristate cv_pipeline_lazy_load hint carried on `v`'s direct
  // users (looking for an `annotation.mark` op with attr key
  // `cv_pipeline_lazy_load`):
  //   * Enable  -> attribute present and set to `true`
  //   * Disable -> attribute present and set to `false`
  //   * None    -> no such mark / attribute on `v`
  static LazyLoadHint getLazyLoadHint(Value v);

  // Returns true if the input op should be treated as lazy-loaded.  Two
  // shapes of input are accepted (dispatched via `dyn_cast`):
  //   * `bufferization::ToTensorOp`: the to_tensor must be registered in
  //     `outputMemrefMap` and its backing writer must be a LoadOp.  When
  //     not registered (or backed by a non-LoadOp), the answer is `false`.
  //   * `LoadOp`: the matching to_tensor is found by reverse-looking up the
  //     LoadOp in `outputMemrefMap`.  When no match is found, the answer
  //     falls back to the kernel-level `enableLazyLoading` (or auto
  //     cross-core if the load has a tensor result).
  //
  // The auto cross-core check is a LEGALITY signal: when the load's
  // tensor result is consumed by both CUBE and VECTOR cores, lazy
  // loading is required for correctness (otherwise the consumer core
  // has no local copy of the data).  This signal cannot be vetoed by
  // the per-tensor hint.
  //
  // Precedence (in order):
  //   * isCrossCoreLoad signals failure (CUBE_OR_VECTOR consumer
  //     encountered): propagate failure -- the caller should bail out
  //     of pipelining since we cannot prove safety.
  //   * isCrossCore = true: always true; if the hint is Disable, a
  //     "hint is ignored" warning is emitted.  Users who really need
  //     to opt out can disable cv-pipelining entirely.
  //   * hint = Enable  -> true
  //   * hint = Disable -> false; if `enableLazyLoading` is on, a
  //     "hint overrides kernel switch" warning is emitted.
  //   * hint = None    -> `enableLazyLoading`
  FailureOr<bool> shouldLazyLoadFor(Operation *op);

  // Returns true if `loaded` (the tensor value produced by a load, either
  // a bufferization.to_tensor over the load's output memref or the load's
  // own tensor result) has consumers on both the CUBE and the VECTOR
  // cores.  Walks transitive users, descending through view-like
  // passthrough ops (tensor.extract_slice / cast / expand_shape /
  // collapse_shape / reshape) and skipping annotation/debug ops.  For
  // each remaining user, the core type is read from `pipeline.cubeonly`
  // / `pipeline.veconly` attrs (set by `illegalRegionedOp` on regioned
  // ops) or, failing that, from `queryCoreTypeHelper`.  CUBE_AND_VECTOR
  // runs on both cores and sets both flags on its own.  CUBE_OR_VECTOR
  // is ambiguous (either core could execute the op); we cannot safely
  // classify the load and return `failure()` so the caller can bail
  // out of pipelining for this loop.
  FailureOr<bool> isCrossCoreLoad(const Value loaded) const;

  // Emit a one-time warning explaining that a per-tensor
  // `cv_pipeline_lazy_load = false` hint on `v` has been IGNORED because
  // the load's tensor result is consumed by both CUBE and VECTOR cores;
  // lazy loading is required for correctness and cannot be vetoed by the
  // hint.  Uses the same warning bookkeeping as `warnHintOverride` so
  // each mark is reported at most once.
  void warnHintIgnoredForCrossCore(Value v);

  // Emit a one-time warning explaining that the per-tensor
  // `cv_pipeline_lazy_load = false` hint on `v` overrides the kernel-level
  // enable-lazy-loading switch.  The warning is emitted on the underlying
  // `hivm.hir.load` op (looked up via outputMemrefMap) for source-location
  // clarity, with a note attached to the originating `annotation.mark`.
  // Falls back to emitting on the mark itself if no load is found.  Does
  // nothing if the same mark op has already been warned for.
  void warnHintOverride(Value v);

  // Walk all `annotation.mark` ops inside pipelineLoop carrying
  // `cv_pipeline_lazy_load` and diagnose three classes of misuse:
  //   * Duplicate marks: a value carries >=2 such marks; warns on the first
  //     and attaches notes on the rest (current "first-wins" policy
  //     preserved in `getLazyLoadHint`).
  //   * Non-to_tensor target: mark.src is not produced by
  //     `bufferization::ToTensorOp`; the hint can never be honored.
  //   * Non-load-backed target: src is a to_tensor whose backing writer is
  //     not a `LoadOp`; the hint will be silently ignored otherwise.
  // Requires `outputMemrefMap` to be populated (call after
  // `populateDependencies` runs for every separator).
  void diagnoseLazyLoadHints();
  LogicalResult traceNonInitOperands(Operation *op, WorkItem *item,
                                     SmallVector<Operation *> &workingStack);

  void collectAtomicEffects();

  LogicalResult markOutputs();

  LogicalResult expandOutputInits(WorkItem *item);
  LogicalResult expandOutputInitsForPreload(WorkItem *item);

  LogicalResult createNewLoops();

  void mapOpToItem(Operation *op, WorkItem *item);

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
                                        Value expanded, OpOperand *initOperand,
                                        Value iv);

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

  // When true, load ops are cloned into each consuming work item rather than
  // passing their results via expanded multi-buffered tensors.
  bool enableLazyLoading;

  // Pipelines we focus on that will be pipelined, everything else will be
  // traced from these based on the operands
  DenseSet<Operation *> toBePipelined;

  // Mapping from the converted memref to the op that writes to it (i.e.
  // FixPipeOp)
  DenseMap<bufferization::ToTensorOp, DestinationStyleOpInterface>
      outputMemrefMap;

  // Separator ops that form the boundry of vector and cube ops (i.e. FixPipeOp
  // or CopyOp)
  SmallVector<Operation *> separators;

  // Map of each operation and what it depends on
  DenseMap<Operation *, DenseSet<Operation *>> dependenceMap;

  // Lookup for yielded values
  SetVector<Value> yieldedVals;

  // Map of each operation with yielded tensor and what depends on it (reverse
  // of dependenceMap)
  DenseMap<Operation *, DenseSet<Operation *>> loopCarriedDependenceMap;

  // Since work items need to be referenced in multiple locations, we use
  // shared_ptr to avoid references being destroyed by vector reallocations
  SmallVector<std::shared_ptr<WorkItem>> worklist;

  // Non-DPS ops could potentially be cloned to various different work items
  DenseMap<Operation *, SmallVector<WorkItem *>> opToWorkItemMap;

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

  // annotation.mark ops we've already emitted a "hint overrides kernel
  // switch" warning on, to avoid duplicate diagnostics for the same tensor.
  DenseSet<Operation *> warnedOverrideMarks;
};

struct CVPipeliningPass
    : public ::mlir::impl::CVPipeliningBase<CVPipeliningPass> {
  using Base::Base;
  void runOnOperation() final;
};
} // namespace

static int getMultibufferCount(annotation::MarkOp marker) {
  auto multibufferAttr = llvm::cast_if_present<IntegerAttr>(
      marker->getAttr(MultiBufferAttr::name));
  if (!multibufferAttr)
    return -1;
  return multibufferAttr.getInt();
}

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

static bool isCrossCoreCopy(Operation *copy) {
  auto copyOp = dyn_cast<CopyOp>(copy);
  if (!copyOp)
    return false;
  Value dst = copyOp.getDst();
  memref::AllocOp alloc = traceAlloc(dst);
  if (!alloc)
    return false;
  auto memSpaceAttr =
      dyn_cast_or_null<AddressSpaceAttr>(alloc.getType().getMemorySpace());
  if (!memSpaceAttr)
    return false;

  return memSpaceAttr.getAddressSpace() == AddressSpace::L1;
}

/// True if `op` is a CV-pipelining "separator" — a store-like op that forms a
/// boundary between vector and cube workitems. Splitting the pipeline loop on
/// these ops yields the per-core workitems that CVPipelining schedules.
/// Covers `hivm.hir.fixpipe`, `hivm.hir.store`, and the cross-core variant of
/// `hivm.hir.copy`.
static bool isSeparator(Operation *op) {
  return isa<FixpipeOp, StoreOp>(op) || isCrossCoreCopy(op);
}

/// Check to see if op is what we consider a "core op" that is only available on
/// either a cube or vector core
static bool isCoreOp(Operation *op) {
  return op->hasAttr(CubeOnlyAttrName) || op->hasAttr(VecOnlyAttrName) ||
         (isa_and_nonnull<HIVMDialect>(op->getDialect()) &&
          isa<DestinationStyleOpInterface>(op));
}

/// True if `op` is a HIVM DMA op whose alloc-backed memref destination may have
/// a `bufferization.to_tensor` reader that crosses CV-pipelining workitem
/// boundaries — i.e. we need `traceMemrefSubnet` to register it in
/// `outputMemrefMap` so `migrateOps` can find the writer for any
/// cross-workitem `localOutput`.
///
/// Includes plain `hivm.hir.load`, `hivm.hir.fixpipe`, the cross-core variant
/// of `hivm.hir.copy`, and `hivm.hir.nd2nz` (the fused GM->L1 load with NZ
/// layout conversion that `CombineOptimizedConvertLayout` emits in place of a
/// LoadOp under `--enable-layout-optimization=true`).
static bool isMemrefSubnetWriter(Operation *op) {
  return isa<LoadOp, FixpipeOp, ND2NZOp>(op) || isCrossCoreCopy(op);
}

/// True if `op` is a HIVM "load-like" DMA op that pulls data from GM into
/// UB/L1. Such ops are not workitem seeds (extractAvailableOps skips them);
/// instead they are pulled into each consumer workitem during dependency
/// tracing. Under `enableLazyLoading`, they may be cloned into multiple
/// consumer workitems so each stage loads independently from GM.
///
/// Currently covers `hivm.hir.load` and `hivm.hir.nd2nz` — both read GM and
/// write to a UB/L1 alloc that the consumer mmadL1/vector op will read via
/// `bufferization.to_tensor`.
static bool isLoadLikeOp(Operation *op) {
  return isa<LoadOp, ND2NZOp>(op);
}

/// Validate if we can pipeline ops with respect to its regions.
/// Returns false if we can operate on it, otherwise true
static bool illegalRegionedOp(Operation *op) {
  if (op->getRegions().empty())
    return false;
  bool hasCube = false;
  bool hasVector = false;
  WalkResult result = op->walk([&hasCube, &hasVector](Operation *curOp) {
    if (!isa_and_nonnull<HIVMDialect>(curOp->getDialect()))
      return WalkResult::advance();
    auto core = queryCoreTypeHelper(curOp).value_or(TCoreType::CUBE_OR_VECTOR);
    if (core == TCoreType::CUBE_OR_VECTOR && isCrossCoreCopy(curOp))
      core = TCoreType::VECTOR;
    if (core == TCoreType::VECTOR) {
      if (hasCube)
        return WalkResult::interrupt();
      hasVector = true;
    } else if (core == TCoreType::CUBE) {
      if (hasVector)
        return WalkResult::interrupt();
      hasCube = true;
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    op->emitWarning("[cv-pipelining] unsupported regioned op");
    return true;
  }

  auto unit = UnitAttr::get(op->getContext());
  if (hasCube)
    op->setAttr(CubeOnlyAttrName, unit);
  else if (hasVector)
    op->setAttr(VecOnlyAttrName, unit);
  return false;
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

static Operation *getContainedParent(Operation *containing, Value inner) {
  Operation *defining = inner.getDefiningOp();
  if (defining)
    return getContainedParent(containing, defining);
  return nullptr;
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

void CVPipelineImpl::mapOpToItem(Operation *op, WorkItem *item) {
  if (item->ops.contains(op))
    return;
  if (opToWorkItemMap.contains(op))
    opToWorkItemMap[op].push_back(item);
  else
    opToWorkItemMap[op] = {item};
  item->ops.insert(op);
}

/// DFS to find all ops that are dependent on separator
LogicalResult CVPipelineImpl::populateDependencies(Operation *separator) {
  SmallVector<Operation *> dfsStack = {separator};
  DenseSet<Operation *> visited;

  while (!dfsStack.empty()) {
    Operation *op = dfsStack.pop_back_val();
    if (visited.contains(op) || !pipelineLoop->isAncestor(op))
      continue;
    visited.insert(op);
    if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
      Value init = dpsOp.getDpsInitOperand(0)->get();
      if (isa<MemRefType>(init.getType())) {
        Value src = traceValueDef(init);
        auto blkArg = dyn_cast<BlockArgument>(src);
        if (!blkArg) {
          op = src.getDefiningOp();
          if (!isa<memref::AllocOp>(op))
            return failure();
        } else if (!isa<func::FuncOp>(blkArg.getOwner()->getParentOp())) {
          return failure();
        }
      }
    }
    for (Value result : op->getResults()) {
      if (!isa<ShapedType>(result.getType()))
        continue;

      for (OpOperand &use : result.getUses()) {
        Operation *usr = use.getOwner();

        Operation *scopedUsr = getContainedParent(pipelineLoop, usr);
        if (isa<scf::YieldOp, scf::ConditionOp>(scopedUsr) ||
            scopedUsr == separator)
          continue;
        dfsStack.push_back(scopedUsr);
        if (dependenceMap.contains(scopedUsr))
          dependenceMap[scopedUsr].insert(separator);
        else
          dependenceMap[scopedUsr] = DenseSet<Operation *>({separator});
      }
    }
  }
  return success();
}

/// Populate dependencies that are carried between loop iterations (ItarArgs,
/// Yield Operands)
void CVPipelineImpl::populateLoopCarriedDependencies() {
  auto maybeYield = pipelineLoop.getYieldedValuesMutable();
  if (!maybeYield.has_value())
    return;
  for (OpOperand &yieldOperand : *maybeYield) {
    Value yieldVal = yieldOperand.get();
    // We only care about the tensor values
    if (!isa<TensorType>(yieldVal.getType()))
      continue;
    Operation *defining = yieldVal.getDefiningOp();
    if (!defining || !pipelineLoop->isAncestor(defining))
      continue;
    BlockArgument iterArg =
        pipelineLoop.getRegionIterArgs()[yieldOperand.getOperandNumber()];
    SmallVector<Operation *> dfsStack(iterArg.getUsers());
    DenseSet<Operation *> visited;
    while (!dfsStack.empty()) {
      Operation *op = getContainedParent(pipelineLoop, dfsStack.pop_back_val());
      if (visited.contains(op) || op == defining)
        continue;
      visited.insert(op);
      if (isa<DestinationStyleOpInterface>(op)) {
        if (loopCarriedDependenceMap.contains(op))
          loopCarriedDependenceMap[op].insert(defining);
        else
          loopCarriedDependenceMap[op] = {defining};
        continue;
      }
      for (Operation *usr : op->getUsers()) {
        if (isa<scf::YieldOp>(usr))
          continue;
        dfsStack.push_back(usr);
      }
    }
  }
  LLVM_DEBUG({
    for (auto &[val, set] : loopCarriedDependenceMap) {
      dbgs() << *val << " depends on:\n";
      for (auto *op : set) {
        dbgs() << "\t";
        op->dump();
      }
    }
  });
}

/// Helper to trace the alloc (if within pipelineLoop), toTensor, and
/// potentially various casts along the way
LogicalResult
CVPipelineImpl::traceMemrefSubnet(Operation *start,
                                  SmallVector<Operation *> &workingStack) {
  // When we get here, `start` should be one of three ops:
  // 1. Fixpipe
  // 2. Copy
  // 3. Load
  // All of which have the `outs` as second operand
  DestinationStyleOpInterface writer = nullptr;
  Value targetOperand = start->getOperand(1);
  if (isa<TensorType>(targetOperand.getType()))
    writer = cast<DestinationStyleOpInterface>(start);

  // Remember the original separator so we don't re-queue it onto
  // workingStack — otherwise it would be popped again and re-enter
  // traceMemrefSubnet in an infinite loop (e.g. when a Fixpipe writes
  // directly to a func-arg memref and the upward trace yields no alloc).
  Operation *separatorOp = start;
  Operation *defining = targetOperand.getDefiningOp();
  // First trace all the way up
  while (defining) {
    if (!pipelineLoop->isAncestor(defining))
      break;
    start = defining;
    if (isa<memref::AllocOp, bishengir::memref_ext::AllocWorkspaceOp>(defining))
      break;
    if (isa<memref::CastOp, memref::ReinterpretCastOp,
            memref::MemorySpaceCastOp, memref::CollapseShapeOp,
            memref::ExpandShapeOp, memref::SubViewOp, memref::ViewOp,
            bufferization::ToTensorOp, tensor::ExtractSliceOp>(defining))
      defining = defining->getOperand(0).getDefiningOp();
    else
      return defining->emitWarning(
          "[cv-pipelining] unexpected memref op in chain");
  }
  SmallVector<Operation *> userTraceStack = {start};
  bufferization::ToTensorOp toTensor = nullptr;

  while (!userTraceStack.empty()) {
    Operation *def = userTraceStack.pop_back_val();
    if (def != separatorOp)
      workingStack.push_back(def);
    for (OpOperand &use : def->getUses()) {
      Operation *usr = use.getOwner();
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(usr)) {
        // Only count dps as a writer if this use is its init operand.
        // Reads via `ins` (e.g. hivm.hir.store's src) do not constitute
        // a write to the traced memref.
        OpOperand *init = dps.getDpsInitOperand(0);
        if (!init || init != &use)
          continue;
        if (writer)
          return usr->emitWarning("[cv-pipelining] expecting only one op "
                                  "writing to a defined memref");
        writer = dps;
        continue;
      }
      if (auto tt = dyn_cast<bufferization::ToTensorOp>(usr)) {
        if (toTensor)
          return usr->emitWarning(
              "[cv-pipelining] expecting only one toTensor for a "
              "defined memref");
        toTensor = tt;
        workingStack.push_back(usr);
        continue;
      }
      if (isa<memref::CastOp, memref::ReinterpretCastOp,
              memref::MemorySpaceCastOp, memref::CollapseShapeOp,
              memref::ExpandShapeOp, memref::SubViewOp, memref::ViewOp>(usr)) {
        userTraceStack.push_back(usr);
      }
    }
  }
  if (toTensor && !writer) {
    LLVM_DEBUG(dbgs() << "toTensor: "; toTensor->dump());
    return toTensor->emitWarning(
        "[cv-pipelining] expecting toTensor to have dps op to write to it");
  }
  if (toTensor && writer)
    outputMemrefMap[toTensor] = writer;
  return success();
}

LazyLoadHint CVPipelineImpl::getLazyLoadHint(Value v) {
  auto maybeMark =
      utils::getAnnotateOpWithAttr(v, CVPipelineLazyLoadAttrName);
  if (!maybeMark)
    return LazyLoadHint::None;
  auto mark = cast<annotation::MarkOp>(*maybeMark);
  auto attr = mark->getAttrOfType<BoolAttr>(CVPipelineLazyLoadAttrName);
  if (!attr)
    return LazyLoadHint::None;
  return attr.getValue() ? LazyLoadHint::Enable : LazyLoadHint::Disable;
}

void CVPipelineImpl::warnHintOverride(Value v) {
  auto maybeMark =
      utils::getAnnotateOpWithAttr(v, CVPipelineLazyLoadAttrName);
  if (!maybeMark)
    return;
  auto mark = cast<annotation::MarkOp>(*maybeMark);
  if (!warnedOverrideMarks.insert(mark).second)
    return;

  // Prefer printing on the LoadOp for source-location clarity: it's the op
  // that would have been cloned across stages had the hint not vetoed it.
  Operation *warnTarget = mark;
  if (auto tt = dyn_cast_or_null<bufferization::ToTensorOp>(v.getDefiningOp()))
    if (auto it = outputMemrefMap.find(tt); it != outputMemrefMap.end())
      if (auto load = dyn_cast<LoadOp>(it->second.getOperation()))
        warnTarget = load;

  auto diag = warnTarget->emitWarning()
              << "[cv-pipelining] " << CVPipelineLazyLoadAttrName
              << "=false overrides kernel-level enable-lazy-loading=true; "
                 "lazy loading is disabled for this tensor";
  if (warnTarget != mark)
    diag.attachNote(mark->getLoc())
        << "see `" << CVPipelineLazyLoadAttrName << " = false` hint here";
}

void CVPipelineImpl::warnHintIgnoredForCrossCore(Value v) {
  auto maybeMark =
      utils::getAnnotateOpWithAttr(v, CVPipelineLazyLoadAttrName);
  if (!maybeMark)
    return;
  auto mark = cast<annotation::MarkOp>(*maybeMark);
  if (!warnedOverrideMarks.insert(mark).second)
    return;

  Operation *warnTarget = mark;
  if (auto tt = dyn_cast_or_null<bufferization::ToTensorOp>(v.getDefiningOp()))
    if (auto it = outputMemrefMap.find(tt); it != outputMemrefMap.end())
      if (auto load = dyn_cast<LoadOp>(it->second.getOperation()))
        warnTarget = load;

  auto diag =
      warnTarget->emitWarning()
      << "[cv-pipelining] " << CVPipelineLazyLoadAttrName
      << "=false is ignored: load result is consumed by both CUBE and "
         "VECTOR cores, lazy loading is required for correctness; "
         "disable cv-pipelining entirely to opt out";
  if (warnTarget != mark)
    diag.attachNote(mark->getLoc())
        << "see `" << CVPipelineLazyLoadAttrName << " = false` hint here";
}

void CVPipelineImpl::diagnoseLazyLoadHints() {
  // Walk pipelineLoop to collect, in IR order, the distinct values that carry
  // at least one `cv_pipeline_lazy_load` mark.  We then query all marks per
  // value via `utils::getAllAnnotateOpsWithAttr` to keep grouping bookkeeping
  // out of this function.
  llvm::SetVector<Value> markedSrcs;
  pipelineLoop.getBody()->walk([&](annotation::MarkOp mark) {
    if (mark.isAnnotatedBy(CVPipelineLazyLoadAttrName))
      markedSrcs.insert(mark.getSrc());
  });

  for (Value src : markedSrcs) {
    SmallVector<Operation *> marks =
        utils::getAllAnnotateOpsWithAttr(src, CVPipelineLazyLoadAttrName);
    if (marks.empty())
      continue;
    auto probe = cast<annotation::MarkOp>(marks.front());

    // (1) Duplicate `cv_pipeline_lazy_load` marks on the same value.
    if (marks.size() > 1) {
      auto diag = probe->emitWarning()
                  << "[cv-pipelining] tensor carries " << marks.size()
                  << " `" << CVPipelineLazyLoadAttrName
                  << "` annotation.mark ops; only the first one will be "
                     "honored";
      for (size_t i = 1; i < marks.size(); ++i)
        diag.attachNote(marks[i]->getLoc())
            << "duplicate `" << CVPipelineLazyLoadAttrName << "` mark here";
    }

    // (2) Target must be a bufferization::ToTensorOp result.
    auto tt =
        dyn_cast_or_null<bufferization::ToTensorOp>(src.getDefiningOp());
    if (!tt) {
      probe->emitWarning()
          << "[cv-pipelining] `" << CVPipelineLazyLoadAttrName
          << "` hint is ignored: marked value is not produced by "
             "`bufferization.to_tensor`";
      continue;
    }

    // (3) The to_tensor must be backed by a `hivm.hir.load`.
    auto it = outputMemrefMap.find(tt);
    if (it == outputMemrefMap.end() ||
        !isa<LoadOp>(it->second.getOperation())) {
      probe->emitWarning()
          << "[cv-pipelining] `" << CVPipelineLazyLoadAttrName
          << "` hint is ignored: tensor is not backed by `hivm.hir.load`";
      continue;
    }
  }
}

FailureOr<bool> CVPipelineImpl::isCrossCoreLoad(const Value loaded) const {
  if (!loaded)
    return false;
  bool hasCube = false;
  bool hasVec = false;
  SmallVector<Value> stack;
  DenseSet<Operation *> visited;
  stack.push_back(loaded);
  while (!stack.empty()) {
    Value v = stack.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      // Descend through view-like / passthrough ops that just rename the
      // loaded tensor; the real consumer (and thus the real core type)
      // is downstream.
      if (isa<tensor::ExtractSliceOp, tensor::CastOp, tensor::ExpandShapeOp,
              tensor::CollapseShapeOp, tensor::ReshapeOp>(user)) {
        for (Value r : user->getResults())
          stack.push_back(r);
        continue;
      }
      if (isa<annotation::MarkOp, DebugOp>(user))
        continue;
      // Per-op core hints set by `illegalRegionedOp` take precedence over
      // the trait-based query.
      std::optional<TCoreType> core;
      if (user->hasAttr(CubeOnlyAttrName))
        core = TCoreType::CUBE;
      else if (user->hasAttr(VecOnlyAttrName))
        core = TCoreType::VECTOR;
      else
        core = queryCoreTypeHelper(user);
      if (!core)
        continue;
      switch (*core) {
      case TCoreType::CUBE:
        hasCube = true;
        break;
      case TCoreType::VECTOR:
        hasVec = true;
        break;
      case TCoreType::CUBE_AND_VECTOR:
        // Op runs on both cores; that alone makes the load cross-core.
        hasCube = true;
        hasVec = true;
        break;
      case TCoreType::CUBE_OR_VECTOR:
        // Ambiguous — either core could end up running this op.  We
        // cannot safely classify the load; signal failure so the caller
        // bails out of pipelining for this loop (lazy load is a
        // legality requirement when cross-core, and we can't prove the
        // load isn't cross-core here).
        return failure();
      }
      if (hasCube && hasVec)
        return true;
    }
  }
  return hasCube && hasVec;
}

FailureOr<bool> CVPipelineImpl::shouldLazyLoadFor(Operation *op) {
  // Find the candidate to_tensor whose hint we should consult, with
  // shape-specific fallbacks when no candidate exists.
  bufferization::ToTensorOp tt;
  if (auto asTT = dyn_cast<bufferization::ToTensorOp>(op)) {
    auto it = outputMemrefMap.find(asTT);
    if (it == outputMemrefMap.end())
      return false;
    if (!isa<LoadOp>(it->second.getOperation()))
      return false;
    tt = asTT;
  } else if (auto load = dyn_cast<LoadOp>(op)) {
    for (auto &kv : outputMemrefMap) {
      if (kv.second.getOperation() == load.getOperation()) {
        tt = kv.first;
        break;
      }
    }
    if (!tt) {
      // No backing to_tensor; the load is tensor-form or otherwise opaque
      // to outputMemrefMap.  Still auto-enable lazy load when the load's
      // own tensor result has consumers on both cores.
      if (load->getNumResults() > 0 &&
          isa<TensorType>(load->getResult(0).getType())) {
        FailureOr<bool> crossCore = isCrossCoreLoad(load->getResult(0));
        if (failed(crossCore))
          return failure();
        if (*crossCore)
          return true;
      }
      return enableLazyLoading;
    }
  } else {
    return false;
  }

  FailureOr<bool> isCrossCore = isCrossCoreLoad(tt.getResult());
  if (failed(isCrossCore))
    return failure();

  switch (getLazyLoadHint(tt.getResult())) {
  case LazyLoadHint::Enable:
    return true;
  case LazyLoadHint::Disable:
    if (*isCrossCore) {
      // Legality: cross-core consumption forces lazy load; the explicit
      // `false` hint cannot veto it without producing incorrect results.
      warnHintIgnoredForCrossCore(tt.getResult());
      return true;
    }
    if (enableLazyLoading)
      warnHintOverride(tt.getResult());
    return false;
  case LazyLoadHint::None:
    return enableLazyLoading || *isCrossCore;
  }
  llvm_unreachable("invalid LazyLoadHint enumerator");
}

// Given memref value, populate users with all operations that uses any aliasing
// memrefs as `memrefVal`
static void memrefDFS(Value memrefVal, SmallVector<Operation *> &users) {
  SmallVector<Operation *> traceStack;
  DenseSet<Operation *> visited;
  Value rootVal = traceValueDef(memrefVal);
  if (!rootVal)
    return;
  traceStack.append(rootVal.user_begin(), rootVal.user_end());
  while (!traceStack.empty()) {
    Operation *op = traceStack.pop_back_val();
    if (visited.contains(op))
      continue;
    visited.insert(op);
    users.push_back(op);

    // If not memref result, dont need to trace any more
    if (op->getNumResults() == 1 &&
        !isa<MemRefType>(op->getResult(0).getType()))
      continue;
    traceStack.append(op->user_begin(), op->user_end());
  }
}

LogicalResult
CVPipelineImpl::traceOperands(Value operand, WorkItem *item,
                              SmallVector<Operation *> &workingStack) {
  if (!operand)
    return success();
  Operation *defining = getContainedParent(pipelineLoop, operand);
  if (item->ops.contains(defining))
    return success();
  if (!defining) {
    auto iterArg = dyn_cast<BlockArgument>(operand);
    if (!iterArg)
      return pipelineLoop->emitWarning(
          "[cv-pipelining] expected non-op-defined value to be block argument");
    for (Operation *usr : iterArg.getUsers()) {
      if (isa<DebugOp>(usr) && !item->ops.contains(usr) &&
          usr->getParentOp() == pipelineLoop)
        workingStack.push_back(usr);
    }
    if (iterArg.getOwner()->getParentOp() != pipelineLoop ||
        iterArg.getArgNumber() == 0)
      return success();
    // Need to pull defining op into this work item to guarentee safety,
    // should already be guarenteed by extractAvailableOps
    if (isa<TensorType>(operand.getType()))
      return success();
    Value yieldVal = pipelineLoop.getTiedLoopYieldedValue(iterArg)->get();
    defining = yieldVal.getDefiningOp();
    if (!defining || defining->getParentOp() != pipelineLoop)
      return success();
  }
  if (defining->getParentOp() != pipelineLoop)
    return success();
  // If defining is a memref, then trace everything that also uses that memref.
  if (isa<MemRefType>(operand.getType()))
    memrefDFS(operand, workingStack);
  // To tensor ops are handled as a part of the memref operand for
  // load/fixpipe/copy
  if (!item->ops.contains(defining))
    workingStack.push_back(defining);
  return success();
}

/// Trace producers of every operand of `op` *except* its DPS-init (writeback
/// destination) operands. Used by `traceDependentOps` for memref-subnet
/// writers (Load / Fixpipe / cross-core Copy / ND2NZ): the init is the `dst`
/// memref reached separately via `traceMemrefSubnet`, while every other
/// operand — ins, plus scalar params like LoadOp's init-condition and
/// padding values — is a real data dependency that must be queued onto
/// `workingStack`.
LogicalResult CVPipelineImpl::traceNonInitOperands(
    Operation *op, WorkItem *item, SmallVector<Operation *> &workingStack) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  for (Value operand : op->getOperands()) {
    if (dps && llvm::is_contained(dps.getDpsInits(), operand))
      continue;
    if (failed(traceOperands(operand, item, workingStack)))
      return failure();
  }
  return success();
}

/// Trace each op in the initial set of ops in each WorkItem, get non-HIVM ops
/// that are operands for each op
LogicalResult CVPipelineImpl::traceDependentOps(WorkItem *item) {
  SmallVector<Operation *> workingStack(item->ops.begin(), item->ops.end());

  while (!workingStack.empty()) {
    Operation *op = workingStack.pop_back_val();
    // If op is nested inside a top-level op already part of this workitem
    // (e.g. a Fixpipe inside an scf.if separator), skip it — it will be
    // cloned along with its enclosing region by migrateOps. Its outputs are
    // tracked via outputMemrefMap populated elsewhere.
    {
      Operation *top = getContainedParent(pipelineLoop, op);
      if (top != op && item->ops.contains(top))
        continue;
    }
    if (isCoreOp(op)) {
      if (opToWorkItemMap.contains(op)) {
        // If Core Op is already inserted into a different work item, then we
        // don't include it here
        if (!item->ops.contains(op)) {
          // With lazy loading (kernel-level switch, per-tensor compile
          // hint, or auto cross-core legality), allow Load ops to be
          // cloned into multiple work items so each stage loads
          // independently from GM.
          if (!isLoadLikeOp(op))
            continue;
          FailureOr<bool> shouldLazy = shouldLazyLoadFor(op);
          if (failed(shouldLazy))
            return failure();
          if (!*shouldLazy)
            continue;
        }
      } else if (!isLoadLikeOp(op)) {
        // Separators (Store/Fixpipe/cross-core Copy) that reach here via a
        // shared memref alias chain have not been assigned to any workitem
        // yet — they will be picked up in a subsequent extractAvailableOps
        // round for the other core. Skip rather than fail.
        if (isSeparator(op))
          continue;
        // Load-like ops are pulled into their consuming work items; apart from
        // that, if we get here we depend on an op that has not satisfied its
        // dependency.
        return op->emitWarning(
            "[cv-pipelining] cannot pipeline op due to dependency");
      }
    }
    // Other than core ops, we can skip them if we already inserted them into
    // this work item
    else if (item->ops.contains(op))
      continue;
    // Determine whether a to_tensor should be skipped because it has already
    // been allocated to another work item.  The default guard fires for all
    // to_tensor ops, but with lazy loading we lift the restriction for
    // to_tensor ops whose backing writer is a LoadOp: those are cloned into
    // every consuming work item independently.  Lazy loading can be enabled
    // either by the kernel-level switch or by a per-tensor compile hint
    // (annotation.mark on the tensor result).
    bool skipToTensor =
        isa<bufferization::ToTensorOp>(op) && opToWorkItemMap.contains(op);
    if (skipToTensor) {
      FailureOr<bool> shouldLazy =
          shouldLazyLoadFor(cast<bufferization::ToTensorOp>(op));
      if (failed(shouldLazy))
        return failure();
      if (*shouldLazy)
        skipToTensor = false;
    }
    if (op->getParentOp() != pipelineLoop || isa<scf::YieldOp>(op) ||
        skipToTensor)
      continue;
    LLVM_DEBUG(dbgs() << "Inserting \t"; op->dump());
    mapOpToItem(op, item);
    toBePipelined.erase(op);
    for (Operation *usr : op->getUsers())
      if (isa<annotation::MarkOp, DebugOp>(usr))
        mapOpToItem(usr, item);

    // Handle load/fixpipe/copy dealing with memref memref
    if (isMemrefSubnetWriter(op)) {
      if (failed(traceMemrefSubnet(op, workingStack)))
        return failure();
      if (failed(traceNonInitOperands(op, item, workingStack)))
        return failure();
      continue;
    }

    // Handle nested ops as well
    WalkResult walkResult = op->walk([&](Operation *nestedOp) {
      // For nested Load/Fixpipe/CrossCoreCopy (e.g. inside a lifted scf.if
      // separator), populate outputMemrefMap via traceMemrefSubnet and trace
      // their ins operand so migrateOps/expandOutputInits can resolve outputs
      // back to their nested writer.
      if (nestedOp != op && isMemrefSubnetWriter(nestedOp)) {
        if (failed(traceMemrefSubnet(nestedOp, workingStack)))
          return WalkResult::interrupt();
        if (failed(traceNonInitOperands(nestedOp, item, workingStack)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      for (Value operand : nestedOp->getOperands())
        if (failed(traceOperands(operand, item, workingStack)))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();
  }
  return success();
}

/// Fill each WorkItem with ops that will eventually go into their own jam loops
LogicalResult
CVPipelineImpl::populateWorkItem(SmallVector<Operation *> &availableOps,
                                 TCoreType core) {
  auto item = std::make_shared<WorkItem>();
  item->core = core;

#ifndef NDEBUG
  static int id = 0;
  item->id = id++;
#endif

  // ExtractOps made sure that there are only one core type of ops in
  // availableOps, no need to check here
  for (Operation *op : availableOps) {
    mapOpToItem(op, item.get());
  }
  LLVM_DEBUG({
    dbgs() << "[populateWorkItem] Initial set{\n";
    for (Operation *op : item->ops) {
      dbgs() << '\t';
      op->dump();
    }
    dbgs() << "[populateWorkItem] } // Initial set\n";
  });

  if (traceDependentOps(item.get()).failed())
    return failure();
  worklist.push_back(item);
  return success();
}

/// Find ops that have no dependencies, i.e. ops that can be executed if all
/// other previously extracted ops are done executing
LogicalResult
CVPipelineImpl::extractAvailableOps(SmallVector<Operation *> &extractedOps,
                                    TCoreType &core) {
  SetVector<Operation *> potentiallyAvailable;

  for (Operation &op : *pipelineLoop.getBody()) {
    if (opToWorkItemMap.contains(&op))
      continue;
    TCoreType maybeCore = op.hasAttr(CubeOnlyAttrName) ? TCoreType::CUBE
                          : op.hasAttr(VecOnlyAttrName)
                              ? TCoreType::VECTOR
                              : TCoreType::CUBE_OR_VECTOR;
    if (maybeCore == hivm::TCoreType::CUBE_OR_VECTOR) {
      if (!isCoreOp(&op) || isLoadLikeOp(&op))
        continue;
      maybeCore = queryCoreTypeHelper(&op).value_or(TCoreType::CUBE_OR_VECTOR);
      if (maybeCore != TCoreType::VECTOR && isCrossCoreCopy(&op))
        maybeCore = TCoreType::VECTOR;
    }

    if (maybeCore != TCoreType::VECTOR && maybeCore != TCoreType::CUBE)
      return op.emitWarning("[cv-pipelining] unexpected core type for op");
    // Only gather ops of the same core type
    if (((maybeCore == TCoreType::VECTOR || isCrossCoreCopy(&op)) &&
         core == TCoreType::CUBE) ||
        ((maybeCore == TCoreType::CUBE && core == TCoreType::VECTOR)))
      continue;
    core = maybeCore;
    if (!dependenceMap.contains(&op) || dependenceMap[&op].empty())
      potentiallyAvailable.insert(&op);
  }

  DenseSet<Operation *> deferredOps;
  for (Operation *op : potentiallyAvailable) {
    if (!loopCarriedDependenceMap.contains(op))
      continue;
    if (llvm::all_of(loopCarriedDependenceMap[op], [&](Operation *dependantOp) {
          return potentiallyAvailable.contains(dependantOp);
        }))
      continue;
    deferredOps.insert(op);
  }

  // Propagate the loop carried dependencies throughout the potentially
  // available ops
  SmallVector<Operation *> dfsStack;
  dfsStack.append(deferredOps.begin(), deferredOps.end());
  while (!dfsStack.empty()) {
    Operation *op = dfsStack.pop_back_val();
    if (deferredOps.contains(op))
      continue;
    if (potentiallyAvailable.contains(op))
      deferredOps.insert(op);
    for (Operation *usr : op->getUsers()) {
      dfsStack.push_back(usr);
    }
  }

  // Coalesce same-core DPS-init chains: when an op's sole result feeds the
  // init operand of another same-core DPS core-op `usr` that is still
  // unavailable this round, defer the producer so it lands in the same
  // WorkItem as `usr`. Without this, the producer's result becomes a cross-
  // WorkItem localOutput, and expandOutputInits cannot expand the backing
  // buffer when the init is not a tensor.empty / to_tensor (e.g. an
  // accumulator chained from another mmad's result).
  DenseSet<Operation *> chainDeferred;
  auto findBlockedCoChainConsumer = [&](Operation *op) -> Operation * {
    if (op->getNumResults() != 1)
      return nullptr;
    Value res = op->getResult(0);
    if (!res.hasOneUse())
      return nullptr;
    OpOperand &use = *res.getUses().begin();
    Operation *usr = use.getOwner();
    if (usr->getParentOp() != pipelineLoop)
      return nullptr;
    if (!isCoreOp(usr))
      return nullptr;
    auto userDps = dyn_cast<DestinationStyleOpInterface>(usr);
    if (!userDps || userDps.getNumDpsInits() != 1)
      return nullptr;
    if (userDps.getDpsInitOperand(0) != &use)
      return nullptr;
    TCoreType usrCore = usr->hasAttr(CubeOnlyAttrName)
                            ? TCoreType::CUBE
                            : usr->hasAttr(VecOnlyAttrName)
                                ? TCoreType::VECTOR
                                : queryCoreTypeHelper(usr).value_or(
                                      TCoreType::CUBE_OR_VECTOR);
    if (usrCore != core)
      return nullptr;
    if (opToWorkItemMap.contains(usr))
      return nullptr;
    // `usr` must be unavailable this round: blocked by deps (not in
    // potentiallyAvailable) or already deferred earlier in this pass.
    if (potentiallyAvailable.contains(usr) && !chainDeferred.contains(usr) &&
        !deferredOps.contains(usr))
      return nullptr;
    return usr;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : potentiallyAvailable) {
      if (deferredOps.contains(op) || chainDeferred.contains(op))
        continue;
      if (Operation *usr = findBlockedCoChainConsumer(op)) {
        LLVM_DEBUG({
          dbgs() << "[extractAvailableOps] deferring acc-chain producer:\n  ";
          op->dump();
          dbgs() << "  for blocked consumer:\n  ";
          usr->dump();
        });
        chainDeferred.insert(op);
        changed = true;
      }
    }
  }

  // Commit chain deferrals only if doing so leaves at least one op for this
  // round; otherwise we would starve the round and createWorkItems would
  // terminate prematurely with `cannot pipeline loop`.
  size_t remaining = 0;
  for (Operation *op : potentiallyAvailable)
    if (!deferredOps.contains(op) && !chainDeferred.contains(op))
      ++remaining;
  if (remaining > 0) {
    deferredOps.insert(chainDeferred.begin(), chainDeferred.end());
  } else if (!chainDeferred.empty()) {
    LLVM_DEBUG(dbgs() << "[extractAvailableOps] skipping acc-chain deferrals "
                         "to avoid empty round\n");
  }

  potentiallyAvailable.set_subtract(deferredOps);
  extractedOps.append(potentiallyAvailable.takeVector());

  return success();
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

/// Split loop based on separator ops into individual work items
LogicalResult CVPipelineImpl::createWorkItems() {
  int multibuffer = numMultibuffer > 1 ? numMultibuffer : 2;
  Block *blk = pipelineLoop.getBody();
  for (Operation &op : blk->getOperations()) {
    if (isCoreOp(&op))
      toBePipelined.insert(&op);
    if (isSeparator(&op))
      separators.push_back(&op);
    else if (auto mark = dyn_cast<annotation::MarkOp>(&op)) {
      // Compile option override
      if (numMultibuffer != -1)
        continue;
      int markMultibuffer = getMultibufferCount(mark);
      if (markMultibuffer == -1)
        continue;
      if (multibuffer < 2)
        multibuffer = markMultibuffer;
      else if (multibuffer != markMultibuffer) {
        // Conflict in multibuffer count, use smallest one
        multibuffer = std::min(multibuffer, markMultibuffer);
      }
    } else if (illegalRegionedOp(&op)) {
      // Illegal op, do nothing and return
      return failure();
    }
  } // end for op

  // Lift nested separators (e.g. a Fixpipe inside an scf.if) to their
  // top-level ancestor within pipelineLoop so downstream partitioning still
  // sees a workitem boundary across the enclosing region.
  pipelineLoop.walk([&](Operation *nested) {
    if (nested->getBlock() == blk)
      return; // already scanned above
    if (!isSeparator(nested))
      return;
    Operation *top = getContainedParent(pipelineLoop, nested);
    if (llvm::find(separators, top) == separators.end())
      separators.push_back(top);
  });

  LLVM_DEBUG({
    dbgs() << "[createWorkItems] Separators:\n";
    for (Operation *op : separators) {
      dbgs() << "\t";
      op->dump();
    }
    dbgs() << "\tmultibuffer = " << multibuffer << "\n\n";
  });
  if (multibuffer < 2)
    return failure();

  if (numMultibuffer < 1)
    numMultibuffer = multibuffer;

  // Set up dependencies
  for (Operation *separator : separators)
    if (populateDependencies(separator).failed())
      return failure();
  populateLoopCarriedDependencies();

  SmallVector<Operation *> independentOps;
  bool done = false;
  TCoreType core = hivm::TCoreType::CUBE_OR_VECTOR;
  while (!done) {
    if (extractAvailableOps(independentOps, core).failed() ||
        core == hivm::TCoreType::CUBE_OR_VECTOR)
      return failure();

    if (independentOps.empty()) {
      done = true;
      break;
    }

    if (populateWorkItem(independentOps, core).failed())
      return failure();

    for (auto &[op, dependant] : dependenceMap) {
      for (Operation *processed : independentOps)
        dependant.erase(processed);
    }
    independentOps.clear();
    // Alternate the core type being extracted.
    if (core == TCoreType::VECTOR)
      core = TCoreType::CUBE;
    else if (core == TCoreType::CUBE)
      core = TCoreType::VECTOR;
    else
      return failure();
  }
  if (!toBePipelined.empty()) {
    LLVM_DEBUG({
      for (Operation *op : toBePipelined)
        op->dump();
    });
    return pipelineLoop->emitWarning("[cv-pipelining] cannot pipeline loop due "
                                     "to loop carried dependencies");
  }
  if (worklist.size() < 2)
    return failure();

  // outputMemrefMap is fully populated now that traceMemrefSubnet has run for
  // every separator -- emit non-fatal diagnostics about misplaced or
  // duplicated `cv_pipeline_lazy_load` hints inside this loop.
  diagnoseLazyLoadHints();

  return success();
}

/// Check ops in each work item to see if they will be used by other WorkItems
/// (localOutputs) or yielded into the next iteration (yieldedOutputs)
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
      if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op)) {
        FailureOr<bool> shouldLazy = shouldLazyLoadFor(toTensor);
        if (failed(shouldLazy))
          return failure();
        if (*shouldLazy)
          continue;
      }
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
              !item->ops.contains(usrTop)) {
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
LogicalResult CVPipelineImpl::expandOutputInits(WorkItem *item) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(newLoop.getBody());
  for (auto &[output, expanded] : item->localOutputs) {
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
    expanded = builder.create<memref::AllocOp>(loc, newType, ValueRange(),
                                               alloc.getAlignmentAttr());
  }
  return success();
}

LogicalResult CVPipelineImpl::expandOutputInitsForPreload(WorkItem *item) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(pipelineLoop.getBody());
  for (auto &[output, expanded] : item->localOutputs) {
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
    item->ops.erase(alloc);
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
    if (failed(expandOutputInits(item.get())))
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
                                                      OpOperand *initOperand,
                                                      Value iv) {
  auto subview =
      dyn_cast<memref::SubViewOp>(initOperand->get().getDefiningOp());
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
  auto targetTy = cast<MemRefType>(initOperand->get().getType());
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
            updateMaskingSubview(builder, loc, expanded, initOperand, iv);
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
    if (failed(expandOutputInitsForPreload(item.get())))
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
  if (createWorkItems().failed()) {
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
