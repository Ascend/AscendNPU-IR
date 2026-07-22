//===- WorklistBuilder.h - Build worklists by core type ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides WorklistBuilder, which partitions operations into WorkItems grouped
// by core type (CUBE vs VECTOR). Used by:
//   - CV pipelining (loop mode, scope = scf::ForOp)
//   - Split mixed-if conditionals (block mode, scope = the block's parent op)
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_HIVM_UTILS_WORKLISTBUILDER_H
#define MLIR_DIALECT_HIVM_UTILS_WORKLISTBUILDER_H

#include "bishengir/Dialect/HIVM/Utils/WorkItem.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace mlir {
namespace hivm {

using bishengir::memref_ext::AllocWorkspaceOp;
using hivm::detail::queryCoreTypeHelper;
/// Tristate result of reading a `cv_pipeline_lazy_load` hint off of a value
/// via an `annotation.mark` op. Used to override (Enable / Disable) or defer
/// to (None) the kernel-level `enableLazyLoading` switch on a per-tensor basis.
enum class LazyLoadHint { None, Enable, Disable };

/// Result bundle returned by WorklistBuilder::build().
struct WorklistBuildResult {
  /// Partitioned work items, in dependence order.
  SmallVector<std::shared_ptr<WorkItem>> worklist;

  /// Reverse map: each op to the WorkItem(s) that own it. Multiple WorkItems
  /// can share an op when CUBE_OR_VECTOR helpers are pulled in as deps, or
  /// (in lazy-loading mode) when the same LoadOp is cloned across stages.
  DenseMap<Operation *, SmallVector<WorkItem *>> opToWorkItemMap;

  /// For each `bufferization::to_tensor`, the DPS op that writes its memref.
  /// Populated by tracing the memref subnet for Load/Fixpipe/Copy ops.
  DenseMap<bufferization::ToTensorOp, DestinationStyleOpInterface>
      outputMemrefMap;

  /// In loop mode, the resolved multibuffer count after annotation::MarkOp
  /// reconciliation. -1 in block mode.
  int resolvedMultibuffer;
};

/// Partitions operations into WorkItems grouped by core type (CUBE vs VECTOR).
class WorklistBuilder {
public:
  /// Loop mode: partition a for-loop's body for CV pipelining.
  /// `enableLazyLoading=true` permits the same LoadOp (and its backing
  /// to_tensor) to be pulled into multiple consuming WorkItems instead of
  /// being shared through expanded multi-buffered tensors.
  WorklistBuilder(scf::ForOp loop, int numMultibuffer,
                  bool enableLazyLoading = false);

  /// Block mode: partition a block's operations for if-else splitting.
  explicit WorklistBuilder(Block *block);

  /// Register counter-advance clones keyed by their counter alloca. When a
  /// VECTOR work item pulls in a load of the alloca, the clone is pulled in
  /// alongside so it migrates through the normal dependency flow.
  void setCounterClones(const DenseMap<Value, Operation *> &clones) {
    counterClones = clones;
  }

  /// Analyze operations and produce a partitioned worklist. May modify the
  /// IR by attaching pipeline.cubeonly / pipeline.veconly attributes to
  /// uniform-core region ops as a side effect.
  FailureOr<WorklistBuildResult> build();

  /// Returns true if the input op should be treated as lazy-loaded.  Two
  /// shapes of input are accepted (dispatched via `dyn_cast`):
  ///   * `bufferization::ToTensorOp`: the to_tensor must be registered in
  ///     `outputMemrefMap` and its backing writer must be a LoadOp.  When
  ///     not registered (or backed by a non-LoadOp), the answer is `false`.
  ///   * `LoadOp`: the matching to_tensor is found by reverse-looking up the
  ///     LoadOp in `outputMemrefMap`.  When no match is found, the answer
  ///     falls back to the kernel-level `enableLazyLoading` (or auto
  ///     cross-core if the load has a tensor result).
  ///
  /// The auto cross-core check is a LEGALITY signal: when the load's
  /// tensor result is consumed by both CUBE and VECTOR cores, lazy
  /// loading is required for correctness (otherwise the consumer core
  /// has no local copy of the data).  This signal cannot be vetoed by
  /// the per-tensor hint.
  ///
  /// Precedence (in order):
  ///   * isCrossCoreLoad signals failure (CUBE_OR_VECTOR consumer
  ///     encountered): propagate failure -- the caller should bail out
  ///     of pipelining since we cannot prove safety.
  ///   * isCrossCore = true: always true; if the hint is Disable, a
  ///     "hint is ignored" warning is emitted.  Users who really need
  ///     to opt out can disable cv-pipelining entirely.
  ///   * hint = Enable  -> true
  ///   * hint = Disable -> false; if `enableLazyLoading` is on, a
  ///     "hint overrides kernel switch" warning is emitted.
  ///   * hint = None    -> `enableLazyLoading`
  FailureOr<bool> shouldLazyLoadFor(Operation *op);

  /// Returns true if `loaded` (the tensor value produced by a load, either
  /// a bufferization.to_tensor over the load's output memref or the load's
  /// own tensor result) has consumers on both the CUBE and the VECTOR
  /// cores.  Walks transitive users, descending through view-like
  /// passthrough ops (tensor.extract_slice / cast / expand_shape /
  /// collapse_shape / reshape) and skipping annotation/debug ops.  For
  /// each remaining user, the core type is read from `pipeline.cubeonly`
  /// / `pipeline.veconly` attrs (set by `illegalRegionedOp` on regioned
  /// ops) or, failing that, from `queryCoreTypeHelper`.  CUBE_AND_VECTOR
  /// runs on both cores and sets both flags on its own.  CUBE_OR_VECTOR
  /// is ambiguous (either core could execute the op); we cannot safely
  /// classify the load and return `failure()` so the caller can bail
  /// out of pipelining for this loop.
  FailureOr<bool> isCrossCoreLoad(const Value loaded) const;

  /// Emit a one-time warning explaining that a per-tensor
  /// `cv_pipeline_lazy_load = false` hint on `v` has been IGNORED because
  /// the load's tensor result is consumed by both CUBE and VECTOR cores;
  /// lazy loading is required for correctness and cannot be vetoed by the
  /// hint.  Uses the same warning bookkeeping as `warnHintOverride` so
  /// each mark is reported at most once.
  void warnHintIgnoredForCrossCore(Value v);

private:
  void mapOpToItem(Operation &op, WorkItem &item);
  LogicalResult populateDependencies(Operation &separator);
  void populateLoopCarriedDependencies();
  LogicalResult extractAvailableOps(SmallVector<Operation *> &extractedOps,
                                    TCoreType &core);
  LogicalResult populateWorkItem(SmallVector<Operation *> &availableOps,
                                 TCoreType core);
  LogicalResult traceDependentOps(WorkItem &item);
  LogicalResult traceMemrefSubnet(Operation &start,
                                  SmallVector<Operation *> &workingStack);
  LogicalResult traceOperands(Value operand, WorkItem &item,
                              SmallVector<Operation *> &workingStack) const;
  LogicalResult traceNonInitOperands(Operation &op, WorkItem &item,
                                     SmallVector<Operation *> &workingStack)
      const;
  void computeLocalOutputs();

  /// Returns the tristate `cv_pipeline_lazy_load` hint carried by `v`'s
  /// direct `annotation.mark` users.
  static LazyLoadHint getLazyLoadHint(Value v);
  /// Emit a one-time warning when a per-tensor `cv_pipeline_lazy_load=false`
  /// hint overrides the kernel-level lazy-load switch.
  void warnHintOverride(Value v);
  /// Walk targetBlock to diagnose misuse of `cv_pipeline_lazy_load` hints
  /// (duplicates, non-to_tensor target, non-load-backed target). Non-fatal.
  void diagnoseLazyLoadHints();

  // Block to scan for ops.
  Block *targetBlock = nullptr;
  // Scope anchor for ancestor / parent checks:
  //   loop mode  → the scf::ForOp
  //   block mode → the block's parent op (e.g. scf::IfOp)
  Operation *scopeOp = nullptr;

  scf::ForOp pipelineLoop;
  bool isLoopMode = false;
  int numMultibuffer = -1;
  bool enableLazyLoading = false;

  DenseSet<Operation *> toBePipelined;
  SmallVector<Operation *> separators;
  // Counter alloca value -> vector-safe clone advancing it (set by CV pipeline).
  DenseMap<Value, Operation *> counterClones;
  DenseMap<Operation *, DenseSet<Operation *>> dependenceMap;
  DenseMap<Operation *, DenseSet<Operation *>> loopCarriedDependenceMap;
  SetVector<Value> yieldedVals;

  DenseMap<Operation *, SmallVector<WorkItem *>> opToWorkItemMap;
  DenseMap<bufferization::ToTensorOp, DestinationStyleOpInterface>
      outputMemrefMap;
  SmallVector<std::shared_ptr<WorkItem>> worklist;

  /// `annotation.mark` ops we've already emitted a "hint overrides kernel
  /// switch" warning on, to avoid duplicate diagnostics for the same tensor.
  DenseSet<Operation *> warnedOverrideMarks;
};

struct CVPipelineRegbaseImpl {
  CVPipelineRegbaseImpl(LoopLikeOpInterface loop, int multibuffer,
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

  DenseMap<Value, Value> localOuputsToRedurnRes;

  DenseSet<Operation *> toErase;

  // Checkpoint for revert in case things go wrong
  Operation *checkpoint;

  // Marked counter alloca -> vector-safe clone of the op that increments it.
  DenseMap<Value, Operation *> counterCloneMap;
};

} // namespace hivm
} // namespace mlir

#endif // MLIR_DIALECT_HIVM_UTILS_WORKLISTBUILDER_H
