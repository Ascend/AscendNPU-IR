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
  /// (in lazy-loading mode) when the same load-like op is cloned across
  /// stages.
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
  /// `enableLazyLoading=true` permits the same LoadOp or ND2NZOp (and its
  /// backing to_tensor) to be pulled into multiple consuming WorkItems instead
  /// of being shared through expanded multi-buffered tensors.
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
  ///     `outputMemrefMap` and its backing writer must be load-like (LoadOp or
  ///     ND2NZOp). When not registered (or backed by another op), the answer
  ///     is `false`.
  ///   * `LoadOp` / `ND2NZOp`: the matching to_tensor is found by reverse
  ///     lookup in `outputMemrefMap`. When no match is found, the answer falls
  ///     back to the kernel-level `enableLazyLoading` (or auto cross-core if
  ///     the load-like op has a tensor result).
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
  /// (duplicates, non-to_tensor target, non-load-like-backed target).
  /// Non-fatal.
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

} // namespace hivm
} // namespace mlir

#endif // MLIR_DIALECT_HIVM_UTILS_WORKLISTBUILDER_H
