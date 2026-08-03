//===- WorkItem.h - Shared WorkItem for HIVM passes -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// WorkItem groups operations partitioned by core type (CUBE vs VECTOR) for
// HIVM passes. Used by:
//   - CV pipelining (loop mode)
//   - Split mixed-if conditionals (block mode)
//
// Common fields are core to the partitioning algorithm. The trailing fields
// (forOp / irMap / reconstructedIV / scopeOp) are CV-pipelining codegen state
// populated during the unroll-and-jam pass; block-mode consumers (split-if)
// leave them default-constructed.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_HIVM_UTILS_WORKITEM_H
#define MLIR_DIALECT_HIVM_UTILS_WORKITEM_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace hivm {

struct WorkItem {
  /// Values crossing work-item boundaries (original, expanded). The expanded
  /// form is written by CV pipelining's expandOutputInits; left null in block
  /// mode.
  SmallVector<std::pair<Value, Value>> localOutputs;

  /// Operations assigned to this work item. SetVector preserves insertion
  /// order for deterministic cloning (split-if relies on this).
  SetVector<Operation *> ops;

  /// Values yielded across the parent for-loop's iteration boundary, paired
  /// with their iter-arg position.
  SmallVector<std::pair<Value, unsigned>> yieldedOutputs;

  /// Store-like ops writing CV pipeline intermediates through
  /// memref_ext.alloc_workspace. Preload-mode CV pipelining expands and slices
  /// these separately from tensor localOutputs.
  SmallVector<Operation *> workspaceOutputs;

  /// CUBE or VECTOR. CUBE_OR_VECTOR may appear for the block-mode "remainder"
  /// work item that absorbs flexibly-typed ops.
  TCoreType core;

  // ===========================================================================
  // CV-pipelining codegen state (loop mode only). Block-mode consumers leave
  // these default-constructed.
  // ===========================================================================

  /// The for-op corresponding to the multibuffering. Constructed in
  /// CVPipelineImpl::createNewLoops.
  scf::ForOp forOp;

  /// IR mapping used while cloning ops into the per-WorkItem for-op.
  IRMapping irMap;

  /// Reconstructed original induction variable.
  Value reconstructedIV;

  /// ScopeOp wrapper for skew-mode preload pipelining.
  scope::ScopeOp scopeOp;

  /// Skew-mode lazy-load counter privatization record. When a load-like op is
  /// cloned into several work items, the GM address counter chain comes along,
  /// so more than one scope would read and advance the same loop iter_arg while
  /// only one can own the yield slot. Each extra reader gets a private iter_arg
  /// described here.
  struct PrivateCounter {
    /// The (shared) advance value, i.e. the original `arith.addi` result that
    /// `markOutputs` registered in `yieldedOutputs`.
    Value advanceVal;
    /// Iter-arg index of the shared counter in the pipeline loop.
    unsigned sharedArgIdx;
    /// The shared iter_arg this item must stop reading.
    BlockArgument sharedIterArg;
    /// The private iter_arg this item reads instead.
    BlockArgument privateIterArg;
    /// scf.yield operand index this item writes its own advance value to.
    unsigned privateYieldIdx;
  };
  SmallVector<PrivateCounter> privateCounters;

#ifndef NDEBUG
  int id = -1;
#endif
};

} // namespace hivm
} // namespace mlir

#endif // MLIR_DIALECT_HIVM_UTILS_WORKITEM_H
