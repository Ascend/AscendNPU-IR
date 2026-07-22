//===- WorkItem.h - Shared WorkItem for HIVM passes -------------*- C++ -*-===//
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
#ifndef BISHENGIR_DIALECT_HIVM_UTILS_WORKITEM_H
#define BISHENGIR_DIALECT_HIVM_UTILS_WORKITEM_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

namespace mlir {
namespace hivm {

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

  // After unrolling the parent for loop, the upper bound for "reroll"ed loops
  // are computed and inserted here. Created in "unrollOuterLoop"
  Value upperBound;

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

  // The containing for loop
  scf::ForOp parentFor;

  // Number of multibuffer
  unsigned multibuffer;

  // Guesstimate of runtime cost of this work item
  float cost;

#ifndef NDEBUG
  int id = -1;
#endif
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_UTILS_WORKITEM_H
