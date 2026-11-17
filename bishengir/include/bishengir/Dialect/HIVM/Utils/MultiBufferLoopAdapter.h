//===- MultiBufferLoopAdapter.h ----------------------------------*- C++-*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
// MultiBufferLoopAdapter wraps an scf.for or scf.while loop and exposes a
// uniform "counter" abstraction used by the four multi-buffer passes
// (MarkMultiBuffer, PlanMemory, GraphSyncSolver, EnableMultiBuffer) to drive
// per-iteration slot rotation.
//
// Counter strategy (unified for both scf.for and scf.while; legacy
// affine.apply((iv - lb)/step) % N codegen for scf.for is retired):
//   The adapter materializes a single hivm.hir.multi_buffer_counter op per
//   block and reuses it across all clients of that block. The op is a pure
//   anchor with no attributes; the later lowering pass keys the concrete
//   counter storage off the op's block.
//
//   A later lowering pass rewrites that HIVM op to the concrete stateful form:
//   memref.alloca<1xi64>() + initial store at FunctionOpInterface entry,
//   block-head load, and block-tail increment-store. The loop signature
//   (iter_args / yields / result types) is *not* modified.
//
//   Slot rotation anchored inside a conditional block uses that block's own
//   counter: the block runs fewer times than the loop, so a loop-iteration
//   index reproduces neither its rotation nor its parity.
//   Its increment sits before the block's terminator, so it advances once per
//   entry.
//   Callers pass the block that owns the counter to getModuloIndex;
//   getIterationCounter keeps the loop body counter.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_HIVM_UTILS_MULTIBUFFER_LOOP_ADAPTER_H
#define MLIR_DIALECT_HIVM_UTILS_MULTIBUFFER_LOOP_ADAPTER_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace hivm {

class MultiBufferLoopAdapter {
public:
  /// Returns failure if `loop` is neither scf.for nor scf.while.
  static FailureOr<MultiBufferLoopAdapter> create(LoopLikeOpInterface loop);

  /// Returns the slot-select index value, computed as `counter mod modular`,
  /// where counter is produced by the hivm.hir.multi_buffer_counter op of
  /// `anchorBlock`. Materialization is idempotent on the first call.
  ///
  /// `anchorBlock` := the block whose counter drives the rotation; pass the
  /// block that consumes it, never the buffer's definition block.
  /// Post: a null `anchorBlock` keeps the loop body counter.
  ///
  /// For scf.while, `anchorBlock` also selects before vs after.
  /// When null, the builder insertion point selects the region, falling back
  /// to after if unavailable.
  Value getModuloIndex(OpBuilder &builder, int64_t modular,
                       Block *anchorBlock = nullptr);

  /// Returns the raw counter SSA value (no modulo), index-typed, from the
  /// shared hivm.hir.multi_buffer_counter op for both for and while.
  /// See getModuloIndex for the while before/after region rule.
  ///
  /// The counter always stays on the loop body, even for a nested
  /// `anchorBlock`: this value drives iteration parity, which a block cadence
  /// does not reproduce.
  Value getIterationCounter(OpBuilder &builder, Block *anchorBlock = nullptr);

  /// Idempotent. Ensures the shared counter op exists. The concrete increment
  /// store is produced by the lowering pass.
  void finalizeIncrement(OpBuilder &builder, Block *anchorBlock = nullptr);

  LoopLikeOpInterface loop() const { return loop_; }
  // Important: dyn_cast on the LoopLikeOpInterface wrapper itself does *not*
  // round-trip back to the underlying op; we must dyn_cast the
  // Operation* instead. Otherwise both methods would return null even when
  // loop_ legitimately wraps an scf.for or scf.while.
  // LoopLikeOpInterface is a thin pointer-sized wrapper, so copying it into
  // a local lets us call the non-const getOperation() without const_cast on
  // `this`.
  scf::ForOp asForOp() const {
    LoopLikeOpInterface loopCopy = loop_;
    return dyn_cast_or_null<scf::ForOp>(loopCopy.getOperation());
  }
  scf::WhileOp asWhileOp() const {
    LoopLikeOpInterface loopCopy = loop_;
    return dyn_cast_or_null<scf::WhileOp>(loopCopy.getOperation());
  }

private:
  explicit MultiBufferLoopAdapter(LoopLikeOpInterface loop) : loop_(loop) {}

  /// Find or create the counter op of the block that owns the current use.
  /// Works uniformly for both scf.for (body = forOp.getBody()) and scf.while
  /// (before or after region selected by `anchorBlock` or the builder
  /// insertion point).
  /// Post: `counterBlock == null` materializes the loop-body counter,
  ///       otherwise that block's own.
  void ensureCounterMaterialized(OpBuilder &builder, Block *anchorBlock,
                                 Block *counterBlock);

  LoopLikeOpInterface loop_;
  Value cachedCounter_ = {};
};

} // namespace hivm
} // namespace mlir

#endif // MLIR_DIALECT_HIVM_UTILS_MULTIBUFFER_LOOP_ADAPTER_H
