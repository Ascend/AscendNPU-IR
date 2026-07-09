//===- MultiBufferLoopAdapter.h ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
//   The adapter materializes a single hivm.hir.multi_buffer_counter op in the
//   loop body and reuses it across all clients. The op is a pure anchor with no
//   attributes; the later lowering pass keys the concrete counter storage off
//   the op's owning loop.
//
//   A later lowering pass rewrites that HIVM op to the concrete stateful form:
//   memref.alloca<1xi64>() + initial store at FunctionOpInterface entry,
//   body-head load, and body-tail increment-store. The loop signature
//   (iter_args / yields / result types) is *not* modified.
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
  /// where counter is produced by the shared hivm.hir.multi_buffer_counter op
  /// for both scf.for and scf.while. Counter-op materialization is done
  /// idempotently on the first call.
  Value getModuloIndex(OpBuilder &builder, int64_t modular);

  /// Returns the raw counter SSA value (no modulo), index-typed, from the
  /// shared hivm.hir.multi_buffer_counter op for both for and while.
  Value getIterationCounter(OpBuilder &builder);

  /// Idempotent. Ensures the shared counter op exists. The concrete increment
  /// store is produced by the lowering pass.
  void finalizeIncrement(OpBuilder &builder);

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

  /// Find or create the shared counter op in the loop body. Works uniformly for
  /// both scf.for (body = forOp.getBody()) and scf.while (body =
  /// whileOp.getAfter().front()).
  void ensureCounterMaterialized(OpBuilder &builder);

  LoopLikeOpInterface loop_;
  Value cachedCounter_ = {};
};

} // namespace hivm
} // namespace mlir

#endif // MLIR_DIALECT_HIVM_UTILS_MULTIBUFFER_LOOP_ADAPTER_H
