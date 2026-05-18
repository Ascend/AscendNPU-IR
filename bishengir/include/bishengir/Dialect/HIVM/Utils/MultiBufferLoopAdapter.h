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
// The two backing strategies are:
//   - scf.for : reuse the legacy (iv - lb) / step path implemented in
//               EnableMultiBuffer.cpp's createNestedIndexModular helper.
//               The scf.for body is guaranteed to dominate every use, so no
//               new state is materialized.
//
//   - scf.while: an i64 counter is materialized as a memref.alloca() at the
//                top of the parent FunctionOpInterface. The alloca carries the
//                kMultiBufferCounterAttr whose value matches a
//                kMultiBufferLoopIdAttr placed on the scf.while op itself.
//                This makes counter discovery purely IR-driven: a second pass
//                that constructs a fresh adapter for the same loop will find
//                and reuse the existing alloca instead of creating a duplicate.
//                The scf.while signature (init / condition / yield /
//                iter_args / result types) is *not* modified.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_HIVM_UTILS_MULTIBUFFER_LOOP_ADAPTER_H
#define MLIR_DIALECT_HIVM_UTILS_MULTIBUFFER_LOOP_ADAPTER_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

  /// Returns the slot-select index value, computed as `counter mod modular`.
  /// For scf.for this defers to createNestedIndexModular which flattens
  /// nested for loops. For scf.while this performs `arith.remui (load,
  /// modular)` at the builder's current insertion point. Counter materialization
  /// for scf.while (alloca + init + body-head load + body-tail increment) is
  /// done idempotently on first call.
  Value getModuloIndex(OpBuilder &builder, int64_t modular);

  /// Returns the raw counter SSA value (no modulo). For scf.for this is
  /// `(iv - lb) / step` flattened across nests. For scf.while this is the
  /// cached body-head load value.
  Value getIterationCounter(OpBuilder &builder);

  /// Idempotent. For scf.while ensures the body-tail "+1; store back to
  /// alloca" pair exists. No-op for scf.for. Safe to call from any pass.
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

  /// Find existing counter alloca via attribute lookup, or create one at the
  /// top of the parent FunctionOpInterface (alloca + init store + body-head
  /// load are all materialized at once and tagged so subsequent calls can
  /// reuse them).
  void ensureCounterMaterialized(OpBuilder &builder);

  LoopLikeOpInterface loop_;
  memref::AllocaOp counterAlloca_ = {};
  Value cachedLoad_ = {};
};

} // namespace hivm
} // namespace mlir

#endif // MLIR_DIALECT_HIVM_UTILS_MULTIBUFFER_LOOP_ADAPTER_H
