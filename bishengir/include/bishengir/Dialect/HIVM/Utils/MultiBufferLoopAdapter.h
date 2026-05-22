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
//   An i64 counter is materialized as a memref.alloca<1xi64>() at the top
//   of the parent FunctionOpInterface. The alloca carries
//   kMultiBufferCounterAttr whose value matches a kMultiBufferLoopIdAttr
//   placed on the owning loop op itself. Counter discovery is purely
//   IR-driven: a second pass that constructs a fresh adapter for the same
//   loop will find and reuse the existing alloca/load/store instead of
//   creating duplicates. The loop signature (iter_args / yields / result
//   types) is *not* modified.
//
//   For scf.for the body block is `forOp.getBody()`; for scf.while it is
//   `whileOp.getAfter().front()`. Body-head load + body-tail
//   increment-store are inserted into this body block, and the alloca
//   sits at funcOp entry so the counter persists across any outer-loop
//   re-entries (equivalent to the legacy affine-flattening (iv-lb)/step
//   semantics but expressed as stateful IR instead of a pure expression).
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

  /// Returns the slot-select index value, computed as `counter mod modular`,
  /// where counter is the body-head load of the i64 alloca counter for both
  /// scf.for and scf.while. Counter materialization (alloca + init +
  /// body-head load + body-tail increment-store) is done idempotently on
  /// the first call.
  Value getModuloIndex(OpBuilder &builder, int64_t modular);

  /// Returns the raw counter SSA value (no modulo), index-typed, as the
  /// body-head load of the i64 alloca counter for both for and while.
  Value getIterationCounter(OpBuilder &builder);

  /// Idempotent. Ensures the body-tail "+1; store back to alloca" pair
  /// exists. Safe to call from any pass; primarily a backstop in case
  /// some client bypassed getModuloIndex / getIterationCounter.
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
  /// load + body-tail increment-store are all materialized at once and
  /// tagged so subsequent calls can reuse them). Works uniformly for both
  /// scf.for (body = forOp.getBody()) and scf.while (body =
  /// whileOp.getAfter().front()).
  void ensureCounterMaterialized(OpBuilder &builder);

  LoopLikeOpInterface loop_;
  memref::AllocaOp counterAlloca_ = {};
  Value cachedLoad_ = {};
};

} // namespace hivm
} // namespace mlir

#endif // MLIR_DIALECT_HIVM_UTILS_MULTIBUFFER_LOOP_ADAPTER_H
