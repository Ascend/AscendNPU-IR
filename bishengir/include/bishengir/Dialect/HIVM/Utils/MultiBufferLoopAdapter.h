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
// MultiBufferLoopAdapter wraps an scf.for or scf.while loop and exposes one
// uniform counter abstraction to MarkMultiBuffer, PlanMemory, GraphSyncSolver
// and EnableMultiBuffer.
// The counter decides what the slot rotates on:
//   loop counter      -> one step per loop iteration
//   conditional block -> one step per entry of that block
//
// One hivm.hir.multi_buffer_counter op per block, reused by every client of
// that block. The op is a pure anchor with no attributes;
// LowerMultiBufferCounter keys the storage off the op's block.
//
// LowerMultiBufferCounter rewrites the op to the stateful form:
// memref.alloca<1xi64>() + initial store at FunctionOpInterface entry,
// block-head load, block-tail increment-store.
// The loop signature (iter_args, yields, result types) is left alone.
//
// A conditional block runs fewer times than the loop:
// a loop-iteration index reproduces neither its rotation nor its parity.
// So that block owns its counter, incremented before its own terminator
// => one step per entry.
// getModuloIndex takes the owning block; getIterationCounter keeps the loop
// body counter.
//
// Legacy affine.apply((iv - lb)/step) % N codegen for scf.for is retired.
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

  /// Slot-select index: `counter mod modular`.
  /// counter := the hivm.hir.multi_buffer_counter op of `anchorBlock`.
  /// The first call materializes it, later calls reuse it.
  ///
  /// `anchorBlock` := the block whose counter drives the rotation.
  /// Pass the consuming block, never the buffer's definition block.
  ///
  /// scf.for: the counter sits in the loop body, whatever `anchorBlock` names.
  /// scf.while: it sits in the region `anchorBlock` names.
  /// null `anchorBlock` -> the builder insertion point's region, the after
  /// region when that names none.
  Value getModuloIndex(OpBuilder &builder, int64_t modular,
                       Block *anchorBlock = nullptr);

  /// Raw counter SSA value, index-typed, no modulo.
  /// Same hivm.hir.multi_buffer_counter op as getModuloIndex, for for and
  /// while.
  /// while region rule: see getModuloIndex.
  ///
  /// The counter always stays on the loop body, even for a nested
  /// `anchorBlock`: this value drives iteration parity, which a block counter
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
  /// scf.for -> body; scf.while -> the region `anchorBlock` names, else the
  /// builder insertion point.
  /// null `counterBlock` -> the loop body counter.
  /// Otherwise -> the counter of `counterBlock`.
  void ensureCounterMaterialized(OpBuilder &builder, Block *anchorBlock,
                                 Block *counterBlock);

  LoopLikeOpInterface loop_;
  Value cachedCounter_ = {};
};

} // namespace hivm
} // namespace mlir

#endif // MLIR_DIALECT_HIVM_UTILS_MULTIBUFFER_LOOP_ADAPTER_H
