//===- MultiBufferLoopAdapter.cpp -------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/MultiBufferLoopAdapter.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Body block of a multi-buffer-supported LoopLike op (for / while).
///   - scf.for   : the single body block (`forOp.getBody()`).
///   - scf.while : the after-region single block (where the actual iteration
///                 body lives; the before-region only carries the predicate
///                 `scf.condition`).
/// Returns nullptr for any unsupported loop type, which lets callers fail
/// gracefully (the adapter's `create()` factory also gates on the same types).
Block *getLoopBodyBlock(LoopLikeOpInterface loop) {
  if (!loop)
    return nullptr;
  Operation *op = loop.getOperation();
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return forOp.getBody();
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return &whileOp.getAfter().front();
  return nullptr;
}

/// Locate the shared counter op for `loop`. Returns nullptr if none exists yet.
hivm::MultiBufferCounterOp findExistingCounterOp(LoopLikeOpInterface loop) {
  Block *body = getLoopBodyBlock(loop);
  if (!body)
    return {};
  for (auto &op : *body) {
    auto counter = dyn_cast<hivm::MultiBufferCounterOp>(&op);
    if (counter)
      return counter;
  }
  return {};
}

} // namespace

//===----------------------------------------------------------------------===//
// MultiBufferLoopAdapter
//===----------------------------------------------------------------------===//

FailureOr<MultiBufferLoopAdapter>
MultiBufferLoopAdapter::create(LoopLikeOpInterface loop) {
  if (!loop)
    return failure();
  if (!isa<scf::ForOp, scf::WhileOp>(loop.getOperation()))
    return failure();
  return MultiBufferLoopAdapter(loop);
}

void MultiBufferLoopAdapter::ensureCounterMaterialized(OpBuilder &builder) {
  Block *body = getLoopBodyBlock(loop_);
  if (!body)
    llvm_unreachable("ensureCounterMaterialized only valid for scf.for / "
                     "scf.while; adapter::create gates on this invariant.");
  if (cachedCounter_) {
    builder.setInsertionPointAfter(cachedCounter_.getDefiningOp());
    return;
  }

  Location loc = loop_->getLoc();
  Type i64Ty = builder.getI64Type();

  // ---- Reuse path: a counter op already anchors this loop. ----
  if (auto existing = findExistingCounterOp(loop_)) {
    cachedCounter_ = existing.getResult();
    builder.setInsertionPointAfter(existing);
    return;
  }

  // ---- Fresh materialization: create a single HIVM op at the body head. The
  // concrete alloca/load/increment/store sequence is emitted later by
  // LowerMultiBufferCounter so all counter clients can reuse this SSA value.
  builder.setInsertionPointToStart(body);
  auto counter = builder.create<hivm::MultiBufferCounterOp>(loc, i64Ty);
  cachedCounter_ = counter.getResult();
  builder.setInsertionPointAfter(counter);
}

Value MultiBufferLoopAdapter::getIterationCounter(OpBuilder &builder) {
  // Unified counter-op path for both scf.for and scf.while. The concrete
  // memref-based counter is emitted later by LowerMultiBufferCounter.
  ensureCounterMaterialized(builder);
  Location loc = loop_->getLoc();
  // cachedCounter_ is i64; convert to index so the surface API stays parity
  // with the previous (iv-lb)/step affine.apply value that returned
  // IndexType.
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                            cachedCounter_);
}

Value MultiBufferLoopAdapter::getModuloIndex(OpBuilder &builder,
                                             int64_t modular) {
  // Unified counter-op path for both scf.for and scf.while (see note in
  // getIterationCounter). slot = counter mod modular, where counter is
  // lowered to a monotonically increasing function-scoped alloca counter.
  ensureCounterMaterialized(builder);
  Location loc = loop_->getLoc();
  Type i64Ty = builder.getI64Type();
  Value modVal =
      builder.create<arith::ConstantIntOp>(loc, i64Ty, /*value=*/modular);
  Value remui = builder.create<arith::RemUIOp>(loc, cachedCounter_, modVal);
  // Cast i64 -> index for parity with the prior affine.apply API.
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                            remui);
}

void MultiBufferLoopAdapter::finalizeIncrement(OpBuilder &builder) {
  // The concrete increment is produced by LowerMultiBufferCounter. This entry
  // point is retained as a safety/no-op hook so client passes can call it
  // without conditionals.
  ensureCounterMaterialized(builder);
}
