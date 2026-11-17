//===- MultiBufferLoopAdapter.cpp -------------------------------*- C++-*-===//
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
///   - scf.while : the before or after region that contains `anchorBlock`, or
///                 the builder insertion point when it is null. The latter
///                 supports GSS, which selects events from a LoopLike handle.
///                 After is the fallback when neither context is available.
/// Returns nullptr for any unsupported loop type, which lets callers fail
/// gracefully (the adapter's `create()` factory also gates on the same types).
static Block *getLoopBodyBlock(LoopLikeOpInterface loop, Block *anchorBlock,
                              Block *insertionBlock) {
  if (!loop)
    return nullptr;
  Operation *op = loop.getOperation();
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return forOp.getBody();
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    Region *region = anchorBlock      ? anchorBlock->getParent()
                     : insertionBlock ? insertionBlock->getParent()
                                      : nullptr;
    if (region) {
      while (region && region->getParentOp() != whileOp.getOperation())
        region = region->getParentOp()->getParentRegion();
      if (region == &whileOp.getBefore())
        return &whileOp.getBefore().front();
      if (region == &whileOp.getAfter())
        return &whileOp.getAfter().front();
    }
    return &whileOp.getAfter().front();
  }
  return nullptr;
}

/// Block that owns the counter used by a use site in `anchorBlock`.
/// A conditional block runs on its own cadence, so a loop-iteration index
/// reproduces neither its rotation nor its parity.
/// Such a use gets a counter of its own block.
///
/// `regionHint` := while before/after only, and may name a block outside the
/// loop (GSS asks for an id while the rewriter sits at entry).
///
/// ```
/// counterBlock null or body     -> body       (loop counter)
/// path up crosses another loop  -> body       (that loop's cadence)
/// path up reaches `loop` first  -> counterBlock
/// `loop` never reached          -> body       (outside or sibling region)
/// ```
/// Policy lives only here: coarsening it to one counter per if tree leaves
/// every caller unchanged.
/// Resolve the block whose *entries* the counter counts.
/// Pre: `counterBlock` is the block the sync pass chose, or null.
/// Post: `counterBlock` when it belongs to this loop, the loop body otherwise.
static Block *getCounterBlock(LoopLikeOpInterface loop, Block *regionHint,
                              Block *counterBlock, Block *insertionBlock) {
  Block *body = getLoopBodyBlock(loop, regionHint, insertionBlock);
  if (!body || !counterBlock || counterBlock == body)
    return body;
  for (Operation *parent = counterBlock->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent == loop.getOperation())
      return counterBlock;
    // An inner loop owns its own cadence; the outer counter would be wrong.
    if (isa<LoopLikeOpInterface>(parent))
      return body;
  }
  // Outside the loop, or in a sibling region that does not dominate the use.
  return body;
}

/// Locate the shared counter op of `block`; null when none exists yet.
/// A counter in a sibling while region or a level above/below is ignored:
/// it advances on a different cadence.
static hivm::MultiBufferCounterOp findExistingCounterOp(Block *block) {
  if (!block)
    return {};
  for (auto &op : *block) {
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

void MultiBufferLoopAdapter::ensureCounterMaterialized(OpBuilder &builder,
                                                       Block *anchorBlock,
                                                       Block *counterBlock) {
  Block *body = getCounterBlock(loop_, anchorBlock, counterBlock,
                                builder.getInsertionBlock());
  if (!body)
    llvm::report_fatal_error(
        "ensureCounterMaterialized only valid for scf.for / "
        "scf.while; adapter::create gates on this invariant.");
  if (cachedCounter_) {
    // Cached counter must dominate the call site; only reuse when it lives in
    // the resolved counter block (same nesting level / while region).
    if (cachedCounter_.getParentBlock() == body) {
      builder.setInsertionPointAfter(cachedCounter_.getDefiningOp());
      return;
    }
    cachedCounter_ = {};
  }

  Location loc = loop_->getLoc();
  Type i64Ty = builder.getI64Type();

  // ---- Reuse path: a counter op already anchors this block. ----
  if (auto existing = findExistingCounterOp(body)) {
    cachedCounter_ = existing.getResult();
    builder.setInsertionPointAfter(existing);
    return;
  }

  // ---- Fresh materialization: one HIVM op at the block head. ----
  // LowerMultiBufferCounter emits the alloca/load/increment/store later, so
  // every counter client in the block reuses this SSA value.
  // Its increment lands before the block's own terminator => the counter counts
  // that block's entries.
  builder.setInsertionPointToStart(body);
  auto counter = builder.create<hivm::MultiBufferCounterOp>(loc, i64Ty);
  cachedCounter_ = counter.getResult();
  builder.setInsertionPointAfter(counter);
}

Value MultiBufferLoopAdapter::getIterationCounter(OpBuilder &builder,
                                                  Block *anchorBlock) {
  // Unified counter-op path for both scf.for and scf.while. The concrete
  // memref-based counter is emitted later by LowerMultiBufferCounter.
  // The counter stays on the loop body on purpose: this entry feeds iteration
  // parity (DB conditions).
  // A conditional block's cadence does not reproduce that parity.
  ensureCounterMaterialized(builder, anchorBlock, /*counterBlock=*/nullptr);
  Location loc = loop_->getLoc();
  // cachedCounter_ is i64; convert to index so the surface API stays parity
  // with the previous (iv-lb)/step affine.apply value that returned
  // IndexType.
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                            cachedCounter_);
}

Value MultiBufferLoopAdapter::getModuloIndex(OpBuilder &builder,
                                             int64_t modular,
                                             Block *anchorBlock) {
  // Unified counter-op path for both scf.for and scf.while (see
  // getIterationCounter).
  // slot = counter mod modular, where counter is lowered to a monotonically
  // increasing function-scoped alloca counter.
  // `anchorBlock` owns the counter: a conditional block counts its entries,
  // not the loop's iterations.
  ensureCounterMaterialized(builder, anchorBlock,
                            /*counterBlock=*/anchorBlock);
  Location loc = loop_->getLoc();
  Type i64Ty = builder.getI64Type();
#if !defined(__LLVM_MAJOR_VERSION_22_COMPATIBLE__) &&                          \
    !defined(BSPUB_DAVINCI_BISHENGIR_A5)
  Value modVal =
      builder.create<arith::ConstantIntOp>(loc, /*value=*/modular, i64Ty);
#else
  Value modVal =
      builder.create<arith::ConstantIntOp>(loc, i64Ty, /*value=*/modular);
#endif
  Value remui = builder.create<arith::RemUIOp>(loc, cachedCounter_, modVal);
  // Cast i64 -> index for parity with the prior affine.apply API.
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), remui);
}

void MultiBufferLoopAdapter::finalizeIncrement(OpBuilder &builder,
                                               Block *anchorBlock) {
  // The concrete increment is produced by LowerMultiBufferCounter. This entry
  // point is retained as a safety/no-op hook so client passes can call it
  // without conditionals.
  ensureCounterMaterialized(builder, anchorBlock, /*counterBlock=*/nullptr);
}
