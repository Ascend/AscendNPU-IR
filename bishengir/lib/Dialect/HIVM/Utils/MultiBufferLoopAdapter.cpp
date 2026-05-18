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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

/// Allocate a fresh per-funcOp loop ID (i64) for `whileOp`. The id is stored
/// on the op as kMultiBufferLoopIdAttr; subsequent calls return the existing
/// id. Uniqueness is ensured by tracking the max id already in use within the
/// parent FunctionOpInterface.
IntegerAttr getOrAssignWhileLoopId(scf::WhileOp whileOp) {
  if (auto existing =
          whileOp->getAttrOfType<IntegerAttr>(kMultiBufferLoopIdAttr))
    return existing;

  auto funcOp = whileOp->getParentOfType<FunctionOpInterface>();
  if (!funcOp)
    llvm_unreachable("scf.while must live inside a FunctionOpInterface");

  int64_t maxId = -1;
  funcOp.walk([&](scf::WhileOp w) {
    if (auto a = w->getAttrOfType<IntegerAttr>(kMultiBufferLoopIdAttr))
      maxId = std::max(maxId, a.getInt());
  });

  auto attr = IntegerAttr::get(
      IntegerType::get(whileOp.getContext(), 64), maxId + 1);
  whileOp->setAttr(kMultiBufferLoopIdAttr, attr);
  return attr;
}

/// Locate the counter alloca created earlier for `whileOp` (matched by
/// kMultiBufferCounterAttr == kMultiBufferLoopIdAttr). Returns nullptr if
/// none exists yet.
memref::AllocaOp findExistingCounterAlloca(scf::WhileOp whileOp,
                                           IntegerAttr loopId) {
  auto funcOp = whileOp->getParentOfType<FunctionOpInterface>();
  if (!funcOp || funcOp.getFunctionBody().empty())
    return {};

  Block &entry = funcOp.getFunctionBody().front();
  for (auto &op : entry) {
    auto alloca = dyn_cast<memref::AllocaOp>(&op);
    if (!alloca)
      continue;
    auto a = alloca->getAttrOfType<IntegerAttr>(kMultiBufferCounterAttr);
    if (a && a == loopId)
      return alloca;
  }
  return {};
}

/// Locate a previously-cached body-head load of `alloca` inside the after
/// region of `whileOp`. Returns nullptr if none exists.
memref::LoadOp findExistingCounterLoad(scf::WhileOp whileOp,
                                       memref::AllocaOp alloca) {
  Block &after = whileOp.getAfter().front();
  for (auto &op : after) {
    auto load = dyn_cast<memref::LoadOp>(&op);
    if (!load)
      continue;
    if (load.getMemref() == alloca.getResult())
      return load;
  }
  return {};
}

/// Returns true iff there is already a memref.store back to `alloca` inside
/// the after region of `whileOp`. Used by finalizeIncrement to remain
/// idempotent across multiple adapter instances.
bool hasExistingCounterStore(scf::WhileOp whileOp, memref::AllocaOp alloca) {
  Block &after = whileOp.getAfter().front();
  for (auto &op : after) {
    auto store = dyn_cast<memref::StoreOp>(&op);
    if (!store)
      continue;
    if (store.getMemref() == alloca.getResult())
      return true;
  }
  return false;
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
  auto whileOp = asWhileOp();
  if (!whileOp)
    llvm_unreachable("ensureCounterMaterialized only valid for scf.while");
  if (counterAlloca_ && cachedLoad_)
    return;

  IntegerAttr loopId = getOrAssignWhileLoopId(whileOp);

  auto funcOp = whileOp->getParentOfType<FunctionOpInterface>();
  if (!funcOp)
    llvm_unreachable("scf.while must live inside a FunctionOpInterface");

  Location loc = whileOp.getLoc();
  Type i64Ty = builder.getI64Type();
  auto memTy = MemRefType::get(/*shape=*/{1}, i64Ty);

  // ---- Reuse path: alloca already present, must mean load+store are too. ----
  if (auto existing = findExistingCounterAlloca(whileOp, loopId)) {
    counterAlloca_ = existing;
    auto load = findExistingCounterLoad(whileOp, existing);
    if (!load)
      llvm_unreachable("counter alloca exists but body-head load is missing; "
                       "IR was rewritten unexpectedly between passes.");
    cachedLoad_ = load.getResult();
    builder.setInsertionPointAfter(load);
    return;
  }

  // ---- Fresh materialization: build alloca + init + body-head load +
  // body-tail increment+store atomically. Multi-buffer passes that later
  // construct a fresh adapter for the same loop will hit the reuse path
  // above; the alloca, load, and store are always created together so the
  // invariant "alloca present iff increment-store present" holds. ----

  // Phase 1: alloca + initial store at the top of the function body.
  {
    OpBuilder::InsertionGuard g(builder);
    Block &entry = funcOp.getFunctionBody().front();
    builder.setInsertionPointToStart(&entry);
    counterAlloca_ = builder.create<memref::AllocaOp>(loc, memTy);
    counterAlloca_->setAttr(kMultiBufferCounterAttr, loopId);

    Value zero =
        builder.create<arith::ConstantIntOp>(loc, i64Ty, /*value=*/0);
    Value zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
    builder.create<memref::StoreOp>(loc, zero, counterAlloca_.getResult(),
                                    ValueRange{zeroIdx});
  }

  // Phase 2: body-head load.
  {
    OpBuilder::InsertionGuard g(builder);
    Block &after = whileOp.getAfter().front();
    builder.setInsertionPointToStart(&after);
    Value zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto load = builder.create<memref::LoadOp>(loc, counterAlloca_.getResult(),
                                               ValueRange{zeroIdx});
    cachedLoad_ = load.getResult();
  }

  // Phase 3: body-tail increment + store-back, inserted before the
  // scf.yield terminator. Created together with the load so the invariant
  // "alloca present <=> store-back present" holds for all later passes.
  {
    OpBuilder::InsertionGuard g(builder);
    Block &after = whileOp.getAfter().front();
    Operation *terminator = after.getTerminator();
    if (!terminator)
      llvm_unreachable("scf.while after region must have a terminator");
    builder.setInsertionPoint(terminator);
    Value one =
        builder.create<arith::ConstantIntOp>(loc, i64Ty, /*value=*/1);
    Value next = builder.create<arith::AddIOp>(loc, cachedLoad_, one);
    Value zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
    builder.create<memref::StoreOp>(loc, next, counterAlloca_.getResult(),
                                    ValueRange{zeroIdx});
  }

  // Position builder right after the load so that subsequent caller ops
  // (e.g. remui, arith.select cascade) end up in body-head order with the
  // load dominating them.
  builder.setInsertionPointAfter(cachedLoad_.getDefiningOp());
}

Value MultiBufferLoopAdapter::getIterationCounter(OpBuilder &builder) {
  if (auto forOp = asForOp()) {
    // Reuse legacy createNestedIndexModular with modular=-1 (no mod, raw
    // flattened iv expression). Returns IndexType.
    return mlir::hivm::createNestedIndexModular(
        builder, cast<LoopLikeOpInterface>(forOp.getOperation()),
        /*modular=*/-1);
  }
  ensureCounterMaterialized(builder);
  Location loc = loop_->getLoc();
  // cachedLoad_ is i64; convert to index for parity with the for-path API.
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                            cachedLoad_);
}

Value MultiBufferLoopAdapter::getModuloIndex(OpBuilder &builder,
                                             int64_t modular) {
  if (auto forOp = asForOp()) {
    return mlir::hivm::createNestedIndexModular(
        builder, cast<LoopLikeOpInterface>(forOp.getOperation()),
        static_cast<int>(modular));
  }
  ensureCounterMaterialized(builder);
  Location loc = loop_->getLoc();
  Type i64Ty = builder.getI64Type();
  Value modVal =
      builder.create<arith::ConstantIntOp>(loc, i64Ty, /*value=*/modular);
  Value remui = builder.create<arith::RemUIOp>(loc, cachedLoad_, modVal);
  // Cast i64 -> index for parity with the for-path API (affine.apply yields
  // IndexType). Downstream callers wrap further (e.g. arith.index_cast to
  // i1 or i64) and IndexType is what they expect.
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                            remui);
}

void MultiBufferLoopAdapter::finalizeIncrement(OpBuilder &builder) {
  // ensureCounterMaterialized atomically creates alloca + load + increment-
  // store, so the increment is already in place by the time any caller
  // observes the counter. This entry point is retained as a safety/no-op
  // hook so client passes can call it without conditionals; it also catches
  // the case where someone bypasses getModuloIndex / getIterationCounter
  // and we still need to ensure the increment exists.
  auto whileOp = asWhileOp();
  if (!whileOp)
    return;
  ensureCounterMaterialized(builder);
  // NB: extract the helper call into a local so we never violate G.AST.03
  // ("assert calls a function with potentially desired side effects"). The
  // call is the actual check; the llvm_unreachable below is the unconditional
  // bail-out kept across release builds.
  bool incrementStorePresent =
      hasExistingCounterStore(whileOp, counterAlloca_);
  if (!incrementStorePresent)
    llvm_unreachable("counter increment store missing after materialization");
}
