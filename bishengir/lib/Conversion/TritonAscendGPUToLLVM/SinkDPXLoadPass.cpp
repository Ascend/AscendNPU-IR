//===- SinkDPXLoadPass.cpp - Sink ascend_dpx.load to reduce reg pressure --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// After ConvertTritonAscendGPUToLLVM, tensor operations are fully unrolled into
// per-thread scalar operations. The lowering packs all per-element results into
// LLVM structs via insertvalue chains and unpacks them via extractvalue.
// This creates an artificial sequential dependency through the struct that
// prevents interleaving of independent load-compute-store chains.
//
// This pass has two phases:
//   1. SROA-like struct bypass: replace extractvalue(insertvalue_chain, idx)
//      with the value that was inserted at that index, eliminating the struct
//      as a dependency bottleneck.
//   2. Bottom-up scheduling: reorder operations so each store's dependency
//      tree (load, compute, store) is emitted contiguously.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TritonAscendGPUToLLVM/Passes.h"
#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::ascend {

#define GEN_PASS_DEF_SINKDPXLOAD
#include "bishengir/Conversion/TritonAscendGPUToLLVM/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Phase 1: SROA - Scalar Replacement of Aggregates
//===----------------------------------------------------------------------===//

/// Given the final value of an insertvalue chain, trace back and build a map
/// from field index to the value inserted at that index.
/// For example, for the chain:
///   %s0 = llvm.undef : !llvm.struct<(i64, i64)>
///   %s1 = llvm.insertvalue %a, %s0[0]
///   %s2 = llvm.insertvalue %b, %s1[1]
/// Returns {0: %a, 1: %b}.
static void traceInsertValueChain(
    Value structVal,
    llvm::DenseMap<int64_t, Value> &fieldMap) {
  while (auto insertOp =
             structVal.getDefiningOp<LLVM::InsertValueOp>()) {
    auto position = insertOp.getPosition();
    // Only handle single-level struct indexing (no nested structs).
    if (position.size() != 1)
      return;
    int64_t idx = position[0];
    // Only record the first (outermost) insertion for each index,
    // since later insertions overwrite earlier ones.
    if (!fieldMap.contains(idx))
      fieldMap[idx] = insertOp.getValue();
    structVal = insertOp.getContainer();
  }
}

/// Perform SROA on extractvalue operations: if an extractvalue reads from
/// an insertvalue chain, replace it with the value that was directly inserted
/// at that index.
static bool sroaStructs(Block &block) {
  bool changed = false;
  llvm::SmallVector<LLVM::ExtractValueOp> extracts;

  for (auto &op : block) {
    if (auto extractOp = dyn_cast<LLVM::ExtractValueOp>(&op))
      extracts.push_back(extractOp);
  }

  for (auto extractOp : extracts) {
    auto position = extractOp.getPosition();
    if (position.size() != 1)
      continue;
    int64_t idx = position[0];

    llvm::DenseMap<int64_t, Value> fieldMap;
    traceInsertValueChain(extractOp.getContainer(), fieldMap);

    auto it = fieldMap.find(idx);
    if (it == fieldMap.end())
      continue;

    extractOp.getResult().replaceAllUsesWith(it->second);
    changed = true;
  }

  return changed;
}

/// Remove dead operations (ops with no users and no side effects).
static void removeDeadOps(Block &block) {
  // Walk in reverse to handle chains of dead ops.
  llvm::SmallVector<Operation *> toDelete;
  for (auto it = block.rbegin(); it != block.rend(); ++it) {
    Operation *op = &*it;
    if (op->hasTrait<OpTrait::IsTerminator>())
      continue;
    if (op->use_empty() && isMemoryEffectFree(op))
      toDelete.push_back(op);
  }
  for (auto *op : toDelete)
    op->erase();
}

//===----------------------------------------------------------------------===//
// Phase 2: Bottom-up store-rooted scheduling
//===----------------------------------------------------------------------===//

/// Check whether any operation in the range [begin, end) produces or consumes
/// a vector<2xf16> value.  This type is the hallmark of f16 element packing
/// introduced by ConvertTritonAscendGPUToLLVM.  Scheduling segments that
/// contain these packed operations triggers a miscompile in the downstream
/// backend (hivmc), so we conservatively skip them.
static bool containsF16VectorPacking(Block::iterator begin,
                                     Block::iterator end) {
  for (auto it = begin; it != end; ++it) {
    for (auto result : it->getResults()) {
      if (auto vecTy = dyn_cast<VectorType>(result.getType())) {
        if (vecTy.getElementType().isF16() && vecTy.getNumElements() == 2)
          return true;
      }
    }
    for (auto operand : it->getOperands()) {
      if (auto vecTy = dyn_cast<VectorType>(operand.getType())) {
        if (vecTy.getElementType().isF16() && vecTy.getNumElements() == 2)
          return true;
      }
    }
  }
  return false;
}

/// Schedule a range of operations [segBegin, segEnd) using bottom-up
/// store-rooted scheduling. Only operations within this segment are reordered;
/// dependency-tree walks stop at the segment boundary.
static void scheduleSegment(Block &block,
                            Block::iterator segBegin,
                            Block::iterator segEnd) {
  // Skip scheduling for segments with f16 vector packing — the downstream
  // backend is sensitive to instruction ordering for these patterns.
  if (containsF16VectorPacking(segBegin, segEnd))
    return;

  // Collect stores in this segment.
  llvm::SmallVector<ascend_dpx::StoreOp> stores;
  for (auto it = segBegin; it != segEnd; ++it) {
    if (auto storeOp = dyn_cast<ascend_dpx::StoreOp>(&*it))
      stores.push_back(storeOp);
  }

  if (stores.empty())
    return;

  // Build the set of operations that belong to this segment so we can
  // confine dependency walks to within the segment boundary.
  llvm::DenseSet<Operation *> segOps;
  for (auto it = segBegin; it != segEnd; ++it)
    segOps.insert(&*it);

  // `insertPt` is the operation at segEnd (a barrier or the block terminator).
  // All scheduled ops will be placed just before it.
  Operation *insertPt = &*segEnd;
  llvm::DenseSet<Operation *> scheduled;
  scheduled.insert(insertPt);

  // Recursively schedule an operation and all its in-segment dependencies,
  // placing them just before `insertPt`.
  std::function<void(Operation *)> scheduleOp;
  scheduleOp = [&](Operation *op) {
    if (scheduled.contains(op))
      return;
    if (!segOps.contains(op))
      return;
    scheduled.insert(op);

    for (auto operand : op->getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != &block)
        continue;
      scheduleOp(defOp);
    }

    op->moveBefore(insertPt);
  };

  // Schedule each store's dependency tree contiguously.
  for (auto storeOp : stores)
    scheduleOp(storeOp.getOperation());

  // Move any remaining unscheduled segment ops before the store trees.
  // We iterate the entire block (not segBegin..segEnd) because scheduling
  // may have moved segBegin, invalidating that iterator range.
  llvm::SmallVector<Operation *> remaining;
  for (auto &op : block) {
    if (segOps.contains(&op) && !scheduled.contains(&op))
      remaining.push_back(&op);
  }
  for (auto *op : remaining)
    op->moveBefore(insertPt);
}

/// Process a single basic block using bottom-up store-rooted scheduling.
///
/// After SROA, the struct intermediaries are gone and each store's dependency
/// tree is independent (loads → extractelement → div → insertelement → store).
/// We reorder operations so each store's full dependency tree is contiguous.
///
/// Scheduling respects barrier boundaries (e.g. ascend_dpx.sync_threads):
/// operations are never reordered across a barrier. The block is split into
/// segments at each barrier, and each segment is scheduled independently.
static void scheduleBlock(Block &block) {
  if (block.empty())
    return;

  // Split the block into segments at barrier operations and schedule each
  // segment independently. This ensures barriers maintain their relative
  // ordering with respect to loads and stores.
  Operation *terminator = block.getTerminator();
  Block::iterator termIt = terminator->getIterator();
  Block::iterator segBegin = block.begin();
  for (auto it = block.begin(); it != termIt; ++it) {
    if (isa<ascend_dpx::SyncThreadsOp>(&*it)) {
      // Schedule the segment [segBegin, it) — everything before this barrier.
      if (segBegin != it)
        scheduleSegment(block, segBegin, it);
      // Skip past the barrier; next segment starts after it.
      segBegin = std::next(it);
    }
  }
  // Schedule the final segment (after the last barrier, up to the terminator).
  if (segBegin != termIt)
    scheduleSegment(block, segBegin, termIt);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct SinkDPXLoadPass : public impl::SinkDPXLoadBase<SinkDPXLoadPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod->walk([](LLVM::LLVMFuncOp func) {
      for (auto &block : func.getBody()) {
        // Phase 1: Break struct pack/unpack patterns.
        sroaStructs(block);
        removeDeadOps(block);

        // Phase 2: Reorder for minimal register pressure.
        scheduleBlock(block);
      }
    });
  }
};

} // namespace

} // namespace mlir::triton::ascend
