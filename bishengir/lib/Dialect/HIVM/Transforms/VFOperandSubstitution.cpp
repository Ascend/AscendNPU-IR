//===- VFOperandSubstitution.cpp - reuse a VF input buffer for its output -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal VF operand substitution.
//
// When a VF callee proves that writing `dstArg` may reuse the storage read from
// `srcArg`, and the callsite feeds `srcArg` with a *dynamic* buffer (an
// `arith.select` between two buffers) while `dstArg` is a caller-private output
// alloc, PlanMemory cannot express this reuse as a static StorageEntry merge
// (there is no single alloc to merge with). Instead we rewrite the destination
// operand to be the same value as the source operand and delete the now-dead
// output alloc, so the output lands on whichever buffer the select picked at
// runtime.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Analysis/VFInplaceReuseAnalyzer.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
#define GEN_PASS_DEF_VFOPERANDSUBSTITUTION
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "vf-operand-substitution"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

/// A destination is only substitutable when it is a caller-private local alloc
/// that does not escape its block, so deleting it after the rewrite is safe. A
/// destination reached by a terminator (returned from the function, or
/// loop-carried through an `scf.yield`) outlives the VF call and is rejected. We
/// accept identity-preserving views/casts between the operand and the alloc.
static memref::AllocOp getPrivateDstAlloc(Value dst) {
  // Peel identity-preserving view/cast ops off the destination operand.
  Value cur = dst;
  while (Operation *def = cur.getDefiningOp()) {
    if (isa<memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp,
            memref::MemorySpaceCastOp>(def)) {
      cur = def->getOperand(0);
      continue;
    }
    break;
  }
  auto alloc = cur.getDefiningOp<memref::AllocOp>();
  if (!alloc) {
    return nullptr;
  }

  for (Operation *user : alloc->getUsers()) {
    if (user->hasTrait<OpTrait::IsTerminator>()) {
      return nullptr;
    }
  }
  return alloc;
}

/// Returns true if a cross-core `sync_block_set` sits in program order between
/// `call` and `user`, within the block that holds `call`. Such a set releases
/// the reused source buffer to the producer core (AIC), so a destination use
/// that lives past it would read a slot the producer may already have
/// overwritten for the next epoch. `set` and `user` may be nested inside ops of
/// the call's block (e.g. an `scf.if`), so each straddled op is walked.
static bool hasSyncSetBetween(func::CallOp call, Operation *user) {
  Block *blk = call->getBlock();
  Operation *userInBlk = blk->findAncestorOpInBlock(*user);
  if (!userInBlk || userInBlk == call.getOperation() ||
      userInBlk->isBeforeInBlock(call)) {
    return false;
  }
  for (Operation *op = call->getNextNode(); op && op != userInBlk;
       op = op->getNextNode()) {
    bool found = false;
    op->walk([&](hivm::SyncBlockSetOp) {
      found = true;
      return WalkResult::interrupt();
    });
    if (found) {
      return true;
    }
  }
  return false;
}

struct Candidate {
  func::CallOp call;
  unsigned dstIdx;
  unsigned srcIdx;
  memref::AllocOp dstAlloc;
};

struct VFOperandSubstitutionPass
    : public impl::VFOperandSubstitutionBase<VFOperandSubstitutionPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    VFInplaceReuseAnalysis analysis(module);
    SymbolTableCollection symbolTable;

    SmallVector<Candidate> candidates;
    module.walk([&](func::CallOp call) {
      if (!hivm::isVFCall(call)) {
        return;
      }
      auto callee = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee) {
        return;
      }
      for (auto [dstIdx, srcIdx] :
           analysis.getInplaceReusableArgPairs(callee)) {
        if (dstIdx >= call.getNumOperands() || srcIdx >= call.getNumOperands()) {
          continue;
        }
        Value dstOperand = call.getOperand(dstIdx);
        Value srcOperand = call.getOperand(srcIdx);
        if (dstOperand == srcOperand) {
          continue;
        }
        // Minimal, targeted scope: only the dynamic-select source case, which
        // is exactly what static StorageEntry merge cannot handle. Normal
        // alloc-to-alloc VF reuse is left to PlanMemory untouched.
        if (!srcOperand.getDefiningOp<arith::SelectOp>()) {
          continue;
        }
        if (dstOperand.getType() != srcOperand.getType()) {
          continue;
        }
        // The selected source must be consumed only by this call, so that
        // overwriting it here does not clobber a value still needed elsewhere.
        if (!srcOperand.hasOneUse()) {
          continue;
        }
        memref::AllocOp dstAlloc = getPrivateDstAlloc(dstOperand);
        if (!dstAlloc) {
          continue;
        }
        // After substitution the destination's users read the overwritten
        // source buffer. If any such user is separated from the call by a
        // `sync_block_set`, the reused slot is released to the producer core
        // before that read, so the reuse is unsafe. The source itself has a
        // single use (this call, checked above), so the destination users are
        // the only extra readers of the merged buffer to consider.
        bool crossesSyncSet = false;
        for (Operation *user : dstOperand.getUsers()) {
          if (user == call.getOperation()) {
            continue;
          }
          if (hasSyncSetBetween(call, user)) {
            crossesSyncSet = true;
            break;
          }
        }
        if (crossesSyncSet) {
          continue;
        }
        candidates.push_back({call, dstIdx, srcIdx, dstAlloc});
        LDBG("substitution candidate: call " << call.getCallee() << " dst arg "
                                             << dstIdx << " <- src arg "
                                             << srcIdx);
      }
    });

    for (Candidate &c : candidates) {
      Value dstOperand = c.call.getOperand(c.dstIdx);
      Value srcOperand = c.call.getOperand(c.srcIdx);
      // Redirect every use of the dead output buffer to the reused source
      // buffer, then erase the now-unused output alloc.
      dstOperand.replaceAllUsesWith(srcOperand);
      if (c.dstAlloc->use_empty()) {
        c.dstAlloc.erase();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createVFOperandSubstitutionPass() {
  return std::make_unique<VFOperandSubstitutionPass>();
}
