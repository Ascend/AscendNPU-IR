//===- HoistTightlyCoupledAlloc.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hoists tightly-coupled `memref.alloc`s that are yielded out of an inner
// region up to the outermost region they escape to.
//
// Background: an L1/UB alloc created inside an inner loop whose tensor view is
// yielded out (e.g. it carries an mmad/fixpipe result that is consumed after
// the loop) is, on the AIV side, kept live past the loop (so auto-multi-buffer
// anchors its slot rotation on the *outer* loop), while on the AIC side only
// the in-loop producer survives SplitMixKernel (so the same buffer anchors on
// the *inner* loop). For a CV tightly-coupled buffer this anchor mismatch makes
// the producer and consumer rotate physical slots on different loop counters
// and corrupts results.
//
// Moving the alloc out of the inner region (to the region the yielded value
// escapes to) makes the buffer live at the same loop level on both cores, so
// the two anchors agree again.
//
// This runs on the MIX function before SplitMixKernel, so both AIC/AIV clones
// inherit the hoisted placement.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_HOISTTIGHTLYCOUPLEDALLOC
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

// Debug type for this pass, used with llvm's DEBUG_WITH_TYPE.
constexpr const char *kDebugType = "hivm-hoist-tightly-coupled-alloc";

/// If `op` just forwards/views another value (without changing the underlying
/// buffer), return that source value; otherwise return a null Value.
static Value getViewSource(Operation *op) {
  if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op))
    return toTensor.getMemref();
  if (auto spaceCast = dyn_cast<memref::MemorySpaceCastOp>(op))
    return spaceCast.getSource();
  if (auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(op))
    return viewLikeOp.getViewSource();
  if (auto memCast = dyn_cast<memref::CastOp>(op))
    return memCast.getSource();
  if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op))
    return expand.getSrc();
  if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op))
    return collapse.getSrc();
  if (auto tensorCast = dyn_cast<tensor::CastOp>(op))
    return tensorCast.getSource();
  return {};
}

/// Returns true if `v` is derived from `root` purely through forwarding/view
/// ops (`to_tensor`, `memory_space_cast`, `subview`, `extract_slice`,
/// `expand_shape`, `collapse_shape`, `cast`).
static bool tracesToValue(Value v, Value root) {
  while (v) {
    if (v == root)
      return true;
    Operation *def = v.getDefiningOp();
    if (!def)
      return false;
    v = getViewSource(def);
  }
  return false;
}

/// In `values`, find the first one that traces back to `srcVal`.
static int findTracedIndex(ValueRange values, Value srcVal) {
  for (auto [i, value] : llvm::enumerate(values)) {
    if (tracesToValue(value, srcVal))
      return static_cast<int>(i);
  }
  return -1;
}

static unsigned getBlockDepth(Block *block) {
  unsigned depth = 0;
  for (Operation *parent = block ? block->getParentOp() : nullptr; parent;
       parent = parent->getBlock() ? parent->getBlock()->getParentOp()
                                   : nullptr) {
    ++depth;
  }
  return depth;
}

static Block *getOuterBlock(Block *lhs, Block *rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return getBlockDepth(lhs) <= getBlockDepth(rhs) ? lhs : rhs;
}

static Block *findHoistTargetBlock(Value carried, Block *curBlock);

/// Follow the escape chains of the `scf.while` iteration slot `idx` and return
/// the outermost block, seeded with the while op's own block.
///
/// An scf.while value can leave the loop two ways:
///  - init / loop-carried chain: the before-region iter_arg `idx` is fed by
///    `getInits()[idx]` (and the after-region yield).
///  - result chain: a value forwarded by `scf.condition` becomes both the
///    after-region argument and the matching `getResult(idx)`.
///
/// `traceInit` / `traceResult` force-follow the corresponding chain. Whenever
/// the before-region iter_arg `idx` is forwarded unchanged by `scf.condition`,
/// or the after-region arg `idx` is yielded unchanged, the two chains are
/// equivalent, so both are followed regardless of the flags.
static Block *traceWhileSlot(scf::WhileOp whileOp, int idx, bool traceInit,
                             bool traceResult) {
  Block *whileBlock = whileOp->getBlock();
  Block *best = whileBlock;
  if (idx < 0)
    return best;

  auto condOp =
      dyn_cast<scf::ConditionOp>(whileOp.getBeforeBody()->getTerminator());
  auto beforeArgs = whileOp.getBeforeArguments();
  bool conditionCheck = condOp &&
                        idx < static_cast<int>(condOp.getArgs().size()) &&
                        idx < static_cast<int>(beforeArgs.size()) &&
                        condOp.getArgs()[idx] == beforeArgs[idx];

  auto yieldOp =
      dyn_cast<scf::YieldOp>(whileOp.getAfterBody()->getTerminator());
  auto afterArgs = whileOp.getAfterArguments();
  bool yieldCheck = yieldOp &&
                    idx < static_cast<int>(yieldOp.getOperands().size()) &&
                    idx < static_cast<int>(afterArgs.size()) &&
                    yieldOp.getOperands()[idx] == afterArgs[idx];

  bool forwarded = conditionCheck || yieldCheck;

  if ((traceInit || forwarded) &&
      idx < static_cast<int>(whileOp.getInits().size()))
    best = getOuterBlock(
        best, findHoistTargetBlock(whileOp.getInits()[idx], whileBlock));

  if ((traceResult || forwarded) &&
      idx < static_cast<int>(whileOp->getNumResults()))
    best = getOuterBlock(
        best, findHoistTargetBlock(whileOp.getResult(idx), whileBlock));

  return best;
}

/// Walk outward from `alloc`, following the chain of yields that carry a value
/// derived from the alloc, and return the outermost block the value escapes to.
/// Returns nullptr when the alloc's view is not yielded anywhere (no hoist).
static Block *findHoistTargetBlock(Value carried, Block *curBlock) {
  Block *targetBlock = nullptr;

  while (curBlock) {
    if (auto blockArg = dyn_cast<BlockArgument>(carried)) {
      Block *argBlock = blockArg.getOwner();
      Operation *argParent = argBlock->getParentOp();

      // `scf.while` is handled specially: it carries values through two regions,
      // so a value can escape via both the init chain and the result chain.
      if (auto whileOp = dyn_cast_if_present<scf::WhileOp>(argParent)) {
        int idx = static_cast<int>(blockArg.getArgNumber());
        // Before-region iter_arg: always trace its init operand outward; if it
        // is forwarded unchanged by scf.condition, also trace the result.
        if (argBlock == whileOp.getBeforeBody())
          return getOuterBlock(
              targetBlock, traceWhileSlot(whileOp, idx, /*traceInit=*/true,
                                          /*traceResult=*/false));
        // After-region arg is the value forwarded by scf.condition, i.e. the
        // matching while result: always trace the result outward; if it is the
        // forwarded before-region iter_arg, also trace the init.
        if (argBlock == whileOp.getAfterBody())
          return getOuterBlock(
              targetBlock, traceWhileSlot(whileOp, idx, /*traceInit=*/false,
                                          /*traceResult=*/true));
      }

      // Generic loop iter_arg (scf.for, scf.forall, ...): hop to the init
      // operand defined in the enclosing block and keep walking outward.
      auto loopOp = dyn_cast_if_present<LoopLikeOpInterface>(argParent);
      if (OpOperand *initOperand =
              loopOp ? loopOp.getTiedLoopInit(blockArg) : nullptr) {
        Block *iterArgLoopBlock = loopOp->getBlock();
        targetBlock = getOuterBlock(targetBlock, iterArgLoopBlock);
        curBlock = iterArgLoopBlock;
        carried = initOperand->get();
        continue;
      }
    }

    Operation *parent = curBlock->getParentOp();
    if (!parent)
      break;

    Value nextCarried;
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      // scf.for body terminator: yield operand i -> next iteration iter_arg i
      // and final result i. Follow both the init_arg chain and the result
      // chain, then choose the outermost target. The init_arg branch is needed
      // for nested loops where an inner loop result is not yielded outward, but
      // the yielded value is carried by an enclosing loop's iter_arg.
      auto yieldOp = dyn_cast<scf::YieldOp>(curBlock->getTerminator());
      if (!yieldOp)
        break;
      int idx = findTracedIndex(yieldOp.getOperands(), carried);
      if (idx < 0)
        break;

      Block *forBlock = parent->getBlock();
      Block *bestTarget = forBlock;
      bestTarget = getOuterBlock(
          bestTarget,
          findHoistTargetBlock(forOp.getInitArgs()[idx], forBlock));
      bestTarget = getOuterBlock(
          bestTarget, findHoistTargetBlock(forOp.getResult(idx), forBlock));
      return bestTarget;
    } else if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      // scf.if then/else terminator: yield operand i -> result i.
      auto yieldOp = dyn_cast<scf::YieldOp>(curBlock->getTerminator());
      if (!yieldOp)
        break;
      int idx = findTracedIndex(yieldOp.getOperands(), carried);
      if (idx < 0)
        break;
      nextCarried = ifOp.getResult(idx);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      // scf.while has two regions:
      //  - "after" terminated by scf.yield %next... : the yielded value feeds
      //    back as the next-iteration init arg, so trace the init chain (and the
      //    result chain when that iter_arg is forwarded by scf.condition).
      //  - "before" terminated by scf.condition(cond) %fwd... : a forwarded
      //    value escapes through the matching while result, so trace the result
      //    chain (and the init chain when the forwarded value is the iter_arg).
      if (curBlock == whileOp.getAfterBody()) {
        auto yieldOp = dyn_cast<scf::YieldOp>(curBlock->getTerminator());
        if (!yieldOp)
          break;
        int idx = findTracedIndex(yieldOp.getOperands(), carried);
        if (idx < 0)
          break;
        return getOuterBlock(
            targetBlock, traceWhileSlot(whileOp, idx, /*traceInit=*/true,
                                        /*traceResult=*/false));
      } else if (curBlock == whileOp.getBeforeBody()) {
        auto condOp = dyn_cast<scf::ConditionOp>(curBlock->getTerminator());
        if (!condOp)
          break;
        int idx = findTracedIndex(condOp.getArgs(), carried);
        if (idx < 0)
          break;
        return getOuterBlock(
            targetBlock, traceWhileSlot(whileOp, idx, /*traceInit=*/false,
                                        /*traceResult=*/true));
      } else {
        break;
      }
    } else {
      break;
    }

    targetBlock = parent->getBlock();
    curBlock = targetBlock;
    carried = nextCarried;
  }

  return targetBlock;
}

static Block *findHoistTargetBlock(memref::AllocOp alloc) {
  return findHoistTargetBlock(alloc.getResult(), alloc->getBlock());
}

struct HoistTightlyCoupledAllocPass
    : public impl::HoistTightlyCoupledAllocBase<HoistTightlyCoupledAllocPass> {
  void runOnOperation() override;
};

void HoistTightlyCoupledAllocPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (hacc::utils::isHost(func))
    return;

  SmallVector<memref::AllocOp> worklist;
  func.walk([&](memref::AllocOp allocOp) {
    // Only L1/UB local buffers are tightly-coupled candidates.
    auto addressSpace =
        mlir::hivm::getOptionalHIVMAddressSpace(allocOp.getMemref().getType());
    if (addressSpace != AddressSpace::L1 && addressSpace != AddressSpace::UB)
      return;
    // Only allocs carrying the tightly-coupled-buffer mark.
    if (!utils::getAnnotateOpWithAttr(allocOp.getMemref(),
                                      hivm::HIVMTightlyCoupledBufferAttr::name)
             .has_value())
      return;
    // Only static allocs (no dynamic/symbol operands) can be hoisted freely
    // without recomputing operands at the new location.
    if (allocOp->getNumOperands() != 0)
      return;
    worklist.push_back(allocOp);
  });

  for (memref::AllocOp allocOp : worklist) {
    auto maybeMark = utils::getAnnotateOpWithAttr(
        allocOp.getMemref(), hivm::HIVMTightlyCoupledBufferAttr::name);
    if (maybeMark.has_value())
      (*maybeMark)->moveAfter(allocOp);
  }

  for (memref::AllocOp allocOp : worklist) {
    Block *target = findHoistTargetBlock(allocOp);
    if (!target || target == allocOp->getBlock())
      continue;
    DEBUG_WITH_TYPE(kDebugType, llvm::dbgs()
                                    << "[" << kDebugType << "]: "
                                    << "hoisting tightly-coupled alloc: "
                                    << allocOp << "\n");
    // Move the tightly-coupled mark (if any) together with the alloc so the
    // annotation stays adjacent to its buffer.
    auto maybeMark = utils::getAnnotateOpWithAttr(
        allocOp.getMemref(), hivm::HIVMTightlyCoupledBufferAttr::name);
    allocOp->moveBefore(&target->front());
    if (maybeMark.has_value())
      (*maybeMark)->moveAfter(allocOp);
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createHoistTightlyCoupledAllocPass() {
  return std::make_unique<HoistTightlyCoupledAllocPass>();
}
