//===- HoistTightlyCoupledAlloc.cpp ---------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025~2026. All rights reserved.
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/TightlyCoupledBufferUtils.h"

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

#define DEBUG_TYPE "hivm-hoist-tightly-coupled-alloc"

using namespace mlir;
using namespace mlir::hivm;

namespace {

/// In `values`, find the first one that traces back to `srcVal`.
static int findTracedIndex(ValueRange values, Value srcVal) {
  for (auto [i, value] : llvm::enumerate(values)) {
    if (tracesToTightlyCoupledValue(value, srcVal))
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

      if (auto whileOp = dyn_cast_if_present<scf::WhileOp>(argParent)) {
        int idx = static_cast<int>(blockArg.getArgNumber());
        if (argBlock == whileOp.getBeforeBody())
          return getOuterBlock(targetBlock,
                               traceWhileSlot(whileOp, idx, /*traceInit=*/true,
                                              /*traceResult=*/false));
        if (argBlock == whileOp.getAfterBody())
          return getOuterBlock(targetBlock,
                               traceWhileSlot(whileOp, idx, /*traceInit=*/false,
                                              /*traceResult=*/true));
      }

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
      auto yieldOp = dyn_cast<scf::YieldOp>(curBlock->getTerminator());
      if (!yieldOp)
        break;
      int idx = findTracedIndex(yieldOp.getOperands(), carried);
      if (idx < 0)
        break;

      Block *forBlock = parent->getBlock();
      Block *bestTarget = forBlock;
      bestTarget = getOuterBlock(
          bestTarget, findHoistTargetBlock(forOp.getInitArgs()[idx], forBlock));
      bestTarget = getOuterBlock(
          bestTarget, findHoistTargetBlock(forOp.getResult(idx), forBlock));
      return bestTarget;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      auto yieldOp = dyn_cast<scf::YieldOp>(curBlock->getTerminator());
      if (!yieldOp)
        break;
      int idx = findTracedIndex(yieldOp.getOperands(), carried);
      if (idx < 0)
        break;
      nextCarried = ifOp.getResult(idx);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      if (curBlock == whileOp.getAfterBody()) {
        auto yieldOp = dyn_cast<scf::YieldOp>(curBlock->getTerminator());
        if (!yieldOp)
          break;
        int idx = findTracedIndex(yieldOp.getOperands(), carried);
        if (idx < 0)
          break;
        return getOuterBlock(targetBlock,
                             traceWhileSlot(whileOp, idx, /*traceInit=*/true,
                                            /*traceResult=*/false));
      }
      if (curBlock == whileOp.getBeforeBody()) {
        auto condOp = dyn_cast<scf::ConditionOp>(curBlock->getTerminator());
        if (!condOp)
          break;
        int idx = findTracedIndex(condOp.getArgs(), carried);
        if (idx < 0)
          break;
        return getOuterBlock(targetBlock,
                             traceWhileSlot(whileOp, idx, /*traceInit=*/false,
                                            /*traceResult=*/true));
      }
      break;
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

  // Match MarkTightlyCoupledBuffer: only RegBase Ascend950 uses CV tightly-
  // coupled buffers, so hoisting is a no-op on other arches.
  auto module = func->getParentOfType<ModuleOp>();
  if (!module || !hacc::utils::isAscend950(module))
    return;

  SmallVector<memref::AllocOp> worklist;
  func.walk([&](memref::AllocOp allocOp) {
    if (!isL1OrUBAlloc(allocOp))
      return;
    if (!getTightlyCoupledMark(allocOp.getMemref()).has_value())
      return;
    // Only static allocs (no dynamic/symbol operands) can be hoisted freely
    // without recomputing operands at the new location.
    if (allocOp->getNumOperands() != 0)
      return;
    worklist.push_back(allocOp);
  });

  for (memref::AllocOp allocOp : worklist) {
    Block *target = findHoistTargetBlock(allocOp);
    if (!target || target == allocOp->getBlock())
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "]: "
               << "hoisting tightly-coupled alloc: " << allocOp << "\n");
    auto maybeMark = getTightlyCoupledMark(allocOp.getMemref());
    allocOp->moveBefore(&target->front());
    if (maybeMark.has_value())
      (*maybeMark)->moveAfter(allocOp);
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createHoistTightlyCoupledAllocPass() {
  return std::make_unique<HoistTightlyCoupledAllocPass>();
}
