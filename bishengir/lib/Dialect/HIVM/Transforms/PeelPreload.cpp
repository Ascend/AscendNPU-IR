//===- PeelPreload.cpp - Peel preload prologue/epilogue ------------------===//
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_PEELPRELOAD
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-peel-preload"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct PeelPreloadPass : public impl::PeelPreloadBase<PeelPreloadPass> {
  void runOnOperation() override;
};

/// Stage index recorded by CreatePreload on `annotation.mark` of the guard
/// condition. Discardable attrs on `scf.if` / `arith.andi` do not survive
/// canonicalization.
static IntegerAttr getPreloadNumAttr(scf::IfOp ifOp) {
  std::optional<Operation *> mark = utils::getAnnotateOpWithAttr(
      ifOp.getCondition(), hivm::PreloadNumAttr::name);
  if (!mark)
    return {};
  return dyn_cast_or_null<IntegerAttr>(
      cast<annotation::MarkOp>(*mark).getStaticAttrValue(
          hivm::PreloadNumAttr::name));
}

static bool isPreloadNumMark(Operation *op) {
  return utils::isAnnotationWithAttr(op, hivm::PreloadNumAttr::name);
}

static void erasePreloadNumMarks(Operation *root) {
  SmallVector<annotation::MarkOp> marks;
  root->walk([&](annotation::MarkOp mark) {
    if (isPreloadNumMark(mark))
      marks.push_back(mark);
  });
  for (annotation::MarkOp mark : marks)
    mark.erase();
}

/// Inline `region` in place of `op`, replacing results with the terminator
/// operands.
static void replaceOpWithRegion(RewriterBase &rewriter, Operation *op,
                                Region &region) {
  assert(llvm::hasSingleElement(region) && "expected single-block region");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// mappedIV(preloadNum) = IV - (maxPreloadNum - 1 - preloadNum) with unit step.
/// Under the peel gate (trip >= 2 * maxPreloadNum):
///   prologue IV = lb + i (i in [0, max)): run iff preloadNum >= max - 1 - i
///   epilogue IV = ub - i (i in [1, max]): run iff i >= preloadNum + 2
static bool shouldRunPeeledPreloadIf(int64_t preloadNum, int64_t maxPreloadNum,
                                     std::optional<int64_t> prologueIndex,
                                     std::optional<int64_t> epilogueIndex) {
  assert(preloadNum >= 0 && preloadNum < maxPreloadNum &&
         "preload_num out of range");
  if (prologueIndex) {
    assert(*prologueIndex >= 0 && *prologueIndex < maxPreloadNum);
    return preloadNum >= maxPreloadNum - 1 - *prologueIndex;
  }
  assert(epilogueIndex && *epilogueIndex >= 1 &&
         *epilogueIndex <= maxPreloadNum);
  return *epilogueIndex >= preloadNum + 2;
}

static bool needsMultiBufferExpansion(PointerCastOp pc) {
  return pc.getAddrs().size() > 1 && !util::isGMPointerCastOp(pc);
}

static bool isMultiBufferMark(annotation::MarkOp mark) {
  return mark->hasAttr(hivm::MultiBufferAttr::name);
}

/// True when both casts select from the same multi-buffer address ring.
static bool samePointerCastAddrs(PointerCastOp a, PointerCastOp b) {
  if (a.getAddrs().size() != b.getAddrs().size())
    return false;
  for (auto [lhs, rhs] : llvm::zip(a.getAddrs(), b.getAddrs())) {
    if (lhs != rhs)
      return false;
  }
  return true;
}

/// True when `region` uses a pointer_cast from the same address ring as
/// `yieldPC` (typically the write target sibling of a yielded multi-buffer
/// view).
static bool regionUsesPointerCastAddrs(Region &region, PointerCastOp yieldPC) {
  if (region.empty())
    return false;
  bool found = false;
  region.walk([&](Operation *op) {
    if (isa<annotation::MarkOp>(op))
      return WalkResult::advance();
    for (Value operand : op->getOperands()) {
      auto pc = dyn_cast_or_null<PointerCastOp>(operand.getDefiningOp());
      if (pc && samePointerCastAddrs(pc, yieldPC)) {
        found = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

/// Yields that are multi-buffer pointer_casts defined in the loop body are
/// often siblings of the real write target (same address ring). Empty peel
/// slots still evaluate that yield even when the producer preload_if did not
/// run, which would replace a carried result with an unused buffer view.
/// Keep the previous iter_arg unless a running preload_if uses that ring.
static bool shouldUpdateMultiBufferYield(Value yieldVal, scf::ForOp forOp,
                                         int64_t peelCount,
                                         std::optional<int64_t> prologueIndex,
                                         std::optional<int64_t> epilogueIndex) {
  auto yieldPC = dyn_cast_or_null<PointerCastOp>(yieldVal.getDefiningOp());
  if (!yieldPC || !needsMultiBufferExpansion(yieldPC))
    return true;
  if (yieldPC->getParentOp() != forOp)
    return true;

  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto preloadIf = dyn_cast<scf::IfOp>(&op);
    if (!preloadIf)
      continue;
    IntegerAttr preloadNumAttr = getPreloadNumAttr(preloadIf);
    if (!preloadNumAttr)
      continue;
    if (!shouldRunPeeledPreloadIf(preloadNumAttr.getInt(), peelCount,
                                  prologueIndex, epilogueIndex))
      continue;
    if (regionUsesPointerCastAddrs(preloadIf.getThenRegion(), yieldPC))
      return true;
  }
  return false;
}

/// Create a function-scoped counter alloca matching LowerMultiBufferCounter.
/// Shared by prologue/middle/epilogue on the peel-then path so event-ID
/// rotation and multi-buffer pointer_cast selection stay consistent.
static Value createCounterAlloca(RewriterBase &rewriter, Location loc,
                                 func::FuncOp funcOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  Type i64Ty = rewriter.getI64Type();
  auto memTy = MemRefType::get(/*shape=*/{1}, i64Ty);
  auto alloca = rewriter.create<memref::AllocaOp>(loc, memTy);
#ifndef BSPUB_DAVINCI_BISHENGIR_A5
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, /*value=*/0, i64Ty);
#else
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, i64Ty, /*value=*/0);
#endif
  Value initIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  rewriter.create<memref::StoreOp>(loc, zero, alloca.getResult(),
                                   ValueRange{initIdx});
  return alloca.getResult();
}

static Value createSharedCounterAlloca(RewriterBase &rewriter,
                                       scf::ForOp forOp) {
  bool needsAlloca = false;
  forOp.walk([&](Operation *op) {
    if (isa<MultiBufferCounterOp>(op)) {
      needsAlloca = true;
      return WalkResult::interrupt();
    }
    if (auto pc = dyn_cast<PointerCastOp>(op);
        pc && needsMultiBufferExpansion(pc)) {
      needsAlloca = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!needsAlloca)
    return {};

  auto funcOp = forOp->getParentOfType<func::FuncOp>();
  assert(funcOp && "preload for must be inside a func");
  return createCounterAlloca(rewriter, forOp.getLoc(), funcOp);
}

static Value loadCounter(RewriterBase &rewriter, Location loc, Value alloca) {
  Value idx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  return rewriter.create<memref::LoadOp>(loc, alloca, ValueRange{idx});
}

static void incrementCounter(RewriterBase &rewriter, Location loc, Value alloca,
                             Value cur) {
#ifndef BSPUB_DAVINCI_BISHENGIR_A5
  Value one =
      rewriter.create<arith::ConstantIntOp>(loc, /*value=*/1, cur.getType());
#else
  Value one =
      rewriter.create<arith::ConstantIntOp>(loc, cur.getType(), /*value=*/1);
#endif
  Value next = rewriter.create<arith::AddIOp>(loc, cur, one);
  Value idx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  rewriter.create<memref::StoreOp>(loc, next, alloca, ValueRange{idx});
}

/// Select `mappedAddrs[iterCounter % N]` and emit a single-addr pointer_cast.
/// Matches EnableMultiBuffer / GraphSync, which key buffer and event rotation
/// off the same multi-buffer counter.
static Value expandMultiBufferedPointerCast(RewriterBase &rewriter,
                                            PointerCastOp pc,
                                            ValueRange mappedAddrs,
                                            ValueRange mappedDynSizes,
                                            Value iterCounter) {
  Location loc = pc.getLoc();
  unsigned factor = mappedAddrs.size();
  assert(factor > 1 && "expected multi-addr pointer_cast");

  Type i64Ty = iterCounter.getType();
#ifndef BSPUB_DAVINCI_BISHENGIR_A5
  Value factorVal = rewriter.create<arith::ConstantIntOp>(
      loc, /*value=*/static_cast<int64_t>(factor), i64Ty);
#else
  Value factorVal = rewriter.create<arith::ConstantIntOp>(
      loc, i64Ty, /*value=*/static_cast<int64_t>(factor));
#endif
  Value idx = rewriter.create<arith::RemUIOp>(loc, iterCounter, factorVal);
  Value selectedAddr = mappedAddrs.front();
  for (unsigned i = 1; i < factor; ++i) {
#ifndef BSPUB_DAVINCI_BISHENGIR_A5
    Value iVal = rewriter.create<arith::ConstantIntOp>(
        loc, /*value=*/static_cast<int64_t>(i), i64Ty);
#else
    Value iVal = rewriter.create<arith::ConstantIntOp>(
        loc, i64Ty, /*value=*/static_cast<int64_t>(i));
#endif
    Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                idx, iVal);
    selectedAddr = rewriter.create<arith::SelectOp>(
        loc, selectedAddr.getType(), cond, mappedAddrs[i], selectedAddr);
  }
  return rewriter.create<PointerCastOp>(loc, pc.getType(), selectedAddr,
                                        mappedDynSizes);
}

/// Erase multi_buffer marks on `value`. Marks that also carry other attrs keep
/// those attrs; a mark that only had multi_buffer is removed.
static void stripMultiBufferMarks(RewriterBase &rewriter, Value value) {
  SmallVector<annotation::MarkOp> marks;
  for (Operation *user : value.getUsers()) {
    if (auto mark = dyn_cast<annotation::MarkOp>(user);
        mark && isMultiBufferMark(mark))
      marks.push_back(mark);
  }
  for (annotation::MarkOp mark : marks) {
    NamedAttrList attrs(mark->getAttrDictionary());
    attrs.erase(hivm::MultiBufferAttr::name);
    bool hasOther = llvm::any_of(
        attrs, [](NamedAttribute na) { return na.getName() != "effects"; });
    if (!hasOther) {
      rewriter.eraseOp(mark);
      continue;
    }
    mark->setAttrs(attrs.getDictionary(mark.getContext()));
  }
}

/// Replace every top-level `multi_buffer_counter` in `block` with one load and
/// one increment. Returns the loaded counter value (or null if none).
static Value lowerCountersInBlock(RewriterBase &rewriter, Block *block,
                                  Value alloca) {
  SmallVector<MultiBufferCounterOp> counters;
  for (Operation &op : block->without_terminator()) {
    if (auto counter = dyn_cast<MultiBufferCounterOp>(&op))
      counters.push_back(counter);
  }
  if (counters.empty())
    return {};

  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = counters.front().getLoc();
  rewriter.setInsertionPoint(counters.front());
  Value val = loadCounter(rewriter, loc, alloca);
  for (MultiBufferCounterOp counter : counters)
    rewriter.replaceOp(counter, val);
  rewriter.setInsertionPoint(block->getTerminator());
  incrementCounter(rewriter, loc, alloca, val);
  return val;
}

/// Expand top-level multi-addr pointer_casts in `block` with
/// `iterCounter % N` and strip their multi_buffer marks.
static void expandMultiBufferCastsInBlock(RewriterBase &rewriter, Block *block,
                                          Value iterCounter) {
  if (!iterCounter)
    return;
  SmallVector<PointerCastOp> toExpand;
  for (Operation &op : block->without_terminator()) {
    if (auto pc = dyn_cast<PointerCastOp>(&op);
        pc && needsMultiBufferExpansion(pc))
      toExpand.push_back(pc);
  }
  OpBuilder::InsertionGuard guard(rewriter);
  for (PointerCastOp pc : toExpand) {
    rewriter.setInsertionPoint(pc);
    Value selected = expandMultiBufferedPointerCast(
        rewriter, pc, pc.getAddrs(), pc.getDynamicSizes(), iterCounter);
    stripMultiBufferMarks(rewriter, pc.getResult());
    pc.replaceAllUsesWith(selected);
    rewriter.eraseOp(pc);
  }
}

/// Lower counters, expand multi-addr casts via the same counter, and strip
/// multi_buffer marks in `block`. Nested loop bodies get their own allocas (as
/// LowerMultiBufferCounter would). Nested if reuses `sharedAlloca`.
static void processMultiBufferInBlock(RewriterBase &rewriter, Block *block,
                                      Value sharedAlloca, func::FuncOp funcOp) {
  Value alloca = sharedAlloca;
  bool hasTopLevelCounter =
      llvm::any_of(block->without_terminator(),
                   [](Operation &op) { return isa<MultiBufferCounterOp>(op); });
  bool hasTopLevelMultiCast =
      llvm::any_of(block->without_terminator(), [](Operation &op) {
        auto pc = dyn_cast<PointerCastOp>(op);
        return pc && needsMultiBufferExpansion(pc);
      });
  if ((hasTopLevelCounter || hasTopLevelMultiCast) && !alloca)
    alloca = createCounterAlloca(rewriter, block->front().getLoc(), funcOp);

  Value iterCounter = lowerCountersInBlock(rewriter, block, alloca);
  if (!iterCounter && hasTopLevelMultiCast) {
    OpBuilder::InsertionGuard bodyGuard(rewriter);
    rewriter.setInsertionPointToStart(block);
    iterCounter = loadCounter(rewriter, block->front().getLoc(), alloca);
    rewriter.setInsertionPoint(block->getTerminator());
    incrementCounter(rewriter, block->front().getLoc(), alloca, iterCounter);
  }
  expandMultiBufferCastsInBlock(rewriter, block, iterCounter);

  // Strip any remaining multi_buffer marks in this block (e.g. on subviews).
  SmallVector<annotation::MarkOp> marks;
  for (Operation &op : block->without_terminator()) {
    if (auto mark = dyn_cast<annotation::MarkOp>(&op);
        mark && isMultiBufferMark(mark))
      marks.push_back(mark);
  }
  for (annotation::MarkOp mark : marks)
    stripMultiBufferMarks(rewriter, mark.getSrc());

  // Nested regions: each loop body owns an independent counter alloca.
  for (Operation &op : block->without_terminator()) {
    if (auto nestFor = dyn_cast<scf::ForOp>(&op)) {
      processMultiBufferInBlock(rewriter, nestFor.getBody(),
                                /*sharedAlloca=*/{}, funcOp);
    } else if (auto nestWhile = dyn_cast<scf::WhileOp>(&op)) {
      processMultiBufferInBlock(rewriter, &nestWhile.getBefore().front(),
                                /*sharedAlloca=*/{}, funcOp);
      processMultiBufferInBlock(rewriter, &nestWhile.getAfter().front(),
                                /*sharedAlloca=*/{}, funcOp);
    } else if (auto nestIf = dyn_cast<scf::IfOp>(&op)) {
      if (!nestIf.getThenRegion().empty())
        processMultiBufferInBlock(rewriter, &nestIf.getThenRegion().front(),
                                  alloca, funcOp);
      if (!nestIf.getElseRegion().empty())
        processMultiBufferInBlock(rewriter, &nestIf.getElseRegion().front(),
                                  alloca, funcOp);
    }
  }
}

/// Clone one peeled iteration. `hivm.preload_num` is read from the
/// `annotation.mark` of the guard condition so IRMapping stays valid.
/// When `counterAlloca` is set, map every `multi_buffer_counter` in this clone
/// to one load, expand multi-buffer pointer_casts with that counter, drop
/// multi_buffer marks, and increment once after the clone (unrolled iteration).
static void clonePeeledIteration(RewriterBase &rewriter, scf::ForOp forOp,
                                 Value iv, SmallVector<Value> &initArgs,
                                 int64_t peelCount,
                                 std::optional<int64_t> prologueIndex,
                                 std::optional<int64_t> epilogueIndex,
                                 Value counterAlloca) {
  auto funcOp = forOp->getParentOfType<func::FuncOp>();
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), iv);
  mapping.map(forOp.getRegionIterArgs(), initArgs);
  Value iterCounter;
  auto ensureCounter = [&](Location loc) {
    if (!iterCounter && counterAlloca)
      iterCounter = loadCounter(rewriter, loc, counterAlloca);
    return iterCounter;
  };
  std::function<void(Operation &)> cloneOp = [&](Operation &op) {
    // Stage index is read from the original mark; do not clone it into the
    // peeled body (it would keep the guard condition live).
    if (isPreloadNumMark(&op))
      return;
    if (counterAlloca) {
      if (auto counter = dyn_cast<MultiBufferCounterOp>(&op)) {
        mapping.map(counter.getResult(), ensureCounter(op.getLoc()));
        return;
      }
      if (auto pc = dyn_cast<PointerCastOp>(&op);
          pc && needsMultiBufferExpansion(pc)) {
        SmallVector<Value> addrs = llvm::map_to_vector(
            pc.getAddrs(), [&](Value v) { return mapping.lookupOrDefault(v); });
        SmallVector<Value> dynSizes =
            llvm::map_to_vector(pc.getDynamicSizes(), [&](Value v) {
              return mapping.lookupOrDefault(v);
            });
        Value selected = expandMultiBufferedPointerCast(
            rewriter, pc, addrs, dynSizes, ensureCounter(op.getLoc()));
        mapping.map(pc.getResult(), selected);
        return;
      }
      if (auto mark = dyn_cast<annotation::MarkOp>(&op)) {
        if (isMultiBufferMark(mark)) {
          // Drop multi_buffer. Keep the mark only when other attrs remain.
          NamedAttrList attrs(mark->getAttrDictionary());
          attrs.erase(hivm::MultiBufferAttr::name);
          bool hasOther = llvm::any_of(attrs, [](NamedAttribute na) {
            return na.getName() != "effects";
          });
          if (!hasOther)
            return;
          auto newMark = rewriter.create<annotation::MarkOp>(
              mark.getLoc(), mark->getResultTypes(),
              mapping.lookupOrDefault(mark.getSrc()));
          newMark->setAttrs(attrs.getDictionary(mark.getContext()));
          return;
        }
      }
    }
    if (auto preloadIf = dyn_cast<scf::IfOp>(&op)) {
      if (IntegerAttr preloadNumAttr = getPreloadNumAttr(preloadIf)) {
        bool run = shouldRunPeeledPreloadIf(preloadNumAttr.getInt(), peelCount,
                                            prologueIndex, epilogueIndex);
        Region &chosen =
            run ? preloadIf.getThenRegion() : preloadIf.getElseRegion();
        if (chosen.empty()) {
          assert(preloadIf.getNumResults() == 0 &&
                 "preload_num else missing for valued if");
          return;
        }
        for (Operation &inner : chosen.front().without_terminator())
          cloneOp(inner);
        auto yieldOp = cast<scf::YieldOp>(chosen.front().getTerminator());
        for (auto [res, operand] :
             llvm::zip(preloadIf.getResults(), yieldOp.getOperands()))
          mapping.map(res, mapping.lookupOrDefault(operand));
        return;
      }
    }
    // Deep-clone region ops, then lower multi-buffer inside nested bodies so
    // counters / multi-addr casts / multi_buffer marks do not survive peel.
    if (counterAlloca && (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op))) {
      Operation *cloned = rewriter.clone(op, mapping);
      if (auto nestFor = dyn_cast<scf::ForOp>(cloned))
        processMultiBufferInBlock(rewriter, nestFor.getBody(),
                                  /*sharedAlloca=*/{}, funcOp);
      else if (auto nestWhile = dyn_cast<scf::WhileOp>(cloned)) {
        processMultiBufferInBlock(rewriter, &nestWhile.getBefore().front(),
                                  /*sharedAlloca=*/{}, funcOp);
        processMultiBufferInBlock(rewriter, &nestWhile.getAfter().front(),
                                  /*sharedAlloca=*/{}, funcOp);
      } else if (auto nestIf = dyn_cast<scf::IfOp>(cloned)) {
        if (!nestIf.getThenRegion().empty())
          processMultiBufferInBlock(rewriter, &nestIf.getThenRegion().front(),
                                    counterAlloca, funcOp);
        if (!nestIf.getElseRegion().empty())
          processMultiBufferInBlock(rewriter, &nestIf.getElseRegion().front(),
                                    counterAlloca, funcOp);
      }
      return;
    }
    rewriter.clone(op, mapping);
  };
  for (Operation &op : forOp.getBody()->without_terminator())
    cloneOp(op);
  if (iterCounter)
    incrementCounter(rewriter, forOp.getLoc(), counterAlloca, iterCounter);
  SmallVector<Value> nextArgs;
  nextArgs.reserve(initArgs.size());
  for (auto [idx, yieldVal] : llvm::enumerate(forOp.getYieldedValues())) {
    if (!shouldUpdateMultiBufferYield(yieldVal, forOp, peelCount, prologueIndex,
                                      epilogueIndex)) {
      nextArgs.push_back(initArgs[idx]);
      continue;
    }
    nextArgs.push_back(mapping.lookupOrDefault(yieldVal));
  }
  initArgs = std::move(nextArgs);
}

static void peelPreloadLoop(scf::ForOp forOp, size_t maxPreloadNum) {
  // The peeled IVs advance by one. Keep other steps unchanged.
  APInt step;
  if (!matchPattern(forOp.getStep(), m_ConstantInt(&step)) || !step.isOne())
    return;

  IRRewriter rewriter(forOp.getContext());
  rewriter.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  int64_t peelCount = static_cast<int64_t>(maxPreloadNum);
  auto constant = [&](int64_t value) -> Value {
    return rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(lb.getType(), value));
  };
  // Both peeled ranges fit, including an empty middle loop at equality.
  Value tripCount = rewriter.createOrFold<arith::SubIOp>(loc, ub, lb);
  Value threshold = constant(2 * peelCount);
  Value canInline = rewriter.createOrFold<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, tripCount, threshold);
  auto ifOp = rewriter.create<scf::IfOp>(loc, forOp.getResultTypes(), canInline,
                                         /*withElseRegion=*/true);

  // Shared by then-path prologue/middle/epilogue. Else (fallback for) keeps
  // its own counters for LowerMultiBufferCounter — the branches are exclusive.
  Value counterAlloca = createSharedCounterAlloca(rewriter, forOp);

  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  SmallVector<Value> initArgs(forOp.getInitArgs());

  for (int64_t i = 0; i < peelCount; ++i)
    clonePeeledIteration(
        rewriter, forOp,
        rewriter.createOrFold<arith::AddIOp>(loc, lb, constant(i)), initArgs,
        peelCount, /*prologueIndex=*/i,
        /*epilogueIndex=*/std::nullopt, counterAlloca);
  Value middleLb =
      rewriter.createOrFold<arith::AddIOp>(loc, lb, constant(peelCount));
  Value middleUb =
      rewriter.createOrFold<arith::SubIOp>(loc, ub, constant(peelCount));
  auto middle = cast<scf::ForOp>(rewriter.clone(*forOp));
  middle->removeAttr(hivm::MaxPreloadNumAttr::name);
  middle.setLowerBound(middleLb);
  middle.setUpperBound(middleUb);
  middle.getInitArgsMutable().assign(initArgs);
  initArgs.assign(middle.getResults().begin(), middle.getResults().end());
  {
    OpBuilder::InsertionGuard guard(rewriter);
    SmallVector<scf::IfOp> middleGuards;
    middle.getBody()->walk([&](scf::IfOp guardIf) {
      if (getPreloadNumAttr(guardIf))
        middleGuards.push_back(guardIf);
    });
    for (scf::IfOp guardIf : llvm::reverse(middleGuards))
      replaceOpWithRegion(rewriter, guardIf, guardIf.getThenRegion());
    if (counterAlloca) {
      auto funcOp = forOp->getParentOfType<func::FuncOp>();
      processMultiBufferInBlock(rewriter, middle.getBody(), counterAlloca,
                                funcOp);
    }
    erasePreloadNumMarks(middle);
  }
  // Resume after the middle for for epilogue clones.
  rewriter.setInsertionPointAfter(middle);
  for (int64_t i = peelCount; i > 0; --i)
    clonePeeledIteration(
        rewriter, forOp,
        rewriter.createOrFold<arith::SubIOp>(loc, ub, constant(i)), initArgs,
        peelCount, /*prologueIndex=*/std::nullopt,
        /*epilogueIndex=*/i, counterAlloca);
  // Resultless scf.if builders already insert terminators.
  if (ifOp.getNumResults())
    rewriter.create<scf::YieldOp>(loc, initArgs);

  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
  auto fallback = cast<scf::ForOp>(rewriter.clone(*forOp));
  fallback->removeAttr(hivm::MaxPreloadNumAttr::name);
  erasePreloadNumMarks(fallback);
  if (ifOp.getNumResults())
    rewriter.create<scf::YieldOp>(loc, fallback.getResults());
  rewriter.replaceOp(forOp, ifOp.getResults());
}

void PeelPreloadPass::runOnOperation() {
  SmallVector<std::pair<scf::ForOp, size_t>> toPeel;
  getOperation()->walk([&](scf::ForOp forOp) {
    auto maxAttr =
        forOp->getAttrOfType<IntegerAttr>(hivm::MaxPreloadNumAttr::name);
    if (!maxAttr)
      return;
    int64_t maxPreloadNum = maxAttr.getInt();
    if (maxPreloadNum <= 0)
      return;
    toPeel.emplace_back(forOp, static_cast<size_t>(maxPreloadNum));
  });

  for (auto [forOp, maxPreloadNum] : toPeel) {
    LDBG("Peeling preload loop:\n" << forOp);
    peelPreloadLoop(forOp, maxPreloadNum);
  }

  if (toPeel.empty())
    return;

  PassManager pm(&getContext());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(getOperation())))
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createPeelPreloadPass() {
  return std::make_unique<PeelPreloadPass>();
}
