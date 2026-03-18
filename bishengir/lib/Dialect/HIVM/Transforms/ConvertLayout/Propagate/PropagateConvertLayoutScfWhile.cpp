//===------------------- PropagateConvertLayoutScfWhile.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// https://llvm.org/LICENSE.txt
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "hivm-propagate-convert-layout"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

SmallVector<Type> replaceTypeAt(TypeRange tys, uint32_t idx, Type newTy) {
  SmallVector<Type> out(tys.begin(), tys.end());
  out[idx] = newTy;
  return out;
}

scf::WhileOp createWhileWithExplicitBlockTypes(
    PatternRewriter &rewriter, scf::WhileOp oldWhile, ValueRange newInits,
    TypeRange newResultTypes, TypeRange newBeforeArgTypes,
    TypeRange newAfterArgTypes) {
  auto newWhile = rewriter.create<scf::WhileOp>(oldWhile.getLoc(),
                                                newResultTypes, newInits);

  // If builder already made placeholder blocks, clear them first.
  newWhile.getBefore().getBlocks().clear();
  newWhile.getAfter().getBlocks().clear();

  SmallVector<Location> beforeLocs(newBeforeArgTypes.size(), oldWhile.getLoc());
  SmallVector<Location> afterLocs(newAfterArgTypes.size(), oldWhile.getLoc());

  rewriter.createBlock(&newWhile.getBefore(), newWhile.getBefore().end(),
                       newBeforeArgTypes, beforeLocs);
  rewriter.createBlock(&newWhile.getAfter(), newWhile.getAfter().end(),
                       newAfterArgTypes, afterLocs);
  return newWhile;
}

void cloneOpsWithoutTerminator(PatternRewriter &rewriter, Block *src, Block *dst,
                               IRMapping &mapping, Operation *skipOp = nullptr) {
  rewriter.setInsertionPointToStart(dst);
  for (Operation &op : src->without_terminator()) {
    if (&op == skipOp)
      continue;
    rewriter.clone(op, mapping);
  }
}

Value mapOrSelf(IRMapping &m, Value v) { return m.lookupOrDefault(v); }

SmallVector<Value> buildConditionArgsWithOptionalOverride(
    scf::ConditionOp oldCond, IRMapping &mapping, int overrideIdx,
    Value overrideVal) {
  SmallVector<Value> args;
  args.reserve(oldCond.getArgs().size());
  for (uint32_t i = 0; i < oldCond.getArgs().size(); ++i) {
    if (static_cast<int>(i) == overrideIdx && overrideVal) {
      args.push_back(overrideVal);
    } else {
      args.push_back(mapOrSelf(mapping, oldCond.getArgs()[i]));
    }
  }
  return args;
}

SmallVector<Value> buildYieldOperandsWithOptionalOverride(
    scf::YieldOp oldYield, IRMapping &mapping, int overrideIdx,
    Value overrideVal) {
  SmallVector<Value> ys;
  ys.reserve(oldYield.getNumOperands());
  for (uint32_t i = 0; i < oldYield.getNumOperands(); ++i) {
    if (static_cast<int>(i) == overrideIdx && overrideVal) {
      ys.push_back(overrideVal);
    } else {
      ys.push_back(mapOrSelf(mapping, oldYield.getOperand(i)));
    }
  }
  return ys;
}

void replaceWhileResults(PatternRewriter &rewriter, scf::WhileOp oldWhile,
                         scf::WhileOp newWhile, uint32_t overrideIdx,
                         Value overrideVal) {
  SmallVector<Value> repl(newWhile.getResults().begin(),
                          newWhile.getResults().end());
  repl[overrideIdx] = overrideVal;
  rewriter.replaceOp(oldWhile, repl);
}

enum class WhileChannel {
  Init,   // inits <-> before args <-> after yield operands
  Result  // results <-> before cond args <-> after args
};

struct WhileParts {
  Block *before = nullptr;
  Block *after = nullptr;
  scf::ConditionOp cond;
  scf::YieldOp yield;
};

static FailureOr<WhileParts> getWhileParts(scf::WhileOp whileOp,
                                           PatternRewriter &rewriter,
                                           Operation *anchor,
                                           StringRef reasonPrefix = "") {
  WhileParts p;
  p.before = whileOp.getBeforeBody();
  p.after = whileOp.getAfterBody();
  p.cond = dyn_cast<scf::ConditionOp>(p.before->getTerminator());
  p.yield = dyn_cast<scf::YieldOp>(p.after->getTerminator());
  if (!p.cond || !p.yield)
    return rewriter.notifyMatchFailure(anchor, (reasonPrefix + "malformed while").str());
  return p;
}

static void mapBlockArgs(IRMapping &map, Block *oldB, Block *newB) {
  for (auto [oldA, newA] : llvm::zip(oldB->getArguments(), newB->getArguments()))
    map.map(oldA, newA);
}

struct RetypePlan {
  SmallVector<Value> inits;
  SmallVector<Type> resultTypes;
  SmallVector<Type> beforeArgTypes;
  SmallVector<Type> afterArgTypes;
};

// Retype exactly one channel slot.
static FailureOr<RetypePlan> buildRetypePlan(PatternRewriter &rewriter,
                                             Operation *anchor,
                                             scf::WhileOp whileOp,
                                             WhileParts &p,
                                             WhileChannel channel,
                                             uint32_t slot,
                                             Type newTy,
                                             Value newInitForInitChannel = nullptr) {
  RetypePlan plan;
  plan.inits.assign(whileOp.getInits().begin(), whileOp.getInits().end());
  plan.resultTypes.assign(whileOp.getResultTypes().begin(), whileOp.getResultTypes().end());

  for (BlockArgument a : p.before->getArguments())
    plan.beforeArgTypes.push_back(a.getType());
  for (BlockArgument a : p.after->getArguments())
    plan.afterArgTypes.push_back(a.getType());

  if (channel == WhileChannel::Init) {
    if (slot >= whileOp.getInits().size() || slot >= p.before->getNumArguments() ||
        slot >= p.yield.getNumOperands()) {
      return rewriter.notifyMatchFailure(anchor, "init-channel slot out of range");
    }
    if (!newInitForInitChannel)
      return rewriter.notifyMatchFailure(anchor, "missing new init value for init-channel");
    plan.inits[slot] = newInitForInitChannel;
    plan.beforeArgTypes[slot] = newTy;
  } else {
    if (slot >= whileOp.getResultTypes().size() || slot >= p.cond.getArgs().size() ||
        slot >= p.after->getNumArguments()) {
      return rewriter.notifyMatchFailure(anchor, "result-channel slot out of range");
    }
    plan.resultTypes[slot] = newTy;
    plan.afterArgTypes[slot] = newTy;
  }

  return plan;
}

static scf::WhileOp createRetypedWhile(PatternRewriter &rewriter,
                                       scf::WhileOp whileOp,
                                       const RetypePlan &plan) {
  rewriter.setInsertionPoint(whileOp);
  return createWhileWithExplicitBlockTypes(
      rewriter, whileOp, plan.inits, plan.resultTypes,
      plan.beforeArgTypes, plan.afterArgTypes);
}

// Safe clone helper: guarantees clone happens AFTER optional bridge op.
static void cloneWithoutTerminatorAfter(PatternRewriter &rewriter,
                                        Block *oldB, Block *newB,
                                        IRMapping &map,
                                        Operation *bridgeOrNull,
                                        Operation *skipOpOrNull) {
  if (bridgeOrNull)
    rewriter.setInsertionPointAfter(bridgeOrNull);
  else
    rewriter.setInsertionPointToStart(newB);
  for (Operation &op : oldB->without_terminator()) {
    if (&op == skipOpOrNull)
      continue;
    rewriter.clone(op, map);
  }
}

static SmallVector<Value> remapValues(IRMapping &map, ValueRange vals) {
  SmallVector<Value> out;
  out.reserve(vals.size());
  for (Value v : vals)
    out.push_back(map.lookupOrDefault(v));
  return out;
}


//===----------------------------------------------------------------------===//
// 1) Propagate UP from scf.while "after" region block arg
//===----------------------------------------------------------------------===//

struct PropagateConvertLayoutScfWhileAfterArgUp
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (!isPropagatingUp(convertOp))
      return rewriter.notifyMatchFailure(convertOp, "not propagating-up");

    auto afterArg = dyn_cast<BlockArgument>(convertOp.getSource());
    if (!afterArg)
      return rewriter.notifyMatchFailure(convertOp,
                                         "source is not block argument");

    auto *afterBlock = afterArg.getOwner();
    auto whileOp =
        dyn_cast<scf::WhileOp>(afterBlock->getParent()->getParentOp());
    if (!whileOp || afterBlock != whileOp.getAfterBody())
      return rewriter.notifyMatchFailure(convertOp,
                                         "source is not while.after arg");

    uint32_t k = afterArg.getArgNumber();

    auto partsOr = getWhileParts(whileOp, rewriter, convertOp);
    if (failed(partsOr))
      return failure();
    WhileParts p = *partsOr;

    auto planOr =
        buildRetypePlan(rewriter, convertOp, whileOp, p, WhileChannel::Result,
                        k, convertOp.getResult().getType());
    if (failed(planOr))
      return failure();
    scf::WhileOp newWhile = createRetypedWhile(rewriter, whileOp, *planOr);
    // Clone before region; condition arg k gets explicit up-convert.
    IRMapping beforeMap;
    mapBlockArgs(beforeMap, p.before, newWhile.getBeforeBody());
    cloneWithoutTerminatorAfter(rewriter, p.before, newWhile.getBeforeBody(),
                                beforeMap, /*bridge*/nullptr, /*skip*/nullptr);


    rewriter.setInsertionPointToEnd(newWhile.getBeforeBody());
    Value mappedCond = mapOrSelf(beforeMap, p.cond.getCondition());
    Value mappedK = mapOrSelf(beforeMap, p.cond.getArgs()[k]);
    Value upK = createConvertLayoutLike(rewriter, convertOp, mappedK);
    auto newCondArgs =
        buildConditionArgsWithOptionalOverride(p.cond, beforeMap, k, upK);
    rewriter.create<scf::ConditionOp>(p.cond.getLoc(), mappedCond, newCondArgs);

    // Clone after region; old-layout users get down(newArg_k_up).
    IRMapping afterMap;
    mapBlockArgs(afterMap, p.after, newWhile.getAfterBody());

    rewriter.setInsertionPointToStart(newWhile.getAfterBody());
    Value downArgK = createInverseConvertLayout(
        rewriter, convertOp, newWhile.getAfterBody()->getArgument(k));
    afterMap.map(p.after->getArgument(k), downArgK);
    afterMap.map(convertOp.getResult(), newWhile.getAfterBody()->getArgument(k));

    cloneWithoutTerminatorAfter(rewriter, p.after, newWhile.getAfterBody(),
                                afterMap, downArgK.getDefiningOp(),
                                convertOp.getOperation());

    rewriter.setInsertionPointToEnd(newWhile.getAfterBody());
    auto newYieldOps =
        buildYieldOperandsWithOptionalOverride(p.yield, afterMap, -1, Value());
    rewriter.create<scf::YieldOp>(p.yield.getLoc(), newYieldOps);

    // Old while result[k] remains old layout via down-convert.
    rewriter.setInsertionPointAfter(newWhile);
    Value rDown =
        createInverseConvertLayout(rewriter, convertOp, newWhile.getResult(k));
    replaceWhileResults(rewriter, whileOp, newWhile, k, rDown);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// 2) Propagate UP from scf.while result (result-channel only)
//===----------------------------------------------------------------------===//
struct PropagateConvertLayoutScfWhileResultUp
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (!isPropagatingUp(convertOp))
      return rewriter.notifyMatchFailure(convertOp, "not propagating-up");

    auto whileRes = dyn_cast<OpResult>(convertOp.getSource());
    if (!whileRes)
      return rewriter.notifyMatchFailure(convertOp, "source is not while result");
    auto whileOp = dyn_cast<scf::WhileOp>(whileRes.getOwner());
    if (!whileOp)
      return rewriter.notifyMatchFailure(convertOp, "source owner not scf.while");
    uint32_t k = whileRes.getResultNumber();

    auto partsOr = getWhileParts(whileOp, rewriter, convertOp);
    if (failed(partsOr)) return failure();
    WhileParts p = *partsOr;

    auto planOr = buildRetypePlan(rewriter, convertOp, whileOp, p,
                                  WhileChannel::Result, k,
                                  convertOp.getResult().getType());
    if (failed(planOr)) return failure();
    scf::WhileOp newWhile = createRetypedWhile(rewriter, whileOp, *planOr);

    // before
    IRMapping beforeMap;
    mapBlockArgs(beforeMap, p.before, newWhile.getBeforeBody());
    cloneWithoutTerminatorAfter(rewriter, p.before, newWhile.getBeforeBody(),
                                beforeMap, /*bridge*/nullptr, /*skip*/nullptr);

    rewriter.setInsertionPointToEnd(newWhile.getBeforeBody());
    Value mappedCond = mapOrSelf(beforeMap, p.cond.getCondition());
    SmallVector<Value> condArgs = remapValues(beforeMap, p.cond.getArgs());
    condArgs[k] = createConvertLayoutLike(rewriter, convertOp, condArgs[k]); // up
    rewriter.create<scf::ConditionOp>(p.cond.getLoc(), mappedCond, condArgs);

    // after
    IRMapping afterMap;
    mapBlockArgs(afterMap, p.after, newWhile.getAfterBody());

    rewriter.setInsertionPointToStart(newWhile.getAfterBody());
    Value downBridge = createInverseConvertLayout(
        rewriter, convertOp, newWhile.getAfterBody()->getArgument(k));
    afterMap.map(p.after->getArgument(k), downBridge);

    cloneWithoutTerminatorAfter(rewriter, p.after, newWhile.getAfterBody(),
                                afterMap, downBridge.getDefiningOp(), nullptr);

    rewriter.setInsertionPointToEnd(newWhile.getAfterBody());
    rewriter.create<scf::YieldOp>(p.yield.getLoc(),
                                  remapValues(afterMap, p.yield.getOperands()));

    rewriter.setInsertionPointAfter(newWhile);
    Value origK = createInverseConvertLayout(rewriter, convertOp, newWhile.getResult(k));

    rewriter.replaceOp(convertOp, newWhile.getResult(k));
    replaceWhileResults(rewriter, whileOp, newWhile, k, origK);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 3) Propagate DOWN from scf.while init arg (init-channel only)
//===----------------------------------------------------------------------===//

struct PropagateConvertLayoutScfWhileInitDown
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (isPropagatingUp(convertOp))
      return rewriter.notifyMatchFailure(convertOp, "not propagating-down");

    if (!convertOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(
          convertOp, "converted init must have one use (strict policy)");

    auto whileOp = dyn_cast<scf::WhileOp>(*convertOp.getResult().user_begin());
    if (!whileOp)
      return rewriter.notifyMatchFailure(convertOp,
                                         "result not used by scf.while");

    std::optional<uint32_t> iOpt;
    for (uint32_t i = 0; i < whileOp.getInits().size(); ++i) {
      if (whileOp.getInits()[i] == convertOp.getResult()) {
        iOpt = i;
        break;
      }
    }
    if (!iOpt)
      return rewriter.notifyMatchFailure(convertOp, "not a while init operand");
    uint32_t i = *iOpt;

    auto partsOr = getWhileParts(whileOp, rewriter, convertOp);
    if (failed(partsOr))
      return failure();
    WhileParts p = *partsOr;

    auto planOr =
        buildRetypePlan(rewriter, convertOp, whileOp, p, WhileChannel::Init, i,
                        convertOp.getSource().getType(), convertOp.getSource());
    if (failed(planOr))
      return failure();
    scf::WhileOp newWhile = createRetypedWhile(rewriter, whileOp, *planOr);

    // -------------------------------------------------------------------------
    // Before region
    // old before arg i expected "down" => synthesize down(newBeforeArg_i_up)
    // -------------------------------------------------------------------------
    IRMapping beforeMap;
    mapBlockArgs(beforeMap, p.before, newWhile.getBeforeBody());

    rewriter.setInsertionPointToStart(newWhile.getBeforeBody());
    Value beforeArgDown =
        createConvertLayoutLike(rewriter, convertOp,
                                newWhile.getBeforeBody()->getArgument(i));
    beforeMap.map(p.before->getArgument(i), beforeArgDown);

    // Ensure cloned users come AFTER bridge convert.
    rewriter.setInsertionPointAfter(beforeArgDown.getDefiningOp());
    cloneWithoutTerminatorAfter(rewriter, p.before, newWhile.getBeforeBody(),
                                beforeMap, beforeArgDown.getDefiningOp(),
                                /*skip*/nullptr);

    // Recreate condition unchanged in result-channel typing.
    rewriter.setInsertionPointToEnd(newWhile.getBeforeBody());
    Value mappedCond = mapOrSelf(beforeMap, p.cond.getCondition());
    rewriter.create<scf::ConditionOp>(
       p.cond.getLoc(), mappedCond, remapValues(beforeMap, p.cond.getArgs()));
    // -------------------------------------------------------------------------
    // After region
    // old yield operand i was "down" => yield "up" for new init-channel slot i
    // -------------------------------------------------------------------------
    IRMapping afterMap;
    mapBlockArgs(afterMap, p.after, newWhile.getAfterBody());
    cloneWithoutTerminatorAfter(rewriter, p.after, newWhile.getAfterBody(),
                                afterMap, /*bridge*/nullptr, /*skip*/nullptr);
    // Build new yield with slot i converted to up.
    rewriter.setInsertionPointToEnd(newWhile.getAfterBody());

    Value yi = mapOrSelf(afterMap, p.yield.getOperand(i));
    yi = createInverseConvertLayout(rewriter, convertOp, yi);
    auto newYieldOps =
        buildYieldOperandsWithOptionalOverride(p.yield, afterMap, i, yi);
    rewriter.create<scf::YieldOp>(p.yield.getLoc(), newYieldOps);

    // Replace old while with new while results (unchanged result-channel).
    rewriter.replaceOp(whileOp, newWhile.getResults());

    // Old init convert should now be dead.
    rewriter.eraseOp(convertOp);
    return success();
  }
};

struct PropagateConvertLayoutScfWhileBeforeArgUp
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (!isPropagatingUp(convertOp))
      return rewriter.notifyMatchFailure(convertOp, "not propagating-up");

    auto beforeArg = dyn_cast<BlockArgument>(convertOp.getSource());
    if (!beforeArg)
      return rewriter.notifyMatchFailure(convertOp,
                                         "source is not block argument");

    auto *beforeBlock = beforeArg.getOwner();
    auto whileOp =
        dyn_cast<scf::WhileOp>(beforeBlock->getParent()->getParentOp());
    if (!whileOp || beforeBlock != whileOp.getBeforeBody())
      return rewriter.notifyMatchFailure(convertOp,
                                         "source is not while.before arg");

    uint32_t i = beforeArg.getArgNumber();
    auto partsOr = getWhileParts(whileOp, rewriter, convertOp);
    if (failed(partsOr)) return failure();
    WhileParts p = *partsOr;

    rewriter.setInsertionPoint(whileOp);
    Value initUp = createConvertLayoutLike(rewriter, convertOp, whileOp.getInits()[i]);
    auto planOr = buildRetypePlan(rewriter, convertOp, whileOp, p,
                                  WhileChannel::Init, i,
                                  convertOp.getResult().getType(), initUp);
    if (failed(planOr)) return failure();
    scf::WhileOp newWhile = createRetypedWhile(rewriter, whileOp, *planOr);

    // ---- Before region ----
    // old before arg i (down) := down(new before arg i up)
    // old convert result (up) := new before arg i up
    IRMapping beforeMap;
    mapBlockArgs(beforeMap, p.before, newWhile.getBeforeBody());

    rewriter.setInsertionPointToStart(newWhile.getBeforeBody());
    Value beforeArgDown = createInverseConvertLayout(
        rewriter, convertOp, newWhile.getBeforeBody()->getArgument(i));
    beforeMap.map(p.before->getArgument(i), beforeArgDown);
    beforeMap.map(convertOp.getResult(), newWhile.getBeforeBody()->getArgument(i));

    // Clone after bridge so dominance is valid.
    cloneWithoutTerminatorAfter(rewriter, p.before, newWhile.getBeforeBody(),
                                beforeMap, beforeArgDown.getDefiningOp(),
                                convertOp.getOperation());

    // Recreate scf.condition unchanged on result-channel values.
    rewriter.setInsertionPointToEnd(newWhile.getBeforeBody());
    Value mappedCond = mapOrSelf(beforeMap, p.cond.getCondition());
    rewriter.create<scf::ConditionOp>(
        p.cond.getLoc(), mappedCond, remapValues(beforeMap, p.cond.getArgs()));
    // ---- After region ----
    // after args (result-channel) unchanged; yield slot i must become up.
    IRMapping afterMap;
    mapBlockArgs(afterMap, p.after, newWhile.getAfterBody());

    cloneWithoutTerminatorAfter(rewriter, p.after, newWhile.getAfterBody(),
                                afterMap, /*bridge*/nullptr, /*skip*/nullptr);

    rewriter.setInsertionPointToEnd(newWhile.getAfterBody());
    Value yi = mapOrSelf(afterMap, p.yield.getOperand(i));
    yi = createConvertLayoutLike(rewriter, convertOp, yi);
    auto newYieldOps =
        buildYieldOperandsWithOptionalOverride(p.yield, afterMap, i, yi);
    rewriter.create<scf::YieldOp>(p.yield.getLoc(), newYieldOps);
    // while results unchanged for this transform.
    rewriter.replaceOp(whileOp, newWhile.getResults());
    return success();
  }
};

} // namespace

void mlir::hivm::populateConvertLayoutScfWhile(RewritePatternSet &patterns,
                                               MLIRContext *context) {
  patterns.add<
      PropagateConvertLayoutScfWhileResultUp,
      PropagateConvertLayoutScfWhileAfterArgUp,
      PropagateConvertLayoutScfWhileBeforeArgUp,
      PropagateConvertLayoutScfWhileInitDown>(context);
}