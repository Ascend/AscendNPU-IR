//===-------------------- PropagateConvertLayoutScfIf.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
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
// Helper Functions
//===----------------------------------------------------------------------===//

static LogicalResult verifyIfWithElseAndResults(scf::IfOp ifOp,
                                                PatternRewriter &rewriter,
                                                Operation *anchor) {
  if (!ifOp.elseBlock())
    return rewriter.notifyMatchFailure(anchor, "scf.if has no else region");
  if (ifOp.getNumResults() == 0)
    return rewriter.notifyMatchFailure(anchor, "scf.if has no results");
  return success();
}

static bool equivalentConvertMeta(ConvertLayoutOp a, ConvertLayoutOp b) {
  // Strict check: all attributes must match.
  return a->getAttrs() == b->getAttrs();
}

static SmallVector<Value> buildYieldOperands(scf::YieldOp oldYield,
                                             IRMapping &mapping,
                                             int overrideIdx = -1,
                                             Value overrideValue = Value()) {
  SmallVector<Value> newOps;
  newOps.reserve(oldYield.getNumOperands());
  for (uint32_t i = 0; i < oldYield.getNumOperands(); ++i) {
    if (static_cast<int>(i) == overrideIdx && overrideValue) {
      newOps.push_back(overrideValue);
      continue;
    }
    newOps.push_back(mapping.lookupOrDefault(oldYield.getOperand(i)));
  }
  return newOps;
}

static void eraseAutoYieldIfPresent(PatternRewriter &rewriter, Block *block) {
  if (block && block->mightHaveTerminator())
    rewriter.eraseOp(block->getTerminator());
}

static void cloneBlockWithoutTerminator(PatternRewriter &rewriter,
                                        Block &oldBlock,
                                        IRMapping &mapping,
                                        Operation *skipOp = nullptr) {
  for (Operation &op : oldBlock.without_terminator()) {
    if (&op == skipOp)
      continue;
    rewriter.clone(op, mapping);
  }
}

//===----------------------------------------------------------------------===//
// Propagate DOWN from scf.if branch yields
//===----------------------------------------------------------------------===//
//
// Before:
//   %v = scf.if %cond -> (..., Tk_down, ...) {
//     ...
//     %a_down = convert_layout %a {down}
//     scf.yield ..., %a_down, ...
//   } else {
//     ...
//     %b_down = convert_layout %b {down}
//     scf.yield ..., %b_down, ...
//   }
//
// After:
//   %v_raw = scf.if %cond -> (..., Tk_raw, ...) {
//     ...
//     scf.yield ..., %a, ...
//   } else {
//     ...
//     scf.yield ..., %b, ...
//   }
//   %vk_down = convert_layout %v_raw#k {down}
//
struct PropagateConvertLayoutScfIfYieldDown
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (isPropagatingUp(convertOp))
      return rewriter.notifyMatchFailure(convertOp, "not propagating-down");

    if (!convertOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(convertOp,
                                         "yielded conversion must have one use");

    auto thisYield =
        dyn_cast<scf::YieldOp>(*convertOp.getResult().user_begin());
    if (!thisYield)
      return rewriter.notifyMatchFailure(convertOp,
                                         "convert result not used by scf.yield");

    auto ifOp = dyn_cast<scf::IfOp>(thisYield->getParentOp());
    if (!ifOp)
      return rewriter.notifyMatchFailure(convertOp, "yield is not in scf.if");

    if (failed(verifyIfWithElseAndResults(ifOp, rewriter, convertOp)))
      return failure();
      
    if (!ifOp.thenBlock() || !ifOp.elseBlock()) {
      return rewriter.notifyMatchFailure(convertOp, "scf.if block is null");
    }
    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());

    // Find yield index k in this branch.
    std::optional<uint32_t> kOpt;
    for (uint32_t i = 0; i < thisYield.getNumOperands(); ++i) {
      if (thisYield.getOperand(i) == convertOp.getResult()) {
        kOpt = i;
        break;
      }
    }
    if (!kOpt)
      return rewriter.notifyMatchFailure(convertOp, "not found in yield operands");
    uint32_t k = *kOpt;

    if (k >= ifOp.getNumResults())
      return rewriter.notifyMatchFailure(convertOp, "invalid result index");

    Value thenK = thenYield.getOperand(k);
    Value elseK = elseYield.getOperand(k);

    auto thenConv = thenK.getDefiningOp<ConvertLayoutOp>();
    auto elseConv = elseK.getDefiningOp<ConvertLayoutOp>();
    if (!thenConv || !elseConv)
      return rewriter.notifyMatchFailure(
          convertOp, "both branches must yield convert_layout at index");

    if (isPropagatingUp(thenConv) || isPropagatingUp(elseConv))
      return rewriter.notifyMatchFailure(convertOp,
                                         "branch conversion is not down");

    if (!thenConv.getResult().hasOneUse() || !elseConv.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(
          convertOp, "branch convert result has extra users");

    if (!equivalentConvertMeta(thenConv, elseConv))
      return rewriter.notifyMatchFailure(convertOp,
                                         "branch conversions metadata mismatch");

    if (thenConv.getSource().getType() != elseConv.getSource().getType())
      return rewriter.notifyMatchFailure(convertOp,
                                         "branch raw source type mismatch");

    // New if result types: result[k] becomes raw/source type.
    SmallVector<Type> newResultTypes(ifOp.getResultTypes().begin(),
                                     ifOp.getResultTypes().end());
    newResultTypes[k] = thenConv.getSource().getType();

    rewriter.setInsertionPoint(ifOp);
    auto newIfOp = rewriter.create<scf::IfOp>(
        ifOp.getLoc(), newResultTypes, ifOp.getCondition(), /*withElseRegion=*/true);

    eraseAutoYieldIfPresent(rewriter, newIfOp.thenBlock());
    eraseAutoYieldIfPresent(rewriter, newIfOp.elseBlock());

    // Clone THEN, replacing yield[k] with thenConv.source.
    {
      IRMapping mapping;
      rewriter.setInsertionPointToStart(newIfOp.thenBlock());
      cloneBlockWithoutTerminator(rewriter, *ifOp.thenBlock(), mapping,
                                  thenConv.getOperation());
      Value rawThenK = mapping.lookupOrDefault(thenConv.getSource());
      auto newThenYieldOps = buildYieldOperands(thenYield, mapping, k, rawThenK);
      rewriter.create<scf::YieldOp>(thenYield.getLoc(), newThenYieldOps);
    }

    // Clone ELSE, replacing yield[k] with elseConv.source.
    {
      IRMapping mapping;
      rewriter.setInsertionPointToStart(newIfOp.elseBlock());
      if (!ifOp.elseBlock()) {
        return rewriter.notifyMatchFailure(convertOp, "scf.if else block is null");
      }
      cloneBlockWithoutTerminator(rewriter, *ifOp.elseBlock(), mapping,
                                  elseConv.getOperation());
      Value rawElseK = mapping.lookupOrDefault(elseConv.getSource());
      auto newElseYieldOps = buildYieldOperands(elseYield, mapping, k, rawElseK);
      rewriter.create<scf::YieldOp>(elseYield.getLoc(), newElseYieldOps);
    }

    // Convert after merge.
    rewriter.setInsertionPointAfter(newIfOp);
    Value mergedDown =
        createConvertLayoutLike(rewriter, thenConv, newIfOp.getResult(k));

    SmallVector<Value> replacements(newIfOp.getResults().begin(),
                                    newIfOp.getResults().end());
    replacements[k] = mergedDown;
    rewriter.replaceOp(ifOp, replacements);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Propagate UP from scf.if result
//===----------------------------------------------------------------------===//
//
// Before:
//   %v = scf.if %cond -> (..., Tk, ...) {
//     ...
//     scf.yield ..., %yk, ...
//   } else {
//     ...
//     scf.yield ..., %zk, ...
//   }
//   %vk_up = convert_layout %v#k {up}
//
// After:
//   %v_new = scf.if %cond -> (..., Tk_up, ...) {
//     ...
//     %yk_up = convert_layout %yk {up}
//     scf.yield ..., %yk_up, ...
//   } else {
//     ...
//     %zk_up = convert_layout %zk {up}
//     scf.yield ..., %zk_up, ...
//   }
//   %vk_orig = convert_layout %v_new#k {down}
//
// Replacements:
//   - old if result #k users -> %vk_orig
//   - old convert op users   -> %v_new#k
//
struct PropagateConvertLayoutScfIfResultUp
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (!isPropagatingUp(convertOp))
      return rewriter.notifyMatchFailure(convertOp, "not propagating-up");

    auto ifRes = dyn_cast<OpResult>(convertOp.getSource());
    if (!ifRes)
      return rewriter.notifyMatchFailure(convertOp, "source is not OpResult");

    auto ifOp = dyn_cast<scf::IfOp>(ifRes.getOwner());
    if (!ifOp)
      return rewriter.notifyMatchFailure(convertOp, "source owner is not scf.if");

    if (failed(verifyIfWithElseAndResults(ifOp, rewriter, convertOp)))
      return failure();

    uint32_t k = ifRes.getResultNumber();
    if (!ifOp.thenBlock() || !ifOp.elseBlock()) {
      return rewriter.notifyMatchFailure(convertOp, "scf.if block is null");
    }
    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());

    // New if type at k becomes convert result type (up type).
    SmallVector<Type> newResultTypes(ifOp.getResultTypes().begin(),
                                     ifOp.getResultTypes().end());
    newResultTypes[k] = convertOp.getResult().getType();

    rewriter.setInsertionPoint(ifOp);
    auto newIfOp = rewriter.create<scf::IfOp>(
        ifOp.getLoc(), newResultTypes, ifOp.getCondition(), /*withElseRegion=*/true);

    eraseAutoYieldIfPresent(rewriter, newIfOp.thenBlock());
    eraseAutoYieldIfPresent(rewriter, newIfOp.elseBlock());

    // Clone THEN and convert yield[k] up.
    {
      IRMapping mapping;
      rewriter.setInsertionPointToStart(newIfOp.thenBlock());
      if (!ifOp.thenBlock()) {
        return rewriter.notifyMatchFailure(convertOp, "scf.if then block is null");
      }
      cloneBlockWithoutTerminator(rewriter, *ifOp.thenBlock(), mapping);
      Value mappedThenK = mapping.lookupOrDefault(thenYield.getOperand(k));
      Value thenUp =
          createConvertLayoutLike(rewriter, convertOp, mappedThenK);
      auto newThenYieldOps = buildYieldOperands(thenYield, mapping, k, thenUp);
      rewriter.create<scf::YieldOp>(thenYield.getLoc(), newThenYieldOps);
    }

    // Clone ELSE and convert yield[k] up.
    {
      IRMapping mapping;
      rewriter.setInsertionPointToStart(newIfOp.elseBlock());
      cloneBlockWithoutTerminator(rewriter, *ifOp.elseBlock(), mapping);
      Value mappedElseK = mapping.lookupOrDefault(elseYield.getOperand(k));
      Value elseUp =
          createConvertLayoutLike(rewriter, convertOp, mappedElseK);
      auto newElseYieldOps = buildYieldOperands(elseYield, mapping, k, elseUp);
      rewriter.create<scf::YieldOp>(elseYield.getLoc(), newElseYieldOps);
    }

    // Materialize down cast for old users of if result #k.
    rewriter.setInsertionPointAfter(newIfOp);
    Value kOrig =
        createInverseConvertLayout(rewriter, convertOp, newIfOp.getResult(k));

    SmallVector<Value> ifRepls(newIfOp.getResults().begin(),
                               newIfOp.getResults().end());
    ifRepls[k] = kOrig;
    rewriter.replaceOp(ifOp, ifRepls);

    // Old convert users now use the new if's converted result directly.
    rewriter.replaceOp(convertOp, newIfOp.getResult(k));
    return success();
  }
};

} // namespace

void mlir::hivm::populateConvertLayoutScfIf(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add<
      PropagateConvertLayoutScfIfYieldDown,
      PropagateConvertLayoutScfIfResultUp>(context);
}