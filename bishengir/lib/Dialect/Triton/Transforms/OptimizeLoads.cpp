//===------------------------OptimizeLoads.cpp ----------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt  for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"

#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#define DEBUG_TYPE "optimize-loads"

namespace bishengir {
namespace triton {
#define GEN_PASS_DEF_OPTIMIZELOADS
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::arith;

/// collected info for a load found under a select-tree
struct CollectedLoadInfo {
  triton::LoadOp loadOp;
  // the op that led us to this load
  Operation *reachingUser;
  // chain of (condValue, isNegated) going from outer-most select to inner-most
  // isNegated==true means this path requires ~condValue
  SmallVector<std::pair<Value, bool>> condChain;
};

/// step through broadcast/splat/expand_dims to return the narrowest defining
/// operand we can find
static Value stepThroughBroadcastIfAny(Value v) {
  if (!v)
    return v;
  Operation *def = v.getDefiningOp();
  while (def) {
    // handle common textual op names used in this repo/dialect
    if (isa<BroadcastOp, ExpandDimsOp, SplatOp>(def)) {
      if (def->getNumOperands() > 0) {
        v = def->getOperand(0);
        def = v.getDefiningOp();
        continue;
      }
    }
    break;
  }
  return v;
}

/// check if src type can be broadcast to tgt type
static bool isBroadcastableTo(Type srcTy, Type tgtTy) {
  auto srcRT = dyn_cast<RankedTensorType>(srcTy);
  auto tgtRT = dyn_cast<RankedTensorType>(tgtTy);
  if (!srcRT || !tgtRT)
    return false;
  if (!srcRT.hasStaticShape() || !tgtRT.hasStaticShape())
    return false;
  if (srcRT.getRank() != tgtRT.getRank())
    return false;
  for (int i = 0, e = srcRT.getRank(); i < e; ++i) {
    int64_t sdim = srcRT.getDimSize(i);
    int64_t tdim = tgtRT.getDimSize(i);
    // src dimension must be 1 or equal to target dimension
    if (sdim != 1 && sdim != tdim)
      return false;
  }
  return true;
}

/// number of elements for ranked static tensor
static uint64_t numElementsOfType(Type t) {
  if (auto rt = dyn_cast<RankedTensorType>(t)) {
    if (!rt.hasStaticShape())
      return UINT64_MAX;
    return rt.getNumElements();
  }
  return UINT64_MAX;
}

/// create a tt.broadcast from `src` to `targetType`
static Value ensureBroadcast(Value src, Type targetType, Location loc,
                             RewriterBase &rewriter) {
  if (!src)
    return src;
  if (src.getType() == targetType)
    return src;

  // check if target shape is larger than source shape
  uint64_t srcNumElems = numElementsOfType(src.getType());
  uint64_t targetNumElems = numElementsOfType(targetType);

  if (targetNumElems <= srcNumElems) {
    // target is not bigger, no need to broadcast
    return src;
  }

  // target is bigger, proceed with broadcast
  auto op = rewriter.create<BroadcastOp>(loc, targetType, src);
  return op->getResult(0);
}

/// create a tensor of same shape of all one
// TODO: only supports ranked static right now, maybe need to improve if it is
// not
static Value createOnesConst(Type t, Location loc, RewriterBase &rewriter) {
  if (auto rt = dyn_cast<RankedTensorType>(t)) {
    Type elt = rt.getElementType();
    auto intElt = dyn_cast<IntegerType>(elt);
    if (!intElt)
      return Value();
    APInt one(intElt.getWidth(), 1);
    SmallVector<APInt> vals(rt.getNumElements(), one);
    DenseElementsAttr attr = DenseElementsAttr::get(rt, ArrayRef<APInt>(vals));
    auto c = rewriter.create<ConstantOp>(loc, rt, attr);
    return c.getResult();
  }
  if (auto intTy = dyn_cast<IntegerType>(t)) {
    auto c = rewriter.create<ConstantIntOp>(loc, /*value*/ 1,
                                            /*width*/ intTy.getWidth());
    return c.getResult();
  }
  return Value();
}

/// create a tt.splat from scalar src to targetType .
static Value createSplatFromScalar(Value src, Type targetType, Location loc,
                                   RewriterBase &rewriter) {
  if (!src)
    return src;
  auto rt = dyn_cast<RankedTensorType>(targetType);
  if (!rt)
    return Value();
  auto op = rewriter.create<SplatOp>(loc, targetType, src);
  return op->getResult(0);
}

/// ensure src has type targetType:
/// scalar -> create tt.splat(targetType)
/// otherwise create tt.broadcast(src -> targetType)
static Value ensureValueOfType(Value src, Type targetType, Location loc,
                               RewriterBase &rewriter) {
  if (!src || !targetType)
    return src;
  if (src.getType() == targetType)
    return src;

  // scalar -> splat
  if (src.getType().isa<IntegerType, FloatType, IndexType>()) {
    return createSplatFromScalar(src, targetType, loc, rewriter);
  }

  // try stepping through to find scalar inner
  Value inner = stepThroughBroadcastIfAny(src);
  if (inner && inner.getType().isa<IntegerType>()) {
    return createSplatFromScalar(inner, targetType, loc, rewriter);
  }

  // fallback to broadcast
  return ensureBroadcast(src, targetType, loc, rewriter);
}

/// produce `notVal = xor(val, ones)` where `ones` is an all-ones constant.
static Value createNotBool(Value val, RewriterBase &rewriter) {
  if (!val)
    return val;
  Location loc = val.getDefiningOp() ? val.getDefiningOp()->getLoc()
                                     : rewriter.getUnknownLoc();
  Type t = val.getType();
  Value ones = createOnesConst(t, loc, rewriter);
  if (!ones)
    return val;
  auto x = rewriter.create<arith::XOrIOp>(loc, t, val, ones);
  return x.getResult();
}

/// bitwise AND of a and b
static Value createAnd(Value a, Value b, Type finalTarget,
                       RewriterBase &rewriter) {
  if (!a && !b)
    return Value();
  if (!a)
    return b;
  if (!b)
    return a;
  Location loc = a.getDefiningOp()
                     ? a.getDefiningOp()->getLoc()
                     : (b.getDefiningOp() ? b.getDefiningOp()->getLoc()
                                          : rewriter.getUnknownLoc());
  // handle broadcasting: broadcast smaller shape to larger shape
  Type aType = a.getType();
  Type bType = b.getType();

  if (aType != bType) {
    // determine which is the "larger" shape for broadcasting purposes
    uint64_t aNumElems = numElementsOfType(aType);
    uint64_t bNumElems = numElementsOfType(bType);

    Value aToUse = a;
    Value bToUse = b;

    if (aNumElems < bNumElems) {
      if (isBroadcastableTo(a.getType(), bType))
        aToUse = ensureValueOfType(a, bType, loc, rewriter);
      else if (isBroadcastableTo(a.getType(), finalTarget) &&
               isBroadcastableTo(b.getType(), finalTarget)) {
        aToUse = ensureValueOfType(a, finalTarget, loc, rewriter);
        bToUse = ensureValueOfType(b, finalTarget, loc, rewriter);
      }
    } else if (bNumElems < aNumElems) {
      if (isBroadcastableTo(b.getType(), aType))
        bToUse = ensureValueOfType(b, aType, loc, rewriter);
      else if (isBroadcastableTo(a.getType(), finalTarget) &&
               isBroadcastableTo(b.getType(), finalTarget)) {
        aToUse = ensureValueOfType(a, finalTarget, loc, rewriter);
        bToUse = ensureValueOfType(b, finalTarget, loc, rewriter);
      }
    }

    if (!aToUse || !bToUse)
      return Value();

    auto andOp =
        rewriter.create<arith::AndIOp>(loc, aToUse.getType(), aToUse, bToUse);
    return andOp.getResult();
  }

  auto andOp = rewriter.create<arith::AndIOp>(loc, a.getType(), a, b);
  return andOp.getResult();
}

/// build combined mask by handling shapes dynamically during combination
/// instead of using a single target type

static Value buildCombinedMask(CollectedLoadInfo &info,
                               RewriterBase &rewriter) {
  Location loc = info.loadOp.getLoc();
  Value origLoadMask = info.loadOp.getMask();

  Type correctType;
  if (origLoadMask && origLoadMask.getType()) {
    correctType = origLoadMask.getType();
  } else {
    // try to derive shape from the pointer
    Value ptr = info.loadOp.getPtr();
    if (ptr) {
      Type pty = ptr.getType();
      auto *ctx = rewriter.getContext();
      Type i1 = IntegerType::get(ctx, 1);

      if (auto rt = dyn_cast<RankedTensorType>(pty)) {
        ArrayRef<int64_t> shape = rt.getShape();
        correctType = RankedTensorType::get(shape, i1);
      }
    }
  }

  // determine pointer element count (if possible). If unknown -> UINT64_MAX.
  uint64_t ptrElemCount = UINT64_MAX;
  if (Value ptr = info.loadOp.getPtr()) {
    Type pty = ptr.getType();
    if (auto rt = dyn_cast<RankedTensorType>(pty)) {
      if (rt.hasStaticShape())
        ptrElemCount = numElementsOfType(rt);
    }
  }

  Type targetType = Type();
  uint64_t bestNum = UINT64_MAX;
  auto consider = [&](Type t) {
    if (!t)
      return;
    uint64_t ne = numElementsOfType(t);
    if (ne < bestNum) {
      bestNum = ne;
      targetType = t;
    }
  };

  for (auto &pr : info.condChain) {
    Value cond = pr.first;
    if (!cond)
      continue;
    Value stepped = stepThroughBroadcastIfAny(cond);
    if (stepped && stepped.getType())
      consider(stepped.getType());
    if (cond.getType())
      consider(cond.getType());
  }

  if (origLoadMask && origLoadMask.getType()) {
    Value stepped = stepThroughBroadcastIfAny(origLoadMask);
    if (stepped && stepped.getType())
      consider(stepped.getType());
    consider(origLoadMask.getType());
  }

  if (!targetType && origLoadMask)
    targetType = origLoadMask.getType();
  if (!targetType)
    return Value();

  // build chainMask at targetType
  Value chainMask;
  for (auto &pr : info.condChain) {
    Value cond = pr.first;
    bool isNeg = pr.second;
    if (!cond)
      continue;

    // determine the non-broadcast (real) type to compare against ptr shape.
    Value stepped = stepThroughBroadcastIfAny(cond);
    Type realType =
        (stepped && stepped.getType()) ? stepped.getType() : cond.getType();

    // if realType is known and pointer element count is known, skip this
    // condition if the mask's element count is strictly greater than the ptr.
    if (realType && ptrElemCount != UINT64_MAX) {
      uint64_t maskElems = numElementsOfType(realType);
      if (maskElems != UINT64_MAX && maskElems > ptrElemCount) {
        LLVM_DEBUG({
          llvm::dbgs() << "buildCombinedMask: skipping cond (maskElems="
                       << maskElems << " > ptrElems=" << ptrElemCount << "): ";
          cond.print(llvm::dbgs());
          llvm::dbgs() << "\n";
        });
        continue;
      }
    }

    Value toUse = stepped ? stepped : cond;

    // ensure toUse materialized at targetType
    if (toUse.getType() != targetType)
      toUse = ensureValueOfType(toUse, targetType, loc, rewriter);

    if (isNeg)
      toUse = createNotBool(toUse, rewriter);

    chainMask = createAnd(chainMask, toUse, correctType, rewriter);
  }

  // AND with original load mask converted to targetType (if any)
  Value origMaskNarrow = origLoadMask;
  if (origLoadMask && origLoadMask.getType() != targetType) {
    Value stepped = stepThroughBroadcastIfAny(origLoadMask);
    Value toUse = stepped ? stepped : origLoadMask;
    origMaskNarrow = ensureValueOfType(toUse, targetType, loc, rewriter);
  }

  Value finalMaskNarrow =
      createAnd(chainMask, origMaskNarrow, correctType, rewriter);

  // if we computed a correctType and finalMaskNarrow doesn't match it,
  // broadcast.
  if (finalMaskNarrow && correctType &&
      finalMaskNarrow.getType() != correctType) {
    finalMaskNarrow =
        ensureBroadcast(finalMaskNarrow, correctType, loc, rewriter);
  }

  return finalMaskNarrow;
}

/// dfs go from store to load, keeping track select
static void collectLoads(Value val, Operation *userOp,
                         SmallVectorImpl<CollectedLoadInfo> &out,
                         SmallVectorImpl<std::pair<Value, bool>> &currentChain,
                         llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (!val)
    return;

  // if value is a block argument, it has no defining op to descend from.
  Operation *def = val.getDefiningOp();
  if (!def)
    return;

  // avoid revisiting same op (handles DAGs / cycles)
  if (visited.contains(def))
    return;
  visited.insert(def);

  // if this value is coming from an arith.select, descend the select tree
  if (auto sel = dyn_cast<arith::SelectOp>(def)) {
    Value cond = sel.getCondition();
    // true branch (cond)
    currentChain.push_back({cond, false});
    collectLoads(sel.getTrueValue(), sel, out, currentChain, visited);
    currentChain.pop_back();

    // false branch (~cond)
    currentChain.push_back({cond, true});
    collectLoads(sel.getFalseValue(), sel, out, currentChain, visited);
    currentChain.pop_back();
    return;
  }

  // if this value *is* a load, collect it with the current condChain
  if (auto ld = dyn_cast<triton::LoadOp>(def)) {
    CollectedLoadInfo info;
    info.loadOp = ld;
    info.reachingUser = userOp;
    info.condChain.assign(currentChain.begin(), currentChain.end());
    out.push_back(std::move(info));
    LLVM_DEBUG({
      llvm::dbgs() << "collectLoads: found load -> ";
      ld.print(llvm::dbgs());
      llvm::dbgs() << "\n  condChain: [";
      for (auto &p : out.back().condChain) {
        if (p.second)
          llvm::dbgs() << "~";
        if (p.first)
          p.first.print(llvm::dbgs());
        else
          llvm::dbgs() << "<null>";
        llvm::dbgs() << ", ";
      }
      llvm::dbgs() << "]\n";
    });
    return;
  }

  // not a select or load, keep going up
  for (Value opnd : def->getOperands()) {
    collectLoads(opnd, def, out, currentChain, visited);
  }
}

static void refineLoads(SmallVectorImpl<CollectedLoadInfo> &loadInfo) {
  LLVM_DEBUG(llvm::dbgs() << "=== Starting load refinement ===\n");

  // set the max number of mask to use in the chain (-1) means all
  static constexpr int GLOBAL_MASK_BUDGET = -1;

  // filter out conditions defined after the load
  for (auto &info : loadInfo) {
    auto loadOp = info.loadOp;
    Operation *loadOperation = loadOp.getOperation();
    Block *loadBlock = loadOperation->getBlock();

    LLVM_DEBUG({
      llvm::dbgs() << "--- Processing load: ";
      loadOperation->print(llvm::dbgs());
      llvm::dbgs() << "\n";
      llvm::dbgs() << "Original condChain size: " << info.condChain.size()
                   << "\n";
    });

    // create a new filtered condChain
    SmallVector<std::pair<Value, bool>> filteredCondChain;
    bool removedAny = false;

    for (auto &condPair : info.condChain) {
      Value condValue = condPair.first;

      // step through any broadcast operations to get the underlying value
      Value underlyingValue = stepThroughBroadcastIfAny(condValue);
      Operation *definingOp = underlyingValue.getDefiningOp();

      LLVM_DEBUG({
        if (underlyingValue != condValue) {
          llvm::dbgs() << "  Stepped through broadcast: original value = "
                       << condValue
                       << ", underlying value = " << underlyingValue << "\n";
        }
      });

      // if there's no defining op (e.g., block argument), keep it
      if (!definingOp) {
        filteredCondChain.push_back(condPair);
        continue;
      }

      Block *definingBlock = definingOp->getBlock();

      // TODO: only consider same block right now, expand to other blocks
      if (definingBlock != loadBlock) {
        filteredCondChain.push_back(condPair);
        continue;
      }

      // check if the defining op appears BEFORE the load in the same block
      if (definingOp->isBeforeInBlock(loadOperation)) {
        filteredCondChain.push_back(condPair);
      } else {
        LLVM_DEBUG({
          llvm::dbgs() << "  Removing condition defined after load: ";
          definingOp->print(llvm::dbgs());
          llvm::dbgs() << "\n";
          if (underlyingValue != condValue) {
            llvm::dbgs() << "    (after stepping through broadcast)\n";
          }
        });
        removedAny = true;
      }
    }

    LLVM_DEBUG({
      if (removedAny) {
        llvm::dbgs() << "  Removed "
                     << (info.condChain.size() - filteredCondChain.size())
                     << " conditions defined after load\n";
      }
      llvm::dbgs() << "New condChain size: " << filteredCondChain.size()
                   << "\n";
    });

    info.condChain = std::move(filteredCondChain);
  }
  if (GLOBAL_MASK_BUDGET >= 0) {
    // find the load with the longest (filtered) condChain and pick the first K
    // entries from its chain (these are the global mask candidates).
    int longestIdx = -1;
    size_t longestSize = 0;
    for (int i = 0, e = loadInfo.size(); i < e; ++i) {
      if (loadInfo[i].condChain.size() > longestSize) {
        longestSize = loadInfo[i].condChain.size();
        longestIdx = i;
      }
    }

    SmallVector<Value> globalMaskUnderlyingValues;
    SmallVector<std::pair<Value, bool>> globalMaskPairs;
    if (longestIdx != -1 && longestSize > 0) {
      auto &longestChain = loadInfo[longestIdx].condChain;
      size_t take = std::min<size_t>(GLOBAL_MASK_BUDGET, longestChain.size());
      globalMaskPairs.reserve(take);
      globalMaskUnderlyingValues.reserve(take);

      for (size_t i = 0; i < take; ++i) {
        Value condValue = longestChain[i].first;
        Value underlying = stepThroughBroadcastIfAny(condValue);
        globalMaskPairs.push_back(longestChain[i]);
        globalMaskUnderlyingValues.push_back(underlying);
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Selected global mask budget = " << GLOBAL_MASK_BUDGET
                   << ", longest chain size = " << longestSize << "\n";
      llvm::dbgs() << "Global masks chosen (in order):\n";
      for (auto &p : globalMaskPairs) {
        llvm::dbgs() << "  pair: ";
        p.first.print(llvm::dbgs());
        llvm::dbgs() << " (negated=" << (p.second ? "true" : "false") << ")";
        llvm::dbgs() << "\n";
        Value u = stepThroughBroadcastIfAny(p.first);
        if (u != p.first) {
          llvm::dbgs() << "    underlying: ";
          u.print(llvm::dbgs());
          llvm::dbgs() << "\n";
        }
      }
    });

    // build a quick lookup set for global underlying values for membership
    // tests.
    llvm::SmallPtrSet<Value, 8> globalUnderlyingSet;
    for (auto &v : globalMaskUnderlyingValues)
      if (v)
        globalUnderlyingSet.insert(v);

    // take X amount from start (global masks)
    for (auto &info : loadInfo) {
      // keep original ordering for matching; we will create the final newChain.
      SmallVector<std::pair<Value, bool>> original = info.condChain;
      SmallVector<std::pair<Value, bool>> finalChain;
      finalChain.reserve(original.size());

      // add global masks in the same order as globalMaskPairs if this load had
      // them.
      for (Value gUnderlying : globalMaskUnderlyingValues) {
        // find the first occurrence in the load's chain that matches this
        // underlying value.
        bool found = false;
        for (auto &p : original) {
          Value pUnderlying = stepThroughBroadcastIfAny(p.first);
          if (pUnderlying == gUnderlying) {
            // include using the load's original negation flag
            finalChain.push_back(p);
            found = true;
            break;
          }
        }
        if (!found) {
          // load doesn't have this global predicate -> skip
        }
      }

      // append the load's unique masks that are
      for (auto &p : original) {
        Value pUnderlying = stepThroughBroadcastIfAny(p.first);
        if (globalUnderlyingSet.count(pUnderlying))
          continue; // already included via global masks

        // keep only the true masks as "unique masks".
        if (!p.second) {
          finalChain.push_back(p);
        }
      }

      LLVM_DEBUG({
        llvm::dbgs() << "Rebuilt condChain for load: ";
        info.loadOp.getOperation()->print(llvm::dbgs());
        llvm::dbgs() << "\n";
        llvm::dbgs() << "  Old size: " << original.size()
                     << ", New size: " << finalChain.size() << "\n";
      });

      // update the condChain
      info.condChain = std::move(finalChain);
    }

    // dump all loads and their final condChains in the requested format
    LLVM_DEBUG({
      for (auto &info : loadInfo) {
        Operation *loadOperation = info.loadOp.getOperation();
        llvm::dbgs() << "collectLoads: found load -> ";
        llvm::dbgs() << *loadOperation;
        llvm::dbgs() << "\n";
        llvm::dbgs() << "  condChain: [";
        for (size_t i = 0; i < info.condChain.size(); ++i) {
          auto &p = info.condChain[i];
          Value condVal = p.first;
          bool isNeg = p.second;

          if (isNeg)
            llvm::dbgs() << "~";

          Value underlying = stepThroughBroadcastIfAny(condVal);
          Operation *defOp = underlying.getDefiningOp();
          if (defOp) {
            llvm::dbgs() << *defOp;
          } else {
            llvm::dbgs() << underlying;
          }

          if (i + 1 < info.condChain.size())
            llvm::dbgs() << ", ";
        }
        llvm::dbgs() << "]\n";
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "=== Finished load refinement ===\n");
  }
}

/// replace old load with new load using finalMask
triton::LoadOp buildNewMaskedLoad(triton::LoadOp oldLoad, Value finalMask,
                                      RewriterBase &rewriter) {
  NamedAttrList attrs(oldLoad->getAttrs());

  // Set operandSegmentSizes correctly based on the actual operands
  // Format should be: [pointer_count, mask_count, other_count]
  SmallVector<int32_t> segmentSizes{1, 1, (oldLoad.getOther() ? 1 : 0)};
  attrs.set("operandSegmentSizes", rewriter.getDenseI32ArrayAttr(segmentSizes));

  SmallVector<Value> operands = {oldLoad.getPtr(), finalMask};
  if (oldLoad.getOther())
    operands.push_back(oldLoad.getOther());

  auto newLoad = rewriter.create<triton::LoadOp>(oldLoad.getLoc(),
                                                 TypeRange{oldLoad.getType()},
                                                 operands, attrs.getAttrs());
  return newLoad;
}

static void optimizeLoadMasks(triton::StoreOp storeOp,
                              IRRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "work on " << storeOp << "\n");
  Value valueToStore = storeOp.getValue();

  SmallVector<CollectedLoadInfo> collected;
  SmallVector<std::pair<Value, bool>> chain;
  llvm::SmallPtrSet<Operation *, 32> visited;
  collectLoads(valueToStore, storeOp, collected, chain, visited);
  refineLoads(collected);
  if (collected.empty())
    return;

  for (auto &info : collected) {
    Operation *oldLoadOp = info.loadOp.getOperation();
    if (!oldLoadOp)
      continue;
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(oldLoadOp);

    Value finalMask = buildCombinedMask(info, rewriter);
    if (!finalMask)
      continue;

    auto newLoad = buildNewMaskedLoad(info.loadOp, finalMask, rewriter);
    rewriter.replaceOpUsesWithIf(
      info.loadOp,
      newLoad->getResults(),
      [&](OpOperand &use) {
        return use.getOwner() == info.reachingUser;
    });

    if (info.loadOp.use_empty())
      rewriter.eraseOp(info.loadOp);        
  }
  return;
}

// the logic is find the first user of the load and then move right above it
struct MoveLoadsAsLateAsPossiblePattern
    : public OpRewritePattern<triton::LoadOp> {
  MoveLoadsAsLateAsPossiblePattern(MLIRContext *context)
      : OpRewritePattern<triton::LoadOp>(context) {}

  LogicalResult matchAndRewrite(triton::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    // find the earliest user in the same block as the load
    Operation *earliestUser = nullptr;
    Block *loadBlock = loadOp->getBlock();

    for (auto* user : loadOp->getUsers()) {
      // TODO: we only consider users in the same block for simplicity right
      // now might need to expand
      if (user->getBlock() != loadBlock) {
        continue;
      }

      // if this is the first user we've seen, or it appears before the
      // current earliest user
      if (!earliestUser || user->isBeforeInBlock(earliestUser)) {
        earliestUser = user;
      }
    }

    if (!earliestUser) {
      return failure();
    }
    if (loadOp->getNextNode() == earliestUser) {
      return failure();
    }

    // move load operation right before the earliest user
    rewriter.moveOpBefore(loadOp, earliestUser);

    return success();
  }
};

class OptimizeLoadsPass : public impl::OptimizeLoadsBase<OptimizeLoadsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto module = getOperation();
    llvm::StringRef myPassName = this->getArgument();
    int digit = mlir::triton::util::getPassColumnDigit(module, myPassName);
    if (digit != 0) {
      RewritePatternSet patterns(context);
      patterns.add<MoveLoadsAsLateAsPossiblePattern>(context);
      auto success = applyPatternsAndFoldGreedily(module, std::move(patterns));
      LLVM_DEBUG(if (failed(success))
        llvm::dbgs() << "apply MoveLoadsAsLateAsPossiblePattern failed\n";);

      IRRewriter rewriter(module.getContext());
      module.walk([&rewriter](triton::StoreOp storeOp) {
        optimizeLoadMasks(storeOp, rewriter);
      });
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createOptimizeLoadsPass() {
  return std::make_unique<OptimizeLoadsPass>();
}

} // namespace triton
} // namespace bishengir
