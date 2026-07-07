//===- OptimizeLayouts.cpp ----------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt  for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"

#include "OptimizeLayoutsAnalysis.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#define DEBUG_TYPE "optimize-layouts"
namespace bishengir {
namespace triton {
#define GEN_PASS_DEF_OPTIMIZELAYOUTS
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

/// returns the encoding attribute for a RankedTensorType, or null attr.
static Attribute getEncodingAttr(RankedTensorType tt) {
  if (!tt)
    return Attribute();
  return tt.getEncoding();
}

/// return the printed string of an attribute.
static std::string attrToString(Attribute a) {
  if (!a)
    return std::string();
  std::string s;
  llvm::raw_string_ostream os(s);
  a.print(os);
  os.flush();
  return s;
}

/// try to build a new encoding Attribute by replacing occurrences of oldEncStr
/// with newEncStr in the text form of `encAttr`. Returns {} if parsing failed.
static Attribute replaceEncodingSubstring(Attribute encAttr,
                                          StringRef oldEncStr,
                                          StringRef newEncStr,
                                          MLIRContext *ctx) {
  std::string encText = attrToString(encAttr);
  if (encText.find(oldEncStr) == std::string::npos)
    return Attribute();

  // replace all occurrences of oldEncStr with newEncStr.
  std::string newText;
  newText.reserve(encText.size() + (newEncStr.size() - oldEncStr.size()) * 4);
  size_t pos = 0;
  while (true) {
    size_t found = encText.find(oldEncStr, pos);
    if (found == std::string::npos) {
      newText.append(encText.begin() + pos, encText.end());
      break;
    }
    newText.append(encText.begin() + pos, encText.begin() + found);
    newText.append(newEncStr.begin(), newEncStr.end());
    pos = found + oldEncStr.size();
  }

  // parse the new attribute string.
  Attribute parsed = mlir::parseAttribute(newText, ctx);
  return parsed;
}

static RankedTensorType maybeReplaceEncoding(RankedTensorType rtt,
                                             Attribute oldEnc, Attribute newEnc,
                                             MLIRContext *ctx) {
  Attribute enc = getEncodingAttr(rtt);
  if (!enc)
    return {};

  Attribute finalEnc;
  if (enc == oldEnc) {
    finalEnc = newEnc;
  } else {
    Attribute replaced = replaceEncodingSubstring(enc, attrToString(oldEnc),
                                                  attrToString(newEnc), ctx);
    if (replaced)
      finalEnc = replaced;
  }

  if (!finalEnc)
    return {};

  return RankedTensorType::get(rtt.getShape(), rtt.getElementType(), finalEnc);
}

// rebuild DenseElementsAttr with new type (if necessary)
static void maybeReplaceDenseAttr(arith::ConstantOp constOp, Attribute oldEnc,
                                  Attribute newEnc, MLIRContext *ctx) {
  Attribute valAttr = constOp.getValue();
  if (auto elems = dyn_cast<DenseElementsAttr>(valAttr)) {
    if (auto attrType = dyn_cast<RankedTensorType>(elems.getType())) {
      if (maybeReplaceEncoding(attrType, oldEnc, newEnc, ctx)) {
        auto newAttrType = maybeReplaceEncoding(attrType, oldEnc, newEnc, ctx);
        SmallVector<Attribute> values(elems.getValues<Attribute>().begin(),
                                      elems.getValues<Attribute>().end());
        DenseElementsAttr newElems =
            DenseElementsAttr::get(newAttrType, values);
        constOp->setAttr("value", newElems);
      }
    }
  }
}

/// Check if replacing oldEnc with newEnc would break operations that require
/// specific encodings (like expand_dims requiring SliceEncodingAttr)
static bool isSafeToReplaceEncoding(ModuleOp module, Attribute oldEnc,
                                    Attribute newEnc) {
  if (!module || !oldEnc || !newEnc || oldEnc == newEnc)
    return false;

  // Check if oldEnc is SliceEncodingAttr and newEnc is not
  bool oldIsSlice = mlir::isa<SliceEncodingAttr>(oldEnc);
  bool newIsSlice = mlir::isa<SliceEncodingAttr>(newEnc);

  // If we're removing a slice encoding, check if any expand_dims depends on it
  if (oldIsSlice && !newIsSlice) {
    bool foundExpandDimsUser = false;
    module.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        if (auto rtt = dyn_cast<RankedTensorType>(result.getType())) {
          if (rtt.getEncoding() == oldEnc) {
            // This result has the old encoding
            // Check if any user requires SliceEncoding
            for (Operation *user : result.getUsers()) {
              if (isa<triton::ExpandDimsOp>(user)) {
                LLVM_DEBUG(llvm::dbgs()
                           << "isSafeToReplaceEncoding: found expand_dims "
                           << "consuming value with SliceEncoding -> UNSAFE\n");
                foundExpandDimsUser = true;
                return WalkResult::interrupt();
              }
            }
          }
        }
      }

      return WalkResult::advance();
    });

    if (foundExpandDimsUser) {
      return false;
    }
  }
  return true;
}

static void replaceEncodingInModule(Operation *topLevel, Attribute oldEnc,
                                    Attribute newEnc) {
  if (!topLevel || !oldEnc || !newEnc || oldEnc == newEnc)
    return;

  ModuleOp module = dyn_cast<ModuleOp>(topLevel);
  if (!module)
    module = topLevel->getParentOfType<ModuleOp>();
  if (!module)
    return;

  MLIRContext *ctx = module.getContext();

  // walk all operations
  module.walk([&](Operation *op) {
    // update results
    for (Value r : op->getResults()) {
      if (auto rtt = dyn_cast<RankedTensorType>(r.getType())) {
        if (auto newType = maybeReplaceEncoding(rtt, oldEnc, newEnc, ctx))
          r.setType(newType);
      }
    }

    // update constants
    if (auto constOp = dyn_cast<arith::ConstantOp>(op))
      maybeReplaceDenseAttr(constOp, oldEnc, newEnc, ctx);

    // update operand block arguments
    for (Value v : op->getOperands()) {
      if (auto bbArg = dyn_cast<BlockArgument>(v)) {
        if (auto rtt = dyn_cast<RankedTensorType>(bbArg.getType())) {
          if (auto newType = maybeReplaceEncoding(rtt, oldEnc, newEnc, ctx))
            bbArg.setType(newType);
        }
      }
    }
  });

  // update all function/block arguments
  module.walk([&](FuncOp func) {
    for (Block &block : func.getBody()) {
      for (BlockArgument &arg : block.getArguments()) {
        if (auto rtt = dyn_cast<RankedTensorType>(arg.getType())) {
          if (auto newType = maybeReplaceEncoding(rtt, oldEnc, newEnc, ctx))
            arg.setType(newType);
        }
      }
    }
  });
}

static Attribute makeBlockedEncodingWithWarpOnAxis(Attribute encAttr,
                                                   unsigned axis,
                                                   MLIRContext *ctx) {
  if (!encAttr)
    return Attribute();

  if (auto blocked = dyn_cast<triton::BlockedEncodingAttr>(encAttr)) {
    auto sizePerThread = llvm::to_vector(blocked.getSizePerThread());
    auto threadsPerWarp = llvm::to_vector(blocked.getThreadsPerWarp());
    auto warpsPerCTA = llvm::to_vector(blocked.getWarpsPerCTA());
    auto order = blocked.getOrder();
    auto CTALayout = blocked.getCTALayout();

    // Todo: incrase to N-D right now only 2D
    if (threadsPerWarp.size() != 2 || warpsPerCTA.size() != 2 ||
        sizePerThread.size() != 2)
      return Attribute();

    unsigned otherAxis = 1 - axis;

    sizePerThread[axis] = 4;
    sizePerThread[otherAxis] = 1;

    threadsPerWarp[axis] = 32;
    threadsPerWarp[otherAxis] = 1;

    warpsPerCTA[axis] = 1;
    warpsPerCTA[otherAxis] = 32;

    return triton::BlockedEncodingAttr::get(
        ctx, ArrayRef<unsigned>(sizePerThread),
        ArrayRef<unsigned>(threadsPerWarp), ArrayRef<unsigned>(warpsPerCTA),
        order, CTALayout);
  }

  return Attribute();
}

static RankedTensorType getReduceResultType(RankedTensorType inputType,
                                            unsigned axis, MLIRContext *ctx) {
  auto shape = inputType.getShape();
  SmallVector<int64_t> newShape;
  for (size_t i = 0; i < shape.size(); ++i)
    if (i != axis)
      newShape.push_back(shape[i]);

  // change the block
  auto inputEnc =
      dyn_cast<triton::BlockedEncodingAttr>(inputType.getEncoding());
  if (!inputEnc)
    return RankedTensorType::get(newShape, inputType.getElementType());

  // create a slice layout for the reduced axis
  auto sliceLayout = triton::SliceEncodingAttr::get(ctx, axis, inputEnc);
  return RankedTensorType::get(newShape, inputType.getElementType(),
                               sliceLayout);
}

struct ReduceOpPattern : public OpRewritePattern<triton::ReduceOp> {
  MLIRContext *ctx;
  ReduceOpPattern(MLIRContext *context)
      : OpRewritePattern<triton::ReduceOp>(context), ctx(context) {}

  LogicalResult matchAndRewrite(triton::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    unsigned axis = op.getAxis();
    if (op->getNumOperands() == 0)
      return failure();

    Value src = op.getSrcs().front();
    auto srcType = dyn_cast<RankedTensorType>(src.getType());
    if (!srcType || !srcType.getEncoding())
      return failure();

    Attribute srcEnc = srcType.getEncoding();

    // create new encoding with warp on reduction axis
    Attribute newEnc = makeBlockedEncodingWithWarpOnAxis(srcEnc, axis, ctx);
    if (!newEnc || newEnc == srcEnc)
      return failure();

    Location loc = op.getLoc();

    // convert input to new encoding
    RankedTensorType newInputType = RankedTensorType::get(
        srcType.getShape(), srcType.getElementType(), newEnc);
    Value convertedInput =
        rewriter.create<ConvertLayoutOp>(loc, newInputType, src);

    // compute reduce result type (slice layout)
    RankedTensorType reduceResultType =
        getReduceResultType(newInputType, axis, ctx);

    // create new reduce op
    auto newReduce = rewriter.create<triton::ReduceOp>(
        loc, TypeRange{reduceResultType}, convertedInput, axis);

    // clone combine region if present
    if (!op.getCombineOp().empty()) {
      Region &newRegion = newReduce.getCombineOp();
      rewriter.cloneRegionBefore(op.getCombineOp(), newRegion,
                                 newRegion.begin());
    }

    op->getResult(0).replaceAllUsesWith(newReduce->getResult(0));
    rewriter.eraseOp(op);
    return success();
  }
};

// Propagate replacement oldEnc -> newEnc starting from
// `startConvert`'s result going downward. Stops at inner ConvertLayoutOp
// boundaries (but updates inner convert's source type if possible) and also
// updates function result types / block arguments when the chain reaches the
// end-of-function/block. This does in-place, text-style type updates using
// maybeReplaceEncoding / maybeReplaceDenseAttr.
static void replaceEncodingInModuleDownProp(Operation *topLevel,
                                            ConvertLayoutOp startConvert,
                                            Attribute oldEnc,
                                            Attribute newEnc) {
  if (!topLevel || !oldEnc || !newEnc || oldEnc == newEnc)
    return;

  ModuleOp module = dyn_cast<ModuleOp>(topLevel);
  if (!module)
    module = topLevel->getParentOfType<ModuleOp>();
  if (!module)
    return;

  MLIRContext *ctx = module.getContext();

  Value startVal = startConvert.getResult();

  // visited ops and chain values
  llvm::DenseSet<Operation *> visitedOps;
  llvm::DenseSet<Value> chainValues;
  chainValues.insert(startVal);

  // worklist: start from users of the start value
  SmallVector<Operation *> worklist;
  for (Operation *u : startVal.getUsers())
    worklist.push_back(u);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!visitedOps.insert(op).second)
      continue;

    // If we hit another ConvertLayoutOp, update its *source* type if it has
    // oldEnc, but do not descend beyond it (boundary).
    if (auto innerConv = dyn_cast<ConvertLayoutOp>(op)) {
      Value innerSrc = innerConv.getSrc();
      if (auto srcRT = dyn_cast<RankedTensorType>(innerSrc.getType())) {
        if (auto newType = maybeReplaceEncoding(srcRT, oldEnc, newEnc, ctx)) {
          innerSrc.setType(newType);
        }
      }
      // stop descent past this convert
      continue;
    }

    // Update this op's result types: oldEnc -> newEnc where applicable.
    for (Value res : op->getResults()) {
      if (auto resRT = dyn_cast<RankedTensorType>(res.getType())) {
        if (auto newType = maybeReplaceEncoding(resRT, oldEnc, newEnc, ctx)) {
          res.setType(newType);
        }
      }
      chainValues.insert(res);
    }

    // Update constant DenseElementsAttr values if present
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      maybeReplaceDenseAttr(constOp, oldEnc, newEnc, ctx);
    }

    // Add downstream users of results to the worklist
    for (Value res : op->getResults()) {
      for (Operation *user : res.getUsers())
        worklist.push_back(user);
    }
  }

  // If the chain reaches function returns, update those function result types.
  // Also update any block arguments that are part of the chain.
  module.walk([&](func::FuncOp func) {
    // Update function result types if any return operand is in chainValues.
    FunctionType fTy = func.getFunctionType();
    SmallVector<Type, 4> newResultTypes(fTy.getResults().begin(),
                                        fTy.getResults().end());
    bool funcTypeChanged = false;

    // Find return ops inside the function and check operands
    func.walk([&](func::ReturnOp ret) {
      for (unsigned i = 0; i < ret.getNumOperands(); ++i) {
        Value retOperand = ret.getOperand(i);
        if (chainValues.count(retOperand)) {
          if (i < newResultTypes.size()) {
            if (auto rtt = dyn_cast<RankedTensorType>(newResultTypes[i])) {
              if (auto replaced =
                      maybeReplaceEncoding(rtt, oldEnc, newEnc, ctx)) {
                newResultTypes[i] = replaced;
                funcTypeChanged = true;
              }
            }
          }
        }
      }
    });

    if (funcTypeChanged) {
      SmallVector<Type, 8> inputTypes(fTy.getInputs().begin(),
                                      fTy.getInputs().end());
      FunctionType newFT = FunctionType::get(ctx, inputTypes, newResultTypes);
      func.setType(newFT);
    }

    // Update block arguments inside the function if they belong to the chain.
    for (Block &block : func.getBody()) {
      for (BlockArgument &arg : block.getArguments()) {
        if (!chainValues.count(arg))
          continue;
        if (auto argRT = dyn_cast<RankedTensorType>(arg.getType())) {
          if (auto replaced =
                  maybeReplaceEncoding(argRT, oldEnc, newEnc, ctx)) {
            arg.setType(replaced);
          }
        }
      }
    }
  });
}

/// Check whether applying newEnc to values used by op is compatible
static bool isLayoutCompatibleToOp(Operation *op, Attribute newEnc) {
  if (!op || !newEnc)
    return false;
  // disallow propagation through split and join because these ops change how
  // multiple inputs/outputs mao to logical axes and are generally
  // layout-sensitive
  if (isa<triton::SplitOp, triton::JoinOp, triton::TransOp>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Split/Join/TransOp are layout-sensitive");
    return false;
  }
  return true;
}

// recursively check whether we can propagate the target encoding upwards from
// start
static bool canPropagateUpFrom(Value start, Attribute targetEnc,
                               const llvm::DenseSet<Value> &srcChainValues) {
  // DFS over producer values
  SmallPtrSet<Value, 32> visited;
  SmallVector<Value, 32> stack;
  stack.push_back(start);

  while (!stack.empty()) {
    Value cur = stack.pop_back_val();
    if (!visited.insert(cur).second)
      continue;

    // if we reach the chain root it is fine to stop
    if (srcChainValues.count(cur))
      continue;

    Operation *def = cur.getDefiningOp();

    if (!def)
      return false;

    // The producer itself must be compatible with the new encoding
    if (!isLayoutCompatibleToOp(def, targetEnc))
      return false;

    for (Value opd : def->getOperands())
      if (!visited.count(opd))
        stack.push_back(opd);
  }

  return true;
}

/*
This pattern remove convert layout by propagate down layout
*/

struct RemoveConvertLayoutByPropagatingDownPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  MLIRContext *ctx;
  RemoveConvertLayoutByPropagatingDownPattern(MLIRContext *context)
      : OpRewritePattern<ConvertLayoutOp>(context, /*benefit=*/1),
        ctx(context) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {

    auto actionAttr = convertOp->getAttrOfType<StringAttr>("layout.action");
    if (!actionAttr)
      return failure();

    StringRef action = actionAttr.getValue();

    // ensure convert source comes from a ReduceOp
    Value srcVal = convertOp.getSrc();
    Operation *srcOp = srcVal.getDefiningOp();
    auto srcType = dyn_cast<RankedTensorType>(srcVal.getType());
    // get the correct layout
    Attribute srcEnc = srcType ? srcType.getEncoding() : Attribute();
    Value dstVal = convertOp.getResult();
    auto dstType = dyn_cast<RankedTensorType>(dstVal.getType());
    Attribute dstEnc = dstType ? dstType.getEncoding() : Attribute();

    // Don't propagate linear layout for now
    if (isa<LinearEncodingAttr>(srcEnc))
      return failure();

    if (action != "propagate_down")
      return failure();

    if (!srcEnc)
      return failure();

    Attribute targetEnc;
    int dstRank = dstType.getRank();

    if (auto sliceEnc = dyn_cast<SliceEncodingAttr>(srcEnc)) {
      if (auto parentBlocked =
              dyn_cast<BlockedEncodingAttr>(sliceEnc.getParent())) {
        unsigned parentRank = parentBlocked.getSizePerThread().size();
        if (static_cast<int>(parentRank) == dstRank) {
          // Safe: parent blocked matches destination rank
          targetEnc = parentBlocked;
        } else {
          // Rank mismatch -> slice
          targetEnc = srcEnc;
        }
      } else {
        // Non-blocked parent -> keep slice
        targetEnc = srcEnc;
      }
    } else if (isa<LinearEncodingAttr>(srcEnc) ||
               isa<BlockedEncodingAttr>(srcEnc)) {
      targetEnc = srcEnc;
    } else {
      // Unsupported encoding
      return failure();
    }

    if (!dstEnc)
      return failure();

    // TODO: Slice to slice layout is not supported. To support we must
    // propagate slice to slice but also the parent block to parent block. Take
    // expandDimOp for example
    if (isa<SliceEncodingAttr>(srcEnc) && isa<SliceEncodingAttr>(dstEnc)) {
      return failure();
    }

    // collect all user ops from dstVal
    SetVector<Operation *> affectedOps;
    SmallVector<Value> worklist;
    worklist.push_back(dstVal);
    while (!worklist.empty()) {
      Value cur = worklist.pop_back_val();
      for (Operation *user : cur.getUsers()) {
        if (!affectedOps.count(user)) {
          affectedOps.insert(user);
          for (Value r : user->getResults()) {
            if (isa<RankedTensorType>(r.getType())) {
              worklist.push_back(r);
            }
          }
        }
      }
    }
    SmallVector<Operation *> orderedOps(affectedOps.begin(), affectedOps.end());
    llvm::sort(orderedOps, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });

    // build srcChainValues set (values derived from the reduce's src/dst).
    // We treat srcVal and dstVal as chain roots.
    llvm::DenseSet<Value> srcChainValues({srcVal, dstVal});

    // helper to test if a value is derived from src (directly/indirectly)
    auto isDerivedFromSrc = [&](Value start) -> bool {
      SmallPtrSet<Value, 16> visited;
      SmallVector<Value> stack;
      stack.push_back(start);

      while (!stack.empty()) {
        Value cur = stack.pop_back_val();
        if (!visited.insert(cur).second)
          continue;

        if (cur == srcVal || cur == dstVal)
          return true;

        // if we already know a replacement mapping (not used here) would
        // short-circuit, but for this check we only care about dataflow to
        // src/dst
        Operation *def = cur.getDefiningOp();
        if (!def)
          continue;

        // stop if we hit a layout sensitve op
        if (!isLayoutCompatibleToOp(def, targetEnc))
          continue;

        if (def == srcOp)
          return true;

        for (Value opnd : def->getOperands()) {
          if (!visited.count(opnd))
            stack.push_back(opnd);
        }
      }
      return false;
    };

    // populate srcChainValues by scanning orderedOps results and testing
    // derivation
    for (Operation *op : orderedOps) {
      for (Value r : op->getResults()) {
        if (isDerivedFromSrc(r))
          srcChainValues.insert(r);
      }
    }

    // check if we need to propagate up ie have operand that is not from chain
    bool needsGlobalReplace = false;
    for (Operation *op : orderedOps) {
      for (Value operand : op->getOperands()) {
        // skip operands that are part of the chain
        if (srcChainValues.count(operand))
          continue;

        // only consider RankedTensorType operands with encodings
        if (auto opRT = dyn_cast<RankedTensorType>(operand.getType())) {
          Attribute opEnc = opRT.getEncoding();
          if (!opEnc)
            continue;

          // ensure the producers we would have to change are compatible
          if (!canPropagateUpFrom(operand, targetEnc, srcChainValues)) {
            return failure();
          }

          if (opEnc != targetEnc) {
            needsGlobalReplace = true;
            break;
          }
        }
      }
      if (needsGlobalReplace)
        break;
    }

    ModuleOp module = convertOp->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (!isSafeToReplaceEncoding(module, dstEnc, targetEnc))
      return failure();

    // check layout compatibility against affect users
    for (Operation *op : orderedOps) {
      if (!isLayoutCompatibleToOp(op, targetEnc)) {
        return failure();
      }
    }
    // Propgate downwards text-style replacement starting at this convert op.
    // This updates the reachable chain's result types and constants and will
    // update function result types/block args if we reached the end.
    replaceEncodingInModuleDownProp(module, convertOp, dstEnc, targetEnc);

    // Only if there are non-chain operands that do not already have the target
    // encoding, call the global module-wide textual replacement to update
    // remaining occurrences of dstEnc -> targetEnc. This avoids a costly global
    // replacement when it's unnecessary.
    if (needsGlobalReplace) {
      replaceEncodingInModule(module, dstEnc, targetEnc);
    }

    // ConvertLayout is now redundant: replace uses of its result with its
    // source (which now has targetEnc semantics) and erase it.
    rewriter.replaceOp(convertOp, srcVal);

    return success();
  }
};

struct RemoveConvertLayoutByPropagatingUpPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  MLIRContext *ctx;
  RemoveConvertLayoutByPropagatingUpPattern(MLIRContext *context)
      : OpRewritePattern<ConvertLayoutOp>(context, /*benefit=*/1),
        ctx(context) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {

    auto actionAttr = convertOp->getAttrOfType<StringAttr>("layout.action");
    if (!actionAttr)
      return failure();

    StringRef action = actionAttr.getValue();
    if (action != "propagate_up")
      return failure();

    // get source and destination types
    Value srcVal = convertOp.getSrc();
    Value dstVal = convertOp.getResult();

    Type srcType = srcVal.getType();
    Type dstType = dstVal.getType();

    auto srcRT = dyn_cast<RankedTensorType>(srcType);
    auto dstRT = dyn_cast<RankedTensorType>(dstType);
    if (!srcRT || !dstRT)
      return failure();

    Attribute srcEnc = getEncodingAttr(srcRT);
    Attribute dstEnc = getEncodingAttr(dstRT);

    // Don't propagate linear layout for now
    if (isa<LinearEncodingAttr>(srcEnc))
      return failure();

    if (!srcEnc || !dstEnc)
      return failure();

    // TODO: Slice to slice layout is not supported. To support we must
    // propagate slice to slice but also the parent block to parent block. Take
    // expandDimOp for example
    if (isa<SliceEncodingAttr>(srcEnc) && isa<SliceEncodingAttr>(dstEnc)) {
      return failure();
    }

    // if we propagate up and hit a convert layout we stop
    // TODO: make it propagate up partial ops between the convert layout
    // (including stopping at reshape as reshape just like convert layout can
    // change the layouts)
    SmallVector<Value> worklist;
    DenseSet<Value> visited;
    worklist.push_back(srcVal);
    visited.insert(srcVal);

    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      Operation *defOp = current.getDefiningOp();
      if (!defOp) {
        continue;
      }
      if (dyn_cast<ConvertLayoutOp>(defOp)) {
        return failure();
      }

      // compatibility check
      if (!isLayoutCompatibleToOp(defOp, dstEnc)) {
        return failure();
      }

      for (Value operand : defOp->getOperands()) {
        if (visited.insert(operand).second) {
          worklist.push_back(operand);
        }
      }
    }

    ModuleOp module = convertOp->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    if (!isSafeToReplaceEncoding(module, srcEnc, dstEnc))
      return failure();

    // update encodings across the module
    replaceEncodingInModule(module, srcEnc, dstEnc);

    // replace all uses of the convert_layout result with its source
    dstVal.replaceAllUsesWith(srcVal);

    // erase the convert_layout op
    rewriter.eraseOp(convertOp);

    return success();
  }
};

class OptimizeLayoutsPass
    : public impl::OptimizeLayoutsBase<OptimizeLayoutsPass> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *ctx = module.getContext();
    llvm::StringRef myPassName = this->getArgument();
    int digit = mlir::triton::util::getPassColumnDigit(module, myPassName);
    if (digit != 0) {
      mlir::triton::OptimizeLayoutsAnalysis analysis(module);

      // improved logic in DecomposeReduction pass
      if (isInDigitList(digit, {2})) {
        RewritePatternSet reducePatterns(ctx);
        reducePatterns.add<ReduceOpPattern>(ctx);
        if (failed(applyPatternsGreedily(module, std::move(reducePatterns))))
          return signalPassFailure();
      }
      {
        analysis.runAnalysis();
        RewritePatternSet patterns(ctx);
        patterns.add<RemoveConvertLayoutByPropagatingDownPattern>(ctx);
        if (failed(applyPatternsGreedily(module, std::move(patterns))))
          return signalPassFailure();
      }

      LLVM_DEBUG(llvm::dbgs() << "Function after phase 1: \n"; module.dump();
                 llvm::dbgs() << "\n";);

      {
        analysis.refineAfterPropDown();
        RewritePatternSet cleanupPatterns(ctx);
        cleanupPatterns.add<RemoveConvertLayoutByPropagatingUpPattern>(ctx);
        if (failed(applyPatternsGreedily(module, std::move(cleanupPatterns))))
          return signalPassFailure();
      }
    }
  }

  bool isInDigitList(int digit, std::initializer_list<int> list) {
    return llvm::find(list, digit) != list.end();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createOptimizeLayoutsPass() {
  return std::make_unique<OptimizeLayoutsPass>();
}

} // namespace triton
} // namespace bishengir