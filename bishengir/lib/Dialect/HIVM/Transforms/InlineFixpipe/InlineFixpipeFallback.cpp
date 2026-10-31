//===- InlineFixpipeFallback.cpp - fallback_not_exec sink -----------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/HIVM/Transforms/InlineFixpipe/InlineFixpipeFallback.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Transforms/InlineFixpipe/InlineFixpipeUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define DEBUG_TYPE "hivm-inline-fixpipe"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace mlir::hivm {
namespace {

struct FallbackConsumerChain {
  SmallVector<Operation *> ops; // from if-result toward store (exclusive)
  hivm::StoreOp store;
};

/// Collect a single-use chain from if-result to StoreOp through ops that the
/// existing InlineFixpipe patterns already know how to fold (Cube path) or
/// that are valid to replay as vector ops (no_exec path). This pattern only
/// sinks the chain; fusion is left to greedy rewrite.
FailureOr<FallbackConsumerChain>
collectFallbackStoreChain(PatternRewriter &rewriter, Operation *notifyOp,
                          hivm::FixpipeOp fixpipe, Value ifResult) {
  FallbackConsumerChain chain;
  Value cur = ifResult;
  while (true) {
    Operation *user = getSingleSiftedUser(cur);
    if (!user)
      return rewriter.notifyMatchFailure(
          notifyOp, "fallback_not_exec consumer chain is not single-use");

    // The whole chain must sit in the same block as the fallback if. If it
    // is nested in another region (e.g. a guarding scf.if / scf.for), its
    // external operands (store dst, dynamic slice indices) are defined in
    // that region and cannot dominate the sunk if, and sinking the store
    // would also drop the guarding condition.
    if (user->getBlock() != notifyOp->getBlock())
      return rewriter.notifyMatchFailure(
          user, "fallback_not_exec consumer chain is nested in another region");

    if (auto storeOp = dyn_cast<hivm::StoreOp>(user)) {
      if (!isFixpipeStoreInlinable(storeOp))
        return rewriter.notifyMatchFailure(
            storeOp, "fallback_not_exec store has unsupported atomic kind");
      if (isInsideVectorScope(storeOp))
        return rewriter.notifyMatchFailure(
            storeOp, "fallback_not_exec store is inside a vector scope");
      chain.store = storeOp;
      return chain;
    }

    if (auto castOp = dyn_cast<hivm::VCastOp>(user)) {
      // getQuantMode includes isVCastInlinableIntoFixpipe check
      if (!getQuantMode(castOp).has_value())
        return rewriter.notifyMatchFailure(
            castOp, "fallback_not_exec vcast is not inlinable as pre_quant");
      if (castOp.getSrc()[0] != cur)
        return rewriter.notifyMatchFailure(
            castOp, "fallback_not_exec vcast src is not the chain value");
      chain.ops.push_back(castOp);
      cur = castOp.getResult()[0];
      continue;
    }

    if (auto reluOp = dyn_cast<hivm::VReluOp>(user)) {
      if (!getReluMode(reluOp).has_value())
        return rewriter.notifyMatchFailure(
            reluOp, "fallback_not_exec vrelu is not inlinable as pre_relu");
      if (reluOp.getSrc()[0] != cur)
        return rewriter.notifyMatchFailure(
            reluOp, "fallback_not_exec vrelu src is not the chain value");
      chain.ops.push_back(reluOp);
      cur = reluOp.getResult()[0];
      continue;
    }

    if (isa<hivm::VTransposeOp>(user)) {
      if (!isUserTransposeInlinable(fixpipe, user))
        return rewriter.notifyMatchFailure(
            user, "fallback_not_exec vtranspose is not inlinable into fixpipe");
      auto transposeOp = cast<hivm::VTransposeOp>(user);
      if (transposeOp.getSrc() != cur)
        return rewriter.notifyMatchFailure(
            transposeOp,
            "fallback_not_exec vtranspose src is not the chain value");
      chain.ops.push_back(transposeOp);
      cur = transposeOp.getResult()[0];
      continue;
    }

    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
      if (extractSliceOp.getSource() != cur)
        return rewriter.notifyMatchFailure(
            extractSliceOp,
            "fallback_not_exec extract_slice source is not the chain value");
      chain.ops.push_back(extractSliceOp);
      cur = extractSliceOp.getResult();
      continue;
    }

    return rewriter.notifyMatchFailure(
        user, "unsupported op in fallback_not_exec consumer chain");
  }
}

/// Replay \p chain after \p src (branch-local value) and emit the terminal
/// store. Used identically for both fallback_not_exec branches.
void emitSunkChainAndStore(PatternRewriter &rewriter, Location loc, Value src,
                           const FallbackConsumerChain &chain) {
  Value cur = src;
  for (Operation *op : chain.ops) {
    if (auto castOp = dyn_cast<hivm::VCastOp>(op)) {
      Value init = utils::createEmptyOp(rewriter, loc, castOp.getResult()[0]);
      IRMapping mapping;
      mapping.map(castOp.getSrc()[0], cur);
      mapping.map(castOp.getDst()[0], init);
      auto *cloned = rewriter.clone(*castOp, mapping);
      cur = cloned->getResult(0);
      continue;
    }
    if (auto reluOp = dyn_cast<hivm::VReluOp>(op)) {
      Value init = utils::createEmptyOp(rewriter, loc, reluOp.getResult()[0]);
      IRMapping mapping;
      mapping.map(reluOp.getSrc()[0], cur);
      mapping.map(reluOp.getDst()[0], init);
      auto *cloned = rewriter.clone(*reluOp, mapping);
      cur = cloned->getResult(0);
      continue;
    }
    if (auto transposeOp = dyn_cast<hivm::VTransposeOp>(op)) {
      Value init =
          utils::createEmptyOp(rewriter, loc, transposeOp.getResult()[0]);
      auto newTranspose = rewriter.create<hivm::VTransposeOp>(
          transposeOp.getLoc(), TypeRange{init.getType()}, cur, init,
          transposeOp.getPermutationAttr());
      cur = newTranspose.getResult()[0];
      continue;
    }
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      auto newExtract = rewriter.create<tensor::ExtractSliceOp>(
          extractSliceOp.getLoc(), extractSliceOp.getResultType(), cur,
          extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
          extractSliceOp.getMixedStrides());
      cur = newExtract.getResult();
      continue;
    }
    llvm_unreachable("unexpected fallback_not_exec consumer op");
  }

  auto storeOp = chain.store;
  auto newStore = rewriter.create<hivm::StoreOp>(storeOp.getLoc(), TypeRange{},
                                                 cur, storeOp.getDst());
  if (auto atomicAttr = storeOp.getAtomicKindAttr())
    newStore.setAtomicKindAttr(atomicAttr);
}

/// Clone ops from \p srcBlock into the builder's current block, excluding
/// the terminator. Returns the mapped value of \p yieldedValue.
Value cloneBlockBodyExceptTerminator(PatternRewriter &rewriter, Block &srcBlock,
                                     Value yieldedValue) {
  IRMapping mapping;
  for (Operation &op : srcBlock.without_terminator())
    rewriter.clone(op, mapping);
  return mapping.lookupOrDefault(yieldedValue);
}

/// Values from the sunk chain that are external to the if-result pipeline
/// (store dst, dynamic extract_slice indices). These must dominate the new
/// if after the sink.
void collectSunkChainExternalValues(const FallbackConsumerChain &chain,
                                    SmallVectorImpl<Value> &values) {
  // StoreOp::getDst() is non-const; copy the lightweight handle first.
  hivm::StoreOp storeOp = chain.store;
  values.push_back(storeOp.getDst());
  for (Operation *op : chain.ops) {
    auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op);
    if (!extractSliceOp)
      continue;
    // Source is the chain value; only offsets/sizes/strides are external.
    values.append(extractSliceOp.getOffsets().begin(),
                  extractSliceOp.getOffsets().end());
    values.append(extractSliceOp.getSizes().begin(),
                  extractSliceOp.getSizes().end());
    values.append(extractSliceOp.getStrides().begin(),
                  extractSliceOp.getStrides().end());
  }
}

/// Hoist effect-free defs of \p values (and their deps) that sit after
/// \p ifOp in the same block so they dominate uses sunk into the if.
/// Returns failure if a required def cannot be safely hoisted.
LogicalResult hoistValuesBeforeIf(PatternRewriter &rewriter, scf::IfOp ifOp,
                                  ArrayRef<Value> values) {
  Block *block = ifOp->getBlock();
  SmallVector<Operation *> toHoist;
  SmallPtrSet<Operation *, 8> seen;

  std::function<LogicalResult(Value)> collect = [&](Value v) -> LogicalResult {
    Operation *def = v.getDefiningOp();
    if (!def)
      return success(); // block/arg — already dominates
    if (def->getBlock() != block)
      return success(); // defined in a dominating block
    if (def->isBeforeInBlock(ifOp))
      return success();
    if (!seen.insert(def).second)
      return success();
    if (!isMemoryEffectFree(def))
      return rewriter.notifyMatchFailure(
          ifOp, "cannot hoist side-effecting def before fallback_not_exec if");
    for (Value operand : def->getOperands())
      if (failed(collect(operand)))
        return failure();
    toHoist.push_back(def);
    return success();
  };

  for (Value v : values)
    if (failed(collect(v)))
      return failure();

  for (Operation *op : toHoist)
    rewriter.moveOpBefore(op, ifOp);
  return success();
}

/// Sink the outer consumer chain into a resultless fallback_not_exec if.
///
/// After this rewrite the Cube branch still has the original UB-dest
/// fixpipe followed by the sunk cast/transpose/extract_slice/store ops.
/// Subsequent greedy applications of the existing InlineFixpipe patterns
/// fold those consumers into fixpipe (including extract_slice swap, which
/// only requires same-block dominance — satisfied after the sink).
///
/// TODO: consider the wrap-around-store alternative: keep the original
/// scf.if, move it to just before the outer store, and rebuild it in place
/// as a resultless if containing the sunk chain in both branches. That
/// removes the need for hoistValuesBeforeIf

} // namespace

LogicalResult sinkFixpipeIntoFallbackIf(PatternRewriter &rewriter, Location loc,
                                        hivm::FixpipeOp fixpipe,
                                        scf::IfOp ifOp) {
  if (ifOp.getElseRegion().empty())
    return rewriter.notifyMatchFailure(
        ifOp, "fallback_not_exec if has no else region");
  // v1: only handle single-result ifs so we can rewrite to a void if.
  if (ifOp.getNumResults() != 1)
    return rewriter.notifyMatchFailure(
        ifOp, "fallback_not_exec if does not have exactly one result");

  Value fixpipeRes = fixpipe.getResultTensor();
  if (!fixpipeRes)
    return rewriter.notifyMatchFailure(
        fixpipe, "fallback_not_exec fixpipe has no result tensor");

  Value thenYielded = ifOp.thenYield().getOperand(0);
  Value elseYielded = ifOp.elseYield().getOperand(0);

  bool fixpipeInThen = thenYielded == fixpipeRes;
  bool fixpipeInElse = elseYielded == fixpipeRes;
  if (fixpipeInThen == fixpipeInElse)
    return rewriter.notifyMatchFailure(
        ifOp, "fixpipe is yielded by both branches or by neither");

  Value vectorYielded = fixpipeInThen ? elseYielded : thenYielded;
  if (vectorYielded.getDefiningOp<hivm::FixpipeOp>())
    return rewriter.notifyMatchFailure(ifOp, "fixpipe found in both branches");

  Value ifResult = ifOp.getResult(0);
  auto maybeChain =
      collectFallbackStoreChain(rewriter, ifOp, fixpipe, ifResult);
  if (failed(maybeChain))
    return failure();
  FallbackConsumerChain chain = *maybeChain;

  SmallVector<Value> externalValues;
  collectSunkChainExternalValues(chain, externalValues);
  // Store dst (e.g. memref.subview) and dynamic slice indices often sit
  // between the if and the outer store; hoist them before the if so sunk
  // uses dominate.
  if (failed(hoistValuesBeforeIf(rewriter, ifOp, externalValues)))
    return failure();

  LDBG("sinkFixpipeIntoFallbackIf");

  SmallVector<Operation *> toErase;
  toErase.push_back(chain.store);
  for (Operation *op : llvm::reverse(chain.ops))
    toErase.push_back(op);

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(ifOp);
  auto newIf = rewriter.create<scf::IfOp>(loc, TypeRange{}, ifOp.getCondition(),
                                          /*withElseRegion=*/true);
  newIf->setAttr(kFallbackNotExec, rewriter.getUnitAttr());

  // Resultless scf.if already has empty scf.yield terminators from the
  // builder (ensureTerminator). Insert branch bodies before those yields;
  // do not create a second terminator.
  auto populateBranch = [&rewriter, &loc, &chain](
                            Block *destBlock, Block &srcBlock, Value yielded) {
    rewriter.setInsertionPointToStart(destBlock);
    Value mapped = cloneBlockBodyExceptTerminator(rewriter, srcBlock, yielded);
    emitSunkChainAndStore(rewriter, loc, mapped, chain);
  };

  populateBranch(newIf.thenBlock(), ifOp.getThenRegion().front(), thenYielded);
  populateBranch(newIf.elseBlock(), ifOp.getElseRegion().front(), elseYielded);

  // Erase outer consumer chain (store first), ignored users of the if
  // result, then the old if (which owns the original in-branch fixpipe).
  // Marks / debug / dim on the if-result cannot be preserved: the rewrite
  // deletes that SSA value. Warn so the drop is visible, but do not fail.
  for (Operation *op : toErase)
    rewriter.eraseOp(op);
  SmallVector<Operation *> ignoredUsers(ifResult.getUsers().begin(),
                                        ifResult.getUsers().end());
  for (Operation *user : ignoredUsers) {
    if (isa<annotation::MarkOp, hivm::DebugOp, tensor::DimOp>(user)) {
      user->emitWarning()
          << "dropping " << user->getName()
          << " on fallback_not_exec if result; the tensor no longer exists "
             "after InlineFixpipe sink";
      rewriter.eraseOp(user);
    }
  }
  rewriter.eraseOp(ifOp);
  return success();
}
} // namespace mlir::hivm
