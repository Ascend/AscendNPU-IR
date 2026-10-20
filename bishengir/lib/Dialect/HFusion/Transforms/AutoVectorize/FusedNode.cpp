//===--------- FusedNode.cpp - Fusion group for AV2 planning
//---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "hfusion-auto-vectorize-v2"

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/FusedNode.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/PlanContext.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace mlir {
namespace hfusion {

//===----------------------------------------------------------------------===//
// FusedNode member functions
//===----------------------------------------------------------------------===//

void FusedNode::addProducer(Operation *op) {
  producers.insert(op);
  insertOrdered(op);
  ctx.getInfo(op).fusedNode = shared_from_this();
  consolidateToTail();
  updateConflicts(op);
}

void FusedNode::addLeaf(Operation *op) {
  leaf.insert(op);
  insertOrdered(op);
  ctx.getInfo(op).fusedNode = shared_from_this();
  consolidateToTail();
  updateConflicts(op);
}

void FusedNode::insertOrdered(Operation *op) {
  // The node is contiguous at entry (consolidateToTail ran on the previous
  // insert), so op is either before front() or after back(); insert at the
  // matching end to keep block order.
  if (opsOrdered.empty() || op->isBeforeInBlock(opsOrdered.front()))
    opsOrdered.insert(opsOrdered.begin(), op);
  else
    opsOrdered.push_back(op);
  assert(opsOrdered.size() == size());
}

// Move the group as a unit, preserving order and contiguity. pos is pinned to
// an extreme of the node containing it so the sweep never splits that node:
//   a member anchor becomes this group's head;
//   an anchor in another node is pinned to that node's tail, so this group
//   lands wholly after it.
void FusedNode::moveAfter(Operation *pos) const {
  if (contains(pos))
    pos = opsOrdered.front();
  else if (auto *node = ctx.tryGetNode(pos))
    pos = node->opsOrdered.back();
  for (auto *op : rops())
    if (op != pos)
      op->moveAfter(pos);
}

// Move the group as a unit, preserving order and contiguity. pos is pinned to
// an extreme of the node containing it so the sweep never splits that node:
//   a member anchor becomes this group's tail;
//   an anchor in another node is pinned to that node's head, so this group
//   lands wholly before it.
void FusedNode::moveBefore(Operation *pos) const {
  if (contains(pos))
    pos = opsOrdered.back();
  else if (auto *node = ctx.tryGetNode(pos))
    pos = node->opsOrdered.front();
  for (auto *op : ops())
    if (op != pos)
      op->moveBefore(pos);
}

bool FusedNode::hasCommonAxis(Operation *op) const {
  const auto &shape1 = ctx.getInfo(op).shape;
  const auto &shape2 = ctx.getInfo(*leaf.begin()).shape;
  if (shape1 == shape2)
    return true;
  if (shape1.size() > 1 && shape2.size() > 1 && shape1[0] == shape2[0])
    return true;
  return false;
}

bool FusedNode::canFitStack(Operation *candidate) const {
  // Keep AutoVectorize planning aligned with the VF stack-slot budget.
  SmallVector<Operation *> existingOps(ops().begin(), ops().end());
  if (!contains(candidate))
    existingOps.push_back(candidate);
  return ctx.canFitStack(existingOps);
}

bool FusedNode::isMemref() const {
  assert(!leaf.empty());
  return isMemrefLinalgOp(*leaf.begin());
}

bool FusedNode::hasRankReducing() const {
  return llvm::any_of(ops(), hasRankReducingIndexingMap);
}

bool FusedNode::conflictsWith(const FusableOpInfo &otherInfo) const {
  return llvm::any_of(
      ops(), [&](Operation *op) { return ctx.hasConflict(otherInfo, op); });
}

bool FusedNode::canAccept(Operation *candidate, AcceptContext actx) const {
  // TODO: Unify per-call-site flags after verifying no regression.
  // checkReverseVsstb and countAllOpsForSize are pure bugfixes;
  // checkIsMemref needs investigation (may be a legitimate distinction).
  // See AcceptContext doc in FusedNode.h.
  bool checkIsMemref = (actx != AcceptContext::Producer);

  if (size() >= ctx.getMaxFusedOps())
    return false;
  if (checkIsMemref && isMemref())
    return false;
  if (!canFitStack(candidate))
    return false;
  if (hasVsstb() && hasRankReducingIndexingMap(candidate))
    return false;
  if (hasRankReducing() && analysis::isVsstbPatternTransposeOp(candidate))
    return false;
  return hasCommonAxis(candidate) && !conflictsWith(ctx.getInfo(candidate));
}

//===----------------------------------------------------------------------===//
// Helpers for consolidateToTail
//===----------------------------------------------------------------------===//

using SuperOpTy = llvm::PointerUnion<Operation *, FusedNode const *>;

static void climbedForwardSlice(PlanContext &ctx, SuperOpTy sop,
                                SetVector<SuperOpTy> &forwardSlice,
                                DenseSet<SuperOpTy> &visited,
                                const DominanceInfo &domInfo,
                                Operation *anchor) {
  if (!visited.insert(sop).second)
    llvm::report_fatal_error(
        "Find circular dependency between the fused nodes.");

  auto climb = [anchor](auto *op) {
    return anchor->getBlock()->findAncestorOpInBlock(*op);
  };
  auto visit = [&](auto *op) {
    if (domInfo.dominates(anchor, op))
      return;
    FusedNode const *node = ctx.tryGetNode(op);
    SuperOpTy user = node ? SuperOpTy(node) : SuperOpTy(op);
    if (forwardSlice.count(user))
      return;
    climbedForwardSlice(ctx, user, forwardSlice, visited, domInfo, anchor);
  };

  if (auto *node = sop.dyn_cast<FusedNode const *>()) {
    for (auto *op : node->rops()) {
      for (Operation *userOp : map_range(op->getUsers(), climb)) {
        if (node->contains(userOp))
          continue;
        visit(userOp);
      }
    }
  } else {
    auto *op = sop.get<Operation *>();
    for (Operation *userOp : map_range(op->getUsers(), climb))
      visit(userOp);
  }

  forwardSlice.insert(sop);
}

// Restore contiguity after an insertion: move the node's forward-slice users
// past the tail, then pull the node's own ops back together before it.
void FusedNode::consolidateToTail() const {
  if (opsOrdered.size() == 1)
    return;

  // opsOrdered is maintained in block order by insertOrdered, so back() is the
  // node tail.
  Operation *lastOp = opsOrdered.back();

  DominanceInfo domInfo(opsOrdered.front());
  DenseSet<SuperOpTy> visited;
  SetVector<SuperOpTy> forwardSlice;
  climbedForwardSlice(ctx, this, forwardSlice, visited, domInfo, lastOp);

  // drop_end skips the node itself.
  for (auto userOp : drop_end(forwardSlice)) {
    if (auto *node = userOp.dyn_cast<FusedNode const *>()) {
      node->moveAfter(lastOp);
      continue;
    }
    auto *op = userOp.get<Operation *>();
    op->moveAfter(lastOp);
  }
  moveBefore(lastOp);

  // Guard the contiguity invariant insertOrdered depends on.
  for (auto node : ctx.nodes())
    for (auto [prev, next] :
         zip(drop_end(node->ops()), drop_begin(node->ops())))
      if (prev->getNextNode() != next)
        llvm::report_fatal_error(StringRef(
            "consolidateToTail make wrong order in " + node->loopLabel));
  if (failed(verify(opsOrdered.back()->getParentOp())))
    llvm::report_fatal_error("consolidateToTail produced invalid IR");
}

void FusedNode::updateConflicts(Operation *newOp) const {
  Block *block = newOp->getBlock();
  ctx.dissolvePivot(newOp, this, block);

  DenseSet<Operation *> upstreamOps;
  DenseSet<Operation *> visitedUpstreamOps;
  DenseSet<Operation *> downstreamOps;
  DenseSet<Operation *> visitedDownstreamOps;
  for (Operation *op : ops()) {
    findUpstreamFusableOpOf(op, block, upstreamOps, visitedUpstreamOps);
    findDownstreamFusableOpOf(op, block, downstreamOps, visitedDownstreamOps);
  }
  for (Operation *op : ops()) {
    upstreamOps.erase(op);
    downstreamOps.erase(op);
  }
  for (auto upstreamOp : upstreamOps) {
    for (auto downstreamOp : downstreamOps) {
      ctx.addGroupConflict(upstreamOp, downstreamOp, this);
    }
  }
}

void FusedNode::estimateTileSizeForOp(
    Operation *fusedOp, SmallVectorImpl<int64_t> &tileSize,
    SmallVectorImpl<int64_t> &tileInterchange) const {
  const unsigned vectorLength = ctx.getVectorLength();
  unsigned maxElemBitWidthInFusedNode = 1;
  for (Operation *nodeOp : ops()) {
    maxElemBitWidthInFusedNode = std::max(maxElemBitWidthInFusedNode,
                                          ctx.getInfo(nodeOp).maxElemBitWidth);
  }

  bool shouldMultiAxisVectorize = hasVsstb();

  FusableOpInfo &opInfo = ctx.getInfo(fusedOp);
  assert(opInfo.numLoops > 0);
  tileSize.assign(opInfo.numLoops, 1);
  if (shouldMultiAxisVectorize && opInfo.numLoops > 2) {
    int64_t maxElemByteWidthInFusedNode =
        maxElemBitWidthInFusedNode / utils::INTR_BITS_PER_BYTE;
    int64_t remainBytes = vectorLength;
    int64_t allocAxisNum = 0;
    for (int64_t i = opInfo.numLoops - 1; i >= 0; --i) {
      allocAxisNum++;
      if (maxElemByteWidthInFusedNode == 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "maxElemByteWidthInFusedNode is 0, "
                   << "use default tile size " << opInfo.shape[i] << "\n");
        tileSize[i] = opInfo.shape[i];
        continue;
      }
      if (allocAxisNum == 2) {
        tileSize[i] = remainBytes / maxElemByteWidthInFusedNode;
        break;
      } else {
        tileSize[i] = std::min(opInfo.shape[i],
                               remainBytes / maxElemByteWidthInFusedNode);
        remainBytes /= tileSize[i];
      }
    }
    if (analysis::isVsstbPatternTransposeOp(fusedOp)) {
      auto transpose = dyn_cast<linalg::TransposeOp>(fusedOp);
      tileInterchange.clear();
      SmallVector<int64_t> transposeTileSize;
      for (int64_t idx : transpose.getPermutation()) {
        transposeTileSize.push_back(tileSize[idx]);
        tileInterchange.push_back(idx);
      }
      tileSize = transposeTileSize;
    }
    return;
  }

  unsigned bytes = maxElemBitWidthInFusedNode != 1
                       ? maxElemBitWidthInFusedNode / utils::INTR_BITS_PER_BYTE
                       : 1;
  tileSize[opInfo.numLoops - 1] = vectorLength / bytes;
}

// here we do not fuse op into user in DpsInits, see issue:
// https://codehub-y.huawei.com/CompilerKernel/BiShengKernel/BiSheng/issues/3687
bool FusedNode::useAsDpsInits(Operation *newOp) const {
  auto results = newOp->getResults();
  return any_of(ops(), [results](Operation *fusedOp) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(fusedOp);
    if (!linalgOp)
      return false;
    return llvm::any_of(results, [&linalgOp](OpResult res) {
      return llvm::is_contained(linalgOp.getDpsInits(), res);
    });
  });
}

bool FusedNode::canFuseProducer(Operation *producer) const {
  if (useAsDpsInits(producer))
    return false;

  SmallVector<int64_t> estimatedTileSize;
  return !wouldBecomeTileLocal(producer, estimatedTileSize) ||
         !reductionConsumerNeedsFullDomain(producer, estimatedTileSize);
}

bool FusedNode::wouldBecomeTileLocal(
    Operation *producer, SmallVectorImpl<int64_t> &estimatedTileSize) const {
  const FusableOpInfo &producerInfo = ctx.getInfo(producer);
  if (producerInfo.shape.empty())
    return false;
  SmallVector<int64_t> ignoredInterchange;
  estimateTileSizeForOp(producer, estimatedTileSize, ignoredInterchange);

  if (producerInfo.shape.size() != estimatedTileSize.size())
    return false;
  for (auto [shapeDim, tileDim] :
       llvm::zip_equal(producerInfo.shape, estimatedTileSize)) {
    if (ShapedType::isDynamic(shapeDim) || shapeDim <= 0 || tileDim <= 0)
      return false;
    if (tileDim < shapeDim)
      return true;
  }
  return false;
}

namespace {
bool canLegallySplitReductionConsumer(const Operation *producer,
                                      linalg::LinalgOp reductionConsumer) {
  if (!producer || !reductionConsumer ||
      reductionConsumer.getNumReductionLoops() == 0)
    return false;

  // Reuse the pass' own reduction-tiling legality gate so this predicate stays
  // aligned with the actual lowering path that produces partial accumulation
  // plus final vector reduction.
  if (!shouldUseTileReductionUsingForV2(reductionConsumer))
    return false;

  SmallVector<unsigned> reductionDims;
  reductionConsumer.getReductionDims(reductionDims);

  OpOperand *producerUse = nullptr;
  for (OpOperand &use : reductionConsumer->getOpOperands()) {
    if (use.get().getDefiningOp() == producer) {
      producerUse = &use;
      break;
    }
  }
  if (!producerUse)
    return false;

  AffineMap indexingMap = reductionConsumer.getMatchingIndexingMap(producerUse);
  return indexingMap.isIdentity();
}
} // namespace

bool FusedNode::reductionConsumerNeedsFullDomain(
    Operation *producer, ArrayRef<int64_t> estimatedTileSize) const {
  if (!producer)
    return false;
  SmallVector<linalg::LinalgOp> reductionConsumers;
  for (Operation *user : producer->getUsers()) {
    if (!contains(user))
      continue;

    auto linalgUser = dyn_cast<linalg::LinalgOp>(user);
    if (!linalgUser || linalgUser.getNumReductionLoops() == 0)
      continue;
    reductionConsumers.push_back(linalgUser);
  }

  for (linalg::LinalgOp linalgUser : reductionConsumers) {
    SmallVector<unsigned> reductionDims;
    linalgUser.getReductionDims(reductionDims);
    if (reductionConsumers.size() == 1 &&
        canLegallySplitReductionConsumer(producer, linalgUser)) {
      continue;
    }

    OpOperand *producerUse = nullptr;
    for (OpOperand &use : linalgUser->getOpOperands()) {
      if (use.get().getDefiningOp() == producer) {
        producerUse = &use;
        break;
      }
    }
    if (!producerUse)
      return true;

    AffineMap indexingMap = linalgUser.getMatchingIndexingMap(producerUse);
    DenseSet<unsigned> reductionDimSet(reductionDims.begin(),
                                       reductionDims.end());

    for (auto [producerDim, expr] : llvm::enumerate(indexingMap.getResults())) {
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr)
        continue;
      unsigned consumerLoopDim = dimExpr.getPosition();
      if (!reductionDimSet.contains(consumerLoopDim))
        continue;
      if (producerDim >= estimatedTileSize.size())
        return true;
      if (estimatedTileSize[producerDim] <
          cast<ShapedType>(producer->getResult(0).getType())
              .getShape()[producerDim])
        return true;
    }
  }
  return false;
}

} // namespace hfusion
} // namespace mlir
