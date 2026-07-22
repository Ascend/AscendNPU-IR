//===------- Transforms.cpp - Transform Extend Fuse Into ContainingOp -----===//
//
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
#include "bishengir/Dialect/HFusion/IR/HFusion.h"

#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Transforms/Transforms.h"

#define DEBUG_TYPE "hfusion-transform-op"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::transform;
using namespace bishengir;

static SmallVector<Value> recursiveClone(RewriterBase &rewriter,
                                         SmallVector<Value> values,
                                         Operation *clonePoint) {
  SmallVector<Value> newValues;
  for (auto value : values) {
    // If target value is a block argument, we can use it anywhere we want.
    if (isa<BlockArgument>(value)) {
      newValues.push_back(value);
      continue;
    }
    // If target value is a result of a value defined before the target
    // cloning point, we need to recursively clone its operands.
    auto *defOperation = value.getDefiningOp();
    if (defOperation == nullptr) {
      return newValues;
    }
    if (clonePoint->getBlock() == defOperation->getBlock() &&
        clonePoint->isBeforeInBlock(defOperation)) {
      auto operands = defOperation->getOperands();
      auto clonedValues = recursiveClone(rewriter, operands, clonePoint);

      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(clonePoint);

      IRMapping mapping;
      mapping.map(operands, clonedValues);
      auto *clonedOp = rewriter.clone(*defOperation, mapping);

      newValues.push_back(
          clonedOp->getResult(cast<OpResult>(value).getResultNumber()));
    } else {
      newValues.push_back(value);
    }
  }

  return newValues;
}

static bool isValidSliceOpInContainingOp(tensor::ExtractSliceOp sliceOp,
                                         Operation *containingOp) {
  if (!sliceOp || !containingOp->isProperAncestor(sliceOp)) {
    return false;
  }

  auto staticStrides = sliceOp.getStaticStrides();
  if (llvm::count_if(staticStrides, [](int64_t s) { return s != 1; }) > 0) {
    // only union extract slice with stride 1
    return false;
  }

  return true;
}

static void getFirstSliceUserInContainingOp(
    Operation *producerOp, Operation *containingOp,
    llvm::DenseMap<Value, tensor::ExtractSliceOp> *result2FirstSliceOp,
    llvm::DenseMap<Value, int> *result2ValidNum) {
  for (auto res : producerOp->getResults()) {
    tensor::ExtractSliceOp firstSliceOp;
    int validNum = 0;
    for (auto user : res.getUsers()) {
      auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!isValidSliceOpInContainingOp(sliceOp, containingOp)) {
        continue;
      }

      if (!firstSliceOp || sliceOp->isBeforeInBlock(firstSliceOp)) {
        firstSliceOp = sliceOp;
      }

      validNum++;
    }
    result2ValidNum->insert(std::pair(res, validNum));
    if (firstSliceOp) {
      assert(validNum > 0);
      result2FirstSliceOp->insert(std::pair(res, firstSliceOp));
    }
  }
}

enum class MODE {
  UNION_MAX,
  UNION_MIN,
  COMPUTE_SLICE_MAX,
  COMPUTE_SUB,
  COMPUTE_DISTANCE
};

static SmallVector<Value> compute(RewriterBase &rewriter, MODE mode,
                                  const SmallVectorImpl<Value> &lhs,
                                  const SmallVectorImpl<Value> &rhs,
                                  Location loc) {
  auto symA = rewriter.getAffineSymbolExpr(0);
  auto symB = rewriter.getAffineSymbolExpr(1);
  auto one = rewriter.getAffineConstantExpr(1);
  AffineMap map;
  if (mode == MODE::UNION_MAX || mode == MODE::UNION_MIN)
    map = AffineMap::get(0, 2, {symA, symB}, rewriter.getContext());
  else if (mode == MODE::COMPUTE_SLICE_MAX)
    map = AffineMap::get(0, 2, {symA + symB - one}, rewriter.getContext());
  else if (mode == MODE::COMPUTE_SUB)
    map = AffineMap::get(0, 2, {symA - symB}, rewriter.getContext());
  else {
    assert(mode == MODE::COMPUTE_DISTANCE);
    map = AffineMap::get(0, 2, {symA - symB + one}, rewriter.getContext());
  }

  SmallVector<Value> results;
  for (auto it : llvm::zip(lhs, rhs)) {
    auto l = std::get<0>(it);
    auto r = std::get<1>(it);
    Value result;
    switch (mode) {
    case MODE::UNION_MAX:
      result = rewriter.create<affine::AffineMaxOp>(loc, map, ValueRange{l, r});
      break;
    case MODE::UNION_MIN:
      result = rewriter.create<affine::AffineMinOp>(loc, map, ValueRange{l, r});
      break;
    case MODE::COMPUTE_SLICE_MAX:
      result =
          rewriter.create<affine::AffineApplyOp>(loc, map, ValueRange{l, r});
      break;
    case MODE::COMPUTE_SUB:
    case MODE::COMPUTE_DISTANCE:
      result =
          rewriter.create<affine::AffineApplyOp>(loc, map, ValueRange{l, r});
      break;
    }
    results.push_back(result);
  }
  return results;
}

SmallVector<OpFoldResult> convert(SmallVectorImpl<Value> &values) {
  SmallVector<OpFoldResult> results;
  for (auto it : values) {
    results.push_back(OpFoldResult(it));
  }
  return results;
}

static SmallVector<Value> createEqualZeroOp(const SmallVector<Value> &targets,
                                            RewriterBase &rewriter,
                                            Location loc) {
  SmallVector<Value> results;
  for (Value target : targets) {
    // cast to i64 because arith CmpIOp does not support index type operand
    Value castResult =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), target);
#ifndef BSPUB_DAVINCI_BISHENGIR_A5
    Value zero =
        rewriter.create<arith::ConstantIntOp>(loc, 0, rewriter.getI64Type());
#else
    Value zero =
        rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI64Type(), 0);
#endif
    Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                castResult, zero);
    results.push_back(cond);
  }
  return results;
}

static SmallVector<Value> createSelectOp(const SmallVector<Value> &conds,
                                         const SmallVector<Value> &trues,
                                         const SmallVector<Value> &falses,
                                         RewriterBase &rewriter, Location loc) {
  SmallVector<Value> results;
  size_t size = conds.size();
  for (size_t i = 0; i < size; ++i) {
    Value result =
        rewriter.create<arith::SelectOp>(loc, conds[i], trues[i], falses[i]);
    results.push_back(result);
  }
  return results;
}

/// Initializes the union offsets and maxes based on the first slice op.
/// The initial union_offsets should be
/// 1. MAX_VALUE (default to slice src sizes), slice_size == 0
/// 2. slice_offset,                           slice_size != 0
/// The initial union maxes should be
/// 1. MIN_VALUE (default to zero),            slice_size == 0
/// 2. union_offset + slice_size - 1,          slice_size != 0
static void unionFirstProducerUser(RewriterBase &rewriter,
                                   tensor::ExtractSliceOp firstSliceOp,
                                   SmallVector<Value> &unionOffsets,
                                   SmallVector<Value> &unionMaxes) {
  LDBG("first SliceOp \n" << firstSliceOp);
  rewriter.setInsertionPoint(firstSliceOp);
  auto sliceOffsets = getValueOrCreateConstantIndexOp(
      rewriter, firstSliceOp.getLoc(), firstSliceOp.getMixedOffsets());
  auto sliceSizes = getValueOrCreateConstantIndexOp(
      rewriter, firstSliceOp.getLoc(), firstSliceOp.getMixedSizes());
  auto srcMixedSizes = tensor::getMixedSizes(rewriter, firstSliceOp.getLoc(),
                                             firstSliceOp.getSource());
  auto srcSizes = getValueOrCreateConstantIndexOp(
      rewriter, firstSliceOp.getLoc(), srcMixedSizes);

  auto isSizesZero =
      createEqualZeroOp(sliceSizes, rewriter, firstSliceOp->getLoc());

  // unionOffsets = select(sliceSlize == 0, MAX_VALUE, sliceOffsets),
  // where `MAX_VALUE` is the srcSizes of slice offset.
  unionOffsets = createSelectOp(isSizesZero, srcSizes, sliceOffsets, rewriter,
                                firstSliceOp->getLoc());

  // unionMaxes = select(sliceSlize == 0, MIN_VALUE, initMaxes),
  // where `MIN_VALUE` is the sliceSizes when sliceSize is 0.
  auto initMaxes = compute(rewriter, MODE::COMPUTE_SLICE_MAX, unionOffsets,
                           sliceSizes, firstSliceOp->getLoc());
  unionMaxes = createSelectOp(isSizesZero, sliceSizes, initMaxes, rewriter,
                              firstSliceOp->getLoc());
}

/// Updates the union offsets and maxes based on slice offsets and sizes.
/// The updated union_offsets should be
/// 1. origin union_offsets,                              slice_size == 0
/// 2. min(union_offsets, slice_offsets),                 slice_size != 0
/// The updated union maxes should be
/// 1. origin union_maxes,                                slice_size == 0
/// 2. max(union_maxes, slice_offsets + slice_sizes - 1), slice_size != 0
static void unionNextProducerUser(RewriterBase &rewriter, Location loc,
                                  const SmallVector<Value> &offsets,
                                  const SmallVector<Value> &sizes,
                                  SmallVector<Value> &unionOffsets,
                                  SmallVector<Value> &unionMaxes) {
  auto isSizesZero = createEqualZeroOp(sizes, rewriter, loc);
  // union offsets
  // unionOffsets = select(sliceSlize == 0, unionOffsets, newOffsets),
  auto newOffsets =
      createSelectOp(isSizesZero, unionOffsets, offsets, rewriter, loc);
  unionOffsets =
      compute(rewriter, MODE::UNION_MIN, unionOffsets, newOffsets, loc);
  // union maxes
  // unionMaxes = select(sliceSlize == 0, unionMaxes, computeMaxes),
  auto computeMaxes =
      compute(rewriter, MODE::COMPUTE_SLICE_MAX, newOffsets, sizes, loc);
  auto clonedMaxes =
      createSelectOp(isSizesZero, unionMaxes, computeMaxes, rewriter, loc);
  unionMaxes = compute(rewriter, MODE::UNION_MAX, unionMaxes, clonedMaxes, loc);
}

/// Adjust the origin slice op by slicing from unioned slice.
/// The adjusted slice_offsets should be
/// 1. MAX_OFFSET(default to union_offsets), slice_size == 0
/// 2. slice_offsets - union_offsets,        slice_size != 0
/// The slice_sizes remain the same
static tensor::ExtractSliceOp
sliceFromUnion(RewriterBase &rewriter, tensor::ExtractSliceOp unionSlice,
               const SmallVector<Value> &unionOffsets,
               tensor::ExtractSliceOp sliceOp) {
  rewriter.setInsertionPoint(sliceOp.getOperation());

  auto offsets = getValueOrCreateConstantIndexOp(rewriter, sliceOp.getLoc(),
                                                 sliceOp.getMixedOffsets());
  auto sizes = getValueOrCreateConstantIndexOp(rewriter, sliceOp.getLoc(),
                                               sliceOp.getMixedSizes());
  auto isSizesZero = createEqualZeroOp(sizes, rewriter, sliceOp->getLoc());
  // if current slice user is zero sized, it should not be unioned, so
  // simpliy reset its start offset to the unioned offset
  offsets = createSelectOp(isSizesZero, unionOffsets, offsets, rewriter,
                           sliceOp->getLoc());
  auto newOffsets = compute(rewriter, MODE::COMPUTE_SUB, offsets, unionOffsets,
                            sliceOp->getLoc());
  auto newSlice = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp.getLoc(), unionSlice.getResult(), convert(newOffsets),
      sliceOp.getMixedSizes(), unionSlice.getMixedStrides());
  return newSlice;
}

void bishengir::unionProducerUsers(RewriterBase &rewriter, Diagnostic &diag,
                                   Operation *producerOp,
                                   Operation *containingOp) {
  llvm::DenseMap<Value, tensor::ExtractSliceOp> result2FirstSliceOp;
  llvm::DenseMap<Value, int> result2ValidNum;
  getFirstSliceUserInContainingOp(producerOp, containingOp,
                                  &result2FirstSliceOp, &result2ValidNum);

  for (auto produceResult : producerOp->getResults()) {
    int validSliceOpNum = result2ValidNum[produceResult];
    LDBG("produce res : " << produceResult
                          << ", slice op number : " << validSliceOpNum);
    if (validSliceOpNum < 2) { // minimum 2 valid slice ops
      continue;
    }
    assert(result2FirstSliceOp.find(produceResult) !=
           result2FirstSliceOp.end());
    auto firstSliceOp = result2FirstSliceOp[produceResult];

    // init unionOffsets and unionMaxes based on the first producer slice user
    SmallVector<Value> unionOffsets;
    SmallVector<Value> unionMaxes;
    LDBG("begin to union \n" << *containingOp);
    unionFirstProducerUser(rewriter, firstSliceOp, unionOffsets, unionMaxes);

    for (auto *user : produceResult.getUsers()) {
      auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!isValidSliceOpInContainingOp(sliceOp, containingOp) ||
          sliceOp == firstSliceOp) {
        continue;
      }

      LDBG("union slice \n" << sliceOp);
      // get and clone offsets if it is defined below inserted point
      auto curOffsets = getValueOrCreateConstantIndexOp(
          rewriter, sliceOp->getLoc(), sliceOp.getMixedOffsets());
      auto clonedOffsets =
          recursiveClone(rewriter, curOffsets, firstSliceOp.getOperation());
      // get and clone sizes if it is defined below inserted point
      auto curSizes = getValueOrCreateConstantIndexOp(
          rewriter, sliceOp.getLoc(), sliceOp.getMixedSizes());
      auto clonedSizes =
          recursiveClone(rewriter, curSizes, firstSliceOp.getOperation());

      // update unionOffsets and unionMaxes based on next producer slice user
      unionNextProducerUser(rewriter, sliceOp->getLoc(), clonedOffsets,
                            clonedSizes, unionOffsets, unionMaxes);
    }

    auto unionSizes = compute(rewriter, MODE::COMPUTE_DISTANCE, unionMaxes,
                              unionOffsets, firstSliceOp->getLoc());

    auto unionSlice = rewriter.create<tensor::ExtractSliceOp>(
        firstSliceOp.getLoc(), firstSliceOp.getSource(), convert(unionOffsets),
        convert(unionSizes), firstSliceOp.getMixedStrides());

    LDBG("insert union slice \n" << unionSlice);
    LDBG(*containingOp);

    // update users to use union slice result
    for (auto *user : llvm::make_early_inc_range(produceResult.getUsers())) {
      auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!isValidSliceOpInContainingOp(sliceOp, containingOp) ||
          sliceOp == unionSlice) {
        continue;
      }
      auto newSliceOp =
          sliceFromUnion(rewriter, unionSlice, unionOffsets, sliceOp);
      rewriter.replaceOp(sliceOp.getOperation(), newSliceOp.getResult());
    }
    LDBG("unioned containingOp: \n" << *containingOp);
  }
}

//===----------------------------------------------------------------------===//
// This file contains code from the LLVM Project.
// Original License: Apache License v2.0 with LLVM Exceptions
// Original Copyright: NA
// Original Source:
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp
//===----------------------------------------------------------------------===//
Operation *replaceForAllWithNewSignature(
    RewriterBase &rewriter, Diagnostic &diag, Operation *producerOp,
    Operation *containingOp, TilingResult &tileAndFuseResult,
    int64_t resultNumber, SmallVector<OpFoldResult> &offsets,
    SmallVector<OpFoldResult> &sizes) {
  // Count number of users not including the containing op
  SetVector<Operation *> dominatedUsers;
  DominanceInfo domInfo(containingOp);
  for (Operation *user : producerOp->getResult(resultNumber).getUsers()) {
    if (!containingOp->isAncestor(user) &&
        (domInfo.dominates(containingOp, user))) {
      dominatedUsers.insert(user);
    }
  }
  if (dominatedUsers.empty())
    return nullptr;

  // Create new scf.forall op
  auto forallOp = cast<scf::ForallOp>(containingOp);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forallOp);

  // Get new output
  Location loc = forallOp.getLoc();
  auto genericOp = dyn_cast<linalg::GenericOp>(producerOp);
  if (!genericOp)
    return nullptr;
  SmallVector<Value> outputs = genericOp.getOutputs();
  SmallVector<Value> newOuts(forallOp.getOutputs());
  newOuts.push_back(outputs[resultNumber]);

  // Create new scf.forall op
  auto newforallOp = rewriter.create<scf::ForallOp>(
      loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
      forallOp.getMixedStep(), newOuts, forallOp.getMapping());
  rewriter.eraseBlock(newforallOp.getBody());
  newforallOp.getRegion().takeBody(forallOp.getRegion());

  // Add additional block argument for new value being returned
  // and replaces all uses of the new output with corresponding bbArg
  // inside the scf.forall to enable fusion into this new scf.forall.
  newforallOp.getBody()->addArgument(newOuts.back().getType(),
                                     newOuts.back().getLoc());
  auto bbArgs = newforallOp.getBody()->getArguments();
  rewriter.replaceUsesWithIf(newOuts.back(), bbArgs.back(),
                             [&](OpOperand &use) {
                               Operation *op = use.getOwner();
                               return newforallOp->isProperAncestor(op);
                             });

  // Fix terminator
  scf::InParallelOp terminatorOp = newforallOp.getTerminator();
  SmallVector<Operation *> yieldingOps = llvm::to_vector<4>(llvm::map_range(
      terminatorOp.getYieldingOps(), [](Operation &op) { return &op; }));
  Operation *firstYieldOp = yieldingOps.front();
  rewriter.setInsertionPoint(firstYieldOp);
  Value src = tileAndFuseResult.tiledValues[0];
  Value dst = newforallOp.getRegionIterArgs().back();
  SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
  rewriter.create<tensor::ParallelInsertSliceOp>(firstYieldOp->getLoc(), src,
                                                 dst, offsets, sizes, strides);

  for (auto result : llvm::enumerate(forallOp.getResults())) {
    rewriter.replaceAllUsesWith(result.value(),
                                newforallOp->getResult(result.index()));
  }
  rewriter.replaceUsesWithIf(producerOp->getResult(resultNumber),
                             newforallOp->getResults().back(),
                             [&](OpOperand &use) {
                               Operation *user = use.getOwner();
                               return dominatedUsers.contains(user);
                             });
  return newforallOp;
}

/// Find the first "extract" user of `producerOp` and tile it right before its
/// use. The tiled op is fused under the `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
/// If tiled op has uses that are dominated by `containingOp`, return
/// a new `containingOp` with results of the fused op appended to
/// results of the `containingOp` or nullptr if there are no dominated uses.
/// However, if `duplicateProducer` is set to true, then the `producerOp` is
/// expected to be tiled and fused into all users.
std::tuple<SmallVector<Operation *>, Operation *>
bishengir::tileAndFuseFirstExtractUse(RewriterBase &rewriter, Diagnostic &diag,
                                      Operation *producerOp,
                                      Operation *containingOp,
                                      bool duplicateProducer) {
  LLVM_DEBUG(DBGS() << "Try to fuse a direct extract use\n");
  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TileableInterface: " << *producerOp;
    return {};
  }

  // Search the producer slices accessed within the containing operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples, maybe
  // evolve into an interface.
  auto it = llvm::find_if(tileableProducer->getUsers(), [&](Operation *user) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });
  // Find a fusion opportunity.
  if (it == tileableProducer->getUsers().end()) {
    diag.attachNote(tileableProducer->getLoc())
        << "could not find fusion opportunity for: " << *tileableProducer;
    return {};
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*it);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Tile the producer.
  int64_t resultNumber =
      cast<OpResult>(sliceOpToTile.getSource()).getResultNumber();
  LLVM_DEBUG(DBGS() << "resultNumber: " << resultNumber << "\n");

  SmallVector<OpFoldResult> offsets = sliceOpToTile.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = sliceOpToTile.getMixedSizes();

  FailureOr<TilingResult> tileAndFuseResult =
      tileableProducer.generateResultTileValue(rewriter, resultNumber, offsets,
                                               sizes);

  if (failed(tileAndFuseResult)) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to tile producer op: " << *tileableProducer;
    return {};
  }

#ifndef NDEBUG
  for (auto *tiledOp : tileAndFuseResult->tiledOps) {
    LLVM_DEBUG(DBGS() << "tiledProducer: " << *tiledOp << "\n");
  }
#endif

  // Replace the extract op.
  auto maybeRankReduced = tensor::ExtractSliceOp::rankReduceIfNeeded(
      rewriter, sliceOpToTile->getLoc(), tileAndFuseResult->tiledValues[0],
      cast<RankedTensorType>(sliceOpToTile->getResult(0).getType()).getShape());
  if (failed(maybeRankReduced)) {
    diag.attachNote(producerOp->getLoc())
        << "shape types don't match (missing canonicalization?):\nTiledOp: "
        << tileAndFuseResult->tiledValues[0]
        << "\nSliceOp: " << sliceOpToTile.getOperation() << '\n';
    return {};
  }
  rewriter.replaceOp(sliceOpToTile, *maybeRankReduced);

  if (duplicateProducer)
    return std::make_tuple(tileAndFuseResult->tiledOps, nullptr);

  // Add new outputs to containing op, if required
  Operation *newContainingOp = replaceForAllWithNewSignature(
      rewriter, diag, producerOp, containingOp, *tileAndFuseResult,
      resultNumber, offsets, sizes);

  return std::make_tuple(tileAndFuseResult->tiledOps, newContainingOp);
}

/// First, find the first "scf::ForallOp" user of `producerOp` and ensure
/// it is exactly the `containingOp`, otherwise bail.
/// Then, find the first "extract" user of the tied block argument and tile it
/// right before its "extract" use. The tiled op is fused under the
/// `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
SmallVector<Operation *>
bishengir::tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
    RewriterBase &rewriter, Diagnostic &diag, Operation *producerOp,
    Operation *containingOp) {
  LLVM_DEBUG(DBGS() << "Try to fuse an extract use through block argument\n");

  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TileableInterface: " << *producerOp;
    return {};
  }

  // Search the first use by a "scf::ForallOp" user.
  scf::ForallOp forallOp;
  auto itProducerUses =
      llvm::find_if(tileableProducer->getUses(), [&](OpOperand &use) {
        forallOp = dyn_cast<scf::ForallOp>(use.getOwner());
        return forallOp;
      });
  // If it's not from the containing op, return.
  if (!forallOp || forallOp != containingOp) {
    diag.attachNote(tileableProducer->getLoc())
        << "could not find a use by the containing op: " << *tileableProducer;
    return {};
  }

  // Search the producer slices accessed within the containing
  // operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples.
  //   Maybe evolve into an interface.
  OpOperand *pUse = &(*itProducerUses);
  BlockArgument bbArg = forallOp.getTiedBlockArgument(pUse);

  // Search the producer slices accessed within the containing operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples, maybe
  // evolve into an interface.
  auto itBBArgUsers = llvm::find_if(bbArg.getUsers(), [&](Operation *user) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });
  // Find a fusion opportunity.
  if (itBBArgUsers == bbArg.getUsers().end()) {
    diag.attachNote(containingOp->getLoc())
        << "could not find fusion opportunity for bbArg: " << bbArg;
    return {};
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*itBBArgUsers);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Replace the use in the tileableProducer before tiling: clone, replace and
  // then tile.
  int64_t resultNumber = cast<OpResult>(pUse->get()).getResultNumber();
  LLVM_DEBUG(DBGS() << "resultNumber: " << resultNumber << "\n");

  // Gather destination tensors.
  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(
          rewriter, tileableProducer->getLoc(), tileableProducer,
          destinationTensors))) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to get destination tensors for: " << *tileableProducer;
    return {};
  }

  IRMapping bvm;
  bvm.map(destinationTensors[resultNumber], bbArg);
  auto tileableProducerClone =
      cast<TilingInterface>(rewriter.clone(*tileableProducer, bvm));
  auto scopeGuard =
      llvm::make_scope_exit([&]() { rewriter.eraseOp(tileableProducerClone); });

  // Tile the producer.
  FailureOr<TilingResult> tileAndFuseResult =
      tileableProducerClone.generateResultTileValue(
          rewriter, resultNumber, sliceOpToTile.getMixedOffsets(),
          sliceOpToTile.getMixedSizes());
  if (failed(tileAndFuseResult)) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to tile producer op: " << *tileableProducer;
    return {};
  }

  // Replace the extract op.
  auto maybeRankReduced = tensor::ExtractSliceOp::rankReduceIfNeeded(
      rewriter, sliceOpToTile->getLoc(), tileAndFuseResult->tiledValues[0],
      cast<RankedTensorType>(sliceOpToTile->getResult(0).getType()).getShape());
  assert(succeeded(maybeRankReduced) && "unexpected shape");
  rewriter.replaceOp(sliceOpToTile, *maybeRankReduced);

  // Replace the use in containingOp.
  rewriter.modifyOpInPlace(containingOp, [&]() {
    containingOp->setOperand(pUse->getOperandNumber(),
                             destinationTensors.front());
  });

  return tileAndFuseResult->tiledOps;
}

Operation *bishengir::cloneAndFuseFirstUse(RewriterBase &rewriter,
                                           Diagnostic &diag,
                                           Operation *producerOp,
                                           Operation *containingOp) {
  LLVM_DEBUG(DBGS() << "Try to fuse an use by cloning\n");

  // Gather all uses inside the containing op.
  SmallVector<OpOperand *> uses;
  for (OpResult result : producerOp->getOpResults()) {
    for (OpOperand &use : result.getUses()) {
      if (containingOp->isProperAncestor(use.getOwner())) {
        uses.push_back(&use);
        continue;
      }
      // Cannot clone and fuse if the use is by the containing op itself: fail
      // immediately.
      if (containingOp == use.getOwner()) {
        diag.attachNote(producerOp->getLoc())
            << "producer op use by containing op cannot be fused by cloning";
        return nullptr;
      }
    }
  }

  // Check for a non-empty list of fusion opportunities.
  if (uses.empty()) {
    diag.attachNote(producerOp->getLoc()) << "no fusion opportunity by cloning";
    return nullptr;
  }

  // Clone and fuse inside the containing op.
  Operation *fusedOp = nullptr;
  OpOperand *use = uses.front();
  // Parallel insert slice is not a valid clone destination.
  // TODO: Generalize to other type of ops.
  assert(!isa<tensor::ParallelInsertSliceOp>(use->getOwner()) &&
         "Parallel insert slice is not a valid clone destination");
  unsigned resultNumber = cast<OpResult>(use->get()).getResultNumber();
  LLVM_DEBUG(DBGS() << "resultNumber: " << resultNumber << "\n");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(use->getOwner());
  fusedOp = rewriter.clone(*producerOp);
  rewriter.modifyOpInPlace(
      use->getOwner(), [&] { use->set(fusedOp->getOpResult(resultNumber)); });

  return fusedOp;
}

//===----------------------------------------------------------------------===//
// fuseNestedSiblingLoops helpers
//===----------------------------------------------------------------------===//

static bool isProducerConsumed(Operation *producer, Operation *consumer) {
  SmallPtrSet<Operation *, 8> visited{producer};
  SmallVector<Operation *> worklist{producer};
  while (!worklist.empty()) {
    for (auto *user : worklist.pop_back_val()->getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (consumer->isAncestor(user))
        return true;
      worklist.push_back(user);
    }
  }
  return false;
}

static SmallVector<SmallVector<scf::ForOp>>
groupInnerSiblingLoops(SmallVector<scf::ForOp> &loops) {
  SmallVector<SmallVector<scf::ForOp>> result;
  for (auto &loop : loops) {
    bool added = false;
    for (auto &group : result) {
      if (llvm::all_of(group, [&](scf::ForOp other) {
            return !isProducerConsumed(loop, other) &&
                   !isProducerConsumed(other, loop);
          })) {
        group.push_back(loop);
        added = true;
        break;
      }
    }
    if (!added)
      result.push_back({loop});
  }
  return result;
}

// Groups sibling loops by iteration count. Each group is paired with an
// optional iters value that signals whether normalization is needed:
// - iters.has_value() — loops have the same count but differing bounds (cnt %
//   step == 0); caller normalizes to uniform bounds before fusing.
// - iters == nullopt  — loops share identical (lb, ub, step), so bounds are
//   already uniform; normalization is skipped to preserve the tail-block signal
//   (cnt % step != 0) for downstream passes.
static SmallVector<std::pair<SmallVector<scf::ForOp>, std::optional<int64_t>>>
groupByIterCount(SmallVector<scf::ForOp> &loops) {
  enum CmpK { Signed, Unsigned };
  llvm::MapVector<std::pair<int64_t, CmpK>, SmallVector<scf::ForOp>>
      alignedGroups;
  using BoundsKey = std::tuple<int64_t, int64_t, int64_t, int64_t>;
  llvm::MapVector<BoundsKey, SmallVector<scf::ForOp>> tailGroups;
  for (auto &loop : loops) {
    auto lb = getConstantIntValue(loop.getLowerBound());
    auto ub = getConstantIntValue(loop.getUpperBound());
    auto st = getConstantIntValue(loop.getStep());
    // TODO: getUnsignedCmp not implemented
    // CmpK ucmp = loop.getUnsignedCmp() ? Unsigned : Signed;
    CmpK ucmp = Signed;
    if (!lb || !ub || !st)
      continue;
    int64_t cnt = *ub - *lb;
    if (cnt == 0 || *st == 0 || (cnt > 0) != (*st > 0))
      continue;
    if (cnt % *st == 0)
      alignedGroups[{llvm::divideCeilSigned(cnt, *st), ucmp}].push_back(loop);
    else
      tailGroups[{*lb, *ub, *st, ucmp}].push_back(loop);
  }
  SmallVector<std::pair<SmallVector<scf::ForOp>, std::optional<int64_t>>>
      result;
  for (auto &[k, v] : alignedGroups)
    result.push_back({std::move(v), k.first});
  for (auto &v : llvm::make_second_range(tailGroups))
    result.push_back({std::move(v), std::nullopt});
  return result;
}

// Normalize all loops in the group to shared GCD bounds so they can be fused.
// See normalizeGroupBounds in SCFTransformOps for the full algorithm comment.
static void normalizeGroupBounds(RewriterBase &rewriter,
                                 SmallVector<scf::ForOp> &group,
                                 int64_t iters) {
  int64_t gcdStep = 0;
  for (auto &l : group)
    gcdStep = std::gcd(gcdStep, getConstantIntValue(l.getStep()).value());

  Location loc = group.front().getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(group.front().getOperation());
  Value newLb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value newUb = rewriter.create<arith::ConstantIndexOp>(loc, iters * gcdStep);
  Value newSt = rewriter.create<arith::ConstantIndexOp>(loc, gcdStep);

  for (auto &loop : group) {
    auto lb = getConstantIntValue(loop.getLowerBound()).value();
    auto st = getConstantIntValue(loop.getStep()).value();
    int64_t factor = st / gcdStep;

    rewriter.modifyOpInPlace(loop, [&]() {
      loop.setLowerBound(newLb);
      loop.setUpperBound(newUb);
      loop.setStep(newSt);
    });

    if (lb == 0 && factor == 1)
      continue;

    rewriter.setInsertionPointToStart(loop.getBody());

    AffineExpr remap =
        rewriter.getAffineConstantExpr(lb) +
        rewriter.getAffineDimExpr(0) * rewriter.getAffineConstantExpr(factor);
    auto map = AffineMap::get(1, 0, remap, rewriter.getContext());

    Value inductionVar = loop.getInductionVar();
    auto affineApplyOp = rewriter.create<affine::AffineApplyOp>(
        loop.getLoc(), map, inductionVar);
    rewriter.replaceUsesWithIf(inductionVar, affineApplyOp,
                               [&](OpOperand &operand) {
                                 return operand.getOwner() != affineApplyOp;
                               });
  }
}

static void climbedForwardSlice(Operation *op,
                                SetVector<Operation *> &forwardSlice,
                                const DominanceInfo &domInfo,
                                Operation *fusedLoop) {
  while (op->getParentOp() != fusedLoop->getParentOp())
    op = op->getParentOp();
  if (forwardSlice.count(op) || domInfo.properlyDominates(fusedLoop, op))
    return;
  for (Operation *userOp : op->getUsers())
    climbedForwardSlice(userOp, forwardSlice, domInfo, fusedLoop);
  forwardSlice.insert(op);
}

static void adjustEarlierLoopUsers(Operation *source, Operation *target,
                                   Operation *fusedLoop,
                                   RewriterBase &rewriter) {
  Operation *earlier = source->isBeforeInBlock(target) ? source : target;
  DominanceInfo domInfo(fusedLoop);
  SetVector<Operation *> forwardSlice;
  climbedForwardSlice(earlier, forwardSlice, domInfo, fusedLoop);
  for (auto *userOp : drop_end(forwardSlice))
    rewriter.moveOpAfter(userOp, fusedLoop);
}

static scf::ForOp fuseSiblingForLoops(scf::ForOp target, scf::ForOp source,
                                      RewriterBase &rewriter) {
  unsigned numTargetOuts = target.getNumResults();
  unsigned numSourceOuts = source.getNumResults();

  SmallVector<Value> fusedInitArgs;
  llvm::append_range(fusedInitArgs, target.getInitArgs());
  llvm::append_range(fusedInitArgs, source.getInitArgs());

  rewriter.setInsertionPointAfter(source->isBeforeInBlock(target) ? target
                                                                  : source);

  scf::ForOp fusedLoop = rewriter.create<scf::ForOp>(
      source.getLoc(), source.getLowerBound(), source.getUpperBound(),
      source.getStep(), fusedInitArgs,
      /*bodyBuilder=*/nullptr);

  IRMapping mapping;
  mapping.map(target.getInductionVar(), fusedLoop.getInductionVar());
  mapping.map(target.getRegionIterArgs(),
              fusedLoop.getRegionIterArgs().take_front(numTargetOuts));
  mapping.map(source.getInductionVar(), fusedLoop.getInductionVar());
  mapping.map(source.getRegionIterArgs(),
              fusedLoop.getRegionIterArgs().take_back(numSourceOuts));

  rewriter.setInsertionPointToStart(fusedLoop.getBody());
  for (Operation &op : target.getBody()->without_terminator())
    rewriter.clone(op, mapping);
  for (Operation &op : source.getBody()->without_terminator())
    rewriter.clone(op, mapping);

  SmallVector<Value> yieldResults;
  for (Value operand : target.getBody()->getTerminator()->getOperands())
    yieldResults.push_back(mapping.lookupOrDefault(operand));
  for (Value operand : source.getBody()->getTerminator()->getOperands())
    yieldResults.push_back(mapping.lookupOrDefault(operand));
  if (!yieldResults.empty())
    rewriter.create<scf::YieldOp>(source.getLoc(), yieldResults);

  adjustEarlierLoopUsers(source, target, fusedLoop, rewriter);
  fusedLoop->setAttrs(source->getAttrs());
  for (auto &attr : target->getAttrs())
    fusedLoop->setAttr(attr.getName(), attr.getValue());

  rewriter.replaceOp(target, fusedLoop.getResults().take_front(numTargetOuts));
  rewriter.replaceOp(source, fusedLoop.getResults().take_back(numSourceOuts));

  return fusedLoop;
}

static scf::ForOp loopsFusion(SmallVector<scf::ForOp> &group, size_t lo,
                              size_t hi, RewriterBase &rw) {
  return lo + 1 < hi
             ? fuseSiblingForLoops(loopsFusion(group, lo, (lo + hi) / 2, rw),
                                   loopsFusion(group, (lo + hi) / 2, hi, rw),
                                   rw)
             : group[lo];
}

void bishengir::normalizeLoop(RewriterBase &rewriter, scf::ForOp op,
                              Value oldStep) {
  MLIRContext *ctx = rewriter.getContext();
  Location loopLoc = op.getLoc();
  // compute new upperbound as oldUB ceildiv oldStep
  AffineExpr symbolUB = getAffineSymbolExpr(0, ctx);
  AffineExpr symbolStep = getAffineSymbolExpr(1, ctx);
  AffineExpr UBCalculation = symbolUB.ceilDiv(symbolStep);
  Value newStep = rewriter.create<arith::ConstantIndexOp>(loopLoc, 1);
  Value newUB = rewriter.create<affine::AffineApplyOp>(
      loopLoc, UBCalculation, ValueRange{op.getUpperBound(), oldStep});

  // updates loop ub and steps in place
  rewriter.modifyOpInPlace(op, [&]() {
    op.getUpperBoundMutable().assign(newUB);
    op.getStepMutable().assign(newStep);
  });

  // update iv, so loop body's behavior stays the same
  rewriter.setInsertionPointToStart(op.getBody());
  AffineExpr newIVCalculation = symbolUB * symbolStep;
  Value newIV = rewriter.create<affine::AffineApplyOp>(
      loopLoc, newIVCalculation, ValueRange{op.getInductionVar(), oldStep});
  rewriter.replaceAllUsesExcept(op.getInductionVar(), newIV,
                                newIV.getDefiningOp());
}

void bishengir::fuseNestedSiblingLoops(Operation *fusedOp,
                                       RewriterBase &rewriter, bool recursive) {
  auto outerForOp = dyn_cast<scf::ForOp>(fusedOp);
  if (!outerForOp)
    return;
  auto innerForOps =
      llvm::to_vector(outerForOp.getBody()->getOps<scf::ForOp>());
  if (innerForOps.empty())
    return;

  for (auto &[sameItersGroup, iters] : groupByIterCount(innerForOps)) {
    if (sameItersGroup.size() < 2)
      continue;
    for (auto &siblingGroup : groupInnerSiblingLoops(sameItersGroup)) {
      if (siblingGroup.size() < 2)
        continue;
      if (iters)
        normalizeGroupBounds(rewriter, siblingGroup, *iters);
      auto fused = loopsFusion(siblingGroup, 0, siblingGroup.size(), rewriter);
      if (recursive)
        fuseNestedSiblingLoops(fused, rewriter, recursive);
    }
  }
}
