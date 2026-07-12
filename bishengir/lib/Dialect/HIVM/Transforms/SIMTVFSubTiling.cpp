//===- SIMTVFSubTiling.cpp -----------------------------------------------===//
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
//
// This pass serially splits oversized tiles inside SIMT VF functions.
//
// High-level idea:
// 1. Stay on HIVM, before SIMT mem-scope materialization and TTIR lowering, so
//    the transform can still reason in terms of tensor/memref slices.
// 2. Use DimensionAnalyzer with hivm.local_store enabled as a store-like sink,
//    and only accept SIMT VF functions whose local_store boundaries agree on
//    one oversized tiling dimension and the same dimension extent.
// 3. Clone the function into a serial scf.for with split count
//    dimSize / maxTileSize. Only evenly divisible cases are rewritten, so odd
//    tails are skipped. Unlike 1:2 sub-block tiling, all iterations still
//    execute on the same sub-block; only the tile size is reduced.
// 4. Seed per-tile extract_slice/subview at each local_store boundary, then
//    reuse the existing BubbleUpExtractSlice infrastructure to propagate those
//    slices upward through the producer DAG.
// 5. If DimensionAnalyzer cannot derive a tiling dimension, or if any rewrite
//    or bubble-up step fails, erase the clone and keep the original SIMT VF
//    unchanged.
//
// Sketch:
//
//   before sub-tiling:
//     %ub = load %gm -> %tmp_ub
//     %t  = to_tensor %tmp_ub
//     local_store %dst_ub, %t              // tensor<192xf32>
//
//   after sub-tiling (maxTileSize = 64, splitNum = 3):
//     scf.for %tile = 0 to 3 step 1 {
//       %off = %tile * 64
//       // tile size is always 64
//       %src_sub = subview %gm[%off] [%tile_size]
//       %tmp_sub = subview %tmp_ub[%off] [%tile_size]
//       load %src_sub -> %tmp_sub
//       %t_sub = to_tensor %tmp_sub        // tensor<64xf32>
//       %dst_sub = subview %dst_ub[%off] [%tile_size]
//       local_store %dst_sub, %t_sub
//     }

#include "bishengir/Dialect/HIVM/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/CSEPattern.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/HoistAffine.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/Pattern.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <memory>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_SIMTVFSUBTILING
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-simt-vf-sub-tiling"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct TilePlan {
  int64_t tilingDim;
  int64_t dimSize;
  int64_t splitNum;
};

bool isAllowedWriteEffect(Operation *op) {
  return isa<hivm::LoadOp, hivm::GatherLoadOp, hivm::LocalStoreOp>(op);
}

bool hasUnsupportedWriteEffect(func::FuncOp func) {
  bool hasUnsupported = false;
  func.walk([&](Operation *op) {
    if (hasUnsupported)
      return WalkResult::interrupt();
    auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectOp)
      return WalkResult::advance();

    SmallVector<MemoryEffects::EffectInstance> effects;
    effectOp.getEffects(effects);
    bool writes = llvm::any_of(effects, [](const auto &effect) {
      return isa<MemoryEffects::Write>(effect.getEffect());
    });
    if (writes && !isAllowedWriteEffect(op)) {
      hasUnsupported = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasUnsupported;
}

std::optional<TilePlan> analyzeTilePlan(func::FuncOp func,
                                        int64_t maxTileSize) {
  // Only rewrite simple SIMT VFs that can be cloned wholesale and then
  // narrowed at the local_store boundary.
  if (maxTileSize <= 0 || !llvm::hasSingleElement(func.getBody()))
    return std::nullopt;
  auto returnOp = dyn_cast<func::ReturnOp>(func.front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 0)
    return std::nullopt;
  if (hasUnsupportedWriteEffect(func))
    return std::nullopt;

  hivm::detail::DimensionAnalyzer analyzer(func);
  if (failed(analyzer.initialize()))
    return std::nullopt;
  analyzer.computeTilingDim(/*isVectorOp=*/true);

  TilePlan plan;
  bool foundBoundary = false;
  bool isEligible = true;
  auto walkResult = func.walk([&](Operation *op) {
    auto boundaryWrite = dyn_cast<hivm::LocalStoreOp>(op);
    if (!boundaryWrite)
      return WalkResult::advance();

    // All rewritten boundary writes must agree on the same oversized tiling
    // dimension so one serial loop can cover the whole SIMT VF.
    Value src = boundaryWrite.getData();
    auto srcType = cast<RankedTensorType>(src.getType());
    if (!srcType.hasStaticShape()) {
      isEligible = false;
      return WalkResult::interrupt();
    }

    int64_t tilingDim = analyzer.getTilingDim(src);
    if (tilingDim < 0) {
      LDBG("Skip SIMT VF sub-tiling for " << func.getSymName()
                                          << ": failed to derive tiling dim "
                                             "from local_store boundary "
                                          << boundaryWrite);
      isEligible = false;
      return WalkResult::interrupt();
    }
    int64_t dimSize = srcType.getDimSize(tilingDim);
    if (dimSize <= maxTileSize) {
      isEligible = false;
      return WalkResult::interrupt();
    }
    if (dimSize % maxTileSize != 0) {
      LDBG("Skip SIMT VF sub-tiling for " << func.getSymName()
                                          << ": odd tail is not supported yet "
                                          << "(dimSize="
                                          << dimSize
                                          << ", maxTileSize="
                                          << maxTileSize << ")");
      isEligible = false;
      return WalkResult::interrupt();
    }

    if (!foundBoundary) {
      plan.tilingDim = tilingDim;
      plan.dimSize = dimSize;
      foundBoundary = true;
    } else if (plan.tilingDim != tilingDim || plan.dimSize != dimSize) {
      isEligible = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!isEligible || walkResult.wasInterrupted() || !foundBoundary)
    return std::nullopt;

  plan.splitNum = llvm::divideCeilSigned(plan.dimSize, maxTileSize);
  return plan;
}

scf::ForOp createTileLoop(Location loc, OpBuilder &builder, int64_t splitNum) {
  auto lowerBound = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto upperBound = builder.create<arith::ConstantIndexOp>(loc, splitNum);
  auto step = builder.create<arith::ConstantIndexOp>(loc, 1);
  auto loop = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step);
  // Unlike 1:2 sub-block tiling, sub-tiling stays on the same sub-block and
  // serializes the oversized tile into splitNum smaller slices.
  loop->setAttr(kSimtVFTileLoopAttrName, builder.getUnitAttr());
  return loop;
}

LogicalResult replicateMultiUseViews(func::FuncOp func) {
  IRRewriter rewriter(func.getContext());
  SmallVector<Operation *> multiUseViews;
  func.walk([&](Operation *op) {
    if (isa<tensor::ExtractSliceOp, memref::SubViewOp>(op) && !op->hasOneUse())
      multiUseViews.push_back(op);
  });

  for (Operation *op : multiUseViews) {
    rewriter.setInsertionPoint(op);
    SmallVector<OpOperand *> uses;
    for (auto &use : op->getUses())
      uses.push_back(&use);
    for (OpOperand *use : uses) {
      use->set(rewriter.clone(*op)->getResult(0));
    }
  }
  return success();
}

FailureOr<func::FuncOp> cloneIntoTileLoop(func::FuncOp func, int64_t splitNum) {
  OpBuilder builder(func->getContext());
  builder.setInsertionPoint(func);
  // Rewrite on a cloned function first so any unsupported pattern can simply
  // drop the clone and leave the original SIMT VF untouched.
  auto newFunc = cast<func::FuncOp>(builder.cloneWithoutRegions(func));
  newFunc.setSymName((func.getSymName() + "__simt_tiled").str());
  newFunc.addEntryBlock();
  builder.setInsertionPointToStart(&newFunc.front());

  auto tileLoop = createTileLoop(func.getLoc(), builder, splitNum);
  IRMapping mapping;
  for (auto [idx, arg] : llvm::enumerate(func.getArguments()))
    mapping.map(arg, newFunc.getArgument(idx));
  func.getBody().cloneInto(&tileLoop.getBodyRegion(0), mapping);

  auto &loopBlock = tileLoop.getBodyRegion(0).front();
  auto *clonedBlock = loopBlock.getNextNode();
  if (!clonedBlock)
    return failure();
  auto returnOp = dyn_cast<func::ReturnOp>(clonedBlock->getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 0)
    return failure();

  // cloneInto() leaves the original function body in a nested block after the
  // loop header. Splice that block into the loop body to form the serial tile
  // loop, then restore a single func.return at top level.
  Operation *yieldOp = loopBlock.getTerminator();
  clonedBlock->getOperations().pop_back();
  loopBlock.getOperations().splice(yieldOp->getIterator(),
                                   clonedBlock->getOperations());
  clonedBlock->erase();

  builder.setInsertionPointToEnd(&newFunc.front());
  builder.create<func::ReturnOp>(func.getLoc());
  if (failed(replicateMultiUseViews(newFunc)))
    return failure();
  return newFunc;
}

struct TileWindow {
  SmallVector<OpFoldResult, 4> strides;
  SmallVector<OpFoldResult, 4> offsets;
  SmallVector<OpFoldResult, 4> sizes;
};

FailureOr<TileWindow> computeTileWindow(IRRewriter &rewriter, Location loc,
                                        Value value, int64_t tilingDim,
                                        scf::ForOp containingLoop) {
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (!shapedType)
    return failure();

  auto maybeTileSize =
      getSingleTileSize(rewriter, loc, value, tilingDim, containingLoop);
  if (failed(maybeTileSize))
    return failure();

  rewriter.setInsertionPointToStart(containingLoop.getBody());
  auto offsetAtTileDim = calculateOffsetAtTilingDim(
      rewriter, loc, containingLoop, value, tilingDim);

  TileWindow window;
  SmallVector<int64_t, 4> ignoredShape;
  if (failed(findCorrespondingSizesOffsetsStrides(
          rewriter, shapedType, tilingDim, offsetAtTileDim,
          maybeTileSize.value(), window.strides, window.offsets, window.sizes,
          ignoredShape)))
    return failure();
  return window;
}

FailureOr<Value> createSlicedValue(IRRewriter &rewriter, Location loc,
                                   Value value, ArrayRef<OpFoldResult> offsets,
                                   ArrayRef<OpFoldResult> sizes,
                                   ArrayRef<OpFoldResult> strides) {
  if (isa<RankedTensorType>(value.getType())) {
    auto slice = rewriter.create<tensor::ExtractSliceOp>(loc, value, offsets,
                                                         sizes, strides);
    markCreatedExtractSliceOp(rewriter, slice);
    return slice.getResult();
  }
  if (isa<MemRefType>(value.getType())) {
    auto subView =
        rewriter.create<memref::SubViewOp>(loc, value, offsets, sizes, strides);
    markCreatedExtractSliceOp(rewriter, subView);
    return subView.getResult();
  }
  return failure();
}

// Seed per-tile slices at each local_store boundary so later bubble-up can
// push those tiled views through the producer DAG. The local_store data tensor
// and destination memref must be sliced with the exact same tile window to
// preserve the original write footprint for each serial tile iteration.
//
// Example:
//   %t = ... : tensor<160xf32>
//   hivm.local_store %dst, %t
// =>
//   %t_tile = tensor.extract_slice %t[%off] [%size] [1]
//   %dst_tile = memref.subview %dst[%off] [%size] [1]
//   hivm.local_store %dst_tile, %t_tile
LogicalResult seedLocalStoreBoundarySlices(func::FuncOp func,
                                           const TilePlan &plan) {
  SmallVector<hivm::LocalStoreOp> localStores;
  func.walk([&](hivm::LocalStoreOp op) { localStores.push_back(op); });

  IRRewriter rewriter(func.getContext());
  for (hivm::LocalStoreOp localStore : localStores) {
    auto maybeLoop = findContainingTilingLoop(localStore);
    if (failed(maybeLoop))
      return failure();
    // The rewritten producer chain and the local_store address must use the
    // exact same tile window to preserve the original write footprint.
    auto maybeWindow =
        computeTileWindow(rewriter, localStore.getLoc(), localStore.getData(),
                          plan.tilingDim, maybeLoop.value());
    if (failed(maybeWindow))
      return failure();

    rewriter.setInsertionPoint(localStore);
    auto maybeTiledData = createSlicedValue(
        rewriter, localStore.getLoc(), localStore.getData(),
        maybeWindow->offsets, maybeWindow->sizes, maybeWindow->strides);
    if (failed(maybeTiledData))
      return failure();
    auto maybeTiledAddr = createSlicedValue(
        rewriter, localStore.getLoc(), localStore.getAddr(),
        maybeWindow->offsets, maybeWindow->sizes, maybeWindow->strides);
    if (failed(maybeTiledAddr))
      return failure();

    rewriter.modifyOpInPlace(localStore, [&]() {
      localStore.getAddrMutable().set(maybeTiledAddr.value());
      localStore.getDataMutable().set(maybeTiledData.value());
      localStore->setAttr(tiledOp, rewriter.getUnitAttr());
    });
  }
  return success();
}

void populateProducerBubbleUpPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();

  SmallVector<std::shared_ptr<hivm::detail::BubbleUpStrategy>> strategies;
  strategies.push_back(
      std::make_shared<hivm::detail::BroadcastBubbleUpStrategy>());
  strategies.push_back(
      std::make_shared<hivm::detail::ReduceBubbleUpStrategy>());
  strategies.push_back(
      std::make_shared<hivm::detail::ExpandBubbleUpStrategy>());
  strategies.push_back(
      std::make_shared<hivm::detail::CollapseBubbleUpStrategy>());
  strategies.push_back(
      std::make_shared<hivm::detail::ElementwiseBubbleUpStrategy>());
  strategies.push_back(std::make_shared<hivm::detail::LoopBubbleUpStrategy>());
  strategies.push_back(
      std::make_shared<hivm::detail::LoopArgsBubbleUpStrategy>());
  strategies.push_back(
      std::make_shared<hivm::detail::BitcastBubbleUpStrategy>());
  strategies.push_back(std::make_shared<hivm::detail::IfBubbleUpStrategy>());
  strategies.push_back(
      std::make_shared<hivm::detail::VarangeBubbleUpStrategy>());
  strategies.push_back(std::make_shared<hivm::detail::ScopeBubbleUpStrategy>());
  strategies.push_back(
      std::make_shared<hivm::detail::SelectBubbleUpStrategy>());
  strategies.push_back(
      std::make_shared<hivm::detail::GatherLoadBubbleUpStrategy>());

  patterns.add<hivm::detail::BubbleUpPattern>(context, std::move(strategies));
}

struct MaterializeToTensorSlicePattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!sliceOp->hasAttrOfType<UnitAttr>(toBeBubbleUpSlice))
      return failure();
    if (!sliceOp.hasUnitStride())
      return failure();

    auto toTensorOp = dyn_cast_or_null<bufferization::ToTensorOp>(
        sliceOp.getSource().getDefiningOp());
    if (!toTensorOp)
      return failure();

    IRRewriter irRewriter(rewriter);
    hivm::LoadOp producerLoad = nullptr;
    SmallVector<Operation *> memrefUsers(
        toTensorOp.getMemref().getUsers().begin(),
        toTensorOp.getMemref().getUsers().end());
    for (Operation *user : memrefUsers) {
      auto loadOp = dyn_cast<hivm::LoadOp>(user);
      if (loadOp && loadOp.getDst() == toTensorOp.getMemref()) {
        producerLoad = loadOp;
        break;
      }
    }

    irRewriter.setInsertionPoint(producerLoad ? producerLoad.getOperation()
                                              : sliceOp.getOperation());
    auto maybeDstSubview =
        createSlicedValue(irRewriter, sliceOp.getLoc(), toTensorOp.getMemref(),
                          sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
                          sliceOp.getMixedStrides());
    if (failed(maybeDstSubview))
      return failure();

    if (producerLoad) {
      auto maybeSrcSubview =
          createSlicedValue(irRewriter, producerLoad.getLoc(),
                            producerLoad.getSrc(), sliceOp.getMixedOffsets(),
                            sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
      if (failed(maybeSrcSubview))
        return failure();

      irRewriter.modifyOpInPlace(producerLoad, [&]() {
        producerLoad.getSrcMutable().set(maybeSrcSubview.value());
        producerLoad.getDstMutable().set(maybeDstSubview.value());
        producerLoad->setAttr(tiledOp, irRewriter.getUnitAttr());
      });
    }

    irRewriter.setInsertionPoint(sliceOp);
    auto newToTensorOp = irRewriter.create<bufferization::ToTensorOp>(
        sliceOp.getLoc(), maybeDstSubview.value(), true, true);
    irRewriter.modifyOpInPlace(newToTensorOp, [&]() {
      newToTensorOp->setAttr(tiledOp, irRewriter.getUnitAttr());
    });

    irRewriter.replaceOp(sliceOp, newToTensorOp->getResults());
    if (toTensorOp->use_empty())
      irRewriter.eraseOp(toTensorOp);
    return success();
  }
};

LogicalResult runProducerBubbleUp(func::FuncOp func) {
  GreedyRewriteConfig config;
  config.maxIterations = kMaxIterations;

  auto applyBubbleRound = [&](func::FuncOp currentFunc) -> LogicalResult {
    RewritePatternSet patterns(currentFunc.getContext());
    hivm::detail::populateHoistAffinePattern(patterns);
    populateProducerBubbleUpPatterns(patterns);
    hivm::detail::populateCSEPattern(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns, true);
    return applyPatternsGreedily(currentFunc, std::move(patterns), config);
  };

  if (failed(applyBubbleRound(func)))
    return failure();

  PassManager cleanupPm(func.getContext());
  cleanupPm.addPass(createCanonicalizerPass());
  cleanupPm.addPass(createCSEPass());
  if (failed(cleanupPm.run(func)))
    return failure();

  return applyBubbleRound(func);
}

LogicalResult verifyNoPendingBoundarySlices(func::FuncOp func) {
  auto walkResult = func.walk([](Operation *op) {
    if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op);
        insertSliceOp && isMarkedInsertSliceOp(insertSliceOp))
      return WalkResult::interrupt();
    if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op);
        sliceOp && sliceOp->hasAttrOfType<UnitAttr>(toBeBubbleUpSlice))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted() ? failure() : success();
}

// Drive the post-tiling cleanup pipeline to convergence:
// 1. Bubble the seeded local_store boundary slices up through the producer DAG.
// 2. Materialize extract_slice(to_tensor(...)) back onto the memref side.
// 3. Fold nested tiling subviews.
// 4. Run canonicalize + cse cleanup.
// 5. Succeed only if no marked boundary slices or cancel-out insert_slices
//    remain at the end.
LogicalResult runBubbleUpCleanupPipeline(func::FuncOp func) {
  if (failed(runProducerBubbleUp(func)))
    return failure();

  RewritePatternSet toTensorPatterns(func.getContext());
  toTensorPatterns.add<MaterializeToTensorSlicePattern>(func.getContext());
  if (failed(applyPatternsGreedily(func, std::move(toTensorPatterns))))
    return failure();

  RewritePatternSet patternsPost(func.getContext());
  patternsPost.add<hivm::detail::BubbleUpSubviewFromTiling>(func.getContext());
  if (failed(applyPatternsGreedily(func, std::move(patternsPost))))
    return failure();

  PassManager cleanupPm(func.getContext());
  cleanupPm.addPass(createCanonicalizerPass());
  cleanupPm.addPass(createCSEPass());
  if (failed(cleanupPm.run(func)))
    return failure();

  return verifyNoPendingBoundarySlices(func);
}

struct SIMTVFSubTilingPass
    : public impl::SIMTVFSubTilingBase<SIMTVFSubTilingPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SmallVector<func::FuncOp> simtFuncs;
    moduleOp.walk([&](func::FuncOp func) {
      if (util::isSIMTVF(func))
        simtFuncs.push_back(func);
    });

    for (func::FuncOp func : simtFuncs) {
      auto maybePlan = analyzeTilePlan(func, maxTileSize);
      if (!maybePlan.has_value())
        continue;

      auto maybeNewFunc = cloneIntoTileLoop(func, maybePlan->splitNum);
      if (failed(maybeNewFunc))
        continue;
      func::FuncOp newFunc = maybeNewFunc.value();

      // Keep the transform transactional: verification or rewrite failure drops
      // the clone and preserves the original SIMT VF for the rest of the
      // mixed-compilation pipeline.
      if (failed(seedLocalStoreBoundarySlices(newFunc, *maybePlan)) ||
          failed(runBubbleUpCleanupPipeline(newFunc)) ||
          failed(newFunc.verify())) {
        newFunc.erase();
        continue;
      }

      auto originalName = func.getSymName();
      func.erase();
      newFunc.setSymName(originalName);
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::hivm::createSIMTVFSubTilingPass(const SIMTVFSubTilingOptions &options) {
  return std::make_unique<SIMTVFSubTilingPass>(options);
}
