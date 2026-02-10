//===--------------------- TileAndBindSubBlock.cpp-------------------------===//
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
// This pass tiles and binds sub block for mix cv function.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Dialect/HIVM/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/Pattern.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Passes.h"
#include "bishengir/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <utility>
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_TILEANDBINDSUBBLOCK
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-bind-sub-block"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
static constexpr llvm::StringLiteral kLimitedSubBlockOpAttrName =
    "limit_sub_block_id0";
static constexpr llvm::StringLiteral tiledOp = "tiled_op";
static constexpr llvm::StringLiteral tileAndBindLeaf = "hivm.tile_and_bind_leaf";
} // namespace

namespace {

struct TileAndBindSubBlockPass
    : public impl::TileAndBindSubBlockBase<TileAndBindSubBlockPass> {
  using Base::Base;
  FailureOr<func::FuncOp> attemptBindSubBlock(func::FuncOp func);
  void runOnOperation() override;
};
} // namespace

/// Calculates the buffer size in bytes for a tensor value.
///
/// This function computes the total buffer size needed for a tensor by:
/// 1. Getting the total size in bits (shape × element type size)
/// 2. Dividing by the number of tiles (kSubBlockDim)
/// 3. Converting from bits to bytes (with ceiling division)
///
/// @param v The tensor value whose buffer size is to be calculated
/// @return The buffer size in bytes
static int64_t calculateBufferSize(Value v) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  assert(tensorType.hasStaticShape() &&
         "Tensor must have static shape for buffer size calculation");
  auto shape = tensorType.getShape();
  auto elementType = tensorType.getElementType();
  auto totalBits = mlir::utils::getStaticTotalSizeInBits(shape, elementType);
  if (!totalBits.has_value())
    llvm::report_fatal_error("Failed to calculate total size in bits");

  // Calculate buffer size: (totalBits / tileNum) converted to bytes
  constexpr int64_t tileCount = kSubBlockDim; // Currently 2 sub-blocks
  int64_t bitsPerTile = totalBits.value() / tileCount;
  int64_t bytesPerTile = llvm::divideCeilSigned(
      bitsPerTile, static_cast<int64_t>(utils::kBitsToByte));

  return bytesPerTile;
}

void setBufferSizeInLoopOp(RewriterBase &rewriter, Location loc,
                           Operation *loop,
                           DenseMap<Operation *, Operation *> &map) {
  auto forOp = dyn_cast<scf::ForOp>(loop);
  assert(forOp && "tile loop must be scf.for");
  Block *block = &forOp.getRegion().front();
  for (Operation &bodyOp : *block) {
    if (map.find(&bodyOp) == map.end())
      continue;
    for (OpResult result : bodyOp.getResults()) {
      auto maybeShapedType = dyn_cast<ShapedType>(result.getType());
      if (!maybeShapedType || maybeShapedType.hasStaticShape())
        continue;

      if (bodyOp.getDialect()->getNamespace() !=
          HIVMDialect::getDialectNamespace())
        continue;

      // calculate buffer size
      auto maybeInit =
          traceDefOp<tensor::EmptyOp>(map[&bodyOp]->getOperands().back());
      if (!maybeInit.has_value()) {
        llvm::report_fatal_error("Cannot trace inits for op");
      }
      auto calculationBufferSizeResult =
          calculateBufferSize(maybeInit.value()->getResult(0));
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfter(&bodyOp);
      auto mark = rewriter.create<annotation::MarkOp>(
          loc, bodyOp.getResult(result.getResultNumber()));
      rewriter.modifyOpInPlace(mark, [&]() {
        mark->setAttr(kBufferSizeInByteAttr,
                      rewriter.getI64IntegerAttr(calculationBufferSizeResult));
      });
    }
  }
}

static void modifyStoreToSliced(RewriterBase &rewriter, OpOperand *operand,
                                SmallVector<OpFoldResult, 4> mixedOffsets,
                                SmallVector<OpFoldResult, 4> mixedSize,
                                SmallVector<OpFoldResult, 4> mixedStrides,
                                SmallVector<int64_t, 4> newShape) {
  auto operandValue = operand->get();
  auto loc = operandValue.getLoc();

  auto newType = operandValue.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(newType)) {
    auto slicedValue = rewriter.create<tensor::ExtractSliceOp>(
        loc, operandValue, mixedOffsets,
        mixedSize, mixedStrides);
    operand->set(slicedValue.getResult());
    markCreatedExtractSliceOp(rewriter, slicedValue);
  } else if (auto memrefType = dyn_cast<MemRefType>(newType)) {
    auto slicedValue = rewriter.create<memref::SubViewOp>(
        loc, operandValue, mixedOffsets,
        mixedSize, mixedStrides);
    operand->set(slicedValue.getResult());
    markCreatedExtractSliceOp(rewriter, slicedValue);
  }
}

namespace {

/// try to tile store ops and bind sub block mapping
class TileAndSliceStore : public OpRewritePattern<hivm::StoreOp> {
public:
  explicit TileAndSliceStore(MLIRContext *context,
                             hivm::detail::DimensionAnalyzer &analyzer)
      : OpRewritePattern<hivm::StoreOp>(context, /*benefit=*/1),
        analyzer(analyzer) {}
  LogicalResult matchAndRewrite(hivm::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (storeOp->hasAttrOfType<UnitAttr>(tiledOp))
      return failure();
    int64_t tilingDim = analyzer.getTilingDim(storeOp.getSrc());
    auto maybeContainingLoop = findContainingSubblockLoop(storeOp);
    if (tilingDim == -1 || failed(maybeContainingLoop))
      return failure();

    auto containingLoop = maybeContainingLoop.value();
    auto srcType = dyn_cast<ShapedType>(storeOp.getSrc().getType());
    if (!srcType)
      return failure();

    // Handling special case
    if (ShapedType::isDynamicShape(srcType.getShape())) {
      if (failed(handleDynamicShape(storeOp, tilingDim, containingLoop, rewriter)))
        return failure();
    } else {
      auto *srcOpr = &storeOp.getSrcMutable();
      auto *dstOpr = &storeOp.getDstMutable();
      if (failed(modifyStoreOp(storeOp, tilingDim, srcOpr, dstOpr,
                               containingLoop, rewriter)))
        return failure();
    }

    // Maybe we need to maintain this map when doing bubble up.
    DenseMap<Operation *, Operation *> map;
    map[storeOp] = storeOp;
    setBufferSizeInLoopOp(rewriter, storeOp.getLoc(), containingLoop, map);

    return success();
  }

private:
  LogicalResult handleDynamicShape(hivm::StoreOp storeOp, int64_t tilingDim,
                                   scf::ForOp containingLoop,
                                   PatternRewriter &rewriter) const {
    auto *srcOpr = &storeOp.getSrcMutable();
    auto *dstOpr = &storeOp.getDstMutable();
    auto src = srcOpr->get();
    auto dst = dstOpr->get();
    SmallVector<OpFoldResult, 4> sizes;

        rewriter.setInsertionPointAfterValue(src);
    if (auto extractSliceOp = src.getDefiningOp<tensor::ExtractSliceOp>()) {
      extractSliceOp = cast<tensor::ExtractSliceOp>(rewriter.clone(*extractSliceOp.getOperation()));
      rewriter.modifyOpInPlace(storeOp, [&]() {
        srcOpr->set(extractSliceOp);
      });
      src = extractSliceOp.getSource();
      srcOpr = &extractSliceOp.getSourceMutable();
      sizes = extractSliceOp.getMixedSizes();
    } else if (auto subViewOp = src.getDefiningOp<memref::SubViewOp>()) {
      subViewOp = cast<memref::SubViewOp>(rewriter.clone(*subViewOp.getOperation()));
      rewriter.modifyOpInPlace(storeOp, [&]() {
        srcOpr->set(subViewOp);
      });
      src = subViewOp.getSource();
      srcOpr = &subViewOp.getSourceMutable();
      sizes = extractSliceOp.getMixedSizes();
    } else {
      return failure();
    }
    rewriter.setInsertionPointAfterValue(dst);
    if (auto extractSliceOp = dst.getDefiningOp<tensor::ExtractSliceOp>()) {
      extractSliceOp = cast<tensor::ExtractSliceOp>(rewriter.clone(*extractSliceOp.getOperation()));
      rewriter.modifyOpInPlace(storeOp, [&]() {
        dstOpr->set(extractSliceOp);
      });
      dst = extractSliceOp.getSource();
      dstOpr = &extractSliceOp.getSourceMutable();
    } else if (auto subViewOp = dst.getDefiningOp<memref::SubViewOp>()) {
      subViewOp = cast<memref::SubViewOp>(rewriter.clone(*subViewOp.getOperation()));
      rewriter.modifyOpInPlace(storeOp, [&]() {
        dstOpr->set(subViewOp);
      });
      dst = subViewOp.getSource();
      dstOpr = &subViewOp.getSourceMutable();
    } else {
      return failure();
    }

    rewriter.setInsertionPoint(storeOp);
    auto srcShape = llvm::to_vector(cast<ShapedType>(src.getType()).getShape());
    auto dstShape = llvm::to_vector(cast<ShapedType>(dst.getType()).getShape());
    auto srcOp =
        storeOp.getSrc().getDefiningOp<OffsetSizeAndStrideOpInterface>();
    auto dstOp =
        storeOp.getDst().getDefiningOp<OffsetSizeAndStrideOpInterface>();
    if (srcShape != dstShape || ShapedType::isDynamicShape(srcShape) ||
        !srcOp.hasZeroOffset() || !srcOp.hasUnitStride() ||
        !dstOp.hasZeroOffset() || !dstOp.hasUnitStride())
      return failure();
    if (failed(modifyStoreOp(storeOp, tilingDim, srcOpr, dstOpr, containingLoop,
                             rewriter)))
      return failure();
    auto loc = storeOp.getLoc();
    rewriter.setInsertionPoint(storeOp);
    auto maybeSingleTileSize =
        getSingleTileSize(rewriter, loc, src, tilingDim, containingLoop);
    rewriter.setInsertionPointToStart(containingLoop.getBody());
    auto offsetAtTileDim = calculateOffsetAtTilingDim(
        rewriter, loc, containingLoop, maybeSingleTileSize.value());

    auto offsetValue =
        getValueOrCreateConstantIndexOp(rewriter, loc, offsetAtTileDim);
    auto sizeVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, sizes[tilingDim]);
    auto tilingSize = getValueOrCreateConstantIndexOp(
        rewriter, loc, maybeSingleTileSize.value());
    rewriter.setInsertionPointAfterValue(sizeVal);

    offsetValue = rewriter.create<arith::MinSIOp>(offsetValue.getLoc(),
                                                  offsetValue, sizeVal);
    sizeVal =
        rewriter.create<arith::SubIOp>(sizeVal.getLoc(), sizeVal, offsetValue);
    sizeVal =
        rewriter.create<arith::MinSIOp>(sizeVal.getLoc(), sizeVal, tilingSize);
    sizes[tilingDim] = sizeVal;

    src = storeOp.getSrc();
    dst = storeOp.getDst();

    if (auto extractSliceOp = src.getDefiningOp<tensor::ExtractSliceOp>()) {
      rewriter.setInsertionPoint(extractSliceOp);
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          extractSliceOp, extractSliceOp.getSource(),
          extractSliceOp.getMixedOffsets(), sizes,
          extractSliceOp.getMixedStrides());
    } else if (auto subViewOp = src.getDefiningOp<memref::SubViewOp>()) {
      rewriter.setInsertionPoint(subViewOp);
      rewriter.replaceOpWithNewOp<memref::SubViewOp>(
          subViewOp, subViewOp.getSource(), subViewOp.getMixedOffsets(), sizes,
          subViewOp.getMixedStrides());
    }
    if (auto extractSliceOp = dst.getDefiningOp<tensor::ExtractSliceOp>()) {
      rewriter.setInsertionPoint(extractSliceOp);
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          extractSliceOp, extractSliceOp.getSource(),
          extractSliceOp.getMixedOffsets(), sizes,
          extractSliceOp.getMixedStrides());
    } else if (auto subViewOp = dst.getDefiningOp<memref::SubViewOp>()) {
      rewriter.setInsertionPoint(subViewOp);
      rewriter.replaceOpWithNewOp<memref::SubViewOp>(
          subViewOp, subViewOp.getSource(), subViewOp.getMixedOffsets(), sizes,
          subViewOp.getMixedStrides());
    }
    return success();
  }

  LogicalResult modifyStoreOp(hivm::StoreOp storeOp, int64_t tilingDim,
                              OpOperand *srcOpr, OpOperand *dstOpr,
                              scf::ForOp containingLoop,
                              PatternRewriter &rewriter) const {
    auto loc = storeOp.getLoc();
    auto src = srcOpr->get();
    auto srcType = cast<ShapedType>(src.getType());
    auto maybeSingleTileSize =
        getSingleTileSize(rewriter, loc, src, tilingDim, containingLoop);
    if (failed(maybeSingleTileSize))
      return failure();
    rewriter.setInsertionPointToStart(containingLoop.getBody());
    auto offsetAtTileDim = calculateOffsetAtTilingDim(
        rewriter, loc, containingLoop, maybeSingleTileSize.value());

    rewriter.setInsertionPoint(storeOp);

    SmallVector<OpFoldResult, 4> mixedStrides, mixedOffsets, mixedSize;
    SmallVector<int64_t, 4> newShape;
    if (failed(findCorrespondingSizesOffsetsStrides(
            rewriter, srcType, tilingDim, offsetAtTileDim,
            maybeSingleTileSize.value(), mixedStrides, mixedOffsets, mixedSize,
            newShape)))
      return failure();

    if (containingLoop.getRegion().isAncestor(srcOpr->get().getParentRegion())) {
      rewriter.setInsertionPointAfterValue(srcOpr->get());
    } else {
      rewriter.setInsertionPointAfterValue(offsetAtTileDim.get<Value>());
    }
    modifyStoreToSliced(rewriter, srcOpr, mixedOffsets, mixedSize, mixedStrides,
                        newShape);
    if (containingLoop.getRegion().isAncestor(dstOpr->get().getParentRegion())) {
      rewriter.setInsertionPointAfterValue(dstOpr->get());
    } else {
      rewriter.setInsertionPointAfterValue(offsetAtTileDim.get<Value>());
    }
    modifyStoreToSliced(rewriter, dstOpr, mixedOffsets, mixedSize, mixedStrides,
                        newShape);
    rewriter.modifyOpInPlace(storeOp, [&]() {
      if (storeOp->getNumResults() > 0)
        storeOp->getResult(0).setType(storeOp.getDst().getType());
      storeOp->setAttr(tiledOp, rewriter.getUnitAttr());
    });
    return success();
  }

  hivm::detail::DimensionAnalyzer &analyzer;
};

template <typename OpTy>
class TileAndSliceLeaf : public OpRewritePattern<OpTy> {
public:
  explicit TileAndSliceLeaf(MLIRContext *context,
                            hivm::detail::DimensionAnalyzer &analyzer)
      : OpRewritePattern<OpTy>(context),
        analyzer(analyzer) {}
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    LogicalResult result = failure();
    auto maybeContainingLoop = findContainingSubblockLoop(op);
    if (failed(maybeContainingLoop))
      return failure();
    auto containingLoop = maybeContainingLoop.value();
    for (auto res : op->getResults()) {
      int64_t tilingDim = analyzer.getTilingDim(res);
      if (tilingDim == -1 || !res.use_empty())
        continue;

      auto loc = res.getLoc();
      auto maybeSingleTileSize = getSingleTileSize(
          rewriter, loc, res, tilingDim, containingLoop);
      if (failed(maybeSingleTileSize))
        continue;
      rewriter.setInsertionPointToStart(containingLoop.getBody());
      auto offsetAtTileDim = calculateOffsetAtTilingDim(
          rewriter, loc, containingLoop, maybeSingleTileSize.value());

      rewriter.setInsertionPointAfter(op);

      SmallVector<OpFoldResult, 4> mixedStrides, mixedOffsets, mixedSize;
      SmallVector<int64_t, 4> newShape;
      auto rankType = cast<ShapedType>(res.getType());
      assert(!ShapedType::isDynamicShape(rankType.getShape()));
      if (failed(findCorrespondingSizesOffsetsStrides(
              rewriter, rankType, tilingDim, offsetAtTileDim,
              maybeSingleTileSize.value(), mixedStrides, mixedOffsets, mixedSize,
              newShape)))
        continue;

      auto newType = RankedTensorType::get(newShape, rankType.getElementType());
      auto slicedValue = rewriter.create<tensor::ExtractSliceOp>(
          loc, newType, res, mixedOffsets, mixedSize,
          mixedStrides);
      markCreatedExtractSliceOp(rewriter, slicedValue);

      auto mark = rewriter.create<annotation::MarkOp>(loc, slicedValue);
      mark->setAttr(tileAndBindLeaf, rewriter.getUnitAttr());
      result = success();
    }
    return result;
  }

private:
  hivm::detail::DimensionAnalyzer &analyzer;
};

/// add if (sublock_id == 0) guard for each store op.
/// e.g.
/// case 1: store op without results
///   store op
/// is changed to
///   if (subblock_id == 0)
///     store op
/// case 2: store op with results
///   %res = store op
/// is changed to
///   if (subblock_id == 0)
///     %res = store op
///     yield %res
///   else
///     yield store's outs
struct LimitUniqueSubBlockIdToStore : public OpRewritePattern<hivm::StoreOp> {
public:
  using OpRewritePattern<hivm::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::StoreOp op,
                                PatternRewriter &rewriter) const override {
    if (auto ifOpOld = dyn_cast_if_present<scf::IfOp>(op->getParentOp())) {
      if (ifOpOld->hasAttrOfType<UnitAttr>(kLimitedSubBlockOpAttrName))
        return failure();
    }
    auto loc = op.getLoc();
    auto subBlockIdxOp =
        rewriter.create<hivm::GetSubBlockIdxOp>(loc, rewriter.getI64Type());
    auto subBlockIndex =
        rewriter
            .create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                        subBlockIdxOp.getResult())
            .getResult();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto cond = rewriter.create<arith::CmpIOp>(loc, rewriter.getI1Type(),
                                               arith::CmpIPredicate::eq,
                                               subBlockIndex, zero);

    if (op.getResults().empty()) {
      // case 1: store op without results
      auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange(), cond, false);
      auto thenBodyBuilder = ifOp.getThenBodyBuilder(rewriter.getListener());
      thenBodyBuilder.clone(*op.getOperation());
      rewriter.replaceOp(op, ifOp);
      rewriter.modifyOpInPlace(ifOp, [&]() {
        ifOp->setAttr(kLimitedSubBlockOpAttrName,
                      UnitAttr::get(ifOp->getContext()));
      });
      return success();
    }

    // case 2: store op with results
    Type dstType = op.getDst().getType();
    auto ifOp = rewriter.create<scf::IfOp>(loc, dstType, cond, true);
    // then block
    {
      PatternRewriter::InsertionGuard insertionGuard(rewriter);
      auto thenBodyBuilder = ifOp.getThenBodyBuilder(rewriter.getListener());
      auto cloneStoreOp = thenBodyBuilder.clone(*op.getOperation());
      Value thenYield = cloneStoreOp->getResults()[0];
      ifOp.getThenBodyBuilder().create<scf::YieldOp>(loc, thenYield);
    }

    // else block
    {
      rewriter.setInsertionPointToEnd(&ifOp.getElseRegion().front());
      rewriter.create<scf::YieldOp>(loc, op.getDst());
    }
    rewriter.modifyOpInPlace(ifOp, [&]() {
      ifOp->setAttr(kLimitedSubBlockOpAttrName,
                    UnitAttr::get(ifOp->getContext()));
    });
    rewriter.replaceOp(op, ifOp);
    return success();
  }
};

} // namespace

static LogicalResult limitUniqueSubBlockToStore(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<LimitUniqueSubBlockIdToStore>(funcOp.getContext());
  GreedyRewriteConfig config;
  config.maxIterations = kMaxIterations;
  return applyPatternsGreedily(funcOp, std::move(patterns));
}

static scf::ForOp createSubBlockLoop(Location loc, OpBuilder &builder,
                                     int64_t lowerBound, int64_t step,
                                     int64_t upperBound) {
  auto loopLowerBound =
      builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(lowerBound));
  auto loopStep =
      builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(step));
  auto loopUpperBound =
      builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(upperBound));
  auto subBlockLoop =
      builder.create<scf::ForOp>(loc, loopLowerBound, loopUpperBound, loopStep);
  subBlockLoop->setAttr(kMapForToForallAttrName,
                        UnitAttr::get(subBlockLoop->getContext()));

  SmallVector<Attribute> mappingNames;
  mappingNames.push_back(HIVMSubBlockMappingAttr::get(
      subBlockLoop->getContext(), hivm::MappingId::DimX));
  subBlockLoop->setAttr(
      kMappingAttrName,
      ArrayAttr::get(subBlockLoop->getContext(), mappingNames));
  return subBlockLoop;
}

static void failAndRevert(func::FuncOp func) {
  LLVM_DEBUG(DBGS() << "tile and bind subblock fail for "
                    << func.getSymNameAttr().str() << "\n\n");
  LLVM_DEBUG(func->dump());
  func->erase();
}

static void populateBindSubBlockBubbleUpPassManager(PassManager &pm) {
  pm.addPass(createHIVMBubbleUpExtractSlicePass());
  CanonicalizerOptions options;
  SmallVector<std::string> disabledPatterns(
      {"ReinterpretCastConstantArgumentFolder"});
  options.disabledPatterns = disabledPatterns;
  pm.addPass(bishengir::createExtendedCanonicalizerPass(options));
  pm.addPass(createCSEPass());
}

static LogicalResult tileAndSliceStore(func::FuncOp func) {
  hivm::detail::DimensionAnalyzer analyzer(func);
  if (failed(analyzer.initialize()))
    return failure();

  analyzer.computeTilingDim();

  // Check there is no dynamic shape store, if there is, we cannot tile it to 2
  // for now.
  std::vector<hivm::StoreOp> allStoreOps;
  func->walk([&allStoreOps](hivm::StoreOp storeOp) {
    allStoreOps.push_back(storeOp);
  });
  if (llvm::any_of(allStoreOps, [&](hivm::StoreOp storeOp) {
        auto srcShapedType = dyn_cast<ShapedType>(storeOp.getSrcOperandType());
        auto dstShapedType = dyn_cast<ShapedType>(storeOp.getDstOperandType());
        if (!srcShapedType || !dstShapedType)
          return true;
        if (ShapedType::isDynamicShape(srcShapedType.getShape()) ||
            ShapedType::isDynamicShape(dstShapedType.getShape())) {
          auto src = storeOp.getSrc();
          auto dst = storeOp.getDst();
          if (auto extractSliceOp = src.getDefiningOp<tensor::ExtractSliceOp>()) {
            src = extractSliceOp.getSource();
          } else if (auto subViewOp = src.getDefiningOp<memref::SubViewOp>()) {
            src = subViewOp.getSource();
          }
          if (auto extractSliceOp = dst.getDefiningOp<tensor::ExtractSliceOp>()) {
            dst = extractSliceOp.getSource();
          } else if (auto subViewOp = dst.getDefiningOp<memref::SubViewOp>()) {
            dst = subViewOp.getSource();
          }
          srcShapedType = cast<ShapedType>(src.getType());
          dstShapedType = cast<ShapedType>(dst.getType());
          return ShapedType::isDynamicShape(srcShapedType.getShape()) ||
                 ShapedType::isDynamicShape(dstShapedType.getShape()) ||
                 srcShapedType.getShape() != dstShapedType.getShape();
        }
        return false;
      })) {
    return failure();
  }

  RewritePatternSet patterns(func->getContext());
  patterns.add<TileAndSliceStore,
               TileAndSliceLeaf<scf::ForOp>,
               TileAndSliceLeaf<scf::WhileOp>,
               TileAndSliceLeaf<scf::IfOp>>(func->getContext(), analyzer);
  GreedyRewriteConfig config;
  config.maxIterations = kMaxIterations;
  if (failed(applyPatternsGreedily(func, std::move(patterns), config))) {
    return failure();
  }

  bool isFailed = true;
  auto walkResult = func->walk([&isFailed](hivm::StoreOp op) {
    if (!op->hasAttr(tiledOp)) {
      if (op->hasAttr(hivm::AtomicKindAttr::name)) {
        isFailed = true;
        return WalkResult::interrupt();
      }
    } else {
      isFailed = false;
    }
    return WalkResult::advance();
  });
  if (isFailed)
    return failure();
  return success();
}

/// Attempts to tile and bind sub-blocks within a function
///
/// This function performs a series of transformations on vector functions:
/// 1. Creates a BindSubBlock Loop that includes whole function body
/// i.e.
/// func {
///   for {
///     func_body
///   } {sub_block_loop}
/// }
/// 2. Insert a extract slice before all storeOps
/// And then we rely on run bubbleUpExtractSlice to tile all ops
///
/// @param func The function to transform (should be a clone if rollback is
/// needed)
/// @return Success if transformation completed, failure otherwise
FailureOr<func::FuncOp>
TileAndBindSubBlockPass::attemptBindSubBlock(func::FuncOp func) {
  // This only apply for aiv func. Should be check before calling.
  OpBuilder builder(func->getContext());
  builder.setInsertionPoint(func);
  // We cloned newFunc for processing.
  func::FuncOp newFunc = cast<func::FuncOp>(builder.cloneWithoutRegions(func));
  newFunc.addEntryBlock();
  builder.setInsertionPointToStart(&newFunc.getBody().getBlocks().front());

  auto subBlockLoop =
      createSubBlockLoop(func->getLoc(), builder, 0, 1, kSubBlockDim);

  IRMapping map;
  for (size_t i = 0; i < func.getNumArguments(); i++) {
    map.map(func.getArgument(i), newFunc.getArgument(i));
  }

  builder.setInsertionPointToStart(subBlockLoop.getBody(0));
  // We are trying to wrap subblock loop to the whole function body.
  // so we clone the whole function body inside the loop.
  func.getBody().cloneInto(&subBlockLoop.getBodyRegion(0), map);

  // bb0 is the loop body when the loop is created (empty with a terminator)
  // bb1 is the cloned function body
  auto &bb0 = subBlockLoop.getBodyRegion(0).getBlocks().front();
  auto *bb1 = bb0.getNextNode();
  if (!bb1)
    llvm::report_fatal_error("Failed to find function body");

  Operation *terminator = bb0.getTerminator();
  // We need to merge bb0 and bb1 because a loop body can only have 1 blocks
  if (bb1->mightHaveTerminator()) {
    builder.setInsertionPointToEnd(&newFunc.getBody().getBlocks().front());
    builder.clone(*bb1->getTerminator(), map);
    bb1->getOperations().pop_back();
  }
  bb0.getOperations().splice(terminator->getIterator(), bb1->getOperations());
  // We need to handle the terminators. clone function body's (bb1) terminator
  // outside of subblock loop body and use as cloned newFunc's terminator.
  bb1->erase();

  if (failed(tileAndSliceStore(newFunc))) {
    failAndRevert(newFunc);
    return failure();
  }

  PassManager pm(newFunc->getContext());
  populateBindSubBlockBubbleUpPassManager(pm);

  LogicalResult bubbleUpResult = pm.run(newFunc);
  if (bubbleUpResult.failed() || newFunc.verify().failed() ||
      newFunc.verifyBody().failed() || newFunc.verifyRegions().failed()) {
    failAndRevert(newFunc);
    return failure();
  }

  SmallVector<Operation *> toBeRemoved;
  func->walk([&](annotation::MarkOp op) {
    if (op->hasAttr(tileAndBindLeaf))
        toBeRemoved.push_back(op);
  });

  for (auto *op : toBeRemoved)
    op->erase();

  RewritePatternSet patternsPost(&getContext());
  patternsPost.add<mlir::hivm::detail::BubbleUpSubviewFromTiling>(
      &getContext());
  if (failed(applyPatternsGreedily(newFunc, std::move(patternsPost)))) {
    failAndRevert(newFunc);
    return failure();
  }

  return newFunc;
}

/// Walks through all functions in the module and attempts to tile and bind
/// sub-blocks for vector functions.
///
/// Functions are cloned before transformation to allow rollback on failure.
/// If attempt to bind some block fail it will rollback to 1:1 and limit to
/// unique block to store.
void TileAndBindSubBlockPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
#ifndef NDEBUG
  uint64_t tiledFunctionCount = 0;
#endif

  PassManager pm(&getContext());
  CanonicalizerOptions options;
  SmallVector<std::string> disabledPatterns(
      {"ReinterpretCastConstantArgumentFolder"});
  options.disabledPatterns = disabledPatterns;
  pm.addPass(bishengir::createExtendedCanonicalizerPass(options));

  // Collect functions to process (can't modify while iterating)
  SmallVector<func::FuncOp> functionsToProcess;
  moduleOp->walk(
      [&](func::FuncOp funcOp) { functionsToProcess.push_back(funcOp); });

  // Process each function
  for (func::FuncOp originalFunc : functionsToProcess) {
    // Only process vector functions
    auto symNameStr = originalFunc.getSymNameAttr().str();
    auto funcCoreType = queryFuncCoreType(originalFunc);
    if (!funcCoreType.has_value() ||
        funcCoreType.value() != TFuncCoreType::AIV ||
        !originalFunc->hasAttrOfType<UnitAttr>(hivm::TPartOfMixAttr::name))
      continue;
    // Clone the function for safe transformation
    OpBuilder builder(originalFunc);
    // Attempt transformation on the clone
    FailureOr<func::FuncOp> res = attemptBindSubBlock(originalFunc);
    if (failed(res)) {
      if (failed(limitUniqueSubBlockToStore(originalFunc))) {
        LLVM_DEBUG(DBGS() << "Failed to limit unique subblock: " << symNameStr
                          << "\n");
        signalPassFailure();
      }
      LLVM_DEBUG(DBGS() << "Failed to transform function: " << symNameStr
                        << ", keeping original\n");
      return;
    }
    auto processedFunc = res.value();
    processedFunc.setName(originalFunc.getName().str() + "_processing");
    if (succeeded(res)) {
      // Success: Remove original and rename clone
      originalFunc.erase();
      processedFunc.setName(symNameStr);
#ifndef NDEBUG
      tiledFunctionCount++;
      LLVM_DEBUG(DBGS() << "Successfully transformed function #"
                        << tiledFunctionCount << ": " << symNameStr << "\n");
#endif
    }
  }

#ifndef NDEBUG
  LLVM_DEBUG(DBGS() << "TileAndBindSubBlock pass completed. "
                    << "Successfully transformed " << tiledFunctionCount
                    << " functions.\n");
#endif
}

std::unique_ptr<Pass> mlir::hivm::createTileAndBindSubBlockPass() {
  return std::make_unique<TileAndBindSubBlockPass>();
}
