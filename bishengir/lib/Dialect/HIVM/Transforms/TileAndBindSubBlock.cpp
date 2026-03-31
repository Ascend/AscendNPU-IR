//===--------------------- TileAndBindSubBlock.cpp-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/HoistAffine.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/Pattern.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
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
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <string>
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
static constexpr llvm::StringLiteral tileAndBindLeaf =
    "hivm.tile_and_bind_leaf";
static constexpr llvm::StringLiteral tileAndSliceFailure = "slice_failure";
} // namespace

namespace {

struct TileAndBindSubBlockPass
    : public impl::TileAndBindSubBlockBase<TileAndBindSubBlockPass> {
public:
  using Base::Base;
  FailureOr<func::FuncOp> attemptBindSubBlock(func::FuncOp func);
  void runOnOperation() override;

private:
  DenseMap<int32_t, int64_t> tightlyCoupledBufferToTilingDim;
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

template <typename OpType>
static void modifyOpToSliced(RewriterBase &rewriter, OpType Op,
                             SmallVector<OpFoldResult, 4> mixedOffsets,
                             SmallVector<OpFoldResult, 4> mixedSize,
                             SmallVector<OpFoldResult, 4> mixedStrides,
                             SmallVector<int64_t, 4> newShape) {
  auto rankType = cast<RankedTensorType>(Op.getSrc().getType());
  auto loc = Op->getLoc();

  auto newType =
      mlir::RankedTensorType::get(newShape, rankType.getElementType());
  auto slicedStore = rewriter.create<tensor::ExtractSliceOp>(
      loc, newType, Op->getOperand(0), mixedOffsets, mixedSize, mixedStrides);
  markCreatedExtractSliceOp(rewriter, slicedStore);

  auto initsType = Op.getDpsInitOperand(0)->get().getType();
  if (isa<mlir::RankedTensorType>(initsType)) {
    auto slicedInit = rewriter.create<tensor::ExtractSliceOp>(
        loc, newType, Op.getDpsInitOperand(0)->get(), mixedOffsets, mixedSize,
        mixedStrides);
    rewriter.modifyOpInPlace(Op,
                             [&]() { Op.setDpsInitOperand(0, slicedInit); });
    markCreatedExtractSliceOp(rewriter, slicedInit);
  } else if (isa<mlir::MemRefType>(initsType)) {
    auto subviewedInits = rewriter.create<memref::SubViewOp>(
        loc, Op.getDpsInitOperand(0)->get(), mixedOffsets, mixedSize,
        mixedStrides);
    markCreatedExtractSliceOp(rewriter, subviewedInits);

    rewriter.modifyOpInPlace(
        Op, [&]() { Op.setDpsInitOperand(0, subviewedInits); });
  }
  rewriter.modifyOpInPlace(Op, [&]() {
    Op->setOperand(0, slicedStore);
    if (Op->getNumResults() > 0)
      Op->getResult(0).setType(newType);
    Op->setAttr(tiledOp, UnitAttr::get(Op->getContext()));
  });
}

namespace {

/// try to tile storeOp and copyOp and bind sub block mapping
template <typename OpType>
class TileAndSliceStoreCopyOp : public OpRewritePattern<OpType> {
public:
  hivm::detail::DimensionAnalyzer &analyzer;

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
          loc, operandValue, mixedOffsets, mixedSize, mixedStrides);
      operand->set(slicedValue);
      markCreatedExtractSliceOp(rewriter, slicedValue);
    } else if (auto memrefType = dyn_cast<MemRefType>(newType)) {
      auto slicedValue = rewriter.create<memref::SubViewOp>(
          loc, operandValue, mixedOffsets, mixedSize, mixedStrides);
      operand->set(slicedValue);
      markCreatedExtractSliceOp(rewriter, slicedValue);
    }
  }

  explicit TileAndSliceStoreCopyOp(MLIRContext *context,
                                   hivm::detail::DimensionAnalyzer &analyzer)
      : OpRewritePattern<OpType>(context, /*benefit=*/1), analyzer(analyzer) {}
  LogicalResult matchAndRewrite(OpType Op,
                                PatternRewriter &rewriter) const override {
    if (Op->template hasAttrOfType<UnitAttr>(tiledOp) ||
        Op->template hasAttrOfType<UnitAttr>(tileAndSliceFailure))
      return failure();

    int64_t tilingDim = analyzer.getTilingDim(Op.getSrc());
    if (std::is_same_v<hivm::CopyOp, OpType>){
      if (!Op.getResults().empty()){  // If copy Op with results
        if (!llvm::any_of(Op->getUsers(), [](Operation *user) {
              return isa<annotation::MarkOp>(user);
            })) {
          return failure(); // If the user of CopyOp is not MarkOp, it cannot be
                            // a tiling start point.
        }
      }
      LLVM_DEBUG(DBGS() << "The copy op tiling dim is: " << tilingDim << "\n");
    } else {
      LLVM_DEBUG(DBGS() << "The store op tiling dim is: "<<tilingDim<<"\n");
    }
    auto maybeContainingLoop = findContainingSubblockLoop(Op);
    if (tilingDim == -1 || failed(maybeContainingLoop)) {
      Op->setAttr(tileAndSliceFailure, rewriter.getUnitAttr());
      return failure();
    }

    auto containingLoop = maybeContainingLoop.value();

    if (std::is_same_v<hivm::StoreOp, OpType>) {
      auto storeOp = cast<hivm::StoreOp>(Op);
      auto srcType = dyn_cast<ShapedType>(storeOp.getSrc().getType());
      if (!srcType)
        return failure();
      // Handling special case
      if (ShapedType::isDynamicShape(srcType.getShape())) {
        return handleDynamicShape(storeOp, tilingDim, containingLoop, rewriter);
      }
    }

    auto loc = Op.getLoc();
    auto maybeSingleTileSize = getSingleTileSize(rewriter, loc, Op.getSrc(),
                                                 tilingDim, containingLoop);
    if (failed(maybeSingleTileSize)) {
      Op->setAttr(tileAndSliceFailure, rewriter.getUnitAttr());
      return failure();
    }

    rewriter.setInsertionPointToStart(containingLoop.getBody());
    auto offsetAtTileDim = calculateOffsetAtTilingDim(
        rewriter, loc, containingLoop, maybeSingleTileSize.value());

    rewriter.setInsertionPoint(Op);

    SmallVector<OpFoldResult, 4> mixedStrides, mixedOffsets, mixedSize;
    SmallVector<int64_t, 4> newShape;
    auto rankType = cast<RankedTensorType>(Op.getSrc().getType());
    if (ShapedType::isDynamicShape(rankType.getShape())) {
      return failure();
    }
    if (failed(findCorrespondingSizesOffsetsStrides(
            rewriter, rankType, tilingDim, offsetAtTileDim,
            maybeSingleTileSize.value(), mixedStrides, mixedOffsets, mixedSize,
            newShape)))
      return failure();

    modifyOpToSliced(rewriter, Op, mixedOffsets, mixedSize, mixedStrides,
                     newShape);

    // Maybe we need to maintain this map when doing bubble up.
    DenseMap<Operation *, Operation *> map;
    map[Op] = Op;
    setBufferSizeInLoopOp(rewriter, loc, containingLoop, map);

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
    auto modifyFailure = [&]() -> LogicalResult {
      storeOp->setAttr(tileAndSliceFailure, rewriter.getUnitAttr());
      return failure();
    };

    // Clone new src and dst
    rewriter.setInsertionPointAfterValue(src);
    if (auto extractSliceOp = src.getDefiningOp<tensor::ExtractSliceOp>()) {
      extractSliceOp = cast<tensor::ExtractSliceOp>(
          rewriter.clone(*extractSliceOp.getOperation()));
      rewriter.modifyOpInPlace(storeOp, [&]() { srcOpr->set(extractSliceOp); });
      src = extractSliceOp.getSource();
      srcOpr = &extractSliceOp.getSourceMutable();
      sizes = extractSliceOp.getMixedSizes();
    } else if (auto subViewOp = src.getDefiningOp<memref::SubViewOp>()) {
      subViewOp =
          cast<memref::SubViewOp>(rewriter.clone(*subViewOp.getOperation()));
      rewriter.modifyOpInPlace(storeOp, [&]() { srcOpr->set(subViewOp); });
      src = subViewOp.getSource();
      srcOpr = &subViewOp.getSourceMutable();
      sizes = extractSliceOp.getMixedSizes();
    } else {
      return modifyFailure();
    }
    rewriter.setInsertionPointAfterValue(dst);
    if (auto extractSliceOp = dst.getDefiningOp<tensor::ExtractSliceOp>()) {
      extractSliceOp = cast<tensor::ExtractSliceOp>(
          rewriter.clone(*extractSliceOp.getOperation()));
      rewriter.modifyOpInPlace(storeOp, [&]() { dstOpr->set(extractSliceOp); });
      dst = extractSliceOp.getSource();
      dstOpr = &extractSliceOp.getSourceMutable();
    } else if (auto subViewOp = dst.getDefiningOp<memref::SubViewOp>()) {
      subViewOp =
          cast<memref::SubViewOp>(rewriter.clone(*subViewOp.getOperation()));
      rewriter.modifyOpInPlace(storeOp, [&]() { dstOpr->set(subViewOp); });
      dst = subViewOp.getSource();
      dstOpr = &subViewOp.getSourceMutable();
    } else {
      return modifyFailure();
    }

    rewriter.setInsertionPoint(storeOp);
    auto srcShape = llvm::to_vector(cast<ShapedType>(src.getType()).getShape());
    auto dstShape = llvm::to_vector(cast<ShapedType>(dst.getType()).getShape());
    if (srcShape != dstShape || ShapedType::isDynamicShape(srcShape))
      return modifyFailure();
    if (failed(modifyStoreOp(storeOp, tilingDim, srcOpr, dstOpr, containingLoop,
                             rewriter)))
      return modifyFailure();
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

    // Replace with new src and dst
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
    auto srcType = cast<RankedTensorType>(src.getType());
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

    if (containingLoop.getRegion().isAncestor(
            srcOpr->get().getParentRegion())) {
      rewriter.setInsertionPointAfterValue(srcOpr->get());
    } else {
      rewriter.setInsertionPointAfterValue(offsetAtTileDim.get<Value>());
    }
    modifyStoreToSliced(rewriter, srcOpr, mixedOffsets, mixedSize, mixedStrides,
                        newShape);
    if (containingLoop.getRegion().isAncestor(
            dstOpr->get().getParentRegion())) {
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

  LogicalResult findCorrespondingSizesOffsetsStrides(
      RewriterBase &rewriter, RankedTensorType rankType, int64_t tilingDim,
      OpFoldResult offsetAtTileDim, OpFoldResult tileSize,
      SmallVector<OpFoldResult, 4> &mixedStrides,
      SmallVector<OpFoldResult, 4> &mixedOffsets,
      SmallVector<OpFoldResult, 4> &mixedSize,
      SmallVector<int64_t, 4> &newShape) const {
    for (int i = 0; i < rankType.getRank(); i++) {
      mixedStrides.push_back(rewriter.getIndexAttr(1));
      if (i != tilingDim) {
        mixedOffsets.push_back(rewriter.getIndexAttr(0));
        mixedSize.push_back(getAsIndexOpFoldResult(rewriter.getContext(),
                                                   rankType.getDimSize(i)));
        newShape.push_back(rankType.getDimSize(i));
      } else {
        mixedOffsets.push_back(offsetAtTileDim);
        mixedSize.push_back(tileSize);
        if (!getConstantIntValue(tileSize)) {
          return failure();
        }
        newShape.push_back(getConstantIntValue(tileSize).value());
      }
    }
    return success();
  }
};

/// Try to tile the leaf nodes that have only annotation::MarkOp as users.
/// Take scf.for Op as example. We insert tensor.extract_sliceOp before
/// annotation::MarkOp, it will change to: %42:1 = scf.for {
///   scf.yield %42#0
///   }
///   %43 = tensor.extract_sliceOp %42#0
///   annotation.mark %43
template <typename OpTy>
class TileAndSliceLeaf : public OpRewritePattern<OpTy> {
public:
  hivm::detail::DimensionAnalyzer &analyzer;
  explicit TileAndSliceLeaf(MLIRContext *context,
                            hivm::detail::DimensionAnalyzer &analyzer)
      : OpRewritePattern<OpTy>(context), analyzer(analyzer) {}
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    LogicalResult result = failure();
    auto maybeContainingLoop = findContainingSubblockLoop(op);
    if (failed(maybeContainingLoop))
      return failure();
    for (auto res : op->getResults()) {
      int64_t tilingDim = 0;
      if (!res.use_empty()){
        continue;
      }
      auto containingLoop = maybeContainingLoop.value();
      auto loc = res.getLoc();
      auto maybeSingleTileSize =
          getSingleTileSize(rewriter, loc, res, tilingDim, containingLoop);
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
              maybeSingleTileSize.value(), mixedStrides, mixedOffsets,
              mixedSize, newShape)))
        continue;

      auto newType = RankedTensorType::get(newShape, rankType.getElementType());
      auto slicedValue = rewriter.create<tensor::ExtractSliceOp>(
          loc, newType, res, mixedOffsets, mixedSize, mixedStrides);

      markCreatedExtractSliceOp(rewriter, slicedValue);

      auto mark = rewriter.create<annotation::MarkOp>(loc, slicedValue);
 	    mark->setAttr(tileAndBindLeaf, rewriter.getUnitAttr());
    }
    return result;
  }
};

// If there are memref-to-memref copy op before the 1:2 split, convert
// them into equivalent tensor-to-tensor copy op.
// oldcopy: hivm.hir.copy ins(%1 : memref<..., ub>) outs(%alloc : memref<...,
// cbuf>) newcopy: %2 = hivm.hir.copy ins(%1 : tensor<...>) outs(%alloc :
// tensor<...>)
class ConvertMemRefUBToL1TensorCopyPattern
    : public OpRewritePattern<hivm::CopyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    // get src and dst, hivm.copy %src, %dst
    Value srcMemRef = copyOp->getOperand(0);
    Value dstMemRef = copyOp->getOperand(1);
    // check if memref type
    auto srcType = srcMemRef.getType().mlir::dyn_cast<MemRefType>();
    auto dstType = dstMemRef.getType().mlir::dyn_cast<MemRefType>();
    if (!srcType || !dstType)
      return failure();
    // check if hivm address space
    auto srcAS = srcType.getMemorySpace().mlir::dyn_cast<hivm::AddressSpaceAttr>();
    auto dstAS = dstType.getMemorySpace().mlir::dyn_cast<hivm::AddressSpaceAttr>();
    if (!srcAS || !dstAS)
      return failure();
    // check if ub to l1
    if (srcAS.getAddressSpace() != hivm::AddressSpace::UB ||
        dstAS.getAddressSpace() != hivm::AddressSpace::L1)
      return failure();

    auto toMemRefOp = srcMemRef.getDefiningOp<bufferization::ToMemrefOp>();
    if (!toMemRefOp)
      return failure();
    // get origin tensor
    Value srcTensor = toMemRefOp.getTensor();
    // support vtranspose -> tensor -> to_memref -> copy to rewrite
    if (!srcTensor.getDefiningOp<hivm::VTransposeOp>())
      return failure();

    rewriter.setInsertionPoint(copyOp);
    // memref<..., L1> to memref<...>
    auto castedMemRefType =
        MemRefType::get(dstType.getShape(), dstType.getElementType());
    auto memSpaceCast = rewriter.create<memref::MemorySpaceCastOp>(
        copyOp.getLoc(), castedMemRefType, dstMemRef);

    // bufferization.to_tensor
    auto dstTensor = rewriter.create<bufferization::ToTensorOp>(
        copyOp.getLoc(), memSpaceCast.getResult(),
        /*restrict=*/true,
        /*writable=*/true);

    auto newCopy = rewriter.create<hivm::CopyOp>(
        copyOp.getLoc(), srcTensor.getType(), srcTensor, dstTensor.getResult());

    rewriter.setInsertionPointAfter(newCopy);
    rewriter.create<annotation::MarkOp>(newCopy.getLoc(), newCopy.getResult(0));

    rewriter.eraseOp(copyOp);

    if (toMemRefOp->use_empty())
      rewriter.eraseOp(toMemRefOp);

    return success();
  }
};

/// add if (sublock_id == 0) guard for each store/copy op.
/// case 1: store/copy op without results
///   store/copy op
/// is changed to
///   if (subblock_id == 0)
///     store/copy op
/// case 2: store/copy op with results
///   %res = store/copy op
/// is changed to
///   if (subblock_id == 0)
///     %res = store/copy op
///     yield %res
///   else
///     yield store/copy's outs
template <typename OpType>
struct LimitUniqueSubBlockIdToStoreCopy : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (auto ifOpOld = dyn_cast_if_present<scf::IfOp>(op->getParentOp())) {
      if (ifOpOld->template hasAttrOfType<UnitAttr>(kLimitedSubBlockOpAttrName))
        return failure();
    }

    // Copy operations on A2/A3 represent ub-to-ub transfers, whereas on A5 they
    // can be either ub-to-ub or ub-to-l1, with only ub-to-l1 used for CV1:1.
    if constexpr (std::is_same_v<hivm::CopyOp, OpType>) {
      if (!isCopytoL1(op.getOperation())) {
        return failure();
      }
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
      ifOp.getThenBodyBuilder().template create<scf::YieldOp>(loc, thenYield);
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
  patterns.add<LimitUniqueSubBlockIdToStoreCopy<hivm::StoreOp>>(
      funcOp.getContext());
  patterns.add<LimitUniqueSubBlockIdToStoreCopy<hivm::CopyOp>>(
      funcOp.getContext());
  GreedyRewriteConfig config;
  config.maxIterations = kMaxIterations;
  return applyPatternsGreedily(funcOp, std::move(patterns), config);
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
  subBlockLoop->setAttr(utils::kMapForToForallAttrName,
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

static void populateBindSubBlockBubbleUpPassManager(PassManager &pm,
                                                    bool strictMode) {
  HIVMBubbleUpExtractSliceOptions options;
  options.strictMode = strictMode;
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createHIVMBubbleUpExtractSlicePass(options));
  pm.addPass(createCSEPass());
}

static LogicalResult
tileAndSliceOp(func::FuncOp func,
               DenseMap<int32_t, int64_t> &tightlyCoupledBufferToTilingDim,
               bool &isBroadcastAxisCase) {
  hivm::detail::DimensionAnalyzer analyzer(func);
  LDBG("Before analyzer: " << func);
  if (failed(analyzer.initialize()))
    return failure();

  if (analyzer.computeTilingDim()) {
    isBroadcastAxisCase = true;
  }

  func->walk([&](annotation::MarkOp markOp) {
    if (auto attr = markOp->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
            hivm::HIVMTightlyCoupledBufferAttr::name)) {
      auto tilingDim = analyzer.getTilingDim(markOp.getSrc());
      markOp->setAttr(
          "hivm.tiling_dim",
          IntegerAttr::get(IndexType::get(markOp.getContext()), tilingDim));
      auto maybeId = attr.getId();
      if (!maybeId) {
        markOp.emitError() << "Missing id in HIVMTightlyCoupledBufferAttr";
        return;
      }
      tightlyCoupledBufferToTilingDim[maybeId.value()] = tilingDim;
    }
  });

  // Check there is no dynamic shape store, if there is, we cannot tile it to 2
  // for now.
  std::vector<hivm::StoreOp> allStoreOps;
  func->walk([&allStoreOps](hivm::StoreOp storeOp) {
    allStoreOps.push_back(storeOp);
  });

  RewritePatternSet patterns(func->getContext());
  patterns.add<ConvertMemRefUBToL1TensorCopyPattern>(func->getContext());
  patterns.add<TileAndSliceStoreCopyOp<hivm::StoreOp>>(func->getContext(),
                                                       analyzer);
  patterns.add<TileAndSliceStoreCopyOp<hivm::CopyOp>>(func->getContext(),
                                                      analyzer);
  patterns.add<TileAndSliceLeaf<scf::ForOp>>(func->getContext(), analyzer);
  patterns.add<TileAndSliceLeaf<scf::IfOp>>(func->getContext(), analyzer);
  GreedyRewriteConfig config;
  config.maxIterations = kMaxIterations;
  auto ret = applyPatternsGreedily(func, std::move(patterns), config);
  if (func.walk([](hivm::StoreOp storeOp) {
            return storeOp->hasAttrOfType<UnitAttr>(tileAndSliceFailure)
                       ? mlir::WalkResult::interrupt()
                       : mlir::WalkResult::advance();
          })
          .wasInterrupted()) {
    return failure();
  }
  return ret;
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

  bool isBroadcastAxisCase = false;
  
  PassManager pm(newFunc->getContext());
  pm.addPass(tensor::createReplicateOutEmptyTensorPass());

  if (failed(pm.run(newFunc)) ||
      failed(tileAndSliceOp(newFunc, tightlyCoupledBufferToTilingDim, isBroadcastAxisCase))) {
    failAndRevert(newFunc);
    return failure();
  }

  if (isBroadcastAxisCase) {
    strictMode = false;
  }

  // If all the pattern fails due to the tilingDim=-1
  // walk through the store op and copy op
  bool isFailed = true;
  newFunc->walk([&isFailed](Operation *op) {
    if (!isa<hivm::StoreOp, hivm::CopyOp>(op)) {
      return WalkResult::advance();
    }
    if (op->hasAttr(tileAndSliceFailure)) {
      op->removeAttr(tileAndSliceFailure);
      if (op->hasAttr(hivm::AtomicKindAttr::name)) {
        isFailed = true;
        return WalkResult::interrupt();
      }
    } else {
      isFailed = false;
    }
    return WalkResult::advance();
  });

  SmallVector<Operation *> toBeRemoved;
  newFunc->walk([&](annotation::MarkOp op) {
    if (op.isAnnotatedBy(kTilingDimMappingAttrName)) {
      toBeRemoved.push_back(op);
    }
  });
  for (auto *op : toBeRemoved) {
    op->erase();
  }

  if (isFailed) {
    failAndRevert(newFunc);
    return failure();
  }

  PassManager pm2(newFunc->getContext());
  populateBindSubBlockBubbleUpPassManager(pm2, strictMode);

  LogicalResult bubbleUpResult = pm2.run(newFunc);
  if (bubbleUpResult.failed() || newFunc.verify().failed() ||
      newFunc.verifyBody().failed() || newFunc.verifyRegions().failed()) {
    failAndRevert(newFunc);
    return failure();
  }

  RewritePatternSet patternsPost(&getContext());
  patternsPost.add<mlir::hivm::detail::BubbleUpSubviewFromTiling>(
      &getContext());
  if (failed(applyPatternsGreedily(newFunc, std::move(patternsPost)))) {
    failAndRevert(newFunc);
    return failure();
  }

  return newFunc;
}

namespace {
struct splitFixpipe : public OpRewritePattern<hivm::FixpipeOp> {
public:
  using OpRewritePattern<hivm::FixpipeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::FixpipeOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr(tileAndSliceFailure)) {
      return failure();
    }
    Value dst = op.getDst();
    if (op.getDualDstModeAttr()) {
      return failure();
    }
    // Determine the address space of the destination operand of the fixpipe
    // instruction.
    auto dstMemrefType = dyn_cast<MemRefType>(dst.getType());
    if (!dstMemrefType)
      return failure();
    auto dstMemorySpace = dstMemrefType.getMemorySpace();
    if (!dstMemorySpace)
      return failure();
    auto toAddrSpace =
        cast<hivm::AddressSpaceAttr>(dstMemorySpace).getAddressSpace();
    if ((!dstMemorySpace) || (toAddrSpace != hivm::AddressSpace::UB)) {
      return success();
    }

    // Determine whether to enable the CV pipeline.
    bool cvpipeFlag = true;
    auto subviewOp = dst.getDefiningOp<memref::SubViewOp>();
    if (!subviewOp)
      cvpipeFlag = false;
    auto maybeAllocOp = traceDefOp<memref::AllocOp>(dst);
    if (!maybeAllocOp)
      return failure();
    memref::AllocOp allocOp = cast<memref::AllocOp>(*maybeAllocOp);
    mlir::Value allocVal = allocOp.getResult();
    auto maybeMarkOpRaw =
        utils::getAnnotateOpWithAttr(allocVal, "hivm.tightly_coupled_buffer");
    if (!maybeMarkOpRaw)
      return failure();
    auto markOp = dyn_cast<annotation::MarkOp>(*maybeMarkOpRaw);
    if (!markOp)
      return failure();
    auto attr = markOp->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
        "hivm.tightly_coupled_buffer");
    if (!attr || !attr.getId().has_value())
      return failure();

    auto tilingDimAttr = markOp->getAttrOfType<IntegerAttr>("hivm.tiling_dim");
    if (!tilingDimAttr)
      return failure();
    int64_t tilingDim = tilingDimAttr.getValue().getSExtValue();
    auto rank = allocOp.getType().getRank();
    if (tilingDim == -1 || (tilingDim != rank - 2 && tilingDim != rank - 1))
      return failure();
    auto splitMode = tilingDim == rank - 2
                         ? hivm::FixpipeDualDstMode::ROW_SPLIT
                         : hivm::FixpipeDualDstMode::COLUMN_SPLIT;
    auto oldTy = cast<MemRefType>(allocVal.getType());
    auto shape = llvm::to_vector(oldTy.getShape());
    // TODO: support NZ2DN
    auto splitShape = [](bool cvpipeFlag, hivm::FixpipeDualDstMode splitMode,
                         SmallVector<int64_t> &shape) -> LogicalResult {
      int64_t splitIdx = 0;
      splitIdx += cvpipeFlag;
      int64_t constraints;
      if (splitMode == FixpipeDualDstMode::ROW_SPLIT)
        constraints = 2;
      else {
        constraints = 32;
        ++splitIdx;
      }
      auto size = shape[splitIdx];
      if (ShapedType::isDynamicShape(size)) {
        return failure();
      }
      if ((size % constraints) != 0) {
        return failure();
      }
      shape[splitIdx] = size / 2;
      return success();
    };

    if (!cvpipeFlag) {
      if (llvm::failed(splitShape(cvpipeFlag, splitMode, shape))) {
        op->setAttr(tileAndSliceFailure, rewriter.getUnitAttr());
        return failure();
      }
      auto newTy = MemRefType::get(shape, oldTy.getElementType(),
                                   oldTy.getLayout(), oldTy.getMemorySpace());
      // new alloc + new mark + new fixpipe
      rewriter.setInsertionPoint(allocOp);
      auto newAlloc = rewriter.create<memref::AllocOp>(allocOp.getLoc(), newTy);

      rewriter.setInsertionPoint(markOp);
      auto newMark =
          rewriter.create<annotation::MarkOp>(markOp->getLoc(), newAlloc);
      rewriter.modifyOpInPlace(newMark,
                               [&] { newMark->setAttrs(markOp->getAttrs()); });
      auto dualAttr =
          hivm::FixpipeDualDstModeAttr::get(rewriter.getContext(), splitMode);
      rewriter.setInsertionPoint(op);
      SmallVector<Value> oprs({op.getSrc(), newAlloc});
      if (auto quantScale = op.getQuantScale())
        oprs.push_back(quantScale);
      auto newFixpipeOp = rewriter.create<hivm::FixpipeOp>(
          op.getLoc(), TypeRange{}, oprs, op->getAttrs());
      newFixpipeOp.setDualDstModeAttr(dualAttr);

      rewriter.replaceAllUsesWith(allocVal, newAlloc.getResult());
      rewriter.replaceOp(op, newFixpipeOp->getResults());
      rewriter.eraseOp(markOp);
      rewriter.eraseOp(allocOp);
      return success();
    }

    if (llvm::failed(splitShape(cvpipeFlag, splitMode, shape))) {
      op->setAttr(tileAndSliceFailure, rewriter.getUnitAttr());
      return failure();
    }
    auto newTy = MemRefType::get(shape, oldTy.getElementType(),
                                 oldTy.getLayout(), oldTy.getMemorySpace());

    rewriter.setInsertionPoint(allocOp);
    auto newAlloc = rewriter.create<memref::AllocOp>(allocOp.getLoc(), newTy);

    rewriter.setInsertionPoint(markOp);
    auto newMark =
        rewriter.create<annotation::MarkOp>(markOp->getLoc(), newAlloc);
    rewriter.modifyOpInPlace(newMark,
                             [&] { newMark->setAttrs(markOp->getAttrs()); });
    rewriter.setInsertionPoint(subviewOp);
    SmallVector<OpFoldResult> sizes = subviewOp.getMixedSizes();
    switch (splitMode) {
    case hivm::FixpipeDualDstMode::ROW_SPLIT:
      if (sizes[1].is<Attribute>()) {
        int64_t oldSize = cast<IntegerAttr>(sizes[1].get<Attribute>()).getInt();
        sizes[1] = rewriter.getIndexAttr(oldSize / 2);
      }
      break;
    case hivm::FixpipeDualDstMode::COLUMN_SPLIT:
      if (sizes[2].is<Attribute>()) {
        int64_t oldSize = cast<IntegerAttr>(sizes[2].get<Attribute>()).getInt();
        sizes[2] = rewriter.getIndexAttr(oldSize / 2);
      }
      break;
    default:
      break;
    }

    int64_t dim1 = sizes[1].is<Attribute>()
                       ? cast<IntegerAttr>(sizes[1].get<Attribute>()).getInt()
                       : ShapedType::kDynamic;
    int64_t dim2 = sizes[2].is<Attribute>()
                       ? cast<IntegerAttr>(sizes[2].get<Attribute>()).getInt()
                       : ShapedType::kDynamic;
    SmallVector<int64_t> new2DShape = {dim1, dim2};

    auto srcType = cast<MemRefType>(newAlloc.getType());
    Type elementType = srcType.getElementType();
    Attribute memorySpace = srcType.getMemorySpace();
    auto layout = StridedLayoutAttr::get(rewriter.getContext(),
                                         ShapedType::kDynamic, {dim2, 1});
    auto result2DType =
        MemRefType::get(new2DShape, elementType, layout, memorySpace);

    auto newSubview = rewriter.create<memref::SubViewOp>(
        subviewOp.getLoc(), result2DType, newAlloc, subviewOp.getMixedOffsets(),
        sizes, subviewOp.getMixedStrides());

    auto dualAttr =
        hivm::FixpipeDualDstModeAttr::get(rewriter.getContext(), splitMode);
    rewriter.setInsertionPoint(op);
    NamedAttrList attrs(op->getAttrs());
    attrs.set(op.getDualDstModeAttrName(), dualAttr);

    auto newFixpipeOp = rewriter.create<hivm::FixpipeOp>(
        op.getLoc(), TypeRange{}, ValueRange{op.getSrc(), newSubview},
        attrs.getAttrs());

    rewriter.replaceAllUsesWith(allocVal, newAlloc.getResult());
    rewriter.replaceAllUsesWith(subviewOp.getResult(), newSubview.getResult());
    rewriter.replaceOp(op, newFixpipeOp->getResults());

    rewriter.eraseOp(subviewOp);
    rewriter.eraseOp(markOp);
    rewriter.eraseOp(allocOp);
    return success();
  }
};
} // namespace

static LogicalResult runSplitFixpipe(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<splitFixpipe>(funcOp.getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return failure();
  }
  if (funcOp
          .walk([](hivm::FixpipeOp fixpipeOp) {
            return fixpipeOp->hasAttrOfType<UnitAttr>(tileAndSliceFailure)
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted()) {
    return failure();
  }
  return success();
}

static LogicalResult tileAndSliceOpAIC(func::FuncOp func) {
  return runSplitFixpipe(func);
}

namespace {
struct FuncRollbackBackup {
  std::string originalName;
  Operation *backupOp = nullptr;
};
} // namespace

static void createFuncBackups(ArrayRef<func::FuncOp> funcs,
                              SmallVectorImpl<FuncRollbackBackup> &backups) {
  backups.reserve(backups.size() + funcs.size());
  for (func::FuncOp func : funcs) {
    backups.push_back({func.getSymNameAttr().str(), func->clone()});
  }
}

static void destroyFuncBackups(SmallVectorImpl<FuncRollbackBackup> &backups) {
  for (auto &entry : backups) {
    if (entry.backupOp) {
      entry.backupOp->destroy();
      entry.backupOp = nullptr;
    }
  }
  backups.clear();
}

static LogicalResult
restoreFunctionsFromBackups(ModuleOp moduleOp,
                            SmallVectorImpl<FuncRollbackBackup> &backups,
                            bool limitSubBlockToStore) {
  for (auto &entry : backups) {
    if (!entry.backupOp) {
      continue;
    }
    if (auto currentFunc =
            moduleOp.lookupSymbol<func::FuncOp>(entry.originalName)) {
      currentFunc.erase();
    }
    moduleOp.push_back(entry.backupOp);
    auto restoredFunc = cast<func::FuncOp>(entry.backupOp);
    restoredFunc.setName(entry.originalName);
    entry.backupOp = nullptr;

    if (limitSubBlockToStore &&
        failed(limitUniqueSubBlockToStore(restoredFunc)))
      return failure();
  }
  backups.clear();
  return success();
}

/// Walks through all functions in the module and attempts to tile and bind
/// sub-blocks for vector functions.
///
/// Functions are cloned before transformation to allow rollback on failure.
/// If attempt to bind some block fail it will rollback to 1:1 and limit to
/// unique block to store.
void TileAndBindSubBlockPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  if (moduleOp->hasAttr("hivm.disable_auto_tile_and_bind_subblock")) {
    return;
  }
#ifndef NDEBUG
  uint64_t tiledFunctionCount = 0;
#endif

  // Collect AIC and AIV functions to process (can't modify while iterating)
  SmallVector<func::FuncOp> aicFunctions;
  SmallVector<func::FuncOp> aivFunctions;
  auto collectMixAicAndAivFuncs = [&moduleOp, &aicFunctions, &aivFunctions]() {
    moduleOp.walk([&aicFunctions, &aivFunctions](func::FuncOp func) {
      auto funcCoreType = queryFuncCoreType(func);
      if (!funcCoreType.has_value() ||
          !func->hasAttrOfType<UnitAttr>(hivm::TPartOfMixAttr::name)) {
        return;
      }
      if (funcCoreType.value() == TFuncCoreType::AIC) {
        aicFunctions.push_back(func);
      } else if (funcCoreType.value() == TFuncCoreType::AIV) {
        aivFunctions.push_back(func);
      }
    });
  };

  auto limitAllAivToSubBlock0 = [this, &aivFunctions]() -> LogicalResult {
    for (func::FuncOp aivFunc : aivFunctions) {
      auto symNameStr = aivFunc.getSymNameAttr().str();
      if (failed(limitUniqueSubBlockToStore(aivFunc))) {
        LLVM_DEBUG(DBGS() << "Failed to limit unique subblock: " << symNameStr
                          << "\n");
        signalPassFailure();
        return failure();
      }
    }
    return success();
  };

  auto hasBatchMatmulLoopInAic = [&aicFunctions]() -> bool {
    return llvm::any_of(aicFunctions, [](func::FuncOp aicFunc) {
      return aicFunc
          .walk([](hivm::MmadL1Op mmad) {
            return mmad->hasAttrOfType<UnitAttr>(hivm::batchMatmulAttr)
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted();
    });
  };

  auto eraseTilingDimMappingMarks = [moduleOp]() {
    SmallVector<Operation *> toBeErased;
    moduleOp->walk([&toBeErased](Operation *op) {
      if (auto markOp = dyn_cast<annotation::MarkOp>(op);
          markOp && markOp->hasAttr(kTilingDimMappingAttrName)) {
        toBeErased.push_back(op);
      }
    });
    for (auto *op : toBeErased) {
      op->erase();
    }
  };

  collectMixAicAndAivFuncs();

  if (!this->enableTile) {
    (void)limitAllAivToSubBlock0();
    return;
  }

  // limitUniqueSubBlockToStore vector function and skip this pass if
  // BatchMatmul is found
  if (hasBatchMatmulLoopInAic()) {
    (void)limitAllAivToSubBlock0();
    return;
  }

  SmallVector<FuncRollbackBackup> aivRollbackBackups;
  SmallVector<FuncRollbackBackup> aicRollbackBackups;
  createFuncBackups(aivFunctions, aivRollbackBackups);
  createFuncBackups(aicFunctions, aicRollbackBackups);
  auto destroyAllBackups = [&aivRollbackBackups, &aicRollbackBackups]() {
    destroyFuncBackups(aivRollbackBackups);
    destroyFuncBackups(aicRollbackBackups);
  };

  // Tile AIV functions
  bool aivSuccessFlag = false;
  auto tileAivFuncs = [this, &aivFunctions, &eraseTilingDimMappingMarks,
                       &aivSuccessFlag
#ifndef NDEBUG
                       ,
                       &tiledFunctionCount
#endif
  ]() -> LogicalResult {
    for (func::FuncOp originalFunc : aivFunctions) {
      auto symNameStr = originalFunc.getSymNameAttr().str();
      FailureOr<func::FuncOp> res = attemptBindSubBlock(originalFunc);
      eraseTilingDimMappingMarks();
      if (failed(res)) {
        if (failed(limitUniqueSubBlockToStore(originalFunc))) {
          LLVM_DEBUG(DBGS() << "Failed to limit unique subblock: " << symNameStr
                            << "\n");
          signalPassFailure();
        }
        LLVM_DEBUG(DBGS() << "Failed to transform function: " << symNameStr
                          << ", keeping original\n");
        return failure();
      }

      auto processedFunc = res.value();
      processedFunc.setName(originalFunc.getName().str() + "_processing");
      aivSuccessFlag = true;
      // Success: Remove original and rename clone
      originalFunc.erase();
      processedFunc.setName(symNameStr);
#ifndef NDEBUG
      tiledFunctionCount++;
      LLVM_DEBUG(DBGS() << "Successfully transformed function #"
                        << tiledFunctionCount << ": " << symNameStr << "\n");
#endif
    }
    return success();
  };

  if (failed(tileAivFuncs())) {
    destroyAllBackups();
    return;
  }

  // Tile AIC functions for Ascend 950
  bool archIs950 = hacc::utils::isAscend950(moduleOp);
  auto tileAicFixpipeFuncs = [this, &aivSuccessFlag, archIs950,
                              &aicFunctions]() -> LogicalResult {
    if (!(aivSuccessFlag && archIs950)) {
      return success();
    }

    for (func::FuncOp originalFunc : aicFunctions) {
      originalFunc->walk([this](annotation::MarkOp markOp) {
        if (auto attr =
                markOp->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
                    hivm::HIVMTightlyCoupledBufferAttr::name)) {
          auto maybeId = attr.getId();
          if (!maybeId) {
            markOp.emitError() << "Missing id in HIVMTightlyCoupledBufferAttr";
            return;
          }
          auto id = maybeId.value();
          int64_t tilingDim = -1;
          if (tightlyCoupledBufferToTilingDim.contains(id)) {
            tilingDim = tightlyCoupledBufferToTilingDim.at(id);
          }
          markOp->setAttr(
              "hivm.tiling_dim",
              IntegerAttr::get(IndexType::get(markOp.getContext()), tilingDim));
        }
      });
      if (failed(tileAndSliceOpAIC(originalFunc))) {
        return failure();
      }
    }
    return success();
  };

  if (failed(tileAicFixpipeFuncs())) {
    if (failed(restoreFunctionsFromBackups(moduleOp, aicRollbackBackups,
                                           /*limitSubBlockToStore=*/false)) ||
        failed(restoreFunctionsFromBackups(moduleOp, aivRollbackBackups,
                                           /*limitSubBlockToStore=*/true))) {
      signalPassFailure();
    }
    destroyAllBackups();
    return;
  }
  destroyAllBackups();

#ifndef NDEBUG
  LLVM_DEBUG(DBGS() << "TileAndBindSubBlock pass completed. "
                    << "Successfully transformed " << tiledFunctionCount
                    << " functions.\n");
#endif
}

std::unique_ptr<Pass> mlir::hivm::createTileAndBindSubBlockPass(
    const TileAndBindSubBlockOptions &options) {
  return std::make_unique<TileAndBindSubBlockPass>(options);
}
