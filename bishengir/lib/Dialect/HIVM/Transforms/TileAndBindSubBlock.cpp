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
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/HoistAffine.h"
#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/Pattern.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Transforms/TileUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Passes.h"
#include "bishengir/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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
#include "llvm/ADT/ScopeExit.h"
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

static void modifyOpToSliced(RewriterBase &rewriter, OpOperand *operand,
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

template <typename OpType>
LogicalResult modifyStoreCopyOp(OpType Op, int64_t tilingDim, OpOperand *srcOpr,
                                OpOperand *dstOpr, scf::ForOp containingLoop,
                                PatternRewriter &rewriter) {
  auto loc = Op->getLoc();
  auto src = srcOpr->get();
  auto srcType = cast<ShapedType>(src.getType());
  auto maybeSingleTileSize =
      getSingleTileSize(rewriter, loc, src, tilingDim, containingLoop);
  if (failed(maybeSingleTileSize))
    return failure();
  rewriter.setInsertionPointToStart(containingLoop.getBody());
  auto offsetAtTileDim = calculateOffsetAtTilingDim(
      rewriter, loc, containingLoop, maybeSingleTileSize.value());

  rewriter.setInsertionPoint(Op);

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
    rewriter.setInsertionPointAfterValue(offsetAtTileDim.template get<Value>());
  }
  modifyOpToSliced(rewriter, srcOpr, mixedOffsets, mixedSize, mixedStrides,
                   newShape);
  if (containingLoop.getRegion().isAncestor(dstOpr->get().getParentRegion())) {
    rewriter.setInsertionPointAfterValue(dstOpr->get());
  } else {
    rewriter.setInsertionPointAfterValue(offsetAtTileDim.template get<Value>());
  }
  modifyOpToSliced(rewriter, dstOpr, mixedOffsets, mixedSize, mixedStrides,
                   newShape);
  rewriter.modifyOpInPlace(Op, [&]() {
    if (Op->getNumResults() > 0)
      Op->getResult(0).setType(Op.getDst().getType());
    Op->setAttr(tiledOp, rewriter.getUnitAttr());
  });
  return success();
}

namespace {

/// try to tile storeOp and copyOp and bind sub block mapping
template <typename OpType>
class TileAndSliceStoreCopyOp : public OpRewritePattern<OpType> {
public:
  hivm::detail::DimensionAnalyzer &analyzer;

  explicit TileAndSliceStoreCopyOp(MLIRContext *context,
                                   hivm::detail::DimensionAnalyzer &analyzer)
      : OpRewritePattern<OpType>(context, /*benefit=*/1), analyzer(analyzer) {}
  LogicalResult matchAndRewrite(OpType Op,
                                PatternRewriter &rewriter) const override {
    if (Op->template hasAttrOfType<UnitAttr>(tiledOp))
      return failure();
    int64_t tilingDim = analyzer.getTilingDim(Op.getSrc());
    auto inputType = Op.getOperand(0).getType();
 	  if (!inputType) {
 	    return failure();
 	  }
    /// We differentiate storeOp and copyOp
    if constexpr (std::is_same_v<hivm::CopyOp, OpType>) {
      if (!Op.getResults().empty()) { // If copy Op with results
        // TODO: Check if there is case of copyOp with result, if no, delete here.
        if (!llvm::any_of(Op->getUsers(), [](Operation *user) {
              return isa<annotation::MarkOp>(user);
            })) {
          return failure(); // If the user of CopyOp is not MarkOp, it cannot be
                            // a tiling start point.
        }
      }
      // If the copy input is memref, we cannot tile it.
      if (isa<mlir::MemRefType>(inputType)) {
        Op->emitWarning(
            "Copy input memref is not supported, skip tile and bind.");
        return failure();
      }
      LLVM_DEBUG(DBGS() << "The copy op tiling dim is: " << tilingDim << "\n");
    } else {
      LLVM_DEBUG(DBGS() << "The store op tiling dim is: " << tilingDim << "\n");
    }
    auto maybeContainingLoop = findContainingSubblockLoop(Op);
    if (tilingDim == -1 || failed(maybeContainingLoop)) {
      Op->setAttr(tileAndSliceFailure, rewriter.getUnitAttr());
      return failure();
    }
    auto containingLoop = maybeContainingLoop.value();
    auto *srcOpr = &Op.getSrcMutable();
    auto *dstOpr = &Op.getDstMutable();
    if constexpr (std::is_same_v<hivm::StoreOp, OpType>) {
      auto storeOp = cast<hivm::StoreOp>(Op);
      auto srcType = dyn_cast<ShapedType>(storeOp.getSrc().getType());
      if (!srcType) {
        Op->setAttr(tileAndSliceFailure, rewriter.getUnitAttr());
        return failure();
      }

      /// Handling masked store
      /// Maskedstore contains the static_mask and dynamic_mask cases, so we
      /// handle the maskedStore first, this will also handle the dynamicStore.
      /// After handleMasked, it should not have dynamicShape case.
      if (failed(handleMaskedStore(storeOp, tilingDim, containingLoop,
                                   rewriter))) {
        if (ShapedType::isDynamicShape(srcType.getShape())) {
          Op->setAttr(tileAndSliceFailure, rewriter.getUnitAttr());
          return failure();
        }

        /// Handle the storeOp without mask
        if (failed(modifyStoreCopyOp(Op, tilingDim, srcOpr, dstOpr,
                                     containingLoop, rewriter))) {
          Op->setAttr(tileAndSliceFailure, rewriter.getUnitAttr());
          return failure();
        }
      }
    } else {
      /// Handle the copyOp, we assump copyOp does not have a mask
      if (failed(modifyStoreCopyOp(Op, tilingDim, srcOpr, dstOpr,
                                   containingLoop, rewriter))) {
        Op->setAttr(tileAndSliceFailure, rewriter.getUnitAttr());
        return failure();
      }
    }

    // Maybe we need to maintain this map when doing bubble up.
    DenseMap<Operation *, Operation *> map;
    map[Op] = Op;
    setBufferSizeInLoopOp(rewriter, Op.getLoc(), containingLoop, map);
    LDBG("Success");
    return success();
  }

private:
  LogicalResult handleMaskedStore(hivm::StoreOp storeOp, int64_t tilingDim,
                                  scf::ForOp containingLoop,
                                  PatternRewriter &rewriter) const {
    auto *srcOpr = &storeOp.getSrcMutable();
    auto *dstOpr = &storeOp.getDstMutable();
    auto src = srcOpr->get();
    auto dst = dstOpr->get();
    SmallVector<OpFoldResult, 4> srcOffsets;
    SmallVector<OpFoldResult, 4> dstOffsets;
    SmallVector<OpFoldResult, 4> srcSizes;
    SmallVector<OpFoldResult, 4> dstSizes;

    if (auto extractSliceOp = src.getDefiningOp<tensor::ExtractSliceOp>()) {
      src = extractSliceOp.getSource();
      srcOpr = &extractSliceOp.getSourceMutable();
      srcOffsets = extractSliceOp.getMixedOffsets();
      srcSizes = extractSliceOp.getMixedSizes();
    } else if (auto subViewOp = src.getDefiningOp<memref::SubViewOp>()) {
      src = subViewOp.getSource();
      srcOpr = &subViewOp.getSourceMutable();
      srcOffsets = subViewOp.getMixedOffsets();
      srcSizes = subViewOp.getMixedSizes();
    } else {
      return failure();
    }
    if (auto extractSliceOp = dst.getDefiningOp<tensor::ExtractSliceOp>()) {
      dst = extractSliceOp.getSource();
      dstOpr = &extractSliceOp.getSourceMutable();
      dstOffsets = extractSliceOp.getMixedOffsets();
      dstSizes = extractSliceOp.getMixedSizes();
    } else if (auto subViewOp = dst.getDefiningOp<memref::SubViewOp>()) {
      dst = subViewOp.getSource();
      dstOpr = &subViewOp.getSourceMutable();
      dstOffsets = subViewOp.getMixedOffsets();
      dstSizes = subViewOp.getMixedSizes();
    } else {
      return failure();
    }
    auto srcShape = llvm::to_vector(cast<ShapedType>(src.getType()).getShape());
    auto dstShape = llvm::to_vector(cast<ShapedType>(dst.getType()).getShape());
    auto srcOp =
        storeOp.getSrc().getDefiningOp<OffsetSizeAndStrideOpInterface>();
    auto dstOp =
        storeOp.getDst().getDefiningOp<OffsetSizeAndStrideOpInterface>();
    if (srcShape != dstShape || ShapedType::isDynamicShape(srcShape) ||
        !srcOp.hasUnitStride() || !dstOp.hasUnitStride())
      return failure();
    if (failed(modifyStoreCopyOp(storeOp, tilingDim, srcOpr, dstOpr,
                                 containingLoop, rewriter)))
      return failure();
    auto loc = storeOp.getLoc();
    rewriter.setInsertionPoint(storeOp);
    auto maybeSingleTileSize =
        getSingleTileSize(rewriter, loc, src, tilingDim, containingLoop);
    rewriter.setInsertionPointToStart(containingLoop.getBody());
    auto offsetAtTileDim = calculateOffsetAtTilingDim(
        rewriter, loc, containingLoop, maybeSingleTileSize.value());

    auto lb = getValueOrCreateConstantIndexOp(rewriter, loc, offsetAtTileDim);
    auto ub = getValueOrCreateConstantIndexOp(rewriter, loc,
                                              maybeSingleTileSize.value());

    src = storeOp.getSrc();
    dst = storeOp.getDst();

    if (auto extractSliceOp = src.getDefiningOp<tensor::ExtractSliceOp>()) {
      rewriter.setInsertionPoint(extractSliceOp);
      handleExtractOfExtract(srcOffsets[tilingDim], srcSizes[tilingDim], lb, ub,
                             loc, rewriter);
      src = rewriter.create<tensor::ExtractSliceOp>(
          extractSliceOp.getLoc(), extractSliceOp.getSource(), srcOffsets,
          srcSizes, extractSliceOp.getMixedStrides());
    } else if (auto subViewOp = src.getDefiningOp<memref::SubViewOp>()) {
      rewriter.setInsertionPoint(subViewOp);
      handleExtractOfExtract(srcOffsets[tilingDim], srcSizes[tilingDim], lb, ub,
                             loc, rewriter);
      src = rewriter.create<memref::SubViewOp>(
          subViewOp.getLoc(), subViewOp.getSource(), srcOffsets, srcSizes,
          subViewOp.getMixedStrides());
    }
    if (auto extractSliceOp = dst.getDefiningOp<tensor::ExtractSliceOp>()) {
      rewriter.setInsertionPoint(extractSliceOp);
      handleExtractOfExtract(dstOffsets[tilingDim], dstSizes[tilingDim], lb, ub,
                             loc, rewriter);
      dst = rewriter.create<tensor::ExtractSliceOp>(
          extractSliceOp.getLoc(), extractSliceOp.getSource(), dstOffsets,
          dstSizes, extractSliceOp.getMixedStrides());
    } else if (auto subViewOp = dst.getDefiningOp<memref::SubViewOp>()) {
      rewriter.setInsertionPoint(subViewOp);
      handleExtractOfExtract(dstOffsets[tilingDim], dstSizes[tilingDim], lb, ub,
                             loc, rewriter);
      dst = rewriter.create<memref::SubViewOp>(
          subViewOp.getLoc(), subViewOp.getSource(), dstOffsets, dstSizes,
          subViewOp.getMixedStrides());
    }

    rewriter.modifyOpInPlace(storeOp, [&]() {
      storeOp.getSrcMutable().set(src);
      storeOp.getDstMutable().set(dst);
    });

    LDBG("New dynamic store:\n" << src << '\n' << dst << '\n' << storeOp);

    return success();
  }
};

static void modifyDebugOpToSliced(RewriterBase &rewriter, hivm::DebugOp debugOp,
                                  SmallVector<OpFoldResult, 4> mixedOffsets,
                                  SmallVector<OpFoldResult, 4> mixedSize,
                                  SmallVector<OpFoldResult, 4> mixedStrides,
                                  SmallVector<int64_t, 4> newShape) {
  auto rankType = cast<RankedTensorType>(debugOp.getArg().getType());
  auto loc = debugOp->getLoc();

  auto newType =
      mlir::RankedTensorType::get(newShape, rankType.getElementType());
  auto slicedDebug = rewriter.create<tensor::ExtractSliceOp>(
      loc, newType, debugOp->getOperand(0), mixedOffsets, mixedSize,
      mixedStrides);
  markCreatedExtractSliceOp(rewriter, slicedDebug);
  rewriter.modifyOpInPlace(debugOp, [&]() {
    debugOp->setOperand(0, slicedDebug);
    debugOp->setAttr(tiledOp, UnitAttr::get(debugOp->getContext()));
  });
}

/// try to tile debug ops and bind sub block mapping
class TileAndSliceDebugOp : public OpRewritePattern<hivm::DebugOp> {
public:
  hivm::detail::DimensionAnalyzer &analyzer;

  explicit TileAndSliceDebugOp(MLIRContext *context,
                               hivm::detail::DimensionAnalyzer &analyzer)
      : OpRewritePattern<hivm::DebugOp>(context, /*benefit=*/1),
        analyzer(analyzer) {}
  LogicalResult matchAndRewrite(hivm::DebugOp debugOp,
                                PatternRewriter &rewriter) const override {
    if (debugOp->hasAttrOfType<UnitAttr>(tiledOp))
      return failure();
    int64_t tilingDim = analyzer.getTilingDim(debugOp.getArg());
    auto maybeContainingLoop = findContainingSubblockLoop(debugOp);
    if (tilingDim == -1 || failed(maybeContainingLoop))
      return failure();

    auto containingLoop = maybeContainingLoop.value();
    auto loc = debugOp.getLoc();
    auto maybeSingleTileSize = getSingleTileSize(
        rewriter, loc, debugOp.getArg(), tilingDim, containingLoop);
    if (failed(maybeSingleTileSize))
      return failure();
    rewriter.setInsertionPointToStart(containingLoop.getBody());
    auto offsetAtTileDim = calculateOffsetAtTilingDim(
        rewriter, loc, containingLoop, maybeSingleTileSize.value());

    rewriter.setInsertionPoint(debugOp);

    SmallVector<OpFoldResult, 4> mixedStrides, mixedOffsets, mixedSize;
    SmallVector<int64_t, 4> newShape;
    auto rankType = cast<ShapedType>(debugOp.getArg().getType());
    assert(!ShapedType::isDynamicShape(rankType.getShape()));
    if (failed(findCorrespondingSizesOffsetsStrides(
            rewriter, rankType, tilingDim, offsetAtTileDim,
            maybeSingleTileSize.value(), mixedStrides, mixedOffsets, mixedSize,
            newShape)))
      return failure();

    modifyDebugOpToSliced(rewriter, debugOp, mixedOffsets, mixedSize,
                          mixedStrides, newShape);

    // Maybe we need to maintain this map when doing bubble up.
    DenseMap<Operation *, Operation *> map;
    map[debugOp] = debugOp;
    setBufferSizeInLoopOp(rewriter, loc, containingLoop, map);

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
      int64_t tilingDim = analyzer.getTilingDim(res);
      if (tilingDim == -1 || !res.use_empty())
        continue;

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

/// add if (sublock_id == 0) guard for each store/copy op.
/// e.g.
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

LogicalResult mlir::hivm::limitUniqueSubBlockToStore(func::FuncOp funcOp) {
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

static void populateBindSubBlockBubbleUpPassManager(PassManager &pm,
                                                    bool strictMode) {
  HIVMBubbleUpExtractSliceOptions bubbleUpOptions;
  bubbleUpOptions.strictMode = strictMode;
  pm.addPass(createHIVMBubbleUpExtractSlicePass(bubbleUpOptions));
  CanonicalizerOptions options;
  SmallVector<std::string> disabledPatterns(
      {"ReinterpretCastConstantArgumentFolder"});
  options.disabledPatterns = disabledPatterns;
  pm.addPass(bishengir::createExtendedCanonicalizerPass(options));
  pm.addPass(createCSEPass());
}

static LogicalResult
tileAndSliceOp(func::FuncOp func,
               DenseMap<int32_t, int64_t> &tightlyCoupledBufferToTilingDim,
               bool &isBroadcastAxisCase) {
  IRRewriter rewriter(func.getContext());
  func->walk([&rewriter](Operation *op) {
    if (!isa<tensor::ExtractSliceOp, memref::SubViewOp>(op) || op->hasOneUse())
      return;
    rewriter.setInsertionPoint(op);
    SmallVector<OpOperand *> uses;
    for (auto &use : op->getUses())
      uses.push_back(&use);
    for (auto *use : uses) {
      auto *user = use->getOwner();
      rewriter.modifyOpInPlace(
          user, [&]() { use->set(rewriter.clone(*op)->getResult(0)); });
    }
  });
  hivm::detail::DimensionAnalyzer analyzer(func);
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
          AICAttrTilingDim,
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
  if (llvm::any_of(allStoreOps, [&](hivm::StoreOp storeOp) {
        auto srcShapedType = dyn_cast<ShapedType>(storeOp.getSrcOperandType());
        auto dstShapedType = dyn_cast<ShapedType>(storeOp.getDstOperandType());
        if (!srcShapedType || !dstShapedType)
          return true;
        auto parentForOp = storeOp->getParentOfType<scf::ForOp>();
        if (parentForOp && parentForOp->hasAttr(ExtractLoadStoreAttr))
          return true;
        if (ShapedType::isDynamicShape(srcShapedType.getShape()) ||
            ShapedType::isDynamicShape(dstShapedType.getShape())) {
          auto src = storeOp.getSrc();
          auto dst = storeOp.getDst();
          if (auto extractSliceOp =
                  src.getDefiningOp<tensor::ExtractSliceOp>()) {
            src = extractSliceOp.getSource();
          } else if (auto subViewOp = src.getDefiningOp<memref::SubViewOp>()) {
            src = subViewOp.getSource();
          }
          if (auto extractSliceOp =
                  dst.getDefiningOp<tensor::ExtractSliceOp>()) {
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
  patterns.add<TileAndSliceStoreCopyOp<hivm::StoreOp>>(func->getContext(),
                                                       analyzer);
  patterns.add<TileAndSliceStoreCopyOp<hivm::CopyOp>>(func->getContext(),
                                                      analyzer);
  patterns.add<TileAndSliceDebugOp>(func->getContext(), analyzer);
  patterns.add<TileAndSliceLeaf<scf::ForOp>, TileAndSliceLeaf<scf::WhileOp>,
               TileAndSliceLeaf<scf::IfOp>>(func->getContext(), analyzer);
  GreedyRewriteConfig config;
  config.maxIterations = kMaxIterations;
  if (failed(applyPatternsGreedily(func, std::move(patterns), config))) {
    return failure();
  }
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

  bool isBroadcastAxisCase = false;

  PassManager pm(newFunc->getContext());
  pm.addPass(tensor::createReplicateOutEmptyTensorPass());

  if (failed(pm.run((newFunc))) ||
      failed(tileAndSliceOp(newFunc, tightlyCoupledBufferToTilingDim,
                            isBroadcastAxisCase))) {
    failAndRevert(newFunc);
    return failure();
  }

  if (isBroadcastAxisCase) {
    emitRemark(newFunc.getLoc())
        << "Selected tiling dim might have broadcast two different axis. "
           "Automatically disables strict mode.";
    if (!func.walk([](scf::ForOp forOp) {
               return forOp->hasAttr(ExtractLoadStoreAttr)
                          ? WalkResult::interrupt()
                          : WalkResult::advance();
             })
             .wasInterrupted()) {
      strictMode = false;
    }
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
      if (auto storeOp = dyn_cast<hivm::StoreOp>(op);
          storeOp && storeOp.isAtomic()) {
        isFailed = true;
        return WalkResult::interrupt();
      }
    } else {
      isFailed = false;
    }
    return WalkResult::advance();
  });

  if (isFailed) {
    failAndRevert(newFunc);
    return failure();
  }

  LDBG("After tileAndSliceStore: " << newFunc);

  PassManager pm2(newFunc->getContext());
  populateBindSubBlockBubbleUpPassManager(pm2, strictMode);

  LogicalResult bubbleUpResult = pm2.run(newFunc);
  if (bubbleUpResult.failed() || newFunc.verify().failed() ||
      newFunc.verifyBody().failed() || newFunc.verifyRegions().failed()) {
    failAndRevert(newFunc);
    return failure();
  }

  SmallVector<Operation *> toBeRemoved;
  newFunc->walk([&](annotation::MarkOp op) {
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
  // Ensure temporary tiling-dim mapping marks are removed on every exit path.
  auto removeTilingDimMappingMarksOnExit = llvm::make_scope_exit(
      [moduleOp]() { removeTilingDimMappingMarksFromModule(moduleOp); });

  if (moduleOp->hasAttr("hivm.disable_auto_tile_and_bind_subblock")) {
    return;
  }
#ifndef NDEBUG
  uint64_t tiledFunctionCount = 0;
#endif

  runTileAndBindSubBlockEarlyPatterns(moduleOp);

  // Collect the AIV and AIC funcs in moduleOp to tile.
  SmallVector<func::FuncOp> aicFunctions;
  SmallVector<func::FuncOp> aivFunctions;
  collectMixAicAndAivFuncs(moduleOp, aicFunctions, aivFunctions);

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

  if (!this->enableTile) {
    (void)limitAllAivToSubBlock0();
    return;
  }

  // limitUniqueSubBlockToStore vector function and skip this pass if
  // BatchMatmul is found
  if (hasBatchMatmulLoopInAicFuncs(aicFunctions)) {
    (void)limitAllAivToSubBlock0();
    return;
  }

  // FIXME: Currently, implicit tranpose's load is not tiled. The data is fully
  // loaded and extracted to use. In some cases, the extract slice is not fused
  // into the vector function, which will lead to precision error because of the
  // function boundary setting of one-shot-bufferize. So we don't tile cases
  // with implicit transpose at all to avoid the problem for now.
  if (hasImplicitTransposeWithLastAxisInAiv(aivFunctions)) {
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

  // Step 1: Tile AIV functions
  bool aivSuccessFlag = false;
  auto tileAivFuncs = [this, &aivFunctions, &aivSuccessFlag
#ifndef NDEBUG
                       ,
                       &tiledFunctionCount
#endif
  ]() -> LogicalResult {
    for (func::FuncOp originalFunc : aivFunctions) {
      auto symNameStr = originalFunc.getSymNameAttr().str();  
      FailureOr<func::FuncOp> res = attemptBindSubBlock(originalFunc);
      removeTilingDimMappingMarksFromModule(
          originalFunc->getParentOfType<ModuleOp>());
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

  // Step 2: Tile the AIC funcs, tiling AIC is only needed in Ascend910_95 Arch
  bool archIs950 = hacc::utils::isAscend910_95(moduleOp);
  if (failed(tileAicFixpipeFuncsIfNeeded(
          aicFunctions, aivSuccessFlag, archIs950,
          tightlyCoupledBufferToTilingDim))) {
    if (failed(restoreFunctionsFromBackups(moduleOp, aicRollbackBackups,
                                           /*limitSubBlockToStore=*/false)) ||
        failed(restoreFunctionsFromBackups(moduleOp, aivRollbackBackups,
                                           /*limitSubBlockToStore=*/true))) {
      LLVM_DEBUG(DBGS() << "Failed to restore from backups.\n " );
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

std::unique_ptr<Pass> mlir::hivm::createTileAndBindSubBlockPass() {
  return std::make_unique<TileAndBindSubBlockPass>();
}
