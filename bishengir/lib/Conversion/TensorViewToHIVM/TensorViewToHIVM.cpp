//===- TensorViewToHIVM.cpp - TensorView to HIVM conversion ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TensorViewToHIVM/TensorViewToHIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/TensorView/IR/TensorViewDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTENSORVIEWTOHIVM
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct TensorViewInfo {
  tv::MakeTensorViewOp rootOp;
  tv::TensorViewType viewType;
  SmallVector<int64_t> tileShape;
  SmallVector<int64_t> traversalStrides;
  SmallVector<int64_t> sparseDims;
  tv::PaddingValue paddingValue;
};

static FailureOr<TensorViewInfo> getTensorViewInfo(Value view) {
  auto encoding =
      dyn_cast_or_null<tv::TensorViewEncodingOpInterface>(view.getDefiningOp());
  if (!encoding)
    return failure();
  auto rootOp = encoding.getSource().getDefiningOp<tv::MakeTensorViewOp>();
  if (!rootOp)
    return failure();
  auto viewType = cast<tv::TensorViewType>(view.getType());
  return TensorViewInfo{rootOp,
                        viewType,
                        tv::getEncodingTileShape(viewType.getEncoding()),
                        tv::getEncodingTraversal(viewType.getEncoding()),
                        tv::getEncodingSparseDims(viewType.getEncoding()),
                        tv::getEncodingPaddingValue(viewType.getEncoding())};
}

static Value createPaddingValue(OpBuilder &builder, Location loc,
                                Type elementType,
                                tv::PaddingValue paddingValue) {
  if (paddingValue == tv::PaddingValue::ZERO)
    return builder.create<arith::ConstantOp>(loc,
                                             builder.getZeroAttr(elementType));
  auto floatType = cast<FloatType>(elementType);
  const auto &semantics = floatType.getFloatSemantics();
  llvm::APFloat value =
      paddingValue == tv::PaddingValue::NAN_VALUE
          ? llvm::APFloat::getNaN(semantics)
          : llvm::APFloat::getInf(semantics,
                                  paddingValue == tv::PaddingValue::NEG_INF);
  return builder.create<arith::ConstantOp>(loc,
                                           FloatAttr::get(floatType, value));
}

// Materialize a GM descriptor while preserving an existing memref layout.
static FailureOr<Value> getOrCreateGMBuffer(Operation *op,
                                            TensorViewInfo &viewInfo,
                                            PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto gmSpace =
      hivm::AddressSpaceAttr::get(op->getContext(), hivm::AddressSpace::GM);
  Value gmBase = viewInfo.rootOp.getSource();
  // Recover a bounded GM descriptor when the frontend supplied an address.
  if (gmBase.getType().isInteger(64)) {
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value addressSpan = one;
    for (auto [size, stride] :
         llvm::zip(viewInfo.rootOp.getSizes(), viewInfo.rootOp.getStrides())) {
      Value extent = rewriter.create<arith::SubIOp>(loc, size, one);
      extent = rewriter.create<arith::MaxSIOp>(loc, extent, zero);
      Value span = rewriter.create<arith::MulIOp>(loc, extent, stride);
      Value negativeSpan = rewriter.create<arith::SubIOp>(loc, zero, span);
      span = rewriter.create<arith::MaxSIOp>(loc, span, negativeSpan);
      addressSpan = rewriter.create<arith::AddIOp>(loc, addressSpan, span);
    }
    auto gmType = MemRefType::get({ShapedType::kDynamic},
                                  viewInfo.viewType.getElementType(),
                                  MemRefLayoutAttrInterface{}, gmSpace);
    gmBase = rewriter.create<hivm::PointerCastOp>(loc, gmType, gmBase,
                                                  ValueRange{addressSpan});
  }
  auto gmBaseType = dyn_cast<MemRefType>(gmBase.getType());
  if (!gmBaseType ||
      gmBaseType.getElementType() != viewInfo.viewType.getElementType()) {
    op->emitOpError("requires a GM memref or i64 base address");
    return failure();
  }
  if (!gmBaseType.getMemorySpace()) {
    auto gmType =
        MemRefType::get(gmBaseType.getShape(), gmBaseType.getElementType(),
                        gmBaseType.getLayout(), gmSpace);
    gmBase = rewriter.create<memref::MemorySpaceCastOp>(loc, gmType, gmBase);
    gmBaseType = gmType;
  }
  if (gmBaseType.getMemorySpace() != gmSpace) {
    op->emitOpError("view source must be in GM");
    return failure();
  }
  return gmBase;
}

struct TensorViewWindow {
  Value gmOffset;
  Value hasElements;
  SmallVector<OpFoldResult> tileOffsets;
  SmallVector<OpFoldResult> validSizes;
  SmallVector<unsigned> transferDims;
};

// Clip a dense window for the selected sparse coordinates.
static TensorViewWindow createTensorViewWindow(
    OpBuilder &builder, Location loc, TensorViewInfo &viewInfo, Value gmBase,
    int64_t baseOffset, ValueRange indices, ArrayRef<Value> sparsePositions) {
  unsigned rank = viewInfo.tileShape.size();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<OpFoldResult> tileOffsets, validSizes;
  SmallVector<unsigned> transferDims;
  Value gmOffset =
      ShapedType::isDynamic(baseOffset)
          ? builder.create<memref::ExtractStridedMetadataOp>(loc, gmBase)
                .getOffset()
          : builder.create<arith::ConstantIndexOp>(loc, baseOffset).getResult();
  Value hasElements = builder.create<arith::ConstantIntOp>(loc, 1, 1);
  for (unsigned dim = 0; dim < rank; ++dim) {
    Value startIndex;
    int64_t tileSize;
    if (sparsePositions[dim]) {
      Value index = builder.create<memref::LoadOp>(
          loc, indices[dim], ValueRange{sparsePositions[dim]});
      startIndex = index.getType().isIndex()
                       ? index
                       : builder.create<arith::IndexCastOp>(
                             loc, builder.getIndexType(), index);
      tileSize = 1;
    } else {
      Value traversalStride = builder.create<arith::ConstantIndexOp>(
          loc, viewInfo.traversalStrides[dim]);
      startIndex =
          builder.create<arith::MulIOp>(loc, indices[dim], traversalStride);
      tileSize = viewInfo.tileShape[dim];
    }
    Value tileExtent = builder.create<arith::ConstantIndexOp>(loc, tileSize);
    Value extent = viewInfo.rootOp.getSizes()[dim];
    // Clip to the GM extent and preserve the offset within the logical tile.
    Value validStart = builder.create<arith::MaxSIOp>(loc, startIndex, zero);
    validStart = builder.create<arith::MinSIOp>(loc, validStart, extent);
    Value validEnd = builder.create<arith::AddIOp>(loc, startIndex, tileExtent);
    validEnd = builder.create<arith::MinSIOp>(loc, validEnd, extent);
    Value size = builder.create<arith::SubIOp>(loc, validEnd, validStart);
    size = builder.create<arith::MaxSIOp>(loc, size, zero);
    Value valid = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                                size, zero);
    hasElements = builder.create<arith::AndIOp>(loc, hasElements, valid);
    Value physicalOffset = builder.create<arith::MulIOp>(
        loc, validStart, viewInfo.rootOp.getStrides()[dim]);
    gmOffset = builder.create<arith::AddIOp>(loc, gmOffset, physicalOffset);
    if (sparsePositions[dim]) {
      tileOffsets.push_back(sparsePositions[dim]);
      validSizes.push_back(builder.getIndexAttr(1));
    } else {
      Value tileOffset =
          builder.create<arith::SubIOp>(loc, validStart, startIndex);
      tileOffset = builder.create<arith::MaxSIOp>(loc, tileOffset, zero);
      tileOffset = builder.create<arith::MinSIOp>(loc, tileOffset, tileExtent);
      tileOffsets.push_back(tileOffset);
      validSizes.push_back(viewInfo.tileShape[dim] == 1
                               ? OpFoldResult(builder.getIndexAttr(1))
                               : OpFoldResult(size));
      if (viewInfo.tileShape[dim] != 1)
        transferDims.push_back(dim);
    }
  }
  // Rank-reduce unit axes, keeping one dimension for scalar transfers.
  if (transferDims.empty())
    transferDims.push_back(rank - 1);
  return {gmOffset, hasElements, tileOffsets, validSizes, transferDims};
}

// Copy valid elements between the logical tile and aligned UB storage.
template <bool IsLoad>
static void copyTile(OpBuilder &builder, Location loc, Value dataBuffer,
                     Value tempBuffer, const TensorViewWindow &window) {
  unsigned rank = cast<MemRefType>(dataBuffer.getType()).getRank();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Value> tileIndices, tempIndices(rank, zero);
  for (OpFoldResult offset : window.tileOffsets)
    tileIndices.push_back(
        getValueOrCreateConstantIndexOp(builder, loc, offset));
  for (unsigned dim : window.transferDims) {
    Value upper =
        getValueOrCreateConstantIndexOp(builder, loc, window.validSizes[dim]);
    auto loop = builder.create<scf::ForOp>(loc, zero, upper, one);
    builder.setInsertionPointToStart(loop.getBody());
    Value index = loop.getInductionVar();
    tempIndices[dim] = index;
    tileIndices[dim] =
        builder.create<arith::AddIOp>(loc, tileIndices[dim], index);
  }
  if constexpr (IsLoad) {
    Value value = builder.create<memref::LoadOp>(loc, tempBuffer, tempIndices);
    builder.create<memref::StoreOp>(loc, value, dataBuffer, tileIndices);
  } else {
    Value value = builder.create<memref::LoadOp>(loc, dataBuffer, tileIndices);
    builder.create<memref::StoreOp>(loc, value, tempBuffer, tempIndices);
  }
}

// Emit the rank-reduced GM/UB transfer only for a nonempty window.
template <bool IsLoad>
static void createViewTransfer(OpBuilder &builder, Location loc,
                               TensorViewInfo &viewInfo, Value gmBase,
                               Value dataBuffer, Value tempBuffer,
                               const TensorViewWindow &window) {
  unsigned rank = viewInfo.tileShape.size();
  auto gmSpace =
      hivm::AddressSpaceAttr::get(builder.getContext(), hivm::AddressSpace::GM);
  auto tempType = cast<MemRefType>(tempBuffer.getType());
  SmallVector<OpFoldResult> transferSizes, gmStrides;
  SmallVector<int64_t> transferShape, staticStrides;
  for (unsigned dim : window.transferDims) {
    transferSizes.push_back(window.validSizes[dim]);
    transferShape.push_back(getConstantIntValue(window.validSizes[dim])
                                .value_or(ShapedType::kDynamic));
    int64_t stride = viewInfo.viewType.getStrides()[dim];
    staticStrides.push_back(stride);
    gmStrides.push_back(ShapedType::isDynamic(stride)
                            ? OpFoldResult(viewInfo.rootOp.getStrides()[dim])
                            : OpFoldResult(builder.getIndexAttr(stride)));
  }
  builder.create<scf::IfOp>(
      loc, window.hasElements, [&](OpBuilder &builder, Location nestedLoc) {
        auto layout = StridedLayoutAttr::get(
            builder.getContext(), ShapedType::kDynamic, staticStrides);
        auto gmType = MemRefType::get(
            transferShape, viewInfo.viewType.getElementType(), layout, gmSpace);
        Value gmView = builder.create<memref::ReinterpretCastOp>(
            nestedLoc, gmType, gmBase, OpFoldResult(window.gmOffset),
            transferSizes, gmStrides);
        SmallVector<OpFoldResult> tempOffsets(rank, builder.getIndexAttr(0));
        SmallVector<OpFoldResult> unitStrides(rank, builder.getIndexAttr(1));
        auto ubType =
            cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
                transferShape, tempType, tempOffsets, window.validSizes,
                unitStrides));
        Value ubView = builder.create<memref::SubViewOp>(
            nestedLoc, ubType, tempBuffer, tempOffsets, window.validSizes,
            unitStrides);
        if constexpr (IsLoad) {
          builder.create<hivm::LoadOp>(nestedLoc, TypeRange{}, gmView, ubView);
          copyTile<IsLoad>(builder, nestedLoc, dataBuffer, tempBuffer, window);
        } else {
          copyTile<IsLoad>(builder, nestedLoc, dataBuffer, tempBuffer, window);
          builder.create<hivm::StoreOp>(nestedLoc, TypeRange{}, ubView, gmView);
        }
        builder.create<scf::YieldOp>(nestedLoc);
      });
}

template <typename OpTy, bool IsLoad>
static LogicalResult convertViewAccessOp(OpTy viewOp,
                                         PatternRewriter &rewriter) {
  auto info = getTensorViewInfo(viewOp.getView());
  if (failed(info))
    return viewOp.emitOpError("expected a directly constructed encoded view");
  TensorViewInfo &viewInfo = *info;
  Value dataBuffer;
  if constexpr (IsLoad)
    dataBuffer = viewOp.getDst();
  else
    dataBuffer = viewOp.getValue();
  auto dataType = dyn_cast<MemRefType>(dataBuffer.getType());
  if (!dataType || viewOp->getNumResults() != 0)
    return viewOp.emitOpError(
        "requires buffer semantics before HIVM conversion");
  auto ubSpace =
      hivm::AddressSpaceAttr::get(viewOp.getContext(), hivm::AddressSpace::UB);
  if (dataType.getMemorySpace() && dataType.getMemorySpace() != ubSpace)
    return viewOp.emitOpError("requires GM to UB or UB to GM access");
  unsigned rank = viewInfo.tileShape.size();
  if (rank == 0)
    return viewOp.emitOpError("requires a nonempty view shape");
  Location loc = viewOp.getLoc();
  rewriter.setInsertionPoint(viewOp);
  auto gmBuffer = getOrCreateGMBuffer(viewOp, viewInfo, rewriter);
  if (failed(gmBuffer))
    return failure();
  Value gmBase = *gmBuffer;
  SmallVector<int64_t> baseStrides;
  int64_t baseOffset;
  if (failed(getStridesAndOffset(cast<MemRefType>(gmBase.getType()),
                                 baseStrides, baseOffset)))
    return viewOp.emitOpError("requires a strided GM base");
  Type elementType = dataType.getElementType();
  if (!elementType.isIntOrFloat())
    return viewOp.emitOpError("requires integer or floating-point elements");
  int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;
  // DMA windows use aligned UB storage; tile offsets are handled by local
  // copies.
  int64_t alignmentElements = std::max<int64_t>(1, 32 / elementBytes);
  SmallVector<int64_t> tempShape(viewInfo.tileShape);
  tempShape.back() = llvm::alignTo(tempShape.back(), alignmentElements);
  auto tempType = MemRefType::get(tempShape, elementType,
                                  MemRefLayoutAttrInterface{}, ubSpace);
  Value tempBuffer = rewriter.create<memref::AllocOp>(
      loc, tempType, ValueRange{}, rewriter.getI64IntegerAttr(32));
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  // Initialize the logical tile so clipped and empty windows retain padding.
  if constexpr (IsLoad) {
    Value paddingValue =
        createPaddingValue(rewriter, loc, elementType, viewInfo.paddingValue);
    OpBuilder::InsertionGuard guard(rewriter);
    SmallVector<Value> indices;
    for (unsigned dim = 0; dim < rank; ++dim) {
      Value upper =
          rewriter.create<arith::ConstantIndexOp>(loc, viewInfo.tileShape[dim]);
      auto loop = rewriter.create<scf::ForOp>(loc, zero, upper, one);
      rewriter.setInsertionPointToStart(loop.getBody());
      indices.push_back(loop.getInductionVar());
    }
    rewriter.create<memref::StoreOp>(loc, paddingValue, dataBuffer, indices);
  }

  // Sparse dimensions select coordinates independently; dense axes form a DMA
  // window.
  SmallVector<Value> sparsePositions(rank);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    for (int64_t dim : viewInfo.sparseDims) {
      Value upper =
          rewriter.create<arith::ConstantIndexOp>(loc, viewInfo.tileShape[dim]);
      auto loop = rewriter.create<scf::ForOp>(loc, zero, upper, one);
      rewriter.setInsertionPointToStart(loop.getBody());
      sparsePositions[dim] = loop.getInductionVar();
    }
    auto window =
        createTensorViewWindow(rewriter, loc, viewInfo, gmBase, baseOffset,
                               viewOp.getIndices(), sparsePositions);
    createViewTransfer<IsLoad>(rewriter, loc, viewInfo, gmBase, dataBuffer,
                               tempBuffer, window);
  }
  rewriter.eraseOp(viewOp);
  return success();
}

struct TensorViewToHIVMLoadOp : public OpRewritePattern<tv::ViewLoadOp> {
  using OpRewritePattern<tv::ViewLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tv::ViewLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    return convertViewAccessOp<tv::ViewLoadOp, true>(loadOp, rewriter);
  }
};

struct TensorViewToHIVMStoreOp : public OpRewritePattern<tv::ViewStoreOp> {
  using OpRewritePattern<tv::ViewStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tv::ViewStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    return convertViewAccessOp<tv::ViewStoreOp, false>(storeOp, rewriter);
  }
};

struct TensorViewToHIVMConversionPass
    : public impl::ConvertTensorViewToHIVMBase<TensorViewToHIVMConversionPass> {
  void runOnOperation() override;
};
} // namespace

void mlir::hivm::populateTensorViewToHIVMConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TensorViewToHIVMLoadOp, TensorViewToHIVMStoreOp>(
      patterns.getContext());
}

void TensorViewToHIVMConversionPass::runOnOperation() {
  auto module = getOperation();
  RewritePatternSet patterns(&getContext());
  hivm::populateTensorViewToHIVMConversionPatterns(patterns);
  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    return signalPassFailure();

  auto hasViewTypes = [](TypeRange types) {
    return llvm::any_of(types, [](Type type) {
      return isa<tv::PtrType, tv::TensorViewType>(type);
    });
  };
  bool hasIllegalOps = false;
  module.walk([&](Operation *op) {
    bool hasViewType = hasViewTypes(op->getOperandTypes()) ||
                       hasViewTypes(op->getResultTypes());
    if (auto funcOp = dyn_cast<func::FuncOp>(op))
      hasViewType |= hasViewTypes(funcOp.getFunctionType().getInputs()) ||
                     hasViewTypes(funcOp.getFunctionType().getResults());
    if (op->getName().getDialectNamespace() == "tv" || hasViewType) {
      op->emitError("TensorView access must be bufferized and directly "
                    "constructed before HIVM conversion");
      hasIllegalOps = true;
    }
  });
  if (hasIllegalOps)
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createTensorViewToHIVMConversionPass() {
  return std::make_unique<TensorViewToHIVMConversionPass>();
}
