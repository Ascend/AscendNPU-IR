#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/BubbleUpUtils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/TileUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::hivm::detail {

const llvm::StringLiteral kBubbleUpPropagateUp = "bubble_up_propagate_up";
const llvm::StringLiteral kBubbleUpPropagateDown = "bubble_up_propagate_down";
const llvm::StringLiteral kTilingDimInfo = "tiling_dim_info";

static UnrealizedConversionCastOp
createBubblePropagationCast(Value input, Type outputType,
                            StringRef directionAttrName, OpFoldResult offset,
                            OpFoldResult size, int64_t tilingDim,
                            PatternRewriter &rewriter) {
  SmallVector<int64_t> tilingDimInfo{tilingDim};
  SmallVector<Value> inputs{input};
  dispatchIndexOpFoldResult(offset, inputs, tilingDimInfo);
  dispatchIndexOpFoldResult(size, inputs, tilingDimInfo);
  auto castOp = rewriter.create<UnrealizedConversionCastOp>(input.getLoc(),
                                                            outputType, inputs);
  castOp->setAttr(directionAttrName, rewriter.getUnitAttr());
  auto tilingDimAttrs =
      llvm::map_to_vector(tilingDimInfo, [&](auto v) -> Attribute {
        return rewriter.getIndexAttr(v);
      });
  castOp->setAttr(kTilingDimInfo, rewriter.getArrayAttr(tilingDimAttrs));
  return castOp;
}

/// propagate_up: old (full) value -> sliced type; consumed by the new halved op
/// created one level below in the chain (e.g. new to_tensor or new_cast
/// source).
UnrealizedConversionCastOp
createBubblePropagatorUpLink(Value oldValue, Type slicedType,
                             OpFoldResult offset, OpFoldResult size,
                             int64_t tilingDim, PatternRewriter &rewriter) {
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(oldValue);
  auto propagateOp =
      createBubblePropagationCast(oldValue, slicedType, kBubbleUpPropagateUp,
                                  offset, size, tilingDim, rewriter);
  return propagateOp;
}

UnrealizedConversionCastOp
createBubblePropagatorDown(Value oldValue, Value newValue, OpFoldResult offset,
                           OpFoldResult size, int64_t tilingDim,
                           PatternRewriter &rewriter) {
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(newValue);
  auto propagateOp = createBubblePropagationCast(newValue, oldValue.getType(),
                                                 kBubbleUpPropagateDown, offset,
                                                 size, tilingDim, rewriter);
  return propagateOp;
}

MemRefType getSlicedMemRefType(MemRefType oldType, ShapedType slicedType) {
  return MemRefType::get(slicedType.getShape(), slicedType.getElementType(),
                         oldType.getLayout(), oldType.getMemorySpace());
}

FailureOr<MemRefType> getSlicedMemRefType(MemRefType oldType,
                                          int64_t tilingDim) {
  auto shape = llvm::to_vector(oldType.getShape());
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(getStridesAndOffset(oldType, strides, offset)))
    return failure();

  auto layout = StridedLayoutAttr::get(oldType.getContext(),
                                       ShapedType::kDynamic, strides);
  if (!ShapedType::isDynamic(shape[tilingDim])) {
    if (shape[tilingDim] % 2 == 0)
      shape[tilingDim] /= 2;
    else
      shape[tilingDim] = ShapedType::kDynamic;
  }

  return MemRefType::get(shape, oldType.getElementType(), layout,
                         oldType.getMemorySpace());
}

void markTiledTightlyCoupledAllocIfNeeded(RewriterBase &rewriter,
                                          Value memrefValue) {
  auto maybeAlloc = mlir::utils::tracebackMemRefToAlloc(memrefValue);
  if (!maybeAlloc)
    return;
  Value allocResult = maybeAlloc->getResult();
  auto maybeMark = mlir::utils::getAnnotateOpWithAttr(
      allocResult, hivm::HIVMTightlyCoupledBufferAttr::name);
  if (!maybeMark.has_value())
    return;
  auto markOp = dyn_cast<annotation::MarkOp>(maybeMark.value());
  if (!markOp)
    return;
  auto attr = markOp->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
      hivm::HIVMTightlyCoupledBufferAttr::name);
  if (!attr || !attr.getId().has_value())
    return;
  rewriter.modifyOpInPlace(markOp, [&]() {
    markOp->setAttr(kTiledTightlyCoupledAlloc,
                    UnitAttr::get(rewriter.getContext()));
  });
}

static LogicalResult
markOddTilingBufferSizeIfNeeded(MemRefType slicedType, MemRefType sourceType,
                                Value buffer, int64_t tilingDim,
                                PatternRewriter &rewriter) {
  auto bufferType = dyn_cast<ShapedType>(buffer.getType());
  if (!bufferType || bufferType.hasStaticShape())
    return success();

  auto bufferSize =
      calculateBufferSizeInBytes(slicedType, sourceType.getShape(), tilingDim);

  auto newMarkOp = rewriter.create<annotation::MarkOp>(buffer.getLoc(), buffer);
  newMarkOp->setAttr(kBufferSizeInByteAttr,
                     rewriter.getI64IntegerAttr(bufferSize));
  return success();
}

FailureOr<memref::AllocOp>
createSlicedAllocLike(UnrealizedConversionCastOp propagateOp,
                      memref::AllocOp oldAllocOp, PatternRewriter &rewriter) {
  auto originalType = dyn_cast<MemRefType>(oldAllocOp.getResult().getType());
  if (!originalType)
    return failure();

  auto slicedMemRefType =
      dyn_cast<MemRefType>(propagateOp.getResult(0).getType());
  if (!slicedMemRefType)
    return failure();

  auto memrefType = MemRefType::get(
      slicedMemRefType.getShape(), slicedMemRefType.getElementType(),
      originalType.getLayout(), originalType.getMemorySpace());

  auto [tilingDim, tiledOffset, tiledSize] = getTilingDimInfo(propagateOp);
  SmallVector<Value> dynamicSizes;
  if (ShapedType::isDynamic(
          getConstantIntValue(tiledSize).value_or(ShapedType::kDynamic)))
    dynamicSizes.push_back(cast<Value>(tiledSize));

  rewriter.setInsertionPoint(oldAllocOp);
  auto newAllocOp = rewriter.create<memref::AllocOp>(oldAllocOp.getLoc(),
                                                     memrefType, dynamicSizes);
  if (!dynamicSizes.empty() &&
      failed(markOddTilingBufferSizeIfNeeded(slicedMemRefType, originalType,
                                             newAllocOp.getResult(), tilingDim,
                                             rewriter)))
    return failure();
  return newAllocOp;
}

void insertDownPropagators(Operation *op, Operation *newOp, OpFoldResult offset,
                           OpFoldResult size, int64_t tilingDim,
                           PatternRewriter &rewriter) {

  for (auto [res, newRes] :
       llvm::zip_equal(op->getResults(), newOp->getResults())) {
    SmallVector<OpOperand *> uses;
    for (auto &use : res.getUses())
      uses.push_back(&use);
    for (auto *use : uses) {
      auto *user = use->getOwner();
      auto newPropagateOp = createBubblePropagatorDown(
          use->get(), newRes, offset, size, tilingDim, rewriter);
      rewriter.modifyOpInPlace(
          user, [&]() { use->set(newPropagateOp->getResult(0)); });
    }
  }
}

TilingDimInfo getTilingDimInfo(UnrealizedConversionCastOp propagateOp) {
  auto inputIter = ++propagateOp.getInputs().begin();
  auto tilingDimInfoAttr =
      propagateOp->getAttrOfType<ArrayAttr>(kTilingDimInfo);
  if (!tilingDimInfoAttr)
    return TilingDimInfo{};
  if (tilingDimInfoAttr.size() != 3)
    return TilingDimInfo{};
  TilingDimInfo tilingDimInfo;
  tilingDimInfo.tilingDim = cast<IntegerAttr>(tilingDimInfoAttr[0]).getInt();
  tilingDimInfo.offset = tilingDimInfoAttr[1];
  if (ShapedType::isDynamic(cast<IntegerAttr>(tilingDimInfoAttr[1]).getInt())) {
    tilingDimInfo.offset = *inputIter;
    ++inputIter;
  }
  tilingDimInfo.size = tilingDimInfoAttr[2];
  if (ShapedType::isDynamic(cast<IntegerAttr>(tilingDimInfoAttr[2]).getInt())) {
    tilingDimInfo.size = *inputIter;
    ++inputIter;
  }
  return tilingDimInfo;
}

} // namespace mlir::hivm::detail
