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

/// Creates an `unrealized_conversion_cast` tagged with `directionAttrName` and
/// `kTilingDimInfo`. Static offset/size are embedded in the attribute; dynamic
/// ones are appended to the cast's operand list after `input`.
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

/// Creates a downward propagator: `newValue` (sliced) is cast to `oldValue`'s
/// type so existing users can be rewired before memref ops are updated.
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

/// Rebuild a memref type for `newShape`.
/// Identity layouts stay identity (compact for the new extents, matching
/// `createSlicedAllocLike`). Explicit strided layouts keep their strides so
/// views into a larger parent buffer (e.g. GM reinterpret_cast / subview)
/// remain valid when only the viewed size is halved.
static MemRefType
cloneMemRefTypeWithNewShape(MemRefType oldType, ArrayRef<int64_t> newShape) {
  assert(static_cast<int64_t>(newShape.size()) == oldType.getRank() &&
         "rank must be preserved when cloning a sliced memref type");

  if (oldType.getLayout().isIdentity())
    return MemRefType::get(newShape, oldType.getElementType(),
                           MemRefLayoutAttrInterface{},
                           oldType.getMemorySpace());

  int64_t offset = 0;
  SmallVector<int64_t> strides;
  if (succeeded(getStridesAndOffset(oldType, strides, offset)))
    return MemRefType::get(
        newShape, oldType.getElementType(),
        StridedLayoutAttr::get(oldType.getContext(), offset, strides),
        oldType.getMemorySpace());

  return MemRefType::get(newShape, oldType.getElementType(),
                         oldType.getLayout(), oldType.getMemorySpace());
}

/// Builds a memref type with the sliced tensor shape. Compact layouts are
/// recomputed for the new extents (see `cloneMemRefTypeWithNewShape`).
MemRefType getSlicedMemRefType(MemRefType oldType, ShapedType slicedType) {
  return cloneMemRefTypeWithNewShape(oldType, slicedType.getShape());
}

/// Halves `tilingDim` in the memref shape (or marks it dynamic when the extent
/// is odd). Identity layouts stay compact; strided views keep their strides.
FailureOr<MemRefType> getSlicedMemRefType(MemRefType oldType,
                                          int64_t tilingDim) {
  if (tilingDim < 0 || tilingDim >= oldType.getRank())
    return failure();

  auto shape = llvm::to_vector(oldType.getShape());
  if (!ShapedType::isDynamic(shape[tilingDim])) {
    if (shape[tilingDim] % 2 == 0)
      shape[tilingDim] /= 2;
    else
      shape[tilingDim] = ShapedType::kDynamic;
  }

  return cloneMemRefTypeWithNewShape(oldType, shape);
}

/// If `memrefValue` traces back to a tightly-coupled alloc, tag that
/// `annotation.mark` with `kTiledTightlyCoupledAlloc` after tiling.
void markTiledTightlyCoupledAllocIfNeeded(RewriterBase &rewriter,
                                          Value memrefValue) {
  auto maybeAlloc = mlir::utils::tracebackMemRefToAlloc(memrefValue);
  if (!maybeAlloc)
    return;
  Value allocResult = maybeAlloc->getResult();
  auto allMarks = mlir::utils::getAllAnnotateOpsWithAttr(
      allocResult, hivm::HIVMTightlyCoupledBufferAttr::name);
  for (auto *mark : allMarks) {
    auto markOp = dyn_cast<annotation::MarkOp>(mark);
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
}

/// For dynamic sliced allocs, attach `kBufferSizeInByteAttr` so later passes
/// know the byte size of the odd-sized tile relative to the original buffer.
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

/// Replaces `oldAllocOp` with a smaller alloc whose shape comes from the
/// upward propagator's result type. Dynamic tile sizes become alloc operands and
/// may trigger an odd-buffer-size annotation.
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

/// After creating `newOp` (e.g. a sliced alloc), insert a downward propagator
/// on every use of `op`'s results and redirect those uses to the propagator.
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

/// Decodes tiling metadata stored on a propagator cast by
/// `createBubblePropagationCast`.
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
