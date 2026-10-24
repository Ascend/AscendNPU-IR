//===- HFusionTransformOps.cpp - Implementation of HFusion transform ops --===//
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

#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-transform-op"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::transform;
using namespace mlir::hfusion;

//===----------------------------------------------------------------------===//
// CacheReadOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
CacheReadOp::apply(TransformRewriter &rewriter,
                   TransformResults &transformResults, TransformState &state) {
  SmallVector<Operation *> cachedOps;
  for (Value target : state.getPayloadValues(getTargets())) {
    // skip values that does not have tensor types
    if (!isa<TensorType>(target.getType())) {
      continue;
    }
    hfusion::LoadOp cachedOp;
    if (auto opResult = dyn_cast_or_null<OpResult>(target)) {
      auto *definingOp = opResult.getOwner();
      rewriter.setInsertionPointAfter(definingOp);
      cachedOp = createCacheRead(rewriter, opResult, definingOp->getLoc());
    } else if (auto blockArgument = dyn_cast_or_null<BlockArgument>(target)) {
      auto *insertPoint = &(blockArgument.getParentBlock()->front());
      rewriter.setInsertionPoint(insertPoint);
      cachedOp =
          createCacheRead(rewriter, blockArgument, insertPoint->getLoc());
    } else {
      llvm::report_fatal_error("unsupported type");
    }
    cachedOps.push_back(cachedOp.getOperation());
  }
  transformResults.set(llvm::cast<OpResult>(getCached()), cachedOps);
  return DiagnosedSilenceableFailure::success();
}

void CacheReadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// CacheWriteOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
CacheWriteOp::apply(TransformRewriter &rewriter,
                    TransformResults &transformResults, TransformState &state) {
  SmallVector<Operation *> cachedOps;
  for (Value target : state.getPayloadValues(getTargets())) {
    // skip values that does not have tensor types
    if (!isa<TensorType>(target.getType())) {
      continue;
    }
    FailureOr<hfusion::StoreOp> maybeCachedOp;
    if (auto opResult = dyn_cast_or_null<OpResult>(target)) {
      CacheWriteOptions options = {
          /*outputOnly=*/getOutputOnly(),
          /*cacheWriteToOutputInit=*/getCacheWriteToOutputInit(),
          /*reshapeTrace=*/std::nullopt};
      maybeCachedOp = createCacheWrite(rewriter, opResult, options);
    } else {
      llvm::report_fatal_error("unsupported type");
    }
    if (failed(maybeCachedOp))
      return DiagnosedSilenceableFailure::definiteFailure();
    cachedOps.push_back((*maybeCachedOp).getOperation());
  }
  transformResults.set(llvm::cast<OpResult>(getCached()), cachedOps);
  return DiagnosedSilenceableFailure::success();
}

void CacheWriteOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// SetBufferSizeOp
//===----------------------------------------------------------------------===//

struct SetBufferSizeResult {
  DiagnosedSilenceableFailure diag{DiagnosedSilenceableFailure::success()};
  int64_t bufferSizeInBytes;
};

SetBufferSizeResult
calculateBufferSize(int64_t bufferSize, SetBufferSizeMode unitMode,
                    Type elementType, std::optional<Type> referenceElementType,
                    Location loc) {
  SetBufferSizeResult result;
  result.bufferSizeInBytes = bufferSize;
  // Adjust size for element mode by multiplying byte size.
  auto elementBitWidth = elementType.getIntOrFloatBitWidth();
  if (unitMode == SetBufferSizeMode::kPerElement) {
    int perElementByte = static_cast<int>(
        llvm::divideCeil(elementBitWidth, mlir::utils::kBitsToByte));
    result.bufferSizeInBytes *= perElementByte;
  }
  if (!referenceElementType.has_value())
    return result;

  // Adjust size by reference type.
  if (!(*referenceElementType).isIntOrFloat()) {
    result.diag = emitDefiniteFailure(
        loc, "reference type must be an int or float type!");
    return result;
  }
  auto referenceTypeWidth = referenceElementType->getIntOrFloatBitWidth();
  if (referenceTypeWidth > elementBitWidth) {
    result.diag = emitDefiniteFailure(
        loc, "Reference type's bit width should be less than or equal to the "
             "current element type!");
    return result;
  }
  if (referenceTypeWidth == 0) {
    llvm::report_fatal_error("Reference type's width should be positive");
    result.diag =
        emitDefiniteFailure(loc, "reference type's width should be positive");
    return result;
  }
  auto factor = elementBitWidth / referenceTypeWidth;
  if (referenceTypeWidth == 0)
    llvm::report_fatal_error("Reference type's with should be positive");
  if (elementBitWidth % referenceTypeWidth != 0)
    factor = (elementBitWidth + referenceTypeWidth - 1) / referenceTypeWidth;
  result.bufferSizeInBytes *= static_cast<int>(factor);
  return result;
}

template <typename AllocOpTy>
void setBufferSizeForAllocLikeOp(AllocOpTy op, int64_t bufferSize,
                                 transform::TransformRewriter &rewriter) {
  assert(op);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(op);
  Location loc = op->getLoc();
  MemRefType oldType = op.getType();
  // Create new alloc with static size.
  auto newType = MemRefType::get({bufferSize}, rewriter.getI8Type(),
                                 mlir::AffineMap{}, oldType.getMemorySpace());
  auto newAllocOp = rewriter.create<AllocOpTy>(loc, newType);
  // Create view from new alloc to old alloc's sizes and replace its use.
  auto startOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto viewOp = rewriter.create<memref::ViewOp>(
      loc, oldType, newAllocOp.getResult(), startOffset, op->getOperands());
  rewriter.replaceOp(op, viewOp);
}

void setBufferSizeForOpResult(Operation *op, int64_t resultNumber,
                              int64_t bufferSize,
                              transform::TransformRewriter &rewriter) {
  assert(op);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(op);
  auto mark = rewriter.create<annotation::MarkOp>(op->getLoc(),
                                                  op->getResult(resultNumber));
  mark->setAttr(utils::kBufferSizeInByteAttr,
                rewriter.getI64IntegerAttr(bufferSize));
}

DiagnosedSilenceableFailure
SetBufferSizeOp::apply(transform::TransformRewriter &rewriter,
                       transform::TransformResults &transformResults,
                       transform::TransformState &state) {
  auto staticBufferSizes = getStaticBufferSizes();
  if (getTarget().size() != staticBufferSizes.size())
    return emitDefiniteFailure(
        "Number of operands to set does not match buffer size count!");

  SetBufferSizeMode unitMode = getUnitMode();
  std::optional<Type> maybeReferenceType = getReferenceType();
  for (const auto &targetHandle : llvm::enumerate(getTarget())) {
    auto payloadOps = state.getPayloadOps(targetHandle.value());
    for (Operation *payloadOp : payloadOps) {
      auto staticBufferSize = staticBufferSizes[targetHandle.index()];
      if (staticBufferSize < 0)
        return emitDefiniteFailure("buffer size should be greater than 0!");

      for (OpResult result : payloadOp->getResults()) {
        auto maybeShapedType = dyn_cast<ShapedType>(result.getType());
        // If the op result is not a shaped type, or has static shape type, do
        // nothing.
        if (!maybeShapedType || maybeShapedType.hasStaticShape())
          continue;

        auto calculationResult = calculateBufferSize(
            staticBufferSize, unitMode,
            /*elementType=*/maybeShapedType.getElementType(),
            /*referenceElementType=*/maybeReferenceType, result.getLoc());
        if (!calculationResult.diag.succeeded())
          return std::move(calculationResult.diag);

        TypeSwitch<Operation *>(payloadOp)
            .Case<memref::AllocaOp>([&](memref::AllocaOp allocaOp) {
              setBufferSizeForAllocLikeOp(
                  allocaOp, calculationResult.bufferSizeInBytes, rewriter);
            })
            .Case<memref::AllocOp>([&](memref::AllocOp allocOp) {
              setBufferSizeForAllocLikeOp(
                  allocOp, calculationResult.bufferSizeInBytes, rewriter);
            })
            .Default([&](Operation *) {
              setBufferSizeForOpResult(payloadOp, result.getResultNumber(),
                                       calculationResult.bufferSizeInBytes,
                                       rewriter);
            });
      }
    }
  }
  return DiagnosedSilenceableFailure::success();
}

void SetBufferSizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// MultiBufferOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
MultiBufferOp::apply(transform::TransformRewriter &rewriter,
                     transform::TransformResults &transformResults,
                     transform::TransformState &state) {
  auto factor = getFactor();
  if (factor < 1) {
    emitError("factor should be >= 1.");
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  for (const auto &targetHandle : getTarget()) {
    auto payloadOps = state.getPayloadOps(targetHandle);
    for (Operation *definingOp : payloadOps) {
      assert(definingOp && "definingOp shouldn't be null.");
      if (!definingOp->getResults().empty()) {
        rewriter.setInsertionPointAfter(definingOp);
        for (auto res : definingOp->getResults()) {
          auto markOp =
              rewriter.create<annotation::MarkOp>(definingOp->getLoc(), res);
          markOp->setAttr(hfusion::MultiBufferAttr::name,
                          rewriter.getI32IntegerAttr(factor));
        }
      }
    }
  }

  return DiagnosedSilenceableFailure::success();
}

void MultiBufferOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

namespace {

static bool mayOverlap(tensor::ExtractSliceOp extract,
                       tensor::InsertSliceOp insert) {
  HyperrectangularSlice readSlice(extract);
  HyperrectangularSlice writeSlice(insert);

  FailureOr<bool> overlap = ValueBoundsConstraintSet::areOverlappingSlices(
      extract.getContext(), readSlice, writeSlice);

  // If we cannot prove they do not overlap, conservatively assume they may.
  return failed(overlap) || overlap.value();
}

struct ForwardExtract final : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extract,
                                PatternRewriter &rewriter) const override {
    Value source = extract.getSource();
    while (auto insert = source.getDefiningOp<tensor::InsertSliceOp>()) {
      if (mayOverlap(extract, insert))
        break;

      source = insert.getDest();
    }

    if (source == extract.getSource())
      return failure();

    rewriter.modifyOpInPlace(extract, [&extract, source] {
      extract.getSourceMutable().assign(source);
    });
    return success();
  }
};

static LogicalResult
getCollapsedExtractSliceInfo(OpBuilder &b, tensor::ExtractSliceOp sliceOp,
                             ArrayRef<ReassociationIndices> reassociation,
                             SmallVectorImpl<OpFoldResult> &collapsedOffsets,
                             SmallVectorImpl<OpFoldResult> &collapsedSizes,
                             SmallVectorImpl<OpFoldResult> &collapsedStrides) {
  if (!sliceOp.hasUnitStride()) {
    return failure();
  }

  SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();

  if (static_cast<size_t>(sliceOp.getResultType().getRank()) != sizes.size()) {
    return failure();
  }

  auto isZeroOffsetAndFullSize = [&sliceOp](OpFoldResult offset,
                                            OpFoldResult sliceSize,
                                            int64_t inputDim) {
    if (!isZeroIndex(offset))
      return false;
    ValueBoundsConstraintSet::Variable inputSize(sliceOp.getSource(), inputDim);
    FailureOr<bool> maybeEqual =
        ValueBoundsConstraintSet::areEqual(sliceSize, inputSize);
    return llvm::succeeded(maybeEqual) && maybeEqual.value();
  };

  // Check that the slice is contiguous within each reassociation group.
  // The slice is contiguous only if after the first dimension where a non
  // unit slice is taken, the slice size on all subsequent dimensions of the
  // group is equal to the entire size of the dimension.
  // Examples of contiguous slices:
  //   full sizes: [8, 8, 10] slice offsets: [0, 0, 0] slice sizes: [1, 1, 10]
  //   full sizes: [5, 10] slice offsets: [3, 0] slice sizes: [2, 10]
  // Examples of non contiguous slices:
  //   full sizes: [8, 8, 10] slice offsets: [0, 0, 0] slice sizes: [1, 2, 5]
  //   full sizes: [5, 10] slice offsets: [0, 4] slice sizes: [2, 5]
  for (const ReassociationIndices &indices : reassociation) {
    int64_t i = 0;
    int64_t e = indices.size();
    // Find the first expanded dim after the first dim with non-unit extracted
    // size.
    for (; i < e; ++i) {
      if (!isConstantIntValue(sizes[indices[i]], 1)) {
        // +1 to skip the first non-unit size dim.
        i++;
        break;
      }
    }

    // Verify that all subsequent dimensions extract the full size of the
    // source tensor.
    for (; i < e; ++i) {
      int64_t expandedDim = indices[i];
      if (!isZeroOffsetAndFullSize(offsets[expandedDim], sizes[expandedDim],
                                   expandedDim)) {
        return failure();
      }
    }
  }

  // The tensor.extract_slice before applying the pattern works on the result
  // of the tensor.expand_shape, so variables (i.e. inputs for ExtractSliceOp)
  // referring to the state before applying the pattern are named with the
  // prefix "expanded", and ones referring to the state after applying the
  // pattern are named with the prefix "collapsed".
  Location loc = sliceOp.getLoc();
  SmallVector<OpFoldResult> expandedOffsets = sliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> expandedSizes = sliceOp.getMixedSizes();
  SmallVector<OpFoldResult> expandedShape =
      tensor::getMixedSizes(b, loc, sliceOp.getSource());

  // Helper variables and function for accumulating the size values.
  AffineExpr d0, d1;
  bindDims(b.getContext(), d0, d1);
  // Multiply two integers.
  auto mul = [&b, loc, d0, d1](OpFoldResult v1, OpFoldResult v2) {
    auto mulMap = AffineMap::get(2, 0, {d0 * d1});
    return affine::makeComposedFoldedAffineApply(b, loc, mulMap, {v1, v2});
  };

  // Compute new offsets, sizes, and strides for tensor.extract_slice.
  // The new tensor.extract_slice will work on a tensor that has has a rank of
  // ReassociationIndices.size(). In the loop a single offset, size, and
  // stride value is computed per reassociation group.
  for (const ReassociationIndices &indices : reassociation) {
    // collapsedSize will hold the size of the single dim that represents the
    // reassociation group in the non expanded tensor.
    OpFoldResult collapsedSize = b.getIndexAttr(1);
    // The reassocGroupSizes and reassocGroupOffsets are used to create an
    // affine.linearize_index op to linearize the single offset value required
    // for this reassociation group.
    SmallVector<OpFoldResult> reassocGroupSizes, reassocGroupOffsets;

    for (long expandedDim : indices) {
      // reassocGroupSizes and reassocGroupOffsets can be obtained directly
      // from the expanded state, but the collapsed size requires calculation
      // as it did not previously exist.
      reassocGroupSizes.push_back(expandedShape[expandedDim]);
      reassocGroupOffsets.push_back(expandedOffsets[expandedDim]);
      collapsedSize = mul(collapsedSize, expandedSizes[expandedDim]);
    }

    // This MLIR revision predates affine.linearize_index. Use the equivalent
    // affine.apply-based utility for the upstream offset linearization.
    ImplicitLocOpBuilder locBuilder(loc, b);
    OpFoldResult collapsedOffset = affine::linearizeIndex(
        reassocGroupOffsets, reassocGroupSizes, locBuilder);
    collapsedOffsets.push_back(collapsedOffset);
    collapsedSizes.push_back(collapsedSize);

    // Only unit stride is supported.
    collapsedStrides.push_back(b.getIndexAttr(1));
  }
  return success();
}

/// Handle an upstream LLVM bug where affine folding creates a static collapsed
/// size while the expanded result type remains dynamic.
/// Rebuild this case with a refined expand_shape and a cast back to the
/// original result type.
static LogicalResult
rewriteDynamicityMismatch(tensor::ExtractSliceOp sliceOp,
                          tensor::ExpandShapeOp expandShapeOp, Value newSliceOp,
                          ArrayRef<OpFoldResult> expandedSizes,
                          PatternRewriter &rewriter) {
  SmallVector<OpFoldResult> refinedSizes;
  SmallVector<int64_t> refinedShape;
  for (OpFoldResult size : expandedSizes) {
    if (std::optional<int64_t> value = getConstantIntValue(size)) {
      refinedSizes.push_back(rewriter.getIndexAttr(*value));
      refinedShape.push_back(*value);
    } else {
      refinedSizes.push_back(size);
      refinedShape.push_back(ShapedType::kDynamic);
    }
  }
  auto refinedType = RankedTensorType::get(
      refinedShape, sliceOp.getResultType().getElementType(),
      sliceOp.getResultType().getEncoding());
  Location loc = sliceOp.getLoc();
  Value expanded = rewriter.create<tensor::ExpandShapeOp>(
      loc, refinedType, newSliceOp, expandShapeOp.getReassociationIndices(),
      refinedSizes);
  Value replacement = expanded;
  if (refinedType != sliceOp.getResultType())
    replacement =
        rewriter.create<tensor::CastOp>(loc, sliceOp.getResultType(), expanded);
  rewriter.replaceOp(sliceOp, replacement);
  return success();
}

/// Converts `tensor.extract_slice(tensor.expand_shape)` to
/// `tensor.expand_shape(tensor.extract_slice)`.
///
/// For this transformation to be possible, the slice must be fully contiguous
/// within each reassociation group of the expand_shape. A slice is defined as
/// fully contiguous within a reassociation group if after flattening the
/// reassociation group to a single 1D range, then the slice taken out of the
/// group could be defined as a single contiguous subrange within that range.
///
/// Rank reducing slices are not supported.
///
/// Example:
/// The transformation is possible because each reassociation group has a
/// contiguous slice (i.e., [2x4->2x4], [2x8->1x5], [4x2x4->1x1x4]).
/// ```
/// BEFORE:
/// %reshape = tensor.expand_shape %in [[0, 1], [2, 3], [4, 5, 6]]
///     tensor<8x16x32xf32> to tensor<2x4x2x8x4x2x4xf32>
/// %slice = tensor.extract_slice %reshape ...
///     tensor<2x4x2x8x4x2x4xf32> to tensor<2x4x1x5x1x1x4xf32>
///
/// AFTER:
/// %slice = tensor.extract_slice %in ...
///     tensor<8x16x32xf32> to tensor<8x5x4xf32>
/// %reshape = tensor.expand_shape %slice [[0, 1], [2, 3], [4, 5, 6]]
///     tensor<8x5x4xf32> to tensor<2x4x1x5x1x1x4xf32>
/// ```
///
/// Note - this pattern could be extended to be a swap pattern between
/// `tensor.expand_shape` and `tensor.extract_slice`, but is currently
/// implemented only as a bubble up pattern for `tensor.extract_slice`.
///
/// This pattern is copied in full from LLVM's
/// mlir/lib/Dialect/Tensor/Transforms/ReshapePatterns.cpp, with API adaptations
/// for this MLIR revision. The only deviation is the isolated handling of the
/// upstream dynamicity bug in rewriteDynamicityMismatch.
/// TODO: Move this pattern and its helpers to our LLVM fork.
struct BubbleUpExtractSliceThroughExpandShape
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto expandShapeOp =
        sliceOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandShapeOp) {
      return rewriter.notifyMatchFailure(
          sliceOp, "tensor.extract_slice source not produced by expand_shape");
    }
    SmallVector<ReassociationIndices> reassociation =
        expandShapeOp.getReassociationIndices();

    SmallVector<OpFoldResult> offsets, sizes, strides;
    if (failed(getCollapsedExtractSliceInfo(rewriter, sliceOp, reassociation,
                                            offsets, sizes, strides)))
      return failure();

    // The shape of the result can be obtained from the sizes passed in.
    SmallVector<OpFoldResult> expandedSizes = sliceOp.getMixedSizes();
    RankedTensorType resultType = sliceOp.getResultType();

    RankedTensorType collapsedType = tensor::ExtractSliceOp::inferResultType(
        expandShapeOp.getSrcType(), offsets, sizes, strides);

    // Create a new ExtractSliceOp and ExpandShapeOp.
    Location loc = sliceOp.getLoc();
    Value newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, expandShapeOp.getSrc(), offsets, sizes, strides);
    bool hasDynamicExpandedDim =
        llvm::any_of(reassociation, [&](const ReassociationIndices &indices) {
          return llvm::any_of(indices, [&](int64_t dim) {
            return resultType.isDynamicDim(dim);
          });
        });
    if (hasDynamicExpandedDim &&
        llvm::any_of(collapsedType.getShape(),
                     [](int64_t dim) { return !ShapedType::isDynamic(dim); }))
      return rewriteDynamicityMismatch(sliceOp, expandShapeOp, newSliceOp,
                                       expandedSizes, rewriter);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        sliceOp, resultType, newSliceOp,
        expandShapeOp.getReassociationIndices(), expandedSizes);
    return success();
  }
};

} // namespace

void transform::ApplyForwardExtractSlicePatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  patterns.add<ForwardExtract, BubbleUpExtractSliceThroughExpandShape>(
      patterns.getContext());
}

#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOpsEnums.cpp.inc"
#define GET_OP_CLASSES
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.cpp.inc"
