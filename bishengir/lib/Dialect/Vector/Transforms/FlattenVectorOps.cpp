//===- FlattenVectorOps.cpp ---- Flatten Vector Ops -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Transforms.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/Vector/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "flatten-vector-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::utils::debugger;

namespace {

// This is required because flatten-vector will be called after infer hivm
// memscope
Type setUBMemScopeAttr(Type type, MLIRContext *ctx) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType) {
    return type;
  }
  auto memSpace = memrefType.getMemorySpace();
  if (memSpace) {
    return type;
  }
  auto ubSpaceAttr = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
  auto newMemRefType =
      MemRefType::Builder(memrefType).setMemorySpace(ubSpaceAttr);
  return newMemRefType;
}

void setUBMemScopeAttr(Value value, MLIRContext *ctx) {
  Type newType = setUBMemScopeAttr(value.getType(), ctx);
  value.setType(newType);
}

Value flattenVector(Value vector, VectorType vecType, Location loc,
                    PatternRewriter &rewriter) {
  if (vecType.getRank() == 1) {
    return vector;
  }
  VectorType flattenVecType = VectorType::get(
      SmallVector<int64_t>{vecType.getNumElements()}, vecType.getElementType());
  return rewriter.create<vector::ShapeCastOp>(loc, flattenVecType, vector);
}

// Unique slice means:
// for rootShape `AxBxCxD`, resultShape `1x1xcxD`, where `c < C`.
// the unique slice dim is `c`, with higher dims being all `1`, and lower dims
// all full loaded
int64_t getUniqSliceDim(ArrayRef<int64_t> rootShape,
                        ArrayRef<int64_t> resultShape) {
  SmallVector<int64_t> alignShape(rootShape.size(), 1);
  int64_t rootRank = rootShape.size();
  int64_t resultRank = resultShape.size();
  int64_t diff = rootRank - resultRank;
  for (int64_t i = diff; i < rootRank; ++i) {
    alignShape[i] = resultShape[i - diff];
  }
  if (llvm::any_of(alignShape, [](int64_t s) { return s < 1; })) {
    return -1;
  }

  if (llvm::all_of(alignShape, [](int64_t s) { return s == 1; })) {
    return rootRank - 1;
  }

  int64_t sliceDim = 0;
  for (int64_t i = 0; i < rootRank; ++i) {
    if (alignShape[i] > 1) {
      sliceDim = i;
      break;
    }
  }

  for (int64_t i = sliceDim + 1; i < rootRank; ++i) {
    if (alignShape[i] != rootShape[i]) {
      return -1;
    }
  }
  return sliceDim;
}

std::optional<Value> collapseAllDims(Value value, OpBuilder &builder,
                                     Location loc) {
  auto inputType = dyn_cast<MemRefType>(value.getType());
  if (!inputType) {
    return value;
  }
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();
  if (rank <= 1) {
    return value;
  }

  int64_t collapsedDimSize = 1;
  bool isStatic = true;
  for (int64_t dim : inputShape) {
    if (dim == ShapedType::kDynamic) {
      collapsedDimSize = ShapedType::kDynamic;
      isStatic = false;
      break;
    }
    collapsedDimSize *= dim;
  }
  if (!isStatic) {
    return std::nullopt;
  }

  MLIRContext *ctx = builder.getContext();
  auto resultType =
      mlir::MemRefType::get({collapsedDimSize}, inputType.getElementType());
  Type typeWithAttr = setUBMemScopeAttr(resultType, ctx);

  SmallVector<ReassociationIndices> reassociations = {
      llvm::to_vector(llvm::seq(0L, rank))};
  Value collapse = builder.create<memref::CollapseShapeOp>(
      loc, typeWithAttr, value, reassociations);
  setUBMemScopeAttr(collapse, ctx);
  return collapse;
}

std::optional<memref::SubViewOp> traceFirstNonRankReduceSubview(Value value) {
  auto subviewOp = value.getDefiningOp<memref::SubViewOp>();
  if (!subviewOp) {
    return std::nullopt;
  }
  Value src = subviewOp.getSource();
  Value dst = subviewOp.getResult();
  ShapedType srcShape = llvm::cast<ShapedType>(src.getType());
  ShapedType dstShape = llvm::cast<ShapedType>(dst.getType());

  if (srcShape.getRank() == dstShape.getRank()) {
    return subviewOp;
  }

  // check if the rank-reduce slice only slice size 1
  ArrayRef<int64_t> srcSizes = srcShape.getShape();
  ArrayRef<int64_t> sliceSizes = subviewOp.getStaticSizes();
  for (auto [srcSize, sliceSize] : llvm::zip(srcSizes, sliceSizes)) {
    if (srcSize != sliceSize && sliceSize != 1) {
      LDBG("strange rank reduce subview: " << value);
      return std::nullopt;
    }
  }
  return traceFirstNonRankReduceSubview(src);
}

Value computeCollapsedSubview(memref::SubViewOp subview, Value collapsedRoot,
                              PatternRewriter &rewriter, Location loc) {
  MemRefType srcType = subview.getSourceType();
  ArrayRef<int64_t> srcShape = srcType.getShape();
  SmallVector<OpFoldResult> sizes = subview.getMixedSizes();
  SmallVector<OpFoldResult> offsets = subview.getMixedOffsets();
  int64_t rank = srcType.getRank();

  Value accumSize = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value collapsedSize = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value collapsedOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  for (int64_t i = rank - 1; i >= 0; --i) {
    Value dimSliceOffset =
        getValueOrCreateConstantIndexOp(rewriter, loc, offsets[i]);
    Value dimSliceSize =
        getValueOrCreateConstantIndexOp(rewriter, loc, sizes[i]);
    Value dimSrcSize =
        rewriter.create<arith::ConstantIndexOp>(loc, srcShape[i]);

    Value curOffset =
        rewriter.create<arith::MulIOp>(loc, accumSize, dimSliceOffset);
    accumSize = rewriter.create<arith::MulIOp>(loc, accumSize, dimSrcSize);
    collapsedOffset =
        rewriter.create<arith::AddIOp>(loc, collapsedOffset, curOffset);
    collapsedSize =
        rewriter.create<arith::MulIOp>(loc, collapsedSize, dimSliceSize);
  }

  SmallVector<OpFoldResult> collapsedSizes = {OpFoldResult(collapsedSize)};
  SmallVector<OpFoldResult> collapsedOffsets = {OpFoldResult(collapsedOffset)};
  SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
  Value newSubview = rewriter.create<memref::SubViewOp>(
      loc, collapsedRoot, collapsedOffsets, collapsedSizes, strides);
  MLIRContext *ctx = rewriter.getContext();
  setUBMemScopeAttr(collapsedRoot, ctx);
  setUBMemScopeAttr(newSubview, ctx);
  return newSubview;
}

// Check if collapsible given assigned reassociations.
// Leading one size dims have individual reassociation indice group.
// Other dims are contained in one reassociation indices group.
// For example:
// 1. 1x1x4x16 -> [[0], [1], [2, 3]]
// 2. 4x1x4x16 -> [[0, 1, 2, 3]]
bool isCollapsible(memref::SubViewOp subview) {
  MemRefType resType = llvm::cast<MemRefType>(subview.getResult().getType());
  ArrayRef<int64_t> shape = resType.getShape();
  int64_t rank = resType.getRank();
  SmallVector<ReassociationIndices> reassociations;
  reassociations.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    if (shape[i] == 1) {
      ReassociationIndices curReasso = {i};
      reassociations.push_back(curReasso);
    } else {
      ReassociationIndices remainReasso = llvm::to_vector(llvm::seq(i, rank));
      reassociations.push_back(remainReasso);
      break;
    }
  }

  bool collapsible =
      memref::CollapseShapeOp::isGuaranteedCollapsible(resType, reassociations);
  LDBG("check collapsible for subview: " << subview);
  LDBG("reassociations: " << utils::debugger::to_string(reassociations));
  LDBG("is collapsible: " << collapsible);
  return collapsible;
}

// Try to flatten contiguous subview into one-dim.
// Returns nullopt if failed.
std::optional<Value> flattenSubview(Value value, PatternRewriter &rewriter,
                                    Location loc) {
  if (isa<BlockArgument>(value)) {
    return collapseAllDims(value, rewriter, loc);
  }
  auto defOp = value.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }
  auto subviewMaybe = traceFirstNonRankReduceSubview(value);
  if (!subviewMaybe.has_value()) {
    return std::nullopt;
  }
  memref::SubViewOp subview = subviewMaybe.value();
  if (!isCollapsible(subview)) {
    return std::nullopt;
  }

  Value root = subview.getSource();
  ShapedType rootType = llvm::cast<ShapedType>(root.getType());
  ShapedType resultType = llvm::cast<ShapedType>(value.getType());
  if (!rootType.hasStaticShape()) {
    return std::nullopt;
  }

  int64_t sliceDim =
      getUniqSliceDim(rootType.getShape(), resultType.getShape());
  if (sliceDim == -1) {
    LDBG("no uniq slice dim found for root: " << root);
    return std::nullopt;
  }

  std::optional<Value> collapsedRoot = collapseAllDims(root, rewriter, loc);
  if (!collapsedRoot.has_value()) {
    LDBG("fail to collapse all dims for root: " << root);
    return std::nullopt;
  }

  return computeCollapsedSubview(subview, collapsedRoot.value(), rewriter, loc);
}

int64_t getNonUnitDimNum(Value value) {
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (!shapedType) {
    return 0;
  }
  ArrayRef<int64_t> shape = shapedType.getShape();
  int64_t num = 0;
  for (int64_t s : shape) {
    if (s > 1) {
      num++;
    }
  }
  return num;
}

/// Flatten a contiguous mask (constant_mask) to 1D for use with flattened
/// transfer_write. Returns nullopt if the mask is not contiguous (e.g. not a
/// constant_mask). Constant_mask produces a hyper-rectangular region [0,d0) x
/// [0,d1) x ... which is always contiguous in row-major flattened layout.
/// Returns optional(null Value) when original has no mask (caller should omit
/// mask in new op).
static std::optional<Value> flattenContiguousMask(Value mask,
                                                  int64_t flattenedVectorSize,
                                                  Location loc,
                                                  PatternRewriter &rewriter) {
  if (!mask) {
    return Value();
  }
  auto constMaskOp = mask.getDefiningOp<vector::ConstantMaskOp>();
  if (!constMaskOp) {
    return std::nullopt;
  }
  ArrayRef<Attribute> maskDimSizes = constMaskOp.getMaskDimSizes().getValue();
  int64_t product = 1;
  for (Attribute attr : maskDimSizes) {
    int64_t dim = cast<IntegerAttr>(attr).getInt();
    if (dim < 0) {
      return std::nullopt;
    }
    product *= dim;
  }
  if (product > flattenedVectorSize) {
    // Invalid mask: mask dim size product is greater than flattened vector size
    return std::nullopt;
  }
  auto flatMaskType =
      VectorType::get({flattenedVectorSize}, rewriter.getI1Type());
  return rewriter
      .create<vector::ConstantMaskOp>(
          loc, flatMaskType,
          rewriter.getArrayAttr(rewriter.getI64IntegerAttr(product)))
      .getResult();
}

// Flatten transfer_write with multiple non-unit dimensions, only works when the
// write source is contiguous memref.
//
// This pattern does flatten by creating a new collapse_shape op that collapse
// write source into one dim, and creating a shape_cast that cast from its
// operand. The shape_cast will be bubbled up in other patterns and should
// eventually be optimized.
//
// e.g. before:
//   %src = ... : memref<4x4x4x16xf16>
//   %view = subview %src [1, 1, 4, 16] ... : memref<1x1x4x16xf16>
//   %read = ... : vector<1x1x4x16xf16>
//   %write = transfer_write %read, %view: vector<1x1x4x16xf16>,
//                                         memref<1x1x4x16xf16>
// after:
//   %collapse = collapse_shape %src [[0, 1, 2, 3]] : memref<1024xf16>
//   %view = subview ... [64] ... : memref<64xf16>
//   %read = ... : vector<1x1x4x16xf16>
//   %cast = shape_cast %read: vector<1x1x4x16xf16> -> vector<64xf16>
//   %write = transfer_write %cast, %view: vector<64xf16>, memref<64xf16>
struct FlattenContiguousTransferWrite
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {

    Value vector = writeOp.getVector();
    Value source = writeOp.getSource();
    Location loc = writeOp->getLoc();
    if (getNonUnitDimNum(vector) <= 1 || getNonUnitDimNum(source) <= 1) {
      // flatten should only work when has more than one non-unit dims,
      // to avoid conflict with other patterns, like drop unit dims.
      return rewriter.notifyMatchFailure(writeOp, "no need to flatten");
    }

    auto newSubviewMaybe = flattenSubview(source, rewriter, loc);
    if (!newSubviewMaybe.has_value()) {
      return rewriter.notifyMatchFailure(writeOp, "fail to flatten subview");
    }
    Value newSubview = newSubviewMaybe.value();
    VectorType flattenedVecType =
        VectorType::get({writeOp.getVectorType().getNumElements()},
                        writeOp.getVectorType().getElementType());
    Value newVector =
        flattenVector(vector, writeOp.getVectorType(), loc, rewriter);

    // Only support contiguous masks (constant_mask). Non-contiguous masks
    // cannot be correctly flattened, so bail out.
    auto flatMaskMaybe = flattenContiguousMask(
        writeOp.getMask(), flattenedVecType.getNumElements(), loc, rewriter);
    if (!flatMaskMaybe.has_value()) {
      return rewriter.notifyMatchFailure(
          writeOp, "mask is not contiguous, skip flatten");
    }

    Value cst0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(1, cst0);
    AffineMap permMap = getTransferMinorIdentityMap(
        llvm::cast<ShapedType>(newSubview.getType()), flattenedVecType);
    // Preserve in_bounds: flattened 1D transfer uses [true] when all original
    // dimensions were in bounds.
    ArrayAttr inBoundsAttr = writeOp.getInBoundsAttr();
    bool allInBounds =
        !inBoundsAttr || llvm::all_of(inBoundsAttr.getAsRange<BoolAttr>(),
                                      [](BoolAttr b) { return b.getValue(); });
    auto newInBoundsAttr =
        rewriter.getBoolArrayAttr(SmallVector<bool>(1, allInBounds));

    auto newWrite = rewriter.create<vector::TransferWriteOp>(
        loc, newVector, newSubview, indices, AffineMapAttr::get(permMap),
        flatMaskMaybe.value(), newInBoundsAttr);
    rewriter.replaceOp(writeOp, newWrite);
    return success();
  }
};

// Flatten vector.gather when result (and thus index_vec, mask, pass_thru) has
// rank > 1 with shape 1x1x...x1xn.
//
// before:
// vector.gather %base[%c0][%idx], %mask, %pass_thru
//        : memref<320xi32>, vector<1x64xi32>, vector<1x64xi1>, vector<1x64xi32>
//        into vector<1x64xi32>
// after:
// vector.gather %base[%c0][%flat_idx], %flat_mask, %flat_pass_thru
//        : memref<320xi32>, vector<64xi32>, vector<64xi1>, vector<64xi32>
//        into vector<64xi32>
struct FlattenVectorGatherOp : public OpRewritePattern<vector::GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    VectorType resType = gatherOp.getVectorType();
    if (resType.getRank() <= 1) {
      return rewriter.notifyMatchFailure(gatherOp, "no need to flatten");
    }
    if (!isLeadingOnesShape(resType.getShape())) {
      return rewriter.notifyMatchFailure(
          gatherOp, "only shape with leading ones is supported for flatten");
    }
    Location loc = gatherOp.getLoc();
    int64_t numElements = resType.getNumElements();
    VectorType flatResType =
        VectorType::get({numElements}, resType.getElementType());

    Value flatIndexVec = flattenVector(
        gatherOp.getIndexVec(), gatherOp.getIndexVectorType(), loc, rewriter);
    Value flatMask = flattenVector(gatherOp.getMask(),
                                   gatherOp.getMaskVectorType(), loc, rewriter);
    Value flatPassThru =
        flattenVector(gatherOp.getPassThru(), gatherOp.getPassThruVectorType(),
                      loc, rewriter);

    auto newGather = rewriter.create<vector::GatherOp>(
        loc, flatResType, gatherOp.getBase(), gatherOp.getIndices(),
        flatIndexVec, flatMask, flatPassThru);
    // Cast back to original shape so existing uses (e.g. transfer_write to
    // memref<1x64xi32>) remain valid.
    Value castBack = rewriter.create<vector::ShapeCastOp>(
        loc, resType, newGather.getResult());
    rewriter.replaceOp(gatherOp, castBack);
    return success();
  }

private:
  bool isLeadingOnesShape(ArrayRef<int64_t> shape) const {
    if (shape.size() <= 1)
      return false;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      if (shape[i] != 1)
        return false;
    }
    return true;
  }
};

// Flatten transfer_read with shape_cast user that reduce rank.
// This pattern does flatten by update the subview to eliminate size-1 dims.
// This pattern works with a higher priority to avoid optimize transfer_read
// into gather in other patterns.
//
// before:
//   %view = subview ... [1, 1, 4, 16] ... : memref<1x1x4x16xf16>
//   %read = transfer_read %view: memref<1x1x4x16xf16> -> vector<4x1x1x16xf16>
//   %cast = shape_cast %read: vector<4x1x1x16xf16> -> vector<4x16xf16>
//   user(%cast)
//
// after:
//   %view = subview ... [1, 1, 4, 16] ... : memref<4x16xf16>
//   %read = transfer_read %view: memref<4x16xf16> -> vector<4x16xf16>
//   user(%read)
struct FlattenTransferReadWithRankReduceShapeCast
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    if (hasConstantPermutationResult(readOp.getPermutationMap())) {
      return rewriter.notifyMatchFailure(
          readOp,
          "reduce rank may be unsafe if has constant permutation result");
    }

    Operation *user = getUniqueUser(readOp.getResult());
    if (!user) {
      return failure();
    }

    auto castUser = dyn_cast<vector::ShapeCastOp>(user);
    if (!castUser) {
      return failure();
    }

    if (!isRankReduceShapeCast(castUser)) {
      return rewriter.notifyMatchFailure(
          readOp, "user should be shape cast for unit dims");
    }

    if (isPrecedingUnitDimShapeCast(castUser)) {
      // make sure this pattern does not affect drop unit dims patterns.
      // this pattern only handles cases shape cast like:
      // - [4, 1, 1, 16] -> [4, 16]
      // not handle shape cast like:
      // - [1, 1, 4, 16] -> [4, 16]
      return rewriter.notifyMatchFailure(
          readOp,
          "user should not be shape cast with unit dims all preceding dims");
    }

    VectorType castType = castUser.getResultVectorType();

    Value source = readOp.getSource();
    auto oldSubview = source.getDefiningOp<memref::SubViewOp>();
    if (!oldSubview) {
      // TODO: relax this constraint by supporting read from arg
      return rewriter.notifyMatchFailure(readOp, "source should be subview");
    }

    auto newSubviewMaybe =
        createRankReducedSubview(rewriter, oldSubview, castUser);
    if (!newSubviewMaybe.has_value()) {
      return rewriter.notifyMatchFailure(readOp,
                                         "fail to createRankReducedSubview");
    }
    Value newSubview = newSubviewMaybe.value();

    Location loc = readOp.getLoc();
    MemRefType newType = llvm::cast<MemRefType>(newSubview.getType());
    Value cst0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(newType.getRank(), cst0);
    auto newRead = rewriter.create<vector::TransferReadOp>(loc, castType,
                                                           newSubview, indices);
    rewriter.replaceOp(user, newRead);
    return success();
  }

private:
  Operation *getUniqueUser(Value value) const {
    if (!value.hasOneUse()) {
      return nullptr;
    }
    return *value.user_begin();
  }

  bool hasConstantPermutationResult(AffineMap map) const {
    return llvm::any_of(map.getResults(), [](AffineExpr expr) {
      return expr.isSymbolicOrConstant();
    });
  }

  bool isPrecedingUnitDimShapeCast(vector::ShapeCastOp cast) const {
    auto sourceType = dyn_cast<VectorType>(cast.getSource().getType());
    auto resultType = dyn_cast<VectorType>(cast.getResult().getType());

    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    // 1. Identify leading unit dimensions in the source
    size_t firstNonUnitIdx = 0;
    while (firstNonUnitIdx < sourceShape.size() &&
           sourceShape[firstNonUnitIdx] == 1) {
      firstNonUnitIdx++;
    }

    // If there were no leading unit dimensions, it doesn't match
    if (firstNonUnitIdx == 0)
      return false;

    // 2. Extract the "right" part of the input shape (non-unit dims)
    ArrayRef<int64_t> trailingSourceDims =
        sourceShape.drop_front(firstNonUnitIdx);

    // 3. Check if any unit dimensions remain in the trailing part
    // (Ensures non-unit dims are strictly on the right)
    for (int64_t dim : trailingSourceDims) {
      if (dim == 1)
        return false;
    }

    // 4. Verify the output shape matches exactly the dropped-prefix version
    // and that all unit dims were actually dropped (result shouldn't have them)
    return trailingSourceDims == resultShape;
  }

  // Create a new rank reduce subview that eliminates size 1 dims.
  // Returns nullopt if not fail to create.
  std::optional<Value> createRankReducedSubview(OpBuilder &builder,
                                                memref::SubViewOp oldSubview,
                                                ShapeCastOp castOp) const {
    ArrayRef<int64_t> srcShape = castOp.getSourceVectorType().getShape();
    ArrayRef<int64_t> dstShape = castOp.getResultVectorType().getShape();
    // check if src can be rank reduced to dst by eliminate size 1 dims
    auto mask = computeRankReductionMask(srcShape, dstShape);
    if (!mask.has_value()) {
      return std::nullopt;
    }

    SmallVector<int64_t> newShape;
    auto oldShape = oldSubview.getType().getShape();
    for (int64_t s : oldShape) {
      if (s != 1) {
        newShape.push_back(s);
      }
    }

    auto sourceMemRef = oldSubview.getSource();
    auto sourceTy = llvm::cast<MemRefType>(sourceMemRef.getType());

    // automatically infer the complicated strides
    Type resultTy = memref::SubViewOp::inferRankReducedResultType(
        newShape, sourceTy, oldSubview.getMixedOffsets(),
        oldSubview.getMixedSizes(), oldSubview.getMixedStrides());

    // use origin offsets/sizes/strides because the reduced rank are decided by
    // resultTy
    return builder.create<memref::SubViewOp>(
        oldSubview.getLoc(), llvm::cast<MemRefType>(resultTy), sourceMemRef,
        oldSubview.getMixedOffsets(), oldSubview.getMixedSizes(),
        oldSubview.getMixedStrides());
  }

  // check if ShapeCastOp op has RankReduce pattern, i.e. reduce size-1 dims
  // these are rank reduce casts:
  // - [1, 64] -> [64]
  // - [1, 64, 1] -> [64]
  // these are not:
  // - [2, 32] -> [64]
  // - [2, 2, 32] -> [4, 32]
  bool isRankReduceShapeCast(Operation *op) const {
    auto shapeCast = dyn_cast<vector::ShapeCastOp>(op);
    if (!shapeCast) {
      return false;
    }

    auto sourceTy = shapeCast.getSourceVectorType();
    auto resultTy = shapeCast.getResultVectorType();
    auto sourceShape = sourceTy.getShape();
    auto resultShape = resultTy.getShape();

    DenseSet<int64_t> unitDims;
    size_t resultIdx = 0;
    for (int64_t i = 0; i < sourceTy.getRank(); ++i) {
      if (resultIdx < resultShape.size() &&
          sourceShape[i] == resultShape[resultIdx]) {
        resultIdx++;
      } else if (sourceShape[i] == 1) {
        unitDims.insert(i);
      } else {
        return false;
      }
    }
    return !unitDims.empty() && resultIdx == resultShape.size();
  }
};

// normalize ineffective transpose by replace transpose with shape_cast.
// Effectiveness is determined by checking the relative order of the non-unit
// dimensions.
//
// example 1:
// before:
// - vector.transpose %1 [1, 0, 2] : vector<1x4x16xf16> to vector<4x1x16xf16>
// after:
// - vector.shape_cast %1 : vector<1x4x16xf16> to vector<4x1x16xf16>
//
// example 2:
// before:
// - vector.transpose %1 [1, 0] : vector<4x16xf16> to vector<16x4xf16>
// after:
// - not optimize, because the transpose is effective
struct NormalizeIneffectiveTranspose
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    VectorType srcType = op.getSourceVectorType();
    VectorType dstType = op.getResultVectorType();

    // Safety check: ShapeCast handles scalable vectors differently,
    // strictly speaking we usually only optimize fixed-width unit dims this
    // way.
    if (srcType.isScalable() || dstType.isScalable()) {
      return failure();
    }

    ArrayRef<int64_t> srcShape = srcType.getShape();
    ArrayRef<int64_t> permutation = op.getPermutation();

    // Logic: Identify the indices of non-unit dimensions as they appear
    // in the OUTPUT (after permutation).
    SmallVector<int64_t> nonUnitIndices;
    nonUnitIndices.reserve(srcShape.size());

    for (int64_t srcIndex : permutation) {
      if (srcShape[srcIndex] != 1) {
        nonUnitIndices.push_back(srcIndex);
      }
    }

    // If the non-unit dimensions are strictly increasing, it means
    // the relative memory layout order hasn't changed.
    // E.g., if we extract non-unit dims and get [1, 2], it's sorted.
    // If we get [2, 1], data is actually being swapped.
    bool isEffectiveTranspose = false;
    for (size_t i = 1; i < nonUnitIndices.size(); ++i) {
      if (nonUnitIndices[i] < nonUnitIndices[i - 1]) {
        isEffectiveTranspose = true;
        break;
      }
    }

    // If it is an effective transpose, we cannot optimize it to a shape_cast.
    if (isEffectiveTranspose) {
      return failure();
    }

    // If logic reaches here, the transpose is just permuting unit dimensions
    // or moving unit dimensions around non-unit ones without reordering data.
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, dstType,
                                                     op.getVector());

    return success();
  }
};

// Enum class for dispatching the creation logic for different arith/math op
// kind.
enum class ComputationOpKind { Unary, Binary, Ternary };

// Base class for flattening computation op, with a call back to for creating
// flattened computation op with different arith/math op kind.
// We use propagate shape cast to flatten each computation op.
//
// before:
// %0 = vector.transfer_read : vector<4x16xf32>
// %1 = vector.transfer_read : vector<4x16xf32>
// %2 = arith.addf %0, %1 : vector<4x16xf32>
// %3 = vector.shape_cast %2 : vector<4x16xf32> to vector<64xf32>
// %4 = use(%3)
//
// after:
// %0 = vector.transfer_read : vector<4x16xf32>
// %cast0 = vector.shape_cast %0 : vector<4x16xf32> to vector<64xf32>
// %1 = vector.transfer_read : vector<4x16xf32>
// %cast1 = vector.shape_cast %1 : vector<4x16xf32> to vector<64xf32>
// %2 = arith.addf %cast0, %cast1 : vector<64xf32>
// %3 = use(%2)
template <typename OpType, ComputationOpKind OpKind>
class FlattenComputationOp : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const final {
    Value result = op->getResult(0);
    auto resType = dyn_cast<VectorType>(result.getType());
    if (!resType) {
      return failure();
    }

    int64_t nonUnitDims = getNonUnitDimNum(result);
    if (nonUnitDims <= 1) {
      return rewriter.notifyMatchFailure(op, "no need to flatten");
    }

    Location loc = op.getLoc();
    int64_t totalElements = resType.getNumElements();

    SmallVector<Value> operands =
        flattenVectorOperands(op, totalElements, rewriter);
    Type dstElemType = resType.getElementType();
    VectorType dstType = VectorType::get({totalElements}, dstElemType);
    // preserve attrs
    auto attrs = op->getAttrs();
    Operation *newOp =
        createNewComputationOp(op, dstType, operands, loc, rewriter);
    newOp->setAttrs(attrs);

    Value newResult = flattenVectorResult(newOp, resType, rewriter);
    rewriter.replaceAllUsesWith(result, newResult);
    return success();
  }

protected:
  // Callback function for creating different kinds of arith/math op.
  virtual Operation *
  createNewComputationOp(OpType op, Type dstType, ArrayRef<Value> operands,
                         Location loc, PatternRewriter &rewriter) const = 0;

private:
  SmallVector<Value> flattenVectorOperands(Operation *op, int64_t totalElements,
                                           PatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      auto oldType = dyn_cast<VectorType>(operand.getType());
      if (!oldType) {
        operands.push_back(operand);
        continue;
      }
      VectorType newType =
          VectorType::get({totalElements}, oldType.getElementType());
      auto newOperand =
          rewriter.create<vector::ShapeCastOp>(loc, newType, operand);
      operands.push_back(newOperand);
    }
    return operands;
  }

  Value flattenVectorResult(Operation *op, Type dstType,
                            PatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto newResult =
        rewriter.create<vector::ShapeCastOp>(loc, dstType, op->getResult(0));
    return newResult;
  }
};

// Partial specialization for flattening arith/math unary op
template <typename OpType>
class FlattenUnaryOp
    : public FlattenComputationOp<OpType, ComputationOpKind::Unary> {
public:
  explicit FlattenUnaryOp(MLIRContext *ctx)
      : FlattenComputationOp<OpType, ComputationOpKind::Unary>(ctx) {}

protected:
  Operation *createNewComputationOp(OpType op, Type dstType,
                                    ArrayRef<Value> operands, Location loc,
                                    PatternRewriter &rewriter) const final {
    return rewriter.create<OpType>(loc, dstType, operands[0]);
  }
};

// Partial specialization for flattening arith/math binary op
template <typename OpType>
class FlattenBinaryOp
    : public FlattenComputationOp<OpType, ComputationOpKind::Binary> {
public:
  explicit FlattenBinaryOp(MLIRContext *ctx)
      : FlattenComputationOp<OpType, ComputationOpKind::Binary>(ctx) {}

protected:
  Operation *createNewComputationOp(OpType op, Type dstType,
                                    ArrayRef<Value> operands, Location loc,
                                    PatternRewriter &rewriter) const override {
    if constexpr (std::is_same_v<OpType, arith::CmpIOp> ||
                  std::is_same_v<OpType, arith::CmpFOp>) {
      // Compare-like op requires extra predicate argument
      auto pred = op.getPredicate();
      return rewriter.create<OpType>(loc, dstType, pred, operands[0],
                                     operands[1]);
    } else {
      return rewriter.create<OpType>(loc, dstType, operands[0], operands[1]);
    }
  }
};

// Partial specialization for flattening ternary op like `select` op
template <typename OpType>
class FlattenTernaryOp
    : public FlattenComputationOp<OpType, ComputationOpKind::Ternary> {
public:
  explicit FlattenTernaryOp(MLIRContext *ctx)
      : FlattenComputationOp<OpType, ComputationOpKind::Ternary>(ctx) {}

protected:
  Operation *createNewComputationOp(OpType op, Type dstType,
                                    ArrayRef<Value> operands, Location loc,
                                    PatternRewriter &rewriter) const override {
    return rewriter.create<arith::SelectOp>(loc, TypeRange{dstType}, operands);
  }
};

} // namespace

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  // register flatten vector for unary and binary arith/math op respectively
  MLIRContext *ctx = patterns.getContext();
  if constexpr (OpType::template hasTrait<OpTrait::NOperands<2>::Impl>()) {
    patterns.add<FlattenBinaryOp<OpType>>(ctx);
  } else if constexpr (OpType::template hasTrait<OpTrait::OneOperand>()) {
    patterns.add<FlattenUnaryOp<OpType>>(ctx);
  } else if constexpr (OpType::template hasTrait<
                           OpTrait::NOperands<3>::Impl>()) {
    patterns.add<FlattenTernaryOp<OpType>>(ctx);
  }
}

/// Variadic helper function.
template <typename... OpTypes>
static void registerAll(RewritePatternSet &patterns) {
  (registerOne<OpTypes>(patterns), ...);
}

void vector::populateFlattenVectorOpsPattern(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<FlattenContiguousTransferWrite>(ctx);
  patterns.add<FlattenVectorGatherOp>(ctx);
  patterns.add<FlattenTransferReadWithRankReduceShapeCast>(ctx);
  patterns.add<NormalizeIneffectiveTranspose>(ctx);
  registerAll<
#define GET_OP_LIST
#include "mlir/Dialect/Arith/IR/ArithOps.cpp.inc"
      >(patterns);
  registerAll<
#define GET_OP_LIST
#include "mlir/Dialect/Math/IR/MathOps.cpp.inc"
      >(patterns);
}
