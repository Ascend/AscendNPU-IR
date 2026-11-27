//===--------------- RemoveRedundantWriteAndReadPair.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_REMOVEREDUNDANTWRITEANDREADPAIR
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "remove-redundant-write-and-read-pair"

using namespace mlir;
using namespace mlir::hfusion;

namespace {
/// Replace the transfer_read by source vector of transfer_write.
/// Example:
/// ```
/// %7 = arith.cmpi ne, %6, %cst_2
/// %8 = vector.transfer_write %7, %extract_slice_4[%c0, %c0]
/// %inserted_slice = tensor.insert_slice %8 into %extract_slice_4
/// %13 = vector.transfer_read %inserted_slice[%c0, %c0]
/// %20 = arith.select %13, %19, %17
/// ```
/// To:
/// ```
/// %7 = arith.cmpi ne, %6, %cst_2
/// %20 = arith.select %7, %19, %17
/// ```
struct FoldTransferReadAfterWriteAndInsertSlice
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    if (readOp.hasOutOfBoundsDim() ||
        !llvm::isa<RankedTensorType>(readOp.getShapedType()))
      return failure();
    auto defInsertSlice = readOp.getSource().getDefiningOp<tensor::InsertSliceOp>();
    if (!defInsertSlice)
      return failure();
    auto defWrite = defInsertSlice.getSource().getDefiningOp<vector::TransferWriteOp>();
    if (!defWrite)
      return failure();
    if (readOp.getIndices() != defWrite.getIndices() ||
        readOp.getVectorType().getNumElements() !=
            defWrite.getVectorType().getNumElements())
      return failure();
    // add shape cast, if the read shape different with write shape
    if (readOp.getVectorType() != defWrite.getVectorType()) {
      Location loc = readOp->getLoc();
      vector::ShapeCastOp shapeCast = rewriter.create<vector::ShapeCastOp>(
          loc, readOp.getVectorType(), defWrite.getVector());
      rewriter.replaceOp(readOp, shapeCast);
    } else {
      rewriter.replaceOp(readOp, defWrite.getVector());
    }
    return success();
  }
};

/// Return the sizes of the leading all-true region of `mask`, if they are
/// statically known. Supports `vector.constant_mask` and `vector.create_mask`
/// with constant bounds.
 SmallVector<int64_t, 4> getStaticMaskSizes(Value mask) {
    SmallVector<int64_t, 4> sizes;

  if (auto constMask = mask.getDefiningOp<vector::ConstantMaskOp>()) {
    for (Attribute attr : constMask.getMaskDimSizes().getValue())
      sizes.push_back(llvm::cast<IntegerAttr>(attr).getInt());

    return sizes;
  }

  if (auto createMask = mask.getDefiningOp<vector::CreateMaskOp>()) {
    for (Value bound : createMask.getOperands()) {
      std::optional<int64_t> cst = getConstantIntValue(bound);
      if (!cst)
        return {};

      sizes.push_back(*cst);
    }
  }

  return sizes;
 }

/// Replace the transfer_read by a broadcast of the source vector of the
/// transfer_write, when the read vector is wider than the written one and the
/// extra lanes are cut off by a static mask.
/// Example:
/// ```
/// %6 = vector.multi_reduction <add>, %4, %5 [2]
///        : vector<1x1x64xf32> to vector<1x1xf32>
/// %7 = vector.transfer_write %6, %extracted_slice_3[%c0, %c0]
///        : vector<1x1xf32>, tensor<1x1xf32>
/// %8 = vector.constant_mask [1, 1] : vector<1x64xi1>
/// %10 = vector.transfer_read %7[%c0, %c0], %cst_0, %8
///        : tensor<1x1xf32>, vector<1x64xf32>
/// %11 = arith.divf %9, %10 : vector<1x64xf32>
/// ```
/// To:
/// ```
/// %6 = vector.multi_reduction <add>, %4, %5 [2]
///        : vector<1x1x64xf32> to vector<1x1xf32>
/// %10 = vector.broadcast %6 : vector<1x1xf32> to vector<1x64xf32>
/// %11 = arith.divf %9, %10 : vector<1x64xf32>
/// ```
struct FoldWidenedTransferReadAfterWrite
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    if (readOp.hasOutOfBoundsDim() ||
        !llvm::isa<RankedTensorType>(readOp.getShapedType()))
      return failure();

    auto defWrite = readOp.getSource().getDefiningOp<vector::TransferWriteOp>();
    if (!defWrite)
      return failure();

    // The write has to define every element of the region it claims to write.
    if (defWrite.getMask() || defWrite.hasOutOfBoundsDim())
      return failure();

    // Same location, same layout. Restricting to minor identity maps keeps the
    // element correspondence trivial once the vector shapes differ.
    if (readOp.getIndices() != defWrite.getIndices() ||
        readOp.getPermutationMap() != defWrite.getPermutationMap() ||
        !readOp.getPermutationMap().isMinorIdentity())
      return failure();

    VectorType readType = readOp.getVectorType();
    VectorType writeType = defWrite.getVectorType();
    if (readType.getElementType() != writeType.getElementType() ||
        readType.getRank() != writeType.getRank() || readType.getRank() == 0)
      return failure();

    // Every element consumed by the read must have been produced by the write,
    // i.e. the masked-in region of the read has to fit into the written vector.
    SmallVector<int64_t, 4> activeSizes(readType.getShape());

    if (Value mask = readOp.getMask()) {
      auto maskSizes = getStaticMaskSizes(mask);
      if (maskSizes.empty() || maskSizes.size() != activeSizes.size())
        return failure();

      activeSizes = maskSizes;
    }

    for (size_t dim = 0, rank = readType.getRank(); dim < rank; ++dim) {
      if (activeSizes[dim] > writeType.getDimSize(dim) ||
          writeType.getDimSize(dim) > readType.getDimSize(dim))
        return failure();
    }

    if (readType == writeType) {
      rewriter.replaceOp(readOp, defWrite.getVector());
      return success();
    }

    // Widening. `vector.broadcast` can only stretch dimensions of size 1, and
    // VecBroadcastOpPattern in VectorToHIVMAVE lowers a vector source only when
    // it holds a single element and the result is a single row.
    if (writeType.getNumElements() != 1 ||
        readType.getNumElements() != readType.getShape().back())
      return failure();

    Location loc = readOp->getLoc();
    vector::BroadcastOp broadcast = rewriter.create<vector::BroadcastOp>(
        loc, readType, defWrite.getVector());
    rewriter.replaceOp(readOp, broadcast);

    return success();
  }
};

static SmallVector<OpFoldResult> getMaskBounds(Value mask) {
  SmallVector<OpFoldResult> bounds;
  if (auto create = mask.getDefiningOp<vector::CreateMaskOp>()) {
    for (Value bound : create.getOperands())
      bounds.push_back(bound);
  } else if (auto constant = mask.getDefiningOp<vector::ConstantMaskOp>()) {
    for (Attribute bound : constant.getMaskDimSizes())
      bounds.push_back(bound);
  }
  return bounds;
}

// Only look through an insert that updates the whole region the transfer
// defined on the same tensor the write consumed. The region is the mask
// bounds when the write is masked and the vector shape otherwise; requiring
// exact equality keeps the proof independent of index wrapping in well-formed
// programs.
static bool insertCoversWrite(tensor::InsertSliceOp insert,
                              vector::TransferWriteOp write) {
  if (insert.getDest() != write.getSource() ||
      !llvm::all_of(insert.getMixedOffsets(),
                    [](OpFoldResult offset) {
                      return getConstantIntValue(offset) == 0;
                    }) ||
      !llvm::all_of(insert.getMixedStrides(), [](OpFoldResult stride) {
        return getConstantIntValue(stride) == 1;
      }))
    return false;

  VectorType writeType = write.getVectorType();
  auto sizes = insert.getMixedSizes();
  if (static_cast<int64_t>(sizes.size()) != writeType.getRank())
    return false;
  if (Value mask = write.getMask())
    return isEqualConstantIntOrValueArray(sizes, getMaskBounds(mask));
  return llvm::all_of(
      llvm::zip_equal(sizes, writeType.getShape()), [](auto pair) {
        return getConstantIntValue(std::get<0>(pair)) == std::get<1>(pair);
      });
}

static bool isScaledBound(OpFoldResult flat, OpFoldResult rows,
                          OpFoldResult rowExtent, int64_t width) {
  if (auto flatConstant = getConstantIntValue(flat))
    return *flatConstant % width == 0 &&
           getConstantIntValue(rows) == *flatConstant / width;

  // A valid reshape makes rows * width equal to the input dimension when
  // rows is the reshape extent. This rules out overflow of the scaled bound.
  auto flatValue = dyn_cast<Value>(flat);
  auto rowsValue = dyn_cast<Value>(rows);
  if (!flatValue || !rowsValue || rows != rowExtent)
    return false;
  auto apply = flatValue.getDefiningOp<affine::AffineApplyOp>();
  if (!apply || apply.getMapOperands().size() != 1 ||
      apply.getMapOperands().front() != rowsValue)
    return false;
  AffineExpr row = getAffineDimExpr(0, apply.getContext());
  return apply.getAffineMap() == AffineMap::get(1, 0, row * width);
}

/// Forward a single vector row through a trailing two-axis expand_shape.
///
/// Work around #1071 by forwarding before bufferization introduces copies
/// that leave a memref result on a vector function. Limit the match to the
/// layouts, masks, vector types, and padding values validated for this fix so
/// forwarding preserves lane order and padding and can be lowered on A5.
/// Coverage can grow as more cases are validated. This pattern can be merged
/// with FoldTransferReadAfterWriteAndInsertSlice once both paths share the
/// same lane and mask checks.
///
/// The source-order read lane [row, col] must correspond to write lane
/// row * width + col. Moving unit vector axes does not change this order.
struct FoldTransferReadOfExpandedWrite
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    auto expand = read.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expand || read.hasOutOfBoundsDim() ||
        read->getParentOfType<vector::MaskOp>())
      return failure();
    Value tensor = expand.getSrc();
    auto insert = tensor.getDefiningOp<tensor::InsertSliceOp>();
    auto write = (insert ? insert.getSource() : tensor)
                     .getDefiningOp<vector::TransferWriteOp>();
    if (!write || write.hasOutOfBoundsDim() ||
        write->getParentOfType<vector::MaskOp>() ||
        !write.getPermutationMap().isIdentity() ||
        (insert && !insertCoversWrite(insert, write)))
      return failure();

    auto isZero = [](Value index) { return getConstantIntValue(index) == 0; };
    if (!llvm::all_of(read.getIndices(), isZero) ||
        !llvm::all_of(write.getIndices(), isZero))
      return failure();

    VectorType writeType = write.getVectorType();
    VectorType readType = read.getVectorType();
    auto reassociation = expand.getReassociationIndices();
    if (reassociation.empty() || reassociation.back().size() != 2 ||
        !llvm::all_of(ArrayRef(reassociation).drop_back(),
                      [](const ReassociationIndices &group) {
                        return group.size() == 1;
                      }) ||
        writeType.getRank() != static_cast<int64_t>(reassociation.size()) ||
        writeType.getRank() > 2 ||
        readType.getRank() != writeType.getRank() + 1 ||
        writeType.getElementType() != readType.getElementType() ||
        writeType.isScalable() || readType.isScalable() ||
        !hivm::util::isOneDimLikeVecType(writeType))
      return failure();

    AffineMap permutation = read.getPermutationMap();
    if (!permutation.isPermutation())
      return failure();
    // Only unit axes may move: non-unit source axis IDs must stay ordered.
    SmallVector<int64_t> sourceShape(readType.getRank());
    int64_t previous = -1;
    for (auto [expr, extent] :
         llvm::zip(permutation.getResults(), readType.getShape())) {
      int64_t dim = cast<AffineDimExpr>(expr).getPosition();
      sourceShape[dim] = extent;
      if (extent == 1)
        continue;
      if (dim <= previous)
        return failure();
      previous = dim;
    }
    int64_t width = expand.getResultType().getShape().back();
    int64_t flatWidth = writeType.getShape().back();
    size_t rowDim = sourceShape.size() - 2;
    if (width <= 0 || sourceShape.back() != width || flatWidth % width != 0 ||
        sourceShape[rowDim] != flatWidth / width ||
        !llvm::all_of(ArrayRef(sourceShape).drop_back(2),
                      [](int64_t dim) { return dim == 1; }))
      return failure();

    Value forwarded = write.getVector();
    if (read.getMask() || write.getMask()) {
      Type elementType = writeType.getElementType();
      // Keep the padding select in the original rank-1/single-row vector, and
      // keep that vector within one hardware vector length.
      if (!read.getMask() || !write.getMask() ||
          !(elementType.isF16() || elementType.isBF16() ||
            elementType.isF32()) ||
          writeType.getNumElements() * elementType.getIntOrFloatBitWidth() >
              static_cast<int64_t>(hivm::util::VL_BITS))
        return failure();
      auto readBounds = getMaskBounds(read.getMask());
      auto writeBounds = getMaskBounds(write.getMask());
      // The padding is broadcast to the write element type, so it must already
      // have that type. A scalar lifted from a multi-dimensional vector is
      // skipped as well: VectorToHIVMAVE rewrites a broadcast of an all-zero
      // extract into a broadcast of the extract's source vector, which only
      // lowers for 1-D sources within one vector length.
      if (read.getPadding().getType() != elementType)
        return failure();
      if (auto extract = read.getPadding().getDefiningOp<vector::ExtractOp>()) {
        auto sourceType = extract.getSourceVectorType();
        if (sourceType.getRank() > 1 ||
            sourceType.getNumElements() *
                    sourceType.getElementType().getIntOrFloatBitWidth() >
                static_cast<int64_t>(hivm::util::VL_BITS))
          return failure();
      }
      auto isUnit = [](OpFoldResult bound) {
        return getConstantIntValue(bound) == 1;
      };
      // A complete inner dimension means the read mask flattens to a prefix.
      // Require the write to define exactly that prefix, including on tails.
      if (readBounds.size() != sourceShape.size() ||
          writeBounds.size() != writeType.getShape().size() ||
          !llvm::all_of(ArrayRef(readBounds).drop_back(2), isUnit) ||
          !llvm::all_of(ArrayRef(writeBounds).drop_back(), isUnit) ||
          getConstantIntValue(readBounds.back()) != width ||
          !isScaledBound(writeBounds.back(), readBounds[rowDim],
                         expand.getMixedOutputShape()[rowDim], width))
        return failure();

      // A masked read produces padding in its inactive lanes, not the
      // original write vector's inactive lanes.
      Location loc = read.getLoc();
      Value padding = rewriter.create<vector::BroadcastOp>(loc, writeType,
                                                           read.getPadding());
      forwarded = rewriter.create<arith::SelectOp>(loc, write.getMask(),
                                                   forwarded, padding);
    }
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(read, readType, forwarded);
    return success();
  }
};

struct RemoveRedundantWriteAndReadPairPass
    : public impl::RemoveRedundantWriteAndReadPairBase<
          RemoveRedundantWriteAndReadPairPass> {
public:
  void runOnOperation() override;
};
} // namespace

void RemoveRedundantWriteAndReadPairPass::runOnOperation() {
  func::FuncOp func = getOperation();
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<FoldTransferReadAfterWriteAndInsertSlice>(ctx);
  patterns.add<FoldWidenedTransferReadAfterWrite>(ctx);
  patterns.add<FoldTransferReadOfExpandedWrite>(ctx);

  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass>
mlir::hfusion::createRemoveRedundantWriteAndReadPairPass() {
  return std::make_unique<RemoveRedundantWriteAndReadPairPass>();
}
