//===--------------- RemoveRedundantWriteAndReadPair.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
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

  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}


std::unique_ptr<Pass>
mlir::hfusion::createRemoveRedundantWriteAndReadPairPass() {
  return std::make_unique<RemoveRedundantWriteAndReadPairPass>();
}