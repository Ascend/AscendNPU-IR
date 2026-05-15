//===-------------------- ConvertLayoutToTranspose.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <bishengir/Dialect/Utils/Util.h>

#define DEBUG_TYPE "hivm-convert-layout-to-transpose"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_CONVERTLAYOUTTOTRANSPOSE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
/// Build reassociation indices for expand/collapse shape
SmallVector<ReassociationIndices> buildReassociation() {
  return {{0, 1}, {2, 3}};
}

/// Apply a given permutation to a shape
template <class T>
SmallVector<T> applyPermutation(ArrayRef<T> shape,
                                ArrayRef<int64_t> permutation) {
  SmallVector<T> result;
  result.reserve(shape.size());
  for (int64_t idx : permutation) {
    result.push_back(shape[idx]);
  }
  return result;
}

enum FractalPart : uint32_t {
  kFractalDst,
  kFractalSrc,
};

/// Common validation result for layout conversion patterns
struct LayoutConversionInfo {
  RankedTensorType inputType;
  RankedTensorType outputType;
  ArrayRef<int64_t> inputShape;
  bool hasBatch;
  Location loc;
};

/// Check if the conversion matches expected src/dst layouts.
bool checkFractalLayout(ConvertLayoutOp op, hivm::DataLayout expectedSrc,
                        hivm::DataLayout expectedDst) {
  auto srcLayout = op.getSrcLayoutAttr();
  auto dstLayout = op.getDstLayoutAttr();
  if (!srcLayout || !dstLayout)
    return false;

  return srcLayout.getDataLayout() == expectedSrc &&
         dstLayout.getDataLayout() == expectedDst;
}

LayoutConversionInfo extractConversionInfo(ConvertLayoutOp op) {
  auto inputType = cast<RankedTensorType>(op.getSource().getType());
  auto outputType = cast<RankedTensorType>(op.getResult().getType());

  bool hasBatch = inputType.getRank() % 2 == 1;

  return LayoutConversionInfo{inputType, outputType, inputType.getShape(),
                              hasBatch, op.getLoc()};
}

const SmallVector<int64_t> kFractalLayoutPermutation = {2, 0, 1, 3};

/// Align `size` up to multiple of `tile`: ceilDiv(size, tile) * tile
static OpFoldResult alignUpOFR(PatternRewriter &rewriter, Location loc,
                               OpFoldResult size, int64_t tile) {
  MLIRContext *ctx = rewriter.getContext();
  AffineExpr d0 = getAffineDimExpr(0, ctx);
  auto map = AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/0,
      {((d0 + (tile - 1)).floorDiv(tile)) * tile}, ctx);
  return affine::makeComposedFoldedAffineApply(rewriter, loc, map, {size});
}

static SmallVector<OpFoldResult> getZeroOffsets(PatternRewriter &rewriter,
                                                int64_t rank) {
  SmallVector<OpFoldResult> v(rank, rewriter.getIndexAttr(0));
  return v;
}

static SmallVector<OpFoldResult> getUnitStrides(PatternRewriter &rewriter,
                                                int64_t rank) {
  SmallVector<OpFoldResult> v(rank, rewriter.getIndexAttr(1));
  return v;
}

struct CanonicalToFractalDecompose : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  /// Create a zero-padded 2D base tensor for ND -> Fractal conversion.
  ///
  /// The source tensor is padded on M/N dimensions to the nearest multiples of
  /// `fractalSizes[0]` and `fractalSizes[1]` respectively. Padding values are
  /// initialized to zero using `hivm::VBrcOp`, then the original source is
  /// inserted at offset [0, 0] via `tensor::InsertSliceOp`.
  ///
  /// If the source shape is statically aligned to fractal tile sizes, this
  /// returns the original source directly (no extra ops).
  static FailureOr<Value>
  createEmptyPaddedBase(ConvertLayoutOp op, PatternRewriter &rewriter,
                        const LayoutConversionInfo &info) {
    auto fractalSizes = op.getDstLayout().getFractalSizesArray().value_or(
        SmallVector<int64_t>());
    if (fractalSizes.size() != 2)
      return rewriter.notifyMatchFailure(op, "invalid fractal size");

    Value src = op.getSource();
    auto srcTy = cast<RankedTensorType>(src.getType());
    Type elemTy = srcTy.getElementType();

    int64_t tileM = fractalSizes[0];
    int64_t tileN = fractalSizes[1];
    if (tileM <= 0 || tileN <= 0)
      return rewriter.notifyMatchFailure(op, "fractal sizes must be > 0");

    // Fast path: if statically known and already aligned, skip padding.
    int64_t mStatic = srcTy.getDimSize(0);
    int64_t nStatic = srcTy.getDimSize(1);
    bool mKnown = mStatic != ShapedType::kDynamic;
    bool nKnown = nStatic != ShapedType::kDynamic;
    if (mKnown && nKnown && (mStatic % tileM == 0) && (nStatic % tileN == 0))
      return src;

    auto toAttr = [](OpFoldResult ofr) -> OpFoldResult {
      if (auto value = getConstantIntValue(ofr)) {
        return getAsIndexOpFoldResult(ofr.getContext(), *value);
      }
      return ofr;
    };
    // Otherwise, build padded base (dynamic-safe).
    OpFoldResult m =
        toAttr(rewriter.createOrFold<tensor::DimOp>(info.loc, src, 0));
    OpFoldResult n =
        toAttr(rewriter.createOrFold<tensor::DimOp>(info.loc, src, 1));

    OpFoldResult alignedM = alignUpOFR(rewriter, info.loc, m, tileM);
    OpFoldResult alignedN = alignUpOFR(rewriter, info.loc, n, tileN);

    SmallVector<OpFoldResult> paddedShape{alignedM, alignedN};
    auto emptyPadded =
        rewriter.create<tensor::EmptyOp>(info.loc, paddedShape, elemTy);

    Value zero = rewriter.create<arith::ConstantOp>(
        info.loc, elemTy, rewriter.getZeroAttr(elemTy));

    auto paddedTy = cast<RankedTensorType>(emptyPadded.getType());
    auto vbrc =
        rewriter.create<hivm::VBrcOp>(info.loc, paddedTy, zero, emptyPadded);
    Value paddedInit = vbrc.getResult().front();

    auto offsets2 = getZeroOffsets(rewriter, 2);
    SmallVector<OpFoldResult> sizes2{m, n};
    auto strides2 = getUnitStrides(rewriter, 2);

    Value paddedSrc = rewriter.create<tensor::InsertSliceOp>(
        info.loc, src, paddedInit, offsets2, sizes2, strides2);

    return paddedSrc;
  }

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {

    if (!checkFractalLayout(op, hivm::DataLayout::ND,
                            hivm::DataLayout::Fractal))
      return rewriter.notifyMatchFailure(op, "not a ND -> Fractal conversion");

    LayoutConversionInfo info = extractConversionInfo(op);
    if (info.hasBatch)
      return rewriter.notifyMatchFailure(
          op, "Batch matmul is currently not supported");

    rewriter.setInsertionPointAfter(op);
    auto paddedSrc = createEmptyPaddedBase(op, rewriter, info);
    if (failed(paddedSrc))
      return paddedSrc;

    auto reassociation = buildReassociation();
    auto resultShape = op.getMixedOutputShape();

    auto inversedPermutation =
        utils::inversePermutation(kFractalLayoutPermutation);
    auto mixedExpandedShape =
        applyPermutation(resultShape, inversedPermutation);

    auto expandedType =
        cast<ShapedType>(op.getResult().getType())
            .clone(decomposeMixedValues(mixedExpandedShape).first);

    auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
        info.loc, expandedType, paddedSrc.value(), reassociation);

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        info.loc, resultShape, getElementTypeOrSelf(op.getResult()));

    auto transposeOp = rewriter.create<hivm::VTransposeOp>(
        info.loc, TypeRange(emptyTensor.getType()), expandOp.getResult(),
        emptyTensor, rewriter.getDenseI64ArrayAttr(kFractalLayoutPermutation));

    rewriter.replaceOp(op, transposeOp.getResult());
    return success();
  }
};

struct FractalToCanonicalDecompose : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    if (!checkFractalLayout(op, hivm::DataLayout::Fractal,
                            hivm::DataLayout::ND))
      return rewriter.notifyMatchFailure(op, "not a Fractal -> ND conversion");

    auto info = extractConversionInfo(op);
    if (info.hasBatch)
      return rewriter.notifyMatchFailure(
          op, "Batch matmul is currently not supported");

    auto permutation = utils::inversePermutation(kFractalLayoutPermutation);
    auto emptyMixedShape =
        tensor::getMixedSizes(rewriter, info.loc, op.getSource());
    auto mixedExpandedShape = applyPermutation(emptyMixedShape, permutation);

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        info.loc, mixedExpandedShape, getElementTypeOrSelf(op.getResult()));

    auto transposeOp = rewriter.create<hivm::VTransposeOp>(
        info.loc, TypeRange(emptyTensor.getType()), op.getSource(), emptyTensor,
        rewriter.getDenseI64ArrayAttr(permutation));

    auto reassociation = buildReassociation();

    // Collapse to dynamic 2D first (aligned ND).
    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        info.loc, transposeOp.getResult()[0], reassociation);

    // Slice to requested ND output shape (e.g. 130x145).
    auto outMixed = op.getMixedOutputShape();
    auto offsets2 = getZeroOffsets(rewriter, 2);
    auto strides2 = getUnitStrides(rewriter, 2);

    auto slice = rewriter.create<tensor::ExtractSliceOp>(
        info.loc, cast<RankedTensorType>(op.getResult().getType()),
        collapseOp.getResult(), offsets2, outMixed, strides2);

    rewriter.replaceOp(op, slice.getResult());
    return success();
  }
};

void populateConvertLayoutToTranspose(RewritePatternSet &patterns,
                                      MLIRContext *context) {
  patterns.add<CanonicalToFractalDecompose, FractalToCanonicalDecompose>(
      context);
}
} // namespace

struct ConvertLayoutToTransposePass
    : public impl::ConvertLayoutToTransposeBase<ConvertLayoutToTransposePass> {
  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    populateConvertLayoutToTranspose(patterns, context);
    ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::hivm::createConvertLayoutToTransposePass() {
  return std::make_unique<ConvertLayoutToTransposePass>();
}