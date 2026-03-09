//===-------------------- ConvertLayoutToTranspose.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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
bool
checkFractalLayout(ConvertLayoutOp op, hivm::DataLayout expectedSrc,
                   hivm::DataLayout expectedDst) {
  auto srcLayout = op.getSrcLayoutAttr();
  auto dstLayout = op.getDstLayoutAttr();
  if (!srcLayout || !dstLayout)
    return false;

  return srcLayout.getDataLayout() == expectedSrc && dstLayout.getDataLayout()
         == expectedDst;
}

LayoutConversionInfo extractConversionInfo(ConvertLayoutOp op) {
  auto inputType = cast<RankedTensorType>(op.getSource().getType());
  auto outputType = cast<RankedTensorType>(op.getResult().getType());

  bool hasBatch = inputType.getRank() % 2 == 1;

  return LayoutConversionInfo{
      inputType,
      outputType,
      inputType.getShape(),
      hasBatch,
      op.getLoc()};
}

const SmallVector<int64_t> kFractalLayoutPermutation = {2, 0, 1, 3};

struct CanonicalToFractalDecompose : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;


  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    auto fractalLayout = checkFractalLayout(
        op, hivm::DataLayout::ND, hivm::DataLayout::Fractal);
    if (!fractalLayout)
      return rewriter.notifyMatchFailure(op, "not a ND -> nZ conversion");
    auto info = extractConversionInfo(op);
    if (info.hasBatch) {
      return rewriter.notifyMatchFailure(
          op, "Batch matmul is currently not supported");
    }
    assert(!info.hasBatch);
    auto fractalSizes = op.getDstLayout().getFractalSizesArray();

    auto reassociation = buildReassociation();
    auto resultShape = op.getMixedOutputShape();
    rewriter.setInsertionPointAfter(op);

    // Apply inverse permutation to the mixedOutputShape
    auto inversedPermutation = utils::inversePermutation(kFractalLayoutPermutation);
    auto mixedExpandedShape =
        applyPermutation(resultShape, inversedPermutation);
    auto expandedType = cast<ShapedType>(op.getResult().getType()).clone(
        decomposeMixedValues(mixedExpandedShape).first);
    auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
        op.getLoc(), expandedType, op.getSource(), reassociation);
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        info.loc, resultShape, getElementTypeOrSelf(op.getResult()));
    auto transposeOp = rewriter.create<hivm::VTransposeOp>(
        info.loc, TypeRange(emptyTensor.getType()), expandOp.getResult(),
        emptyTensor,
        rewriter.getDenseI64ArrayAttr(kFractalLayoutPermutation));
    rewriter.replaceOp(op, transposeOp);
    return success();
  }
};

struct FractalToCanonicalDecompose : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    auto fractalLayout = checkFractalLayout(
        op, hivm::DataLayout::Fractal, hivm::DataLayout::ND);
    if (!fractalLayout)
      return rewriter.notifyMatchFailure(op, "not a Fractal -> ND conversion");

    auto info = extractConversionInfo(op);

    LDBG("Converting Fractal -> ND (down): " << info.inputType << " -> "
        << info.outputType);

    // Fractal to canonical is easier, we can simply do the inverse, no need to even do anything
    auto permutation = utils::inversePermutation(kFractalLayoutPermutation);

    // TODO: ... do we need to keep the input shape as well here
    auto emptyMixedShape = tensor::getMixedSizes(
        rewriter, info.loc, op.getSource());
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        info.loc, emptyMixedShape, getElementTypeOrSelf(op.getResult()));
    auto transposeOp = rewriter.create<hivm::VTransposeOp>(
        info.loc, TypeRange(emptyTensor.getType()), op.getSource(),
        emptyTensor,
        rewriter.getDenseI64ArrayAttr(permutation));
    auto reassociation = buildReassociation();
    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        info.loc, op.getResult().getType(), transposeOp.getResult()[0], reassociation);
    rewriter.replaceOp(op, collapseOp);
    return success();
  }
};

void populateConvertLayoutToTranspose(RewritePatternSet &patterns,
                                      MLIRContext *context) {
  patterns.add<
    CanonicalToFractalDecompose,
    FractalToCanonicalDecompose
  >(context);
}
}

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