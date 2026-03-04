//===------ TensorToHIVM.cpp - conversion from Tensor to HIVM dialect -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TensorToHIVM/TensorToHIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/RWMutex.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTENSORTOHIVM
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

struct TensorToHIVMConcatOp : public OpRewritePattern<tensor::ConcatOp> {
  using OpRewritePattern<tensor::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> outputSizes;
    ReifiedRankedShapedTypeDims reifiedReturnShapes;
    if (failed(concatOp.reifyResultShapes(rewriter, reifiedReturnShapes))) {
      return failure();
    }
    outputSizes = reifiedReturnShapes.front();
    auto emptyDest = rewriter.create<tensor::EmptyOp>(
        concatOp.getLoc(), outputSizes,
        concatOp.getResultType().getElementType());
    rewriter.replaceOpWithNewOp<hivm::VConcatOp>(
        concatOp, concatOp.getResult().getType(), concatOp.getDim(),
        concatOp.getInputs(), emptyDest);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TensorToHIVMPadOp
//===----------------------------------------------------------------------===//
/// Convert tensor.pad to hivm.hir.vpad
/// e.g.
///   tensor.pad %source low[2046] high[0]
///     pad_value = %cst : f32
///     : tensor<2047xf32> to tensor<4093xf32>
/// converts to
///   %empty = tensor.empty() : tensor<4093xf32>
///   hivm.hir.vpad
///     ins(%source : tensor<2047xf32>)
///     outs(%empty: tensor<4093xf32>)
///     low[2046] high[0]
///     pad_value %cst : f32
///     -> tensor<4093xf32>
/// we create an empty tensor to store the init for hivm.vpad output
/// when sizes are dynamic we create affine map to calculate it
struct TensorToHIVMPadOp : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto src = padOp.getSource();
    // We have to create our own destination init
    SmallVector<OpFoldResult> resultSizes;
    auto sourceSizes = tensor::getMixedSizes(rewriter, padOp.getLoc(), src);
    auto lowPads = padOp.getMixedLowPad();
    auto highPads = padOp.getMixedHighPad();
    for (auto [srcSize, lowPad, highPad] :
         llvm::zip(sourceSizes, lowPads, highPads)) {
      // Compute result dimension size: srcSize + lowPad + highPad
      AffineExpr expr = rewriter.getAffineSymbolExpr(0) +
                        rewriter.getAffineSymbolExpr(1) +
                        rewriter.getAffineSymbolExpr(2);
      OpFoldResult sum = affine::makeComposedFoldedAffineApply(
          rewriter, padOp.getLoc(), expr, {srcSize, lowPad, highPad});
      resultSizes.push_back(sum);
    }
    auto res = padOp->getResult(0);
    auto padValue = padOp.getConstantPaddingValue();
    auto lowDynamic = padOp.getLow();
    auto highDynamic = padOp.getHigh();
    auto lowStatic = padOp.getStaticLow();
    auto highStatic = padOp.getStaticHigh();
    auto dst = rewriter.create<tensor::EmptyOp>(
        padOp.getLoc(), resultSizes, padOp.getResultType().getElementType());
    auto hivmPadOp = rewriter.create<hivm::VPadOp>(
        padOp->getLoc(), TypeRange(dst.getResult()), src, dst, padValue,
        lowDynamic, highDynamic, lowStatic, highStatic);
    if (cast<ShapedType>(res.getType()).getShape() !=
        dst.getResult().getType().getShape()) {
      auto castOp = rewriter.create<tensor::CastOp>(
          padOp.getLoc(), res.getType(), hivmPadOp.getResult());
      rewriter.replaceOp(padOp, castOp);
      return success();
    }
    rewriter.replaceOp(padOp, hivmPadOp);
    return success();
  }
};

void mlir::hivm::populateTensorToHIVMConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TensorToHIVMConcatOp, TensorToHIVMPadOp>(patterns.getContext());
}

namespace {
struct TensorToHIVMConversionPass
    : public impl::ConvertTensorToHIVMBase<TensorToHIVMConversionPass> {
  void runOnOperation() override;
};
} // namespace

void TensorToHIVMConversionPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  target.addLegalDialect<hivm::HIVMDialect, func::FuncDialect,
                         tensor::TensorDialect, arith::ArithDialect,
                         affine::AffineDialect>();
  target.addIllegalOp<tensor::ConcatOp, tensor::PadOp>();
  populateTensorToHIVMConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createTensorToHIVMConversionPass() {
  return std::make_unique<TensorToHIVMConversionPass>();
}