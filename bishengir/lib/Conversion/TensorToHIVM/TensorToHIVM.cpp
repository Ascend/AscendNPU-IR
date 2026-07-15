//===------ TensorToHIVM.cpp - conversion from Tensor to HIVM dialect -----===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "bishengir/Conversion/TensorToHIVM/TensorToHIVM.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
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
    auto newConcatOp = rewriter.replaceOpWithNewOp<hivm::VConcatOp>(
        concatOp, concatOp.getResult().getType(), concatOp.getDim(),
        concatOp.getInputs(), emptyDest);
    // membase-only
    if (hacc::utils::isMemBasedArch(
            concatOp->getParentOfType<ModuleOp>())) {
      auto index = traceInsertSliceSourceIndex(concatOp, rewriter);
      if (index.has_value()) {
        newConcatOp->setAttr(hivm::InsertSliceSourceIndexAttr::name,
                             rewriter.getI64IntegerAttr(index.value()));
      }
    }
    return success();
  }

  std::optional<int64_t>
  traceInsertSliceSourceIndex(Operation *op, PatternRewriter &rewriter) const {
    auto markOp = dyn_cast<annotation::MarkOp>(op);
    if (markOp) {
      IntegerAttr attr = markOp->getAttrOfType<IntegerAttr>(
          hfusion::InsertSliceSourceIndexAttr::name);
      if (attr) {
        int64_t result = attr.getInt();
        rewriter.eraseOp(markOp);
        return result;
      }
    }

    for (Operation *user : op->getUsers()) {
      auto tracedIndex = traceInsertSliceSourceIndex(user, rewriter);
      if (tracedIndex.has_value()) {
        return tracedIndex;
      }
    }
    return std::nullopt;
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