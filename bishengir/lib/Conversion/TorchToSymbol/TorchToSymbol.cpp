//===--TorchToSymbol.cpp - Torch to Symbol Dialect Conversion ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TorchToSymbol/TorchToSymbol.h"
#include "bishengir/Dialect/Symbol/IR/Symbol.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTORCHTOSYMBOL
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

//===----------------------------------------------------------------------===//
// Torch SymbolicIntOp to Symbol SymbolicIntOp
//===----------------------------------------------------------------------===//

class TorchSymbolicIntConversionPattern
    : public OpConversionPattern<Torch::SymbolicIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = Torch::SymbolicIntOpAdaptor;
  LogicalResult
  matchAndRewrite(Torch::SymbolicIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: use symbol manager to get a unique symbol name instead of reusing
    // the torch symbol
    auto symbolRef = FlatSymbolRefAttr::get(rewriter.getContext(),
                                            op.getSymbolNameAttr().str());
    auto newOp = rewriter.create<symbol::SymbolicIntOp>(
        op.getLoc(), rewriter.getIndexType(), symbolRef, op.getMinVal(),
        op.getMaxVal());

    // Materialize torch int to index
    const TypeConverter *converter = this->getTypeConverter();
    if (!converter)
      return failure();

    auto materializedValue = converter->materializeSourceConversion(
        rewriter, op->getLoc(), op.getResult().getType(), newOp.getResult());
    rewriter.replaceOp(op, materializedValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Torch BindSymbolicShapeOp to Symbol BindSymbolicShapeOp
//===----------------------------------------------------------------------===//

class TorchBindSymbolicShapeConversionPattern
    : public OpConversionPattern<Torch::BindSymbolicShapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = Torch::BindSymbolicShapeOpAdaptor;
  LogicalResult
  matchAndRewrite(Torch::BindSymbolicShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check the tensor type can be converted correctly
    Value tensorToBind = op.getOperand();
    Type builtinTensorType =
        cast<Torch::ValueTensorType>(tensorToBind.getType()).toBuiltinTensor();
    if (!isa_and_nonnull<RankedTensorType>(builtinTensorType))
      return rewriter.notifyMatchFailure(op, "type cannot be lowered");

    // Materialize torch vtensor to ranked tensor
    const TypeConverter *converter = this->getTypeConverter();
    if (!converter)
      return failure();

    SmallVector<Value> materializedSymbols;
    materializedSymbols.reserve(op.getShapeSymbols().size());
    for (auto symbol : op.getShapeSymbols())
      materializedSymbols.push_back(converter->materializeTargetConversion(
          rewriter, op.getLoc(), rewriter.getIndexType(), symbol));

    Value materializedTensor = converter->materializeTargetConversion(
        rewriter, op->getLoc(), builtinTensorType, tensorToBind);
    rewriter.replaceOpWithNewOp<symbol::BindSymbolicShapeOp>(
        op, materializedTensor, materializedSymbols,
        adaptor.getShapeExpressions());
    return success();
  }
};

void mlir::symbol::populatePatternsAndLegality(TypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               ConversionTarget &target) {
  // Set up torch backend type converter
  typeConverter.addConversion([](Type type) { return type; });
  TorchConversion::setupBackendTypeConversion(target, typeConverter);

  target.addIllegalOp<Torch::SymbolicIntOp, Torch::BindSymbolicShapeOp>();
  target.addLegalDialect<arith::ArithDialect, symbol::SymbolDialect>();

  patterns.add<TorchSymbolicIntConversionPattern,
               TorchBindSymbolicShapeConversionPattern>(typeConverter,
                                                        patterns.getContext());
}

namespace {
class ConvertTorchToSymbol
    : public impl::ConvertTorchToSymbolBase<ConvertTorchToSymbol> {
public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};
} // namespace

void ConvertTorchToSymbol::getDependentDialects(
    DialectRegistry &registry) const {
  TorchConversion::getBackendTypeConversionDependentDialects(registry);
  impl::ConvertTorchToSymbolBase<ConvertTorchToSymbol>::getDependentDialects(
      registry);
}

void ConvertTorchToSymbol::runOnOperation() {
  MLIRContext *context = &getContext();
  ConversionTarget target(*context);
  RewritePatternSet patterns(context);
  TypeConverter typeConverter;
  mlir::symbol::populatePatternsAndLegality(typeConverter, patterns, target);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertTorchToSymbolPass() {
  return std::make_unique<ConvertTorchToSymbol>();
}