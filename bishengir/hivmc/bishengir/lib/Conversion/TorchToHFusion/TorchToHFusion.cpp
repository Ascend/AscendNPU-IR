//===- TorchToHFusion.cpp - Main entry point for Torch to HFusion ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/TorchToHFusion/TorchToHFusion.h"
#include "bishengir/Conversion/TorchToHFusion/PopulatePatterns.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTORCHTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::hfusion;

namespace {
class ConvertTorchToHFusion
    : public impl::ConvertTorchToHFusionBase<ConvertTorchToHFusion> {
public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
    impl::ConvertTorchToHFusionBase<
        ConvertTorchToHFusion>::getDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, func::FuncDialect,
                           cf::ControlFlowDialect, math::MathDialect,
                           scf::SCFDialect, sparse_tensor::SparseTensorDialect,
                           tensor::TensorDialect, arith::ArithDialect,
                           complex::ComplexDialect, hfusion::HFusionDialect>();
    target.addLegalOp<TorchConversion::GetNextSeedOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    ConvertTorchToHFusionOptions options;
    options.ensureNoImplicitBroadcast = ensureNoImplicitBroadcast;

    RewritePatternSet patterns(context);
    populateElementWisePatternsAndLegality(typeConverter, patterns, target);
    populateReductionPatternsAndLegality(typeConverter, patterns, target);
    populateDataMovementPatternsAndLegality(typeConverter, patterns, target,
                                            options);
    populateUncategorizedPatternsAndLegality(typeConverter, patterns, target);
    populateTensorConstructorsPatternsAndLegality(typeConverter, patterns,
                                                  target);

    (void)applyPartialConversion(getOperation(), target, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertTorchToHFusionPass() {
  return std::make_unique<ConvertTorchToHFusion>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertTorchToHFusionPass(
    const ConvertTorchToHFusionOptions &options) {
  return std::make_unique<ConvertTorchToHFusion>(options);
}
