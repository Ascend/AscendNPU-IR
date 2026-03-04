//===-TorchPipelines.cpp ------ BiShengIR Torch Pipelines --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TorchToHFusion/TorchToHFusion.h"
#include "bishengir/Conversion/TorchToSymbol/TorchToSymbol.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Torch/Pipelines/Passes.h"
#include "bishengir/Dialect/Torch/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "torch-mlir/Conversion/TorchConversionToMLProgram/TorchConversionToMLProgram.h"
#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Conversion/TorchToTMTensor/TorchToTMTensor.h"
#include "torch-mlir/Conversion/TorchToTensor/TorchToTensor.h"
#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
#include "stablehlo/transforms/Passes.h"
#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"
#endif

using namespace mlir;
using namespace mlir::torch;
//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace reg {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h.inc"
} // end namespace reg

void bishengir::createTorchBackendToNamedOpBackendPipeline(
    OpPassManager &pm, const TorchToNamedOpPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(Torch::createFuseQuantizedOpsPass());
  pm.addNestedPass<func::FuncOp>(Torch::createScalarizeShapesPass());

  pm.addNestedPass<func::FuncOp>(createConvertTorchToTMTensorPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(createLiteralDataTypeCastPass());
  ConvertTorchToHFusionOptions torchToHFusionOption;
  torchToHFusionOption.ensureNoImplicitBroadcast =
      options.ensureNoImplicitBroadcast;
  pm.addNestedPass<func::FuncOp>(createConvertTorchToSymbolPass());
  pm.addNestedPass<func::FuncOp>(
      createConvertTorchToHFusionPass(torchToHFusionOption));
  pm.addNestedPass<func::FuncOp>(createConvertTorchToLinalgPass());

  // NOTE: Upstream's TorchToLinalg conversion has limitations and sometimes
  // generates tensor.reshape ops, which are unsupported by downstream passes.
  // Normalize them ASAP, otherwise some canonicalization patterns might make
  // them more difficult to optimize.
  pm.nest<func::FuncOp>().addPass(
      tensor::createCanonicalizeTensorReshapePass());

  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(createConvertTorchToSCFPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToArithPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToTensorPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());

  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      memref::createResolveShapedTypeResultDimsPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  pm.addPass(TorchConversion::createFuncBackendTypeConversionPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      TorchConversion::createFinalizingBackendTypeConversionPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void bishengir::registerTorchToHFusionPipelines() {
  reg::registerPasses();
  mlir::PassPipelineRegistration<TorchToNamedOpPipelineOptions>(
      "torch-backend-to-named-op-backend-pipeline",
      "Pipeline lowering torch backend contract to linalg-on-tensors backend "
      "contract.",
      [](OpPassManager &pm, const TorchToNamedOpPipelineOptions &options) {
        bishengir::createTorchBackendToNamedOpBackendPipeline(pm, options);
      });
}