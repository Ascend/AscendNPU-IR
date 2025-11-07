//===-TorchPipelines.cpp ------ BiShengIR Torch Pipelines --------*- C++-*-===//
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

#include "bishengir/Conversion/TorchToHFusion/TorchToHFusion.h"
#include "bishengir/Conversion/TorchToSymbol/TorchToSymbol.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Torch/Pipelines/Passes.h"
#include "bishengir/Dialect/Torch/Transforms/Passes.h"
#include "bishengir/Transforms/Passes.h"

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
  CanonicalizerOptions canonicalizerOpts;
  SmallVector<std::string> disabledPatterns(
      {"ReinterpretCastConstantArgumentFolder"});
  canonicalizerOpts.disabledPatterns = disabledPatterns;
  pm.addNestedPass<func::FuncOp>(Torch::createFuseQuantizedOpsPass());
  pm.addNestedPass<func::FuncOp>(Torch::createScalarizeShapesPass());

  pm.addNestedPass<func::FuncOp>(createConvertTorchToTMTensorPass());
  pm.addNestedPass<func::FuncOp>(
      bishengir::createExtendedCanonicalizerPass(canonicalizerOpts));

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

  pm.addNestedPass<func::FuncOp>(
      bishengir::createExtendedCanonicalizerPass(canonicalizerOpts));

  pm.addNestedPass<func::FuncOp>(createConvertTorchToSCFPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToArithPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToTensorPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());

  pm.addNestedPass<func::FuncOp>(
      bishengir::createExtendedCanonicalizerPass(canonicalizerOpts));
  pm.addNestedPass<func::FuncOp>(
      memref::createResolveShapedTypeResultDimsPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  pm.addPass(TorchConversion::createFuncBackendTypeConversionPass());
  pm.addNestedPass<func::FuncOp>(
      bishengir::createExtendedCanonicalizerPass(canonicalizerOpts));
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