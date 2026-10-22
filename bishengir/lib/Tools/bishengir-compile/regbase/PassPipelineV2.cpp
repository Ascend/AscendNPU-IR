//===- PassPipelineV2.cpp - v2 driver pipeline ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Composes the v2 pipeline for the compiler driver. The pass order is owned by
// bishengir/Dialect/HIVM/Pipelines/v2/HIVMV2Pipelines.cpp.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/bishengir-compile/regbase/PassPipelineV2.h"

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/HACC/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/regbase/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/v2/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/v2/TargetUtils.h"
#include "bishengir/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "llvm/Support/ErrorHandling.h"

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
#include "bishengir/Dialect/Torch/Pipelines/Passes.h"
#endif

using namespace mlir;

namespace bishengir {
namespace regbase {

// Defined in PassPipeline.cpp.
void setupHFusionPipelineOptions(hfusion::HFusionPipelineOptions &options,
                                 const BiShengIRCompileMainConfig &config);
void setupHIVMPipelineOptions(hivm::regbase::HIVMPipelineOptions &options,
                              const BiShengIRCompileMainConfig &config);

namespace {

/// Fill the shared v2 pipeline options for the two driver stages.
void setupV2PipelineOptions(
    const BiShengIRCompileMainConfig &config,
    const hivm::regbase::HIVMPipelineOptions &hivmOptions,
    hivm::v2::V2PipelineOptions &v2Options) {
  v2Options.target = hacc::stringifyTargetDeviceEnum(config.getTarget()).str();
  v2Options.enablePreload = hivmOptions.enablePreload;
  v2Options.enableMultiBuffer = config.getEnableAutoMultiBuffer();
  v2Options.enableCrossCoreGSS = config.getEnableHIVMGraphSyncSolver();
  v2Options.useBlockAllSync = config.getEnableHIVMInjectBlockAllSync();
  v2Options.setTargetSpec = false;
  v2Options.ghostSplitMix = false;
}

enum class V2Stage { Tensor, Final };

/// The v2 pipeline needs reg-based execution and on-chip cube/vector data
/// exchange. Reject unsupported targets here, before the driver builds the
/// stages, instead of failing later inside a stage builder.
void validateV2Target(const BiShengIRCompileMainConfig &config) {
  if (hivm::v2::isSupportedTarget(config.getTarget()))
    return;
  llvm::report_fatal_error(llvm::Twine("hivm-v2-pipeline: target '") +
                           hacc::stringifyTargetDeviceEnum(config.getTarget()) +
                           "' does not support the v2 pipeline");
}

void appendV2Pipeline(OpPassManager &pm,
                      const BiShengIRCompileMainConfig &config,
                      const hivm::regbase::HIVMPipelineOptions &hivmOptions,
                      V2Stage stage) {
  validateV2Target(config);
  hivm::v2::V2PipelineOptions v2Options;
  setupV2PipelineOptions(config, hivmOptions, v2Options);
  if (stage == V2Stage::Tensor) {
    hivm::v2::buildV2TensorPhase(pm, v2Options);
    hivm::v2::buildV2BufferizationPhase(pm, v2Options);
  } else {
    hivm::v2::buildV2PostBufferizationPhase(pm, v2Options);
    hivm::v2::buildV2PerCoreTail(pm, v2Options);
  }
}

} // namespace

void buildV2HIRPipeline(OpPassManager &pm,
                        const BiShengIRCompileMainConfig &config) {
  if (!config.getCompileHost()) {
    pm.addPass(
        hacc::createAppendDeviceSpecPass(hacc::AppendTargetDeviceSpecOptions{
            config.getTarget(), config.getCustomAICNumber(),
            config.getCustomAIVNumber(), config.getHIVMCVersion()}));
  }

  pm.addPass(createCanonicalizeModulePass());
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
  if (config.getEnableTorchCompile()) {
    auto torchOptions = std::make_unique<TorchToNamedOpPipelineOptions>();
    torchOptions->ensureNoImplicitBroadcast =
        config.getEnsureNoImplicitBroadcast();
    createTorchBackendToNamedOpBackendPipeline(pm, *torchOptions);
  }
#endif

  if (config.getEnableHfusionCompile()) {
    auto options = std::make_unique<hfusion::HFusionPipelineOptions>();
    setupHFusionPipelineOptions(*options, config);
    // v2 vectorizes in hivm-vectorize on the unsplit MIX.
    options->disableHfusionVectorize = true;
    hfusion::regbase::buildHFusionPipelines(pm, *options);
  }

  if (config.getEnableHIVMCompile()) {
    auto convertOptions =
        std::make_unique<hivm::regbase::ConvertToHIVMPipelineOptions>();
    convertOptions->enableTritonKernelCompile =
        config.getEnableTritonKernelCompile();
    convertOptions->enableRegBaseHIVMPipe =
        hacc::utils::isRegBasedArch(config.getTarget());
    auto options = std::make_unique<hivm::regbase::HIVMPipelineOptions>();
    setupHIVMPipelineOptions(*options, config);
    hivm::regbase::buildConvertToHIVMPipeline(pm, *convertOptions);
    appendV2Pipeline(pm, config, *options, V2Stage::Tensor);
  }
}

void buildV2FinalHIVMPipelines(OpPassManager &pm,
                               const BiShengIRCompileMainConfig &config) {
  auto options = std::make_unique<hivm::regbase::HIVMPipelineOptions>();
  setupHIVMPipelineOptions(*options, config);
  appendV2Pipeline(pm, config, *options, V2Stage::Final);
}

} // namespace regbase
} // namespace bishengir
