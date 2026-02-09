//===- PassPipeline.cpp - BiShengIR pass pipeline -------------------------===//
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

#include "bishengir/Tools/bishengir-compile/PassPipeline.h"
#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/HACC/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/ConvertToHIVMPipeline.h"
#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Tools/bishengir-hivm-compile/BiShengIRHIVMCompile.h"
#include "bishengir/Transforms/Passes.h"

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
#include "bishengir/Dialect/Torch/Pipelines/Passes.h"
#endif

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace bishengir {

// Helper function to set up HFusionPipelineOptions
void setupHFusionPipelineOptions(hfusion::HFusionPipelineOptions &options,
                                 const BiShengIRCompileMainConfig &config) {
#define GEN_HFUSION_OPTION_SETUP
#include "bishengir/Tools/bishengir-compile/ConfigUtils.cpp.inc"
}

void setupHIVMPipelineOptions(hivm::HIVMPipelineOptions &options,
                              const BiShengIRCompileMainConfig &config) {
// TODO: Delete bishengir/Tools/bishengir-hivm-compile
#define GEN_HIVM_OPTION_SETUP
#include "bishengir/Tools/bishengir-hivm-compile/HIVMConfigUtils.cpp.inc"
}

void buildBiShengHIRPipeline(OpPassManager &pm,
                             const BiShengIRCompileMainConfig &config) {
  pm.addPass(createCanonicalizeModulePass());
  pm.addPass(hacc::createAppendDeviceSpecPass(
      hacc::AppendTargetDeviceSpecOptions{config.getTarget()}));

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
  if (config.getEnableTorchCompile()) {
    TorchToNamedOpPipelineOptions torchToNamedOpOptions;
    torchToNamedOpOptions.ensureNoImplicitBroadcast =
        config.getEnsureNoImplicitBroadcast();
    createTorchBackendToNamedOpBackendPipeline(pm, torchToNamedOpOptions);
  }
#endif

  if (config.getEnableHfusionCompile()) {
    hfusion::HFusionPipelineOptions hfusionPipelineOptions;
    setupHFusionPipelineOptions(hfusionPipelineOptions, config);
    hfusion::buildHFusionPipelines(pm, hfusionPipelineOptions);
  }

  if (config.getEnableHIVMCompile()) {
    // Build convert to HIVM Dialect pipeline.
    hivm::ConvertToHIVMPipelineOptions convertToHIVMOptions;
    convertToHIVMOptions.enableTritonKernelCompile =
        config.getEnableTritonKernelCompile();
    convertToHIVMOptions.enableUbufSaving = config.getEnableUbufSaving();
    hivm::buildConvertToHIVMPipeline(pm, convertToHIVMOptions);
    hivm::HIVMPipelineOptions options;
    setupHIVMPipelineOptions(options, config);
    hivm::buildOptimizeHIVMPipeline(pm, options);
  }
}

/// We define the BiShengIR Compile Pass here not in a tablegen file because
/// there potentially many options that are controlled by cmake options, and
/// it's more flexible to define in cpp.
struct BiShengIRCompilePass
    : public PassWrapper<BiShengIRCompilePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BiShengIRCompilePass)
  BiShengIRCompilePass() = default;
  BiShengIRCompilePass &operator=(const BiShengIRCompilePass &pass) = delete;
  BiShengIRCompilePass(const BiShengIRCompilePass &pass)
      : PassWrapper<BiShengIRCompilePass, OperationPass<ModuleOp>>(pass) {}
  StringRef getArgument() const override { return "bishengir-compile"; }
  StringRef getDescription() const override {
    return "Compile BiShengIR module to binary.";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    BiShengIRCompileMainConfig config;
    // Use fluent API to set the pass option into config.

    // Feature control options
    config
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
        .setEnableTorchCompile(enableTorchCompile)
        .setEnsureNoImplicitBroadcast(ensureNoImplicitBroadcast)
#endif
        .setEnableTritonKernelCompile(enableTritonKernelCompile)
        .setEnableHfusionCompile(enableHfusionCompile)
        .setEnableHIVMCompile(enableHIVMCompile)
        .setEnableManageHostResources(enableManageHostResources)
        .setEnableSymbolAnalysis(enableSymbolAnalysis)
        .setEnableMultiKernel(enableMultiKernel);

    // DFX control options
    config.setEnableSanitizer(enableSanitizer)
        .setEnableDebugInfo(enableDebugInfo);

    // Output setting options
    config.setOutputFile(outputFile);

    // General optimization control options
    config.setEnableAutoMultiBuffer(enableAutoMultiBuffer)
        .setEnableDeterministicComputing(enableDeterministicComputing)
        .setEnableOpsReorder(enableOpsReorder)
        .setEnableTuningMode(enableTuningMode)
        .setBlockDim(blockDim);

    // HFusion optimization control options
    config.setHfusionMaxHorizontalFusionSize(hfusionMaxHorizontalFusionSize)
        .setHfusionMaxBufferCountTuning(hfusionMaxBufferCountTuning)
        .setenableHfusionCountBufferDmaOpt(enableHfusionCountBufferDmaOpt)
        .setCubeTilingTuning(cubeTilingTuning);

    SmallVector<Pass::Option<bool> *> sharedWithHIVMCompileBool = {
      &enableAutoBindSubBlock,
      &enableAutoBlockifyLoop,
      &enableHIVMAutoCVBalance,
      &enableAutoMultiBuffer,
      &enableHIVMAutoStorageAlign,
      &enableBinRelocation,
      &enableCodeMotion,
#if (!BISHENGIR_PUBLISH)
      &enableCpuTraceIntrinsic,
      &enableLIRCompile,
#endif
      &enableDebugInfo,
      &enableHIVMGlobalWorkspaceReuse,
      &enableHIVMCompile,
      &enableHIVMInjectBarrierAllSync,
      &enableHIVMInjectBlockAllSync,
      &enableHivmNd2nzOnVector,
      &enableSanitizer,
      &enableStaticBarePtr,
      &enableTritonKernelCompile,
      &enableHIVMUnitFlagSync,
      &enableHIVMAssumeAliveLoops,
      &enableUbufSaving,
    };

    SmallVector<Pass::Option<unsigned> *> sharedWithHIVMCompileUnsigned = {
        &setWorkspaceMultibuffer,
        &tileMixVectorLoop,
        &tileMixCubeLoop,
    };

    SmallVector<Pass::Option<MultiBufferStrategy> *>
        sharedWithHIVMCompileMultiBuffer = {
            &limitAutoMultiBufferOfLocalBuffer,
            &limitAutoMultiBufferBuffer,
        };

    SmallVector<Pass::Option<mlir::hacc::TargetDevice> *>
        sharedWithHIVMCompileTargetDevice = {
            &target,
        };

    std::vector<std::string> collectedArgs;
    for (auto &opt : sharedWithHIVMCompileBool) {
      std::string arg =
          opt->getArgStr().str() + "=" + (opt->getValue() ? "true" : "false");
      collectedArgs.push_back(arg);
    }
    for (auto &opt : sharedWithHIVMCompileUnsigned) {
      std::string arg =
          opt->getArgStr().str() + "=" + std::to_string(opt->getValue());
      collectedArgs.push_back(arg);
    }
    const std::map<MultiBufferStrategy, std::string> multibufferStrategy2str = {
        {MultiBufferStrategy::NO_LIMIT, "no-limit"},
        {MultiBufferStrategy::ONLY_CUBE, "only-cube"},
        {MultiBufferStrategy::ONLY_VECTOR, "only-vector"},
        {MultiBufferStrategy::CUBE_NO_L0C, "no-l0c"},
    };
    for (auto &opt : sharedWithHIVMCompileMultiBuffer) {
      std::string arg = opt->getArgStr().str() + "=" +
                        multibufferStrategy2str.at(opt->getValue());
      collectedArgs.push_back(arg);
    }
    for (auto &opt : sharedWithHIVMCompileTargetDevice) {
      std::string arg = opt->getArgStr().str() + "=" +
                        hacc::stringifyTargetDeviceEnum(opt->getValue()).str();
      collectedArgs.push_back(arg);
    }
    for (const std::string &arg : this->hIVMCArgs) {
      collectedArgs.push_back(arg);
    }

    config.setHIVMCArgs(collectedArgs);
    auto cloned = moduleOp.clone();
    auto res = runBiShengIRPipeline(cloned, config);
    if (failed(res)) {
      signalPassFailure();
    }
    IRMapping mapper;
    auto blocks =
        llvm::map_to_vector(moduleOp.getBodyRegion().getBlocks(),
                            [](Block &block) -> Block * { return &block; });
    for (auto &block : blocks) {
      block->erase();
    }
    res->get().getBodyRegion().cloneInto(&moduleOp.getBodyRegion(), mapper);
    moduleOp->setAttrs(res->get()->getAttrs());
  }

protected:
#define GEN_ALL_OPTION_REGISTRATION
#include "bishengir/Tools/bishengir-compile/PassOptions.cpp.inc"

  Pass::Option<std::string> outputFile{
      *this, "o", llvm::cl::desc("Specify output bin name"),
      llvm::cl::init("-")};
};

} // namespace bishengir

void bishengir::registerBiShengIRCompilePass() {
  PassRegistration<bishengir::BiShengIRCompilePass>();
}
