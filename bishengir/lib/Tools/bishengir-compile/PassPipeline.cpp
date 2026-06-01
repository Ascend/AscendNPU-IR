//===- PassPipeline.cpp - BiShengIR pass pipeline -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/bishengir-compile/PassPipeline.h"

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Conversion/HIVMToStandard/HIVMToStandard.h"
#include "bishengir/Conversion/HIVMAVEToStandard/HIVMAVEToStandard.h"
#include "bishengir/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.h"
#include "bishengir/Dialect/AscendDPX/Transforms/Passes.h"
#include "bishengir/Dialect/Annotation/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Pipelines/Passes.h"
#include "bishengir/Dialect/HACC/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Pipelines/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/Pipelines/Passes.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "bishengir/Dialect/AscendDPX/Transforms/Passes.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Triton/Pipelines/Passes.h"
#include "bishengir/ExecutionEngine/Passes.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "bishengir/Transforms/InjectIRInstrumentation.h"
#include "bishengir/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToEmitC/ArithToEmitCPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <set>

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
#include "bishengir/Dialect/Torch/Pipelines/Passes.h"
#endif

#include "mlir/CAPI/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToEmitC/ArithToEmitCPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace bishengir {

#if BISHENGIR_ENABLE_TRITON_COMPILE
/// Defined in BiShengIRCompileConfig.cpp.
const mlir::triton::proton::ConvertProtonToProtonGPUOptions &
getProtonGPUCompileConfig();
#endif

// Helper function to set up HFusionPipelineOptions
void setupHFusionPipelineOptions(
    hfusion::HFusionPipelineOptions &hfusionPipelineOptions,
    const BiShengIRCompileMainConfig &config) {
  auto &options = hfusionPipelineOptions;
#define GEN_HFUSION_OPTION_SETUP
#include "bishengir/Tools/bishengir-compile/ConfigUtils.cpp.inc"
  hfusionPipelineOptions.insertFFTS =
      !hfusionPipelineOptions.disableFFTS &&
      hacc::utils::isFFTSSupportedArch(config.getTarget());
  hfusionPipelineOptions.target =
      hacc::stringifyTargetDeviceEnum(config.getTarget());
}

void setupHIVMPipelineOptions(hivm::HIVMPipelineOptions &hivmPipelineOptions,
                              const BiShengIRCompileMainConfig &config) {
  auto &options = hivmPipelineOptions;
#define GEN_HIVM_OPTION_SETUP
#include "bishengir/Tools/bishengir-compile/ConfigUtils.cpp.inc"
  hivmPipelineOptions.target =
      hacc::stringifyTargetDeviceEnum(config.getTarget());
  // UB-aware fusion splits groups to avoid overflow; disable later VF merging
  // so the split is preserved through the HIVM pipeline.
  if (config.isUBAwareVfFusion() && hivmPipelineOptions.enableVfMergeLevel > 0)
    hivmPipelineOptions.enableVfMergeLevel = 0;
}

void setupHIVMAVEPipelineOptions(
    hivmave::HIVMAVEPipelineOptions &hivmAVEPipelineOptions,
    const BiShengIRCompileMainConfig &config) {
  hivmAVEPipelineOptions.enableTritonKernelCompile =
      config.getEnableTritonKernelCompile();
  hivmAVEPipelineOptions.enableMixedCV = config.shouldEnableMixedCV();
  hivmAVEPipelineOptions.enableLayoutOptimization =
      config.shouldEnableLayoutOptimization();
  hivmAVEPipelineOptions.simtVFDynamicSize = config.getSimtVFDynamicSize();
  hivmAVEPipelineOptions.enableAutoBlockifyLoop =
      config.getEnableAutoBlockifyLoop();
  hivmAVEPipelineOptions.enableAutoMultiBuffer =
      config.getEnableAutoMultiBuffer();
  hivmAVEPipelineOptions.limitAutoMultiBufferOnlyForLocalBuffer =
      config.getLimitAutoMultiBufferOnlyForLocalBuffer();
  hivmAVEPipelineOptions.limitAutoMultiBufferOfLocalBuffer =
      config.getLimitAutoMultiBufferOfLocalBuffer();
  hivmAVEPipelineOptions.limitMixAutoMultiBufferBuffer =
      config.getLimitAutoMultiBufferBuffer();
  hivmAVEPipelineOptions.disableMultiBufferOnUB =
      config.getDisableMultiBufferOnUB();
  hivmAVEPipelineOptions.disableMultiBufferOnL0C =
      config.getDisableMultiBufferOnL0C();
  hivmAVEPipelineOptions.disableMultiBufferOnL1 =
      config.getDisableMultiBufferOnL1();
  hivmAVEPipelineOptions.enableAutoBindSubBlock =
      config.getEnableAutoBindSubBlock();
  hivmAVEPipelineOptions.enableAutoStorageAlign =
      config.getEnableHIVMAutoStorageAlign();
  hivmAVEPipelineOptions.enableGlobalWorkspaceReuse =
      config.getEnableHIVMGlobalWorkspaceReuse();
  hivmAVEPipelineOptions.enableHIVMInjectBarrierAllSync =
      config.getEnableHIVMInjectBarrierAllSync();
  hivmAVEPipelineOptions.workspaceMultiBufferNum =
      config.getSetWorkspaceMultibuffer();
  hivmAVEPipelineOptions.enableAutoCVBalance =
      config.getEnableHIVMAutoCVBalance();
  hivmAVEPipelineOptions.enableInjectBlockAllSync =
      config.getEnableHIVMInjectBlockAllSync();
  hivmAVEPipelineOptions.disableAutoInjectBlockSync =
      config.getDisableAutoInjectBlockSync();
  hivmAVEPipelineOptions.enableHIVMGraphSyncSolver =
      config.getEnableHIVMGraphSyncSolver();
  hivmAVEPipelineOptions.enableUnitFlagSync =
      config.getEnableHIVMUnitFlagSync();
  hivmAVEPipelineOptions.enableCodeMotion = config.getEnableCodeMotion();
  hivmAVEPipelineOptions.target =
      hacc::stringifyTargetDeviceEnum(config.getTarget());
  hivmAVEPipelineOptions.enableVfMergeLevel = config.getEnableVfMergeLevel();
  hivmAVEPipelineOptions.useDPX = config.getUseDPX();
  hivmAVEPipelineOptions.enableND2NZOnVector =
      config.getEnableHivmNd2nzOnVector();
  hivmAVEPipelineOptions.enableFusedMultiplyAdd =
      config.getEnableFusedMultiplyAdd();
  hivmAVEPipelineOptions.enablePrintMemoryAllocatedSize =
      config.getEnablePrintMemoryAllocatedSize();
  hivmAVEPipelineOptions.maxReductionSplitNum =
      config.getMaxReductionSplit();
}

void buildSIMTPipeline(OpPassManager &pm, const BiShengIRCompileMainConfig &config) {
  pm.addPass(createCSEPass());
  pm.addPass(createSCCPPass());
  auto tritonGridDim = config.getSimtTritonGrid();
  bishengir::TritonRemapOptions options;
  if (!tritonGridDim.empty()) {
    options.gridDimX = static_cast<int>(tritonGridDim[0]);
    options.useGridFlag = true;
  }
  if (tritonGridDim.size() > 1)
    options.gridDimY = static_cast<int>(tritonGridDim[1]);

  if (tritonGridDim.size() > 2)
    options.gridDimZ = static_cast<int>(tritonGridDim[2]);

  // TODO: When DPX covers all remapper features correctly, remove
  // createTritonRemapPass completely.
  if (!config.getUseDPX())
    pm.addPass(bishengir::triton::createTritonRemapPass(options));
  CanonicalizerOptions canonicalizerOptions;
  pm.addPass(createCanonicalizerPass(canonicalizerOptions));
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass(canonicalizerOptions));
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());

  buildLowerToLLVMPipeline(pm, config);
}

void buildBiShengHIRAVEToLLVMPipeline(
    OpPassManager &pm, const BiShengIRCompileMainConfig &config) {
  if (config.getCompileHost()) {
    hacc::buildLowerHACCToLLVMPipeline(pm, config.getHostOutputFile());
    return;
  }

  if (config.getEnableHIVMCompile()) {
    hivmave::HIVMAVEPipelineOptions hivmAVEPipelineOptions;
    setupHIVMAVEPipelineOptions(hivmAVEPipelineOptions, config);
    hivmave::buildLowerAVEPipelines(pm, hivmAVEPipelineOptions);
  }

  if (config.getLowerToLLVM()) {
    buildLowerToLLVMPipeline(pm, config);
  }
}

/// Build the pipeline to lower BiShengHIR to LLVM Dialect IR
void buildLowerToLLVMPipeline(OpPassManager &pm,
                              const BiShengIRCompileMainConfig &config) {

  const bool ascendDebugPrint =
      StringRef(getenv("ASCEND_DEBUG_PRINT")) == "ALL";
  if (ascendDebugPrint) {
    DebugMemoryOptions debugMemoryOptions;
    pm.addPass(createDebugMemoryPass(debugMemoryOptions));
  }

  pm.addPass(annotation::createAnnotationLoweringPass());
  pm.addPass(hivm::createAllocToAllocaPass());

  // TODO: How does host/device separation compilation flow for triton
  //       compilation look like?
  // if (config.shouldCompileTriton())
  //   pm.nest<func::FuncOp>().addPass(hivm::createInsertInferVFModeFuncPass());

  pm.nest<func::FuncOp>().addPass(
      hivm::createInsertInitAndFinishForDebugPass());
  ConvertHIVMToStandardOptions hivmToStdOptions;
  hivmToStdOptions.isOpsAligned = config.getEnableHIVMAutoStorageAlign();
  pm.addPass(hivm::createMarkDisableLoadPass());
  pm.addPass(createConvertHIVMToStandardPass(hivmToStdOptions));
  pm.addPass(createConvertHIVMAVEToStandardPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createConvertHIVMAVEToAVEIntrinPass());
  pm.addPass(hivmave::createHoistVstasPass());
  if (config.getPureSimt() && config.getUseDPX()) {
    auto tritonGridDim = config.getSimtTritonGrid();
    bishengir::TritonRemapOptions options;
    options.isSimdSimtMixCompile = config.getEnableSimdSimtMixCompile();
    if (!tritonGridDim.empty()) {
      options.gridDimX = static_cast<int>(tritonGridDim[0]);
      options.useGridFlag = true;
    }
    if (tritonGridDim.size() > 1)
      options.gridDimY = static_cast<int>(tritonGridDim[1]);

    if (tritonGridDim.size() > 2)
      options.gridDimZ = static_cast<int>(tritonGridDim[2]);
    pm.addPass(bishengir::triton::createAdaptGPUKernelPass(options));
    pm.addPass(mlir::ascend_dpx::createHoistCallScalarToCallerPass());
    pm.addPass(mlir::ascend_dpx::createDPXDivOptimizationPass(options));
  }
  pm.addPass(createConvertAscendDPXToHIVMRegbaseIntrinPass());
  pm.addPass(bishengir::triton::createDecomposeFRemPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(arith::createArithExpandOpsPass());
}

static void buildDelayedHFusionRegBaseVectorizePipeline(
    mlir::OpPassManager &pm, const BiShengIRCompileMainConfig &config,
    bool shouldInferFuncCoreType = true) {
  if (config.getDisableHfusionVectorize()) {
    return;
  }
  // inferMixedCV populates enableMixedCV before this delayed HFusion pipeline
  // is built; adjust only the local HFusion options consumed by flatten.
  BiShengIRCompileMainConfig hfusionConfig = config;
  auto &registeredOptions = llvm::cl::getRegisteredOptions();
  auto enableFlattenOpt = registeredOptions.find("enable-flatten");
  bool hasExplicitEnableFlatten =
      enableFlattenOpt != registeredOptions.end() &&
      enableFlattenOpt->second->getNumOccurrences() != 0;
  if (hfusionConfig.shouldEnableMixedCV() && !hasExplicitEnableFlatten) {
    hfusionConfig.setEnableFlatten(false);
  }

  hfusion::HFusionPipelineOptions hfusionPipelineOptions;
  setupHFusionPipelineOptions(hfusionPipelineOptions, hfusionConfig);
  ExecutionEngineHIVMToUpstreamConversionOptions upstreamOptions;
  upstreamOptions.convertToNamedOp = 
      hacc::utils::isRegBasedArch(config.getTarget());
  pm.addPass(mlir::execution_engine::createConvertHIVMToUpstreamPass(upstreamOptions));
  hfusion::buildHFusionRegBasePipeline(pm, hfusionPipelineOptions);
  if (shouldInferFuncCoreType) {
    pm.addPass(mlir::hivm::createInferFuncCoreTypePass());
  }

  ConvertHFusionToHIVMOptions hfs2hivmOptions;
  hfs2hivmOptions.mmMapMode =
      config.getEnableTritonKernelCompile() ? hfusion::MmMapMode::MacroInstr
                                            : hfusion::MmMapMode::CoreOp;
  pm.addPass(createHFusionToHIVMConversionPass(hfs2hivmOptions));
}

void buildFinalHIVMPipelines(mlir::OpPassManager &pm,
                             const BiShengIRCompileMainConfig &config) {
  if (config.getEnableHIVMCompile()) {
    hivm::HIVMPipelineOptions hivmPipelineOptions;
    setupHIVMPipelineOptions(hivmPipelineOptions, config);
    if (config.getEnableSimdSimtMixCompile()) {
      buildDelayedHFusionRegBaseVectorizePipeline(
          pm, config, /*shouldInferFuncCoreType=*/true);
    }
    hivm::buildLowerHIVMPipelines(pm, hivmPipelineOptions);
  }
}

#if BISHENGIR_ENABLE_TRITON_COMPILE
// Helper function to set up LowerTritonPipelineOptions
void setupLowerTritonPipelineOptions(
    triton::LowerTritonPipelineOptions &options,
    const BiShengIRCompileMainConfig &config) {
  options.numWarps = config.getNumWarps();
  options.threadsPerWarp = config.getThreadsPerWarp();
  options.enableSIMTFastDiv = config.getEnableSIMTFastDiv();
  options.useDPX = config.getUseDPX();
  options.disableDecomposeReduction = config.getDisableDecomposeReduction();
  options.disableReorderInstruction = config.getDisableReorderInstruction();
  options.enableSinkDPXLoad = config.getEnableSinkDPXLoad();
  options.tritonMetadataOutput = config.getTritonMetadataOutput();
  options.enableSIMTAutoBlockify = config.getEnableAutoBlockifyLoop();
#if BSPUB_DAVINCI_BISHENGIR
  if (config.getSharedMemDynamicSize() < 122880 ||
      config.getSharedMemDynamicSize() > 221184)
    llvm::report_fatal_error(
        "shared-mem-dynamic-size should range from 122880 to 221184.");
  // max size of shared memory available for simt vf.
  options.sharedDynamicSize = config.getSharedMemDynamicSize();
  // encode our own compile optimization
  options.enableBishengirSimtOptimization =
      config.getEnableBishengirSimtOptimization();
  options.enableSimtReorderInstruction = config.getEnableSimtReorderInstruction();
#endif
#if BISHENGIR_ENABLE_TRITON_COMPILE
  options.protonGPUCompileConfig = getProtonGPUCompileConfig();
#endif
}

void buildBiShengTTIRPipeline(OpPassManager &pm,
                              const BiShengIRCompileMainConfig &config) {
  if (config.getEnableSimdSimtMixCompile()) {
    // Materialize SIMT mem scopes only after split so the main module can stay
    // free of address-spaced memrefs before delayed reg-based vectorization.
    pm.addPass(hivm::createMaterializeSimtVFMemScopePass());    
    pm.addPass(createHIVMToTritonGPUConversionPass());
  }

  if (!config.getCompileHost()) {
    pm.addPass(hacc::createAppendDeviceSpecPass(
        hacc::AppendTargetDeviceSpecOptions{config.getTarget()}));
  }
  pm.addPass(createCanonicalizeModulePass());
  triton::LowerTritonPipelineOptions lowerTritonPipelineOptions;
  setupLowerTritonPipelineOptions(lowerTritonPipelineOptions, config);
  bishengir::triton::buildLowerTritonPipeline(pm, lowerTritonPipelineOptions);
}
#endif

void buildBiShengHIRFinishPipeline(mlir::OpPassManager &pm,
                                   const BiShengIRCompileMainConfig &config) {
  pm.addPass(hivm::createWriteBackSharedPass());
}

void buildBiShengHIRPipeline(OpPassManager &pm,
                             const BiShengIRCompileMainConfig &config) {
  if (!config.getCompileHost()) {
    pm.addPass(hacc::createAppendDeviceSpecPass(
        hacc::AppendTargetDeviceSpecOptions{config.getTarget()}));
  }

  pm.addPass(createCanonicalizeModulePass());
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
  if (config.getEnableTorchCompile()) {
    TorchToNamedOpPipelineOptions torchToNamedOpOptions;
    torchToNamedOpOptions.ensureNoImplicitBroadcast =
        config.getEnsureNoImplicitBroadcast();
    createTorchBackendToNamedOpBackendPipeline(pm, torchToNamedOpOptions);
  }
#endif

  hfusion::HFusionPipelineOptions hfusionPipelineOptions;
  if (config.getEnableHfusionCompile()) {
    setupHFusionPipelineOptions(hfusionPipelineOptions, config);
    if (config.getEnableSimdSimtMixCompile()) {
      // Delay reg-based vectorization until SIMT code is split out and we can
      // re-run it only on the main module.
      hfusionPipelineOptions.disableHfusionVectorize = true;
    }
    hfusion::buildHFusionPipelines(pm, hfusionPipelineOptions);
  }

  if (config.getEnableHIVMCompile()) {
    // Build convert to HIVM Dialect pipeline.
    hivm::ConvertToHIVMPipelineOptions convertToHIVMOptions;
    convertToHIVMOptions.enableTritonKernelCompile =
        config.getEnableTritonKernelCompile();
    convertToHIVMOptions.enableRegBaseHIVMPipe =
        hacc::utils::isRegBasedArch(config.getTarget());
    hivm::HIVMPipelineOptions hivmPipelineOptions;
    setupHIVMPipelineOptions(hivmPipelineOptions, config);
    hivm::buildConvertToHIVMPipeline(pm, convertToHIVMOptions);
    hivm::buildHIVMTensorOptimizations(pm, hivmPipelineOptions);
    if (config.shouldEnableMixedCV()) {
      HIVMAggregatedDecomposeOpOptions decomposeOption;
      decomposeOption.decomposePhase = bishengir::DecomposePhase::NO_CONSTRAINT;
      pm.nest<func::FuncOp>().addPass(
          mlir::hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));
      // delay vectorization after split simd/simt
      if (!config.getEnableSimdSimtMixCompile())
        buildDelayedHFusionRegBaseVectorizePipeline(
            pm, config, /*shouldInferFuncCoreType=*/true);
    }
    if (config.getEnableSimdSimtMixCompile()) {
      pm.addPass(hivm::createAutoScopePass());
      pm.addPass(hivm::createInsertMemSemanticForSimtVFPass());
      pm.addPass(scope::createOutlineScopePass());
      pm.addPass(hivm::createInsertAllocBasePlaceholderPass());
      pm.addPass(hivm::createInferSimtVFMemEffectPass());
      // Infer per-argument mem scope hints from the mixed call boundary first;
      // actual address space rewrites are deferred until each SIMT module is
      // split out and lowered independently.
      pm.addPass(hivm::createInferSimtVFMemScopeHintPass());
      pm.addPass(hivm::createSplitSimtModulePass());
    }
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
    // Use generated metadata from Options.td to map pass options back to the
    // compile config.
#define GEN_PASS_OPTION_TO_CONFIG
#include "bishengir/Tools/bishengir-compile/ConfigUtils.cpp.inc"
    config.setOutputFile(outputFile);
    std::string arg;
    std::vector<std::string> args;
    std::set<std::string> skip = {" ", "{", "}", getArgument().str()};
    auto callback = [&arg, &args, skip](MlirStringRef str, void *data) {
      std::string sData = std::string(str.data, str.data + str.length);
      if (skip.count(sData) != 0) {
        if (!arg.empty() && (arg[0] != 'o' || arg[1] != '=')) {
          args.push_back("-" + arg);
        }
        arg.clear();
      } else {
        arg += sData;
      }
    };

    mlir::detail::CallbackOstream stream(callback, nullptr);
    this->printAsTextualPipeline(stream);
    config.setClArgs(args);
    BiShengIRCompileMainConfig::collectHIVMCArgs(config);

    if (failed(runBiShengIRPipeline(moduleOp, config))) {
      signalPassFailure();
    }
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
