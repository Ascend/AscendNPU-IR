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
#include "bishengir/Dialect/Annotation/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Pipelines/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "bishengir/Dialect/Triton/Pipelines/Passes.h"
#include "bishengir/ExecutionEngine/Passes.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
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
#include "llvm/Support/Debug.h"

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

// Helper function to set up HFusionPipelineOptions
void setupHFusionPipelineOptions(
    hfusion::HFusionPipelineOptions &hfusionPipelineOptions,
    const BiShengIRCompileMainConfig &config) {
  hfusionPipelineOptions.enableManageHostResources =
      config.shouldManageHostResource();
  hfusionPipelineOptions.enableTritonKernelCompile =
      config.shouldCompileTriton();
  hfusionPipelineOptions.enableLayoutOptimization =
      config.shouldEnableLayoutOptimization();
  hfusionPipelineOptions.enableMixedCV = config.shouldEnableMixedCV();
  hfusionPipelineOptions.disableHFusionVectorize =
      config.shouldDisableHFusionVectorize();
  hfusionPipelineOptions.disableFFTS = config.shouldDisableFFTS();
  hfusionPipelineOptions.insertFFTS =
      !hfusionPipelineOptions.disableFFTS &&
      hacc::utils::isFFTSSupportedArch(config.getTargetBackend());
  hfusionPipelineOptions.blockDim = config.blockDim();
  hfusionPipelineOptions.maxHorizontalFusionSize =
      config.maxHorizontalFusionSize();
  hfusionPipelineOptions.maxFusedElementwiseOps =
      config.maxFusedElementwiseOps();
  hfusionPipelineOptions.enableDropUnitDims = config.shouldEnableDropUnitDims();
  hfusionPipelineOptions.enableFlatten = config.shouldEnableFlatten();
  hfusionPipelineOptions.enableAutoMultiBuffer =
      config.shouldEnableAutoMultiBuffer();
  hfusionPipelineOptions.enableDeterministicComputing =
      config.isDeterministicComputing();
  hfusionPipelineOptions.enableOpsReorder = config.shouldEnableOpsReorder();
  hfusionPipelineOptions.maxBufferCntTuning = config.maxBufferCountTuning();
  hfusionPipelineOptions.enableMultiKernel = config.shouldEnableMultiKernel();
  hfusionPipelineOptions.enableSymbolAnalysis =
      config.shouldEnableSymbolAnalysis();
  hfusionPipelineOptions.enableAutoVectorizeV2 =
      config.shouldEnableAutoVectorizeV2();
  hfusionPipelineOptions.enableVFFusion = config.shouldEnableVFFusion();
  hfusionPipelineOptions.enableTreeReduce = config.shouldEnableTreeReduce();
  hfusionPipelineOptions.skipScope = config.shouldSkipScope();
  hfusionPipelineOptions.enableCountBufferDmaOpt =
      config.shouldEnableCountBufferDmaOpt();
  hfusionPipelineOptions.cubeTilingTuning = config.cubeTilingTuningParams();
  hfusionPipelineOptions.target =
      hacc::stringifyTargetDeviceEnum(config.getTargetBackend());
  hfusionPipelineOptions.enableHighPrecision =
      config.shouldEnableHighPrecision();
  hfusionPipelineOptions.injectIrFromFile = config.getInjectIrFromFile();
}

void setupHIVMPipelineOptions(hivm::HIVMPipelineOptions &hivmPipelineOptions,
                              const BiShengIRCompileMainConfig &config) {
  hivmPipelineOptions.enableTritonKernelCompile = config.shouldCompileTriton();
  hivmPipelineOptions.enableLayoutOptimization =
      config.shouldEnableLayoutOptimization();
  hivmPipelineOptions.enableDotScaledCompile = config.shouldcompileDotScaled();
  hivmPipelineOptions.enableMixedCV = config.shouldEnableMixedCV();
  hivmPipelineOptions.simtVFDynamicSize = config.getSimtVFDynamicSize();
  hivmPipelineOptions.enableAutoBlockifyLoop = config.shouldAutoBlockifyLoop();
  hivmPipelineOptions.enableAutoMultiBuffer =
      config.shouldEnableAutoMultiBuffer();
  hivmPipelineOptions.limitAutoMultiBufferOnlyForLocalBuffer =
      config.shouldLimitAutoMultiBufferForLocalBuffer();
  hivmPipelineOptions.limitAutoMultiBufferOfLocalBuffer =
      config.getLimitAutoMultiBufferBufferOfLocalBuffer();
  hivmPipelineOptions.limitMixAutoMultiBufferBuffer =
      config.getLimitAutoMultiBufferBuffer();
  hivmPipelineOptions.enableAutoBindSubBlock =
      config.shouldEnableAutoBindSubBlock();
  hivmPipelineOptions.enableAutoStorageAlign =
      config.shouldEnableAutoStorageAlign();
  hivmPipelineOptions.enableGlobalWorkspaceReuse =
      config.shouldEnableGlobalWorkspaceReuse();
  hivmPipelineOptions.enableHIVMInjectBarrierAllSync =
      config.shouldInjectBarrierAllSync();
  hivmPipelineOptions.workspaceMultiBufferNum =
      config.getWorkspaceMultiBufferNum();
  hivmPipelineOptions.enableAutoCVBalance = config.shouldAutoCVBalance();
  hivmPipelineOptions.enableInjectBlockAllSync =
      config.shouldInjectBlockAllSync();
  hivmPipelineOptions.disableAutoInjectBlockSync =
      config.shouldDisableAutoInjectBlockSync();
  hivmPipelineOptions.enableHIVMGraphSyncSolver = 
      config.shouldEnableHIVMGraphSyncSolver();
  hivmPipelineOptions.enableUnitFlagSync = config.shouldEnableUnitFlagSync();
  hivmPipelineOptions.enableCodeMotion = config.shouldEnableCodeMotion();
  hivmPipelineOptions.target =
      hacc::stringifyTargetDeviceEnum(config.getTargetBackend());
  hivmPipelineOptions.enableVfMergeLevel = config.enableVfMergeLevel();
  hivmPipelineOptions.enableDirectHIVMLowering =
      config.enableDirectHIVMLowering();
  hivmPipelineOptions.enableND2NZOnVector = config.shouldEnableND2NZOnVector();
  hivmPipelineOptions.enableFusedMultiplyAdd = config.shouldEnableFusedMultiplyAdd();
  hivmPipelineOptions.enablePrintMemoryAllocatedSize =
      config.shouldenablePrintMemoryAllocatedSize();
  hivmPipelineOptions.maxReductionSplitNum =
      config.getMaxReductionSplitNum();
  hivmPipelineOptions.injectIrFromFile = config.getInjectIrFromFile();
}

void buildFinalHIVMPipelines(mlir::OpPassManager &pm,
                             const BiShengIRCompileMainConfig &config) {
  if (config.shouldCompileHIVM()) {
    hivm::HIVMPipelineOptions hivmPipelineOptions;
    setupHIVMPipelineOptions(hivmPipelineOptions, config);
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
  options.disableDecomposeReduction = config.getDisableDecomposeReduction();
  options.disableReorderInstruction = config.getDisableReorderInstruction();
  options.tritonMetadataOutput = config.getTritonMetadataOutput();
#if BSPRIV_DAVINCI_BISHENGIR
  if (config.getSharedDynamicSize() < 122880 ||
      config.getSharedDynamicSize() > 221184)
    llvm::report_fatal_error(
        "shared-mem-dynamic-size should range from 122880 to 221184.");
  // max size of shared memory available for simt vf.
  options.sharedDynamicSize = config.getSharedDynamicSize();
  // encode our own compile optimization
  options.enableBishengirSimtOptimization =
      config.getEnableBishengirSimtOptimize();
#endif
  options.protonGPUCompileConfig = config.getProtonGPUCompileConfig();
}

void buildBiShengTTIRPipeline(OpPassManager &pm,
                              const BiShengIRCompileMainConfig &config) {
  if (config.shouldEnableSimdSimtMixCompile()) {
    pm.addPass(createHIVMToTritonGPUConversionPass());
  }

  if (!config.shouldCompileHost()) {
    pm.addPass(hacc::createAppendDeviceSpecPass(
        hacc::AppendTargetDeviceSpecOptions{config.getTargetBackend()}));
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
  if (!config.shouldCompileHost()) {
    pm.addPass(hacc::createAppendDeviceSpecPass(
        hacc::AppendTargetDeviceSpecOptions{config.getTargetBackend()}));
  }

  pm.addPass(createCanonicalizeModulePass());
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
  if (config.shouldCompileTorch()) {
    TorchToNamedOpPipelineOptions torchToNamedOpOptions;
    torchToNamedOpOptions.ensureNoImplicitBroadcast =
        config.shouldEnforceNoImplicitBroadcast();
    createTorchBackendToNamedOpBackendPipeline(pm, torchToNamedOpOptions);
  }
#endif

  hfusion::HFusionPipelineOptions hfusionPipelineOptions;
  if (config.shouldCompileHFusion()) {
    setupHFusionPipelineOptions(hfusionPipelineOptions, config);
    hfusion::buildHFusionPipelines(pm, hfusionPipelineOptions);
  }

  if (config.shouldCompileHIVM()) {
    // Build convert to HIVM Dialect pipeline.
    hivm::ConvertToHIVMPipelineOptions convertToHIVMOptions;
    convertToHIVMOptions.enableTritonKernelCompile =
        config.shouldCompileTriton();
    convertToHIVMOptions.enableAutoBlockifyLoop =
        config.shouldAutoBlockifyLoop();
    convertToHIVMOptions.enableRegBaseHIVMPipe =
        hacc::utils::isRegBasedArch(config.getTargetBackend());
    hivm::HIVMPipelineOptions hivmPipelineOptions;
    setupHIVMPipelineOptions(hivmPipelineOptions, config);
    hivm::buildConvertToHIVMPipeline(pm, convertToHIVMOptions);
    hivm::buildHIVMTensorOptimizations(pm, hivmPipelineOptions);
    if (config.shouldEnableMixedCV()) {
      HIVMAggregatedDecomposeOpOptions decomposeOption;
      decomposeOption.decomposePhase = bishengir::DecomposePhase::NO_CONSTRAINT;
      pm.nest<func::FuncOp>().addPass(
          mlir::hivm::createHIVMAggregatedDecomposeOpPass(decomposeOption));
      pm.addPass(mlir::execution_engine::createConvertHIVMToUpstreamPass());
      hfusion::buildHFusionRegBasePipeline(pm, hfusionPipelineOptions);
      pm.addPass(mlir::hivm::createInferFuncCoreTypePass());
      // FIXME: we need convert hfusion back to hivm again beacsue rank-1
      // `linalg.fill` is not  vectorized. It relies one hivm-single-point-opt
      // pass to convert `memref.store` for performace .thie will be fixed
      // after vectorize move to hivm.
      ConvertHFusionToHIVMOptions hfs2hivmOptions;
      hfs2hivmOptions.mmMapMode = convertToHIVMOptions.enableTritonKernelCompile
                                      ? hfusion::MmMapMode::MacroInstr
                                      : hfusion::MmMapMode::CoreOp;
      pm.addPass(createHFusionToHIVMConversionPass(hfs2hivmOptions));
    }
    if (config.shouldEnableSimdSimtMixCompile()) {
      pm.addPass(hivm::createAutoScopePass());
      pm.addPass(hivm::createInsertMemSemanticForSimtVFPass());
      pm.addPass(scope::createOutlineScopePass());
      pm.addPass(hivm::createInsertAllocBasePlaceholderPass());
      pm.addPass(hivm::createInferSimtVFMemEffectPass());
      pm.addPass(hivm::createInferHIVMMemScopePass());
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
      : PassWrapper<BiShengIRCompilePass, OperationPass<ModuleOp>>(pass) {
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
    enableTorchCompile = pass.enableTorchCompile;
#endif
    enableTritonKernelCompile = pass.enableTritonKernelCompile;
    enableAutoBlockifyLoop = pass.enableAutoBlockifyLoop;
    simtVFDynamicSize = pass.simtVFDynamicSize;
    disableFFTS = pass.disableFFTS;
    disableHFusionVectorize = pass.disableHFusionVectorize;
    enableDropUnitDims = pass.enableDropUnitDims;
    enableFlatten = pass.enableFlatten;
    enableHFusionCompile = pass.enableHFusionCompile;
    enableHIVMCompile = pass.enableHIVMCompile;
    enableLIRCompile = pass.enableLIRCompile;
    targetBackend = pass.targetBackend;
    enableManageHostResources = pass.enableManageHostResources;
    enableStaticBarePtr = pass.enableStaticBarePtr;
    enableBinRelocation = pass.enableBinRelocation;
    saveLinkedIR = pass.saveLinkedIR;
    enableSymbolAnalysis = pass.enableSymbolAnalysis;
    enableAutoVectorizeV2 = pass.enableAutoVectorizeV2;
    enableVFFusion = pass.enableVFFusion;
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
    ensureNoImplicitBroadcast = pass.ensureNoImplicitBroadcast;
#endif
#if (!BISHENGIR_PUBLISH)
    enableCpuTraceIntrinsic = pass.enableCpuTraceIntrinsic;
#endif
    enableSanitizer = pass.enableSanitizer;
    enableDebugInfo = pass.enableDebugInfo;
    enablePrintMemoryAllocatedSize = pass.enablePrintMemoryAllocatedSize;
    outputFile = pass.outputFile;
    appendBishengOptions = pass.appendBishengOptions;
    enableAutoMultiBuffer = pass.enableAutoMultiBuffer;
    limitAutoMultiBufferOnlyForLocalBuffer =
        pass.limitAutoMultiBufferOnlyForLocalBuffer;
    limitAutoMultiBufferOfLocalBuffer = pass.limitAutoMultiBufferOfLocalBuffer;
    limitMixAutoMultiBufferBuffer = pass.limitMixAutoMultiBufferBuffer;
    enableCodeMotion = pass.enableCodeMotion;
    enableOpsReorder = pass.enableOpsReorder;
    enableTuningMode = pass.enableTuningMode;
    blockDim = pass.blockDim;
    maxHorizontalFusionSize = pass.maxHorizontalFusionSize;
    maxFusedElementwiseOps = pass.maxFusedElementwiseOps;
    enableMultiKernel = pass.enableMultiKernel;
    enableCountBufferDmaOpt = pass.enableCountBufferDmaOpt;
    maxBufferCntTuning = pass.maxBufferCntTuning;
    cubeTilingTuning = pass.cubeTilingTuning;
    enableHIVMInjectBarrierAllSync = pass.enableHIVMInjectBarrierAllSync;
    enableInjectBlockAllSync = pass.enableInjectBlockAllSync;
    disableAutoInjectBlockSync = pass.disableAutoInjectBlockSync;
    enableHIVMGraphSyncSolver = pass.enableHIVMGraphSyncSolver;
    enableUnitFlagSync = pass.enableUnitFlagSync;
    enableGlobalWorkspaceReuse = pass.enableGlobalWorkspaceReuse;
    enableAutoStorageAlign = pass.enableAutoStorageAlign;
    enableND2NZOnVector = pass.enableND2NZOnVector;
    enablefusedMultiplyAdd = pass.enablefusedMultiplyAdd;
    enableHighPrecision = pass.enableHighPrecision;
  }
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
        .compileTorch(enableTorchCompile)
#endif
        .compileTriton(enableTritonKernelCompile)
        .setDisableHFusionVectorize(disableHFusionVectorize)
        .setSimtVFDynamicSize(simtVFDynamicSize)
        .setDisableFFTS(disableFFTS)
        .compileHFusion(enableHFusionCompile)
        .compileHIVM(enableHIVMCompile)
        .compileLIR(enableLIRCompile)
        .targetBackend(targetBackend)
        .manageHostResource(enableManageHostResources)
        .barePtrCallConvForStaticShape(enableStaticBarePtr)
        .relocateBinary(enableBinRelocation)
        .setSaveLinkedIR(saveLinkedIR)
        .symbolAnalysis(enableSymbolAnalysis)
        .autoVectorizeV2(enableAutoVectorizeV2)
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
        .noImplicitBroadcast(ensureNoImplicitBroadcast)
#endif
        .multiKernel(enableMultiKernel)
        .highPrecision(enableHighPrecision);

    config
#if (!BISHENGIR_PUBLISH)
        // DFX control options
        .cpuTraceIntrinsic(enableCpuTraceIntrinsic)
#endif
        .enableSanitizer(enableSanitizer)
        .enableDebugInfo(enableDebugInfo)
        .enablePrintMemoryAllocatedSize(enablePrintMemoryAllocatedSize);

    // Output setting options
    config.setOutputFile(outputFile);

    // BiSheng options
    config.appendBishengOptions(appendBishengOptions);

    // General optimization control options
    config.autoMultiBuffer(enableAutoMultiBuffer)
        .limitAutoMultiBufferForLocalBuffer(
            limitAutoMultiBufferOnlyForLocalBuffer)
        .limitAutoMultiBufferOfLocalBuffer(limitAutoMultiBufferOfLocalBuffer)
        .limitMixAutoMultiBufferBuffer(limitMixAutoMultiBufferBuffer)
        .autoBindSubBlock(enableAutoBindSubBlock)
        .deterministicComputing(enableDeterministicComputing)
        .codeMotion(enableCodeMotion)
        .reorderOps(enableOpsReorder)
        .tuningMode(enableTuningMode)
        .setBlockDim(blockDim)
        .enableDropUnitDims(enableDropUnitDims)
        .enableFlatten(enableFlatten);

    // HFusion optimization control options
    config.setMaxHorizontalFusionSize(maxHorizontalFusionSize)
        .setMaxFusedElementwiseOps(maxFusedElementwiseOps)
        .setMaxBufferCountTuning(maxBufferCntTuning)
        .optimizeCountBufferForDma(enableCountBufferDmaOpt)
        .setCubeTilingTuningParams(cubeTilingTuning);

    // HIVM optimization control options
    config.injectBarrierAllSync(enableHIVMInjectBarrierAllSync)
        .injectBlockAllSync(enableInjectBlockAllSync)
        .disableAutoInjectBlockSync(disableAutoInjectBlockSync)
        .enableHIVMGraphSyncSolver(enableHIVMGraphSyncSolver)
        .unitFlagSync(enableUnitFlagSync)
        .globalWorkspaceReuse(enableGlobalWorkspaceReuse)
        .autoStorageAlign(enableAutoStorageAlign)
        .enableND2NZOnVector(enableND2NZOnVector)
        .enablefusedMultiplyAdd(enablefusedMultiplyAdd)
        .autoBlockifyLoop(enableAutoBlockifyLoop);

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
    config.clArgs(args);

    if (failed(runBiShengIRPipeline(moduleOp, config))) {
      signalPassFailure();
    }
  }

protected:
  // -------------------------------------------------------------------------//
  //                       Feature control options
  // -------------------------------------------------------------------------//
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
  Pass::Option<bool> enableTorchCompile{
      *this, "enable-torch-compile",
      llvm::cl::desc("Enable compile from Torch dialect"),
      llvm::cl::init(false)};
#endif
  Pass::Option<bool> enableTritonKernelCompile{
      *this, "enable-triton-kernel-compile",
      llvm::cl::desc("Enable Triton kernel compile"), llvm::cl::init(false)};
  Pass::Option<bool> disableHFusionVectorize{
      *this, "disable-hfusion-vectorize",
      llvm::cl::desc("Disable hfusion auto vectorize"), llvm::cl::init(false)};
  Pass::Option<int> simtVFDynamicSize{
      *this, "simt-vf-dynamic-size",
      llvm::cl::desc("Dynamic ub size(KB) for simt VF. Default is 216"),
      llvm::cl::init(216)};
  Pass::Option<bool> disableFFTS{*this, "disable-ffts",
                                 llvm::cl::desc("force disable FFTS"),
                                 llvm::cl::init(false)};
  Pass::Option<bool> enableHFusionCompile{
      *this, "enable-hfusion-compile",
      llvm::cl::desc("Enable BiShengHIR HFusion compile"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableHIVMCompile{
      *this, "enable-hivm-compile",
      llvm::cl::desc("Enable BiShengHIR HIVM compile"), llvm::cl::init(true)};
  Pass::Option<bool> enableLIRCompile{
      *this, "enable-lir-compile", llvm::cl::desc("Enable BiShengLIR compile"),
      llvm::cl::init(true)};
  Pass::Option<bool> enableManageHostResources{
      *this, "enable-manage-host-resources",
      llvm::cl::desc("Enable managing resource for Host functions"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableStaticBarePtr{
      *this, "enable-static-bare-ptr",
      llvm::cl::desc("Enable generating bare ptr calling convention for static "
                     "shaped kernels"),
      llvm::cl::init(true)};
  Pass::Option<bool> enableBinRelocation{
      *this, "enable-bin-relocation",
      llvm::cl::desc("Enable binary relocation"), llvm::cl::init(true)};
  Pass::Option<bool> enableSymbolAnalysis{
      *this, "enable-symbol-analysis", llvm::cl::desc("Enable symbol analysis"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableAutoVectorizeV2{
      *this, "enable-auto-vectorize-v2",
      llvm::cl::desc("Enable auto vectorize v2"), llvm::cl::init(true)};
  Pass::Option<bool> enableVFFusion{*this, "enable-vf-fusion",
                                    llvm::cl::desc("Enable vf fusion"),
                                    llvm::cl::init(false)};
  Pass::Option<bool> enableHighPrecision{
      *this, "enable-high-precision",
      llvm::cl::desc(
          "Enable high precision calculation for sin/cos in HFusion"),
      llvm::cl::init(true)};
#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
  Pass::Option<bool> ensureNoImplicitBroadcast{
      *this, "ensure-no-implicit-broadcast",
      llvm::cl::desc("Whether to ensure that there is no implicit broadcast "
                     "semantics.If there is a dynamic to dynamic dim "
                     "broadcast, raise a runtime error."),
      llvm::cl::init(false)};
#endif
  Pass::Option<bool> saveLinkedIR{
      *this, "save-linked-ir",
      llvm::cl::desc("Enable saving linked ir before compile to binary"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableMultiKernel{
      *this, "enable-hfusion-multi-kernel",
      llvm::cl::desc("When disabled, graph must fuse as single kernel; when "
                     "enabled, outline multiple kernels."),
      llvm::cl::init(false)};
  Pass::Option<bool> enableDropUnitDims{
      *this, "enable-drop-unit-dims",
      llvm::cl::desc("Enable drop-unit-dims pass"), llvm::cl::init(true)};
  Pass::Option<bool> enableFlatten{*this, "enable-flatten",
                                   llvm::cl::desc("Enable flatten pass"),
                                   llvm::cl::init(true)};
  // -------------------------------------------------------------------------//
  //                           DFX control options
  // -------------------------------------------------------------------------//
#if (!BISHENGIR_PUBLISH)
  Pass::Option<bool> enableCpuTraceIntrinsic{
      *this, "enable-cpu-trace-intrinsic",
      llvm::cl::desc("Enable to generate cpu-accepted IR by eliminating HIVM "
                     "special traits"),
      llvm::cl::init(false)};
#endif
  Pass::Option<bool> enableSanitizer{*this, "enable-sanitizer",
                                     llvm::cl::desc("Enable ascend sanitizer"),
                                     llvm::cl::init(false)};
  Pass::Option<bool> enableDebugInfo{*this, "enable-debug-info",
                                     llvm::cl::desc("Enable debug info"),
                                     llvm::cl::init(false)};
  Pass::Option<bool> enablePrintMemoryAllocatedSize{
      *this, "enable-print-memory-allocated-size",
      llvm::cl::desc("Enable print memory allocated size"),
      llvm::cl::init(false)};
  // -------------------------------------------------------------------------//
  //                        Output setting options
  // -------------------------------------------------------------------------//
  Pass::Option<std::string> outputFile{
      *this, "o", llvm::cl::desc("Specify output bin name"),
      llvm::cl::init("-")};
  // -------------------------------------------------------------------------//
  //                  General optimization control options
  // -------------------------------------------------------------------------//
  Pass::Option<bool> enableLayoutOptimization{
      *this, "enable-layout-optimization",
      llvm::cl::desc("Enable Layout Optimization"), llvm::cl::init(false)};

  Pass::Option<bool> enableMixedCV{
      *this, "enable-mixed-cv", llvm::cl::desc("Enable mixed CV compilation"),
      llvm::cl::init(false)};

  Pass::Option<bool> enableAutoMultiBuffer{
      *this, "enable-auto-multi-buffer",
      llvm::cl::desc("Enable auto multi buffer"), llvm::cl::init(false)};

  Pass::Option<bool> limitAutoMultiBufferOnlyForLocalBuffer{
      *this, "limit-auto-multi-buffer-only-for-local-buffer",
      llvm::cl::desc("Disable workspace multi-buffer optimization"),
      llvm::cl::init(false)};

  Pass::Option<MultiBufferStrategy> limitAutoMultiBufferOfLocalBuffer{
      *this, "limit-auto-multi-buffer-of-local-buffer",
      llvm::cl::desc("When enable-auto-multi-buffer = true, limit local buffer "
                     "mode"),
      llvm::cl::init(MultiBufferStrategy::CUBE_NO_L0C),
      llvm::cl::values(
          clEnumValN(MultiBufferStrategy::NO_LIMIT, "no-limit", "No limit"),
          clEnumValN(MultiBufferStrategy::CUBE_NO_L0C, "no-l0c",
                     "Disable l0c multi buffer"))};

  Pass::Option<MultiBufferStrategy> limitMixAutoMultiBufferBuffer{
      *this, "limit-auto-multi-buffer-buffer",
      llvm::cl::desc("When enable-auto-multi-buffer = true, limit it only work"
                     "for NO_LIMIT, ONLY_CUBE, ONLY_VECTOR"),
      llvm::cl::values(clEnumValN(MultiBufferStrategy::NO_LIMIT, "no-limit",
                                  "limited to cube and vector"),
                       clEnumValN(MultiBufferStrategy::ONLY_CUBE, "only-cube",
                                  "limited to cube"),
                       clEnumValN(MultiBufferStrategy::ONLY_VECTOR,
                                  "only-vector", "limited to vector")),
      llvm::cl::init(MultiBufferStrategy::ONLY_CUBE)};

  Pass::Option<bool> enableAutoBindSubBlock{
      *this, "enable-auto-bind-sub-block",
      llvm::cl::desc("Enable auto bind sub block"), llvm::cl::init(true)};

  Pass::Option<bool> enableDeterministicComputing{
      *this, "enable-deterministic-computing",
      llvm::cl::desc("If enabled, the computation result is deterministic. If "
                     "disabled, we will enable extra optimizations that might "
                     "boost performance, e.g. bind reduce to multiple cores. "
                     "However, the result will be non-deterministic."),
      llvm::cl::init(true)};

  Pass::Option<bool> enableCodeMotion{
      *this, "enable-code-motion",
      llvm::cl::desc("Enable code-motion and subset-hoist (Default = ON)"),
      llvm::cl::init(true)};
  Pass::Option<bool> enableOpsReorder{
      *this, "enable-ops-reorder",
      llvm::cl::desc("Enable ops reorder to opt pipeline (Default = ON)"),
      llvm::cl::init(true)};
  Pass::Option<bool> enableTuningMode{
      *this, "enable-tuning-mode",
      llvm::cl::desc("Enable tuning mode and will not try compile multi times"),
      llvm::cl::init(false)};

  Pass::Option<unsigned> blockDim{*this, "block-dim",
                                  llvm::cl::desc("Number of blocks to use"),
                                  llvm::cl::init(1)};
  // -------------------------------------------------------------------------//
  //                  HFusion optimization control options
  // -------------------------------------------------------------------------//
  Pass::Option<int> maxHorizontalFusionSize{
      *this, "hfusion-max-horizontal-fusion-size",
      llvm::cl::desc(
          "Number of horizontal fusion attempt (Default: unlimited)"),
      llvm::cl::init(-1)};
  Pass::Option<int> maxFusedElementwiseOps{
      *this, "hfusion-max-fused-elementwise-ops",
      llvm::cl::desc("Maximum number of elementwise ops to fuse in "
                     "PreVectorizationFusion (Default: unlimited)"),
      llvm::cl::init(-1)};
  Pass::Option<bool> enableCountBufferDmaOpt{
      *this, "enable-hfusion-count-buffer-dma-opt",
      llvm::cl::desc("If enabled, the buffer used by DMA operations will not "
                     "bereused by vector operations"),
      llvm::cl::init(false)};
  Pass::Option<int64_t> maxBufferCntTuning{
      *this, "hfusion-max-buffer-count-tuning",
      llvm::cl::desc("Allow tuning auto-schedule max buffer count"),
      llvm::cl::init(0)};
  Pass::ListOption<int64_t> cubeTilingTuning{
      *this, "hfusion-cube-tiling-tuning",
      llvm::cl::desc("Allow tuning auto-schedule cube block sizes")};
  // -------------------------------------------------------------------------//
  //                  HIVM optimization control options
  // -------------------------------------------------------------------------//
  Pass::Option<bool> enableHIVMInjectBarrierAllSync{
      *this, "enable-hivm-inject-barrier-all-sync",
      llvm::cl::desc("Enable barrier all mode for HIVM inject sync"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableInjectBlockAllSync{
      *this, "enable-hivm-inject-block-all-sync",
      llvm::cl::desc("Enable inject all block sync for HIVM inject block sync"),
      llvm::cl::init(false)};
  Pass::Option<bool> disableAutoInjectBlockSync{
      *this, "disable-auto-inject-block-sync",
      llvm::cl::desc("Disable auto generating sync block wait/set by "
                     "InjectBlockSync pass"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableHIVMGraphSyncSolver{
      *this, "enable-hivm-graph-sync-solver",
      llvm::cl::desc("Enable HIVM Graph-Sync-Solver pass to do auto-sync."),
      llvm::cl::init(false)};
  Pass::Option<bool> enableUnitFlagSync{
      *this, "enable-hivm-unit-flag-sync",
      llvm::cl::desc(
          "Enable inject sync pass to use unit-flag modes for synchronization"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableGlobalWorkspaceReuse{
      *this, "enable-hivm-global-workspace-reuse",
      llvm::cl::desc("Enable global workspace reuse for plan memory"),
      llvm::cl::init(false)};
  Pass::Option<bool> enableAutoStorageAlign{
      *this, "enable-hivm-auto-storage-align",
      llvm::cl::desc("Enable mark/enable HIVM storage align (Default = ON)"),
      llvm::cl::init(true)};
  Pass::Option<bool> enableND2NZOnVector{
      *this, "enable-hivm-nd2nz-on-vector",
      llvm::cl::desc("Enable nd2nz on vector (Default = OFF)"),
      llvm::cl::init(false)};
  Pass::Option<bool> enablefusedMultiplyAdd{
      *this, "enable-fused-multiply-add",
      llvm::cl::desc("Enable fused multiply add"), llvm::cl::init(false)};
  Pass::Option<bool> enableAutoBlockifyLoop{
      *this, "enable-auto-blockify-loop",
      llvm::cl::desc(
          "Enable auto loop on blocks for all parallel (Default = OFF)"),
      llvm::cl::init(false)};
  Pass::Option<std::string> appendBishengOptions{
      *this, "append-bisheng-options",
      llvm::cl::desc("Append options when calling bisheng"),
      llvm::cl::init("")};
  // -------------------------------------------------------------------------//
  //                  Target options
  // -------------------------------------------------------------------------//
  Pass::Option<std::string> targetBackend{*this, "target",
                                          llvm::cl::desc("Target device name"),
                                          llvm::cl::init("Ascend910B1")};
};

} // namespace bishengir

void bishengir::registerBiShengIRCompilePass() {
  PassRegistration<bishengir::BiShengIRCompilePass>();
}
