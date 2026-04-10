//===- Config.h - BiShengIR Compile Tool Support -----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_BISHENGIR_COMPILE_CONFIG_H
#define BISHENGIR_TOOLS_BISHENGIR_COMPILE_CONFIG_H

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Pass/PassManager.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace mlir;

namespace bishengir {

/// Configuration options for the bishengir-compile tool.
/// This is intended to help building tools like bishengir-compile by collecting
/// the supported options.
/// The API is fluent, and the options are ordered by functionality. The options
/// can be exposed to the LLVM command line by registering them with
/// `BiShengIRCompileMainConfig::registerCLOptions();` and creating a
/// config using
/// `auto config = BiShengIRCompileMainConfig::createFromCLOptions();`.
class BiShengIRCompileMainConfig {
public:
  /// Register the options as global LLVM command line options.
  static void registerCLOptions();

  /// Create a new config with the default set from the CL options.
  static BiShengIRCompileMainConfig createFromCLOptions();

  // -------------------------------------------------------------------------//
  //                       Feature control options                            //
  // -------------------------------------------------------------------------//

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
  BiShengIRCompileMainConfig &compileTorch(bool compile) {
    enableTorchCompileFlag = compile;
    return *this;
  }
  bool shouldCompileTorch() const { return enableTorchCompileFlag; };
#endif

#if BISHENGIR_ENABLE_TRITON_COMPILE
  BiShengIRCompileMainConfig &compileTritonDialect(bool compile) {
    enableTritonIRCompileFlag = compile;
    return *this;
  }
  bool shouldCompileTritonDialect() const { return enableTritonIRCompileFlag; };
#endif

  BiShengIRCompileMainConfig &setDisableHFusionVectorize(bool flag) {
    disableHFusionVectorizeFlag = flag;
    return *this;
  }

  bool shouldDisableHFusionVectorize() const {
    return disableHFusionVectorizeFlag;
  }

  BiShengIRCompileMainConfig &compileFullSIMT(bool compile) {
    enableFullSIMTFlag = compile;
    return *this;
  }
  bool shouldCompileFullSIMT() const { return enableFullSIMTFlag; }

  BiShengIRCompileMainConfig &setSimtVFDynamicSize(int simtVFDynamicSize) {
    simtVFDynamicSizeFlag = simtVFDynamicSize;
    return *this;
  }
  int getSimtVFDynamicSize() const { return simtVFDynamicSizeFlag; }

  BiShengIRCompileMainConfig &
  setTritonGridDim(const std::vector<int64_t> &params) {
    gridDimFlags = params;
    return *this;
  }
  std::vector<int64_t> getTritonGridDim() const { return gridDimFlags; }

  BiShengIRCompileMainConfig &setDisableFFTS(bool flag) {
    disableFFTSFlag = flag;
    return *this;
  }
  bool shouldDisableFFTS() const { return disableFFTSFlag; }

  BiShengIRCompileMainConfig &setDisableFMA(bool flag) {
    disableFMAFlag = flag;
    return *this;
  }
  bool shouldDisableFMA() const { return disableFMAFlag; }

  BiShengIRCompileMainConfig &compileTriton(bool compile) {
    enableTritonKernelCompileFlag = compile;
    return *this;
  }
  bool shouldCompileTriton() const { return enableTritonKernelCompileFlag; }

  BiShengIRCompileMainConfig &compileDotScaled(bool compile) {
    enableDotScaledCompileFlag = compile;
    return *this;
  }
  bool shouldcompileDotScaled() const { return enableDotScaledCompileFlag; };

  BiShengIRCompileMainConfig &compileHFusion(bool compile) {
    enableHFusionCompileFlag = compile;
    return *this;
  }
  bool shouldCompileHFusion() const { return enableHFusionCompileFlag; }

  BiShengIRCompileMainConfig &compileHIVM(bool compile) {
    enableHIVMCompileFlag = compile;
    return *this;
  }
  bool shouldCompileHIVM() const { return enableHIVMCompileFlag; }

  BiShengIRCompileMainConfig &compileLIR(bool compile) {
    enableLIRCompileFlag = compile;
    return *this;
  }
  bool shouldCompileLIR() const { return enableLIRCompileFlag; }

  BiShengIRCompileMainConfig &onlyRunHIVMPipeline(bool compile) {
    onlyRunHIVMPipelineFlag = compile;
    return *this;
  }
  bool shouldOnlyRunHIVMPipeline() const { return onlyRunHIVMPipelineFlag; }

  BiShengIRCompileMainConfig &manageHostResource(bool enable) {
    enableManageHostResourcesFlag = enable;
    return *this;
  }
  bool shouldManageHostResource() const {
    return enableManageHostResourcesFlag;
  }

  BiShengIRCompileMainConfig &barePtrCallConvForStaticShape(bool enable) {
    enableStaticBarePtrFlag = enable;
    return *this;
  }
  bool tryBarePtrCallConvForStaticShape() const {
    return enableStaticBarePtrFlag;
  }

  BiShengIRCompileMainConfig &relocateBinary(bool enable) {
    enableBinRelocationFlag = enable;
    return *this;
  }
  bool shouldRelocateBinary() const { return enableBinRelocationFlag; }

  BiShengIRCompileMainConfig &symbolAnalysis(bool enable) {
    enableSymbolAnalysisFlag = enable;
    return *this;
  }
  bool shouldEnableSymbolAnalysis() const { return enableSymbolAnalysisFlag; }

  BiShengIRCompileMainConfig &autoVectorizeV2(bool enable) {
    enableAutoVectorizeV2Flag = enable;
    return *this;
  }
  bool shouldEnableAutoVectorizeV2() const { return enableAutoVectorizeV2Flag; }
  BiShengIRCompileMainConfig &setMaxFusedOpsInAutoVectorizeV2(int32_t count) {
    maxFusedOpsInAutoVectorizeV2Flag = count;
    return *this;
  }
  int32_t maxFusedOpsInAutoVectorizeV2() const {
    return maxFusedOpsInAutoVectorizeV2Flag;
  }

  BiShengIRCompileMainConfig &VFFusion(bool enable) {
    enableVFFusionFlag = enable;
    return *this;
  }
  bool shouldEnableVFFusion() const { return enableVFFusionFlag; }

  BiShengIRCompileMainConfig &treeReduce(bool enable) {
    enableTreeReduceFlag = enable;
    return *this;
  }
  bool shouldEnableTreeReduce() const { return enableTreeReduceFlag; }

  BiShengIRCompileMainConfig &skipScope(bool skip) {
    skipScopeFlag = skip;
    return *this;
  }
  bool shouldSkipScope() const { return skipScopeFlag; }

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
  BiShengIRCompileMainConfig &noImplicitBroadcast(bool ensure) {
    ensureNoImplicitBroadcastFlag = ensure;
    return *this;
  }
  bool shouldEnforceNoImplicitBroadcast() const {
    return ensureNoImplicitBroadcastFlag;
  }
#endif

  BiShengIRCompileMainConfig &setSaveLinkedIR(bool save) {
    saveLinkedIR = save;
    return *this;
  }
  bool shouldSaveLinkedIR() const { return saveLinkedIR; }

  BiShengIRCompileMainConfig &multiKernel(bool enable) {
    enableMultiKernelFlag = enable;
    return *this;
  }
  bool shouldEnableMultiKernel() const { return enableMultiKernelFlag; }

#if BISHENGIR_ENABLE_TRITON_COMPILE
  BiShengIRCompileMainConfig &setNumWarps(int32_t numWarps) {
    numWarpsFlag = numWarps;
    return *this;
  }
  int32_t getNumWarps() const { return numWarpsFlag; };

  BiShengIRCompileMainConfig &setThreadsPerWarp(int32_t threadsPerWarp) {
    threadsPerWarpFlag = threadsPerWarp;
    return *this;
  }
  int32_t getThreadsPerWarp() const { return threadsPerWarpFlag; };

  BiShengIRCompileMainConfig &setSharedDynamicSize(int32_t sharedDynamicSize) {
    sharedDynamicSizeFlag = sharedDynamicSize;
    return *this;
  }
  int32_t getSharedDynamicSize() const { return sharedDynamicSizeFlag; };

  BiShengIRCompileMainConfig &enableSimdSimtMixCompile(bool enable) {
    enableSimdSimtMixCompileFlag = enable;
    return *this;
  }
  bool shouldEnableSimdSimtMixCompile() const {
    return enableSimdSimtMixCompileFlag;
  }
#endif

  BiShengIRCompileMainConfig &highPrecision(bool enable) {
    enableHighPrecisionFlag = enable;
    return *this;
  }
  bool shouldEnableHighPrecision() const { return enableHighPrecisionFlag; }

  // -------------------------------------------------------------------------//
  //                           DFX control options                            //
  // -------------------------------------------------------------------------//

#if (!BISHENGIR_PUBLISH)
  BiShengIRCompileMainConfig &cpuTraceIntrinsic(bool enable) {
    enableCpuTraceIntrinsicFlag = enable;
    return *this;
  }
  bool enableCPUTraceIntrinsic() const { return enableCpuTraceIntrinsicFlag; }
#endif

  BiShengIRCompileMainConfig &enableSanitizer(bool enable) {
    enableSanitizerFlag = enable;
    return *this;
  }
  bool shouldEnableSanitizer() const { return enableSanitizerFlag; }

  BiShengIRCompileMainConfig &enableDebugInfo(bool enable) {
    enableDebugInfoFlag = enable;
    return *this;
  }
  bool shouldEnableDebugInfo() const { return enableDebugInfoFlag; }

  BiShengIRCompileMainConfig &enablePrintMemoryAllocatedSize(bool enable) {
    enablePrintMemoryAllocatedSizeFlag = enable;
    return *this;
  }
  bool shouldenablePrintMemoryAllocatedSize() const {
    return enablePrintMemoryAllocatedSizeFlag;
  }

  BiShengIRCompileMainConfig &setInjectIrFromFile(const std::string &path) {
    injectIrFromFileFlag = path;
    return *this;
  }
  const std::string &getInjectIrFromFile() const {
    return injectIrFromFileFlag;
  }

  BiShengIRCompileMainConfig &setInjectIrBefore(const std::string &path) {
    injectIrBeforeFlag = path;
    return *this;
  }
  const std::string &getInjectIrBefore() const { return injectIrBeforeFlag; }

  BiShengIRCompileMainConfig &setInjectIrAfter(const std::string &path) {
    injectIrAfterFlag = path;
    return *this;
  }
  const std::string &getInjectIrAfter() const { return injectIrAfterFlag; }

  BiShengIRCompileMainConfig &setPrintPassId(bool enable) {
    printPassIdFlag = enable;
    return *this;
  }
  bool shouldPrintPassId() const { return printPassIdFlag; }

  // -------------------------------------------------------------------------//
  //                        Output setting options                            //
  // -------------------------------------------------------------------------//

  BiShengIRCompileMainConfig &setOutputFile(const std::string &file) {
    outputFileFlag = file;
    return *this;
  }
  std::string outputFile() const { return outputFileFlag; }

  /// Configs for compiling host functions. Note that these configs are only
  /// used internally.
  BiShengIRCompileMainConfig &compileHost(bool compile) {
    enableHostCompileFlag = compile;
    return *this;
  }
  bool shouldCompileHost() const { return enableHostCompileFlag; }

  BiShengIRCompileMainConfig &setHostOutputFile(const std::string &file) {
    hostOutputFileFlag = file;
    return *this;
  }
  std::string hostOutputFile() const { return hostOutputFileFlag; }

  BiShengIRCompileMainConfig &lowerToLLVMFlag(bool compile) {
    enableLowerToLLVMFlag = compile;
    return *this;
  }
  bool shouldLowerToLLVM() const { return enableLowerToLLVMFlag; }

  // -------------------------------------------------------------------------//
  //                  General optimization control options                    //
  // -------------------------------------------------------------------------//
  BiShengIRCompileMainConfig &layoutOptimization(bool enable) {
    enableLayoutOptimizationFlag = enable;
    return *this;
  }

  bool shouldEnableLayoutOptimization() const {
    return enableLayoutOptimizationFlag &&
           hacc::utils::isAscend950(this->getTargetBackend());
  }

  BiShengIRCompileMainConfig &mixedCV(bool enable) {
    enableMixedCVFlag = enable;
    return *this;
  }
  bool shouldEnableMixedCV() const {
    return enableMixedCVFlag &&
           hacc::utils::isAscend950(this->getTargetBackend());
  }

  BiShengIRCompileMainConfig &autoMultiBuffer(bool enable) {
    enableAutoMultiBufferFlag = enable;
    return *this;
  }
  bool shouldEnableAutoMultiBuffer() const { return enableAutoMultiBufferFlag; }

  BiShengIRCompileMainConfig &limitAutoMultiBufferForLocalBuffer(bool limit) {
    limitAutoMultiBufferOnlyForLocalBufferFlag = limit;
    return *this;
  }

  BiShengIRCompileMainConfig &
  limitAutoMultiBufferOfLocalBuffer(MultiBufferStrategy limit) {
    limitAutoMultiBufferOfLocalBufferFlag = limit;
    return *this;
  }

  BiShengIRCompileMainConfig &
  limitMixAutoMultiBufferBuffer(MultiBufferStrategy limit) {
    limitMixAutoMultiBufferBufferFlag = limit;
    return *this;
  }

  bool shouldLimitAutoMultiBufferForLocalBuffer() const {
    return limitAutoMultiBufferOnlyForLocalBufferFlag;
  }

  MultiBufferStrategy getLimitAutoMultiBufferBufferOfLocalBuffer() const {
    return limitAutoMultiBufferOfLocalBufferFlag;
  }

  MultiBufferStrategy getLimitAutoMultiBufferBuffer() const {
    return limitMixAutoMultiBufferBufferFlag;
  }

  BiShengIRCompileMainConfig &autoBindSubBlock(bool enable) {
    enableAutoBindSubBlockFlag = enable;
    return *this;
  }
  bool shouldEnableAutoBindSubBlock() const {
    return enableAutoBindSubBlockFlag;
  }

  BiShengIRCompileMainConfig &deterministicComputing(bool enable) {
    enableDeterministicComputingFlag = enable;
    return *this;
  }

  bool isDeterministicComputing() const {
    return enableDeterministicComputingFlag;
  }

  BiShengIRCompileMainConfig &codeMotion(bool enable) {
    enableCodeMotionFlag = enable;
    return *this;
  }
  bool shouldEnableCodeMotion() const { return enableCodeMotionFlag; }

  BiShengIRCompileMainConfig &reorderOps(bool enable) {
    enableOpsReorderFlag = enable;
    return *this;
  }
  bool shouldEnableOpsReorder() const { return enableOpsReorderFlag; }

  BiShengIRCompileMainConfig &tuningMode(bool enable) {
    enableTuningModeFlag = enable;
    return *this;
  }
  bool isTuning() const { return enableTuningModeFlag; }

  BiShengIRCompileMainConfig &setBlockDim(unsigned blockDim) {
    blockDimFlag = blockDim;
    return *this;
  }
  unsigned blockDim() const { return blockDimFlag; }

  BiShengIRCompileMainConfig &setVfMergeLevel(unsigned enableVfMergeLevel) {
    enableVfMergeLevelFlag = (int32_t)enableVfMergeLevel;
    return *this;
  }
  int32_t enableVfMergeLevel() const { return enableVfMergeLevelFlag; }

  BiShengIRCompileMainConfig &enableBishengirSimtOptimization(int enable) {
    enableBishengirSimtOptimizationFlag = enable;
    return *this;
  }
  int getEnableBishengirSimtOptimize() const {
    return enableBishengirSimtOptimizationFlag;
  }

  BiShengIRCompileMainConfig &disableDecomposeReduction(bool disable) {
    disableDecomposeReductionFlag = disable;
    return *this;
  }
  bool getDisableDecomposeReduction() const {
    return disableDecomposeReductionFlag;
  }

  BiShengIRCompileMainConfig &disableReorderInstruction(bool disable) {
    disableReorderInstructionFlag = disable;
    return *this;
  }
  bool getDisableReorderInstruction() const {
    return disableReorderInstructionFlag;
  }

  BiShengIRCompileMainConfig &enableSimtReorderInstruction(bool enable) {
    enableSimtReorderInstructionFlag = enable;
    return *this;
  }
  bool getEnableSimtReorderInstruction() const {
    return enableSimtReorderInstructionFlag;
  }

  BiShengIRCompileMainConfig &simtStackLimit(int32_t limit) {
    simtStackLimitFlag = limit;
    return *this;
  }
  std::optional<int32_t> getSimtStackLimit() const {
    return simtStackLimitFlag;
  }

  BiShengIRCompileMainConfig &tritonMetadataOutput(std::string Path) {
    tritonMetadataOutputPath = Path;
    return *this;
  }
  std::string getTritonMetadataOutput() const {
    return tritonMetadataOutputPath;
  }

  // -------------------------------------------------------------------------//
  //                  HFusion optimization control options                    //
  // -------------------------------------------------------------------------//

  BiShengIRCompileMainConfig &setMaxHorizontalFusionSize(int32_t size) {
    maxHorizontalFusionSizeFlag = size;
    return *this;
  }
  int32_t maxHorizontalFusionSize() const {
    return maxHorizontalFusionSizeFlag;
  }
  BiShengIRCompileMainConfig &setMaxFusedElementwiseOps(int32_t count) {
    maxFusedElementwiseOpsFlag = count;
    return *this;
  }
  int32_t maxFusedElementwiseOps() const { return maxFusedElementwiseOpsFlag; }

  /// Update max buffer count tuning delta.
  BiShengIRCompileMainConfig &setMaxBufferCountTuning(int64_t maxBufferCount) {
    maxBufferCntTuningFlag = maxBufferCount;
    return *this;
  }
  BiShengIRCompileMainConfig &increaseMaxBufferCountTuning(int64_t delta) {
    maxBufferCntTuningFlag += delta;
    return *this;
  }
  int64_t maxBufferCountTuning() const { return maxBufferCntTuningFlag; }

  BiShengIRCompileMainConfig &
  setCubeTilingTuningParams(const std::vector<int64_t> &params) {
    cubeTilingTuningFlags = params;
    return *this;
  }
  std::vector<int64_t> cubeTilingTuningParams() const {
    return cubeTilingTuningFlags;
  }

  BiShengIRCompileMainConfig &optimizeCountBufferForDma(bool enable) {
    enableCountBufferDmaOptFlag = enable;
    return *this;
  }
  bool shouldEnableCountBufferDmaOpt() const {
    return enableCountBufferDmaOptFlag;
  }

  // -------------------------------------------------------------------------//
  //                  HIVM optimization control options                       //
  // -------------------------------------------------------------------------//

  BiShengIRCompileMainConfig &injectBarrierAllSync(bool enable) {
    enableHIVMInjectBarrierAllSyncFlag = enable;
    return *this;
  }
  bool shouldInjectBarrierAllSync() const {
    return enableHIVMInjectBarrierAllSyncFlag;
  }

  BiShengIRCompileMainConfig &unitFlagSync(bool enable) {
    enableUnitFlagSyncFlag = enable;
    return *this;
  }
  bool shouldEnableUnitFlagSync() const { return enableUnitFlagSyncFlag; }

  BiShengIRCompileMainConfig &injectBlockAllSync(bool enable) {
    enableInjectBlockAllSyncFlag = enable;
    return *this;
  }
  bool shouldInjectBlockAllSync() const { return enableInjectBlockAllSyncFlag; }

  BiShengIRCompileMainConfig &disableAutoInjectBlockSync(bool disable) {
    disableAutoInjectBlockSyncFlag = disable;
    return *this;
  }
  bool shouldDisableAutoInjectBlockSync() const {
    return disableAutoInjectBlockSyncFlag;
  }

  BiShengIRCompileMainConfig &enableHIVMGraphSyncSolver(bool enable) {
    enableHIVMGraphSyncSolverFlag = enable;
    return *this;
  }
  bool shouldEnableHIVMGraphSyncSolver() const {
    return enableHIVMGraphSyncSolverFlag;
  }

  BiShengIRCompileMainConfig &enableDropUnitDims(bool enable) {
    enableDropUnitDimsFlag = enable;
    return *this;
  }
  bool shouldEnableDropUnitDims() const { return enableDropUnitDimsFlag; }

  BiShengIRCompileMainConfig &enableFlatten(bool enable) {
    enableFlattenFlag = enable;
    return *this;
  }

  bool shouldEnableFlatten() const { return enableFlattenFlag; }

  BiShengIRCompileMainConfig &enableFuseReductionIntoLoop(bool enable) {
    enableFuseReductionIntoLoopFlag = enable;
    return *this;
  }
  bool shouldEnableFuseReductionIntoLoop() const {
    return enableFuseReductionIntoLoopFlag;
  }

  BiShengIRCompileMainConfig &setWorkspaceMultiBufferNum(unsigned number) {
    workspaceMultiBufferNumFlag = number;
    return *this;
  }
  unsigned getWorkspaceMultiBufferNum() const {
    return workspaceMultiBufferNumFlag;
  }

  BiShengIRCompileMainConfig &enableAutoCVBalance(bool enable) {
    enableAutoCVBalanceFlag = enable;
    return *this;
  }
  bool shouldAutoCVBalance() const { return enableAutoCVBalanceFlag; }

  BiShengIRCompileMainConfig &globalWorkspaceReuse(bool enable) {
    enableGlobalWorkspaceReuseFlag = enable;
    return *this;
  }
  bool shouldEnableGlobalWorkspaceReuse() const {
    return enableGlobalWorkspaceReuseFlag;
  }

  BiShengIRCompileMainConfig &autoStorageAlign(bool enable) {
    enableAutoStorageAlignFlag = enable;
    return *this;
  }
  bool shouldEnableAutoStorageAlign() const {
    return enableAutoStorageAlignFlag;
  }

  BiShengIRCompileMainConfig &enablefusedMultiplyAdd(bool enable) {
    enableFusedMultiplyAddFlag = enable;
    return *this;
  }
  bool shouldEnableFusedMultiplyAdd() const {
    return enableFusedMultiplyAddFlag;
  }

  BiShengIRCompileMainConfig &disableTightlyCoupledBufferReuse(bool disable) {
    disableTightlyCoupledBufferReuseFlag = disable;
    return *this;
  }
  bool shouldDisableTightlyCoupledBufferReuse() const {
    return disableTightlyCoupledBufferReuseFlag;
  }

  BiShengIRCompileMainConfig &enableND2NZOnVector(bool compile) {
    enableND2NZOnVectorFlag = compile;
    return *this;
  }
  bool shouldEnableND2NZOnVector() const { return enableND2NZOnVectorFlag; }

  BiShengIRCompileMainConfig &autoBlockifyLoop(bool enable) {
    enableAutoBlockifyLoopFlag = enable;
    return *this;
  }
  bool shouldAutoBlockifyLoop() const { return enableAutoBlockifyLoopFlag; }

  BiShengIRCompileMainConfig &maxReductionSplitNum(int number) {
    maxReductionSplitNumFlag = number;
    return *this;
  }
  int getMaxReductionSplitNum() const { return maxReductionSplitNumFlag; }

  BiShengIRCompileMainConfig &enableMultipleConsumerFusion(bool enable) {
    enableMultipleConsumerFusionFlag = enable;
    return *this;
  }
  bool shouldEnableMultipleConsumerFusion() const {
    return enableMultipleConsumerFusionFlag;
  }

  // -------------------------------------------------------------------------//
  //                            Target options                                //
  // -------------------------------------------------------------------------//
  BiShengIRCompileMainConfig &targetBackend(mlir::hacc::TargetDevice target) {
    targetBackendFlag = target;
    return *this;
  }

  BiShengIRCompileMainConfig &targetBackend(std::string target) {
    targetBackendFlag = mlir::hacc::symbolizeTargetDeviceEnum(target);
    return *this;
  }

  mlir::hacc::TargetDevice getTargetBackend() const {
    return targetBackendFlag;
  }

  // -------------------------------------------------------------------------//
  //                          CPU Runner options                              //
  // -------------------------------------------------------------------------//

  static bool shouldEnableCPURunner();

  // -------------------------------------------------------------------------//
  //                            Other options                                 //
  // -------------------------------------------------------------------------//

  /// Configs related to BiSheng compiler.
  BiShengIRCompileMainConfig &updateMaxInputParamsSizeInBytes(size_t size) {
    deviceMaxInputParamSizeInBytesFlag =
        std::max(size, deviceMaxInputParamSizeInBytesFlag);
    return *this;
  }
  BiShengIRCompileMainConfig &setUseDPX(bool enable) {
    useDPXFlag = enable;
    return *this;
  }
  bool shouldUseDPX() const { return useDPXFlag; }
  size_t deviceMaxInputParamSizeInBytes() const {
    return deviceMaxInputParamSizeInBytesFlag;
  }

  BiShengIRCompileMainConfig &
  setenableDirectHIVMLowering(unsigned enableDirectHIVMLowering) {
    enableDirectHIVMLoweringFlag = enableDirectHIVMLowering;
    return *this;
  }
  unsigned enableDirectHIVMLowering() const {
    return enableDirectHIVMLoweringFlag;
  }

  /// Allow operation with no registered dialects.
  /// This option is for convenience during testing only and discouraged in
  /// general.
  BiShengIRCompileMainConfig &allowUnregisteredDialects(bool allow) {
    allowUnregisteredDialectsFlag = allow;
    return *this;
  }
  bool shouldAllowUnregisteredDialects() const {
    return allowUnregisteredDialectsFlag;
  }

  /// Allow use to pass customized options to bisheng
  BiShengIRCompileMainConfig &
  appendBishengOptions(std::string appendBishengOptions) {
    appendBishengOptionsFlag = appendBishengOptions;
    return *this;
  }
  std::string getAppendBiShengOptions() const {
    return appendBishengOptionsFlag;
  }

  BiShengIRCompileMainConfig &clArgs(std::vector<std::string> args) {
    clArgsFlag = std::move(args);
    return *this;
  }
  std::vector<std::string> getClArgs() const { return clArgsFlag; }

  void readCLArgs(int argc, char **argv) {
    std::vector<std::string> clArgs;
    for (int i = 1; i < argc; i++) {
      std::string curArg = argv[i];
      if (curArg[0] != '-') {
        continue;
      }
      if (curArg == "-o") {
        i++;
        continue;
      }
      clArgs.push_back(curArg);
    }
    this->clArgs(clArgs);
  }

protected:
  // -------------------------------------------------------------------------//
  //                       Feature control options                            //
  // -------------------------------------------------------------------------//

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
  /// Enable BiShengHIR Torch compile.
  bool enableTorchCompileFlag{false};
#endif

  /// Enable symbol analysis flag
  bool enableSymbolAnalysisFlag{false};

  /// Enable Auto Vectorize V2
  bool enableAutoVectorizeV2Flag{true};
  /// Maximum number of ops to fuse in AutoVectorizeV2, -1 uses pass default.
  int32_t maxFusedOpsInAutoVectorizeV2Flag{-1};

  bool enableVFFusionFlag{false};

  bool enableTreeReduceFlag{false};

  /// skip passes like flattenOps when scope exists
  bool skipScopeFlag{true};

  bool disableHFusionVectorizeFlag{false};

  bool enableFullSIMTFlag{false};

  int simtVFDynamicSizeFlag{216};

  std::vector<int64_t> gridDimFlags{};

  /// Force enable FFTS for cube kernels
  bool disableFFTSFlag{false};

  /// Force enable FMA for simt kernels
  bool disableFMAFlag{false};

  /// Enable Triton kernel compile.
  bool enableTritonKernelCompileFlag{false};

#if BISHENGIR_ENABLE_TRITON_COMPILE
  /// Enable Triton Dialect compile.
  bool enableTritonIRCompileFlag{false};

  /// Enable DotScaled compile.
  bool enableDotScaledCompileFlag{false};
#endif

  /// Enable BiShengHIR HFusion compile.
  bool enableHFusionCompileFlag{false};

  /// Enable BiShengHIR HIVM compile.
  bool enableHIVMCompileFlag{true};

  /// Enable BiShengLIR compile.
  bool enableLIRCompileFlag{true};

  /// Enable regbase architecture compile.
  bool enableRegbaseCompileFlag{false};

  /// Enable managing resource for Host functions.
  bool enableManageHostResourcesFlag{false};

  /// Enable generating bare ptr calling convention for static shaped kernels.
  bool enableStaticBarePtrFlag{true};

  /// Enable binary relocation.
  bool enableBinRelocationFlag{true};

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
  /// Whether to ensure that there is no implicit broadcast semantics. If there
  /// is a dynamic to dynamic dim broadcast, raise a runtime error.
  bool ensureNoImplicitBroadcastFlag{false};
#endif

  /// Enable saving linked IR before compile to binary
  bool saveLinkedIR{false};

  /// When disabled, graph must fuse as single kernel; when enabled, outline
  /// multiple kernels.
  bool enableMultiKernelFlag{false};

#if BISHENGIR_ENABLE_TRITON_COMPILE
  /// Number of warps.
  int32_t numWarpsFlag{4};

  /// Number of threads per warp.
  int32_t threadsPerWarpFlag{32};

  /// Dynamic size of shared memory (in bytes) for SIMT-VF.
  int32_t sharedDynamicSizeFlag{122880};

  bool enableSimdSimtMixCompileFlag;
#endif

  /// Enable high precision calculation for sin/cos in HFusion.
  bool enableHighPrecisionFlag{false};

  // -------------------------------------------------------------------------//
  //                           DFX control options                            //
  // -------------------------------------------------------------------------//

#if (!BISHENGIR_PUBLISH)
  /// Enable to generate host-accepted IR by eliminating HIVM special traits.
  bool enableCpuTraceIntrinsicFlag{false};
#endif

  /// Enable ascend sanitizer.
  ///
  /// When this is enabled `--mlir-print-debuginfo` option will be enabled at
  /// the same time so the lineNo can be displayed correctly.
  /// It may cause some unexpected failures.
  bool enableSanitizerFlag{false};

  /// Requires an additional switch to control whether to enable debug info.
  bool enableDebugInfoFlag{false};

  /// Enable to get Ub allocation.
  bool enablePrintMemoryAllocatedSizeFlag{false};

  /// Path to IR file for inject-ir pass (debug).
  std::string injectIrFromFileFlag{""};

  /// Path to IR file to inject before a specific pass.
  std::string injectIrBeforeFlag{""};

  /// Path to IR file to inject after a specific pass.
  std::string injectIrAfterFlag{""};
  // -------------------------------------------------------------------------//
  //                        Output setting options                            //
  // -------------------------------------------------------------------------//

  /// Output binary file path.
  std::string outputFileFlag{"-"};

  /// Enable BiShengHIR compile for Host module.
  bool enableHostCompileFlag{false};

  /// Enable lower-to-llvm pipeline
  bool enableLowerToLLVMFlag{true};

  /// Output file path for host module. This is only a temporary file path.
  std::string hostOutputFileFlag{"-"};

  /// Whether to print pass IDs during compilation
  bool printPassIdFlag{false};

  // -------------------------------------------------------------------------//
  //                  General optimization control options                    //
  // -------------------------------------------------------------------------//

  /// Enable Layout Optimization
  bool enableLayoutOptimizationFlag{false};

  /// Enable mixed CV compilation
  bool enableMixedCVFlag{false};

  /// Enable auto multi buffer.
  bool enableAutoMultiBufferFlag{false};

  /// Disable drop-unit-dims pass.
  bool enableDropUnitDimsFlag{true};

  /// Enable flatten pass.
  bool enableFlattenFlag{true};

  /// Enable fuse-reduction-into-loop pass.
  bool enableFuseReductionIntoLoopFlag{false};

  /// Number of multibuffers for workspace.
  unsigned workspaceMultiBufferNumFlag{2};

  /// Enable auto balancing during CV Pipelining pass
  bool enableAutoCVBalanceFlag{false};

  /// When `enable-auto-multi-buffer=true`, limit it only work for local buffer
  // TODO: change default value to be false
  bool limitAutoMultiBufferOnlyForLocalBufferFlag{false};

  /// When `enable-auto-multi-buffer=true`, limit it only work for
  /// NO_LIMIT, CUBE_NO_L0C
  MultiBufferStrategy limitAutoMultiBufferOfLocalBufferFlag{
      MultiBufferStrategy::CUBE_NO_L0C};

  /// When `enable-auto-multi-buffer=true`, limit it only work for
  /// NO_LIMIT, ONLY_CUBE, ONLY_VECTOR
  MultiBufferStrategy limitMixAutoMultiBufferBufferFlag{
      MultiBufferStrategy::ONLY_CUBE};

  /// Enable auto bind sub block
  bool enableAutoBindSubBlockFlag{true};

  /// If enabled, the computation result is deterministic. If disabled, we will
  /// enable extra optimizations that might boost performance, e.g. bind reduce
  /// to multiple cores. However, the result will be non-deterministic.
  bool enableDeterministicComputingFlag{false};

  /// Enable code-motion/subset-hoist.
  bool enableCodeMotionFlag{true};

  /// Enable ops reorder to opt pipeline.
  bool enableOpsReorderFlag{true};

  /// Enable tuning mode and will not try compile multi times in case of plan
  /// memory failure.
  bool enableTuningModeFlag{false};

  /// Number of blocks to use.
  unsigned blockDimFlag{1};

  // Enable vector function merging level
  int32_t enableVfMergeLevelFlag{1};

  /// Number of multibuffers for workspace.
  int maxReductionSplitNumFlag{1};

  // -------------------------------------------------------------------------//
  //                  HFusion optimization control options                    //
  // -------------------------------------------------------------------------//

  /// Number of horizontal fusion attempt.
  int32_t maxHorizontalFusionSizeFlag{-1};
  /// Maximum number of elementwise ops to fuse in PreVectorizationFusion.
  int32_t maxFusedElementwiseOpsFlag{-1};

  /// Enable multiple consumer fusion in AutoVectorizeV2
  bool enableMultipleConsumerFusionFlag{false};

  /// Max buffer count tuning in HFusion auto schedule.
  int64_t maxBufferCntTuningFlag{0};

  /// Cube block size tuning in HFusion auto schedule.
  std::vector<int64_t> cubeTilingTuningFlags{};

  /// If enabled, the buffer used by DMA operations will not be reused by Vector
  /// operations.
  bool enableCountBufferDmaOptFlag{false};

  // -------------------------------------------------------------------------//
  //                  HIVM optimization control options                       //
  // -------------------------------------------------------------------------//

  /// Enable barrier all mode for HIVM inject sync.
  bool enableHIVMInjectBarrierAllSyncFlag{false};

  /// Enable inject all block sync for HIVM inject block sync
  bool enableInjectBlockAllSyncFlag{false};

  /// Disable auto generating sync block wait/set by InjectBlockSync pass
  bool disableAutoInjectBlockSyncFlag{false};

  /// Enable HIVM Graph-Sync-Solver pass to do auto-sync.
  bool enableHIVMGraphSyncSolverFlag{false};


  /// Enable inject sync pass to use unit-flag modes for synchronization
  bool enableUnitFlagSyncFlag{false};

  /// Disable plan memory pass to reuse tightly coupled buffer 
  bool disableTightlyCoupledBufferReuseFlag{false};

  /// Enable global workspace reuse.
  bool enableGlobalWorkspaceReuseFlag{false};

  /// Enable mark/enable storage align
  bool enableAutoStorageAlignFlag{true};

  /// Enable fused multiply add
  bool enableFusedMultiplyAddFlag{false};

  /// Enable nd2nz on vector.
  bool enableND2NZOnVectorFlag{false};

  /// Enable auto loop on blocks when logical blocknum is larger than physical
  /// one
  bool enableAutoBlockifyLoopFlag{false};

  int enableBishengirSimtOptimizationFlag{000};

  std::optional<int32_t> simtStackLimitFlag = std::nullopt;

  std::string tritonMetadataOutputPath{""};

  bool useDPXFlag{false};

  /// Disable decompose reduction
  bool disableDecomposeReductionFlag{false};

  /// Disable reorder instruction
  bool disableReorderInstructionFlag{false};

  /// Enable SIMT reorder instruction pattern
  bool enableSimtReorderInstructionFlag{false};

  // -------------------------------------------------------------------------//
  //                            Target options                                //
  // -------------------------------------------------------------------------//

  /// The device target to lower to.
  mlir::hacc::TargetDevice targetBackendFlag{mlir::hacc::TargetDevice::Unknown};

  // -------------------------------------------------------------------------//
  //                            Other options                                 //
  // -------------------------------------------------------------------------//

  /// The default value is set to same as bisheng compiler
  size_t deviceMaxInputParamSizeInBytesFlag{1536};

  /// Allow operation with no registered dialects.
  /// This option is for convenience during testing only and discouraged in
  /// general.
  bool allowUnregisteredDialectsFlag{false};

  bool enableDirectHIVMLoweringFlag{false};

  std::string appendBishengOptionsFlag{""};

  bool onlyRunHIVMPipelineFlag{false};

  std::vector<std::string> clArgsFlag;
};

} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIR_COMPILE_CONFIG_H
