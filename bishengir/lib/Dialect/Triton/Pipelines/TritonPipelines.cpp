//===- TritonPipelines.cpp - BiShengIR Triton pipelines ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Config/bishengir-config.h"

#if BISHENGIR_ENABLE_TRITON_COMPILE
#include "NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "NVGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "bishengir/Conversion/TritonAscendGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "Conversion/ProtonGPUToLLVM/Passes.h"
#include "Conversion/ProtonToProtonGPU/Passes.h"
#endif

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/Triton/Pipelines/Passes.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

/// This is constructed based on triton/third_party/nvidia/backend/compiler.py

#define ADD_CANONICALIZER_PASS                                                 \
  CanonicalizerOptions options;                                                \
  options.enableExtendedPattern = true;                                        \
  std::vector<std::string> disabledPatterns{};                                 \
  options.disabledPatterns = disabledPatterns;                                 \
  pm.addPass(createCanonicalizerPass(options))

#define ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS                             \
  pm.nest<func::FuncOp>().addPass(createCanonicalizerPass(options))

namespace {
#if BISHENGIR_ENABLE_TRITON_COMPILE
void buildTritonGPUOptimizationPipeline(
    OpPassManager &pm,
    const bishengir::triton::LowerTritonPipelineOptions &tritonOptions) {
  pm.addPass(triton::gpu::createTritonGPUCoalesce());
  pm.addPass(triton::gpu::createTritonGPURemoveLayoutConversions());
  pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeThreadLocality());
  pm.addPass(triton::gpu::createTritonGPURemoveLayoutConversions());
  pm.addPass(mlir::triton::createTritonLoopAwareCSE());
  pm.addPass(mlir::triton::gpu::createTritonGPUFuseNestedLoops());
  ADD_CANONICALIZER_PASS;
  pm.addPass(mlir::triton::createTritonLoopInvariantCodeMotion());
  ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
  pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
  pm.addPass(mlir::triton::gpu::createTritonGPUScheduleLoops());
  ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
  pm.addPass(mlir::triton::createTritonLoopAwareCSE());
  pm.addPass(triton::gpu::createTritonGPURemoveLayoutConversions());
  pm.addPass(mlir::triton::gpu::createTritonGPUReduceDataDuplication());
  if (!tritonOptions.disableReorderInstruction) {
#if BSPRIV_DAVINCI_BISHENGIR
    mlir::triton::gpu::TritonGPUReorderInstructionsOptions reorderInstructionsOptions;
    reorderInstructionsOptions.enableSimtReorderInstruction = tritonOptions.enableSimtReorderInstruction;
    pm.addPass(mlir::triton::gpu::createTritonGPUReorderInstructionsPass(reorderInstructionsOptions));
#else
    pm.addPass(mlir::triton::gpu::createTritonGPUReorderInstructionsPass());
#endif
  }
  if (!tritonOptions.disableDecomposeReduction)
    pm.addPass(bishengir::triton::createDecomposeReductionPass());
  pm.addPass(bishengir::triton::createOptimizeLayoutsPass());
  pm.addPass(mlir::triton::createTritonLoopAwareCSE());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createSCCPPass());
  pm.addPass(createCSEPass());
  ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
}

#endif

} // namespace

namespace bishengir {
namespace triton {

#if BISHENGIR_ENABLE_TRITON_COMPILE
void buildLowerTritonPipeline(OpPassManager &pm,
                              const LowerTritonPipelineOptions &options) {
  bishengir::SetBishengirSimtOptAttrOptions optionsSimtOpt;
  optionsSimtOpt.enableBishengirSimtOptimization =
      options.enableBishengirSimtOptimization;
  pm.addPass(
      bishengir::triton::createSetBishengirSimtOptAttrPass(optionsSimtOpt));
  pm.addNestedPass<mlir::triton::FuncOp>(
      bishengir::triton::createAdaptTritonIRKernelPass());
  pm.addPass(bishengir::triton::createOptimizeLoadsPass());
  pm.addPass(bishengir::triton::createLoopRestructureArangeOptimizationPass());
  pm.addNestedPass<mlir::triton::FuncOp>(
      bishengir::triton::createLegalizeF16ForTritonPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSCCPPass());
  {
    ADD_CANONICALIZER_PASS;
  }
  pm.addPass(bishengir::triton::createFixFusedCatPass());
  pm.addPass(mlir::triton::createTritonRewriteTensorPointer());
  pm.addPass(mlir::triton::createTritonRewriteTensorDescriptorToPointer());
  // Convert TTIR to TTGIR
  // TODO: Adapt target for NPU
  mlir::triton::ConvertTritonToTritonGPUOptions convertTritonToTritonGPUOpt;
  convertTritonToTritonGPUOpt.target = "cuda:80";
  convertTritonToTritonGPUOpt.numWarps = options.numWarps;
  convertTritonToTritonGPUOpt.threadsPerWarp = options.threadsPerWarp;
#if BSPRIV_DAVINCI_BISHENGIR
  // max size of shared memory available for simt vf.
  convertTritonToTritonGPUOpt.shared = options.sharedDynamicSize;
#endif
  pm.addPass(mlir::triton::createConvertTritonToTritonGPU(
      convertTritonToTritonGPUOpt));
  // Optimize TTGIR
  buildTritonGPUOptimizationPipeline(pm, options);
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(mlir::triton::ascend::createAllocateAscendSharedMemory());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(mlir::triton::proton::createConvertProtonToProtonGPUPass(
    options.protonGPUCompileConfig.metricType,
    options.protonGPUCompileConfig.samplingStrategy,
    options.protonGPUCompileConfig.samplingOptions,
    options.protonGPUCompileConfig.granularity,
    options.protonGPUCompileConfig.bufferStrategy,
    options.protonGPUCompileConfig.bufferType,
    options.protonGPUCompileConfig.bufferSize,
    options.protonGPUCompileConfig.maxSharedMemSize,
    options.protonGPUCompileConfig.profileScratchSize,
    options.protonGPUCompileConfig.profileScratchAlignment,
    options.protonGPUCompileConfig.clockExtension
  ));
  pm.addPass(createCSEPass());
  pm.addPass(mlir::triton::proton::gpu::createAllocateProtonSharedMemoryPass());
  pm.addPass(mlir::triton::createConvertTritonAscendGPUToLLVMPass());
  pm.addPass(createCSEPass());
  pm.addPass(mlir::triton::proton::gpu::createAllocateProtonGlobalScratchBufferPass());
  pm.addPass(mlir::triton::proton::gpu::createConvertProtonAscendGPUToLLVMPass());
  std::unique_ptr<OperationPass<ModuleOp>> convertTritonGPUToLLVMPass =
      mlir::triton::createConvertTritonGPUToLLVMPass(70, 73);
  pm.addPass(std::unique_ptr<mlir::Pass>(
      dyn_cast<mlir::Pass>(convertTritonGPUToLLVMPass.release())));
  pm.addPass(mlir::triton::createConvertNVGPUToLLVM());
  pm.addPass(createGetTritonMetadataPass({options.tritonMetadataOutput}));
  pm.addPass(createArithToLLVMConversionPass());
}
#endif

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerLowerTritonPipeline() {
#if BISHENGIR_ENABLE_TRITON_COMPILE
  PassPipelineRegistration<LowerTritonPipelineOptions>(
      "lower-triton-pipeline", "Lower Triton to LLVM",
      [](OpPassManager &pm, const LowerTritonPipelineOptions &options) {
        buildLowerTritonPipeline(pm, options);
      });
#endif
}

} // namespace triton
} // namespace bishengir
