//===- Passes.h - HFusion pipeline entry points -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all HFusion pipelines.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_PIPELINES_PASSES_H
#define BISHENGIR_DIALECT_HFUSION_PIPELINES_PASSES_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace hfusion {

struct HFusionPipelineOptions
    : public mlir::PassPipelineOptions<HFusionPipelineOptions> {
  // -------------------------------------------------------------------------//
  //                       feature control options                            //
  // -------------------------------------------------------------------------//
  PassOptions::Option<bool> enableManageHostResources{
      *this, "enable-manage-host-resources",
      llvm::cl::desc("Enable managing resource for Host functions."),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableTritonKernelCompile{
      *this, "enable-triton-kernel-compile",
      llvm::cl::desc("Enable Triton kernel compilation"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> disableFFTS{*this, "disable-ffts",
                                        llvm::cl::desc("Force disabling FFTS"),
                                        llvm::cl::init(false)};

  PassOptions::Option<bool> disableHFusionVectorize{
      *this, "disable-hfusion-vectorize",
      llvm::cl::desc("Disable hfusion auto vectorize"), llvm::cl::init(false)};

  bool insertFFTS{true};

  PassOptions::Option<bool> enableMultiKernel{
      *this, "multi-kernel", llvm::cl::desc("Enable multi-kernel fusion"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableSymbolAnalysis{
      *this, "enable-symbol-analysis", llvm::cl::desc("Enable symbol analysis"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableAutoVectorizeV2{
      *this, "enable-auto-vectorize-v2",
      llvm::cl::desc("Enable auto vectorize v2"), llvm::cl::init(true)};

  PassOptions::Option<bool> enableVFFusion{*this, "enable-vf-fusion",
                                           llvm::cl::desc("Enable vf fusion"),
                                           llvm::cl::init(false)};

  PassOptions::Option<bool> enableTreeReduce{
      *this, "enable-tree-reduce", llvm::cl::desc("Enable tree reduce"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableMultipleConsumerFusion{
      *this, "enable-multiple-consumer-fusion",
      llvm::cl::desc("Enable multiple consumer fusion in AutoVectorizeV2"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> skipScope{
      *this, "skip-scope",
      llvm::cl::desc("Skip passes like flattenOps when scope exists"),
      llvm::cl::init(true)};

  PassOptions::Option<std::string> target{*this, "target",
                                          llvm::cl::desc("Target device name"),
                                          llvm::cl::init("Ascend910B1")};
  PassOptions::Option<bool> enableHighPrecision{
      *this, "enable-high-precision",
      llvm::cl::desc(
          "Enable high precision calculation for sin/cos in HFusion"),
      llvm::cl::init(true)};

  // -------------------------------------------------------------------------//
  //                  optimization control options                            //
  // -------------------------------------------------------------------------//
  PassOptions::Option<bool> enableLayoutOptimization{
      *this, "enable-layout-optimization",
      llvm::cl::desc("Enable Layout Optimization"), llvm::cl::init(false)};

  PassOptions::Option<bool> enableMixedCV{
      *this, "enable-mixed-cv", llvm::cl::desc("Enable mixed CV compilation"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableFuseReductionIntoLoop{
      *this, "enable-fuse-reduction-into-loop",
      llvm::cl::desc("Enable fuse post-loop reductions into the loop body"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableAutoMultiBuffer{
      *this, "enable-auto-multi-buffer",
      llvm::cl::desc("Enable automatic multi-buffer optimization"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableDropUnitDims{
      *this, "enable-drop-unit-dims",
      llvm::cl::desc("enable drop-unit-dims pass"), llvm::cl::init(true)};

  PassOptions::Option<bool> enableFlatten{*this, "enable-flatten",
                                          llvm::cl::desc("enable flatten pass"),
                                          llvm::cl::init(true)};

  PassOptions::Option<bool> enableAutoBindSubBlock{
      *this, "enable-auto-bind-sub-block",
      llvm::cl::desc("Enable auto bind sub block"), llvm::cl::init(true)};

  PassOptions::Option<bool> enableDeterministicComputing{
      *this, "enable-deterministic-computing",
      llvm::cl::desc("If enabled, the computation result is deterministic. If "
                     "disabled, we will enable extra optimizations that might "
                     "boost performance, e.g. bind reduce to multiple cores. "
                     "However, the result will be non-deterministic."),
      llvm::cl::init(true)};

  PassOptions::Option<bool> enableOpsReorder{
      *this, "enable-ops-reorder",
      llvm::cl::desc("Enable operations reordering"), llvm::cl::init(true)};

  PassOptions::Option<int32_t> maxHorizontalFusionSize{
      *this, "max-horizontal-fusion-size",
      llvm::cl::desc("Maximum horizontal fusion size (-1 for unlimited)"),
      llvm::cl::init(-1)};
  PassOptions::Option<int32_t> maxFusedElementwiseOps{
      *this, "max-fused-elementwise-ops",
      llvm::cl::desc("Maximum number of elementwise ops to fuse in "
                     "PreVectorizationFusion (-1 for unlimited)"),
      llvm::cl::init(-1)};
  PassOptions::Option<int32_t> maxFusedOpsInAutoVectorizeV2{
      *this, "max-fused-ops-in-auto-vectorize-v2",
      llvm::cl::desc("Maximum number of ops to fuse in AutoVectorizeV2 "
                     "(-1 uses pass default)"),
      llvm::cl::init(-1)};

  PassOptions::Option<int64_t> maxBufferCntTuning{
      *this, "max-buffer-count-tuning",
      llvm::cl::desc("Maximum buffer count for tuning"), llvm::cl::init(0)};

  PassOptions::ListOption<int64_t> cubeTilingTuning{
      *this, "cube-tiling-tuning", llvm::cl::desc("cube tiling for tuning")};

  PassOptions::Option<bool> enableCountBufferDmaOpt{
      *this, "enable-count-buffer-dma-opt",
      llvm::cl::desc("If enabled, the buffer used by DMA operations will not be"
                     "reused by Vector operations"),
      llvm::cl::init(false)};

  PassOptions::Option<std::string> externalTilingFuncPath{
      *this, "external-tiling-func-path",
      llvm::cl::desc("auto add external tiling func"), llvm::cl::init("-")};

  PassOptions::Option<std::string> injectIrFromFile{
      *this, "inject-ir-from-file",
      llvm::cl::desc(
          "Path to IR file for inject-ir pass; when set, matching "
          "functions are replaced with those from the file for debug"),
      llvm::cl::init("")};

  /// TODO : remove it after add platform info
  PassOptions::Option<unsigned> blockDim{*this, "block-dim",
                                         llvm::cl::desc("Block dimension size"),
                                         llvm::cl::init(1)};
};

void buildHFusionPipelines(OpPassManager &pm,
                           const HFusionPipelineOptions &options);

void buildHFusionRegBasePipeline(OpPassManager &pm,
                                 const HFusionPipelineOptions &options);

void registerLowerHFusionPipelines();

bool enableSIMDVFFusion(const HFusionPipelineOptions &options);
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_PIPELINES_PASSES_H
