//===- Passes.h - HIVM pipeline entry points --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all HIVM pipelines.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_PIPELINES_PASSES_H
#define BISHENGIR_DIALECT_HIVM_PIPELINES_PASSES_H

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace hivm {

struct HIVMPipelineOptions
    : public mlir::PassPipelineOptions<HIVMPipelineOptions> {
  // -------------------------------------------------------------------------//
  //                       feature control options                            //
  // -------------------------------------------------------------------------//
  PassOptions::Option<bool> enableTritonKernelCompile{
      *this, "enable-triton-kernel-compile",
      llvm::cl::desc("Enable triton kernel compile"), llvm::cl::init(false)};
  PassOptions::Option<bool> enableDotScaledCompile{
      *this, "enable-dot-scaled-compile",
      llvm::cl::desc("Enable dot scaled compile"), llvm::cl::init(false)};
  PassOptions::Option<bool> enableMixedCV{
      *this, "enable-mixed-cv", llvm::cl::desc("Enable mixed CV compilation"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> enableLayoutOptimization{
      *this, "enable-layout-optimization",
      llvm::cl::desc("Enable Layout Optimization"), llvm::cl::init(false)};
  PassOptions::Option<bool> enableFullSIMTCompile{
      *this, "pure-simt", llvm::cl::desc("Enable full simt compile"),
      llvm::cl::init(false)};

  // -------------------------------------------------------------------------//
  //                  optimization control options                            //
  // -------------------------------------------------------------------------//
  PassOptions::Option<bool> enableAutoMultiBuffer{
      *this, "enable-auto-multi-buffer",
      llvm::cl::desc("Enable automatic multi-buffer optimization"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> limitAutoMultiBufferOnlyForLocalBuffer{
      *this, "limit-auto-multi-buffer-only-for-local-buffer",
      llvm::cl::desc("When enable-auto-multi-buffer = true, limit it only work "
                     "for local buffer"),
      llvm::cl::init(false)};

  PassOptions::Option<MultiBufferStrategy> limitAutoMultiBufferOfLocalBuffer{
      *this, "limit-auto-multi-buffer-of-local-buffer",
      llvm::cl::desc("When enable-auto-multi-buffer = true, limit local buffer "
                     "mode"),
      llvm::cl::values(
          clEnumValN(MultiBufferStrategy::NO_LIMIT, "no-limit", "No limit"),
          clEnumValN(MultiBufferStrategy::CUBE_NO_L0C, "no-l0c",
                     "Disable l0c multi buffer")),
      llvm::cl::init(MultiBufferStrategy::CUBE_NO_L0C)};

  PassOptions::Option<MultiBufferStrategy> limitMixAutoMultiBufferBuffer{
      *this, "limit-mix-auto-multi-buffer-buffer",
      llvm::cl::desc("When enable-auto-multi-buffer = true, limit it only work"
                     "for NO_LIMIT, ONLY_CUBE, ONLY_VECTOR"),
      llvm::cl::values(clEnumValN(MultiBufferStrategy::NO_LIMIT, "no-limit",
                                  "limited to cube and vector"),
                       clEnumValN(MultiBufferStrategy::ONLY_CUBE, "only-cube",
                                  "limited to cube"),
                       clEnumValN(MultiBufferStrategy::ONLY_VECTOR,
                                  "only-vector", "limited to vector")),
      llvm::cl::init(MultiBufferStrategy::ONLY_CUBE)};

  PassOptions::Option<unsigned> workspaceMultiBufferNum{
      *this, "set-workspace-multibuffer",
      llvm::cl::desc("Set multibuffer number of workspace, defaults to 2"),
      llvm::cl::init(2)};

  PassOptions::Option<bool> enableAutoBindSubBlock{
      *this, "enable-auto-bind-sub-block",
      llvm::cl::desc("Enable auto bind sub block"), llvm::cl::init(true)};

  PassOptions::Option<bool> enableAutoCVBalance{
      *this, "enable-auto-cv-balance",
      llvm::cl::desc("Enable balancing when pipelining CV ops"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableAutoStorageAlign{
      *this, "enable-auto-storage-align",
      llvm::cl::desc("Enable mark/enable storage align"), llvm::cl::init(true)};

  PassOptions::Option<bool> enableGlobalWorkspaceReuse{
      *this, "enable-global-workspace-reuse",
      llvm::cl::desc("Enable global workspace reuse"), llvm::cl::init(false)};

  PassOptions::Option<bool> enablePrintMemoryAllocatedSize{
      *this, "enable-print-memory-allocated-size",
      llvm::cl::desc("Enable print memory allocated size"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> disableTightlyCoupledBufferReuse{
      *this, "disable-tightly-coupled-buffer-reuse",
      llvm::cl::desc("Disable tightly coupled buffer reuse"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableHIVMInjectBarrierAllSync{
      *this, "enable-hivm-inject-barrier-all-sync",
      llvm::cl::desc("Enable barrier all mode for HIVM inject sync"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableInjectBlockAllSync{
      *this, "enable-hivm-inject-block-all-sync",
      llvm::cl::desc("Enable inject all block sync for HIVM injectBlockSync"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> disableAutoInjectBlockSync{
      *this, "disable-auto-inject-block-sync",
      llvm::cl::desc("Disable auto generating sync block wait/set by "
                     "InjectBlockSync pass"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableHIVMGraphSyncSolver{
      *this, "enable-hivm-graph-sync-solver",
      llvm::cl::desc("Enable HIVM Graph-Sync-Solver Auto-Sync Pass"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableUnitFlagSync{
      *this, "enable-hivm-unit-flag-sync",
      llvm::cl::desc("Enable inject sync pass to use unit-flag modes for "
                     "synchronization"),
      llvm::cl::init(false)};

  PassOptions::Option<std::string> target{*this, "target",
                                          llvm::cl::desc("Target device name"),
                                          llvm::cl::init("Ascend910B1")};

  PassOptions::Option<bool> enableCodeMotion{
      *this, "enable-code-motion",
      llvm::cl::desc("Enable code-motion/subset-hoist"), llvm::cl::init(true)};

  PassOptions::Option<int32_t> enableVfMergeLevel{
      *this, "enable-vf-merge-level",
      llvm::cl::desc("Enable merging vector function"), llvm::cl::init(1)};

  PassOptions::Option<bool> enableDirectHIVMLowering{
      *this, "enable-direct-hivm-lowering",
      llvm::cl::desc("enable-direct-hivm-lowering"), llvm::cl::init(false)};

  PassOptions::Option<bool> enableFusedMultiplyAdd{
      *this, "enable-fused-multiply-add",
      llvm::cl::desc("Enable fused multiply add"), llvm::cl::init(false)};

  PassOptions::Option<bool> enableND2NZOnVector{
      *this, "enable-hivm-nd2nz-on-vector",
      llvm::cl::desc("Enable hivm nd2nz on vector"), llvm::cl::init(false)};

  PassOptions::Option<bool> enableAutoBlockifyLoop{
      *this, "enable-auto-blockify-loop",
      llvm::cl::desc("Enable auto loop on blocks when logical blocknum is "
                     "larger than physical one"),
      llvm::cl::init(false)};

  PassOptions::Option<int> enableBishengirSimtOptimization{
      *this, "enable-bishengir-simt-optimization",
      llvm::cl::desc("Enable bishengir simt optimization"),
      llvm::cl::init(900101)};

  PassOptions::Option<bool> useDPX{
      *this, "use-dpx",
      llvm::cl::desc("Enable simt lowering through the DPX dialect."),
      llvm::cl::init(true)};

  PassOptions::Option<int> disableDecomposeReduction{
      *this, "disable-decompose-reduction",
      llvm::cl::desc("Disable decompose reduction"), llvm::cl::init(false)};

  PassOptions::Option<int> disableReorderInstruction{
      *this, "disable-reorder-instruction",
      llvm::cl::desc("Disable reorder instruction"), llvm::cl::init(false)};

  PassOptions::Option<int> enableSimtReorderInstruction{
      *this, "enable-simt-reorder-instruction",
      llvm::cl::desc("Enable SIMT reorder instruction pattern"), llvm::cl::init(false)};

  PassOptions::Option<int> simtVFDynamicSize{
      *this, "simt-vf-dynamic-size",
      llvm::cl::desc("Dynamic ub size(KB) for simt VF. Default is 216"),
      llvm::cl::init(216)};

  PassOptions::Option<int> maxReductionSplitNum{
      *this, "max-reduction-split",
      llvm::cl::desc("Max split times for reductionLoop. Default is 1"),
      llvm::cl::init(1)};

  PassOptions::Option<std::string> injectIrFromFile{
      *this, "inject-ir-from-file",
      llvm::cl::desc("Path to IR file for inject-ir pass; when set, matching "
                     "functions are replaced with those from the file for debug"),
      llvm::cl::init("")};
};

struct ConvertToHIVMPipelineOptions
    : public mlir::PassPipelineOptions<ConvertToHIVMPipelineOptions> {
  PassOptions::Option<bool> enableTritonKernelCompile{
      *this, "enable-triton-kernel-compile",
      llvm::cl::desc("Enable Triton kernel compilation"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableAutoBlockifyLoop{
      *this, "enable-auto-blockify-loop",
      llvm::cl::desc("Enable auto loop on blocks when logical blocknum is "
                     "larger than physical one"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableRegBaseHIVMPipe{
      *this, "enable-regbase-hivmpipe",
      llvm::cl::desc("Enable hivmpipeline on RegBase"), llvm::cl::init(false)};
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the "ConvertToHIVM" pipeline to the `OpPassManager`. This is the
/// standard pipeline for lowering from other dialects to HIVM dialect.
void buildConvertToHIVMPipeline(mlir::OpPassManager &pm,
                                const ConvertToHIVMPipelineOptions &options);

void buildHIVMTensorOptimizations(
    OpPassManager &pm, const HIVMPipelineOptions &hivmPipelineOptions);

/// Adds the "LowerHIVM" pipeline to the `OpPassManager`. This is the
/// standard pipeline for lowering from HIVM dialect to LLVM IR.
/// \note This includes the `ConvertToHIVM` pipeline.
void buildLowerHIVMPipelines(OpPassManager &pm,
                             const HIVMPipelineOptions &hivmPipelineOptions);

/// Register the "ConvertToHIVM" pipeline.
void registerConvertToHIVMPipelines();

/// Register the "LowerHIVM" pipeline.
void registerLowerHIVMPipelines();

/// A canonicalization pipeline for HIVM pipeline.
void canonicalizationHIVMPipeline(OpPassManager &pm);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_PIPELINES_PASSES_H
