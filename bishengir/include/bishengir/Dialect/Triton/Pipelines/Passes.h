//===- Passes.h - Triton pipeline entry points ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all Triton pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TRITON_PIPELINES_PASSES_H
#define BISHENGIR_DIALECT_TRITON_PIPELINES_PASSES_H

#include "bishengir/Tools/bishengir-compile/BiShengIRCompile.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h"

namespace bishengir {
namespace triton {

struct LowerTritonPipelineOptions
    : public mlir::PassPipelineOptions<LowerTritonPipelineOptions> {
  PassOptions::Option<int32_t> numWarps{
      *this, "num-warps", llvm::cl::desc("Number of warps"), llvm::cl::init(4)};
  PassOptions::Option<int32_t> threadsPerWarp{
      *this, "threads-per-warp", llvm::cl::desc("Number of threads per warp"),
      llvm::cl::init(32)};
  PassOptions::Option<int32_t> sharedDynamicSize{
      *this, "shared-memory-size",
      llvm::cl::desc("max size of shared memory available for simt vf"),
      llvm::cl::init(122880)};
  PassOptions::Option<int32_t> enableBishengirSimtOptimization{
      *this, "enable-bishengir-simt-optimization",
      llvm::cl::desc("enable which bishengir simt optimization"),
      llvm::cl::init(900101)};
  PassOptions::Option<bool> disableDecomposeReduction{
      *this, "disable-decompose-reduction",
      llvm::cl::desc("disable decompose reduction"), llvm::cl::init(false)};
  PassOptions::Option<bool> disableReorderInstruction{
      *this, "disable-reorder-instruction",
      llvm::cl::desc("disable reorder instruction"), llvm::cl::init(false)};
  PassOptions::Option<bool> disableSinkDPXLoad{
      *this, "disable-sink-dpx-load",
      llvm::cl::desc("disable post-lowering instruction scheduling that "
                     "reduces register pressure by interleaving "
                     "load-compute-store chains"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> enableSimtReorderInstruction{
      *this, "enable-simt-reorder-instruction",
      llvm::cl::desc("enable simt reorder instruction pattern"), llvm::cl::init(false)};
  PassOptions::Option<bool> enableSIMTFastDiv{
      *this, "enable-simt-fast-div",
      llvm::cl::desc("enable SIMT fast division optimization"),
      llvm::cl::init(true)};
  PassOptions::Option<std::string> tritonMetadataOutput{
      *this, "triton-metadata-output",
      llvm::cl::desc("File to dump triton metadata. -- means stdout"),
      llvm::cl::init("")};
  mlir::triton::proton::ConvertProtonToProtonGPUOptions protonGPUCompileConfig;
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the pipeline to lower Triton dialect to LLVM Dialect.
void buildLowerTritonPipeline(mlir::OpPassManager &pm,
                              const LowerTritonPipelineOptions &options);

/// Register a pipeline to lower Triton dialect.
void registerLowerTritonPipeline();

} // namespace triton
} // namespace bishengir

#endif // BISHENGIR_DIALECT_TRITON_PIPELINES_PASSES_H
