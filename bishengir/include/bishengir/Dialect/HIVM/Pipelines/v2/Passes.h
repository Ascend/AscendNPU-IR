//===- Passes.h - HIVM v2 pipeline entry points -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Entry points for the HIVM v2 pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_PIPELINES_V2_PASSES_H
#define BISHENGIR_DIALECT_HIVM_PIPELINES_V2_PASSES_H

#include "mlir/Pass/PassOptions.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hivm {
namespace v2 {

struct V2PipelineOptions : public mlir::PassPipelineOptions<V2PipelineOptions> {
  PassOptions::Option<std::string> target{*this, "target",
                                          llvm::cl::desc("Target device name"),
                                          llvm::cl::init("Ascend950PR_9579")};

  /// Append the device spec from `target` (bishengir-opt only).
  PassOptions::Option<bool> setTargetSpec{
      *this, "set-target-spec",
      llvm::cl::desc("Append the target device spec from --target"),
      llvm::cl::init(true)};

  PassOptions::Option<bool> enableVectorize{
      *this, "enable-vectorize",
      llvm::cl::desc("Run vectorization on the unsplit MIX"),
      llvm::cl::init(true)};

  PassOptions::Option<bool> enableCrossCoreGSS{
      *this, "enable-cross-core-gss",
      llvm::cl::desc("Solve cross-core sync once on the unsplit MIX"),
      llvm::cl::init(true)};

  PassOptions::Option<bool> enablePreload{*this, "enable-preload",
                                          llvm::cl::desc("Run create-preload"),
                                          llvm::cl::init(false)};

  PassOptions::Option<bool> enableMultiBuffer{
      *this, "enable-multi-buffer",
      llvm::cl::desc("Mark/enable multi-buffering on the unsplit MIX"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> useBlockAllSync{
      *this, "enable-block-all-sync",
      llvm::cl::desc("Use inject-block-sync{block-all-sync} instead of GSS"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> ghostSplitMix{
      *this, "ghost-split-mix",
      llvm::cl::desc("Run the split stage (ghost: keeps the kernel MIX)"),
      llvm::cl::init(true)};

  /// Run the verifier after this phase (none, tensor, bufferized, planned,
  /// solved, split, tail).
  /// TODO(v2): debug-only. The stage verifier must be removed from the final
  /// pipeline once the v2 stages are stable.
  PassOptions::Option<std::string> verifyStage{
      *this, "verify-stage",
      llvm::cl::desc("Run hivm-v2-verify-stage after the named phase"),
      llvm::cl::init("none")};
};

/// Phase builders shared by the opt pipeline and the compile driver.
void buildV2TensorPhase(OpPassManager &pm, const V2PipelineOptions &options);
void buildV2BufferizationPhase(OpPassManager &pm,
                               const V2PipelineOptions &options);
void buildV2PostBufferizationPhase(OpPassManager &pm,
                                   const V2PipelineOptions &options);
void buildV2PerCoreTail(OpPassManager &pm, const V2PipelineOptions &options);

/// Add the "hivm-v2-pipeline" pipeline to `pm`.
void buildV2HIVMPipeline(OpPassManager &pm, const V2PipelineOptions &options);

/// Register the v2 pipeline.
void registerV2HIVMPipelines();

} // namespace v2
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_PIPELINES_V2_PASSES_H
