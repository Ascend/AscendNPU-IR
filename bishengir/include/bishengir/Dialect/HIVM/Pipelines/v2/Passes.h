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

  PassOptions::Option<bool> enablePreload{*this, "enable-preload",
                                          llvm::cl::desc("Run create-preload"),
                                          llvm::cl::init(false)};

  PassOptions::Option<bool> enableMultiBuffer{
      *this, "enable-multi-buffer",
      llvm::cl::desc("Mark/enable multi-buffering on the unsplit MIX"),
      llvm::cl::init(false)};
};

/// Phase builders shared by the opt pipeline and the compile driver.
void buildV2TensorPhase(OpPassManager &pm, const V2PipelineOptions &options);
void buildV2BufferizationPhase(OpPassManager &pm,
                               const V2PipelineOptions &options);
void buildV2PostBufferizationPhase(OpPassManager &pm,
                                   const V2PipelineOptions &options);
void buildV2PerCoreTail(OpPassManager &pm);

/// Add the "hivm-v2-pipeline" pipeline to `pm`.
void buildV2HIVMPipeline(OpPassManager &pm, const V2PipelineOptions &options);

/// Register the v2 pipeline.
void registerV2HIVMPipelines();

} // namespace v2
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_PIPELINES_V2_PASSES_H
