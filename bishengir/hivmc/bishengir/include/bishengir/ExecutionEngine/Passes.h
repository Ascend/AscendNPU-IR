//===- Passes.h - Execution Engine pass entrypoints -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_EXECUTION_ENGINE_TRANSFORMS_PASSES_H
#define BISHENGIR_EXECUTION_ENGINE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL
#include "bishengir/ExecutionEngine/Passes.h.inc"

namespace execution_engine {

/// Create a pass to create wrappers for the only host related functions.
std::unique_ptr<Pass> createCreateHostMainPass(
    const ExecutionEngineHostMainCreatorOptions &options = {});

/// Create a pass to convert HIVM operations to upstream dialect's equivalent.
std::unique_ptr<Pass> createConvertHIVMToUpstreamPass();

struct CPURunnerPipelineOptions
    : public PassPipelineOptions<CPURunnerPipelineOptions> {
  CPURunnerPipelineOptions() = default;

  CPURunnerPipelineOptions(const CPURunnerPipelineOptions &other)
      : CPURunnerPipelineOptions() {
    *this = other;
  }

  CPURunnerPipelineOptions &operator=(const CPURunnerPipelineOptions &other) {
    if (this == &other) {
      return *this;
    }
    this->copyOptionValuesFrom(other);
    return *this;
  }

  PassOptions::Option<std::string> wrapperName{
      *this, "wrapper-name",
      ::llvm::cl::desc("Name of the wrapper function to be generated for the "
                       "single host entry function provided."),
      ::llvm::cl::init("main")};
};

/// Create a pipeline to lower everything to be compatible with the CPU runner.
void buildCPURunnerPipeline(OpPassManager &pm,
                            const CPURunnerPipelineOptions &options = {});

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/ExecutionEngine/Passes.h.inc"

/// Register all Execution-Engine related pipelines.
void registerAllPipelines();

} // namespace execution_engine
} // namespace mlir

#endif // BISHENGIR_EXECUTION_ENGINE_TRANSFORMS_PASSES_H
