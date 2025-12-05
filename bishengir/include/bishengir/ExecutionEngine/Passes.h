//===- Passes.h - Execution Engine pass entrypoints -------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
    if (this != &other) {
      this->copyOptionValuesFrom(other);
    }
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
