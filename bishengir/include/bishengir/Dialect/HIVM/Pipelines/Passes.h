//===- Passes.h - HIVM pipeline entry points --------------------*- C++ -*-===//
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
#define GEN_HIVM_OPTION_REGISTRATION
#include "bishengir/Tools/bishengir-compile/PassPipelineOptions.cpp.inc"

  PassOptions::Option<std::string> target{
      *this, "target", llvm::cl::desc("Target device name"),
      llvm::cl::init("Ascend910B1")};
};

struct ConvertToHIVMPipelineOptions
    : public mlir::PassPipelineOptions<ConvertToHIVMPipelineOptions> {
#define GEN_HFUSION_TO_HIVM_OPTION_REGISTRATION
#include "bishengir/Tools/bishengir-compile/PassPipelineOptions.cpp.inc"

  PassOptions::Option<bool> enableRegBaseHIVMPipe{
      *this, "enable-regbase-hivmpipe",
      llvm::cl::desc("Enable hivmpipeline on RegBase"), llvm::cl::init(false)};
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the "OptimizeHIVM" pipeline to the `OpPassManager`. This is the
/// standard pipeline for optimizing HIVM dialect IR.
void buildOptimizeHIVMPipeline(OpPassManager &pm,
                               const HIVMPipelineOptions &options);
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

/// Adds sync-block-lock finalize passes (mark subblock + insert free_lock_var)
/// before HIVM to Standard conversion.
void addSyncBlockLockFinalizePasses(OpPassManager &pm);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_PIPELINES_PASSES_H
