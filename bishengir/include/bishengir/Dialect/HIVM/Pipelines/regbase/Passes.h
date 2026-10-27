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
#ifndef BISHENGIR_DIALECT_HIVM_PIPELINES_REGBASE_PASSES_H
#define BISHENGIR_DIALECT_HIVM_PIPELINES_REGBASE_PASSES_H

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/HIVMPipelineBase.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace hivm {
namespace regbase {

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

/// Register each RegBase HIVM pass for the kernel kinds in its HIVMPass
/// `kernelKinds` list, and the callback that runs InferFuncCoreType after a
/// pass with `classifyKernel` set. Both come from HIVM Passes.td.
/// Idempotent; called by the pipeline builders themselves.
void registerKernelKindPassesForRegbase();

/// Register the "ConvertToHIVM" pipeline.
void registerConvertToHIVMPipelines();

/// Register the "LowerHIVM" pipeline.
void registerLowerHIVMPipelines();

/// Adds sync-block-lock finalize passes (insert infer-sync-block-lock-num/init
/// + lower create_sync_block_lock + mark subblock + insert free_lock_var)
/// before HIVM to Standard conversion.
void addSyncBlockLockFinalizePasses(OpPassManager &pm);

} // namespace regbase
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_PIPELINES_REGBASE_PASSES_H
