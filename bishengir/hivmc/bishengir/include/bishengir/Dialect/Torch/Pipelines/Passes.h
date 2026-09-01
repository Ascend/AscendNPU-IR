//===- Passes.h - BiShengIR Torch pipeline entry points ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all BiShengIR Torch pipelines.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_TORCH_PIPELINES_PASSES_H
#define BISHENGIR_DIALECT_TORCH_PIPELINES_PASSES_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace bishengir {
struct TorchToNamedOpPipelineOptions
    : public mlir::PassPipelineOptions<TorchToNamedOpPipelineOptions> {
  // -------------------------------------------------------------------------//
  //                       feature control options
  // -------------------------------------------------------------------------//
  PassOptions::Option<bool> ensureNoImplicitBroadcast{
      *this, "ensure-no-implicit-broadcast",
      llvm::cl::desc("Whether to ensure that there is no implicit broadcast "
                     "semantics. If there is a dynamic to dynamic dim "
                     "broadcast, raise a runtime error."),
      llvm::cl::init(false)};
};

void registerTorchToHFusionPipelines();

/// Creates a pipeline that lowers from the torch backend contract to the
/// linalg-on-tensors backend contract.
void createTorchBackendToNamedOpBackendPipeline(
    mlir::OpPassManager &pm, const TorchToNamedOpPipelineOptions &options);
} // namespace bishengir

#endif // BISHENGIR_DIALECT_TORCH_PIPELINES_PASSES_H
