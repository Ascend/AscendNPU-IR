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

#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace hfusion {

struct HFusionPipelineOptions
    : public mlir::PassPipelineOptions<HFusionPipelineOptions> {
#define GEN_HFUSION_OPTION_REGISTRATION
#include "bishengir/Tools/bishengir-compile/PassPipelineOptions.cpp.inc"

  PassOptions::Option<std::string> target{
      *this, "target", llvm::cl::desc("Target device name"),
      llvm::cl::init("Ascend910B1")};

  bool insertFFTS{true};

  PassOptions::Option<std::string> externalTilingFuncPath{
      *this, "external-tiling-func-path",
      llvm::cl::desc("auto add external tiling func"), llvm::cl::init("-")};
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
