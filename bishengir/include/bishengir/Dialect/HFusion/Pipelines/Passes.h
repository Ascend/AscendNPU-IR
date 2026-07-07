//===- Passes.h - HFusion pipeline entry points -----------------*- C++ -*-===//
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
// This header file defines prototypes of all HFusion pipelines.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_PIPELINES_PASSES_H
#define BISHENGIR_DIALECT_HFUSION_PIPELINES_PASSES_H

#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace hfusion {

/// TreeReduce processing mode
enum class TreeReduceMode {
  Off, // disable TreeReduce RA/AR processing
  RA,  // row-reduction, dim=0
  AR,  // column-reduction, dim=1
  All, // enable both RA and AR processing
};

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
