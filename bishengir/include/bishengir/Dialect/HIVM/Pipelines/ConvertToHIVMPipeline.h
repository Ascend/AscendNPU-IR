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
#ifndef BISHENGIR_DIALECT_HIVM_PIPELINES_CONVERT_TO_HIVM_PIPELINE_H
#define BISHENGIR_DIALECT_HIVM_PIPELINES_CONVERT_TO_HIVM_PIPELINE_H

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace hivm {

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
void buildConvertToHIVMPipeline(OpPassManager &pm,
                                const ConvertToHIVMPipelineOptions &options);

/// Register the "ConvertToHIVM" pipeline.
void registerConvertToHIVMPipelines();

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_PIPELINES_CONVERT_TO_HIVM_PIPELINE_H
