//===- ConvertToHIVMPipeline.cpp - HIVM pipelines -------------------------===//
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

#include "bishengir/Dialect/HIVM/Pipelines/ConvertToHIVMPipeline.h"
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace hivm {

void buildConvertToHIVMPipeline(OpPassManager &pm,
                                const ConvertToHIVMPipelineOptions &options) {
  ConvertHFusionToHIVMOptions hfs2hivmOptions;
  hfs2hivmOptions.mmMapMode = options.enableTritonKernelCompile
                                  ? hfusion::MmMapMode::MacroInstr
                                  : hfusion::MmMapMode::CoreOp;
  hfs2hivmOptions.isEnableUbufSaving = options.enableUbufSaving;
  pm.addPass(createHFusionToHIVMConversionPass(hfs2hivmOptions));
  if (options.enableTritonKernelCompile) {
    pm.addPass(createTritonGlobalKernelArgsToHIVMOpPass());
  }
  pm.addPass(createTensorToHIVMConversionPass());
  pm.addPass(createConvertToHIVMOpPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerConvertToHIVMPipelines() {
  PassPipelineRegistration<ConvertToHIVMPipelineOptions>(
      "convert-to-hivm-pipeline", "convert to hivm pipeline",
      [](OpPassManager &pm, const ConvertToHIVMPipelineOptions &options) {
        buildConvertToHIVMPipeline(pm, options);
      });
}

} // namespace hivm
} // namespace mlir
