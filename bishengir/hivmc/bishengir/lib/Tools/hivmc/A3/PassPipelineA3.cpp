//===- PassPipeline.cpp - HIVMC pass pipeline -------------------*- C++ -*-===//
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


#include "bishengir/Tools/hivmc/PassPipelineA3.h"
#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Pipelines/Passes.h"
#include "bishengir/Dialect/HACC/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Pipelines/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Tools/hivmc/HIVMC.h"
#include "bishengir/Transforms/Passes.h"


#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
// using namespace bishengir;

namespace {

// NOT USING IT ... DEBATABLE !!!
// struct FinalizeHIVMToLLVMPipelineOptions
//     : public mlir::PassPipelineOptions<FinalizeHIVMToLLVMPipelineOptions> {
// #define GEN_FINALIZE_HIVM_TO_LLVM_OPTION_REGISTRATION
// // #include "bishengir/Tools/hivmc/PassPipelineOptions.cpp.inc"
// };

/// Adds the "FinalizeHIVMToLLVM" pipeline to the `OpPassManager`. This is the
/// standard pipeline for finalizing conversion from HIVM dialect to LLVM IR.
void buildFinalizeHIVMToLLVMPipeline(
    OpPassManager &pm, const bishengir::HIVMCMainConfig &config) {
  pm.addPass(createArithToHIVMLLVMConversionPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createLowerAffinePass());
  ConvertHIVMToLLVMOptions hivm2llvmOptions;
  hivm2llvmOptions.useBarePtrCallConv = config.tryBarePtrCallConvForStaticShape();
  pm.addPass(arith::createArithExpandOpsPass());
  pm.addPass(createConvertHIVMToLLVMPass(hivm2llvmOptions));
  pm.addPass(createArithToHIVMLLVMConversionPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createConvertMemRefToBarePtrPass());
  pm.addPass(createAppendUsePrintDebugDataPass());
  if (config.shouldEnableSanitizer()) {
    pm.addPass(createAppendSanitizerDataPass());
  }
  if (config.shouldEnableSanitizer() || config.shouldEnableDebugInfo() ||
      config.shouldEnableDebugVariables()) {
    // Add lineNo debug info to the IR
    // The info here Corresponds to that in the source file.
    pm.addPass(LLVM::createDIScopeForLLVMFuncOpPass());
  }
  if (config.shouldEnableDebugVariables()) {
    pm.addPass(LLVM::createLLVMDILocalVariablePass());
  }
}
} // namespace


namespace bishengir {
// void setupFinalizeHIVMToLLVMPipelineOptions(
//     FinalizeHIVMToLLVMPipelineOptions &options, const HIVMCMainConfig &config) {
// #define GEN_FINALIZE_HIVM_TO_LLVM_OPTION_SETUP
// // #include "bishengir/Tools/hivmc/ConfigUtils.cpp.inc"
// }

void buildBiShengHIRHIVMToLLVMPipeline(mlir::OpPassManager &pm,
                                       const HIVMCMainConfig &config) {
  // Host to LLVM pipeline
  if (config.shouldCompileHost()) {
    hacc::buildLowerHACCToLLVMPipeline(pm, config.hostOutputFile());
    return;
  }

  // DEBATABLE !!!
  // Device to LLVM pipeline
  // FinalizeHIVMToLLVMPipelineOptions options;
  // setupFinalizeHIVMToLLVMPipelineOptions(options, config);
  // DEBATABLE !!!

  buildFinalizeHIVMToLLVMPipeline(pm, config);
}

} // namespace bishengir

// void bishengir::registerHIVMCCompilePass() {
//   PassRegistration<bishengir::HIVMCCompilePass>();
// }
