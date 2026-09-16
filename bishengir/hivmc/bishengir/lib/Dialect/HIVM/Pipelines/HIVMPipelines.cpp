//===- HIVMPipelines.cpp - HIVM pipelines ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/ArithToAffine/ArithToAffine.h"
#include "bishengir/Conversion/HFusionToHIVM/HFusionToHIVMPass.h"
#include "bishengir/Conversion/TensorToHIVM/TensorToHIVM.h"
#include "bishengir/Dialect/Annotation/Transforms/Passes.h"
#include "bishengir/Dialect/Arith/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "bishengir/Dialect/SCF/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace hivm {

#define ADD_CANONICALIZER_PASS                                                 \
  CanonicalizerOptions options;                                                \
  options.enableExtendedPattern = true;                                        \
  pm.addPass(createCanonicalizerPass(options))

#define ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS                             \
  pm.nest<func::FuncOp>().addPass(createCanonicalizerPass(options))

void canonicalizationHIVMPipeline(OpPassManager &pm) {
  pm.addPass(createArithToAffineConversionPass());
  ADD_CANONICALIZER_PASS;
  pm.addPass(createSCFForLoopCanonicalizationPass());
  pm.addPass(createCSEPass());
  ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
  pm.nest<func::FuncOp>().addPass(createHIVMOptSinglePointPass());
  ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
  pm.nest<func::FuncOp>().addPass(memref::createDeadStoreEliminationPass());
}

void buildConvertToHIVMPipeline(OpPassManager &pm,
                                const ConvertToHIVMPipelineOptions &options) {
  ConvertHFusionToHIVMOptions hfs2hivmOptions;
  hfs2hivmOptions.mmMapMode = options.enableTritonKernelCompile
                                  ? hfusion::MmMapMode::MacroInstr
                                  : hfusion::MmMapMode::CoreOp;

  if (options.enableRegBaseHIVMPipe)
    pm.nest<func::FuncOp>().addPass(createCanonicalizerPass());
  pm.addPass(createHFusionToHIVMConversionPass(hfs2hivmOptions));
  if (options.enableTritonKernelCompile) {
    pm.addPass(createTritonGlobalKernelArgsToHIVMOpPass());
  }
  pm.addPass(createTensorToHIVMConversionPass());
  pm.addPass(createConvertToHIVMOpPass());
  if (!options.enableRegBaseHIVMPipe) {
    // HIVM brc/reduce op's operands have the same rank, so after converting
    // from Linalg/HFusion to HIVM, reshape ops will be inserted. Need to
    // propagate them.
    PropagateReshapeOptions propagateOption;
    propagateOption.forHIVM = true;
    pm.nest<func::FuncOp>().addPass(
        tensor::createPropagateReshapePass(propagateOption));
  }
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
