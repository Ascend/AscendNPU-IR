//===- ExecutionEnginePipelines.cpp - Pipelines for Execution Engine ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/ExecutionEngine/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void execution_engine::buildCPURunnerPipeline(
    OpPassManager &pm, const CPURunnerPipelineOptions &options) {
  // Preprocessing Passes
  ExecutionEngineHostMainCreatorOptions hostFuncWrapperOpts;
  hostFuncWrapperOpts.wrapperName = options.wrapperName;
  pm.addPass(execution_engine::createCreateHostMainPass(hostFuncWrapperOpts));

  // Bufferization Passes
  // decompose tensor.concat into slices before bufferization
  pm.addNestedPass<func::FuncOp>(tensor::createDecomposeTensorConcatPass());
  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());

  bufferization::OneShotBufferizationOptions bufferizationOpts;
  bufferizationOpts.bufferizeFunctionBoundaries = true;
  bufferizationOpts.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  bufferizationOpts.allowReturnAllocsFromLoops = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOpts));

  bufferization::buildBufferDeallocationPipeline(
      pm, bufferization::BufferDeallocationPipelineOptions());

  // Lower to LLVM Passes
  pm.addPass(createBufferizationToMemRefPass());
  pm.addNestedPass<func::FuncOp>(
      bufferization::createFinalizingBufferizePass());
  pm.addPass(annotation::createAnnotationLoweringPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void execution_engine::registerAllPipelines() {
  PassPipelineRegistration<CPURunnerPipelineOptions>(
      "lower-for-cpu-runner-pipeline",
      "Lower MLIR to LLVM MLIR to prepare it for CPU runner.",
      buildCPURunnerPipeline);
}
