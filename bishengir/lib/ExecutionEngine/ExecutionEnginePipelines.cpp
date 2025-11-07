//===- ExecutionEnginePipelines.cpp - Pipelines for Execution Engine ------===//
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

#include "bishengir/Dialect/Annotation/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/ExecutionEngine/Passes.h"
#include "bishengir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
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
  CanonicalizerOptions canonicalizerOpts;
  SmallVector<std::string> disabledPatterns(
      {"ReinterpretCastConstantArgumentFolder"});
  canonicalizerOpts.disabledPatterns = disabledPatterns;
  pm.addPass(bishengir::createExtendedCanonicalizerPass(canonicalizerOpts));
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(arith::createArithExpandOpsPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
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
