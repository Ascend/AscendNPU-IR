//===- HIVMBufferizationPipeline.cpp - HIVM bufferization pipeline --------===//
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

#include "bishengir/Dialect/HIVM/Pipelines/HIVMBufferizationPipeline.h"

#include "bishengir/Dialect/Bufferization/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir::bufferization;

namespace mlir {
namespace hivm {

void bufferizationPipelineBase(OpPassManager &pm, bool isRegBased) {
  if (isRegBased) {
    // Promote temporary tensors that are read-modify-written inside a nested
    // scf.for into the enclosing loop's iter_args, so that FoldSliceIntoScfFor
    // can handle it, helping one-shot-bufferize bufferize them in place without
    // inserting memref.copy.
    pm.nest<func::FuncOp>().addPass(
        tensor::createMergeConsecutiveInsertExtractSlicePass());
    pm.nest<func::FuncOp>().addPass(hivm::createNormalizeTmpBufferInVfPass());
  }
  // Fold extract_slice/insert_slice pairs around scf.for loops into the loop
  // body, merge the consecutive slice chains this creates and CSE away the
  // duplicated slices, so fold-extract-insert-pair can fold the redundant
  // slice round trips.
  pm.nest<func::FuncOp>().addPass(tensor::createFoldSliceIntoScfForPass());
  pm.nest<func::FuncOp>().addPass(
      tensor::createMergeConsecutiveInsertExtractSlicePass());
  pm.nest<func::FuncOp>().addPass(createCSEPass());
  pm.nest<func::FuncOp>().addPass(tensor::createFoldExtractInsertPairPass());
  // CSE merges identical tensor.empty ops; need to run clone-tensor-empty
  pm.nest<func::FuncOp>().addPass(createCloneTensorEmptyPass());
  // TODO: support process toTensorOp in one-shot-bufferize.
  // Expose memref-level writes (e.g., hivm.hir.load) to tensor-level analysis
  // by inserting an `alloc_tensor(copy...)` for each writable to_tensor.
  pm.nest<func::FuncOp>().addPass(
      bishengir::bufferization::createNormalizeToTensorOpPass());
  pm.nest<func::FuncOp>().addPass(bishengir::bufferization::createFlipIfPass());
  OneShotBufferizationOptions oneShotOptions;
  oneShotOptions.bufferizeFunctionBoundaries = true;
  oneShotOptions.setFunctionBoundaryTypeConversion(
      LayoutMapOption::IdentityLayoutMap);
  oneShotOptions.allowReturnAllocsFromLoops = true;
  oneShotOptions.allowUnknownOps = true;
  if (!isRegBased) {
    // We don't expect dynamic shape beacuse LiftLowestStride pass will generate
    // extract_strided_metadata op, which can't be lowered in planMemory pass.
    // So static identity layout for unknown type conversion should be used to
    // avoid dynamic layout map in the generated memref type.
    oneShotOptions.unknownTypeConverterFn =
        [=](Value value, Attribute memorySpace,
            const BufferizationOptions &options) {
          auto tensorType = cast<TensorType>(value.getType());
          return getMemRefTypeWithStaticIdentityLayout(tensorType, memorySpace);
        };
  }
  oneShotOptions.analysisHeuristic =
      OneShotBufferizationOptions::AnalysisHeuristic::TopDown;
  pm.addPass(createDecomposeAllocTensorCopyPass());
  // Run a first round of analysis + tensor copy insertion.
  pm.addPass(
      bishengir::bufferization::createTensorCopyInsertionPass(oneShotOptions));
  pm.addPass(createDecomposeAllocTensorCopyPass());
  pm.addPass(createOneShotBufferizePass(oneShotOptions));
  bishengir::bufferization::FlipIfOptions restoreIfOptions;
  restoreIfOptions.restoreOnly = true;
  pm.nest<func::FuncOp>().addPass(
      bishengir::bufferization::createFlipIfPass(restoreIfOptions));
}

void bufferizationPipeline(OpPassManager &pm,
                           const HIVMPipelineOptions &hivmPipelineOptions) {
  const bool isRegBased =
      hacc::utils::isRegBasedArch(hivmPipelineOptions.target);
  if (hivmPipelineOptions.enableTritonKernelCompile) {
    pm.nest<func::FuncOp>().addPass(
        tensor::createOptimizeDpsOpWithYieldedInsertSlicePass());
  }
  if (isRegBased ? hivmPipelineOptions.enableTritonKernelCompile
                 : hivmPipelineOptions.enableUbufSaving) {
    pm.nest<func::FuncOp>().addPass(createSinkOpToConsumerInLoopPass());
  }
  if (isRegBased) {
    if (hivmPipelineOptions.enableVfMergeLevel == 1) {
      MergeVecScopeOptions vfMergeOpsOpt;
      vfMergeOpsOpt.mergeLevel = 1;
      pm.addPass(hfusion::createMergeVecScopePass(vfMergeOpsOpt));
    }
    pm.addPass(hfusion::createSimplifyVFArgsPass());
  }
  bufferizationPipelineBase(pm, isRegBased);
  if (isRegBased && hivmPipelineOptions.enableVfMergeLevel == 2) {
    MergeVecScopeOptions vfMergeOpsOpt;
    vfMergeOpsOpt.mergeLevel = 2;
    pm.addPass(hfusion::createMergeVecScopePass(vfMergeOpsOpt));
  }
  canonicalizationHIVMPipeline(pm);
  if (hivmPipelineOptions.enableTritonKernelCompile) {
    // For triton kernels, bufferization will generate `memref.copy` ops,
    // and they need to be converted to `hivm.copy` ops.
    pm.addPass(createConvertToHIVMOpPass());
  }
  pm.addPass(createDropEquivalentBufferResultsPass());
  canonicalizationHIVMPipeline(pm);
  pm.addPass(createDropEquivalentBufferResultsPass());
  if (!hivmPipelineOptions.enableTritonKernelCompile) {
    // For non-triton kernels, there could also be `memref.copy` ops generated
    // during bufferization. But we want to convert them after canonicalizing
    // the IR.
    pm.addPass(createConvertToHIVMOpPass());
  }
}

} // namespace hivm
} // namespace mlir
