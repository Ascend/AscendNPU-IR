//===- HFusionPipelines.cpp - HFusion pipelines -----------------*- C++ -*-===//
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

#include "bishengir/Conversion/ArithToAffine/ArithToAffine.h"
#include "bishengir/Conversion/ArithToHFusion/ArithToHFusion.h"
#include "bishengir/Conversion/GPUToHFusion/GPUToHFusion.h"
#include "bishengir/Conversion/LinalgToHFusion/LinalgToHFusion.h"
#include "bishengir/Conversion/MathToHFusion/MathToHFusion.h"
#include "bishengir/Conversion/TensorToHFusion/TensorToHFusion.h"
#include "bishengir/Dialect/HFusion/Pipelines/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "bishengir/Dialect/Symbol/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "hfusion-pipeline"

namespace mlir {
namespace hfusion {

enum CanonicaliziationPattern {
  FoldFillWithTensorReshapeCollapse = 0,
  FoldFillWithTensorReshapeExpand = 1,
  FoldTransposeWithTranspose = 2
};

static DenseMap<int, std::string> canonicalizationEnumMap = {
    {FoldFillWithTensorReshapeCollapse,
     "(anonymous "
     "namespace)::FoldFillWithTensorReshape<mlir::tensor::CollapseShapeOp>"},
    {FoldFillWithTensorReshapeExpand,
     "(anonymous "
     "namespace)::FoldFillWithTensorReshape<mlir::tensor::ExpandShapeOp>"},
    {FoldTransposeWithTranspose, "FoldTransposeWithTranspose"}};

enum DisableCanonicalizationPhase {
  NoRestriction = 0,
  AfterFlattenBeforeAutoSchedule = 1,
  AfterAutoSchedule = 2
};

static DenseMap<int, std::vector<std::string>> phaseToDisabledMap = {
    {NoRestriction, {}},
    {AfterFlattenBeforeAutoSchedule,
     {canonicalizationEnumMap[FoldFillWithTensorReshapeCollapse],
      canonicalizationEnumMap[FoldFillWithTensorReshapeExpand]}},
    {AfterAutoSchedule, {canonicalizationEnumMap[FoldTransposeWithTranspose]}}};

static void
canonicalizationPipeline(OpPassManager &pm,
                         const HFusionPipelineOptions &hfusionOptions,
                         DisableCanonicalizationPhase phase = NoRestriction) {
  pm.addPass(createCSEPass());
  CanonicalizerOptions options;
  options.disabledPatterns = phaseToDisabledMap[phase];
  pm.addPass(bishengir::createExtendedCanonicalizerPass(options));
  pm.nest<func::FuncOp>().addPass(tensor::createNormalizeTensorOpsPass());
}

static void preProcess(OpPassManager &pm,
                       const HFusionPipelineOptions &options) {
  if (options.enableTritonKernelCompile) {
    pm.addPass(scope::createOutlineScopePass());
  }
  if (!options.enableSymbolAnalysis) {
    pm.nest<func::FuncOp>().addPass(symbol::createEraseSymbolPass());
  }
  pm.addPass(createArithToHFusionConversionPass());
  pm.addPass(createMathToHFusionConversionPass());
  pm.addPass(createLinalgToHFusionConversionPass());
  if (options.enableTritonKernelCompile) {
    pm.addPass(createSymbolDCEPass());
    pm.addPass(createGPUToHFusionConversionPass());
    pm.addPass(createAdaptTritonKernelPass());
  }
  pm.addPass(createTensorToHFusionConversionPass());
  pm.nest<func::FuncOp>().addPass(
      tensor::createCanonicalizeTensorReshapePass());
  canonicalizationPipeline(pm, options);
  // ArithToHFusion should be called after FoldUnitExtentDims and Canonicalize,
  // because with certain unit dims folded, some op (e.g. reduce) can be
  // optimized to lift containing arith op outside its body region
  pm.addPass(createArithToHFusionConversionPass());
  pm.nest<func::FuncOp>().addPass(createConvertGenericToNamedOpPass());
  pm.nest<func::FuncOp>().addPass(createLegalizeBF16Pass());
  DecomposeOptions decomposeOptions;
  decomposeOptions.hfusionDecomposePhase =
      bishengir::DecomposePhase::NO_CONSTRAINT;
  pm.nest<func::FuncOp>().addPass(createDecomposePass(decomposeOptions));
  pm.nest<func::FuncOp>().addPass(createHFusionNormalizeSliceOpsPass(
      /*skipAlignedSlice=*/options.enableTritonKernelCompile));
  pm.nest<func::FuncOp>().addPass(createHFusionNormalizeOpsPass());
  pm.addPass(createLegalizeBoolPass());
  pm.nest<func::FuncOp>().addPass(createSimplifyOpsPass());
  pm.nest<func::FuncOp>().addPass(createHFusionInlineBrcPass());
  // normalize should be called after inline-brc pass:
  //  a) convert scalar-vector ops to vector-scalar ops
  pm.nest<func::FuncOp>().addPass(createHFusionNormalizeOpsPass());
}

static void preFlattenPass(OpPassManager &pm,
                           const HFusionPipelineOptions &options) {
  pm.nest<func::FuncOp>().addPass(tensor::createBubbleUpExtractSlicePass());
  canonicalizationPipeline(pm, options);
  LinalgFoldUnitExtentDimsPassOptions linalgFoldOptions;
  linalgFoldOptions.useRankReducingSlices = false;
  linalgFoldOptions.foldRankReducingSlices = false;
  pm.nest<func::FuncOp>().addPass(
      createLinalgFoldUnitExtentDimsPass(linalgFoldOptions));
  canonicalizationPipeline(pm, options);
  // convert arith operations from canonicalized reduce operations
  pm.nest<func::FuncOp>().addPass(createArithToHFusionConversionPass());
  ComposeMultiReduceOptions composeOptions;
  composeOptions.aggressive = true;
  pm.nest<func::FuncOp>().addPass(createComposeMultiReduce(composeOptions));
  if (options.enableSymbolAnalysis) {
    pm.nest<func::FuncOp>().addPass(symbol::createPropagateSymbolPass());
    pm.nest<func::FuncOp>().addPass(symbol::createUnfoldSymbolicIntPass());
    // cse the unfolded tensor.empty with same tensor.dim
    pm.addPass(createCSEPass());
  }
  pm.nest<func::FuncOp>().addPass(tensor::createPropagateReshapePass());
  pm.nest<func::FuncOp>().addPass(createSimplifyOpsPass());
  canonicalizationPipeline(pm, options);
}

static void postProcessOutlinedKernel(OpPassManager &pm) {
  pm.nest<func::FuncOp>().addPass(createDowngradeFP64CstOpPass());
  pm.nest<func::FuncOp>().addPass(tensor::createTrickleConcatDownPass());
  pm.nest<func::FuncOp>().addPass(tensor::createBubblePadUpPass());
  pm.addPass(createLegalizeBoolPass());
  pm.nest<func::FuncOp>().addPass(tensor::createFoldTensorEmptyPass());
  pm.nest<func::FuncOp>().addPass(
      tensor::createNormalizeLastDimUnalignedTensorOpPass());
}

static void flattenAndFold(OpPassManager &pm,
                           const HFusionPipelineOptions &options) {
  // Add fold tensor empty pass to avoid dimension getting barriers for dynamic
  // shape if the expanded empty is binded to the empty shape
  // e.g:
  // %op = tensor.empty(%dim) <?xf32>
  // %expanded = <?xf32> -> <1x?x1xf32> output_shape = [1, %dim, 1]
  // ? will be getting barrier on its left and right, thus will avoid collapsing
  // the unit dimension
  pm.nest<func::FuncOp>().addPass(tensor::createFoldTensorEmptyPass());
  FlattenOpsOptions flattenOpsOpt;
  flattenOpsOpt.flattenMode = hfusion::FlattenMode::Tidy;
  flattenOpsOpt.skipHost = options.enableMultiKernel;
  flattenOpsOpt.multiDynamicShape = false;
  pm.nest<func::FuncOp>().addPass(createFlattenOpsPass(flattenOpsOpt));
  pm.nest<func::FuncOp>().addPass(
      tensor::createCanonicalizeTensorReshapePass());
  canonicalizationPipeline(pm, options, AfterFlattenBeforeAutoSchedule);
  pm.nest<func::FuncOp>().addPass(createCacheIOForReturnArg());
  // Pass to fold `tensor.empty` ops.
  pm.nest<func::FuncOp>().addPass(tensor::createFoldTensorEmptyPass());
  canonicalizationPipeline(pm, options, AfterFlattenBeforeAutoSchedule);
}

static void inferAndOutlineOp(OpPassManager &pm,
                              const HFusionPipelineOptions &options) {
  pm.nest<func::FuncOp>().addPass(createFoldSymbolicDimPass());
  pm.nest<func::FuncOp>().addPass(createInferFuncFusionKind());
  HFusionOpFusionOptions opFusionPassOption;
  opFusionPassOption.alwaysInline = false;
  opFusionPassOption.moveOutToParam = false;
  opFusionPassOption.outputMode = OutputMode::Multiple;
  opFusionPassOption.maxHorizontalFusionSize =
      options.hfusionMaxHorizontalFusionSize;
  opFusionPassOption.enableMultiKernel = options.enableMultiKernel;
  pm.addPass(createHFusionOpFusionPass(opFusionPassOption));
  canonicalizationPipeline(pm, options, AfterFlattenBeforeAutoSchedule);
  OutlineSingleOpOptions OutlineSingleOpOptions;
  OutlineSingleOpOptions.moveOutToParam = false;
  pm.nest<func::FuncOp>().addPass(
      createOutlineSingleOpPass(OutlineSingleOpOptions));
  canonicalizationPipeline(pm, options, AfterFlattenBeforeAutoSchedule);
  pm.nest<func::FuncOp>().addPass(createUnfoldSymbolicDimPass());
  pm.nest<func::FuncOp>().addPass(createDropSymbolsPass());
  pm.addPass(createEliminateDuplicateFuncsPass());
}

static void
hfusionTilingOptimizationPipeline(OpPassManager &pm,
                                  const HFusionPipelineOptions &options) {
  pm.addPass(createConstantizeTilingDataPass());
  canonicalizationPipeline(pm, options, AfterAutoSchedule);
  PackTilingDataOptions packOptions;
  packOptions.emitGetTilingStructSizeFunction = !options.enableMultiKernel;
  packOptions.packTilingKey = false;
  pm.addPass(createPackTilingDataPass(packOptions));
  // after tiling is all constantized and packed, try to simplify loops
  pm.addPass(createArithToAffineConversionPass());
  canonicalizationPipeline(pm, options, AfterAutoSchedule);
  pm.addPass(createSCFForLoopCanonicalizationPass());
  canonicalizationPipeline(pm, options, AfterAutoSchedule);
}

static void hfusionAutoSchedulePipeline(OpPassManager &pm,
                                        const HFusionPipelineOptions &options) {
  if (options.enableOpsReorder)
    pm.nest<func::FuncOp>().addPass(createReorderOpsByBFS());
  canonicalizationPipeline(pm, options, AfterFlattenBeforeAutoSchedule);
  // Decompose tranpose ops before auto-schedule
  DecomposeOptions decomposeOptions;
  decomposeOptions.hfusionDecomposePhase =
      bishengir::DecomposePhase::AFTER_HFUSION_FLATTEN;
  pm.nest<func::FuncOp>().addPass(createDecomposePass(decomposeOptions));
  // BEGIN AUTO SCHEDULE
  AutoScheduleOptions autoScheduleOptions;
  autoScheduleOptions.blockDim = options.blockDim;
  autoScheduleOptions.enableAutoMultiBuffer = options.enableAutoMultiBuffer;
  autoScheduleOptions.enableDeterministicComputing =
      options.enableDeterministicComputing;
  autoScheduleOptions.maxBufferCntTuning = options.hfusionMaxBufferCountTuning;
  autoScheduleOptions.cubeTilingTuning = options.cubeTilingTuning;
  autoScheduleOptions.enableCountBufferDmaOpt =
      options.enableHfusionCountBufferDmaOpt;
  autoScheduleOptions.externalTilingFuncPath = options.externalTilingFuncPath;
  autoScheduleOptions.enableManageHostResources =
      options.enableManageHostResources;
  autoScheduleOptions.enableSymbolAnalysis = options.enableSymbolAnalysis;
  pm.addPass(createHFusionAutoSchedulePass(autoScheduleOptions));
  // END AUTO SCHEDULE
  pm.nest<func::FuncOp>().addPass(createDecomposeMulti());
  // Auto Schedule might generated generic ops.
  pm.nest<func::FuncOp>().addPass(createConvertGenericToNamedOpPass());
  if (options.enableOpsReorder) {
    canonicalizationPipeline(pm, options, AfterAutoSchedule);
    pm.nest<func::FuncOp>().addPass(createReorderOpsByBFS());
  }
  hfusionTilingOptimizationPipeline(pm, options);
  if (!options.enableMultiKernel) {
    WrapHostFuncOptions wrapOptions{/*removeUnusedArguments=*/true};
    pm.addPass(createWrapHostFuncPass(wrapOptions));
  }
}

static void postProcess(OpPassManager &pm,
                        const HFusionPipelineOptions &options) {
  pm.nest<func::FuncOp>().addPass(createHFusionInlineBrcPass());
  // normalize should be called after auto schedule:
  // - tile reduction may generate unsupported elemwise op requiring normalize
  pm.nest<func::FuncOp>().addPass(createHFusionNormalizeOpsPass());

  // will only operate on functions with ShallowCV fusion kind
  AddFFTSAddrOptions addFFTSAddrOpt;
  if (options.enableTritonKernelCompile) {
    addFFTSAddrOpt.forceAddFFTSAddr = 0;
  }
  pm.addPass(createAddFFTSAddrPass(addFFTSAddrOpt));
  pm.addPass(createHoistTensorEmptyPass());
  // TODO: triton compiler do tiddy flatten as well for performance
  // decompose linalg.transpose into multiple ones that contains only two
  // permutation axes
  DecomposeOptions decomposeOptions;
  decomposeOptions.hfusionDecomposePhase =
      bishengir::DecomposePhase::AFTER_HFUSION_FLATTEN;
  pm.nest<func::FuncOp>().addPass(createDecomposePass(decomposeOptions));
}

void buildHFusionPipelines(OpPassManager &pm,
                           const HFusionPipelineOptions &options) {
  preProcess(pm, options);
  canonicalizationPipeline(pm, options);
  if (!options.enableTritonKernelCompile) {
    if (!options.enableMultiKernel) {
      preFlattenPass(pm, options);
      flattenAndFold(pm, options);
    }
    inferAndOutlineOp(pm, options);
    postProcessOutlinedKernel(pm);
    if (options.enableMultiKernel) {
      preFlattenPass(pm, options);
      flattenAndFold(pm, options);
    }
    hfusionAutoSchedulePipeline(pm, options);
  } else {
    pm.nest<func::FuncOp>().addPass(
        tensor::createCanonicalizeTensorReshapePass());
    // To handle ops that are not decomposed properly in the hivm.hir form
    // TODO: refactor propagate reshape structure and option names
    PropagateReshapeOptions propagateOption;
    propagateOption.forHIVM = true;
    pm.nest<func::FuncOp>().addPass(
        tensor::createPropagateReshapePass(propagateOption));
    pm.nest<func::FuncOp>().addPass(tensor::createFoldTensorEmptyPass());
    pm.nest<func::FuncOp>().addPass(
        tensor::createNormalizeLastDimUnalignedTensorOpPass());
  }
  canonicalizationPipeline(pm, options, AfterAutoSchedule);
  postProcess(pm, options);
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerLowerHFusionPipelines() {
  PassPipelineRegistration<HFusionPipelineOptions>(
      "lower-hfusion-pipeline", "lower hfusion pipeline",
      [](OpPassManager &pm, const HFusionPipelineOptions &options) {
        buildHFusionPipelines(pm, options);
      });
}

} // namespace hfusion
} // namespace mlir
