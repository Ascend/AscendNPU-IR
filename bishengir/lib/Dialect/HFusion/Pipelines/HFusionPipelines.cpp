//===- HFusionPipelines.cpp - HFusion pipelines -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/ArithToAffine/ArithToAffine.h"
#include "bishengir/Conversion/ArithToHFusion/ArithToHFusion.h"
#include "bishengir/Conversion/GPUToHFusion/GPUToHFusion.h"
#include "bishengir/Conversion/LinalgToHFusion/LinalgToHFusion.h"
#include "bishengir/Conversion/MathToHFusion/MathToHFusion.h"
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Conversion/TensorToHFusion/TensorToHFusion.h"
#include "bishengir/Dialect/Analysis/VFFusion/Passes.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Pipelines/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "bishengir/Dialect/Symbol/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
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
  options.enableExtendedPattern = true;
  options.disabledPatterns = phaseToDisabledMap[phase];
  pm.addPass(createCanonicalizerPass(options));
  pm.nest<func::FuncOp>().addPass(tensor::createNormalizeTensorOpsPass(
      /*skipAlignedSlice=*/hfusionOptions.enableTritonKernelCompile));
}

static void convertAllToHFusion(OpPassManager &pm,
                                const HFusionPipelineOptions &options,
                                bool shouldConvertLinalgToNamedOps = true) {
  pm.addPass(createArithToHFusionConversionPass());
  pm.addPass(createMathToHFusionConversionPass());
  // NOTE: linalg.generic should be converted to named equivalent ops after
  // dropping unit dimensions
  if (shouldConvertLinalgToNamedOps)
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
  // NOTE: linalg.generic should be converted to named equivalent ops after
  // dropping unit dimensions
  if (shouldConvertLinalgToNamedOps)
    pm.nest<func::FuncOp>().addPass(createConvertGenericToNamedOpPass());
}

static void preProcess(OpPassManager &pm,
                       const HFusionPipelineOptions &options) {
  LegalizeBoolPassOptions clampOptions;
  clampOptions.enableClamp = true;
  pm.addPass(createLegalizeBoolPass(clampOptions));
  if (!options.enableSymbolAnalysis) {
    pm.nest<func::FuncOp>().addPass(symbol::createEraseSymbolPass());
  }
  auto targetEnum = hacc::symbolizeTargetDeviceEnum(options.target);
  if (options.enableTritonKernelCompile &&
      hacc::utils::isRegBasedArch(targetEnum)) {
    pm.nest<func::FuncOp>().addPass(createLegalizeScalarPass());
    // Convert the operations to HFusion as much as possible to make
    // LinalgFoldUnitExtentDims work on LinalgOp interface more effectively.
    // Avoid converting special linalg.generic (e.g. hfusion.reduce_with_index)
    // because the pass is in the upstream so it cannot handle downstream
    // special operations.
    //
    // FIX: To handle inputs that contain direct HFusion ops, better to use a
    // pass that selectively generalizes HFusion ops (using
    // mlir::linalg::LinalgGeneralizationPattern)
    // delete enableDropUnitDims in future
    if (options.enableFlatten && options.enableDropUnitDims) {
      convertAllToHFusion(pm, options, false);
      pm.nest<func::FuncOp>().addPass(createHFusionGeneralizePass());
      pm.nest<func::FuncOp>().addPass(createHFusionFoldUnitDimsPass());
      // NOTE: Re-converting to HFusion is necessary after applying the
      // canonicalizer because it simplifies some linalg operations
      canonicalizationPipeline(pm, options);
    }
  }
  convertAllToHFusion(pm, options);
  pm.nest<func::FuncOp>().addPass(createLegalizeBF16Pass());
  pm.nest<func::FuncOp>().addPass(createLegalizeFP8Pass());
  DecomposeOptions decomposeOptions;
  decomposeOptions.hfusionDecomposePhase =
      bishengir::DecomposePhase::NO_CONSTRAINT;
  pm.nest<func::FuncOp>().addPass(createDecomposePass(decomposeOptions));
  pm.nest<func::FuncOp>().addPass(createHFusionNormalizeSliceOpsPass());
  pm.nest<func::FuncOp>().addPass(createGenericUnrollerPass());
  NormalizeOptions normalizeOptions;
  normalizeOptions.enableHighPrecision = options.enableHighPrecision;
  pm.nest<func::FuncOp>().addPass(
      createHFusionNormalizeOpsPass(normalizeOptions));
  pm.addPass(createLegalizeBoolPass());
  pm.nest<func::FuncOp>().addPass(createSimplifyOpsPass());
  pm.nest<func::FuncOp>().addPass(createHFusionInlineBrcPass());
  // normalize should be called after inline-brc pass:
  //  a) convert scalar-vector ops to vector-scalar ops
  pm.nest<func::FuncOp>().addPass(
      createHFusionNormalizeOpsPass(normalizeOptions));
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
  opFusionPassOption.maxHorizontalFusionSize = options.maxHorizontalFusionSize;
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
  autoScheduleOptions.maxBufferCntTuning = options.maxBufferCntTuning;
  autoScheduleOptions.cubeTilingTuning = options.cubeTilingTuning;
  autoScheduleOptions.enableCountBufferDmaOpt = options.enableCountBufferDmaOpt;
  autoScheduleOptions.externalTilingFuncPath = options.externalTilingFuncPath;
  autoScheduleOptions.enableManageHostResources =
      options.enableManageHostResources;
  pm.addPass(createHFusionAutoSchedulePass(autoScheduleOptions));
  // END AUTO SCHEDULE
  pm.nest<func::FuncOp>().addPass(createDecomposeMulti());
  // Auto Schedule might generated generic ops.
  // If we want to vectorize, it is easier with generic ops
  if (!hacc::utils::isRegBasedArch(
          hacc::symbolizeTargetDeviceEnum(options.target)))
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
  NormalizeOptions normalizeOptions;
  normalizeOptions.enableHighPrecision = options.enableHighPrecision;
  pm.nest<func::FuncOp>().addPass(
      createHFusionNormalizeOpsPass(normalizeOptions));
  // will only operate on functions with ShallowCV fusion kind
  AddFFTSAddrOptions addFFTSAddrOpt;
  if (options.enableTritonKernelCompile && options.insertFFTS) {
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

static void hfusionVectorizeManualScopePipeline(
    OpPassManager &pm, const HFusionPipelineOptions &hfusionOptions) {
  // vectorize manual vector scope
  pm.addPass(scope::createOutlineScopePass());
  VectorizeOpsOptions vectorizeOptions;
  vectorizeOptions.forManualScope = true;
  pm.addPass(createHFusionVectorizeOpsPass(vectorizeOptions));
  pm.nest<func::FuncOp>().addPass(vector::createLowerVectorMaskPass());
  pm.addPass(createSimplifyVFArgsPass());
  canonicalizationPipeline(pm, hfusionOptions);
  pm.nest<func::FuncOp>().addPass(
      vector::createMaterializeVectorWriteToDestinationPass());
}

static void
hfusionAutoVectorizePipeline(OpPassManager &pm,
                             const HFusionPipelineOptions &hfusionOptions) {
  canonicalizationPipeline(pm, hfusionOptions);
  pm.nest<func::FuncOp>().addPass(hivm::createSinkOpToConsumerInLoopPass());
  pm.nest<func::FuncOp>().addPass(hivm::createCloneSCFIfYieldOperandPass());
  hfusionVectorizeManualScopePipeline(pm, hfusionOptions);
  if (enableSIMDVFFusion(hfusionOptions)) {
    VFFusionOptions vfFusionOptions;
    pm.addPass(analysis::createVFFusionPass(vfFusionOptions));
    canonicalizationPipeline(pm, hfusionOptions);
  }
  PreVectorizationFusionOptions options;
  options.enableTritonCompile = hfusionOptions.enableTritonKernelCompile;
  options.maxFusedElementwiseOps = hfusionOptions.maxFusedElementwiseOps;
  pm.nest<func::FuncOp>().addPass(createPreVectorizationFusionPass(options));
  canonicalizationPipeline(pm, hfusionOptions);
  if (hfusionOptions.enableAutoVectorizeV2) {
    AutoVectorizeV2Options vecOptions;
    vecOptions.enableMultipleConsumerFusion =
        hfusionOptions.enableMultipleConsumerFusion;
    if (hfusionOptions.maxFusedOpsInAutoVectorizeV2 >= 0)
      vecOptions.maxFusedOps =
          static_cast<unsigned>(hfusionOptions.maxFusedOpsInAutoVectorizeV2);
    pm.addPass(createHFusionAutoVectorizeV2Pass(vecOptions));
    pm.addPass(createOutlineVectorFunctionPass());
  } else {
    AutoVectorizeOptions vecOptions;
    // Set maxVectorizeAxes to be 1 for compatiblity.
    // TODO: Remove this constraint after e2e support for multi axes
    // vectorization.
    vecOptions.maxVectorizeAxes = 2;
    vecOptions.treeReduce = hfusionOptions.enableTreeReduce;
    pm.addPass(createHFusionAutoVectorizePass(vecOptions));
  }
  pm.addPass(createPullSliceIntoVectorFunctionPass());
  pm.addPass(mlir::createHFusionToVectorConversionPass());
  pm.nest<func::FuncOp>().addPass(
      createRemoveMaskFromUnalignedReductionLoopPass());
  pm.nest<func::FuncOp>().addPass(vector::createLowerVectorMaskPass());
  pm.addPass(createSimplifyVFArgsPass());
  pm.addPass(createLoopInvariantSubsetHoistingPass());
  canonicalizationPipeline(pm, hfusionOptions);
  pm.addPass(createSCFForLoopCanonicalizationPass());
  canonicalizationPipeline(pm, hfusionOptions);
}

void buildHFusionPipelines(OpPassManager &pm,
                           const HFusionPipelineOptions &options) {
  bool runRegBasePasses = !options.enableMixedCV &&
                          hacc::utils::isRegBasedArch(
                              hacc::symbolizeTargetDeviceEnum(options.target));
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
    // this logic copy from function: "flattenAndFold"
    // step 1: flattenOps
    // step 2: CanonicalizeTensorReshape
    // step 3：FoldTensorEmpty
    // step 4: CanonicalizeTensorReshape
    // FIXME: because of VF fusion, for triton compile, we need flatten pass or
    // other passes we didn't notice. Maybe these passes will be called later.
    if (runRegBasePasses && options.enableFlatten) {
      PropagateReshapeOptions opts;
      opts.forRegbased = true;
      opts.forHIVM = false;
      opts.skipScope = options.skipScope;
      pm.nest<func::FuncOp>().addPass(tensor::createPropagateReshapePass(opts));
      pm.nest<func::FuncOp>().addPass(tensor::createFoldTensorEmptyPass());
      pm.nest<func::FuncOp>().addPass(
          tensor::createCanonicalizeTensorReshapePass());
      canonicalizationPipeline(pm, options);

      FlattenOpsOptions flattenOpsOpt;
      flattenOpsOpt.flattenMode = hfusion::FlattenMode::Tidy;
      flattenOpsOpt.skipHost = options.enableMultiKernel;
      flattenOpsOpt.multiDynamicShape = false;
      flattenOpsOpt.registerBased = true;
      flattenOpsOpt.skipScope = options.skipScope;
      pm.nest<func::FuncOp>().addPass(createFlattenOpsPass(flattenOpsOpt));
    }
    pm.nest<func::FuncOp>().addPass(
        tensor::createCanonicalizeTensorReshapePass());
    pm.nest<func::FuncOp>().addPass(
        tensor::createNormalizeLastDimUnalignedTensorOpPass());

    if (runRegBasePasses) {
      canonicalizationPipeline(pm, options, AfterAutoSchedule);
      pm.nest<func::FuncOp>().addPass(tensor::createFoldTensorEmptyPass());
    }
  }
  canonicalizationPipeline(pm, options, AfterAutoSchedule);
  postProcess(pm, options);

  if (runRegBasePasses) {
    if (!options.disableHFusionVectorize) {
      hfusionAutoVectorizePipeline(pm, options);
    }
  }
}

void buildHFusionRegBasePipeline(OpPassManager &pm,
                                 const HFusionPipelineOptions &options) {
  // TODO: Commonize this with the non-mixed-cv pipeline to prevent potential
  // out-of-sync pass orders
  if (options.enableFlatten) {
    PropagateReshapeOptions opts;
    opts.forRegbased = true;
    opts.forHIVM = false;
    pm.nest<func::FuncOp>().addPass(tensor::createPropagateReshapePass(opts));
    pm.nest<func::FuncOp>().addPass(tensor::createFoldTensorEmptyPass());
    pm.nest<func::FuncOp>().addPass(
        tensor::createCanonicalizeTensorReshapePass());
    canonicalizationPipeline(pm, options);

    FlattenOpsOptions flattenOpsOpt;
    flattenOpsOpt.flattenMode = hfusion::FlattenMode::Tidy;
    flattenOpsOpt.skipHost = options.enableMultiKernel;
    flattenOpsOpt.multiDynamicShape = false;
    flattenOpsOpt.registerBased = true;
    pm.nest<func::FuncOp>().addPass(createFlattenOpsPass(flattenOpsOpt));
    canonicalizationPipeline(pm, options, AfterAutoSchedule);
  }

  pm.nest<func::FuncOp>().addPass(tensor::createFoldTensorEmptyPass());
  NormalizeOptions normalizeOptions;
  normalizeOptions.enableHighPrecision = options.enableHighPrecision;
  pm.nest<func::FuncOp>().addPass(
      createHFusionNormalizeOpsPass(normalizeOptions));
  pm.nest<func::FuncOp>().addPass(createHFusionInlineBrcPass());
  hfusionAutoVectorizePipeline(pm, options);
}

bool enableSIMDVFFusion(const HFusionPipelineOptions &options) {
  return options.enableTritonKernelCompile && options.enableVFFusion;
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
