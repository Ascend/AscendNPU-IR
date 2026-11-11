//===- HIVMV2Pipelines.cpp - HIVM v2 pipeline ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The v2 pipeline keeps the MIX unsplit through bufferization, the memory
// plans and the cross-core solve; the split is the last structural pass.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Pipelines/v2/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/v2/TargetUtils.h"

#include "bishengir/Conversion/LowerMemRefExt/LowerMemRefExt.h"
#include "bishengir/Conversion/TensorViewToHIVM/TensorViewToHIVM.h"
#include "bishengir/Dialect/HACC/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Pipelines/HIVMBufferizationPipeline.h"
#include "bishengir/Dialect/HIVM/Pipelines/regbase/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "bishengir/Dialect/SCF/Transforms/Passes.h"
#include "bishengir/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace mlir {
namespace hivm {
namespace v2 {

static void addFuncPass(OpPassManager &pm, std::unique_ptr<Pass> pass) {
  pm.nest<func::FuncOp>().addPass(std::move(pass));
}

/// Resolve and validate the pipeline target once.
static hacc::TargetDevice resolveTarget(const V2PipelineOptions &options) {
  auto device = hacc::symbolizeTargetDeviceEnum(options.target);
  if (!isSupportedTarget(device)) {
    llvm::report_fatal_error(llvm::Twine("hivm-v2-pipeline: target '") +
                             options.target +
                             "' does not support the v2 pipeline");
  }
  return device;
}

static void buildPreVectorizationPhase(OpPassManager &pm,
                                       const V2PipelineOptions &options) {
  regbase::registerKernelKindPassesForRegbase();
  addFuncPass(pm, createInitEntryKernelPass());
  canonicalizationHIVMPipeline(pm);
  pm.addPass(createNormalizeMatmulPass());

  // Reuse the existing CV movement pass; TCB insertion owns UB -> L1.
  addFuncPass(pm, createInsertCVTightCoupledBufferPass());
  InsertLoadStoreForMixCVOptions cvOptions;
  cvOptions.enableLegacy = false;
  cvOptions.target = options.target;
  addFuncPass(pm, createInsertLoadStoreForMixCVPass(cvOptions));

  addFuncPass(pm, createMarkTightlyCoupledBufferPass());
  addFuncPass(pm, createHoistTightlyCoupledAllocPass());

  // TCB ids must exist on the MIX before the split.
  pm.addPass(createCacheModuleTargetPass());
  pm.addPass(createInsertFixpipePass());
  pm.addPass(createInlineFixpipePass());

  addFuncPass(pm, createInsertConvertLayoutPass());
  PropagateConvertLayoutOptions propagateOptions;
  propagateOptions.allowAgnosticOps = false;
  addFuncPass(pm, createPropagateConvertLayoutPass(propagateOptions));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  addFuncPass(pm, createCombineOptimizedConvertLayoutPass());
  addFuncPass(pm, createConvertLayoutToTransposePass());

  CVPipeliningOptions pipelineOptions;
  pipelineOptions.pipelineMode = CVPipelineMode::Skew;
  addFuncPass(pm, createCVPipeliningPass(pipelineOptions));

  // Re-run normalize after CV pipelining clones matmul users.
  pm.addPass(createNormalizeMatmulPass());

  // CV 1:2 needs operand-parallel sub-block partitioning; 1:1 targets skip it.
  if (needsCV12(hacc::symbolizeTargetDeviceEnum(options.target)))
    pm.addPass(createPartitionAndBindSubBlockPass());
}

/// Tensor phase on the unsplit MIX.
void buildV2TensorPhase(OpPassManager &pm, const V2PipelineOptions &options) {
  buildPreVectorizationPhase(pm, options);

  pm.addPass(createHIVMVectorizePass());
}

/// The tensor -> memref anchor.
void buildV2BufferizationPhase(OpPassManager &pm,
                               const V2PipelineOptions &options) {
  pm.addPass(hfusion::createSimplifyVFArgsPass());

  // The v2 pipeline only supports reg-based targets.
  bufferizationPipelineBase(pm, /*isRegBased=*/true);
  pm.addPass(createTensorViewToHIVMConversionPass());
  canonicalizationHIVMPipeline(pm);
  pm.addPass(createConvertToHIVMOpPass());
  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());

  if (needsCV12(hacc::symbolizeTargetDeviceEnum(options.target)))
    pm.addPass(createSubBlockGuardCleanupPass());
}

static void buildAlignStorage(OpPassManager &pm) {
  pm.addPass(createAlignAllocSizePass());
  addFuncPass(pm, memref::createFoldAllocReshapePass());
  pm.addPass(createEnableStrideAlignPass());
}

/// Post-bufferization: both memory plans and the solve see the unsplit MIX.
void buildV2PostBufferizationPhase(OpPassManager &pm,
                                   const V2PipelineOptions &options) {
  regbase::registerKernelKindPassesForRegbase();
  addFuncPass(pm, createLiftZeroRankPass());
  addFuncPass(pm, scf::createMapForToForallPass());
  addFuncPass(pm, createHIVMMapForallToBlocksPass());
  addFuncPass(pm, createHIVMDecomposeOpPass());
  pm.addPass(createNonContiguousReshapeToCopyPass());
  pm.addPass(createInferHIVMMemScopePass());
  addFuncPass(pm, createHIVMDecomposeOpPass());

  HIVMAggregatedDecomposeOpOptions decomposeOptions;
  decomposeOptions.decomposePhase =
      bishengir::DecomposePhase::BEFORE_HIVM_STRIDE_ALIGNMENT;
  addFuncPass(pm, createHIVMAggregatedDecomposeOpPass(decomposeOptions));
  decomposeOptions.decomposePhase =
      bishengir::DecomposePhase::AFTER_RECOGNIZE_BROADCAST;
  addFuncPass(pm, createHIVMAggregatedDecomposeOpPass(decomposeOptions));

  addFuncPass(pm, createInferBufferSizeFromSlicePass());
  buildAlignStorage(pm);

  decomposeOptions.decomposePhase =
      bishengir::DecomposePhase::AFTER_HIVM_STRIDE_ALIGNMENT;
  addFuncPass(pm, createHIVMAggregatedDecomposeOpPass(decomposeOptions));
  pm.addPass(bishengir::createExtendedCanonicalizerPass());

  addFuncPass(pm, createInferHIVMDataLayoutPass());
  decomposeOptions.decomposePhase =
      bishengir::DecomposePhase::AFTER_INFER_HIVM_DATA_LAYOUT;
  addFuncPass(pm, createHIVMAggregatedDecomposeOpPass(decomposeOptions));

  pm.addPass(bishengir::createExtendedCanonicalizerPass());
  addFuncPass(pm, createAutoInferBufferSizePass());
  addFuncPass(pm, createConstantizeBufferSizePass());
  addFuncPass(pm, createSetBufferSizePass());
  addFuncPass(pm, createFlattenOpsPass());
  decomposeOptions.decomposePhase =
      bishengir::DecomposePhase::AFTER_HIVM_FLATTEN_OPS;
  addFuncPass(pm, createHIVMAggregatedDecomposeOpPass(decomposeOptions));
  addFuncPass(pm, createReduceRankSubviewPass());
  addFuncPass(pm, createLiftLowestStridePass());
  addFuncPass(pm, createAllocExtraBufferPass());
  pm.addPass(createInferHIVMMemScopePass());
  canonicalizationHIVMPipeline(pm);

  // GLOBAL workspace plan on the unsplit MIX.
  PlanMemoryOptions globalPlanOptions;
  globalPlanOptions.memMode = MemPlanMode::GLOBAL_WORKSPACE_PLAN;
  pm.addPass(createPlanMemoryPass(globalPlanOptions));

  // Mark multi-buffer before the local plan allocates slots.
  MarkMultiBufferOptions multiBufferOptions;
  multiBufferOptions.enableAuto = options.enableMultiBuffer;
  multiBufferOptions.limitAutoMultiBufferOnlyForLocalBuffer = true;
  multiBufferOptions.gmMultiBufferNum = 1;
  addFuncPass(pm, createMarkMultiBufferPass(multiBufferOptions));

  // LOCAL plan on the unsplit MIX.
  PlanMemoryOptions localPlanOptions;
  pm.addPass(createPlanMemoryPass(localPlanOptions));

  // Expand slots before the split clones the counters.
  if (options.enableMultiBuffer)
    addFuncPass(pm, createEnableMultiBufferPass());

  // Single-shot cross-core solve on the MIX.
  canonicalizationHIVMPipeline(pm);
  CrossCoreGSSOptions crossCoreOptions;
  crossCoreOptions.forceIsRegBased = true;
  addFuncPass(pm, createCrossCoreGSSPass(crossCoreOptions));

  if (options.enablePreload) {
    CreatePreloadOptions preloadOptions;
    preloadOptions.enablePreload = true;
    pm.addPass(createCreatePreloadPass(preloadOptions));
  }

  // Last structural pass.
  pm.addPass(createV2SplitKernelPass());
}

/// Per-core tail.
void buildV2PerCoreTail(OpPassManager &pm) {
  addFuncPass(pm, createHIVMLowerToLoopsPass());
  addFuncPass(pm, createHIVMDecomposeOpPass());
  pm.addPass(createInferHIVMMemScopePass());

  GraphSyncSolverOptions gssOptions;
  addFuncPass(pm, createGraphSyncSolverPass(gssOptions));

  pm.addPass(mlir::createMemrefExtLoweringPass());
  addFuncPass(pm, createLowerMultiBufferCounterPass());
  addFuncPass(pm, createLiftLowestStridePass());
  canonicalizationHIVMPipeline(pm);
}

void buildV2HIVMPipeline(OpPassManager &pm, const V2PipelineOptions &options) {
  auto device = resolveTarget(options);
  if (options.setTargetSpec) {
    pm.addPass(hacc::createAppendDeviceSpecPass({device, /*customAICNumber=*/-1,
                                                 /*customAIVNumber=*/-1,
                                                 /*hivmcVersion=*/{}}));
  }
  buildV2TensorPhase(pm, options);
  buildV2BufferizationPhase(pm, options);
  buildV2PostBufferizationPhase(pm, options);
  buildV2PerCoreTail(pm);
}

void registerV2HIVMPipelines() {
  PassPipelineRegistration<V2PipelineOptions>(
      "hivm-v2-pre-vectorize-pipeline",
      "Prepare HIVM tensor IR up to the v2 vectorization boundary",
      [](OpPassManager &pm, const V2PipelineOptions &options) {
        auto device = resolveTarget(options);
        if (options.setTargetSpec)
          pm.addPass(hacc::createAppendDeviceSpecPass(
              {device, /*customAICNumber=*/-1, /*customAIVNumber=*/-1,
               /*hivmcVersion=*/{}}));
        buildPreVectorizationPhase(pm, options);
      });
  PassPipelineRegistration<V2PipelineOptions>(
      "hivm-v2-pipeline", "MIX HIVM pipeline (unsplit until the memref split)",
      [](OpPassManager &pm, const V2PipelineOptions &options) {
        buildV2HIVMPipeline(pm, options);
      });
}

} // namespace v2
} // namespace hivm
} // namespace mlir
