//===- HIVMPipelines.cpp - HIVM pipelines ---------------------------------===//
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

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "bishengir/Dialect/SCF/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace hivm {

void canonicalizationHIVMPipeline(OpPassManager &pm) {
  pm.addPass(createArithToAffineConversionPass());
  pm.nest<func::FuncOp>().addPass(scf::createCanonicalizeIterArgPass());
  pm.addPass(bishengir::createExtendedCanonicalizerPass());
  pm.addPass(createSCFForLoopCanonicalizationPass());
  pm.addPass(createCSEPass());
  pm.nest<func::FuncOp>().addPass(bishengir::createExtendedCanonicalizerPass());
  pm.nest<func::FuncOp>().addPass(createHIVMOptSinglePointPass());
  pm.nest<func::FuncOp>().addPass(bishengir::createExtendedCanonicalizerPass());
  pm.nest<func::FuncOp>().addPass(memref::createDeadStoreEliminationPass());
}

static void
hivmNormSyncPipeline(OpPassManager &pm,
                     const HIVMPipelineOptions &hivmPipelineOptions) {
  if (hivmPipelineOptions.enableHIVMGraphSyncSolver &&
      !hivmPipelineOptions.enableHIVMInjectBarrierAllSync) {
    GraphSyncSolverOptions gssOptions;
    gssOptions.enableUnitFlag = hivmPipelineOptions.enableHIVMUnitFlagSync;
    pm.nest<func::FuncOp>().addPass(createGraphSyncSolverPass(gssOptions));
  } else if (!hivmPipelineOptions.disableHIVMAutoInjectSync) {
    InjectSyncOptions syncOptions;
    syncOptions.enableUnitFlag = hivmPipelineOptions.enableHIVMUnitFlagSync;
    syncOptions.assumeAliveLoops =
        hivmPipelineOptions.enableHIVMAssumeAliveLoops;
    if (hivmPipelineOptions.enableHIVMInjectBarrierAllSync) {
      syncOptions.syncMode = SyncMode::BARRIERALL;
    }
    pm.nest<func::FuncOp>().addPass(createInjectSyncPass(syncOptions));
  }
}

static void
hivmCrossCoreSyncPipeline(OpPassManager &pm,
                          const HIVMPipelineOptions &hivmPipelineOptions) {
  // Mark load/store scalar operations with core-type attributes so block
  // synchronization passes recognize cross-core scalar-pipeline conflicts and
  // insert needed sync operations.
  pm.addPass(createMarkRealCoreTypePass());
  InjectBlockSyncOptions blockSyncOption;
  blockSyncOption.blockAllSync =
      hivmPipelineOptions.enableHIVMInjectBlockAllSync;
  blockSyncOption.assumeAliveLoops =
      hivmPipelineOptions.enableHIVMAssumeAliveLoops;
  blockSyncOption.disableAutoInjectBlockSync =
      hivmPipelineOptions.disableAutoInjectBlockSync;
  pm.nest<func::FuncOp>().addPass(createInjectBlockSyncPass(blockSyncOption));
  // Clear inserted core-type attributes as they are not needed for other
  // passes. Note that they are only inserted by mark-real-core-type pass so
  // it's safe to remove them. And after split-mix-kernel pass, they are not
  // needed.
  MarkRealCoreTypeOptions markRealCoreTypeOptions;
  markRealCoreTypeOptions.removeCoreTypeAttrs = true;
  pm.addPass(createMarkRealCoreTypePass(markRealCoreTypeOptions));
}

static void
bufferizationPipeline(OpPassManager &pm,
                      const HIVMPipelineOptions &hivmPipelineOptions) {
  if (hivmPipelineOptions.enableTritonKernelCompile) {
    pm.nest<func::FuncOp>().addPass(
        tensor::createOptimizeDpsOpWithYieldedInsertSlicePass());
    pm.nest<func::FuncOp>().addPass(createCloneTensorEmptyPass());
  }
  if (hivmPipelineOptions.enableUbufSaving) {
    pm.nest<func::FuncOp>().addPass(createSinkOpToConsumerInLoopPass());
  }
  bufferization::OneShotBufferizationOptions oneShotOptions;
  oneShotOptions.bufferizeFunctionBoundaries = true;
  oneShotOptions.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  oneShotOptions.allowReturnAllocsFromLoops = true;
  oneShotOptions.allowUnknownOps = true;
  pm.addPass(bufferization::createOneShotBufferizePass(oneShotOptions));
  canonicalizationHIVMPipeline(pm);
  if (hivmPipelineOptions.enableTritonKernelCompile) {
    // For triton kernels, bufferization will generate `memref.copy` ops,
    // and they need to be converted to `hivm.copy` ops.
    pm.addPass(createConvertToHIVMOpPass());
  }
  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  canonicalizationHIVMPipeline(pm);
  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  if (!hivmPipelineOptions.enableTritonKernelCompile) {
    // For non-triton kernels, there could also be `memref.copy` ops generated
    // during bufferization. But we want to convert them after canonicalizing
    // the IR.
    pm.addPass(createConvertToHIVMOpPass());
  }
}

static void hivmPreBufferizationOptimizationPipeline(
    OpPassManager &pm, const HIVMPipelineOptions &hivmPipelineOptions) {
  // HIVM brc/reduce op's operands have the same rank, so after converting from
  // Linalg/HFusion to HIVM, reshape ops will be inserted. Need to propagate
  // them.
  PropagateReshapeOptions propagateOption;
  propagateOption.forHIVM = true;
  pm.nest<func::FuncOp>().addPass(
      tensor::createPropagateReshapePass(propagateOption));
  pm.addPass(mlir::scf::createRemoveRedundantLoopInitPass());
  pm.addPass(mlir::hivm::createNormalizeMatmulPass());
  pm.addPass(mlir::hivm::createNormalizeBitwiseSelectPass());
  pm.addPass(mlir::hivm::createInlineFixpipePass());
  pm.nest<func::FuncOp>().addPass(createTileBatchMMIntoLoopPass());
  if (!hivmPipelineOptions.disableAutoCVWorkSpaceManage) {
    pm.nest<func::FuncOp>().addPass(
        mlir::hivm::createInsertLoadStoreForMixCVPass());
  }

  pm.addPass(mlir::hivm::createNormalizeMatmulPass());
  pm.addPass(createInsertNZ2NDForDebugPass());
  pm.addPass(mlir::hivm::createInlineFixpipePass());

  if (!hivmPipelineOptions.disableAutoCVWorkSpaceManage) {
    pm.nest<func::FuncOp>().addPass(
        mlir::hivm::createInsertLoadStoreForMixCVPass());
    pm.addPass(createInsertWorkSpaceForMixCVPass());
    pm.nest<func::FuncOp>().addPass(createBindWorkSpaceArgPass());
  }

  pm.addPass(createInferFuncCoreTypePass());
  // AutoBlockifyParallelLoopPass needs to be after infer core type because
  // num. of physical blocks we loop on is dependent on core type
  if (hivmPipelineOptions.enableTritonKernelCompile &&
      hivmPipelineOptions.enableAutoBlockifyLoop) {
    pm.addPass(createAutoBlockifyParallelLoopPass());
  }

  if (!hivmPipelineOptions.disableAutoCVWorkSpaceManage) {
    MarkMultiBufferOptions multiBufferOptions;
    multiBufferOptions.enableAuto = hivmPipelineOptions.enableAutoMultiBuffer;
    multiBufferOptions.limitAutoMultiBufferOnlyForLocalBuffer =
        hivmPipelineOptions.limitAutoMultiBufferOnlyForLocalBuffer;
    multiBufferOptions.limitAutoMultiBufferOfLocalBuffer =
        hivmPipelineOptions.limitAutoMultiBufferOfLocalBuffer;
    multiBufferOptions.limitMixAutoMultiBufferBuffer =
        hivmPipelineOptions.limitAutoMultiBufferBuffer;
    multiBufferOptions.workspaceMultiBufferNum =
        hivmPipelineOptions.setWorkspaceMultibuffer;
    pm.addNestedPass<func::FuncOp>(
        createMarkMultiBufferPass(multiBufferOptions));
  }
  // Call canonicalize before inline OTF broadcast to optimize redundant 1-to-1
  // broadcasts.
  pm.addPass(bishengir::createExtendedCanonicalizerPass());
  pm.nest<func::FuncOp>().addPass(createInlineOTFBroadcastPass());
  if (!hivmPipelineOptions.disableAutoCVWorkSpaceManage) {
    // Software pipelining Cube and Vector operations
    CVPipeliningOptions pipelineOptions;
    pipelineOptions.enableAutoBalance =
        hivmPipelineOptions.enableHIVMAutoCVBalance;
    pm.nest<func::FuncOp>().addPass(createCVPipeliningPass(pipelineOptions));
  }

  if (hivmPipelineOptions.tileMixCubeLoop != 1 ||
      hivmPipelineOptions.tileMixVectorLoop != 1) {
    pm.addPass(createTileCubeVectorLoopPass(
        TileCubeVectorLoopOptions{hivmPipelineOptions.tileMixVectorLoop,
                                  hivmPipelineOptions.tileMixCubeLoop}));
  }

  if (!hivmPipelineOptions.disableAutoCVWorkSpaceManage) {
    PlanMemoryOptions planMemoryOption;
    planMemoryOption.memMode = MemPlanMode::GLOBAL_WORKSPACE_PLAN;
    planMemoryOption.enableGlobalReuse =
        hivmPipelineOptions.enableHIVMGlobalWorkspaceReuse;
    pm.nest<func::FuncOp>().addPass(createPlanMemoryPass(planMemoryOption));
  }
  // cross-core sync (inject-block-sync) passes.
  hivmCrossCoreSyncPipeline(pm, hivmPipelineOptions);
  if (hivmPipelineOptions.enableTritonKernelCompile &&
      !hivmPipelineOptions.disableAutoCVWorkSpaceManage) {
    // Must place after plan-workspace-memory
    pm.nest<func::FuncOp>().addPass(createInsertInferWorkSpaceSizeFuncPass());
  }
  pm.addPass(mlir::createMemrefExtLoweringPass());

  if (hivmPipelineOptions.enableTritonKernelCompile) {
    pm.addPass(createInsertInferTaskTypeFuncPass());
  }
  // Split mix kernel is done before bufferization because it depends on
  // tensor SSA property.
  pm.addPass(createSplitMixKernelPass());
  pm.addPass(scope::createInlineScopePass());
  if (hivmPipelineOptions.enableAutoBindSubBlock)
    pm.addPass(createTileAndBindSubBlockPass());
  pm.nest<func::FuncOp>().addPass(tensor::createFoldTensorEmptyPass());
  canonicalizationHIVMPipeline(pm);
  if (hivmPipelineOptions.enableCodeMotion) {
    // call canonicalization to contantize the variable, then hoist can work for
    // some cases
    pm.addPass(createLoopInvariantCodeMotionPass());
    pm.addPass(createLoopInvariantSubsetHoistingPass());
  }

  pm.nest<func::FuncOp>().addPass(createCloneTensorEmptyPass());
  pm.nest<func::FuncOp>().addPass(createHIVMInlineOTFLoadStorePass());
}

static void
alignStoragePipeline(OpPassManager &pm,
                     const HIVMPipelineOptions &hivmPipelineOptions) {
  pm.nest<func::FuncOp>().addPass(createAlignAllocSizePass());
  if (hivmPipelineOptions.enableHIVMAutoStorageAlign) {
    pm.nest<func::FuncOp>().addPass(createMarkStrideAlignPass());
  }
  pm.nest<func::FuncOp>().addPass(memref::createFoldAllocReshapePass());
  pm.nest<func::FuncOp>().addPass(createEnableStrideAlignPass());
}

static void hivmPostBufferizationOptimizationPipeline(
    OpPassManager &pm, const HIVMPipelineOptions &hivmPipelineOptions) {
  pm.nest<func::FuncOp>().addPass(createLiftZeroRankPass());
  pm.nest<func::FuncOp>().addPass(scf::createMapForToForallPass());
  pm.nest<func::FuncOp>().addPass(createHIVMMapForallToBlocksPass());
  // Op decompose, need mark buffer size for newly allocated buffer.
  pm.nest<func::FuncOp>().addPass(createHIVMDecomposeOpPass());
  pm.nest<func::FuncOp>().addPass(createSyncBlockHoistingPass());
  pm.nest<func::FuncOp>().addPass(createBindSyncBlockLockArgPass());
  pm.nest<func::FuncOp>().addPass(
      createInsertInferSyncBlockLockNumAndInitFuncPass());
  pm.nest<func::FuncOp>().addPass(createSyncBlockLockLoweringPass());
  // Convert non-contiguous reshape to hivm.copy
  // Call this before infer mem scope. Otherwise, there might be UB allocs in
  // AIC function.
  pm.addPass(createNonContiguousReshapeToCopyPass());
  pm.addPass(createInferHIVMMemScopePass());
  // Decompose copy_ub_to_ub after inferHIVMMemScope
  pm.nest<func::FuncOp>().addPass(createHIVMDecomposeOpPass());
  HIVMAggregatedDecomposeOpOptions decomposeOption;
  // Currently no Ops decompose in this phase
  decomposeOption.decomposePhase =
      bishengir::DecomposePhase::BEFORE_HIVM_STRIDE_ALIGNMENT;
  pm.nest<func::FuncOp>().addPass(
      createHIVMAggregatedDecomposeOpPass(decomposeOption));

  // Transform uncontinuous access to deinterleave op
  pm.nest<func::FuncOp>().addPass(createHIVMRecognizeDeinterleaveOpPass());
  decomposeOption.decomposePhase =
      bishengir::DecomposePhase::AFTER_RECOGNIZE_DEINTERLEAVE;
  pm.nest<func::FuncOp>().addPass(
      createHIVMAggregatedDecomposeOpPass(decomposeOption));
  decomposeOption.decomposePhase =
      bishengir::DecomposePhase::AFTER_RECOGNIZE_BROADCAST;
  pm.nest<func::FuncOp>().addPass(
      createHIVMAggregatedDecomposeOpPass(decomposeOption));
  // align alloc size for special hivm op
  alignStoragePipeline(pm, hivmPipelineOptions);
  // Decompose {vconcat} after stride alignment
  decomposeOption.decomposePhase =
      bishengir::DecomposePhase::AFTER_HIVM_STRIDE_ALIGNMENT;
  pm.nest<func::FuncOp>().addPass(
      createHIVMAggregatedDecomposeOpPass(decomposeOption));
  // convert copyOp to nd2nzOp
  pm.nest<func::FuncOp>().addPass(createInferHIVMDataLayoutPass());
  decomposeOption.decomposePhase =
      bishengir::DecomposePhase::AFTER_INFER_HIVM_DATA_LAYOUT;
  pm.nest<func::FuncOp>().addPass(
      createHIVMAggregatedDecomposeOpPass(decomposeOption));

  // Passes to constantize alloc size.
  // Call canonicalize before constantize so that we make sure
  // that constant dimensions are folded into an alloc. We can simply check for
  // the memref type to find the dynamic allocs.
  pm.addPass(bishengir::createExtendedCanonicalizerPass());
  pm.nest<func::FuncOp>().addPass(createAutoInferBufferSizePass());
  // convert arith to affine before constantize buffer size again becuase stride
  // align may bring in arith ops
  pm.addPass(createArithToAffineConversionPass());
  pm.nest<func::FuncOp>().addPass(createConstantizeBufferSizePass());
  pm.nest<func::FuncOp>().addPass(createSetBufferSizePass());
  pm.nest<func::FuncOp>().addPass(createFlattenOpsPass());
  decomposeOption.decomposePhase =
      bishengir::DecomposePhase::AFTER_HIVM_FLATTEN_OPS;
  pm.nest<func::FuncOp>().addPass(
      createHIVMAggregatedDecomposeOpPass(decomposeOption));
  pm.nest<func::FuncOp>().addPass(createReduceRankSubviewPass());
  pm.nest<func::FuncOp>().addPass(createLiftLowestStridePass());
  pm.nest<func::FuncOp>().addPass(createAllocExtraBufferPass());
  // Infer memory scope for newly allocated extra buffer
  pm.addPass(createInferHIVMMemScopePass());
  canonicalizationHIVMPipeline(pm);
  pm.nest<func::FuncOp>().addPass(createInlineLoadCopyPass());

  MarkMultiBufferOptions multiBufferOptions;
  multiBufferOptions.enableAuto = hivmPipelineOptions.enableAutoMultiBuffer;
  // Limit auto multi buffer only work for local buffer at this stage
  multiBufferOptions.limitAutoMultiBufferOnlyForLocalBuffer = true;
  multiBufferOptions.limitAutoMultiBufferOfLocalBuffer =
      hivmPipelineOptions.limitAutoMultiBufferOfLocalBuffer;
  multiBufferOptions.limitMixAutoMultiBufferBuffer =
      hivmPipelineOptions.limitAutoMultiBufferBuffer;
  pm.nest<func::FuncOp>().addPass(
      createMarkMultiBufferPass(multiBufferOptions));
  pm.nest<func::FuncOp>().addPass(createPlanMemoryPass());

  // Lower hivm ops to loops
  pm.nest<func::FuncOp>().addPass(createHIVMLowerToLoopsPass());
  // TODO: move DecomposeI32ScalarExtOp etc. to interface
  pm.nest<func::FuncOp>().addPass(createHIVMDecomposeOpPass());
  // Normal sync (inject-sync, graph-sync-solver) passes.
  hivmNormSyncPipeline(pm, hivmPipelineOptions);
  pm.nest<func::FuncOp>().addPass(createAddFFTSToSyncBlockSetOpPass());
  pm.nest<func::FuncOp>().addPass(createEnableMultiBufferPass());
  pm.nest<func::FuncOp>().addPass(createLiftLowestStridePass());
}

void buildOptimizeHIVMPipeline(OpPassManager &pm,
                               const HIVMPipelineOptions &options) {
  pm.nest<func::FuncOp>().addPass(createInitEntryKernelPass());
  if (!options.disableHIVMTensorCompile) {
    hivmPreBufferizationOptimizationPipeline(pm, options);
    bufferizationPipeline(pm, options);
  }
  hivmPostBufferizationOptimizationPipeline(pm, options);
  // Optimizations that relies on scope should be done after this point. Inline
  // all `scope.scope` ops.
  pm.addPass(
      scope::createInlineScopePass(InlineScopeOptions{/*forceInline=*/true}));
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerLowerHIVMPipelines() {
  PassPipelineRegistration<HIVMPipelineOptions>(
      "optimize-hivm-pipeline", "optimize hivm pipeline",
      [](OpPassManager &pm, const HIVMPipelineOptions &options) {
        buildOptimizeHIVMPipeline(pm, options);
      });
}

} // namespace hivm
} // namespace mlir
