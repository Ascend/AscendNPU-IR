//===- HIVMPipelines.cpp - HIVM pipelines ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/ArithToAffine/ArithToAffine.h"
#include "bishengir/Conversion/HFusionToHIVM/HFusionToHIVMPass.h"
#include "bishengir/Conversion/LowerMemRefExt/LowerMemRefExt.h"
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
#include "bishengir/Transforms/Passes.h"

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
  pm.nest<func::FuncOp>().addPass(scf::createCanonicalizeIterArgPass());
  ADD_CANONICALIZER_PASS;
  pm.addPass(createSCFForLoopCanonicalizationPass());
  pm.addPass(createCSEPass());
  ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
  pm.nest<func::FuncOp>().addPass(createHIVMOptSinglePointPass());
  ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
  pm.nest<func::FuncOp>().addPass(memref::createDeadStoreEliminationPass());
}

static void
hivmCVCommunicationPipeline(OpPassManager &pm,
                            const HIVMPipelineOptions &hivmPipelineOptions) {
  // TODO: Implement the transformation that splits a to_tensor on a
  // space-specific memref into a pure to_tensor + memory_space_cast. It is
  // recommended to do this in canonicalization pass

  if (hacc::utils::isAscend950(
          hacc::symbolizeTargetDeviceEnum(hivmPipelineOptions.target)) &&
      !hivmPipelineOptions.enableDotScaledCompile) {
    // New A5 convert layout pipeline
    if (hivmPipelineOptions.enableLayoutOptimization) {
      pm.nest<func::FuncOp>().addPass(createInsertCVDataMovementPass());
    } else {
      pm.nest<func::FuncOp>().addPass(createInsertCVTightCoupledBufferPass());
    }
    pm.nest<func::FuncOp>().addPass(
        mlir::hivm::createInsertLoadStoreForScalarPass());
  } else {
    pm.nest<func::FuncOp>().addPass(
        mlir::hivm::createInsertLoadStoreForMixCVPass());
  }
}

static void
hivmIntraCoreSyncPipeline(OpPassManager &pm,
                          const HIVMPipelineOptions &hivmPipelineOptions) {
  if (hivmPipelineOptions.enableHIVMGraphSyncSolver &&
      !hivmPipelineOptions.enableHIVMInjectBarrierAllSync) {
    GraphSyncSolverOptions gssOptions;
    gssOptions.enableUnitFlag = hivmPipelineOptions.enableUnitFlagSync;
    pm.nest<func::FuncOp>().addPass(createGraphSyncSolverPass(gssOptions));
  } else {
    InjectSyncOptions syncOptions;
    syncOptions.enableUnitFlag = hivmPipelineOptions.enableUnitFlagSync;
    if (hivmPipelineOptions.enableHIVMInjectBarrierAllSync) {
      syncOptions.syncMode = SyncMode::BARRIERALL;
    }
    pm.nest<func::FuncOp>().addPass(createInjectSyncPass(syncOptions));
  }
}

static void
hivmCrossCoreSyncPipeline(OpPassManager &pm,
                          const HIVMPipelineOptions &hivmPipelineOptions) {
  if (hivmPipelineOptions.disableAutoInjectBlockSync) {
    return;
  }
  // Mark load/store scalar operations with core-type attributes so block
  // synchronization passes recognize cross-core scalar-pipeline conflicts and
  // insert needed sync operations.
  pm.addPass(createMarkRealCoreTypePass());
  if (hivmPipelineOptions.enableHIVMGraphSyncSolver &&
      !hivmPipelineOptions.enableInjectBlockAllSync) {
    pm.nest<func::FuncOp>().addPass(createCrossCoreGSSPass());
  } else {
    InjectBlockSyncOptions blockSyncOption;
    blockSyncOption.blockAllSync = hivmPipelineOptions.enableInjectBlockAllSync;
    pm.nest<func::FuncOp>().addPass(createInjectBlockSyncPass(blockSyncOption));
  }
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
    pm.nest<func::FuncOp>().addPass(createSinkOpToConsumerInLoopPass());
  }
  if (hivmPipelineOptions.enableVfMergeLevel == 1) {
    MergeVecScopeOptions VfMergeOpsOpt;
    VfMergeOpsOpt.mergeLevel = 1;
    pm.addPass(hfusion::createMergeVecScopePass(VfMergeOpsOpt));
  }
  pm.addPass(hfusion::createSimplifyVFArgsPass());
  bufferization::OneShotBufferizationOptions oneShotOptions;
  oneShotOptions.bufferizeFunctionBoundaries = true;
  oneShotOptions.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  oneShotOptions.allowReturnAllocsFromLoops = true;
  oneShotOptions.allowUnknownOps = true;
  pm.addPass(bufferization::createOneShotBufferizePass(oneShotOptions));
  if (hivmPipelineOptions.enableVfMergeLevel == 2) {
    MergeVecScopeOptions VfMergeOpsOpt;
    VfMergeOpsOpt.mergeLevel = 2;
    pm.addPass(hfusion::createMergeVecScopePass(VfMergeOpsOpt));
  }
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
  if (!hacc::utils::isRegBasedArch(
          hacc::symbolizeTargetDeviceEnum(hivmPipelineOptions.target))) {
    // HIVM brc/reduce op's operands have the same rank, so after
    // converting from Linalg/HFusion to HIVM, reshape ops will be
    // inserted. Need to propagate them.
    PropagateReshapeOptions propagateOption;
    propagateOption.forHIVM = true;
    pm.nest<func::FuncOp>().addPass(
        tensor::createPropagateReshapePass(propagateOption));
  }

  pm.addPass(mlir::scf::createRemoveRedundantLoopInitPass());
  pm.addPass(mlir::hivm::createNormalizeMatmulPass());

  if (hivmPipelineOptions.enableLayoutOptimization) {
    // Combine optimized folds:
    // - load + convert layout
    // - convert layout + fixpipe
    // For regbase convert layout optimization is done early in the pass
    // Inserts convert layout before and after cube operations
    pm.nest<func::FuncOp>().addPass(createInsertConvertLayoutPass());

    // Moves layout conversion to the start and end of the kernel
    // TODO: This part needs the most improvement compared to others
    pm.nest<func::FuncOp>().addPass(createPropagateConvertLayoutPass());

    // Add canonicalization passes
    pm.nest<func::FuncOp>().addPass(createCanonicalizerPass());
    pm.nest<func::FuncOp>().addPass(createCSEPass());
    pm.addPass(mlir::hivm::createCombineOptimizedConvertLayoutPass());
    pm.addPass(mlir::hivm::createInlineFixpipeV2Pass());
  } else {
    pm.addPass(mlir::hivm::createInlineFixpipePass());
  }
  hivmCVCommunicationPipeline(pm, hivmPipelineOptions);
  pm.nest<func::FuncOp>().addPass(createTileBatchMMIntoLoopPass());
  pm.addPass(mlir::hivm::createNormalizeMatmulPass());
  if (hivmPipelineOptions.enableLayoutOptimization) {
    pm.addPass(mlir::hivm::createCombineOptimizedConvertLayoutPass());
    pm.addPass(mlir::hivm::createInlineFixpipeV2Pass());
    pm.nest<func::FuncOp>().addPass(createConvertLayoutToTransposePass());
  } else {
    if (hacc::utils::isAscend950(
            hacc::symbolizeTargetDeviceEnum(hivmPipelineOptions.target))) {
      pm.addPass(createInsertL12UBForDebugPass());
    } else {
      pm.addPass(createInsertNZ2NDForDebugPass());
    }
    pm.addPass(mlir::hivm::createInlineFixpipePass());
  }
  hivmCVCommunicationPipeline(pm, hivmPipelineOptions);
  if (!hacc::utils::isAscend950(
          hacc::symbolizeTargetDeviceEnum(hivmPipelineOptions.target)) ||
      hivmPipelineOptions.enableDotScaledCompile) {
    pm.addPass(createInsertWorkSpaceForMixCVPass());
  }
  // keep this for the debug feature (device print, etc.)
  pm.nest<func::FuncOp>().addPass(createBindWorkSpaceArgPass());

  pm.addPass(createInferFuncCoreTypePass());
  // AutoBlockifyParallelLoopPass needs to be after infer core type because
  // num. of physical blocks we loop on is dependent on core type
  if (hivmPipelineOptions.enableTritonKernelCompile &&
      hivmPipelineOptions.enableAutoBlockifyLoop) {
    pm.addPass(createAutoBlockifyParallelLoopPass());
  }

  MarkMultiBufferOptions multiBufferOptions;
  multiBufferOptions.enableAuto = hivmPipelineOptions.enableAutoMultiBuffer;
  multiBufferOptions.limitAutoMultiBufferOnlyForLocalBuffer =
      hivmPipelineOptions.limitAutoMultiBufferOnlyForLocalBuffer;
  multiBufferOptions.limitAutoMultiBufferOfLocalBuffer =
      hivmPipelineOptions.limitAutoMultiBufferOfLocalBuffer;
  multiBufferOptions.limitMixAutoMultiBufferBuffer =
      hivmPipelineOptions.limitMixAutoMultiBufferBuffer;
  multiBufferOptions.workspaceMultiBufferNum =
      hivmPipelineOptions.workspaceMultiBufferNum;
  pm.addNestedPass<func::FuncOp>(createMarkMultiBufferPass(multiBufferOptions));
  // Call canonicalize before inline OTF broadcast to optimize redundant 1-to-1
  // broadcasts.
  ADD_CANONICALIZER_PASS;
  pm.nest<func::FuncOp>().addPass(createInlineOTFBroadcastPass());
  if (hivmPipelineOptions.enableMixedCV) {
    if (hivmPipelineOptions.workspaceMultiBufferNum > 1) {
      pm.nest<func::FuncOp>().addPass(
          mlir::hivm::createSplitMixedIfConditionalsPass());
    }
    // Software pipelining Cube and Vector operations
    CVPipeliningOptions pipelineOptions;
    pipelineOptions.pipelineDepth = (int)hivmPipelineOptions.workspaceMultiBufferNum;
    pm.nest<func::FuncOp>().addPass(createCVPipeliningPass(pipelineOptions));
  }

  pm.nest<func::FuncOp>().addPass(createInferVFModePass());

  PlanMemoryOptions planMemoryOption;
  planMemoryOption.memMode = MemPlanMode::GLOBAL_WORKSPACE_PLAN;
  planMemoryOption.enableGlobalReuse =
      hivmPipelineOptions.enableGlobalWorkspaceReuse;
  planMemoryOption.enablePrintMemoryAllocatedSize =
      hivmPipelineOptions.enablePrintMemoryAllocatedSize;
  planMemoryOption.disableTightlyCoupledBufferReuse =
      hivmPipelineOptions.disableTightlyCoupledBufferReuse;
  pm.addPass(createPlanMemoryPass(planMemoryOption));

  // Cross-Core Auto-Sync passes (Inject-Block-Sync, Cross-Core-GSS)
  hivmCrossCoreSyncPipeline(pm, hivmPipelineOptions);

  if (hivmPipelineOptions.enableTritonKernelCompile)
    // Must place after plan-workspace-memory
    pm.nest<func::FuncOp>().addPass(createInsertInferWorkSpaceSizeFuncPass());
  pm.addPass(mlir::createMemrefExtLoweringPass());
  // Split mix kernel is done before bufferization because it depends on
  // tensor SSA property.
  pm.addPass(createSplitMixKernelPass());
  pm.addPass(scope::createInlineScopePass());
  TileAndBindSubBlockOptions tileOptions;
  tileOptions.enableTile = hivmPipelineOptions.enableAutoBindSubBlock;
  pm.addPass(createTileAndBindSubBlockPass(tileOptions));
  pm.nest<func::FuncOp>().addPass(tensor::createFoldTensorEmptyPass());
  canonicalizationHIVMPipeline(pm);
  if (hivmPipelineOptions.enableCodeMotion) {
    // call canonicalization to contantize the variable, then hoist can work for
    // some cases
    pm.addPass(createLoopInvariantCodeMotionPass());
    pm.addPass(createLoopInvariantSubsetHoistingPass());
    canonicalizationHIVMPipeline(pm);
  }
  pm.addPass(hfusion::createSimplifyVFArgsPass());
  pm.nest<func::FuncOp>().addPass(createCloneTensorEmptyPass());
  pm.nest<func::FuncOp>().addPass(createHIVMInlineOTFLoadStorePass());
}

static void
alignStoragePipeline(OpPassManager &pm,
                     const HIVMPipelineOptions &hivmPipelineOptions) {
  pm.addPass(createAlignAllocSizePass());
  if (hivmPipelineOptions.enableAutoStorageAlign) {
    pm.nest<func::FuncOp>().addPass(createMarkStrideAlignPass());
  }
  pm.nest<func::FuncOp>().addPass(memref::createFoldAllocReshapePass());
  pm.addPass(createEnableStrideAlignPass());
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
  if (hacc::utils::isRegBasedArch(
          hacc::symbolizeTargetDeviceEnum(hivmPipelineOptions.target))) {
    // make sure no alloc within vf and no value returned by vf,
    // so InferHIVMMemScope can work correctly
    pm.addPass(createOutlineAllocInVFPass());
    // Move VF entry DMA on argument buffers to the caller before VF cleanup.
    pm.addPass(createOutlineCopyInVFPass());
    pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
    // make sure VFFusion function can be inlined
    // so InferHIVMMemScope can work correctly
    pm.addPass(scope::createInlineScopePass());
  }
  // Bind buffer should be done after outline alloc in vf because the source
  // allocs might be inside the VF.
  pm.nest<func::FuncOp>().addPass(memref::createBindBufferPass());
  pm.addPass(createInferHIVMMemScopePass());
  // Decompose copy_ub_to_ub after inferHIVMMemScope
  pm.nest<func::FuncOp>().addPass(createHIVMDecomposeOpPass());
  HIVMAggregatedDecomposeOpOptions decomposeOption;
  // Currently no Ops decompose in this phase
  decomposeOption.decomposePhase =
      bishengir::DecomposePhase::BEFORE_HIVM_STRIDE_ALIGNMENT;
  pm.nest<func::FuncOp>().addPass(
      createHIVMAggregatedDecomposeOpPass(decomposeOption));
  if (!hacc::utils::isRegBasedArch(
          hacc::symbolizeTargetDeviceEnum(hivmPipelineOptions.target))) {
    // Transform uncontinuous access to deinterleave op
    pm.nest<func::FuncOp>().addPass(createHIVMRecognizeDeinterleaveOpPass());
    decomposeOption.decomposePhase =
        bishengir::DecomposePhase::AFTER_RECOGNIZE_DEINTERLEAVE;
  }

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
  ADD_CANONICALIZER_PASS;
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
  ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
  pm.nest<func::FuncOp>().addPass(createAutoInferBufferSizePass());
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
  if (hacc::utils::isRegBasedArch(
          hacc::symbolizeTargetDeviceEnum(hivmPipelineOptions.target))) {
    // make sure no alloc within vf and no value returned by vf,
    // so InferHIVMMemScope can work correctly
    pm.addPass(createOutlineAllocInVFPass());
    pm.addPass(createOutlineCopyInVFPass());
    pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  }
  // Infer memory scope for newly allocated extra buffer
  pm.addPass(createInferHIVMMemScopePass());
  canonicalizationHIVMPipeline(pm);

  MarkMultiBufferOptions multiBufferOptions;
  multiBufferOptions.enableAuto = hivmPipelineOptions.enableAutoMultiBuffer;
  // Limit auto multi buffer only work for local buffer at this stage
  multiBufferOptions.limitAutoMultiBufferOnlyForLocalBuffer = true;
  multiBufferOptions.limitAutoMultiBufferOfLocalBuffer =
      hivmPipelineOptions.limitAutoMultiBufferOfLocalBuffer;
  multiBufferOptions.limitMixAutoMultiBufferBuffer =
      hivmPipelineOptions.limitMixAutoMultiBufferBuffer;
  pm.nest<func::FuncOp>().addPass(
      createMarkMultiBufferPass(multiBufferOptions));
  PlanMemoryOptions planMemoryOption;
  planMemoryOption.enablePrintMemoryAllocatedSize =
      hivmPipelineOptions.enablePrintMemoryAllocatedSize;
  planMemoryOption.simtVFDynamicSize = hivmPipelineOptions.simtVFDynamicSize;
  planMemoryOption.disableTightlyCoupledBufferReuse =
      hivmPipelineOptions.disableTightlyCoupledBufferReuse;
  pm.addPass(createPlanMemoryPass(planMemoryOption));

  // Lower hivm ops to loops
  pm.nest<func::FuncOp>().addPass(createHIVMLowerToLoopsPass());
  // TODO: move DecomposeI32ScalarExtOp etc. to interface
  pm.nest<func::FuncOp>().addPass(createHIVMDecomposeOpPass());
  // Intra-Core Auto-Sync passes (Inject-Sync, GSS)
  hivmIntraCoreSyncPipeline(pm, hivmPipelineOptions);
  pm.nest<func::FuncOp>().addPass(createEnableMultiBufferPass());
  pm.nest<func::FuncOp>().addPass(createLiftLowestStridePass());
  canonicalizationHIVMPipeline(pm);
  if (!hivmPipelineOptions.enableDirectHIVMLowering &&
      hacc::utils::isRegBasedArch(
          hacc::symbolizeTargetDeviceEnum(hivmPipelineOptions.target))) {
    pm.nest<func::FuncOp>().addPass(arith::createNormalizeArithPass());
    pm.nest<func::FuncOp>().addPass(arith::createLiftArithIndexCastPass());
    pm.nest<func::FuncOp>().addPass(
        vector::createPeelLoopsContainingTransposePass());
    pm.addPass(createCanonicalizerPass());
    NormalizeVectorOptions normalizeVectorOptions;
    normalizeVectorOptions.enableDotScaledCompile =
        hivmPipelineOptions.enableDotScaledCompile;
    pm.nest<func::FuncOp>().addPass(
        vector::createNormalizeVectorPass(normalizeVectorOptions));
    pm.nest<func::FuncOp>().addPass(createCSEPass());
    pm.nest<func::FuncOp>().addPass(createArithVectorMaskAnalysisPass());
  }
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

void buildHIVMTensorOptimizations(
    OpPassManager &pm, const HIVMPipelineOptions &hivmPipelineOptions) {
  pm.nest<func::FuncOp>().addPass(createInitEntryKernelPass());
  pm.nest<func::FuncOp>().addPass(mlir::hivm::createHIVMNormalizeOpsPass());
  hivmPreBufferizationOptimizationPipeline(pm, hivmPipelineOptions);
}

void buildLowerHIVMPipelines(OpPassManager &pm,
                             const HIVMPipelineOptions &hivmPipelineOptions) {
  bufferizationPipeline(pm, hivmPipelineOptions);
  hivmPostBufferizationOptimizationPipeline(pm, hivmPipelineOptions);
  // Optimizations that relies on scope should be done after this point. Inline
  // all `scope.scope` ops.
  pm.addPass(
      scope::createInlineScopePass(InlineScopeOptions{/*forceInline=*/true}));
  pm.addPass(
      bishengir::createInjectIRPass(hivmPipelineOptions.injectIrFromFile));
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerLowerHIVMPipelines() {
  PassPipelineRegistration<HIVMPipelineOptions>(
      "lower-hivm-pipeline", "lower hivm pipeline",
      [](OpPassManager &pm, const HIVMPipelineOptions &options) {
        buildHIVMTensorOptimizations(pm, options);
        buildLowerHIVMPipelines(pm, options);
      });
}

void registerConvertToHIVMPipelines() {
  PassPipelineRegistration<ConvertToHIVMPipelineOptions>(
      "convert-to-hivm-pipeline", "convert to hivm pipeline",
      [](OpPassManager &pm, const ConvertToHIVMPipelineOptions &options) {
        buildConvertToHIVMPipeline(pm, options);
      });
}

} // namespace hivm
} // namespace mlir
