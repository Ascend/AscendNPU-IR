//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include <memory>

/// Defines a scope for reinterpret map pass.
enum class MultiBufferStrategy {
  NO_LIMIT = 0,
  ONLY_CUBE,
  ONLY_VECTOR,
  CUBE_NO_L0C,
};

/// Cube-Vector pipelining mode
enum class CVPipelineMode {
  Off,     // disable CVPipelining
  Unroll,  // standard unroll-mode pipelining
  Skew,    // skew/preload-mode pipelining
  Dynamic, // dynamic mode pipeling (developing)
};

/// partition-and-bind-sub-block mode
enum class PartitionAndBindSubBlockMode {
  Off = 0,      // disable partition-and-bind sub-block
  DefaultPin,   // enable; pin every free op to sub-block 0
  LoadBalanced, // enable; spread op` across both AIV sub-cores
};

namespace mlir {

namespace hivm {

enum class SyncMode {
  NORMAL,
  BARRIERALL, // only for debug
};

} // namespace hivm
} // namespace mlir

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

namespace hivm {
/// Create a pass to infer the core type of each function.
std::unique_ptr<Pass> createInferFuncCoreTypePass();

/// Create a pass to expose memref-level writes (e.g., hivm.hir.load) to
/// tensor-level analysis by add copyOp for toTensorOp.
std::unique_ptr<Pass> createExposeMemrefWriteToTensorPass();

/// Create a pass to convert ops from other dialects to HIVM Ops.
std::unique_ptr<Pass> createConvertToHIVMOpPass();

/// Create a pass to normalize hivm ops.
std::unique_ptr<Pass> createHIVMNormalizeOpsPass();

/// Create a pass to normalize hivm matmul op.
std::unique_ptr<Pass> createNormalizeMatmulPass();

/// Create a pass to convert args of global kernel function to HIVM Ops.
std::unique_ptr<Pass> createTritonGlobalKernelArgsToHIVMOpPass();

/// Create a pass to infer, propagate, and add memory scope information to
/// HIVM Ops.
std::unique_ptr<Pass> createInferHIVMMemScopePass();

/// Creates an operation pass to convert `memref.AllocOp` with non-global
/// memory space to `memref.AllocaOp`.
std::unique_ptr<Pass> createAllocToAllocaPass();

/// Create a pass to output clones to different empty tensors based on hivmOp.
std::unique_ptr<Pass> createCloneTensorEmptyPass();

/// Create a pass to infer data layout information for HIVM Ops.
std::unique_ptr<Pass> createInferHIVMDataLayoutPass();

/// Create a pass to infer vf mode for Ops.
std::unique_ptr<Pass> createInferVFModePass();

/// Create a pass to mark multi buffer for HIVM Ops.
/// If options is {}, enableAuto is false as default.
/// And this pass contains method for marking workspace multiple buffer, which
/// could be turned off by option 'limitAutoMultiBufferOnlyForLocalBuffer'
std::unique_ptr<Pass>
createMarkMultiBufferPass(const MarkMultiBufferOptions &options = {});

/// Create a pass to enable multi buffer.
std::unique_ptr<Pass> createEnableMultiBufferPass();

/// Create a pass to lower multi-buffer counter anchors.
std::unique_ptr<Pass> createLowerMultiBufferCounterPass();

/// Create a pass to plan memory.
std::unique_ptr<Pass>
createPlanMemoryPass(const PlanMemoryOptions &planMemoryOption = {});

/// Create a pass to inject sync
std::unique_ptr<Pass>
createInjectSyncPass(const InjectSyncOptions &options = {});

/// Create a pass to graph sync solver.
std::unique_ptr<Pass>
createGraphSyncSolverPass(const GraphSyncSolverOptions &options = {});

/// Create a pass to cross-core graph-sync-solver.
std::unique_ptr<Pass>
createCrossCoreGSSPass(const CrossCoreGSSOptions &options = {});

// Create a pass to run delayed cross-core GSS driven by anchors.
std::unique_ptr<Pass>
createDelayedCrossCoreGSSPass(const DelayedCrossCoreGSSOptions &options = {});

// Create a pass to insert anchor operations and backup mixed kernels.
std::unique_ptr<Pass> createInsertAnchorsAndBackupPass(
    const InsertAnchorsAndBackupOptions &options = {});

/// Create a pass to inject block sync
std::unique_ptr<Pass>
createInjectBlockSyncPass(const InjectBlockSyncOptions &options = {});

/// create a pass to decompose
std::unique_ptr<Pass> createHIVMDecomposeOpPass();

/// create a pass to decompose after alignment pipeline
std::unique_ptr<Pass> createHIVMAggregatedDecomposeOpPass(
    const HIVMAggregatedDecomposeOpOptions &options = {});

/// create a pass to lower hivm ops to loops
std::unique_ptr<Pass> createHIVMLowerToLoopsPass();

/// create a pass to opt uncontinuous access to deinterleave
std::unique_ptr<Pass> createHIVMRecognizeDeinterleaveOpPass();

/// create a pass to opt single point operation
std::unique_ptr<Pass> createHIVMOptSinglePointPass();

/// Create a pass to constantize buffers with dynamic sizes.
std::unique_ptr<Pass> createConstantizeBufferSizePass();

/// Create a pass to allocate extra buffer
std::unique_ptr<Pass> createAllocExtraBufferPass();

/// Create a pass to outline memref.alloc with static shape in VF
std::unique_ptr<Pass> createOutlineAllocInVFPass();

/// Create a pass to outline hivm.load in VF by rewriting it to hivm.copy.
std::unique_ptr<Pass> createOutlineCopyInVFPass();

/// Create a pass to remove unnecessary buffer address return
std::unique_ptr<Pass> createHIVMOptFuncOutputPass();

// Create a pass to split davinci aicore and aivector kernel
std::unique_ptr<Pass> createSplitMixKernelPass();

// Create a pass to mark L1/UB allocs with the tightly-coupled-buffer attribute
// before SplitMixKernel clones the MIX function.
std::unique_ptr<Pass> createMarkTightlyCoupledBufferPass();

// Create a pass to hoist yielded tightly-coupled allocs out of inner regions so
// the AIC/AIV multi-buffer anchor stays consistent after SplitMixKernel.
std::unique_ptr<Pass> createHoistTightlyCoupledAllocPass();

// Create a pass to mark scalar operations with core-type attribute.
std::unique_ptr<Pass>
createMarkRealCoreTypePass(const MarkRealCoreTypeOptions &options = {});

// Create a pass to run the HIVM canonicalization pass pipeline on a function.
std::unique_ptr<Pass> createHIVMCanonicalizationPipelinePass();

// Create a pass to set buffer size
std::unique_ptr<Pass> createSetBufferSizePass();

// Create a pass to map forall to hivm blocks.
std::unique_ptr<Pass> createHIVMMapForallToBlocksPass();

// Create a pass to flatten HIVM ops.
std::unique_ptr<Pass> createFlattenOpsPass();

// Create a pass to align alloc size for some HIVM ops that
// has to access aligned size.
std::unique_ptr<Pass> createAlignAllocSizePass();

// Create a pass to pre-analyze which allocs must skip stride alignment
// (e.g., DMA-loaded buffers that will be vload'd in VFs).
std::unique_ptr<Pass> createPreMarkStrideAlignPass();

// Create a pass to annoate storage_align marks for HIVM ops.
std::unique_ptr<Pass> createMarkStrideAlignPass();

// Create a pass to reallocate memrefs according to storage_align marks
std::unique_ptr<Pass> createEnableStrideAlignPass();

// Create a pass to lift the lowest stride of operands
std::unique_ptr<Pass> createLiftLowestStridePass();

// Create a pass to inline OTF broadcast
std::unique_ptr<Pass> createInlineOTFBroadcastPass();

// Create a pass to reduce the rank using subview
std::unique_ptr<Pass> createReduceRankSubviewPass();

// Create a pass to init entry kernel
std::unique_ptr<Pass> createInitEntryKernelPass();

// Create a pass to convert ops to fixpipe
std::unique_ptr<Pass>
createInlineFixpipePass(const InlineFixpipeOptions &options = {});

// Create a pass to insert fixpipe
std::unique_ptr<Pass>
createInsertFixpipePass(const InsertFixpipeOptions &options = {});

// Create a pass to tile batch matmul into loop
std::unique_ptr<Pass> createTileBatchMMIntoLoopPass();

// Create a pass to lift zero rank
std::unique_ptr<Pass> createLiftZeroRankPass();

// Create a pass to insert load/store op for scalar.
std::unique_ptr<Pass> createInsertLoadStoreForScalarPass();

// Create a pass to split if conditionals for mix cv function.
std::unique_ptr<Pass> createSplitMixedIfConditionalsPass();

// Create a pass to insert load/store op for mix cv function.
std::unique_ptr<Pass> createInsertLoadStoreForMixCVPass(
    const InsertLoadStoreForMixCVOptions &options = {});

// Create a pass to insert cv tight coupled buffer for mix cv function.
std::unique_ptr<Pass> createInsertCVTightCoupledBufferPass(
    const InsertCVTightCoupledBufferOptions &options = {});

// Create a pass to insert infer-workspace callback func for host
std::unique_ptr<Pass> createInsertInferWorkSpaceSizeFuncPass();

// Create a pass to insert infer-vf-mode callback func for host
std::unique_ptr<Pass> createInsertInferVFModeFuncPass();

// Create a pass to bind func augument with hacc.workspace to AllocWorkspaceOp
std::unique_ptr<Pass>
createBindWorkSpaceArgPass(const BindWorkSpaceArgOptions &options = {});

// Create a pass to bind func augument with hacc.syncblocklock to
// CreateSyncBlockLockOp.
std::unique_ptr<Pass> createBindSyncBlockLockArgPass();

// Hoist syncblock lock and unlock operation to the parent region if it
// is in the scf.for or scf.while
std::unique_ptr<Pass> createSyncBlockHoistingPass();

// Create a pass to insert infer-sync-block-lock-num and
// infer-sync-block-lock-init callback func for host.
std::unique_ptr<Pass> createInsertInferSyncBlockLockNumAndInitFuncPass();

// Create a pass to lower CreateSyncBlockLockOp.
std::unique_ptr<Pass> createSyncBlockLockLoweringPass();

// Create a pass to insert FreeLockVarOp before return to prevent
// deadlock when control flow skips sync_block_lock/unlock.
std::unique_ptr<Pass> createInsertFreeLockVarBeforeReturnPass();

// Create a pass to auto infer buffer size by inserting Annotation MarkOp
std::unique_ptr<Pass> createAutoInferBufferSizePass();

// Create a pass to insert workspace for mix cv function.
std::unique_ptr<Pass> createInsertWorkSpaceForMixCVPass();

/// Create a pass to Inline Load and Store operation on the fly.
std::unique_ptr<Pass> createHIVMInlineOTFLoadStorePass();

// Create a pass to annotate alias info within VF
std::unique_ptr<Pass> createAnnotateVFAliasPass();

/// Create a pass to reuse a VF input buffer for its output
std::unique_ptr<Pass> createVFOperandSubstitutionPass();

/// Create a pass to remove CopyOps.
std::unique_ptr<Pass> createRemoveCopyOpsPass();

/// Create a pass to analyze arith/vector mask
std::unique_ptr<Pass> createArithVectorMaskAnalysisPass();

/// Create a pass to tile and bind sub block for mix cv function.
std::unique_ptr<Pass>
createTileAndBindSubBlockPass(const TileAndBindSubBlockOptions &options = {});

/// Create the self-gated operand-parallel sub-block pass. Lowers
/// `{sub_block = n}` scope.scope regions to `scf.if` lane guards. No options.
std::unique_ptr<Pass> createPartitionAndBindSubBlockPass(
    const PartitionAndBindSubBlockOptions &options = {});

/// Create the STAGE-3 pass that cleans up value-returning operand-parallel
/// sub-block guards (post-bufferization): bubbles metadata-only memref views off
/// the then-yields, folds the redundant UB->UB copies, and makes each guard
/// result-free. No options.
std::unique_ptr<Pass> createSubBlockGuardCleanupPass();

/// Create a pass to bubble up extract slice for hivm ops.
std::unique_ptr<Pass> createHIVMBubbleUpExtractSlicePass(
    const HIVMBubbleUpExtractSliceOptions &options = {});

/// Create a pass to vectorize hivm ops.
std::unique_ptr<Pass> createHIVMVectorizeOpsPass();

// Create a pass to insert init and finish for debug.
std::unique_ptr<Pass> createInsertInitAndFinishForDebugPass();

// Create a pass to mark memref.loads that need to disable dcache.
std::unique_ptr<Pass> createMarkDisableLoadPass();

/// Create a pass to mark sync_block_lock/sync_block_unlock with
/// sync_block_lock_with_subblock tag when not inside limit_sub_block_id0 if in
/// mix module.
std::unique_ptr<Pass> createMarkSyncBlockLockWithSubblockPass();

// Create a pass to insert nz2nd for debug.
std::unique_ptr<Pass> createInsertNZ2NDForDebugPass();

// Create a pass to insert l12ub for debug.
std::unique_ptr<Pass> createInsertL12UBForDebugPass();

/// Create a pass to loop on blocks when logical blocknum is larger than
/// physical one
std::unique_ptr<Pass> createAutoBlockifyParallelLoopPass();

// Create CV pipelining pass
std::unique_ptr<Pass>
createCVPipeliningPass(const CVPipeliningOptions &options = {});

/// Create a pass to create preload for CV pipelining.
std::unique_ptr<Pass> createCreatePreloadPass();

// Create a pass to compose expands and collapses ops
std::unique_ptr<Pass> createComposeCollapseExpandPass();

std::unique_ptr<Pass> createSinkOpToConsumerInLoopPass();

// Split simt module for every simt vf
std::unique_ptr<Pass> createSplitSimtModulePass();

// Create a pass to infer simt vf func args memory effect.
std::unique_ptr<Pass> createInferSimtVFMemEffectPass();

// Create a pass to infer simt vf func args memory scope hints.
std::unique_ptr<Pass> createInferSimtVFMemScopeHintPass();

// Create a pass to materialize explicit memory scopes inside split simt
// modules.
std::unique_ptr<Pass> createMaterializeSimtVFMemScopePass();

// Create a pass to serially split oversized SIMT VF tiles.
std::unique_ptr<Pass>
createSIMTVFSubTilingPass(const SIMTVFSubTilingOptions &options = {});

// Create a pass to insert convert layout operations for matmul ops
std::unique_ptr<Pass> createInsertConvertLayoutPass();
std::unique_ptr<Pass> createPropagateConvertLayoutPass(
    const PropagateConvertLayoutOptions &options = {});
std::unique_ptr<Pass> createConvertLayoutToTransposePass();
std::unique_ptr<Pass> createInsertCVDataMovementPass();
std::unique_ptr<Pass> createCombineOptimizedConvertLayoutPass();

// Create a pass to insert memory semantic for simt vf.
std::unique_ptr<Pass> createInsertMemSemanticForSimtVFPass();

// Create scope for gather_load and scatter_store
std::unique_ptr<Pass> createAutoScopePass();

std::unique_ptr<Pass> createInsertAllocBasePlaceholderPass();
std::unique_ptr<Pass> createWriteBackSharedPass();

/// Create a pass to fuse linalg.transpose into hivm.hir.load via DMA
/// on-the-fly transpose.
std::unique_ptr<Pass> createFuseTransposeIntoLoadPass();
//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_H
