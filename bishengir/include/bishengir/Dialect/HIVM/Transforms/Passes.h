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
#include "mlir/Pass/Pass.h"
#include <memory>

/// Defines a scope for reinterpret map pass.
enum class MultiBufferStrategy {
  NO_LIMIT = 0,
  ONLY_CUBE,
  ONLY_VECTOR,
  CUBE_NO_L0C,
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

/// Create a pass to clone scf.if.yield operand for PlanMemory.
std::unique_ptr<Pass> createCloneSCFIfYieldOperandPass();

/// Create a pass to convert ops from other dialects to HIVM Ops.
std::unique_ptr<Pass> createConvertToHIVMOpPass();

/// Create a pass to normalize hivm matmul op.
std::unique_ptr<Pass> createNormalizeMatmulPass();

/// Create a pass to convert args of global kernel function to HIVM Ops.
std::unique_ptr<Pass> createTritonGlobalKernelArgsToHIVMOpPass();

/// Create a pass to infer, propagate, and add memory scope information to
/// HIVM Ops.
std::unique_ptr<Pass> createInferHIVMMemScopePass();

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

// Create a pass to mark scalar operations with core-type attribute.
std::unique_ptr<Pass>
createMarkRealCoreTypePass(const MarkRealCoreTypeOptions &options = {});

// Create a pass to set buffer size
std::unique_ptr<Pass> createSetBufferSizePass();

// Create a pass to map forall to hivm blocks.
std::unique_ptr<Pass> createHIVMMapForallToBlocksPass();

// Create a pass to flatten HIVM ops.
std::unique_ptr<Pass> createFlattenOpsPass();

// Create a pass to align alloc size for some HIVM ops that
// has to access aligned size.
std::unique_ptr<Pass> createAlignAllocSizePass();

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
std::unique_ptr<Pass> createInlineFixpipePass();

// Create a pass to convert ops to fixpipe
std::unique_ptr<Pass> createInlineFixpipeV2Pass();

// Create a pass to tile batch matmul into loop
std::unique_ptr<Pass> createTileBatchMMIntoLoopPass();

// Create a pass to lift zero rank
std::unique_ptr<Pass> createLiftZeroRankPass();

// Create a pass to insert load/store op for scalar.
std::unique_ptr<Pass> createInsertLoadStoreForScalarPass();

// Create a pass to split if conditionals for mix cv function.
std::unique_ptr<Pass> createSplitMixedIfConditionalsPass();

// Create a pass to insert load/store op for mix cv function.
std::unique_ptr<Pass> createInsertLoadStoreForMixCVPass();

// Create a pass to insert cv tight coupled buffer for mix cv function.
std::unique_ptr<Pass> createInsertCVTightCoupledBufferPass();

// Create a pass to insert infer-workspace callback func for host
std::unique_ptr<Pass> createInsertInferWorkSpaceSizeFuncPass();

// Create a pass to insert infer-vf-mode callback func for host
std::unique_ptr<Pass> createInsertInferVFModeFuncPass();

// Create a pass to bind func augument with hacc.workspace to AllocWorkspaceOp
std::unique_ptr<Pass> createBindWorkSpaceArgPass();

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

// Create a pass to auto infer buffer size by inserting Annotation MarkOp
std::unique_ptr<Pass> createAutoInferBufferSizePass();

// Create a pass to insert workspace for mix cv function.
std::unique_ptr<Pass> createInsertWorkSpaceForMixCVPass();

// Create a pass to normalize special state of loop iterator before plan-memory
std::unique_ptr<Pass> createNormalizeLoopIteratorPass();

/// Create a pass to Inline Load and Store operation on the fly.
std::unique_ptr<Pass> createHIVMInlineOTFLoadStorePass();

// Create a pass to annotate alias info within VF
std::unique_ptr<Pass> createAnnotateVFAliasPass();

/// Create a pass to remove CopyOps.
std::unique_ptr<Pass> createRemoveCopyOpsPass();

/// Create a pass to analyze arith/vector mask
std::unique_ptr<Pass> createArithVectorMaskAnalysisPass();

/// Create a pass to tile and bind sub block for mix cv function.
std::unique_ptr<Pass> createTileAndBindSubBlockPass();

/// Create a pass to bubble up extract slice for hivm ops.
std::unique_ptr<Pass> createHIVMBubbleUpExtractSlicePass();

/// Create a pass to vectorize hivm ops.
std::unique_ptr<Pass> createHIVMVectorizeOpsPass();

// Create a pass to mark memref.loads that need to disable dcache.
std::unique_ptr<Pass> createMarkDisableLoadPass();

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

// Create a pass to compose expands and collapses ops
std::unique_ptr<Pass> createComposeCollapseExpandPass();

std::unique_ptr<Pass> createSinkOpToConsumerInLoopPass();

// Split simt module for every simt vf
std::unique_ptr<Pass> createSplitSimtModulePass();

// Create a pass to infer simt vf func args memory effect.
std::unique_ptr<Pass> createInferSimtVFMemEffectPass();

// Create a pass to insert convert layout operations for matmul ops
std::unique_ptr<Pass> createInsertConvertLayoutPass();
std::unique_ptr<Pass> createPropagateConvertLayoutPass();
std::unique_ptr<Pass> createConvertLayoutToTransposePass();
std::unique_ptr<Pass> createInsertCVDataMovementPass();
std::unique_ptr<Pass> createCombineOptimizedConvertLayoutPass();

// Create a pass to insert memory semantic for simt vf.
std::unique_ptr<Pass> createInsertMemSemanticForSimtVFPass();

// Create scope for gather_load and scatter_store
std::unique_ptr<Pass> createAutoScopePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_H
