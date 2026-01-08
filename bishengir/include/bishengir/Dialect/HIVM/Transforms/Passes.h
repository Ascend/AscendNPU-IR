//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
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

/// Create a pass to convert ops from other dialects to HIVM Ops.
std::unique_ptr<Pass> createConvertToHIVMOpPass();

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

/// Create a pass to mark multi buffer for HIVM Ops.
/// If options is {}, enableAuto is false as default.
/// And this pass contains method for marking workspace multiple buffer, which
/// could be turned off by option 'limitAutoMultiBufferOnlyForLocalBuffer'
std::unique_ptr<Pass>
createMarkMultiBufferPass(const MarkMultiBufferOptions &options = {});

/// Create a pass to enable multi buffer.
std::unique_ptr<Pass> createEnableMultiBufferPass();

/// Create a pass to add FFTS (arg0) to every SyncBlockSetOp
std::unique_ptr<Pass> createAddFFTSToSyncBlockSetOpPass();

/// Create a pass to plan memory.
std::unique_ptr<Pass>
createPlanMemoryPass(const PlanMemoryOptions &planMemoryOption = {});

/// Create a pass to inject sync
std::unique_ptr<Pass>
createInjectSyncPass(const InjectSyncOptions &options = {});

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

/// Create a pass to remove unnecessary buffer address return
std::unique_ptr<Pass> createHIVMOptFuncOutputPass();

/// Create a pass to infer task type
std::unique_ptr<Pass> createInsertInferTaskTypeFuncPass();

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

// Create a pass to inline Copied load
std::unique_ptr<Pass> createInlineLoadCopyPass();

// Create a pass to reduce the rank using subview
std::unique_ptr<Pass> createReduceRankSubviewPass();

// Create a pass to init entry kernel
std::unique_ptr<Pass> createInitEntryKernelPass();

// Create a pass to convert ops to fixpipe
std::unique_ptr<Pass> createInlineFixpipePass();

// Create a pass to tile batch matmul into loop
std::unique_ptr<Pass> createTileBatchMMIntoLoopPass();

// Create a pass to lift zero rank
std::unique_ptr<Pass> createLiftZeroRankPass();

// Create a pass to insert load/store op for mix cv function.
std::unique_ptr<Pass> createInsertLoadStoreForMixCVPass();

// Create a pass to insert infer-workspace callback func for host
std::unique_ptr<Pass> createInsertInferWorkSpaceSizeFuncPass();

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

/// Create a pass to tile and bind sub block for mix cv function.
std::unique_ptr<Pass> createTileAndBindSubBlockPass();

/// Create a pass to bubble up extract slice for hivm ops.
std::unique_ptr<Pass> createHIVMBubbleUpExtractSlicePass();

// Create a pass to insert init and finish for debug.
std::unique_ptr<Pass> createInsertInitAndFinishForDebugPass();

// Create a pass to mark memref.loads that need to disable dcache.
std::unique_ptr<Pass> createMarkDisableLoadPass();

// Create a pass to insert nz2nd for debug.
std::unique_ptr<Pass> createInsertNZ2NDForDebugPass();

/// Create a pass to loop on blocks when logical blocknum is larger than
/// physical one
std::unique_ptr<Pass> createAutoBlockifyParallelLoopPass();

// Create CV pipelining pass
std::unique_ptr<Pass>
createCVPipeliningPass(const CVPipeliningOptions &options = {});

// Create a pass to compose expands and collapses ops
std::unique_ptr<Pass> createComposeCollapseExpandPass();

/// Create a pass to tile cube and vector loop on local buffer.
std::unique_ptr<Pass>
createTileCubeVectorLoopPass(const TileCubeVectorLoopOptions &options = {});

/// Create a pass to generate copy for reassociative reshape that might be
/// non-contiguous.
std::unique_ptr<Pass> createNonContiguousReshapeToCopyPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_H
