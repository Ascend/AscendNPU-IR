//===- Passes.td - HIVM pass definition file ---------------*- tablegen -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def InferFuncCoreType : Pass<"hivm-infer-func-core-type", "ModuleOp"> {
  let summary = "Infer the core type of each function";
  let constructor = "mlir::hivm::createInferFuncCoreTypePass()";
  let dependentDialects = ["hivm::HIVMDialect", ];
}

def ConvertToHIVMOp : Pass<"convert-to-hivm-op"> {
  let summary = "Convert Ops from other dialects to HIVM Ops";
  let constructor = "mlir::hivm::createConvertToHIVMOpPass()";
  let dependentDialects = ["arith::ArithDialect", "hivm::HIVMDialect",
                           "tensor::TensorDialect",
  ];
}

def NormalizeMatmul : Pass<"hivm-normalize-matmul"> {
  let summary = "Normalize hivm matmul op";
  let constructor = "mlir::hivm::createNormalizeMatmulPass()";
  let dependentDialects = ["arith::ArithDialect", "hivm::HIVMDialect",
                           "tensor::TensorDialect",
  ];
}

def NormalizeBitwiseSelect : Pass<"hivm-normalize-bitwise-select"> {
  let summary = "Normalize hivm bitwise select op";
  let constructor = "mlir::hivm::createNormalizeBitwiseSelectPass()";
  let dependentDialects = ["arith::ArithDialect", "hivm::HIVMDialect",
                           "tensor::TensorDialect",
  ];
}

def TritonGlobalKernelArgsToHIVMOp
    : Pass<"triton-global-kernel-args-to-hivm-op", "func::FuncOp"> {
  let summary = "";
  let constructor = "mlir::hivm::createTritonGlobalKernelArgsToHIVMOpPass()";
  let dependentDialects = ["hivm::HIVMDialect", "hacc::HACCDialect", "annotation::AnnotationDialect"];
}

def InferHIVMMemScope : Pass<"hivm-infer-mem-scope"> {
  let summary = "Infer memory scope for HIVM Ops";
  let constructor = "mlir::hivm::createInferHIVMMemScopePass()";
  let dependentDialects = ["hivm::HIVMDialect", "memref::MemRefDialect", ];
}

def MarkMultiBuffer : Pass<"hivm-mark-multi-buffer", "func::FuncOp"> {
  let summary = "Mark multi buffer for HIVM Ops";
  let description = [{
    This pass mark multi buffer for hivm ops if the option enable-auto is true.
    Note that Buffer with scope L0C would not be marked.
    If enable-auto is false, do nothing.
  }];

  let constructor = "mlir::hivm::createMarkMultiBufferPass()";
  let dependentDialects = [
    "hivm::HIVMDialect",
    "annotation::AnnotationDialect",
  ];
  let options = [
    Option<"enableAuto", "enable-auto", "bool",
           /*default=*/"false", "Mark multi buffer automatically.">,
    Option<"limitAutoMultiBufferOnlyForLocalBuffer",
           "limit-auto-multi-buffer-only-for-local-buffer", "bool",
           /*default=*/"false", "Disable multi-buffer mark on workspace">,
    Option<"limitAutoMultiBufferOfLocalBuffer",
           "limit-auto-multi-buffer-of-local-buffer", "MultiBufferStrategy",
           /*default*/ "MultiBufferStrategy::CUBE_NO_L0C",
           "Limit local buffer auto multi buffer",
           [{::llvm::cl::values(
               clEnumValN(MultiBufferStrategy::NO_LIMIT, "no-limit",
                          "no limit for local multi buffer"),
               clEnumValN(MultiBufferStrategy::CUBE_NO_L0C, "no-l0c",
                          "Disable l0c multi buffer"))}]>,
    Option<"limitMixAutoMultiBufferBuffer",
           "limit-mix-auto-multi-buffer-buffer", "MultiBufferStrategy",
           /*default=*/"MultiBufferStrategy::ONLY_CUBE",
           "Disable multi-buffer-buffer on cube, vector Or no limit">,
    Option<"workspaceMultiBufferNum", "set-workspace-multibuffer", "unsigned",
           /*default=*/"2", "Override for multibuffer number for workspace">,
  ];
}

def EnableMultiBuffer : Pass<"hivm-enable-multi-buffer", "func::FuncOp"> {
  let summary = "Enable multi buffer";
  let description = [{
    This pass enable multi buffer for hivm op if the op is marked "hivm.multi_buffer".
  }];

  let constructor = "mlir::hivm::createEnableMultiBufferPass()";
  let dependentDialects = ["mlir::affine::AffineDialect", ];
}

def AddFFTSToSyncBlockSetOp : Pass<"hivm-add-ffts-to-syncblocksetop", "func::FuncOp"> {
  let summary = "Add FFTS (arg0) to SyncBlockSetOp";
  let description = [{
    This pass adds FFTS (arg0) to SyncBlockSetOp.
  }];

  let constructor = "mlir::hivm::createAddFFTSToSyncBlockSetOpPass()";
  let dependentDialects = ["hivm::HIVMDialect", ];
}

def AllocToAlloca : Pass<"hivm-memref-alloc-to-alloca"> {
  let summary = "Convert local AllocOp to AllocaOp";
  let description = [{This pass replace all memref.alloc with non -
                      global memory space to memref.alloca.}];
  let constructor = "mlir::hivm::createAllocToAllocaPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def CloneTensorEmpty : Pass<"hivm-clone-tensor-empty", "func::FuncOp"> {
  let summary = "Output clones to different empty tensors based on hivmOp.";
  let description = [{This pass clone different tensor.empty to hivmOp output}];
  let constructor = "mlir::hivm::createCloneTensorEmptyPass()";
  let dependentDialects = ["hivm::HIVMDialect", "tensor::TensorDialect"];
}

def InferHIVMDataLayout : Pass<"hivm-infer-data-layout", "func::FuncOp"> {
  let summary = "Infer data layout for HIVM Ops";
  let constructor = "mlir::hivm::createInferHIVMDataLayoutPass()";
  let dependentDialects = ["affine::AffineDialect", "hivm::HIVMDialect"];
}

def PlanMemory : Pass<"hivm-plan-memory", "func::FuncOp"> {
  let summary = "Plan memory for HIVM Ops";
  let constructor = "mlir::hivm::createPlanMemoryPass()";
  let dependentDialects = ["hivm::HIVMDialect", ];
  let options = [
    Option<"memMode", "mem-plan-mode", "hivm::MemPlanMode",
           "hivm::MemPlanMode::LOCAL_MEM_PLAN",
           "plan mem mode (default is LOCAL_MEM_PLAN)",
           [{::llvm::cl::values(
               clEnumValN(hivm::MemPlanMode::LOCAL_MEM_PLAN, "local-mem-plan",
                          "plan mem mode is for memref.alloc"),
               clEnumValN(
                   hivm::MemPlanMode::GLOBAL_WORKSPACE_PLAN,
                   "global-work-space-plan",
                   "plan mem mode is for memref_ext.alloc_workspace"))}]>,
    Option<"enableGlobalReuse", "enable-global-workspace-reuse", "bool",
           /*default=*/"false",
           "Enable global workspace reuse ,default : false">,
    Option<"restrictInplaceAsISA", "restrict-inplace-as-isa", "bool",
           /*default=*/"false",
           "restrict memory inplace as isa, default : false">,
  ];
}

def InjectSync : Pass<"hivm-inject-sync", "func::FuncOp"> {
  let summary = "auto inject sync ";
  let constructor = "mlir::hivm::createInjectSyncPass()";
  let dependentDialects = ["affine::AffineDialect", "hivm::HIVMDialect", ];
  let options = [Option<"syncMode", "sync-mode", "hivm::SyncMode",
                        "hivm::SyncMode::NORMAL",
                        "injetc sync mode (default is inject normal)",
                        [{::llvm::cl::values(
               clEnumValN(
                   hivm::SyncMode::BARRIERALL, "barrier-all",
                   "inject sync mode is inject pipe all, only for debug"),
               clEnumValN(hivm::SyncMode::NORMAL, "normal",
                          "inject sync mode is inject sync"))}]>,
                  Option<"enableUnitFlag", "enable-unit-flag",
                          "bool", /*default=*/"false",
                          "Enable unit-flag modes for synchronization">,
                  Option<"assumeAliveLoops", "assume-alive-loops",
                          "bool", /*default=*/"false",
                          "Assume that all loops (forOp whileOp) will execute at least once.">
  ];
}

def InjectBlockSync : Pass<"hivm-inject-block-sync", "func::FuncOp"> {
  let summary = "auto inject block sync";
  let constructor = "mlir::hivm::createInjectBlockSyncPass()";
  let dependentDialects = ["hacc::HACCDialect", "hivm::HIVMDialect",
                           "tensor::TensorDialect",
  ];
  let options = [Option<"blockAllSync",
           "block-all-sync", "bool",/*default=*/"false",
           "Enable inject all block sync for HIVM injectBlockSync.">,
            Option<"assumeAliveLoops", "assume-alive-loops",
            "bool", /*default=*/"false",
            "Assume that all loops (forOp whileOp) will execute at least once.">,
            Option<"disableAutoInjectBlockSync", "disable-auto-inject-block-sync",
           "bool", /*default=*/"false",
           "Toggle auto set/wait insertion, always keep SetFFTSBaseAddrOp">
  ];
}

def GraphSyncSolver : Pass<"hivm-graph-sync-solver", "func::FuncOp"> {
  let summary = "graph sync solver ";
  let constructor = "mlir::hivm::createGraphSyncSolverPass()";
  let dependentDialects = ["affine::AffineDialect", "hivm::HIVMDialect", ];
  let options = [Option<"enableUnitFlag", "enable-unit-flag", "bool",
                        /*default=*/"false",
                        "Enable unit-flag modes for synchronization">];
}

def HIVMDecomposeOp : Pass<"hivm-decompose-op", "func::FuncOp"> {
  let summary = "decompose compound hivm op to multiple hivm ops according to "
                "hardware ability.";
  let description = [{
    This pass decomposes compound hivm op to multiple hivm ops according to hardware ability.
    For example, hardware cannot cast f32 to i8 type directly, therefore it needs
    to be composed to the combination of f32 to f16 cast op and f16 to i8 cast op.
    In dynamic cases, create annotation.markOp with attr buffer_size_in_byte for allocated extra
    buffer, the value of buffer_size_in_byte is same as the src or dst operand of original op.
  }];
  let constructor = "mlir::hivm::createHIVMDecomposeOpPass()";
  let dependentDialects = ["hivm::HIVMDialect", "memref::MemRefDialect",
                           "tensor::TensorDialect", "annotation::AnnotationDialect"];
}

def HIVMAggregatedDecomposeOp : Pass<"hivm-aggregated-decompose-op", "func::FuncOp"> {
  let summary = "decompose hivm ops that use hivm AggregatedOpInterface";
  let constructor = "mlir::hivm::createHIVMAggregatedDecomposeOpPass()";
  let dependentDialects = ["hivm::HIVMDialect", "memref::MemRefDialect",
                           "tensor::TensorDialect", "annotation::AnnotationDialect"];
  let options = [
    Option<
        "decomposePhase", "decompose-phase", "bishengir::DecomposePhase",
        /*default=*/"bishengir::DecomposePhase::NO_CONSTRAINT",
        "Specify which decompose phase to apply.",
        [{::llvm::cl::values(
            clEnumValN(bishengir::DecomposePhase::NO_CONSTRAINT,
                       "no-constraint",
                       "decompose with no constraint specified"),
            clEnumValN(bishengir::DecomposePhase::BEFORE_HIVM_STRIDE_ALIGNMENT,
                       "before-hivm-align",
                       "for ops that get decompose before stride alignment"),
            clEnumValN(bishengir::DecomposePhase::AFTER_RECOGNIZE_DEINTERLEAVE,
                       "after-hivm-recognize-deinterleave",
                       "for ops that get decompose after recognizing deinterleave"),
            clEnumValN(bishengir::DecomposePhase::AFTER_RECOGNIZE_BROADCAST,
                       "after-hivm-recognize-broadcast",
                       "for ops that get decompose after recognizing broadcast"),
            clEnumValN(bishengir::DecomposePhase::AFTER_HIVM_STRIDE_ALIGNMENT,
                       "after-hivm-align",
                       "for ops that get decompose after stride alignment"),
            clEnumValN(bishengir::DecomposePhase::AFTER_INFER_HIVM_DATA_LAYOUT,
                       "after-infer-hivm-data-layout",
                       "for ops that get decompose after infer HIVM data "
                       "layout"),
            clEnumValN(bishengir::DecomposePhase::AFTER_HIVM_FLATTEN_OPS,
                       "after-hivm-flatten-ops",
                       "for ops that get decompose after HIVM flatten ops"))}]>,
  ];
}

def HIVMLowerToLoops : Pass<"hivm-lower-to-loops", "func::FuncOp"> {
  let summary = "lower hivm ops to loops";
  let constructor = "mlir::hivm::createHIVMLowerToLoopsPass()";
  let dependentDialects = ["hivm::HIVMDialect", "memref::MemRefDialect",
                           "tensor::TensorDialect"];
}

def HIVMRecognizeDeinterleaveOp : Pass<"hivm-recognize-deinterleave-op", "func::FuncOp"> {
  let summary = "optimize uncontinuous access to deinterleave.";
  let description = [{
    This pass optimize uncontinuous memory access using deinterleave.
  }];
  let constructor = "mlir::hivm::createHIVMRecognizeDeinterleaveOpPass()";
  let dependentDialects = ["hivm::HIVMDialect", "memref::MemRefDialect",
                           "tensor::TensorDialect"];
}

def HIVMOptSinglePointOp : Pass<"hivm-opt-single-point", "func::FuncOp"> {
  let summary = "optimize single point hivm op by scalar operation.";
  let description =
      [{This pass optimize the single point hivm op by scalar operation.}];
  let constructor = "mlir::hivm::createHIVMOptSinglePointPass()";
  let dependentDialects = ["hivm::HIVMDialect", "arith::ArithDialect",
                           "math::MathDialect", "memref::MemRefDialect"];
}

def AllocExtraBuffer : Pass<"hivm-alloc-extra-buffer", "func::FuncOp"> {
  let summary = "allocate additional temporary buffer for op needed";
  let constructor = "mlir::hivm::createAllocExtraBufferPass()";
  let dependentDialects = ["hivm::HIVMDialect", ];
}

def ConstantizeBufferSize
    : Pass<"hivm-constantize-buffer-size", "func::FuncOp"> {
  let summary = "Try to constantize dynamic shape buffers.";
  let description = [{
    This pass tries to constantize dyanmic shape buffers by upper-bounding
    their original shape.
    If successful, a new, static shaped alloc will be created and subviewed to
    the original shape for further use.
  }];
  let constructor = "mlir::hivm::createConstantizeBufferSizePass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def HIVMOptFuncOutput : Pass<"hivm-opt-func-output", "ModuleOp"> {
  let summary = "Try to optimize function output after bufferization.";
  let description = [{Try to remove unnecessary address return.}];
  let constructor = "mlir::hivm::createHIVMOptFuncOutputPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def InsertInferTaskTypeFunc : Pass<"hivm-insert-infer-task-type-func", "ModuleOp"> {
  let summary = "Infer the module's task type and emit a host function that returns it.";
  let description = [{
    Detect whether the module is CubeVectorMix, CubeOnly, VectorOnly or Unknown,
    then emit a host‑side function `<original_func>_infer_task_type_function` that
    returns an i8 constant encoding the detected type and is marked with the
    appropriate HACC host‑function attributes.
  }];
  let constructor = "mlir::hivm::createInsertInferTaskTypeFuncPass()";
  let dependentDialects = ["func::FuncDialect", "hacc::HACCDialect"];
}

def SplitMixKernel : Pass<"hivm-split-mix-kernel", "ModuleOp"> {
  let summary =
      "Split Mix device functions into AICube and AIVector functions.";
  let description = [{
    Split mix kernels into sperate AICube and AIVector kernels, and mark the
    parent module as a Mix module.

    Note:
      * If a Mix kernel is called within a Host function, a function declaration
        is generated for the final kernel launch. Currently don't support calling
        Mix kernel within a device function.

    Input:

    ```mlir
    func (workspace) attribute {tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>} {
      t = cube_op ins() outs(workspace)
      ... = vector_op ins(t) ...
    }
    ```

    Output :

    ```mlir
    func (workspace) attribute {tcore_type = #hivm.tcore_type<CUBE>} {
      t = cube_op ins() outs(workspace)
      annotation.mark t // mark to avoid dce
    }

    func (workspace) attribute {tcore_type = #hivm.tcore_type<VECTOR>} {
      ... = vector_op ins(workspace) ...
    }
    ```
  }];
  let constructor = "mlir::hivm::createSplitMixKernelPass()";
  let dependentDialects = ["func::FuncDialect", "annotation::AnnotationDialect",
                           "hacc::HACCDialect"];
}

def MarkRealCoreType : Pass<"hivm-mark-real-core-type", "ModuleOp"> {
  let summary = "Mark scalar operations with core-type attribute.";
  let constructor = "mlir::hivm::createMarkRealCoreTypePass()";
  let dependentDialects = ["func::FuncDialect"];
  let options = [Option<"removeCoreTypeAttrs", "remove-core-type-attrs", "bool",
                        /*default=*/"false",
                        "Remove all core type attributes. If set to true, this "
                        "pass becomes a cleanup pass.">];
}

def SetBufferSize : Pass<"hivm-set-buffer-size", "func::FuncOp"> {
  let constructor = "mlir::hivm::createSetBufferSizePass()";
  let dependentDialects = ["arith::ArithDialect", "memref::MemRefDialect"];
}

def HIVMMapForallToBlocks : Pass<"hivm-map-forall-to-blocks", "func::FuncOp"> {
  let summary = "Map forall to hivm blocks.";
  let description = [{
    This pass maps each scf.forall operations to HIVM block ops.
    Mapping is one-to-one and the induction variables of scf.
    forall are rewritten to hivm block idx ops.
  }];
  let constructor = "mlir::hivm::createHIVMMapForallToBlocksPass()";
  let dependentDialects = ["hivm::HIVMDialect", ];
}

def HIVMFlattenOps : Pass<"hivm-flatten-ops", "func::FuncOp"> {
  let summary = "Flatten HIVM ops.";
  let constructor = "mlir::hivm::createFlattenOpsPass()";
  let dependentDialects = ["memref::MemRefDialect", "hivm::HIVMDialect", ];
}

def SinkOpToConsumerInLoop : Pass<"hivm-sink-op-to-consumer-in-loop", "func::FuncOp"> {
  let constructor = "mlir::hivm::createSinkOpToConsumerInLoopPass()";
  let dependentDialects = ["hivm::HIVMDialect"];
}

def AlignAllocSize : Pass<"hivm-align-alloc-size", "func::FuncOp"> {
  let summary =
      "Automatically align memref.alloc size for special hivm op that "
      "has to access aligned size";
  let description = [{
    For some hivm ops, its access size can only be aligned to hw unit size,
    so this pass need adjust the memref.alloc size for the case to avoid access
    out of bounds.
  }];
  let constructor = "mlir::hivm::createAlignAllocSizePass()";
  let dependentDialects = ["mlir::memref::MemRefDialect", "hivm::HIVMDialect",
                           "annotation::AnnotationDialect"];
}

def MarkStrideAlign : Pass<"hivm-mark-stride-align", "func::FuncOp"> {
  let summary =
      "Automatically annotate stride_align marks for operands of hivm ops";
  let description = [{
    For all hivm ops, annotate their memref operands with storage_align marks automatically
  }];
  let constructor = "mlir::hivm::createMarkStrideAlignPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect", "hivm::HIVMDialect",
                           "annotation::AnnotationDialect",
  ];
}

def EnableStrideAlign : Pass<"hivm-enable-stride-align", "func::FuncOp"> {
  let summary = "Align memref allocations according to stride align marks";
  let description = [{
    Re-allocate memrefs according to anntations of storage_align marks
  }];
  let constructor = "mlir::hivm::createEnableStrideAlignPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect",
                           "annotation::AnnotationDialect",
                           "arith::ArithDialect",
  ];
}

def LiftLowestStride : Pass<"hivm-lift-lowest-stride", "func::FuncOp"> {
  let summary = "Lift lowest stride of operands of hivm ops";
  let description = [{
    For most hivm structured op, lift the lowest stride of operands,
    if the last dim is not contiguous.

    Exceptions: MacroOp and VArangeOp.

    For exmaple, the type of operand is memref<16xf16, strided<[8]>>,
    after LiftLowestStride, the type would be
    memref<16x1xf32, strided<[8, 1]>> with contiguous last dim.
  }];
  let constructor = "mlir::hivm::createLiftLowestStridePass()";
  let dependentDialects = ["mlir::memref::MemRefDialect", "hivm::HIVMDialect",
  ];
}

def ReduceRankSubview : Pass<"hivm-reduce-rank-subview", "func::FuncOp"> {
  let summary = "Reduce rank using subview";
  let constructor = "mlir::hivm::createReduceRankSubviewPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect", "hivm::HIVMDialect",
  ];
}

def InlineOTFBroadcast : Pass<"hivm-inline-otf-broadcast", "func::FuncOp"> {
  let summary = "Inline OTF broadcast";
  let constructor = "mlir::hivm::createInlineOTFBroadcastPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect", "hivm::HIVMDialect",
  ];
}

def InlineLoadCopy : Pass<"hivm-inline-load-copy", "func::FuncOp"> {
  let summary = "Inline Copied load";
  let constructor = "mlir::hivm::createInlineLoadCopyPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect", "hivm::HIVMDialect",
  ];
}

def InitEntryKernel : Pass<"hivm-init-entry-kernel", "func::FuncOp"> {
  let summary = "Insert set_mask_norm() at the beginning of entry kernel";
  let constructor = "mlir::hivm::createInitEntryKernelPass()";
  let dependentDialects = ["hivm::HIVMDialect", ];
}

def InlineFixpipe : Pass<"hivm-inline-fixpipe"> {
  let summary = "Convert ops to HIVM Fixpipe op";
  let constructor = "mlir::hivm::createInlineFixpipePass()";
  let dependentDialects =
      ["hivm::HIVMDialect", "bishengir::memref_ext::MemRefExtDialect"];
}

def TileBatchMMIntoLoop : Pass<"hivm-tile-batchmm-into-loop", "func::FuncOp"> {
  let summary = "Tile batch matmul into loop with iteration on batch dimension";
  let constructor = "mlir::hivm::createTileBatchMMIntoLoopPass()";
  let dependentDialects = ["hivm::HIVMDialect"];
}

def LiftZeroRank : Pass<"hivm-lift-zero-rank", "func::FuncOp"> {
  let constructor = "mlir::hivm::createLiftZeroRankPass()";
  let dependentDialects = ["arith::ArithDialect", "memref::MemRefDialect"];
}

def InsertLoadStoreForMixCV
    : Pass<"hivm-insert-load-store-for-mix-cv", "func::FuncOp"> {
  let summary = "insert load store op for mix cv";
  let constructor = "mlir::hivm::createInsertLoadStoreForMixCVPass()";
  let dependentDialects = ["hivm::HIVMDialect", "bishengir::memref_ext::MemRefExtDialect"];
}

def InsertInferWorkSpaceSizeFunc : Pass<"hivm-insert-infer-workspace-size-func",
                                        "func::FuncOp"> {
  let summary = "Insert infer-workspace callback func for host";
  let description =
                "Calculate total static workspace size after plan-workspace "
                "pass and then create host callback to return this size";

  let constructor = "mlir::hivm::createInsertInferWorkSpaceSizeFuncPass()";
  let dependentDialects = ["memref::MemRefDialect",
                           "hivm::HIVMDialect",
                           "bishengir::memref_ext::MemRefExtDialect"];
}

def BindWorkSpaceArg : Pass<"hivm-bind-workspace-arg", "func::FuncOp"> {
  let summary = "Bind func augument with hacc.workspace to AllocWorkspaceOp";

  let constructor = "mlir::hivm::createBindWorkSpaceArgPass()";
  let dependentDialects = ["memref::MemRefDialect",
                           "hivm::HIVMDialect",
                           "bishengir::memref_ext::MemRefExtDialect"];
}

def BindSyncBlockLockArg : Pass<"hivm-bind-sync-block-lock-arg", "func::FuncOp"> {
  let summary = "Bind func augument with hacc.syncblocklock to CreateSyncBlockLockOp";

  let constructor = "mlir::hivm::createBindSyncBlockLockArgPass()";
  let dependentDialects = ["memref::MemRefDialect",
                           "hivm::HIVMDialect"];
}

def SyncBlockHoisting : Pass<"hivm-sync-block-hoisting", "func::FuncOp"> {
  let summary = "Hoist syncblock lock and unlock operation to the parent region "
                "if it is in the scf.for or scf.while";

  let constructor = "mlir::hivm::createSyncBlockHoistingPass()";
  let dependentDialects = ["memref::MemRefDialect",
                           "hivm::HIVMDialect"];
}

def InsertInferSyncBlockLockNumAndInitFunc : Pass<"hivm-insert-infer-sync-block-lock-num-and-init-func",
                                        "func::FuncOp"> {
  let summary = "Insert infer-sync-block-lock callback func for host";
  let description =
                "Calculate total static sync block lock num and init"
                "and then create host callback to return this size";

  let constructor = "mlir::hivm::createInsertInferSyncBlockLockNumAndInitFuncPass()";
  let dependentDialects = ["memref::MemRefDialect",
                           "hivm::HIVMDialect"];
}

def LowerCreateSyncBlockLock : Pass<"hivm-lower-create-sync-block-lock", "func::FuncOp"> {
  let summary = "Lower CreateSyncBlockLockOp to ViewOp";

  let constructor = "mlir::hivm::createSyncBlockLockLoweringPass()";
  let dependentDialects = ["memref::MemRefDialect",
                           "hivm::HIVMDialect"];
}

def AutoInferBufferSize : Pass<"hivm-auto-infer-buffer-size", "func::FuncOp"> {
  let summary = "Auto infer buffer size";
  let description = "infer buffer size by inserting annotation.mark Op";

  let constructor = "mlir::hivm::createAutoInferBufferSizePass()";
  let dependentDialects = ["annotation::AnnotationDialect"];
}

def InsertWorkSpaceForMixCV : Pass<"insert-workspace-for-mix-cv"> {
  let summary = "insert workspace for mix cv";
  let constructor = "mlir::hivm::createInsertWorkSpaceForMixCVPass()";
  let dependentDialects = [
    "hivm::HIVMDialect", "bishengir::memref_ext::MemRefExtDialect",
                         "bufferization::BufferizationDialect"
  ];
}

def NormalizeLoopIterator : Pass<"hivm-normalize-loop-iterator",
                                 "func::FuncOp"> {
  let summary = "Normalize special state of loop iterator before plan-memory";
  let constructor = "mlir::hivm::createNormalizeLoopIteratorPass()";
  let dependentDialects = ["hivm::HIVMDialect", "scf::SCFDialect"];
}

def HIVMInlineOTFLoadStore : Pass<"hivm-inline-otf-load-store", "func::FuncOp"> {
  let summary = "On the fly Inline Load and Store operations";
  let constructor = "mlir::hivm::createHIVMInlineOTFLoadStorePass()";
  let dependentDialects = ["bufferization::BufferizationDialect",
                           "memref::MemRefDialect","tensor::TensorDialect",
                           "hivm::HIVMDialect"];
}

def TileAndBindSubBlock : Pass<"hivm-bind-sub-block", "ModuleOp"> {
  let summary = "tile and bind sub block";
  let constructor = "mlir::hivm::createTileAndBindSubBlockPass()";
  let dependentDialects = [
    "annotation::AnnotationDialect", "func::FuncDialect", "hivm::HIVMDialect"
  ];
}

def HIVMBubbleUpExtractSlice : Pass<"hivm-bubble-up-extract-slice", "func::FuncOp"> {
  let summary = "tile and bind sub block";
  let constructor = "mlir::hivm::createHIVMBubbleUpExtractSlicePass()";
  let dependentDialects = [
    "annotation::AnnotationDialect", "func::FuncDialect", "hivm::HIVMDialect"
  ];
}

def InsertInitAndFinishForDebug : Pass<"hivm-insert-init-and-finish-for-debug"> {
  let summary = "insert init and finish for debug";
  let constructor = "mlir::hivm::createInsertInitAndFinishForDebugPass()";
  let dependentDialects = ["hivm::HIVMDialect"];
}

def MarkDisableLoad : Pass<"hivm-mark-disable-load"> {
  let summary = "mark the memref.loads that need to disable dcache";
  let constructor = "mlir::hivm::createMarkDisableLoadPass()";
  let dependentDialects = ["hivm::HIVMDialect", "memref::MemRefDialect"];
}

def InsertNZ2NDForDebug : Pass<"hivm-insert-nz2nd-for-debug"> {
  let summary = "insert nz2nd for debug";
  let constructor = "mlir::hivm::createInsertNZ2NDForDebugPass()";
  let dependentDialects = [
    "hivm::HIVMDialect",
    "bishengir::memref_ext::MemRefExtDialect",
    "bufferization::BufferizationDialect"
  ];
}

def CVPipelining : Pass<"cv-pipelining", "func::FuncOp"> {
  let summary = "Cube and vector core pipelining for multi-buffer'ed mix-cv ops";
  let constructor = "mlir::hivm::createCVPipeliningPass()";
  let dependentDialects = ["hivm::HIVMDialect", "scf::SCFDialect"];
  let options = [Option<
      "enableAutoBalance", "enable-auto-balance", "bool",
      /*default=*/"true",
      "Enable balancing of vector subtasks during pipelining.">];
}

def AutoBlockifyParallelLoop
    : Pass<"auto-blockify-parallel-loop", "func::FuncOp"> {
  let summary = "Enable auto loop on blocks when logical blocknum is larger than physical one";
  let constructor = "mlir::hivm::createAutoBlockifyParallelLoopPass()";
  let dependentDialects = ["hivm::HIVMDialect", "hacc::HACCDialect"];
}

def ComposeCollapseExpand : Pass<"compose-collapse-expand"> {
  let summary = "compose collapse and expand op";
  let constructor = "mlir::hivm::createComposeCollapseExpandPass()";
}

def TileCubeVectorLoop : Pass<"tile-cube-vector-loop", "ModuleOp"> {
  let summary = "Tile cube and vector loops on local buffer";
  let description = [{
    This pass will attemp to tile cube and vector ops again on the
    local buffer because:
      1. we can reduce the amount of inter-core synchronizations,
         which is costly.
      2. we can make the tiling size bigger.
  }];
  let constructor = "mlir::hivm::createTileCubeVectorLoopPass()";
  let options = [
    Option<
      "tiledMixVectorLoopNumber", "tile-mix-vector-loop", "unsigned",
      /*default=*/"1",
      "The trip count of the tiled vector loop for mix kernels">,
    Option<
      "tiledMixCubeLoopNumber", "tile-mix-cube-loop", "unsigned",
      /*default=*/"1",
      "The trip count of the tiled cube loop for mix kernels">,
  ];
  let dependentDialects = ["transform::TransformDialect"];
}

def ConvertNonContiguousReshapeToCopy : Pass<"convert-non-contiguous-reshape-to-copy"> {
  let summary = "Generate copy for reassociative reshape that might be non-contiguous";
  let constructor = "mlir::hivm::createNonContiguousReshapeToCopyPass()";
  let dependentDialects = ["memref::MemRefDialect", "hivm::HIVMDialect"];
}

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_TD
