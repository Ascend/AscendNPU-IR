//===- Passes.td - HIVM pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

def InferVFMode : Pass<"hivm-infer-vf-mode", "func::FuncOp"> {
  let summary = "Infer vf mode for Ops";
  let constructor = "mlir::hivm::createInferVFModePass()";
  let dependentDialects = ["hivm::HIVMDialect"];
}

def CloneSCFIfYieldOperand : Pass<"hivm-clone-scf-if-yield-operand", "func::FuncOp"> {
  let summary = "Clone scf.if.yield operand for PlanMemory.";
  let description = [{This pass clones scf.if yield operand if yield operands
    are same or it is used after this scf.if, avoid inplace by PlanMemory}];
  let constructor = "mlir::hivm::createCloneSCFIfYieldOperandPass()";
  let dependentDialects =
    ["hivm::HIVMDialect", "scf::SCFDialect", "tensor::TensorDialect"];
}

def PlanMemory : Pass<"hivm-plan-memory", "ModuleOp"> {
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
    Option<"enablePrintMemoryAllocatedSize", "enable-print-memory-allocated-size", "bool",
           /*default=*/"false",
           "Enable print memory allocated size, default : false">,
    Option<"restrictInplaceAsISA", "restrict-inplace-as-isa", "bool",
           /*default=*/"false",
           "restrict memory inplace as isa, default : false">,
    Option<"simtVFDynamicSize", "simt-vf-dynamic-size", "int",
           /*default=*/"216",
           "Dynamic ub size(KB) for simt VF, default : 216">,
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
            "Assume that all loops (forOp whileOp) will execute at least once.">
  ];
}

def GraphSyncSolver : Pass<"hivm-graph-sync-solver", "func::FuncOp"> {
  let summary = "graph sync solver ";
  let constructor = "mlir::hivm::createGraphSyncSolverPass()";
  let dependentDialects = ["affine::AffineDialect", "hivm::HIVMDialect"];
  let options = [Option<"enableUnitFlag", "enable-unit-flag", "bool",
                        /*default=*/"false",
                        "Enable unit-flag modes for synchronization">,
                 Option<"enableTesterMode", "enable-tester-mode", "bool",
                        /*default=*/"false", "Enable sync-tester mode">,
                 ListOption<"syncTesterOptions", "sync-tester-options",
                            "int64_t",
                            "Sync-tester options (num_runs, init_seed, "
                            "num_ops, num_ptrs, enable-multibuffer)">,
  ];
}

def CrossCoreGSS : Pass<"hivm-cross-core-gss", "func::FuncOp"> {
  let summary = "cross-core graph-sync-solver ";
  let constructor = "mlir::hivm::createCrossCoreGSSPass()";
  let dependentDialects = ["hacc::HACCDialect", "hivm::HIVMDialect",
                           "tensor::TensorDialect"];
  let options =
      [Option<"blockAllSync", "block-all-sync", "bool", /*default=*/"false",
              "Enable inject all block sync for HIVM injectBlockSync.">,
       Option<"disableAutoInjectBlockSync", "disable-auto-inject-block-sync",
              "bool", /*default=*/"false",
              "Toggle auto set/wait insertion, always keep SetFFTSBaseAddrOp">,
       Option<
           "alwaysUsePipeSAsWaitingPipe", "always-use-pipe-s", "bool",
           /*default=*/"false",
           "Always use scalar pipeline as waiting pipeline for wait_flag ops.">,
       Option<"useDifferentMultiBufferFlagIds",
              "use-different-multibuffer-flag-ids", "bool", /*default=*/"false",
              "Use different flag-ids for multibuffer backward sync pairs.">,
       Option<"forceIsMemBased", "force-is-mem-based", "bool",
              /*default=*/"false",
              "Force memory based ai-core architecture mode.">,
       Option<"forceIsRegBased", "force-is-reg-based", "bool",
              /*default=*/"false",
              "Force register based ai-core architecture mode.">,
  ];
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
                           "tensor::TensorDialect"];
}

def HIVMAggregatedDecomposeOp : Pass<"hivm-aggregated-decompose-op", "func::FuncOp"> {
  let summary = "decompose hivm ops that use hivm AggregatedOpInterface";
  let constructor = "mlir::hivm::createHIVMAggregatedDecomposeOpPass()";
  let dependentDialects = ["hivm::HIVMDialect", "memref::MemRefDialect",
                           "tensor::TensorDialect"];
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
  let dependentDialects = ["hivm::HIVMDialect"];
}

def OutlineAllocInVF : Pass<"hivm-outline-alloc-in-VF", "ModuleOp"> {
  let summary = "outline memref.alloc with static shape in VF";
  let constructor = "mlir::hivm::createOutlineAllocInVFPass()";
  let dependentDialects = ["hivm::HIVMDialect"];
}

def OutlineCopyInVF : Pass<"hivm-outline-copy-in-VF", "ModuleOp"> {
  let summary = "rewrite hivm.load to hivm.copy in VF when operands are VF args";
  let constructor = "mlir::hivm::createOutlineCopyInVFPass()";
  let dependentDialects = ["arith::ArithDialect", "hivm::HIVMDialect",
                           "memref::MemRefDialect", "vector::VectorDialect"];
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
  let dependentDialects = ["func::FuncDialect", "annotation::AnnotationDialect",
                           "hivm::HIVMDialect"];
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
  let summary = "Sink Ops To Consumer In Loop.";
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

def EnableStrideAlign : Pass<"hivm-enable-stride-align", "ModuleOp"> {
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

def InlineFixpipeV2 : Pass<"hivm-inline-fixpipe-v2"> {
  let summary = "Convert ops to HIVM Fixpipe op";
  let constructor = "mlir::hivm::createInlineFixpipeV2Pass()";
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

def InsertLoadStoreForScalar
    : Pass<"hivm-insert-load-store-for-scalar", "func::FuncOp"> {
  let summary = "insert load store op for scalar";
  let constructor = "mlir::hivm::createInsertLoadStoreForScalarPass()";
  let dependentDialects = ["hivm::HIVMDialect", "bishengir::memref_ext::MemRefExtDialect"];
}

def SplitMixedIfConditionals
    : Pass<"hivm-split-mixed-if-conditionals", "func::FuncOp"> {
  let summary = "split if conditionals for mix cv";
  let constructor = "mlir::hivm::createSplitMixedIfConditionalsPass()";
  let dependentDialects = ["arith::ArithDialect", "hivm::HIVMDialect",
                           "tensor::TensorDialect"];
}

def InsertLoadStoreForMixCV
    : Pass<"hivm-insert-load-store-for-mix-cv", "func::FuncOp"> {
  let summary = "insert load store op for mix cv";
  let constructor = "mlir::hivm::createInsertLoadStoreForMixCVPass()";
  let dependentDialects = ["hivm::HIVMDialect", "bishengir::memref_ext::MemRefExtDialect"];
}

def InsertCVTightCoupledBuffer
    : Pass<"hivm-insert-cv-tight-coupled-buffer", "func::FuncOp"> {
  let summary = "insert cv tight coupled buffer for mix cv";
  let constructor = "mlir::hivm::createInsertCVTightCoupledBufferPass()";
  let dependentDialects = ["hivm::HIVMDialect",
                           "bufferization::BufferizationDialect",
                           "annotation::AnnotationDialect"];
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

def InsertInferVFModeFunc : Pass<"hivm-insert-vf-mode-func", "func::FuncOp"> {
  let summary = "Insert infer-vf-mode callback func for host";
  let description = "Infer a function's VF mode {SIMD, SIMT, MIX}, "
                    "and then create host callback to return its VF mode";

  let constructor = "mlir::hivm::createInsertInferVFModeFuncPass()";
  let dependentDialects = ["hivm::HIVMDialect", ];
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

def AnnotateVFAlias : Pass<"hivm-annotate-vf-alias", "func::FuncOp"> {
  let summary = "annotate alias info within VF";
  let constructor = "mlir::hivm::createAnnotateVFAliasPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def RemoveCopyOps : Pass<"hivm-remove-copy-ops", "func::FuncOp"> {
  let summary = "remove redundant copy ops";
  let constructor = "mlir::hivm::createRemoveCopyOpsPass()";
  let dependentDialects = ["hivm::HIVMDialect"];
}

def AnalyzeArithVectorMask : Pass<"arith-vector-mask-analyze", "func::FuncOp"> {
  let summary = "Analyze arith/vector mask to eliminate select OP";
  let constructor = "mlir::hivm::createArithVectorMaskAnalysisPass()";
  let dependentDialects = ["arith::ArithDialect", "vector::VectorDialect",
                           "annotation::AnnotationDialect"];
}

def TileAndBindSubBlock : Pass<"hivm-bind-sub-block", "ModuleOp"> {
  let summary = "tile and bind sub block";
  let constructor = "mlir::hivm::createTileAndBindSubBlockPass()";
  let dependentDialects = [
    "annotation::AnnotationDialect", "func::FuncDialect", "hivm::HIVMDialect"
  ];
  let options = [Option<
      "strictMode", "strict-mode", "bool",
      /*default=*/"true",
      "Run on strict mode when bubble up extract slice">];
}

def HIVMBubbleUpExtractSlice : Pass<"hivm-bubble-up-extract-slice", "func::FuncOp"> {
  let summary = "tile and bind sub block";
  let constructor = "mlir::hivm::createHIVMBubbleUpExtractSlicePass()";
  let dependentDialects = [
    "annotation::AnnotationDialect", "func::FuncDialect", "hivm::HIVMDialect"
  ];
  let options = [Option<
      "strictMode", "strict-mode", "bool",
      /*default=*/"true",
      "Run on strict mode when bubble up extract slice">];
}

def HIVMVectorizeOps : Pass<"hivm-vectorize-ops", "func::FuncOp"> {
  let summary = "vectorize hivm ops";
  let constructor = "mlir::hivm::createHIVMVectorizeOpsPass()";
  let dependentDialects = [
    "func::FuncDialect", "hivm::HIVMDialect", "arith::ArithDialect", 
    "vector::VectorDialect"
  ];
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

def InsertL12UBForDebug : Pass<"hivm-insert-l12ub-for-debug"> {
  let summary = "insert l12ub for debug";
  let constructor = "mlir::hivm::createInsertL12UBForDebugPass()";
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
  let options = [Option<"enableAutoBalance", "enable-auto-balance", "bool",
                        /*default=*/"false", "DEPRICATED">,
                 Option<"pipelineDepth", "pipeline-depth", "int",
                        /*default=*/"-1",
                        "Specify pipeline depth for cv-pipelining.">];
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

def InferSimtVFMemEffect : Pass<"infer-simt-vf-memory-effect"> {
  let summary = "Infer simt vf func args memory effect";
  let constructor = "mlir::hivm::createInferSimtVFMemEffectPass()";
}

def InsertConvertLayout : Pass<"hivm-insert-convert-layout", "func::FuncOp"> {
  let summary = "Insert convert layout for matmul operations";
  let constructor = "mlir::hivm::createInsertConvertLayoutPass()";
  let dependentDialects = ["hivm::HIVMDialect", ];
}

def PropagateConvertLayout : Pass<"hivm-propagate-convert-layout", "func::FuncOp"> {
  let summary = "Propagate convert layout for matmul operations";
  let constructor = "mlir::hivm::createPropagateConvertLayoutPass()";
  let dependentDialects = ["hivm::HIVMDialect", ];
}

def ConvertLayoutToTranspose : Pass<"hivm-convert-layout-to-transpose", "func::FuncOp"> {
  let summary = "Convert layout to transpose decomposition";
  let constructor = "mlir::hivm::createConvertLayoutToTransposePass()";
  let dependentDialects = ["hivm::HIVMDialect", ];
}

def InsertCVDataMovement : Pass<"hivm-insert-cv-data-movement", "func::FuncOp"> {
  let summary = "Cube vector data movement";
  let constructor = "mlir::hivm::createInsertCVDataMovementPass()";
  let dependentDialects = ["hivm::HIVMDialect", ];
}

def CombineOptimizedConvertLayout : Pass<"hivm-combine-optimized-convert-layout", "func::FuncOp"> {
  let summary = "Fold some convert layout patterns and change it into optimized form";
  let constructor = "mlir::hivm::createCombineOptimizedConvertLayoutPass()";
  let dependentDialects = ["hivm::HIVMDialect", ];
}


def SplitSimtModule : Pass<"split-simt-module"> {
  let summary = "Split simt module for every simt vf";
  let constructor = "mlir::hivm::createSplitSimtModulePass()";
}

def InsertMemSemanticForSimtVF : Pass<"insert-memory-semantic-for-simtvf"> {
  let summary = "Insert memory semantic for simt vf";
  let constructor = "mlir::hivm::createInsertMemSemanticForSimtVFPass()";
}

def AutoScope : Pass<"auto-scope"> {
  let summary = "Create scope for gather_load and scatter_store";
  let constructor = "mlir::hivm::createAutoScopePass()";
  let dependentDialects = ["hivm::HIVMDialect", "scope::ScopeDialect"];
}

def InsertAllocBasePlaceholder : Pass<"insert-alloc-base-placeholder", "ModuleOp"> {
  let summary = "Insert placeholder memref.alloc with a dummy size";
  let constructor = "mlir::hivm::createInsertAllocBasePlaceholderPass()";
}

def WriteBackShared : Pass<"write-back-shared", "ModuleOp"> {
  let summary = "Set the actual size of memref.alloc which represent used shared memory in simt vf";
  let constructor = "mlir::hivm::createWriteBackSharedPass()";
}
#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_PASSES_TD
