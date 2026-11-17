// RUN: split-file %s %t.inputs
// RUN: bishengir-opt %t.inputs/staged.mlir --hacc-append-device-spec=target=Ascend950PR_9579 --hivm-mark-preload-buffers --hivm-mark-tightly-coupled-buffer --hivm-plan-memory "--hivm-cross-core-gss=solver-version=v2 enable-cv-patterns=false" -o %t
// RUN: FileCheck %s --check-prefix=MIX < %t
// RUN: bishengir-opt %t --hivm-v2-split-kernel | FileCheck %s --check-prefix=SPLIT --implicit-check-not=memref.alloc
// RUN: bishengir-opt %t.inputs/planned.mlir --hivm-cross-core-gss | FileCheck %s --check-prefix=REUSE

// Stages 0, 2 and 4 exchange data through L1 and UB. Two stages of overlap
// require three addresses for each transferred tile. The stage distance is
// even for both cross-core dependencies. The general solver places the
// handshakes at task boundaries so both sides execute them during fill/drain.
// MIX-LABEL: func.func @mix_gap(
// MIX: hivm.hir.pointer_cast({{%[^,]+}}, {{%[^,]+}}, {{%[^)]+}}) : memref<16x16xf16, #hivm.address_space<ub>>
// MIX: hivm.hir.pointer_cast({{%[^,]+}}, {{%[^,]+}}, {{%[^)]+}}) : memref<16x16xf16, #hivm.address_space<cbuf>>
// MIX: memref.load
// MIX: memref.store
// MIX: hivm.hir.copy
// MIX: hivm.preload_num = 4 : i32, hivm.task_name = "prepare"
// MIX-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = [[VC:[0-9]+]]
// MIX: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = [[VC]]
// MIX: hivm.hir.mmadL1
// MIX: hivm.hir.fixpipe
// MIX: hivm.preload_num = 2 : i32, hivm.task_name = "cube"
// MIX-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE3>] flag = [[CV:[0-9]+]]
// MIX: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE3>] flag = [[CV]]
// MIX: hivm.hir.store
// MIX: hivm.preload_num = 0 : i32, hivm.task_name = "store"

// SPLIT-LABEL: func.func @mix_gap_mix_aic(
// SPLIT-NOT: hivm.task_name = "prepare"
// SPLIT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>]
// SPLIT: hivm.hir.mmadL1
// SPLIT: hivm.preload_num = 2 : i32, hivm.task_name = "cube"
// SPLIT-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE3>]
// SPLIT-NOT: hivm.task_name = "store"
// SPLIT-LABEL: func.func @mix_gap_mix_aiv(
// SPLIT: hivm.hir.copy
// SPLIT: hivm.preload_num = 4 : i32, hivm.task_name = "prepare"
// SPLIT-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>]
// SPLIT-NOT: hivm.task_name = "cube"
// SPLIT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE3>]
// SPLIT: hivm.hir.store
// SPLIT: hivm.preload_num = 0 : i32, hivm.task_name = "store"

//--- staged.mlir
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @mix_gap(%gm: memref<?xf16, #hivm.address_space<gm>>, %T: index)
      attributes {
      hivm.func_core_type = #hivm.func_core_type<MIX>,
      hivm.core_ratio = #hivm.core_ratio<1, 1>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %true = arith.constant true
    scf.for %i = %c0 to %T step %c1 {
      %in = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      %l1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %acc = memref.alloc() : memref<16x16xf32, #hivm.address_space<cc>>
      %out = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      %offset = arith.muli %i, %c256 : index
      %view = memref.reinterpret_cast %gm to offset: [%offset],
          sizes: [16, 16], strides: [16, 1] : memref<?xf16, #hivm.address_space<gm>> to
          memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      scope.scope : () -> () {
        hivm.hir.load ins(%view : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>)
            outs(%in : memref<16x16xf16, #hivm.address_space<ub>>)
        %scalar = memref.load %view[%c0, %c0] {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
            : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        memref.store %scalar, %in[%c0, %c0] {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
            : memref<16x16xf16, #hivm.address_space<ub>>
        hivm.hir.copy ins(%in : memref<16x16xf16, #hivm.address_space<ub>>)
            outs(%l1 : memref<16x16xf16, #hivm.address_space<cbuf>>)
        scope.return
      } {hivm.max_preload_num = 5 : i32, hivm.preload_num = 4 : i32,
         hivm.task_name = "prepare", no_inline,
         hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        hivm.hir.mmadL1 ins(%l1, %l1, %true, %c16, %c16, %c16 :
            memref<16x16xf16, #hivm.address_space<cbuf>>,
            memref<16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index)
            outs(%acc : memref<16x16xf32, #hivm.address_space<cc>>)
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>,
            pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
            ins(%acc : memref<16x16xf32, #hivm.address_space<cc>>)
            outs(%out : memref<16x16xf16, #hivm.address_space<ub>>)
        scope.return
      } {hivm.max_preload_num = 5 : i32, hivm.preload_num = 2 : i32,
         hivm.task_name = "cube", no_inline,
         hivm.tcore_type = #hivm.tcore_type<CUBE>}
      scope.scope : () -> () {
        hivm.hir.store ins(%out : memref<16x16xf16, #hivm.address_space<ub>>)
            outs(%view : memref<16x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>)
        scope.return
      } {hivm.max_preload_num = 5 : i32, hivm.preload_num = 0 : i32,
         hivm.task_name = "store", no_inline,
         hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    } {hivm.cv_pipelined_loop}
    return
  }
}

// The two TCB values already have the same planned address. The second Cube
// write must wait for the first Vector read even though the SSA values differ.
// REUSE-LABEL: func.func @reused_tcb(
// REUSE: %[[A:[^ ]+]] = hivm.hir.pointer_cast(%[[ADDR:[^)]+]]) : memref<16x16xf16, #hivm.address_space<ub>>
// REUSE: hivm.hir.store ins(%[[A]]
// REUSE-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = [[REUSE:[0-9]+]]
// REUSE: %[[B:[^ ]+]] = hivm.hir.pointer_cast(%[[ADDR]]) : memref<16x16xf16, #hivm.address_space<ub>>
// REUSE: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_FIX>] flag = [[REUSE]]
// REUSE-NEXT: hivm.hir.fixpipe {{.*}} outs(%[[B]]
//--- planned.mlir
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @reused_tcb(%acc: memref<16x16xf32, #hivm.address_space<cc>>,
                        %dst0: memref<16x16xf16, #hivm.address_space<gm>>,
                        %dst1: memref<16x16xf16, #hivm.address_space<gm>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>,
                  hivm.core_ratio = #hivm.core_ratio<1, 1>} {
    %addr = arith.constant 0 : i64
    %a = hivm.hir.pointer_cast(%addr) : memref<16x16xf16, #hivm.address_space<ub>>
    annotation.mark %a {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
        : memref<16x16xf16, #hivm.address_space<ub>>
    hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
        ins(%acc : memref<16x16xf32, #hivm.address_space<cc>>)
        outs(%a : memref<16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%a : memref<16x16xf16, #hivm.address_space<ub>>)
        outs(%dst0 : memref<16x16xf16, #hivm.address_space<gm>>)
    %b = hivm.hir.pointer_cast(%addr) : memref<16x16xf16, #hivm.address_space<ub>>
    annotation.mark %b {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>}
        : memref<16x16xf16, #hivm.address_space<ub>>
    hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
        ins(%acc : memref<16x16xf32, #hivm.address_space<cc>>)
        outs(%b : memref<16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%b : memref<16x16xf16, #hivm.address_space<ub>>)
        outs(%dst1 : memref<16x16xf16, #hivm.address_space<gm>>)
    return
  }
}
