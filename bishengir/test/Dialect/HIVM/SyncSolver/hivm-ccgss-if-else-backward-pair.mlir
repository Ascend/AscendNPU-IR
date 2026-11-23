// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-cross-core-gss{solver-version=v1 always-use-pipe-s=true use-different-multibuffer-flag-ids=true force-is-mem-based=true}))" -split-input-file -verify-diagnostics %s | FileCheck %s
// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-cross-core-gss{solver-version=v2 always-use-pipe-s=true use-different-multibuffer-flag-ids=true force-is-mem-based=true}))" -split-input-file -verify-diagnostics %s | FileCheck %s

// Cross-core else-scope backward pair: the cube producer is in the else
// branch, so the set/wait pair is placed at the beginning of the vector
// wait scope rather than after the producer.
module {
  // CHECK-LABEL: func.func @sync_solver_test_cross_core_if_else_backward_pair
  func.func @sync_solver_test_cross_core_if_else_backward_pair(%arg0: index, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>, %arg3: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.always_inline, hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.set_ffts_base_addr %arg3
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg4 = %c0 to %arg0 step %c1 {
      scf.if %false {
        %alloc_ub = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
        // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>]
        // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>]
        // CHECK: hivm.hir.load
        hivm.hir.load ins(%arg2 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc_ub : memref<256xf32, #hivm.address_space<ub>>)
      } else {
        %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
        hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc : memref<16xf32, #hivm.address_space<cbuf>>)
        %0 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
        hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
        %1 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
        hivm.hir.mmadL1 ins(%alloc, %0, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%1 : memref<256xf32, #hivm.address_space<cc>>)
        // CHECK: hivm.hir.fixpipe
        // CHECK-NOT: hivm.hir.sync_block_set
        hivm.hir.fixpipe {enable_nz2nd} ins(%1 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
      }
    }
    return
  }
}

// -----
// Cross-core true-scope backward pair: the cube producer is in the true
// branch, so the set/wait pair stays at the end of that scope.
module {
  // CHECK-LABEL: func.func @sync_solver_test_cross_core_if_true_backward_pair
  func.func @sync_solver_test_cross_core_if_true_backward_pair(%arg0: index, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>, %arg3: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}) attributes {hacc.always_inline, hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.set_ffts_base_addr %arg3
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg4 = %c0 to %arg0 step %c1 {
      scf.if %true {
        %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
        hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc : memref<16xf32, #hivm.address_space<cbuf>>)
        %0 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
        hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
        %1 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
        hivm.hir.mmadL1 ins(%alloc, %0, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%1 : memref<256xf32, #hivm.address_space<cc>>)
        hivm.hir.fixpipe {enable_nz2nd} ins(%1 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
        // CHECK: hivm.hir.fixpipe
        // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>]
        // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>]
      } else {
        %alloc_ub = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
        // CHECK: hivm.hir.load
        // CHECK-NOT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>]
        hivm.hir.load ins(%arg2 : memref<256xf32, #hivm.address_space<gm>>) outs(%alloc_ub : memref<256xf32, #hivm.address_space<ub>>)
      }
    }
    return
  }
}
