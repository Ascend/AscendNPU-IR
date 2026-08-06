// RUN: bishengir-opt -hivm-graph-sync-solver -split-input-file %s | FileCheck %s

// Double-buffered load/store in the inner loop of a nest: the multibuffer
// rotation loop is the inner loop (the direct parent of both the load and the
// store), so the two rotating event ids are hoisted around the nest and the
// inner backward sync rotates per iteration.
module {
  func.func @test_db_nested_loop(%arg0: memref<16xf16, #hivm.address_space<gm>>, %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c32_i64 = arith.constant 32 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg2 = %c0 to %c16 step %c4 {
      scf.for %arg3 = %c0 to %c16 step %c4 {
        %0 = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
        annotation.mark %0 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, [[MB:%[0-9]+]]]
        hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%0 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, [[MB]]]
      }
    }
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    return
  }
}

// -----

// A two-slot backward V-to-MTE2 dependency already orders the next load.
// Preserve its event count while checking a same-pipe conflict so that GSS
// does not add a redundant MTE2 barrier.
module {
  // CHECK-LABEL: func.func @test_db_event_info_avoids_redundant_barrier
  func.func @test_db_event_info_avoids_redundant_barrier(
      %arg0: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg1 = %c0 to %c4 step %c1 {
      %0 = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %0 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
      // CHECK: hivm.hir.load
      hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_V>]
      // CHECK: hivm.hir.vadd
      hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
    }
    // CHECK: return
    return
  }
}
