// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v1" -split-input-file %s | FileCheck %s
// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v2" -split-input-file %s | FileCheck %s
// clang-format off

// A multi-buffer handshake that lives inside a conditional block rotates on that
// block's own counter: the block may run fewer times than the loop, so a
// loop-iteration index reproduces neither its rotation nor its parity. Both ends
// of the handshake share the one counter of the block.

// -----
module {
  // CHECK-LABEL: func.func @nested_pair_owns_block_counter(
  func.func @nested_pair_owns_block_counter(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>, %cond: i1) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    // CHECK: scf.for
    scf.for %iv = %c0 to %c8 step %c1 {
      %buf = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %buf {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // The loop body owns no counter for this pair.
      // CHECK-NOT: hivm.hir.multi_buffer_counter
      // CHECK: scf.if
      scf.if %cond {
        // CHECK-NEXT: %[[CTR:.*]] = hivm.hir.multi_buffer_counter -> i64
        // CHECK: %[[MOD:.*]] = arith.remui %[[CTR]], %{{.*}} : i64
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %{{.*}}]
        hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                      outs(%buf : memref<16xf16, #hivm.address_space<ub>>)
        hivm.hir.store ins(%buf : memref<16xf16, #hivm.address_space<ub>>)
                       outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %{{.*}}]
      }
    }
    return
  }
}

// -----
module {
  // CHECK-LABEL: func.func @outer_pair_keeps_loop_counter(
  func.func @outer_pair_keeps_loop_counter(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>, %cond: i1) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    // CHECK: scf.for
    // The buffer is filled unconditionally and consumed under a guard, so the
    // pair keeps the loop parity: its counter stays at the loop body head.
    // CHECK-NEXT: %[[CTR:.*]] = hivm.hir.multi_buffer_counter -> i64
    scf.for %iv = %c0 to %c8 step %c1 {
      %buf = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %buf {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%buf : memref<16xf16, #hivm.address_space<ub>>)
      scf.if %cond {
        hivm.hir.store ins(%buf : memref<16xf16, #hivm.address_space<ub>>)
                       outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      }
      // CHECK: scf.if
      // CHECK-NOT: hivm.hir.multi_buffer_counter
      // CHECK: return
    }
    return
  }
}
// clang-format on
