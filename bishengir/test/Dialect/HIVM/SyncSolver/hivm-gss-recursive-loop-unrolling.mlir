// Unrolling a nested loop recursively, enable-non-recursive-loop-unrolling=
// false, splits every copy of the inner loop instead of linearizing it in the
// second copy of the outer one. The option defaults to true, so this is the
// only coverage that path has.

// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v1 enable-non-recursive-loop-unrolling=false" %s | FileCheck %s
// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v2 enable-non-recursive-loop-unrolling=false" %s | FileCheck %s

// Either way of unrolling must end up with the same sync.
// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v1" %s | FileCheck %s
// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v2" %s | FileCheck %s

module {
  // The store writes the buffer the load of the next iteration fills, a
  // backward MTE3 -> MTE2 dependency carried by both loops.
  // CHECK-LABEL: @nested_cross_pipe_backward
  func.func @nested_cross_pipe_backward(
      %src: memref<16xf16, #hivm.address_space<gm>>,
      %dst: memref<16xf16, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %ub = hivm.hir.pointer_cast(%c0_i64)
      : memref<16xf16, #hivm.address_space<ub>>

    // The first iteration has nothing to wait for.
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    // CHECK-NEXT: scf.for
    // CHECK-NEXT: scf.for
    scf.for %outer = %c0 to %c2 step %c1 {
      scf.for %inner = %c0 to %c2 step %c1 {
        // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        // CHECK-NEXT: hivm.hir.load
        hivm.hir.load
          ins(%src : memref<16xf16, #hivm.address_space<gm>>)
          outs(%ub : memref<16xf16, #hivm.address_space<ub>>)
        // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        // CHECK-NEXT: hivm.hir.store
        hivm.hir.store
          ins(%ub : memref<16xf16, #hivm.address_space<ub>>)
          outs(%dst : memref<16xf16, #hivm.address_space<gm>>)
        // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      }
    }
    // The last iteration posted a flag nobody waits for inside the loop.
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    // CHECK-NEXT: return
    return
  }
}
