// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v1" %s | FileCheck %s
// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v2" %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @sync_solver_sibling_if_backward
  func.func @sync_solver_sibling_if_backward(%n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %gm = hivm.hir.pointer_cast(%c0_i64) : memref<1xi1, #hivm.address_space<gm>>
    %ub = hivm.hir.pointer_cast(%c1_i64) : memref<1xi1, #hivm.address_space<ub>>

    // CHECK: scf.for
    scf.for %arg0 = %c0 to %n step %c1 {
      // CHECK: scf.if
      scf.if %true {
        // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        // CHECK-NEXT: hivm.hir.load
        hivm.hir.load ins(%gm : memref<1xi1, #hivm.address_space<gm>>) outs(%ub : memref<1xi1, #hivm.address_space<ub>>)
      }
      // CHECK: else
      // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]

      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      // CHECK-NEXT: scf.if
      scf.if %true {
        // CHECK-NEXT: hivm.hir.store
        // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.store ins(%ub : memref<1xi1, #hivm.address_space<ub>>) outs(%gm : memref<1xi1, #hivm.address_space<gm>>)
      }
      // CHECK: else
      // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    }
    return
  }
}
