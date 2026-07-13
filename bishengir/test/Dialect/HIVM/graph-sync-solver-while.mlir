// RUN: bishengir-opt -hivm-graph-sync-solver -hivm-lower-multi-buffer-counter -split-input-file %s | FileCheck %s

// -----
// GraphSyncSolver should drive its event-id rotation off the alloca-based
// counter (created by MultiBufferLoopAdapter::ensureCounterMaterialized) when
// the multi-buffer loop is an scf.while. The scf.while signature must NOT be
// extended; the counter lives in a memref<1xi64> alloca at the funcOp top.

module {
// CHECK-LABEL: func.func @while_gss_basic(
  func.func @while_gss_basic(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                             %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c32_i64 = arith.constant 32 : i64
    %c0_i64 = arith.constant 0 : i64
    %true = arith.constant true

    // Funcop-top alloca for the counter, init to 0.
    // CHECK-DAG: %[[CTR:.*]] = memref.alloca() : memref<1xi64>
    // CHECK-DAG: memref.store %{{.*}}, %[[CTR]]

    // Set/wait flag for unmodified MTE3->MTE2 pre-roll plus first-iter set.
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]

    // The scf.while signature must remain (i1) -> i1.
    // CHECK: scf.while {{.*}} : (i1) -> i1
    %r = scf.while (%cond = %true) : (i1) -> i1 {
      scf.condition(%cond) %cond : i1
    } do {
    // CHECK: ^bb0
    ^bb0(%cin: i1):
      // Counter load + remui at body head.
      // CHECK: %[[CUR:.*]] = memref.load %[[CTR]]
      // CHECK: arith.remui %[[CUR]], %{{.*}} : i64
      %3 = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %3 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // wait_flag should use a select-driven event id (not a constant).
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %{{.*}}]
      hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%3 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%3 : memref<16xf16, #hivm.address_space<ub>>)
                     outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      // Counter increment-store before scf.yield.
      // CHECK: arith.addi %[[CUR]], %{{.*}} : i64
      // CHECK: memref.store %{{.*}}, %[[CTR]]
      // CHECK: scf.yield
      scf.yield %cin : i1
    }
    return
  }
}
