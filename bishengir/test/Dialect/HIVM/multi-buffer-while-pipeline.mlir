// RUN: bishengir-opt %s                                                       \
// RUN:   -pass-pipeline="builtin.module(                                      \
// RUN:     hacc-append-device-spec{target=Ascend910B1},                       \
// RUN:     func.func(hivm-mark-multi-buffer{enable-auto=true}),               \
// RUN:     func.func(hivm-plan-memory,hivm-graph-sync-solver,                  \
// RUN:               hivm-enable-multi-buffer,hivm-lower-multi-buffer-counter))" \
// RUN:   -split-input-file -verify-diagnostics                                \
// RUN:   | FileCheck %s

// -----
// 4-pass end-to-end pipeline on a vadd-style scf.while body.
//
// This locks in the contract that all four multi-buffer passes
// (mark, plan-memory, graph-sync-solver, enable-multi-buffer) compose
// transparently for scf.while using the alloca-based counter scheme.
//
// Inspection items (all CHECK below):
//   1. funcOp top: a memref<1xi64> counter alloca, with an initial store of 0.
//   2. The scf.while op result type list (i1) -> i1 stays unchanged.
//   3. Body head: memref.load + arith.remui + arith.select to pick the slot.
//   4. set_flag/wait_flag pairs use a dynamic event id (selected from the
//      counter) - this is what enables N-way buffer rotation.
//   5. Body tail: arith.addi + memref.store back to the alloca.

// CHECK-LABEL: func.func @while_pipeline_vadd(
func.func @while_pipeline_vadd(%arg0: memref<8xf32, #hivm.address_space<gm>>,
                               %arg1: memref<8xf32, #hivm.address_space<gm>>) {
  %true = arith.constant true
  // Counter alloca + init at funcOp top (item 1).
  // CHECK-DAG: %[[CTR:.*]] = memref.alloca() : memref<1xi64>
  // CHECK-DAG: memref.store %{{.*}}, %[[CTR]]

  // Pre-loop set_flag pre-roll for double buffering (one per slot).
  // CHECK-DAG: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  // CHECK-DAG: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]

  // While signature unchanged (item 2).
  // CHECK: scf.while {{.*}} : (i1) -> i1
  %r = scf.while (%cond = %true) : (i1) -> i1 {
    scf.condition(%cond) %cond : i1
  } do {
  // CHECK: ^bb0
  ^bb0(%cin: i1):
    // Body head load + remui (item 3).
    // CHECK: %[[CUR:.*]] = memref.load %[[CTR]]
    // CHECK: arith.remui %[[CUR]], %{{.*}} : i64
    // CHECK: arith.select {{.*}} : i64
    %tmp = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // wait_flag uses dynamic event id (item 4).
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %{{.*}}]
    hivm.hir.load ins(%arg0 : memref<8xf32, #hivm.address_space<gm>>)
                  outs(%tmp : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%tmp, %tmp : memref<8xf32, #hivm.address_space<ub>>,
                                   memref<8xf32, #hivm.address_space<ub>>)
                  outs(%tmp : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%tmp : memref<8xf32, #hivm.address_space<ub>>)
                   outs(%arg1 : memref<8xf32, #hivm.address_space<gm>>)
    // Body tail increment + store-back (item 5).
    // CHECK: arith.addi %[[CUR]], %{{.*}} : i64
    // CHECK: memref.store %{{.*}}, %[[CTR]]
    // CHECK: scf.yield
    scf.yield %cin : i1
  }
  return
}
