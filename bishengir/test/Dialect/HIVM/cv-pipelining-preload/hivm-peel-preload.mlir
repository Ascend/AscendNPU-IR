// RUN: bishengir-opt %s -create-preload -hivm-graph-sync-solver -hivm-peel-preload | FileCheck %s

// GraphSync inserts pipe flags on the structured expanded loop. Peel after
// GraphSync clones those paired flags with prologue/epilogue and drops
// preload_if in the middle loop.
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>} {
  // CHECK-LABEL: func.func @peel_after_graph_sync(
  func.func @peel_after_graph_sync(
      %src: memref<16x16x16xf16, #hivm.address_space<gm>>,
      %dst: memref<16x16x16xf16, #hivm.address_space<gm>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c1 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %buf = hivm.hir.pointer_cast(%c0_i64) : memref<16x16x16xf16, #hivm.address_space<ub>>
    scf.for %i = %c0 to %c8 step %c1 : i32 {
      scope.scope : () -> () {
        hivm.hir.load ins(%src : memref<16x16x16xf16, #hivm.address_space<gm>>)
          outs(%buf : memref<16x16x16xf16, #hivm.address_space<ub>>)
        hivm.hir.store ins(%buf : memref<16x16x16xf16, #hivm.address_space<ub>>)
          outs(%dst : memref<16x16x16xf16, #hivm.address_space<gm>>)
        scope.return
      } {no_inline, hivm.preload_num = 1 : i32, hivm.max_preload_num = 2 : i32}
      scope.scope : () -> () {
        hivm.hir.load ins(%src : memref<16x16x16xf16, #hivm.address_space<gm>>)
          outs(%buf : memref<16x16x16xf16, #hivm.address_space<ub>>)
        hivm.hir.store ins(%buf : memref<16x16x16xf16, #hivm.address_space<ub>>)
          outs(%dst : memref<16x16x16xf16, #hivm.address_space<gm>>)
        scope.return
      } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}
    }
    return
  }
}

// Prologue is inlined before the middle loop.
// CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
// CHECK: hivm.hir.load
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
// CHECK: hivm.hir.store

// Middle loop after peeling max_preload_num from each end; guards are gone.
// CHECK: scf.for {{.*}} = %c2{{.*}} to %c8{{.*}} step {{.*}}
// CHECK-NOT: scf.if
// CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
// CHECK: hivm.hir.load
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
// CHECK: hivm.hir.store
// CHECK: hivm.hir.load
// CHECK: hivm.hir.store
// CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
// CHECK: }
