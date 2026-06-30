// RUN: bishengir-opt -hivm-graph-sync-solver %s | FileCheck %s
//
// Multiple sync_event_slots on the same pipe pair: two wait + two internal.

module {
  func.func @test_custom_macro_multi_slots(
      %arg0: memref<16xf16, #hivm.address_space<gm>>,
      %arg1: memref<16xf16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c32_i64) : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.load ins(%arg1 : memref<16xf16, #hivm.address_space<gm>>) outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE2>]
    hivm.hir.custom_macro
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
         hivm.pipe_out = #hivm.pipe<PIPE_V>,
         symbol = "k_custom_macro_multi",
         sync_event_slots = [
           #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, wait>,
           #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, wait>,
           #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, internal>,
           #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, internal>
         ]}
        "user.macro_multi"
        ins(%0, %1 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
        outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
    // CHECK-NOT: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK-DAG: %c0_i64_2 = arith.constant 0 : i64
    // CHECK-DAG: %c1_i64 = arith.constant 1 : i64
    // CHECK: sync_related_args(%c0_i64_0, %c0_i64, %c0_i64_2, %c1_i64 : i64, i64, i64, i64)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
