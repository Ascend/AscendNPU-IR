// RUN: bishengir-opt -hivm-graph-sync-solver %s | FileCheck %s
//
// set slot: macro sets internally; GSS injects wait_flag after the macro.

module {
  func.func @test_custom_macro_set_slot(
      %arg0: memref<16xf16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
    hivm.hir.custom_macro
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
         hivm.pipe_out = #hivm.pipe<PIPE_V>,
         symbol = "k_custom_macro_set",
         sync_event_slots = [
           #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, set>
         ]}
        "user.macro_set"
        ins(%0 : memref<16xf16, #hivm.address_space<ub>>)
        outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
    // CHECK: sync_related_args(%c0_i64 : i64)
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
