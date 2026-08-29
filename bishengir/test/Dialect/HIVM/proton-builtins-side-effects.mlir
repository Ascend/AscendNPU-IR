// RUN: bishengir-opt %s --cse --split-input-file | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @counter_reads_are_not_cse() -> (i64, i64) {
    %clock0 = hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_S>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>}
        "__builtin_proton_get_sys_cnt" -> i64
    %clock1 = hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_S>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>}
        "__builtin_proton_get_sys_cnt" -> i64
    return %clock0, %clock1 : i64, i64
  }
}

// CHECK-LABEL: func.func @counter_reads_are_not_cse
// CHECK-COUNT-2: "__builtin_proton_get_sys_cnt"

// -----

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @stores_are_not_cse(
      %buffer: memref<?xi32, #hivm.address_space<gm>>, %section_offset: index,
      %clock: i64) {
    %scope_id = arith.constant 7 : i32
    %is_start = arith.constant 1 : i32
    %data_segment_words = arith.constant 16 : i32
    hivm.hir.custom
        {gm_addr_args_indices = array<i32: 0>,
         hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_S>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>}
        "__builtin_proton_circular_store"
        ins(%buffer, %section_offset, %clock, %scope_id, %is_start,
            %data_segment_words : memref<?xi32, #hivm.address_space<gm>>,
            index, i64, i32, i32, i32)
    hivm.hir.custom
        {gm_addr_args_indices = array<i32: 0>,
         hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_S>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>}
        "__builtin_proton_circular_store"
        ins(%buffer, %section_offset, %clock, %scope_id, %is_start,
            %data_segment_words : memref<?xi32, #hivm.address_space<gm>>,
            index, i64, i32, i32, i32)
    return
  }
}

// CHECK-LABEL: func.func @stores_are_not_cse
// CHECK-COUNT-2: "__builtin_proton_circular_store"
