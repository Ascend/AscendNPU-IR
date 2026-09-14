// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend910_9589 -convert-hivm-to-std="mark-libcall-noinline=true" -split-input-file | FileCheck %s

module {
  func.func @lower_proton_get_sys_cnt() -> i64 {
    %clock = hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_S>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         hivm.inline_mode = #hivm.inline_mode<always_inline>}
        "__builtin_proton_get_sys_cnt" -> i64
    return %clock : i64
  }
}

// CHECK-LABEL: func.func private @proton_get_sys_cnt
// CHECK-SAME:  attributes {{{.*}}hacc.always_inline{{.*}}}
// CHECK-LABEL: func.func @lower_proton_get_sys_cnt
// CHECK:       %[[CLOCK:.*]] = call @proton_get_sys_cnt() : () -> i64
// CHECK:       return %[[CLOCK]] : i64

// -----

module {
  func.func @lower_proton_circular_store(
      %buffer: memref<?xi32, #hivm.address_space<gm>>, %section_offset: index,
      %clock: i64) {
    %scope_id = arith.constant 7 : i32
    %is_start = arith.constant 1 : i32
    %data_segment_words = arith.constant 16 : i32
    hivm.hir.custom
        {gm_addr_args_indices = array<i32: 0>,
         hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_S>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         hivm.inline_mode = #hivm.inline_mode<always_inline>}
        "__builtin_proton_circular_store"
        ins(%buffer, %section_offset, %clock, %scope_id, %is_start,
            %data_segment_words : memref<?xi32, #hivm.address_space<gm>>,
            index, i64, i32, i32, i32)
    return
  }
}

// CHECK-LABEL: func.func private @proton_circular_store
// CHECK-SAME:  attributes {{{.*}}hacc.always_inline{{.*}}}
// CHECK-LABEL: func.func @lower_proton_circular_store
// CHECK:       %[[BUFFER:.*]] = memref.cast %{{.*}} : memref<?xi32, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
// CHECK:       call @proton_circular_store(%[[BUFFER]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, index, i64, i32, i32, i32) -> ()
