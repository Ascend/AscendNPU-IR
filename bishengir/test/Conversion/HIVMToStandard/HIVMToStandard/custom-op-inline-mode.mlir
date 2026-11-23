// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivm-to-std="mark-libcall-noinline=true" -split-input-file | FileCheck %s

module {
  func.func @custom_always_inline(%src: memref<16xf32>, %dst: memref<16xf32>) {
    hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_V>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         hivm.inline_mode = #hivm.inline_mode<always_inline>,
         symbol = "custom_always_inline_impl"}
        "user.custom_always_inline"
        ins(%src : memref<16xf32>)
        outs(%dst : memref<16xf32>)
    return
  }
}

// CHECK-LABEL: func.func private @custom_always_inline_impl
// CHECK-SAME:  attributes {{{.*}}hacc.always_inline{{.*}}}
// CHECK-LABEL: func.func @custom_always_inline
// CHECK:       call @custom_always_inline_impl

// -----

module {
  func.func @custom_default_no_inline(%src: memref<16xf32>, %dst: memref<16xf32>) {
    hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_V>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         symbol = "custom_default_no_inline_impl"}
        "user.custom_default_no_inline"
        ins(%src : memref<16xf32>)
        outs(%dst : memref<16xf32>)
    return
  }
}

// CHECK-LABEL: func.func private @custom_default_no_inline_impl
// CHECK-SAME:  attributes {{{.*}}hacc.noinline{{.*}}}
// CHECK-LABEL: func.func @custom_default_no_inline
// CHECK:       call @custom_default_no_inline_impl

// -----

module {
  func.func @custom_macro_no_inline(%src: memref<16xf32>,
                                    %dst: memref<16xf32>) {
    hivm.hir.custom_macro
        {hivm.tcore_type = #hivm.tcore_type<CUBE>,
         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
         hivm.pipe_out = #hivm.pipe<PIPE_M>,
         hivm.inline_mode = #hivm.inline_mode<no_inline>,
         symbol = "custom_macro_no_inline_impl"}
        "user.custom_macro_no_inline"
        ins(%src : memref<16xf32>)
        outs(%dst : memref<16xf32>)
    return
  }
}

// CHECK-LABEL: func.func private @custom_macro_no_inline_impl
// CHECK-SAME:  attributes {{{.*}}hacc.noinline{{.*}}}
// CHECK-LABEL: func.func @custom_macro_no_inline
// CHECK:       call @custom_macro_no_inline_impl

// -----

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @custom_mix_aiv_preserves_symbol(%src: tensor<16xf16>,
                                              %dst: tensor<16xf16>)
      -> tensor<16xf16>
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %result = hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_V>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         no_side_effect,
         symbol = "custom_scale_add_f16"}
        "user.custom_scale_add_f16"
        ins(%src : tensor<16xf16>)
        outs(%dst : tensor<16xf16>) -> tensor<16xf16>
    return %result : tensor<16xf16>
  }
}

// CHECK-NOT:   @_mlir_ciface_custom_scale_add_f16.vector
// CHECK-LABEL: func.func private @_mlir_ciface_custom_scale_add_f16
// CHECK-LABEL: func.func @custom_mix_aiv_preserves_symbol
// CHECK:       call @_mlir_ciface_custom_scale_add_f16

// -----

// A pointer-returning helper has a memref result and no `outs`. The last
// operand is an i32 rank, not a destination buffer; lowering must keep the
// call result rather than forwarding that i32 onto memref users.
module {
  func.func @custom_ptr_result(%src: memref<?xbf16>, %rank: i32) -> memref<?xbf16> {
    %0 = hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_S>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         hivm.is_distributed,
         symbol = "aclshmem_ptr_bfloat16"}
        "dist.aclshmem_ptr_bfloat16"
        ins(%src, %rank : memref<?xbf16>, i32) -> memref<?xbf16>
    return %0 : memref<?xbf16>
  }
}

// CHECK-LABEL: func.func @custom_ptr_result
// CHECK:       %[[PTR:.*]] = call @aclshmem_ptr_bfloat16
// CHECK-SAME:    -> memref<?xbf16>
// CHECK:       return %[[PTR]]

// -----

// Destination-passing custom ops with a leftover memref result lower like
// any other returning call: the result stays on the call (sret), and is not
// forwarded onto `outs`.
module {
  func.func @custom_dps_result(%src: memref<16xf32>, %dst: memref<16xf32>) -> memref<16xf32> {
    %0 = hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_V>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         symbol = "dps_write_impl"}
        "user.dps_write"
        ins(%src : memref<16xf32>)
        outs(%dst : memref<16xf32>) -> memref<16xf32>
    return %0 : memref<16xf32>
  }
}

// CHECK-LABEL: func.func @custom_dps_result
// CHECK:       %[[PTR:.*]] = call @dps_write_impl
// CHECK-SAME:    -> memref<16xf32>
// CHECK:       return %[[PTR]]

// -----

// The op is ins-then-outs, and lowers generically: the library call keeps that
// operand order, so the inputs come first and the L1 tile (`outs`) comes last,
// matching the template's parameter list.
module {
  // CHECK-LABEL: func.func @sparse_gather_dst_last
  func.func @sparse_gather_dst_last(%base: memref<1xf32>,
                                    %index: memref<1xi32>,
                                    %dst: memref<2x1x16x16xf32, #hivm.address_space<cbuf>>) {
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    // CHECK: %[[CAST_BASE:.*]] = memref.cast %{{.*}} : memref<1xf32> to memref<?xf32
    // CHECK: %[[CAST_INDEX:.*]] = memref.cast %{{.*}} : memref<1xi32> to memref<?xi32
    // CHECK: %[[CAST_DST:.*]] = memref.cast %{{.*}} : memref<2x1x16x16xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32
    // CHECK: call @sparse_gather_load_to_L1buffer_float_int32_t(%[[CAST_BASE]], %[[CAST_INDEX]],
    // CHECK-SAME: %[[CAST_DST]]) :
    // CHECK-SAME: -> ()
    hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<CUBE>,
         hivm.pipe = #hivm.pipe<PIPE_MTE2>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         symbol = "sparse_gather_load_to_L1buffer_float_int32_t"}
        "__builtin_sparse_gather_load_to_L1buffer"
        ins(%base, %index, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i64,
            %c1_i32, %c1_i32, %c0_i32
            : memref<1xf32>, memref<1xi32>, i32, i32, i32, i32, i64,
              i32, i32, i32)
        outs(%dst : memref<2x1x16x16xf32, #hivm.address_space<cbuf>>)
    return
  }
}
