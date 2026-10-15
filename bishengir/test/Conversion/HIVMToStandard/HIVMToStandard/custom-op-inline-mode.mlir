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
