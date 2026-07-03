// RUN: bishengir-opt %s --convert-hivm-to-std=mark-libcall-noinline -split-input-file | FileCheck %s

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
