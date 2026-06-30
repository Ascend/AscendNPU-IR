// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

// -----

// Extra buffers types/sizes count mismatch.
func.func @custom_extra_buffer_mismatch(%arg0: memref<1xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<1xf32>
  // expected-error @+1 {{'hivm.hir.custom' op Extra buffers' types and sizes mismatch}}
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_mismatch",
       extra_buffers_types = [f32, f16],
       extra_buffers_sizes = [256 : i64]}
      "user.mismatch"
      ins(%arg0 : memref<1xf32>)
      outs(%empty : tensor<1xf32>) -> tensor<1xf32>
  return
}

// -----

// Extra buffers types present but sizes missing.
func.func @custom_extra_buffer_types_only(%arg0: memref<1xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<1xf32>
  // expected-error @+1 {{'hivm.hir.custom' op Either extra buffers' types or sizes missing}}
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_types_only",
       extra_buffers_types = [f32]}
      "user.types_only"
      ins(%arg0 : memref<1xf32>)
      outs(%empty : tensor<1xf32>) -> tensor<1xf32>
  return
}

// -----

// Extra buffers sizes present but types missing.
func.func @custom_extra_buffer_sizes_only(%arg0: memref<1xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<1xf32>
  // expected-error @+1 {{'hivm.hir.custom' op Either extra buffers' types or sizes missing}}
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_sizes_only",
       extra_buffers_sizes = [256 : i64]}
      "user.sizes_only"
      ins(%arg0 : memref<1xf32>)
      outs(%empty : tensor<1xf32>) -> tensor<1xf32>
  return
}
