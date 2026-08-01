// RUN: bishengir-opt %s -split-input-file | bishengir-opt -split-input-file | FileCheck %s

// -----

// Iterator types.

func.func @custom_iterator_types(%arg0: memref<1xf32>, %arg1: tensor<1xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<1xf32>
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_iter",
       iterator_types = [#hivm.iterator_type<parallel>]}
      "user.iter"
      ins(%arg0, %arg1 : memref<1xf32>, tensor<1xf32>)
      outs(%empty : tensor<1xf32>) -> tensor<1xf32>
  return
}

// CHECK-LABEL: func.func @custom_iterator_types
// CHECK:       iterator_types = [#hivm.iterator_type<parallel>]
// CHECK:       symbol = "k_iter"

// -----

// Named affine maps in indexing_map.

#map2d = affine_map<(d0, d1) -> (d0, d1)>
func.func @custom_indexing_map_named_maps(%arg0: memref<2x2xf32>, %arg1: tensor<2x2xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<2x2xf32>
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_named_maps",
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
       indexing_map = [#map2d, #map2d, #map2d]}
      "user.named_maps"
      ins(%arg0, %arg1 : memref<2x2xf32>, tensor<2x2xf32>)
      outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  return
}

// CHECK-LABEL: func.func @custom_indexing_map_named_maps
// CHECK:       indexing_map
// CHECK:       symbol = "k_named_maps"

// -----

// Extra buffer type/size attributes (used by hivm-alloc-extra-buffer for CustomOp).

func.func @custom_extra_buffer_attrs_single(%arg0: memref<1xf32>, %arg1: tensor<1xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<1xf32>
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_extra_buf_one",
       extra_buffers_types = [f32],
       extra_buffers_sizes = [256 : i64]}
      "user.extra_buf_one"
      ins(%arg0, %arg1 : memref<1xf32>, tensor<1xf32>)
      outs(%empty : tensor<1xf32>) -> tensor<1xf32>
  return
}

// CHECK-LABEL: func.func @custom_extra_buffer_attrs_single
// CHECK-DAG:   extra_buffers_types = [f32]
// CHECK-DAG:   extra_buffers_sizes = [256]

// -----

func.func @custom_extra_buffer_attrs_multi(%arg0: memref<2x2xf32>, %arg1: tensor<2x2xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<2x2xf32>
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_extra_buf_multi",
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
       extra_buffers_types = [f32, f16],
       extra_buffers_sizes = [512 : i64, 128 : i64]}
      "user.extra_buf_multi"
      ins(%arg0, %arg1 : memref<2x2xf32>, tensor<2x2xf32>)
      outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  return
}

// CHECK-LABEL: func.func @custom_extra_buffer_attrs_multi
// CHECK-DAG:   extra_buffers_types = [f32, f16]
// CHECK-DAG:   extra_buffers_sizes = [512, 128]

// -----

// Inline mode.

func.func @custom_inline_mode(%arg0: memref<1xf32>, %arg1: tensor<1xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<1xf32>
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       hivm.inline_mode = #hivm.inline_mode<always_inline>,
       symbol = "k_inline_always"}
      "user.inline_always"
      ins(%arg0, %arg1 : memref<1xf32>, tensor<1xf32>)
      outs(%empty : tensor<1xf32>) -> tensor<1xf32>
  %1 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       hivm.inline_mode = #hivm.inline_mode<no_inline>,
       symbol = "k_inline_no"}
      "user.inline_no"
      ins(%arg0, %arg1 : memref<1xf32>, tensor<1xf32>)
      outs(%empty : tensor<1xf32>) -> tensor<1xf32>
  return
}

// CHECK-LABEL: func.func @custom_inline_mode
// CHECK:       hivm.inline_mode = #hivm.inline_mode<always_inline>
// CHECK:       hivm.inline_mode = #hivm.inline_mode<no_inline>

// -----

func.func @custom_macro_inline_mode(%arg0: memref<1xf32>, %arg1: tensor<1xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<1xf32>
  %0 = hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       hivm.inline_mode = #hivm.inline_mode<no_inline>,
       symbol = "k_inline_never"}
      "user.inline_never"
      ins(%arg0, %arg1 : memref<1xf32>, tensor<1xf32>)
      outs(%empty : tensor<1xf32>) -> tensor<1xf32>
  return
}

// CHECK-LABEL: func.func @custom_macro_inline_mode
// CHECK:       hivm.inline_mode = #hivm.inline_mode<no_inline>
