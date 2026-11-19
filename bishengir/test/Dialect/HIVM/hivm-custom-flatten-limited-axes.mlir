// RUN: bishengir-opt %s -test-flatten-limited-axes -split-input-file -o /dev/null | FileCheck %s

// -----

// CHECK: @custom_2d_all_parallel limited axes: []
//===----------------------------------------------------------------------===//
// CustomOp flattening.
//
// `hivm.hir.custom` / `hivm.hir.custom_macro` are opaque to the compiler, so
// they are flattened only when the author declared the loop semantics through
// `iterator_types` and provided, through `flatten_ranks` / `flatten_symbols`,
// the implementation to call at the collapsed rank. A non-`parallel` iterator
// or a missing or inconsistent declaration keeps the op at its original rank
// and its original `symbol`.
//===----------------------------------------------------------------------===//

// All dimensions are parallel and contiguous: collapse down to rank 1.
func.func @custom_2d_all_parallel(%src: memref<32x7xf32>, %dst: memref<32x7xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_2d_all_parallel",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_2d_all_parallel_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.elemwise"
      ins(%src : memref<32x7xf32>)
      outs(%dst : memref<32x7xf32>)
  return
}

// -----

// CHECK: @custom_macro_2d_all_parallel limited axes: []
// `hivm.hir.custom_macro` follows the very same rules.
func.func @custom_macro_2d_all_parallel(%src: memref<32x7xf32>, %dst: memref<32x7xf32>) {
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_macro_2d",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_macro_2d_1d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.macro_elemwise"
      ins(%src : memref<32x7xf32>)
      outs(%dst : memref<32x7xf32>)
  return
}

// -----

// CHECK: @custom_reduction_middle_dim limited axes: [0, 1, 2]
// A reduction in the middle separates both parallel dimensions, so nothing can
// be merged and the op is left untouched.
func.func @custom_reduction_middle_dim(%src: memref<4x8x16xf32>, %dst: memref<4x8x16xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_reduce_middle",
       flatten_ranks = [1 : i64, 2 : i64], flatten_symbols = ["k_reduce_middle_1d", "k_reduce_middle_2d"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<reduction>,
                         #hivm.iterator_type<parallel>]}
      "user.reduce"
      ins(%src : memref<4x8x16xf32>)
      outs(%dst : memref<4x8x16xf32>)
  return
}

// -----

// CHECK: @custom_malformed_iterator_types limited axes: [0, 1]
// A malformed `iterator_types` states nothing about the loops. No verifier
// rejects it today and the structured-op accessor hard-casts every element, so
// it must be read defensively and treated like a missing declaration.
func.func @custom_malformed_iterator_types(%src: memref<4x8xf32>, %dst: memref<4x8xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_malformed_iter",
       flatten_ranks = [1 : i64], flatten_symbols = ["k_malformed_iter_1d"],
       iterator_types = [0 : i64, 0 : i64]}
      "user.malformed_iter"
      ins(%src : memref<4x8xf32>)
      outs(%dst : memref<4x8xf32>)
  return
}
