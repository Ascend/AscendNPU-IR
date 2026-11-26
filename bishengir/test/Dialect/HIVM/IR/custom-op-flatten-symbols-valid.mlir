// RUN: bishengir-opt %s -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @custom_wide_small
// CHECK: hivm.hir.custom
// CHECK-SAME: flatten_ranks = [1 : i128]
func.func @custom_wide_small(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [1 : i128], flatten_symbols = ["k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @custom_max_rank
// CHECK: hivm.hir.custom
// CHECK-SAME: flatten_ranks = [9223372036854775807 : i128]
func.func @custom_max_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [9223372036854775807 : i128], flatten_symbols = ["k_max"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @custom_empty_table
// CHECK: hivm.hir.custom
// CHECK-SAME: flatten_ranks = []
func.func @custom_empty_table(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [], flatten_symbols = []}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @custom_macro_wide_small
// CHECK: hivm.hir.custom_macro
// CHECK-SAME: flatten_ranks = [1 : i128]
func.func @custom_macro_wide_small(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [1 : i128], flatten_symbols = ["k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @custom_macro_max_rank
// CHECK: hivm.hir.custom_macro
// CHECK-SAME: flatten_ranks = [9223372036854775807 : i128]
func.func @custom_macro_max_rank(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [9223372036854775807 : i128], flatten_symbols = ["k_max"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @custom_macro_empty_table
// CHECK: hivm.hir.custom_macro
// CHECK-SAME: flatten_ranks = []
func.func @custom_macro_empty_table(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [], flatten_symbols = []}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @custom_boundary_ranks
// CHECK: hivm.hir.custom
// CHECK-SAME: flatten_ranks = [9223372036854775807, 9223372036854775806 : i128, 1]
func.func @custom_boundary_ranks(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [9223372036854775807 : i64, 9223372036854775806 : i128, 1 : i64],
       flatten_symbols = ["k_max", "k_max_minus_one", "k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @custom_macro_boundary_ranks
// CHECK: hivm.hir.custom_macro
// CHECK-SAME: flatten_ranks = [9223372036854775807, 9223372036854775806 : i128, 1]
func.func @custom_macro_boundary_ranks(%a: memref<2x2xf32>, %out: memref<2x2xf32>) {
  hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
       flatten_ranks = [9223372036854775807 : i64, 9223372036854775806 : i128, 1 : i64],
       flatten_symbols = ["k_max", "k_max_minus_one", "k_1d"]}
      "user.probe" ins(%a : memref<2x2xf32>) outs(%out : memref<2x2xf32>)
  return
}
