// RUN: bishengir-opt %s -hivm-flatten-ops | FileCheck %s
// These two views address the same elements and would collapse to the same
// memref<6x5x7x22xf32, strided<[10000, 1000, 100, 1]>>. The cumulative axis
// would be 1 for @left and 2 for @right. A single rank-4 symbol cannot distinguish
// identical descriptors. Both calls must keep their original rank and symbol.
// CHECK-LABEL: func.func @left(
// CHECK-NOT: memref.collapse_shape
// CHECK: hivm.hir.custom
// CHECK-SAME: symbol = "cumsum_axis2_r5"
// CHECK-SAME: memref<2x3x5x7x22xf32
// CHECK-NOT: memref.collapse_shape
// CHECK: return
func.func @left(%src: memref<2x3x5x7x22xf32, strided<[30000, 10000, 1000, 100, 1]>>, %dst: memref<2x3x5x7x22xf32, strided<[30000, 10000, 1000, 100, 1]>>) {
  hivm.hir.custom {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
    hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD>,
    symbol = "cumsum_axis2_r5", flatten_ranks = [4 : i64],
    flatten_symbols = ["cumsum_r4"],
    iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>,
                     #hivm.iterator_type<cumulative>, #hivm.iterator_type<parallel>,
                     #hivm.iterator_type<parallel>]}
    "user.cumsum" ins(%src : memref<2x3x5x7x22xf32, strided<[30000, 10000, 1000, 100, 1]>>) outs(%dst : memref<2x3x5x7x22xf32, strided<[30000, 10000, 1000, 100, 1]>>)
  return
}

// CHECK-LABEL: func.func @right(
// CHECK-NOT: memref.collapse_shape
// CHECK: hivm.hir.custom
// CHECK-SAME: symbol = "cumsum_axis2_r5"
// CHECK-SAME: memref<6x5x7x2x11xf32
// CHECK-NOT: memref.collapse_shape
// CHECK: return
func.func @right(%src: memref<6x5x7x2x11xf32, strided<[10000, 1000, 100, 11, 1]>>, %dst: memref<6x5x7x2x11xf32, strided<[10000, 1000, 100, 11, 1]>>) {
  hivm.hir.custom {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
    hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD>,
    symbol = "cumsum_axis2_r5", flatten_ranks = [4 : i64],
    flatten_symbols = ["cumsum_r4"],
    iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>,
                     #hivm.iterator_type<cumulative>, #hivm.iterator_type<parallel>,
                     #hivm.iterator_type<parallel>]}
    "user.cumsum" ins(%src : memref<6x5x7x2x11xf32, strided<[10000, 1000, 100, 11, 1]>>) outs(%dst : memref<6x5x7x2x11xf32, strided<[10000, 1000, 100, 11, 1]>>)
  return
}
