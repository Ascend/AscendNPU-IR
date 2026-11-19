// REQUIRES: expensive-pattern-api-checks
// RUN: bishengir-opt %s -hivm-flatten-ops | FileCheck %s
// Expensive pattern API checks reject a pattern that changes IR and returns
// failure. Exercise both a native op and a custom op through the shared pattern.

// CHECK-LABEL: func.func @native(
// CHECK: memref.collapse_shape
// CHECK: hivm.hir.vadd
func.func @native(%a: memref<2x8xf32>, %b: memref<2x8xf32>, %out: memref<2x8xf32>) {
  hivm.hir.vadd ins(%a, %b : memref<2x8xf32>, memref<2x8xf32>)
                 outs(%out : memref<2x8xf32>)
  return
}

// CHECK-LABEL: func.func @custom(
// CHECK: memref.collapse_shape
// CHECK: hivm.hir.custom
// CHECK-SAME: symbol = "add_r1"
func.func @custom(%a: memref<2x8xf32>, %out: memref<2x8xf32>) {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "add_r2", flatten_ranks = [1 : i64],
       flatten_symbols = ["add_r1"],
       iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>]}
      "user.add" ins(%a : memref<2x8xf32>) outs(%out : memref<2x8xf32>)
  return
}
