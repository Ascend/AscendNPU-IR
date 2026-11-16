// RUN: bishengir-opt %s -hivm-flatten-ops -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -test-flatten-limited-axes -split-input-file -o /dev/null | FileCheck %s --check-prefix=AXES

// Unranked operands are legal, but do not define an iteration rank to flatten.
// CHECK-LABEL: func.func @custom_unranked
// CHECK-NOT: memref.collapse_shape
// CHECK: hivm.hir.custom
// CHECK-SAME: symbol = "user_unranked"
// CHECK-SAME: ins({{.*}} : memref<*xf32>) outs({{.*}} : memref<*xf32>)
// AXES: @custom_unranked limited axes: []
func.func @custom_unranked(%src: memref<*xf32>, %dst: memref<*xf32>) {
  hivm.hir.custom {
    hivm.tcore_type = #hivm.tcore_type<VECTOR>,
    hivm.pipe = #hivm.pipe<PIPE_V>,
    hivm.vf_mode = #hivm.vf_mode<SIMD>,
    symbol = "user_unranked"
  } "user.unranked"
  ins(%src : memref<*xf32>) outs(%dst : memref<*xf32>)
  return
}

// -----

// A ranked output does not make an unranked input safe to flatten.
// CHECK-LABEL: func.func @custom_macro_unranked
// CHECK-NOT: memref.collapse_shape
// CHECK: hivm.hir.custom_macro
// CHECK-SAME: symbol = "user_macro_unranked"
// CHECK-SAME: ins({{.*}} : memref<*xf32>) outs({{.*}} : memref<4x8xf32>)
// AXES: @custom_macro_unranked limited axes: [0, 1]
func.func @custom_macro_unranked(%src: memref<*xf32>, %dst: memref<4x8xf32>) {
  hivm.hir.custom_macro {
    hivm.tcore_type = #hivm.tcore_type<VECTOR>,
    hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
    hivm.pipe_out = #hivm.pipe<PIPE_V>,
    hivm.vf_mode = #hivm.vf_mode<SIMD>,
    symbol = "user_macro_unranked"
  } "user.macro_unranked"
  ins(%src : memref<*xf32>) outs(%dst : memref<4x8xf32>)
  return
}
