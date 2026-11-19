// RUN: bishengir-opt %s -hivm-flatten-ops -split-input-file | FileCheck %s

// -----

// CHECK: hacc.target = #hacc.target<"Ascend910B1">
// CHECK-LABEL: func.func @custom_Ascend910B1
// CHECK: memref.collapse_shape
// CHECK: hivm.hir.custom
// CHECK-SAME: symbol = "k_1d"
// CHECK-SAME: ins(%{{.*}} : memref<32xf32>) outs(%{{.*}} : memref<32xf32>)
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @custom_Ascend910B1(%a: memref<4x8xf32>, %out: memref<4x8xf32>) {
    hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
         iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
         flatten_ranks = [1 : i64], flatten_symbols = ["k_1d"]}
        "user.probe" ins(%a : memref<4x8xf32>) outs(%out : memref<4x8xf32>)
    return
  }
}

// -----

// CHECK: hacc.target = #hacc.target<"Ascend910B1">
// CHECK-LABEL: func.func @custom_macro_Ascend910B1
// CHECK: memref.collapse_shape
// CHECK: hivm.hir.custom_macro
// CHECK-SAME: symbol = "k_1d"
// CHECK-SAME: ins(%{{.*}} : memref<32xf32>) outs(%{{.*}} : memref<32xf32>)
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @custom_macro_Ascend910B1(%a: memref<4x8xf32>, %out: memref<4x8xf32>) {
    hivm.hir.custom_macro
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
         iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
         flatten_ranks = [1 : i64], flatten_symbols = ["k_1d"]}
        "user.probe" ins(%a : memref<4x8xf32>) outs(%out : memref<4x8xf32>)
    return
  }
}

// -----

// CHECK: hacc.target = #hacc.target<"Ascend950">
// CHECK-LABEL: func.func @custom_Ascend950
// CHECK: memref.collapse_shape
// CHECK: hivm.hir.custom
// CHECK-SAME: symbol = "k_1d"
// CHECK-SAME: ins(%{{.*}} : memref<32xf32>) outs(%{{.*}} : memref<32xf32>)
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @custom_Ascend950(%a: memref<4x8xf32>, %out: memref<4x8xf32>) {
    hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
         iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
         flatten_ranks = [1 : i64], flatten_symbols = ["k_1d"]}
        "user.probe" ins(%a : memref<4x8xf32>) outs(%out : memref<4x8xf32>)
    return
  }
}

// -----

// CHECK: hacc.target = #hacc.target<"Ascend950">
// CHECK-LABEL: func.func @custom_macro_Ascend950
// CHECK: memref.collapse_shape
// CHECK: hivm.hir.custom_macro
// CHECK-SAME: symbol = "k_1d"
// CHECK-SAME: ins(%{{.*}} : memref<32xf32>) outs(%{{.*}} : memref<32xf32>)
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @custom_macro_Ascend950(%a: memref<4x8xf32>, %out: memref<4x8xf32>) {
    hivm.hir.custom_macro
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "k_2d",
         iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
         flatten_ranks = [1 : i64], flatten_symbols = ["k_1d"]}
        "user.probe" ins(%a : memref<4x8xf32>) outs(%out : memref<4x8xf32>)
    return
  }
}
