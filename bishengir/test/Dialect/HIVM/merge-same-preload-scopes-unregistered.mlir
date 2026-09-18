// RUN: bishengir-opt %s -hivm-merge-same-preload-scopes | FileCheck %s

// Off-registry functions must not merge scopes that share preload_num.
// CHECK-LABEL: func.func @keep_distinct_same_preload_num_scopes(
// CHECK:         scf.for
// CHECK:           scope.scope
// CHECK:             memref.load %arg0
// CHECK:             scope.return
// CHECK:           } {{{.*}}preload_num = 0
// CHECK:           scope.scope
// CHECK:             memref.load %arg1
// CHECK:             scope.return
// CHECK:           } {{{.*}}preload_num = 0
func.func @keep_distinct_same_preload_num_scopes(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %init: tensor<4xf32>, %lb: index, %ub: index, %step: index) -> tensor<4xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %res = scf.for %iv = %lb to %ub step %step iter_args(%iter = %init) -> (tensor<4xf32>) {
    %0 = scope.scope : () -> f32 {
      %v0 = memref.load %arg0[%iv] : memref<16xf32>
      scope.return %v0 : f32
    } {hivm.preload_num = 0 : i32, no_inline}
    %1 = scope.scope : () -> f32 {
      %v1 = memref.load %arg1[%iv] : memref<16xf32>
      scope.return %v1 : f32
    } {hivm.preload_num = 0 : i32, no_inline}
    %val = arith.addf %0, %1 : f32
    memref.store %val, %arg0[%iv] : memref<16xf32>
    %out = tensor.empty() : tensor<4xf32>
    scf.yield %out : tensor<4xf32>
  }
  return %res : tensor<4xf32>
}

// Registered name is not enough: merge stays off unless preload is on.
// CHECK-LABEL: func.func @cross_core_loop_carry_skew(
// CHECK:         scf.for
// CHECK:           scope.scope
// CHECK:             memref.load %arg0
// CHECK:             scope.return
// CHECK:           } {{{.*}}preload_num = 0
// CHECK:           scope.scope
// CHECK:             memref.load %arg1
// CHECK:             scope.return
// CHECK:           } {{{.*}}preload_num = 0
func.func @cross_core_loop_carry_skew(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %init: tensor<4xf32>, %lb: index, %ub: index, %step: index) -> tensor<4xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %res = scf.for %iv = %lb to %ub step %step iter_args(%iter = %init) -> (tensor<4xf32>) {
    %0 = scope.scope : () -> f32 {
      %v0 = memref.load %arg0[%iv] : memref<16xf32>
      scope.return %v0 : f32
    } {hivm.preload_num = 0 : i32, no_inline}
    %1 = scope.scope : () -> f32 {
      %v1 = memref.load %arg1[%iv] : memref<16xf32>
      scope.return %v1 : f32
    } {hivm.preload_num = 0 : i32, no_inline}
    %val = arith.addf %0, %1 : f32
    memref.store %val, %arg0[%iv] : memref<16xf32>
    %out = tensor.empty() : tensor<4xf32>
    scf.yield %out : tensor<4xf32>
  }
  return %res : tensor<4xf32>
}
