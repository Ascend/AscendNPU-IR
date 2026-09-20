// RUN: bishengir-opt %s -hivm-merge-same-preload-scopes="bypass-shape-registry=true" | FileCheck %s

// Test 1: Merge preload_num = 0 scopes and sink trailing anchor/store while hoisting tensor.empty
// CHECK-LABEL: func.func @merge_preload_scopes_0(
// CHECK:         scf.for %[[IV:[^ ]*]] = %arg3 to %arg4 step %arg5
// CHECK:           %[[EMPTY:.*]] = tensor.empty
// CHECK:           %[[RES:.*]]:2 = scope.scope
// CHECK-DAG:         %[[V0:.*]] = memref.load %arg0[%[[IV]]]
// CHECK-DAG:         %[[V1:.*]] = memref.load %arg1[%[[IV]]]
// CHECK:             %[[VAL:.*]] = arith.addf %[[V0]], %[[V1]]
// CHECK:             memref.store %[[VAL]], %arg0[%[[IV]]]
// CHECK:             hivm.hir.anchor {id = 1 : i64}
// CHECK:             scope.return %[[V0]], %[[V1]]
// CHECK:           } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
// CHECK:           scf.yield %[[EMPTY]]
func.func @merge_preload_scopes_0(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %init: tensor<4xf32>, %lb: index, %ub: index, %step: index) -> tensor<4xf32>
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
    hivm.hir.anchor {id = 1 : i64}
    %out = tensor.empty() : tensor<4xf32>
    scf.yield %out : tensor<4xf32>
  }
  return %res : tensor<4xf32>
}

// Test 2: Distinct preload_num scopes (0 and 1) should merge independently;
// trailing bufferization.to_tensor is skipped when preload_num != 0.
// CHECK-LABEL: func.func @merge_distinct_preload_nums(
// CHECK:         scf.for %[[IV:[^ ]*]] = %arg2 to %arg3 step %arg4
// CHECK:           %[[SCOPE0:.*]]:2 = scope.scope
// CHECK:             hivm.hir.anchor {id = 10 : i64}
// CHECK:             scope.return
// CHECK:           } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 0 : i32, no_inline}
// CHECK:           %[[SCOPE1:.*]]:2 = scope.scope
// CHECK:             scope.return
// CHECK:           } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 1 : i32, no_inline}
// CHECK:           %[[TT:.*]] = bufferization.to_tensor
func.func @merge_distinct_preload_nums(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %lb: index, %ub: index, %step: index)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  scf.for %iv = %lb to %ub step %step {
    %0 = scope.scope : () -> f32 {
      %v0 = memref.load %arg0[%iv] : memref<16xf32>
      scope.return %v0 : f32
    } {hivm.preload_num = 0 : i32, no_inline}
    %1 = scope.scope : () -> f32 {
      %v1 = memref.load %arg1[%iv] : memref<16xf32>
      scope.return %v1 : f32
    } {hivm.preload_num = 0 : i32, no_inline}
    hivm.hir.anchor {id = 10 : i64}

    %2 = scope.scope : () -> f32 {
      %v2 = memref.load %arg0[%iv] : memref<16xf32>
      scope.return %v2 : f32
    } {hivm.preload_num = 1 : i32, no_inline}
    %3 = scope.scope : () -> f32 {
      %v3 = memref.load %arg1[%iv] : memref<16xf32>
      scope.return %v3 : f32
    } {hivm.preload_num = 1 : i32, no_inline}
    %tt = bufferization.to_tensor %arg0 : memref<16xf32>
    scf.yield
  }
  return
}

// Test 3: Propagate hivm.has_loop_carried_dep attribute
// CHECK-LABEL: func.func @propagate_loop_carried_dep(
// CHECK:         scf.for %[[IV:[^ ]*]] = %arg2 to %arg3 step %arg4
// CHECK:           scope.scope
// CHECK:           } {hivm.has_loop_carried_dep, hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
func.func @propagate_loop_carried_dep(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %lb: index, %ub: index, %step: index)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  scf.for %iv = %lb to %ub step %step {
    %0 = scope.scope : () -> f32 {
      %v0 = memref.load %arg0[%iv] : memref<16xf32>
      scope.return %v0 : f32
    } {hivm.has_loop_carried_dep, hivm.preload_num = 0 : i32, no_inline}
    %1 = scope.scope : () -> f32 {
      %v1 = memref.load %arg1[%iv] : memref<16xf32>
      scope.return %v1 : f32
    } {hivm.preload_num = 0 : i32, no_inline}
    scf.yield
  }
  return
}
