// RUN: bishengir-opt %s -wrap-cv-preload-scopes -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -wrap-cv-preload-scopes -wrap-cv-preload-scopes -split-input-file | FileCheck %s

// Three mixed-core scopes, intervening operations, and both scope and
// intervening results escaping the group. A singleton stage stays unchanged.
// CHECK-LABEL: func.func @mixed_stage
// CHECK: %[[GROUP:.*]]:3 = scope.scope : () -> (i32, tensor<?xf32>, i32) {
// CHECK-NEXT: %[[A:.*]] = scope.scope : () -> i32 {
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
// CHECK-NEXT: %[[BETWEEN:.*]] = arith.addi %[[A]], %[[A]] : i32
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() : memref<4xf32>
// CHECK-NEXT: %[[CAST:.*]] = memref.cast %[[ALLOC]]
// CHECK-NEXT: annotation.mark %[[CAST]]
// CHECK-NEXT: %[[TENSOR:.*]] = bufferization.to_tensor %[[CAST]]
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: %{{.*}} = tensor.dim %[[TENSOR]], %{{.*}}
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
// CHECK-NEXT: %[[C:.*]] = scope.scope : () -> i32 {
// CHECK-NEXT: %{{.*}} = arith.addi %[[BETWEEN]], %[[A]] : i32
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
// CHECK-NEXT: scope.return %[[BETWEEN]], %[[TENSOR]], %[[C]] : i32, tensor<?xf32>, i32
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: %{{.*}} = tensor.dim %[[GROUP]]#1, %{{.*}}
// CHECK-NEXT: %{{.*}} = arith.addi %[[GROUP]]#0, %[[GROUP]]#2 : i32
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
// CHECK-NEXT: scf.yield %[[GROUP]]#2 : i32
func.func @mixed_stage(%init: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> i32 {
    %a = scope.scope : () -> i32 {
      %v = arith.addi %acc, %acc : i32
      scope.return %v : i32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
    %between = arith.addi %a, %a : i32
    %alloc = memref.alloc() : memref<4xf32>
    %cast = memref.cast %alloc : memref<4xf32> to memref<?xf32>
    annotation.mark %cast {test} : memref<?xf32>
    %tensor = bufferization.to_tensor %cast restrict writable : memref<?xf32>
    scope.scope : () -> () {
      %dim = tensor.dim %tensor, %c0 : tensor<?xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
    %c = scope.scope : () -> i32 {
      %v = arith.addi %between, %a : i32
      scope.return %v : i32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
    scope.scope : () -> () {
      %outside_dim = tensor.dim %tensor, %c0 : tensor<?xf32>
      %sum = arith.addi %between, %c : i32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scf.yield %c : i32
  } {hivm.cv_pipelined_loop}
  return %result : i32
}

// -----

// Independent groups with no escaping values need no wrapper results.
// CHECK-LABEL: func.func @two_stages
// CHECK: scf.for
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
// CHECK-NEXT: scope.scope : () -> () {
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scope.return
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
func.func @two_stages(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32}
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32}
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32}
    scope.scope : () -> () { scope.return } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// Loops outside PipelineV2 are not interpreted as stage groups.
// CHECK-LABEL: func.func @unrelated
// CHECK: scf.for
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.preload_num = 0 : i32}
// CHECK-NEXT: scope.scope
// CHECK: } {hivm.preload_num = 0 : i32}
func.func @unrelated(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    scope.scope : () -> () { scope.return } {hivm.preload_num = 0 : i32}
    scope.scope : () -> () { scope.return } {hivm.preload_num = 0 : i32}
  }
  return
}
