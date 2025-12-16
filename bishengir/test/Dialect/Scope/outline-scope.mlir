// RUN: bishengir-opt -outline-scope -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_scope_scope_scope_0(
// CHECK-SAME: %[[F0_CST:.*]]: f32,
// CHECK-SAME: %[[F0_ALLOC:.*]]: memref<f32>) attributes {debug = 15 : index, tcore_type = #hivm.tcore_type<VECTOR>} {
// CHECK: memref.store %[[F0_CST]], %[[F0_ALLOC]][] {debug = 16 : index} : memref<f32>

// CHECK-LABEL: func.func @test_scope_scope_scope_1(
// CHECK-SAME: %[[F1_CST:.*]]: f32,
// CHECK-SAME: %[[F1_ALLOC:.*]]: memref<f32>) attributes {debug = 11 : index} {
// CHECK: memref.store %[[F1_CST]], %[[F1_ALLOC]][] {debug = 12 : index} : memref<f32>

// CHECK: func.func @test_scope_scope_scope_2(%[[F2_CST_1:.*]]: f32, %[[F2_ALLOC:.*]]: memref<f32>, %[[F2_IDX_A:.*]]: index, %[[F2_IDX_B:.*]]: index, %[[F2_STEP:.*]]: index, %[[F2_CST_2:.*]]: f32) attributes {debug = 2 : index, tcore_type = #hivm.tcore_type<CUBE>} {
// CHECK: memref.store %[[F2_CST_1]], %[[F2_ALLOC]][] {debug = 3 : index} : memref<f32>
// CHECK: scf.for %[[VAL_6:.*]] = %[[F2_IDX_A]] to %[[F2_IDX_B]] step %[[F2_STEP]] {
// CHECK: memref.store %[[F2_CST_1]], %[[F2_ALLOC]][] {debug = 9 : index} : memref<f32>
// CHECK: } {debug = 8 : index}
// CHECK: call @test_scope_scope_scope_1(%[[F2_CST_2]], %[[F2_ALLOC]]) : (f32, memref<f32>) -> ()

// CHECK-LABEL: func.func @test_scope_scope(
// CHECK-SAME: %[[ALLOC_0:.*]]: memref<f32>) attributes {debug = 0 : index} {
// CHECK: %[[STEP:.*]] = arith.constant {debug = 7 : index} 1 : index
// CHECK: %[[IDX_B:.*]] = arith.constant {debug = 6 : index} 3 : index
// CHECK: %[[IDX_A:.*]] = arith.constant {debug = 5 : index} 0 : index
// CHECK: %[[CST_2:.*]] = arith.constant {debug = 4 : index} 2.000000e-01 : f32
// CHECK: %[[CST_1:.*]] = arith.constant {debug = 1 : index} 1.000000e-01 : f32
// CHECK: call @test_scope_scope_scope_2(%[[CST_1]], %[[ALLOC_0]], %[[IDX_A]], %[[IDX_B]], %[[STEP]], %[[CST_2]]) : (f32, memref<f32>, index, index, index, f32) -> ()
// CHECK: call @test_scope_scope_scope_0(%[[CST_1]], %[[ALLOC_0]]) : (f32, memref<f32>) -> ()
// CHECK: return {debug = 18 : index}

module {
  func.func @test_scope_scope(%arg0: memref<f32>) attributes {debug = 0 : index} {
    %cst = arith.constant {debug = 1 : index} 1.000000e-01 : f32
    scope.scope : () -> () {
      memref.store %cst, %arg0[] {debug = 3 : index} : memref<f32>
      %cst_0 = arith.constant {debug = 4 : index} 2.000000e-01 : f32
      %c0 = arith.constant {debug = 5 : index} 0 : index
      %c3 = arith.constant {debug = 6 : index} 3 : index
      %c1 = arith.constant {debug = 7 : index} 1 : index
      scf.for %arg1 = %c0 to %c3 step %c1 {
        memref.store %cst, %arg0[] {debug = 9 : index} : memref<f32>
      } {debug = 8 : index}
      scope.scope : () -> () {
        memref.store %cst_0, %arg0[] {debug = 12 : index} : memref<f32>
        scope.return {debug = 13 : index}
      } {debug = 11 : index}
      scope.return {debug = 14 : index}
    } {debug = 2 : index, tcore_type = #hivm.tcore_type<CUBE>}
    scope.scope : () -> () {
      memref.store %cst, %arg0[] {debug = 16 : index} : memref<f32>
      scope.return {debug = 17 : index}
    } {debug = 15 : index, tcore_type = #hivm.tcore_type<VECTOR>}
    return {debug = 18 : index}
  }
}

// -----

// CHECK: func.func @test_scope_with_yields_scope_0(%[[ARG_0:.*]]: f32, %[[ARG_1:.*]]: f32) -> (f32, f32, f32) {
// CHECK: return %[[ARG_0]], %[[ARG_1]], %[[ARG_0]] : f32, f32, f32
// CHECK-LABEL: func.func @test_scope_with_yields(
// CHECK: %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[CALL:.*]]:3 = call @test_scope_with_yields_scope_0(%[[CST_0]], %[[CST_1]]) : (f32, f32) -> (f32, f32, f32)
// CHECK: return %[[CALL]]#0, %[[CALL]]#1, %[[CALL]]#2 : f32, f32, f32
module{
  func.func @test_scope_with_yields() -> (f32, f32, f32){
    %cst = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0:3 = scope.scope : () -> (f32, f32, f32) {
      scope.return %cst, %cst_1, %cst : f32, f32, f32
    }
    return %0#0, %0#1, %0#2 : f32, f32, f32
  }
}