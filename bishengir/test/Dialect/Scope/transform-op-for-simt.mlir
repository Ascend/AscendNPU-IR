// RUN: bishengir-opt -transform-op-for-simt %s | FileCheck %s

// Test 1: Multi-elem tensor.extract conversion
// CHECK-LABEL: func.func @test_multi_elem_extract
func.func @test_multi_elem_extract() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  // CHECK: %[[BUF:.*]] = memref.alloc() : memref<128xi32>
  // CHECK: scope.scope : () -> () {
  // CHECK:   %[[TENSOR:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK:   hivm.hir.local_store ins(%[[BUF]] : memref<128xi32>, %[[TENSOR]] : tensor<128xi32>)
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %[[BUF]][%{{.*}}] [1] [1]
  // CHECK:   %[[C0:.*]] = arith.constant 0 : index
  // CHECK:   %[[SCALAR:.*]] = memref.load %[[SUBVIEW]][%[[C0]]]
  // CHECK:   scope.return
  scope.scope : () -> () {
    %tensor = tensor.empty() : tensor<128xi32>
    %extracted = tensor.extract %tensor[%c0] : tensor<128xi32>
    scope.return
  } {hivm.vf_mode = #hivm.vf_mode<SIMT>}
  
  return
}

// -----

// Test 2: Scalar tensor.extract hoisting
// CHECK-LABEL: func.func @test_scalar_extract
func.func @test_scalar_extract() {
  %c0 = arith.constant 0 : index
  
  // CHECK: %[[TENSOR:.*]] = tensor.empty() : tensor<1xi32>
  // CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[TENSOR]][%{{.*}}]
  // CHECK: scope.scope : () -> () {
  // CHECK:   scope.return
  scope.scope : () -> () {
    %tensor = tensor.empty() : tensor<1xi32>
    %extracted = tensor.extract %tensor[%c0] : tensor<1xi32>
    scope.return
  } {hivm.vf_mode = #hivm.vf_mode<SIMT>}
  
  return
}

// -----

// Test 3: tensor.from_elements hoisting
// CHECK-LABEL: func.func @test_from_elements_hoist
func.func @test_from_elements_hoist(%arg0: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  
  // CHECK: %[[LOAD:.*]] = memref.load %{{.*}}[%{{.*}}]
  // CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[LOAD]], %{{.*}}
  // CHECK: %[[FROM_ELEM:.*]] = tensor.from_elements %[[CMP]]
  // CHECK: scope.scope : () -> () {
  // CHECK:   scope.return
  scope.scope : () -> () {
    %val = memref.load %arg0[%c0] : memref<1xi32>
    %cmp = arith.cmpi slt, %val, %c0_i32 : i32
    %from_elem = tensor.from_elements %cmp : tensor<1xi1>
    scope.return
  } {hivm.vf_mode = #hivm.vf_mode<SIMT>}
  
  return
}

// -----

// Test 4: SIMD scope should not be transformed
// CHECK-LABEL: func.func @test_simd_scope_unchanged
func.func @test_simd_scope_unchanged() {
  %c0 = arith.constant 0 : index
  
  // CHECK: scope.scope : () -> () {
  // CHECK:   %[[TENSOR:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK:   %{{.*}} = tensor.extract %[[TENSOR]][%{{.*}}]
  // CHECK:   scope.return
  scope.scope : () -> () {
    %tensor = tensor.empty() : tensor<128xi32>
    %extracted = tensor.extract %tensor[%c0] : tensor<128xi32>
    scope.return
  } {hivm.vf_mode = #hivm.vf_mode<SIMD>}
  
  return
}
