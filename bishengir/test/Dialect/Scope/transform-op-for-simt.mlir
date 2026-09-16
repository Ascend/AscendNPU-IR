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

// -----

// Test 5: A scalar extract hoisted from a SIMT boundary load reuses the
// original SIMD tensor. local_load must remain confined to the SIMT scope.
// CHECK-LABEL: func.func @test_hoisted_local_load
func.func @test_hoisted_local_load(%arg0: tensor<1xi32>) {
  %c0 = arith.constant 0 : index
  %buffer = bufferization.to_memref %arg0 : memref<1xi32>

  // CHECK-NOT: hivm.hir.local_load
  // CHECK: %[[EXTRACTED:.*]] = tensor.extract %arg0[%c0]
  // CHECK-NOT: hivm.hir.local_load
  // CHECK: scope.scope : () -> () {
  // CHECK-NOT: hivm.hir.local_load
  // CHECK:   scope.return
  // CHECK-NOT: hivm.hir.local_load
  // CHECK: return
  scope.scope : () -> () {
    %loaded = hivm.hir.local_load ins(%buffer : memref<1xi32>) -> tensor<1xi32>
    %extracted = tensor.extract %loaded[%c0] : tensor<1xi32>
    scope.return
  } {hivm.vf_mode = #hivm.vf_mode<SIMT>}

  return
}

// -----

// Test 6: The canonical boundary may contain memref casts. Recover the
// original SIMD tensor through the casts and retain scalar hoisting.
// CHECK-LABEL: func.func @test_hoisted_local_load_through_cast
func.func @test_hoisted_local_load_through_cast(%arg0: tensor<1xi32>) {
  %c0 = arith.constant 0 : index
  %buffer = bufferization.to_memref %arg0 : memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<1xi32, strided<[?], offset: ?>>

  // CHECK-NOT: hivm.hir.local_load
  // CHECK: tensor.extract %arg0[%c0]
  // CHECK-NOT: hivm.hir.local_load
  // CHECK: scope.scope : () -> () {
  // CHECK-NOT: hivm.hir.local_load
  // CHECK:   scope.return
  // CHECK-NOT: hivm.hir.local_load
  // CHECK: return
  scope.scope : () -> () {
    %loaded = hivm.hir.local_load ins(%cast : memref<1xi32, strided<[?], offset: ?>>) -> tensor<1xi32>
    %extracted = tensor.extract %loaded[%c0] : tensor<1xi32>
    scope.return
  } {hivm.vf_mode = #hivm.vf_mode<SIMT>}

  return
}

// -----

// Test 7: Without a recoverable SIMD tensor, keep local_load and its users in
// the SIMT scope instead of moving a SIMT-only operation into the SIMD module.
// CHECK-LABEL: func.func @test_unrecoverable_local_load_stays_in_scope
func.func @test_unrecoverable_local_load_stays_in_scope(%buffer: memref<1xi32>) {
  %c0 = arith.constant 0 : index

  // CHECK-NOT: hivm.hir.local_load
  // CHECK: scope.scope : () -> () {
  // CHECK:   %[[LOADED:.*]] = hivm.hir.local_load
  // CHECK:   tensor.extract %[[LOADED]][%c0]
  // CHECK:   scope.return
  // CHECK-NOT: hivm.hir.local_load
  // CHECK: return
  scope.scope : () -> () {
    %loaded = hivm.hir.local_load ins(%buffer : memref<1xi32>) -> tensor<1xi32>
    %extracted = tensor.extract %loaded[%c0] : tensor<1xi32>
    scope.return
  } {hivm.vf_mode = #hivm.vf_mode<SIMT>}

  return
}

// -----

// Test 8: A recovered tensor with a different encoding is not an exact SSA
// replacement. Keep the boundary load and dependent scalar extraction inside.
// CHECK-LABEL: func.func @test_mismatched_local_load_type_stays_in_scope
func.func @test_mismatched_local_load_type_stays_in_scope(
    %arg0: tensor<1xi32>) {
  %c0 = arith.constant 0 : index
  %buffer = bufferization.to_memref %arg0 : memref<1xi32>

  // CHECK-NOT: hivm.hir.local_load
  // CHECK: scope.scope : () -> () {
  // CHECK:   %[[LOADED:.*]] = hivm.hir.local_load
  // CHECK:   tensor.extract %[[LOADED]][%c0]
  // CHECK:   scope.return
  // CHECK-NOT: hivm.hir.local_load
  // CHECK: return
  scope.scope : () -> () {
    %loaded = hivm.hir.local_load ins(%buffer : memref<1xi32>) -> tensor<1xi32, "simt_encoding">
    %extracted = tensor.extract %loaded[%c0] : tensor<1xi32, "simt_encoding">
    scope.return
  } {hivm.vf_mode = #hivm.vf_mode<SIMT>}

  return
}

// -----

// Test 9: The same boundary repair applies to the tensor.from_elements hoist
// path, not only to a directly hoisted tensor.extract.
// CHECK-LABEL: func.func @test_from_elements_hoisted_local_load
func.func @test_from_elements_hoisted_local_load(%arg0: tensor<1xi32>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %buffer = bufferization.to_memref %arg0 : memref<1xi32>

  // CHECK-NOT: hivm.hir.local_load
  // CHECK: %[[VALUE:.*]] = tensor.extract %arg0[%c0]
  // CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[VALUE]], %c0_i32
  // CHECK: tensor.from_elements %[[CMP]]
  // CHECK-NOT: hivm.hir.local_load
  // CHECK: scope.scope : () -> () {
  // CHECK-NOT: hivm.hir.local_load
  // CHECK:   scope.return
  // CHECK-NOT: hivm.hir.local_load
  // CHECK: return
  scope.scope : () -> () {
    %loaded = hivm.hir.local_load ins(%buffer : memref<1xi32>) -> tensor<1xi32>
    %value = tensor.extract %loaded[%c0] : tensor<1xi32>
    %cmp = arith.cmpi slt, %value, %c0_i32 : i32
    %from_elements = tensor.from_elements %cmp : tensor<1xi1>
    scope.return
  } {hivm.vf_mode = #hivm.vf_mode<SIMT>}

  return
}

// -----

// Test 10: Classify every local_load before rewriting uses. An operation that
// consumes both a recoverable and an unrecoverable boundary load must remain
// wholly inside SIMT and keep both loads, independent of traversal order.
// CHECK-LABEL: func.func @test_mixed_recoverable_local_loads_stay_in_scope
func.func @test_mixed_recoverable_local_loads_stay_in_scope(
    %arg0: tensor<1xi32>, %unrecoverable: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %good_buffer = bufferization.to_memref %arg0 : memref<1xi32>

  // CHECK-NOT: hivm.hir.local_load
  // CHECK: scope.scope : () -> () {
  // CHECK:   %[[GOOD:.*]] = hivm.hir.local_load
  // CHECK:   %[[BAD:.*]] = hivm.hir.local_load
  // CHECK:   %[[SUM:.*]] = arith.addi %[[BAD]], %[[GOOD]]
  // CHECK:   tensor.extract %[[SUM]][%c0]
  // CHECK:   scope.return
  // CHECK-NOT: hivm.hir.local_load
  // CHECK: return
  scope.scope : () -> () {
    %good = hivm.hir.local_load ins(%good_buffer : memref<1xi32>) -> tensor<1xi32>
    %bad = hivm.hir.local_load ins(%unrecoverable : memref<1xi32>) -> tensor<1xi32>
    // The recoverable operand is pushed last and visited first by the LIFO
    // backward walk, exercising the mutation order that used to leak %arg0.
    %sum = arith.addi %bad, %good : tensor<1xi32>
    %value = tensor.extract %sum[%c0] : tensor<1xi32>
    %from_elements = tensor.from_elements %value : tensor<1xi32>
    scope.return
  } {hivm.vf_mode = #hivm.vf_mode<SIMT>}

  return
}

// -----

// Test 11: Cache only boundary sources defined outside the scope. Two
// from_elements roots discover the outer local_load before a nested load whose
// recovered source is that outer result; rewriting the first must not leave a
// dangling cached Value for the second.
// CHECK-LABEL: func.func @test_nested_local_load_source_stays_in_scope
func.func @test_nested_local_load_source_stays_in_scope(
    %arg0: tensor<1xi32>) {
  %outer_buffer = bufferization.to_memref %arg0 : memref<1xi32>

  // CHECK-NOT: hivm.hir.local_load
  // CHECK: "test.tensor_to_scalar"(%arg0)
  // CHECK: tensor.from_elements
  // CHECK: %[[INNER_BUFFER:.*]] = bufferization.to_memref %arg0
  // CHECK-NOT: hivm.hir.local_load
  // CHECK: scope.scope : () -> () {
  // CHECK:   %[[INNER:.*]] = hivm.hir.local_load ins(%[[INNER_BUFFER]]
  // CHECK:   "test.tensor_to_scalar"(%[[INNER]])
  // CHECK:   tensor.from_elements
  // CHECK:   scope.return
  // CHECK-NOT: hivm.hir.local_load
  // CHECK: return
  scope.scope : () -> () {
    %outer = hivm.hir.local_load ins(%outer_buffer : memref<1xi32>) -> tensor<1xi32>
    %outer_scalar = "test.tensor_to_scalar"(%outer) : (tensor<1xi32>) -> i32
    %outer_elements = tensor.from_elements %outer_scalar : tensor<1xi32>
    %inner_buffer = bufferization.to_memref %outer : memref<1xi32>
    %inner = hivm.hir.local_load ins(%inner_buffer : memref<1xi32>) -> tensor<1xi32>
    %inner_scalar = "test.tensor_to_scalar"(%inner) : (tensor<1xi32>) -> i32
    %inner_elements = tensor.from_elements %inner_scalar : tensor<1xi32>
    scope.return
  } {hivm.vf_mode = #hivm.vf_mode<SIMT>}

  return
}
