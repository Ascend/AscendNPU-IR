// RUN: bishengir-opt -outline-scope -split-input-file %s | FileCheck %s

// External constants are cloned into the outlined body instead of becoming
// function arguments, so the outlined functions only take the memref and
// re-materialize the constants.
// CHECK-LABEL: func.func @test_scope_scope_scope_0(
// CHECK-SAME: %[[F0_ALLOC:.*]]: memref<f32>) attributes {debug = 15 : index, tcore_type = #hivm.tcore_type<VECTOR>} {
// CHECK: %[[F0_CST:.*]] = arith.constant {debug = 1 : index} 1.000000e-01 : f32
// CHECK: memref.store %[[F0_CST]], %[[F0_ALLOC]][] {debug = 16 : index} : memref<f32>

// CHECK-LABEL: func.func @test_scope_scope_scope_1(
// CHECK-SAME: %[[F1_ALLOC:.*]]: memref<f32>) attributes {debug = 11 : index} {
// CHECK: %[[F1_CST:.*]] = arith.constant {debug = 4 : index} 2.000000e-01 : f32
// CHECK: memref.store %[[F1_CST]], %[[F1_ALLOC]][] {debug = 12 : index} : memref<f32>

// CHECK-LABEL: func.func @test_scope_scope_scope_2(
// CHECK-SAME: %[[F2_ALLOC:.*]]: memref<f32>) attributes {debug = 2 : index, tcore_type = #hivm.tcore_type<CUBE>} {
// CHECK: %[[F2_CST:.*]] = arith.constant {debug = 1 : index} 1.000000e-01 : f32
// CHECK: %[[F2_IDX_A:.*]] = arith.constant {debug = 5 : index} 0 : index
// CHECK: %[[F2_IDX_B:.*]] = arith.constant {debug = 6 : index} 3 : index
// CHECK: %[[F2_STEP:.*]] = arith.constant {debug = 7 : index} 1 : index
// CHECK: memref.store %[[F2_CST]], %[[F2_ALLOC]][] {debug = 3 : index} : memref<f32>
// CHECK: scf.for %[[VAL_6:.*]] = %[[F2_IDX_A]] to %[[F2_IDX_B]] step %[[F2_STEP]] {
// CHECK: memref.store %[[F2_CST]], %[[F2_ALLOC]][] {debug = 9 : index} : memref<f32>
// CHECK: } {debug = 8 : index}
// CHECK: call @test_scope_scope_scope_1(%[[F2_ALLOC]]) : (memref<f32>) -> ()

// CHECK-LABEL: func.func @test_scope_scope(
// CHECK-SAME: %[[ALLOC_0:.*]]: memref<f32>) attributes {debug = 0 : index} {
// CHECK: call @test_scope_scope_scope_2(%[[ALLOC_0]]) : (memref<f32>) -> ()
// CHECK: call @test_scope_scope_scope_0(%[[ALLOC_0]]) : (memref<f32>) -> ()
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

// CHECK-LABEL: func.func @test_scope_with_yields_scope_0() -> (f32, f32, f32) {
// CHECK: %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: return %[[CST_0]], %[[CST_1]], %[[CST_0]] : f32, f32, f32
// CHECK-LABEL: func.func @test_scope_with_yields(
// CHECK: %[[CALL:.*]]:3 = call @test_scope_with_yields_scope_0() : () -> (f32, f32, f32)
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

// -----

// CHECK-LABEL: func.func @test_mixed_inputs_scope_0(
// CHECK-SAME: %[[REINTERPRET1:.*]]: memref<8xf32, strided<[1], offset: ?>>,
// CHECK-SAME: %[[REINTERPRET2:.*]]: memref<8xf32, strided<[1], offset: 2>>) attributes {outline = true}
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: memref.store %[[CST]], %[[REINTERPRET1]][%[[C0]]] : memref<8xf32, strided<[1], offset: ?>>
// CHECK: memref.store %[[CST]], %[[REINTERPRET2]][%[[C0]]] : memref<8xf32, strided<[1], offset: 2>>
// CHECK: return

// CHECK-LABEL: func.func @test_mixed_inputs(
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[REINTERPRET1:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [%[[C4]]]
// CHECK: %[[REINTERPRET2:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [2]
// CHECK: call @test_mixed_inputs_scope_0(%[[REINTERPRET1]], %[[REINTERPRET2]])
module {
  func.func @test_mixed_inputs(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    %c4 = arith.constant 4 : index
    %reinterpret1 = memref.reinterpret_cast %arg0 to offset: [%c4], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1], offset: ?>>
    %reinterpret2 = memref.reinterpret_cast %arg1 to offset: [2], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1], offset: 2>>
    %cst = arith.constant 1.0 : f32
    %c0 = arith.constant 0 : index
    scope.scope : () -> () {
      memref.store %cst, %reinterpret1[%c0] : memref<8xf32, strided<[1], offset: ?>>
      memref.store %cst, %reinterpret2[%c0] : memref<8xf32, strided<[1], offset: 2>>
      scope.return
    } {outline = true}
    return
  }
}

// -----

// Test: SIMT scope with sub-block tiling reverted attribute should be wrapped
// in scf.if guard with limit_sub_block_id0.
// CHECK-LABEL: func.func @test_simt_scope_with_reverted_attr_scope_0(
// CHECK: memref.store
// CHECK: return

// CHECK-LABEL: func.func @test_simt_scope_with_reverted_attr(
// CHECK: hivm.hir.get_sub_block_idx
// CHECK: arith.index_cast
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK: call @test_simt_scope_with_reverted_attr_scope_0
// CHECK: limit_sub_block_id0
module attributes {"hivm.tile_and_bind_subblock_reverted"} {
  func.func @test_simt_scope_with_reverted_attr(%arg0: memref<f32>) {
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      memref.store %cst, %arg0[] : memref<f32>
      scope.return
    } {hivm.vf_mode = #hivm.vf_mode<SIMT>}
    return
  }
}

// -----

// Test: SIMT scope without the reverted attribute should produce a plain call
// (no scf.if guard).
// CHECK-LABEL: func.func @test_simt_scope_without_reverted_attr_scope_0(
// CHECK: memref.store
// CHECK: return

// CHECK-LABEL: func.func @test_simt_scope_without_reverted_attr(
// CHECK-NOT: hivm.hir.get_sub_block_idx
// CHECK: call @test_simt_scope_without_reverted_attr_scope_0
module {
  func.func @test_simt_scope_without_reverted_attr(%arg0: memref<f32>) {
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      memref.store %cst, %arg0[] : memref<f32>
      scope.return
    } {hivm.vf_mode = #hivm.vf_mode<SIMT>}
    return
  }
}

// -----

// Test: Non-SIMT scope with the reverted attribute should produce a plain call
// (no scf.if guard) because both conditions (SIMT AND reverted) must hold.
// CHECK-LABEL: func.func @test_non_simt_scope_with_reverted_attr_scope_0(
// CHECK: memref.store
// CHECK: return

// CHECK-LABEL: func.func @test_non_simt_scope_with_reverted_attr(
// CHECK-NOT: hivm.hir.get_sub_block_idx
// CHECK: call @test_non_simt_scope_with_reverted_attr_scope_0
module attributes {"hivm.tile_and_bind_subblock_reverted"} {
  func.func @test_non_simt_scope_with_reverted_attr(%arg0: memref<f32>) {
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      memref.store %cst, %arg0[] : memref<f32>
      scope.return
    }
    return
  }
}
