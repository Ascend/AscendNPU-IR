// RUN: bishengir-opt %s --canonicalize="enable-extended-patterns=true" -split-input-file | FileCheck %s

// CHECK-LABEL: func @reinterpret_constant_arg_folder_unranked_memref
func.func @reinterpret_constant_arg_folder_unranked_memref(%arg0 : memref<*xf16>) -> memref<?xf16, strided<[?], offset: ?>> {
  %offset = arith.constant 0 : index
  %size = arith.constant 1024 : index
  %stride = arith.constant 1 : index
  // CHECK: memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<*xf16> to memref<1024xf16, strided<[1]>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%offset], sizes: [%size], strides: [%stride] : memref<*xf16> to memref<?xf16, strided<[?], offset: ?>>
  return %reinterpret_cast : memref<?xf16, strided<[?], offset: ?>>
}

// -----

// CHECK-LABEL: func @reinterpret_constant_arg_folder_memref
func.func @reinterpret_constant_arg_folder_memref(%arg0 : memref<?xf16>) -> memref<?xf16, strided<[?], offset: ?>> {
  %offset = arith.constant 0 : index
  %size = arith.constant 1024 : index
  %stride = arith.constant 1 : index
  // CHECK: memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<?xf16> to memref<1024xf16, strided<[1]>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%offset], sizes: [%size], strides: [%stride] : memref<?xf16> to memref<?xf16, strided<[?], offset: ?>>
  return %reinterpret_cast : memref<?xf16, strided<[?], offset: ?>>
}

// -----

func.func @redundant_copy_not_folded_across_blocks(
    %cond: i1, %src: memref<4xf32>, %mid: memref<4xf32>,
    %dst: memref<4xf32>) {
  memref.copy %src, %mid : memref<4xf32> to memref<4xf32>
  scf.if %cond {
    memref.copy %mid, %dst : memref<4xf32> to memref<4xf32>
  }
  return
}

// CHECK-LABEL: func @redundant_copy_not_folded_across_blocks(
//  CHECK-SAME:     %[[COND:.*]]: i1, %[[SRC:.*]]: memref<4xf32>, %[[MID:.*]]: memref<4xf32>, %[[DST:.*]]: memref<4xf32>
//       CHECK:   memref.copy %[[SRC]], %[[MID]] : memref<4xf32> to memref<4xf32>
//       CHECK:   scf.if %[[COND]] {
//       CHECK:     memref.copy %[[MID]], %[[DST]] : memref<4xf32> to memref<4xf32>
//       CHECK:   }
//       CHECK:   return

// -----

func.func @redundant_copy_not_folded_with_nested_side_effect(
    %cond: i1, %src: memref<4xf32>, %mid: memref<4xf32>,
    %dst: memref<4xf32>, %idx: index, %value: f32) {
  memref.copy %src, %mid : memref<4xf32> to memref<4xf32>
  scf.if %cond {
    memref.store %value, %src[%idx] : memref<4xf32>
  }
  memref.copy %mid, %dst : memref<4xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func @redundant_copy_not_folded_with_nested_side_effect(
//  CHECK-SAME:     %[[COND:.*]]: i1, %[[SRC:.*]]: memref<4xf32>, %[[MID:.*]]: memref<4xf32>, %[[DST:.*]]: memref<4xf32>, %[[IDX:.*]]: index, %[[VALUE:.*]]: f32
//       CHECK:   memref.copy %[[SRC]], %[[MID]] : memref<4xf32> to memref<4xf32>
//       CHECK:   scf.if %[[COND]] {
//       CHECK:     memref.store %[[VALUE]], %[[SRC]][%[[IDX]]] : memref<4xf32>
//       CHECK:   }
//       CHECK:   memref.copy %[[MID]], %[[DST]] : memref<4xf32> to memref<4xf32>
//       CHECK:   return