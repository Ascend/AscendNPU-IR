// RUN: bishengir-opt -fold-alloc-reshape -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test_single_use
// CHECK-SAME: %[[DIM0:.*]]: index, %[[DIM1:.*]]: index, %[[DIM2:.*]]: index
// CHECK: memref.alloc(%[[DIM1]], %[[DIM2]])
// CHECK-NOT: memref.expand_shape
func.func @test_single_use(%dim0: index, %dim1: index, %dim2: index) {
  %alloc = memref.alloc(%dim0) : memref<?xf32>
  %res = memref.expand_shape %alloc [[0, 1]] output_shape [%dim1, %dim2] : memref<?xf32> into memref<?x?xf32>
  annotation.mark %res : memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @test_multi_use
// CHECK: memref.alloc
// CHECK-SAME: memref<?xf32>
// CHECK: memref.expand_shape
// CHECK: memref.expand_shape
func.func @test_multi_use(%dim0: index, %dim1: index, %dim2: index) {
  %alloc = memref.alloc(%dim0) : memref<?xf32>
  %res0 = memref.expand_shape %alloc [[0, 1]] output_shape [%dim1, %dim2] : memref<?xf32> into memref<?x?xf32>
  %res1 = memref.expand_shape %alloc [[0, 1]] output_shape [%dim2, %dim1] : memref<?xf32> into memref<?x?xf32>
  annotation.mark %res0 : memref<?x?xf32>
  annotation.mark %res1 : memref<?x?xf32>
  return
}