// RUN: bishengir-opt %s --one-shot-bufferize="dialect-filter=tensor,bufferization copy-before-write unknown-type-conversion=identity-layout-map" -cse -split-input-file | FileCheck %s

// CHECK-LABEL: func @tensor.expand_shape(
// CHECK-SAME:     %[[t1:.*]]: tensor<?xf32>,
// CHECK-SAME:     %[[sz0:.*]]: index,
// CHECK-SAME:     %[[sz1:.*]]: index
func.func @tensor.expand_shape(%t1: tensor<?xf32>, %sz0: index, %sz1: index) -> tensor<?x?xf32> {
  // memref.expand_shape %[[t1]] {{\[\[}}0, 1]] output_shape [%[[sz0]], %[[sz1]]] : memref<?xf32> into memref<?x?xf32>
  %0 = tensor.expand_shape %t1 [[0, 1]] output_shape [%sz0, %sz1]
      : tensor<?xf32> into tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}