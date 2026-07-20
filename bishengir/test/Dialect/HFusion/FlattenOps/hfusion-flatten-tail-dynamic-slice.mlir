// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true multi-dynamic-shape=false" -split-input-file | FileCheck %s

// -----
func.func @flatten_tail_dynamic_slice(%arg0: memref<?xf32>, %arg1: index, %arg2: index, %arg3: index) {
  %c16 = arith.constant 16 : index
  %c80 = arith.constant 80 : index
  %src = arith.constant dense<0.0> : tensor<16xf32>
  %empty = tensor.empty() : tensor<16x16x2xf32>
  %bcast = linalg.broadcast ins(%src : tensor<16xf32>) outs(%empty : tensor<16x16x2xf32>) dimensions = [0, 2]
  %slice = tensor.extract_slice %bcast[0, 0, 0] [%arg1, %arg2, 1] [1, 1, 1] : tensor<16x16x2xf32> to tensor<?x?xf32>
  %view = memref.reinterpret_cast %arg0 to offset: [%arg3], sizes: [16, 16, 2], strides: [80, 2, 1] : memref<?xf32> to memref<16x16x2xf32, strided<[80, 2, 1], offset: ?>>
  %sub = memref.subview %view[0, 0, 0] [%arg1, %arg2, 1] [1, 1, 1] : memref<16x16x2xf32, strided<[80, 2, 1], offset: ?>> to memref<?x?x1xf32, strided<[80, 2, 1], offset: ?>>
  %collapsed = memref.collapse_shape %sub [[0], [1, 2]] : memref<?x?x1xf32, strided<[80, 2, 1], offset: ?>> into memref<?x?xf32, strided<[80, 2], offset: ?>>
  bufferization.materialize_in_destination %slice in writable %collapsed : (tensor<?x?xf32>, memref<?x?xf32, strided<[80, 2], offset: ?>>) -> ()
  return
}

// CHECK-LABEL: func.func @flatten_tail_dynamic_slice
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice
// CHECK: %[[RC_INIT:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}], sizes: [16, 16, 2], strides: [80, 2, 1]
// CHECK: %[[COL_BASE:.*]] = memref.collapse_shape %[[RC_INIT]] {{\[\[}}0{{\]}}, [1, 2]]
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[COL_BASE]][0, 0] [%{{.*}}, %{{.*}}] [1, 2]
// CHECK: %[[EXPAND:.*]] = memref.expand_shape %[[SUBVIEW]] {{\[\[}}0{{\]}}, [1, 2]] output_shape [%[[ARG1:.*]], %[[ARG2:.*]], 1]
// CHECK: %{{.*}}, %[[OFF:.*]], %{{.*}}:3, %{{.*}}:3 = memref.extract_strided_metadata %[[EXPAND]]
// CHECK: %[[RC_FINAL:.*]] = memref.reinterpret_cast %[[EXPAND]] to offset: [%[[OFF]]], sizes: [%[[ARG1]], %[[ARG2]], 1], strides: [80, 2, 1]
// CHECK: %[[FINAL_COL:.*]] = memref.collapse_shape %[[RC_FINAL]] {{\[\[}}0{{\]}}, [1, 2]]
// CHECK: bufferization.materialize_in_destination %[[EXTRACTED]] in writable %[[FINAL_COL]]