// RUN: bishengir-opt -propagate-reshape=for-regbased=true -allow-unregistered-dialect %s -split-input-file | FileCheck %s

// Memref materialize has zero results; after rewiring to collapse src the
// original collapse must be erased when unused.
// CHECK-LABEL: func.func @materialize_through_collapse
// CHECK: %[[SOURCE:.*]] = math.absf %arg0 : tensor<2x3xf32>
// CHECK-NOT: tensor.collapse_shape
// CHECK: %[[DEST:.*]] = memref.expand_shape %arg1 {{\[\[}}0, 1]] output_shape [2, 3]
// CHECK: bufferization.materialize_in_destination %[[SOURCE]] in writable %[[DEST]]
// CHECK-SAME: (tensor<2x3xf32>, memref<2x3xf32>) -> ()
func.func @materialize_through_collapse(
    %arg0: tensor<2x3xf32>, %dest: memref<6xf32>) {
  %source = math.absf %arg0 : tensor<2x3xf32>
  %collapsed = tensor.collapse_shape %source [[0, 1]]
      : tensor<2x3xf32> into tensor<6xf32>
  bufferization.materialize_in_destination %collapsed in writable %dest
      : (tensor<6xf32>, memref<6xf32>) -> ()
  return
}

// -----

// Rewiring materialize must not erase a collapse that still has another user.
// CHECK-LABEL: func.func @materialize_collapse_shared_user
// CHECK: %[[SOURCE:.*]] = math.absf %arg0 : tensor<2x3xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[SOURCE]]
// CHECK: "test.keep"(%[[COLLAPSED]])
// CHECK: %[[DEST:.*]] = memref.expand_shape %arg1 {{\[\[}}0, 1]] output_shape [2, 3]
// CHECK: bufferization.materialize_in_destination %[[SOURCE]] in writable %[[DEST]]
// CHECK: return %[[COLLAPSED]]
func.func @materialize_collapse_shared_user(
    %arg0: tensor<2x3xf32>, %dest: memref<6xf32>) -> tensor<6xf32> {
  %source = math.absf %arg0 : tensor<2x3xf32>
  %collapsed = tensor.collapse_shape %source [[0, 1]]
      : tensor<2x3xf32> into tensor<6xf32>
  "test.keep"(%collapsed) : (tensor<6xf32>) -> ()
  bufferization.materialize_in_destination %collapsed in writable %dest
      : (tensor<6xf32>, memref<6xf32>) -> ()
  return %collapsed : tensor<6xf32>
}
