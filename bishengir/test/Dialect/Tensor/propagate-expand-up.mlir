// RUN: bishengir-opt -propagate-reshape="for-regbased=false" -allow-unregistered-dialect %s -split-input-file | FileCheck %s

// `PropagateExpandUp` deliberately does not lift `tensor.expand_shape` across
// `linalg.fill` (non-termination with collapse-down / concat). The expand must
// stay on the fill result.
//
// CHECK-LABEL: func.func @no_expand_through_fill
func.func @no_expand_through_fill() {
  %cst = arith.constant 1.000000e+00 : f32
  %empty = tensor.empty() : tensor<6xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<6xf32>) -> tensor<6xf32>
  %expanded = tensor.expand_shape %fill [[0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>
  "some_use"(%expanded) : (tensor<2x3xf32>) -> ()
  return
}
// CHECK: linalg.fill
// CHECK-NEXT: tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>

// -----

// CHECK-LABEL: func.func @no_unit_expand_through_rank_reducing_extract
func.func @no_unit_expand_through_rank_reducing_extract(
    %src: tensor<2xi32>) -> tensor<1xi32> {
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %arg0[0] [1] [1]
  %slice = tensor.extract_slice %src[0] [1] [1] :
      tensor<2xi32> to tensor<i32>
  // CHECK: tensor.expand_shape %[[SLICE]] [] output_shape [1]
  %expanded = tensor.expand_shape %slice [] output_shape [1] :
      tensor<i32> into tensor<1xi32>
  return %expanded : tensor<1xi32>
}

// -----

// CHECK-LABEL: func.func @no_expand_through_rank_reducing_insert
func.func @no_expand_through_rank_reducing_insert(
    %src: tensor<i32>, %dest: tensor<2xi32>) -> tensor<2x1xi32> {
  // CHECK: %[[INSERTED:.*]] = tensor.insert_slice %arg0 into %arg1[0] [1] [1]
  %inserted = tensor.insert_slice %src into %dest[0] [1] [1] :
      tensor<i32> into tensor<2xi32>
  // CHECK: tensor.expand_shape %[[INSERTED]] {{\[\[}}0, 1]] output_shape [2, 1]
  %expanded = tensor.expand_shape %inserted [[0, 1]] output_shape [2, 1] :
      tensor<2xi32> into tensor<2x1xi32>
  return %expanded : tensor<2x1xi32>
}
