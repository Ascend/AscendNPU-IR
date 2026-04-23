// RUN: bishengir-opt -propagate-reshape -allow-unregistered-dialect %s -split-input-file | FileCheck %s

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
