// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s

// CHECK-LABEL: func.func @register_tree_selection
// CHECK: linalg.generic
// CHECK-SAME: hfusion.register_tree_reduction_selected
func.func @register_tree_selection(%input: tensor<32x32xf32>,
                                   %init: tensor<32xf32>)
    -> tensor<32xf32> {
  %reduced = linalg.reduce ins(%input : tensor<32x32xf32>)
      outs(%init : tensor<32xf32>) dimensions = [0]
      {hfusion.register_tree_reduction_selected}
    (%in: f32, %out: f32) {
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    }
  return %reduced : tensor<32xf32>
}

// CHECK-LABEL: func.func @regular_tree_selection
// CHECK: linalg.generic
// CHECK-SAME: hfusion.regular_tree_reduction_selected
func.func @regular_tree_selection(%input: tensor<32x32xf32>,
                                  %init: tensor<32xf32>)
    -> tensor<32xf32> {
  %reduced = linalg.reduce ins(%input : tensor<32x32xf32>)
      outs(%init : tensor<32xf32>) dimensions = [0]
      {hfusion.regular_tree_reduction_selected}
    (%in: f32, %out: f32) {
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    }
  return %reduced : tensor<32xf32>
}
