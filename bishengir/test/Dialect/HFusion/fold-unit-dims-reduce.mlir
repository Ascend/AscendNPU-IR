// RUN: bishengir-opt -hfusion-fold-unit-dims -cse -canonicalize %s | FileCheck %s

// Test case: Reduce with output shape 1xN should NOT be reduced to scalar
// This tests the fix for skipping reduce ops where output is 1xdtype


// -----

// CHECK-LABEL: @reduce_unit_dim_output_1x1_should_skip
// CHECK-NOT: tensor.collapse_shape
// CHECK: linalg.reduce ins(%arg0 : tensor<1x8777xf32>) outs(%arg1 : tensor<1xf32>)
func.func @reduce_unit_dim_output_1x1_should_skip(%arg0: tensor<1x8777xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = linalg.reduce ins(%arg0 : tensor<1x8777xf32>) outs(%arg1 : tensor<1xf32>) dimensions = [1]
  (%in: f32, %init: f32) {
    %2 = arith.addf %in, %init : f32
    linalg.yield %2 : f32
  }
  return %0 : tensor<1xf32>
}


// -----

// CHECK-LABEL: @reduce_unit_dim_input_should_be_optimized
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[0, 1\]\]}}
// CHECK: linalg.reduce ins({{.*}} : tensor<2x3xf32>) outs({{.*}} : tensor<2xf32>)
func.func @reduce_unit_dim_input_should_be_optimized(%arg0: tensor<2x1x3xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x1xf32> {
  %0 = linalg.reduce ins(%arg0 : tensor<2x1x3xf32>) outs(%arg1 : tensor<2x1xf32>) dimensions = [2]
  (%in: f32, %init: f32) {
    %2 = arith.addf %in, %init : f32
    linalg.yield %2 : f32
  }
  return %0 : tensor<2x1xf32>
}


