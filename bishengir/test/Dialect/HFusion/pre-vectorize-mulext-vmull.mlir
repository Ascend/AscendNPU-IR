// pre-vectorize-mulext-vmull
// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s

// Test GeneralizeMulextPattern - Signed integer multiplication
// -----
func.func @test_mul_ext_signed_2d(%arg0: tensor<16x32xi32>, %arg1: tensor<16x32xi32>) 
    -> (tensor<16x32xi32>, tensor<16x32xi32>) {
  %0, %1 = hfusion.mulext %arg0, %arg1 : tensor<16x32xi32>
  return %0, %1 : tensor<16x32xi32>, tensor<16x32xi32>
}

// CHECK-LABEL: func.func @test_mul_ext_signed_2d
// CHECK: %[[LOW_EMPTY:.*]] = tensor.empty() : tensor<16x32xi32>
// CHECK: %[[HIGH_EMPTY:.*]] = tensor.empty() : tensor<16x32xi32>
// CHECK:  linalg.generic
// CHECK-SAME: ins(%arg0, %arg1 : tensor<16x32xi32>, tensor<16x32xi32>)
// CHECK-SAME: outs(%[[LOW_EMPTY]], %[[HIGH_EMPTY]] : tensor<16x32xi32>, tensor<16x32xi32>)
// CHECK: arith.mulsi_extended


