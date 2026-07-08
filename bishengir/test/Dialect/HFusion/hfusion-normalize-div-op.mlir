// RUN: bishengir-opt --hfusion-normalize-ops="enable-fast-div=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_linalg_div_to_mul_rec
// CHECK-SAME: (%[[arg0:.*]]: tensor<32xf32>)
// CHECK: %[[rec:.*]] = arith.constant 0.0434782617 : f32
// CHECK: %[[out:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0]], %[[rec]] : tensor<32xf32>, f32) outs(%[[out]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: return %[[res]]
func.func @test_linalg_div_to_mul_rec(%src: tensor<32xf32>) -> tensor<32xf32> {
    %cst = arith.constant 23.0e+00 : f32
    %0 = tensor.empty() : tensor<32xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%src, %cst : tensor<32xf32>, f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
    return %1 : tensor<32xf32>
}

// CHECK-LABEL: func.func @test_linalg_div_to_mul_inf_rec
// CHECK-SAME: (%[[arg0:.*]]: tensor<32xf32>)
// CHECK: %[[cst:.*]] = arith.constant {{.*}} : f32
// CHECK: %[[out:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[arg0]], %[[cst]] : tensor<32xf32>, f32) outs(%[[out]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: return %[[res]]
func.func @test_linalg_div_to_mul_inf_rec(%src: tensor<32xf32>) -> tensor<32xf32> {
    %cst = arith.constant 4.21e-40 : f32
    %0 = tensor.empty() : tensor<32xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%src, %cst : tensor<32xf32>, f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
    return %1 : tensor<32xf32>
}

// CHECK-LABEL: func.func @test_linalg_div_to_mul_denormal_rec
// CHECK-SAME: (%[[arg0:.*]]: tensor<32xf32>)
// CHECK: %[[cst:.*]] = arith.constant {{.*}} : f32
// CHECK: %[[out:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[arg0]], %[[cst]] : tensor<32xf32>, f32) outs(%[[out]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: return %[[res]]
func.func @test_linalg_div_to_mul_denormal_rec(%src: tensor<32xf32>) -> tensor<32xf32> {
    %cst = arith.constant 3.4e38 : f32
    %0 = tensor.empty() : tensor<32xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%src, %cst : tensor<32xf32>, f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
    return %1 : tensor<32xf32>
}
