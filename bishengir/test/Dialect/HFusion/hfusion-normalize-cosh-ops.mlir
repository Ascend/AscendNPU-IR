// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeCosh_hfusion_elemwise_unary_cosh(
// CHECK-SAME: %[[ARG0:.*]]: tensor<16xf32>) -> tensor<16xf32> {
// CHECK-NOT: #hfusion.unary_fn<cosh>
// CHECK: %[[EXP_X:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[ARG0]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[NEG_X:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0]], %{{.*}} : tensor<16xf32>, f32) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EXP_NEG_X:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[NEG_X]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[SUM:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXP_X]], %[[EXP_NEG_X]] : tensor<16xf32>, tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SUM]], %{{.*}} : tensor<16xf32>, f32) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[RES]] : tensor<16xf32>
// CHECK: }
func.func @test_NormalizeCosh_hfusion_elemwise_unary_cosh(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tensor.empty() : tensor<16xf32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<cosh>} ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCosh_hfusion_elemwise_unary_cosh_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK-NOT: #hfusion.unary_fn<cosh>
// CHECK: %[[IN_F32:.*]] = hfusion.cast {{.*}} ins(%[[ARG0]] : tensor<8xf16>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[EXP_X:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[IN_F32]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[NEG_X:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[IN_F32]], %{{.*}} : tensor<8xf32>, f32) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[EXP_NEG_X:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[NEG_X]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[SUM:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXP_X]], %[[EXP_NEG_X]] : tensor<8xf32>, tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[MUL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SUM]], %{{.*}} : tensor<8xf32>, f32) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[RES:.*]] = hfusion.cast {{.*}} ins(%[[MUL]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xf16>) -> tensor<8xf16>
// CHECK: return %[[RES]] : tensor<8xf16>
// CHECK: }
func.func @test_NormalizeCosh_hfusion_elemwise_unary_cosh_f16(%arg0: tensor<8xf16>) -> tensor<8xf16> {
  %0 = tensor.empty() : tensor<8xf16>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<cosh>} ins(%arg0 : tensor<8xf16>) outs(%0 : tensor<8xf16>) -> tensor<8xf16>
  return %1 : tensor<8xf16>
}
