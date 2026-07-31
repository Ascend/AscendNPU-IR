// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeNearbyint_hfusion_elemwise_unary_f32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<16xf32>) -> tensor<16xf32> {
// CHECK-NOT: #hfusion.unary_fn<nearbyint>
// CHECK: %[[RINT:.*]] = hfusion.cast {{.*round_mode = #hfusion.round_mode<rint>.*}} ins(%[[ARG0]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[MAG_BITS:.*]] = hfusion.bitcast ins(%[[RINT]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[SIGN_BITS:.*]] = hfusion.bitcast ins(%[[ARG0]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[MAG:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MAG_BITS]], %{{.*}} : tensor<16xi32>, i32) outs(%{{.*}} : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[SIGN:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[SIGN_BITS]], %{{.*}} : tensor<16xi32>, i32) outs(%{{.*}} : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[RES_BITS:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[MAG]], %[[SIGN]] : tensor<16xi32>, tensor<16xi32>) outs(%{{.*}} : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[RES:.*]] = hfusion.bitcast ins(%[[RES_BITS]] : tensor<16xi32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[RES]] : tensor<16xf32>
// CHECK: }
func.func @test_NormalizeNearbyint_hfusion_elemwise_unary_f32(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tensor.empty() : tensor<16xf32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<nearbyint>} ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeNearbyint_hfusion_elemwise_unary_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK-NOT: #hfusion.unary_fn<nearbyint>
// CHECK: %[[IN_F32:.*]] = hfusion.cast {{.*round_mode = #hfusion.round_mode<rint>.*}} ins(%[[ARG0]] : tensor<8xf16>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[ROUNDED_F32:.*]] = hfusion.cast {{.*round_mode = #hfusion.round_mode<rint>.*}} ins(%[[IN_F32]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[RINT:.*]] = hfusion.cast {{.*round_mode = #hfusion.round_mode<rint>.*}} ins(%[[ROUNDED_F32]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xf16>) -> tensor<8xf16>
// CHECK: %[[MAG_BITS:.*]] = hfusion.bitcast ins(%[[RINT]] : tensor<8xf16>) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[SIGN_BITS:.*]] = hfusion.bitcast ins(%[[ARG0]] : tensor<8xf16>) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[MAG:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MAG_BITS]], %{{.*}} : tensor<8xi16>, i16) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[SIGN:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[SIGN_BITS]], %{{.*}} : tensor<8xi16>, i16) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[RES_BITS:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[MAG]], %[[SIGN]] : tensor<8xi16>, tensor<8xi16>) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[RES:.*]] = hfusion.bitcast ins(%[[RES_BITS]] : tensor<8xi16>) outs(%{{.*}} : tensor<8xf16>) -> tensor<8xf16>
// CHECK: return %[[RES]] : tensor<8xf16>
// CHECK: }
func.func @test_NormalizeNearbyint_hfusion_elemwise_unary_f16(%arg0: tensor<8xf16>) -> tensor<8xf16> {
  %0 = tensor.empty() : tensor<8xf16>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<nearbyint>} ins(%arg0 : tensor<8xf16>) outs(%0 : tensor<8xf16>) -> tensor<8xf16>
  return %1 : tensor<8xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeNearbyint_hfusion_elemwise_unary_bf16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<8xbf16>) -> tensor<8xbf16> {
// CHECK-NOT: #hfusion.unary_fn<nearbyint>
// CHECK: %[[IN_F32:.*]] = hfusion.cast {{.*round_mode = #hfusion.round_mode<rint>.*}} ins(%[[ARG0]] : tensor<8xbf16>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[ROUNDED_F32:.*]] = hfusion.cast {{.*round_mode = #hfusion.round_mode<rint>.*}} ins(%[[IN_F32]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[RINT:.*]] = hfusion.cast {{.*round_mode = #hfusion.round_mode<rint>.*}} ins(%[[ROUNDED_F32]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xbf16>) -> tensor<8xbf16>
// CHECK: %[[MAG_BITS:.*]] = hfusion.bitcast ins(%[[RINT]] : tensor<8xbf16>) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[SIGN_BITS:.*]] = hfusion.bitcast ins(%[[ARG0]] : tensor<8xbf16>) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[MAG:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[MAG_BITS]], %{{.*}} : tensor<8xi16>, i16) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[SIGN:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[SIGN_BITS]], %{{.*}} : tensor<8xi16>, i16) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[RES_BITS:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[MAG]], %[[SIGN]] : tensor<8xi16>, tensor<8xi16>) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[RES:.*]] = hfusion.bitcast ins(%[[RES_BITS]] : tensor<8xi16>) outs(%{{.*}} : tensor<8xbf16>) -> tensor<8xbf16>
// CHECK: return %[[RES]] : tensor<8xbf16>
// CHECK: }
func.func @test_NormalizeNearbyint_hfusion_elemwise_unary_bf16(%arg0: tensor<8xbf16>) -> tensor<8xbf16> {
  %0 = tensor.empty() : tensor<8xbf16>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<nearbyint>} ins(%arg0 : tensor<8xbf16>) outs(%0 : tensor<8xbf16>) -> tensor<8xbf16>
  return %1 : tensor<8xbf16>
}
