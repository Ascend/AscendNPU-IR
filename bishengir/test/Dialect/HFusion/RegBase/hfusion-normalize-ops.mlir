// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend310B4 --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_cast_op_tensor_i64_to_f16
// CHECK: %[[ZERO:.*]] = tensor.empty() : tensor<23xf16>
// CHECK: %[[ONE:.*]] = tensor.empty() : tensor<23xf32>
// CHECK: %[[TWO:.*]] = hfusion.cast {{.*}} ins(%[[arg0:.*]] : tensor<23xi64>) outs(%[[ONE:.*]] : tensor<23xf32>) -> tensor<23xf32>
// CHECK: %[[THREE:.*]] = hfusion.cast {{.*}} ins(%[[TWO:.*]] : tensor<23xf32>) outs(%[[ZERO:.*]] : tensor<23xf16>) -> tensor<23xf16>
func.func @test_cast_op_tensor_i64_to_f16(%arg0: tensor<23xi64>, %arg1: tensor<f16>) -> tensor<23xf16> attributes {hacc.entry} {
    %cst = arith.constant 0.86956521739130432 : f64
    %0 = tensor.empty() : tensor<23xf16>
    %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<23xi64>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<f16>) outs(%0 : tensor<23xf16>) dimensions = [0]
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%1, %broadcasted : tensor<23xf16>, tensor<23xf16>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %3 = arith.truncf %cst : f64 to f16
    %4 = linalg.fill ins(%3 : f16) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %4 : tensor<23xf16>, tensor<23xf16>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    return %5 : tensor<23xf16>
  }
// -----

// CHECK-LABEL: func.func @test_hfusion_powf_cast_fill
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
func.func @test_hfusion_powf_cast_fill(%arg0: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %cst_1 = arith.constant 0.5 : f16
  %1 = tensor.empty(): tensor<16xf16>
  %2 = linalg.fill ins(%cst_1 : f16) outs(%1 : tensor<16xf16>) -> tensor<16xf16>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%2 : tensor<16xf16>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %3: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %res : tensor<16xf32>
}
// -----

// CHECK-LABEL: @lowering_cast_i64_to_bf16(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xbf16>) -> tensor<4x4xbf16>
func.func @lowering_cast_i64_to_bf16(%arg0: tensor<4x4xi64>) -> tensor<4x4xbf16> {
  %0 = tensor.empty() : tensor<4x4xbf16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %1 : tensor<4x4xbf16>
}
// -----

// CHECK-LABEL: @test_i16_cast_i32(
// CHECK: %0 = tensor.empty() : tensor<4x4xi32>
// CHECK: %1 = hfusion.cast {{.*}} ins(%arg0 : tensor<4x4xi16>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: return %1 : tensor<4x4xi32>
func.func @test_i16_cast_i32(%arg0: tensor<4x4xi16>) -> tensor<4x4xi32> {
  %0 = tensor.empty() : tensor<4x4xi32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi16>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}
// -----

// CHECK-LABEL: @test_i8_cast_i32(
// CHECK: %[[CAST:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi8>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: return %[[CAST]] : tensor<4x4xi32>
func.func @test_i8_cast_i32(%arg0: tensor<4x4xi8>) -> tensor<4x4xi32> {
  %0 = tensor.empty() : tensor<4x4xi32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi8>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}
// -----

// CHECK-LABEL: @test_cast_i8_to_bf16
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi8>) outs({{.*}} : tensor<4x4xf16>) -> tensor<4x4xf16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf16>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xbf16>) -> tensor<4x4xbf16>
func.func @test_cast_i8_to_bf16(%arg0: tensor<4x4xi8>) -> tensor<4x4xbf16> {
  %0 = tensor.empty() : tensor<4x4xbf16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi8>) outs(%0 : tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %1 : tensor<4x4xbf16>
}
// -----

// CHECK-LABEL: @test_cast_f32_to_i16
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi32>) outs({{.*}} : tensor<4x4xi16>) -> tensor<4x4xi16>
func.func @test_cast_f32_to_i16(%arg0: tensor<4x4xf32>) -> tensor<4x4xi16> {
  %0 = tensor.empty() : tensor<4x4xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%arg0 : tensor<4x4xf32>) outs(%0 : tensor<4x4xi16>) -> tensor<4x4xi16>
  return %1 : tensor<4x4xi16>
}

// -----
// CHECK-LABEL: @test_cast_i64_to_i16
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xi16>) -> tensor<4x4xi16>
func.func @test_cast_i64_to_i16(%arg0: tensor<4x4xi64>) -> tensor<4x4xi16> {
  %0 = tensor.empty() : tensor<4x4xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xi16>) -> tensor<4x4xi16>
  return %1 : tensor<4x4xi16>
}

// -----
// CHECK-LABEL: @test_cast_i64_to_i8
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xi8>) -> tensor<4x4xi8>
func.func @test_cast_i64_to_i8(%arg0: tensor<4x4xi64>) -> tensor<4x4xi8> {
  %0 = tensor.empty() : tensor<4x4xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xi8>) -> tensor<4x4xi8>
  return %1 : tensor<4x4xi8>
}

// -----
// CHECK-LABEL: @test_normalize_div_uint(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins({{.*}}, {{.*}} : tensor<1024xi16>, tensor<1024xi16>) outs({{.*}} : tensor<1024xi16>) -> tensor<1024xi16>
module {
  func.func @test_normalize_div_uint(%arg0: tensor<1024xi16>, %arg1: tensor<1024xi16>) -> tensor<1024xi16> {
    %0 = tensor.empty() : tensor<1024xi16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : tensor<1024xi16>, tensor<1024xi16>) outs(%0 : tensor<1024xi16>) -> tensor<1024xi16>
    return %1 : tensor<1024xi16>
  }
}