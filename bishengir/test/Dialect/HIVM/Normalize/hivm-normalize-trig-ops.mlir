// RUN: bishengir-opt --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeSin_hivm_vsin_f32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG: %[[PI_REC:.*]] = arith.constant 0.318309873 : f32
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[OUT:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[DIV_PI:.*]] = hivm.hir.vmul ins(%[[ARG0]], %[[PI_REC]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ROUND:.*]] = hivm.hir.vcast ins(%[[DIV_PI]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) round_mode = <round> -> tensor<4xf32>
// CHECK: hivm.hir.vsub
// CHECK: hivm.hir.vmul ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SIGN_PRE:.*]] = hivm.hir.vmul ins(%[[ROUND]], %[[HALF]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SIGN_FLOOR:.*]] = hivm.hir.vcast ins(%[[SIGN_PRE]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) round_mode = <floor> -> tensor<4xf32>
// CHECK: %[[RES:.*]] = hivm.hir.vmul ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>) outs(%[[OUT]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[RES]] : tensor<4xf32>
// CHECK-NOT: hivm.hir.vsin
func.func @test_NormalizeSin_hivm_vsin_f32(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = hivm.hir.vsin ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeSin_hivm_vsin_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf16>) -> tensor<4xf16> {
// CHECK: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[CAST_IN:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<4xf16>) outs(%[[EMPTY_F32]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ROUND_INPUT:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) round_mode = <round> -> tensor<4xf32>
// CHECK: %[[SIGN_FLOOR:.*]] = hivm.hir.vcast
// CHECK-SAME: round_mode = <floor>
// CHECK: %[[EMPTY_F16:.*]] = tensor.empty() : tensor<4xf16>
// CHECK-NEXT: %[[CAST_OUT:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4xf32>) outs(%[[EMPTY_F16]] : tensor<4xf16>) round_mode = <round> -> tensor<4xf16>
// CHECK-NEXT: return %[[CAST_OUT]] : tensor<4xf16>
// CHECK-NOT: hivm.hir.vsin
func.func @test_NormalizeSin_hivm_vsin_f16(%arg0 : tensor<4xf16>) -> tensor<4xf16> {
  %0 = tensor.empty() : tensor<4xf16>
  %1 = hivm.hir.vsin ins(%arg0 : tensor<4xf16>) outs(%0 : tensor<4xf16>) -> tensor<4xf16>
  return %1 : tensor<4xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCos_hivm_vcos_f32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG: %[[PI_REC:.*]] = arith.constant 0.318309873 : f32
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[HALF_PI:.*]] = arith.constant 1.57079637 : f32
// CHECK: %[[OUT:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[DIV_PI:.*]] = hivm.hir.vmul ins(%[[ARG0]], %[[PI_REC]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ROUND_INPUT:.*]] = hivm.hir.vadd ins(%[[DIV_PI]], %[[HALF]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ROUND:.*]] = hivm.hir.vcast ins(%[[ROUND_INPUT]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) round_mode = <round> -> tensor<4xf32>
// CHECK: %[[NORM:.*]] = hivm.hir.vadd ins(%{{.*}}, %[[HALF_PI]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SIGN_PRE:.*]] = hivm.hir.vmul ins(%[[ROUND]], %[[HALF]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SIGN_FLOOR:.*]] = hivm.hir.vcast ins(%[[SIGN_PRE]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) round_mode = <floor> -> tensor<4xf32>
// CHECK: %[[RES:.*]] = hivm.hir.vmul ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>) outs(%[[OUT]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[RES]] : tensor<4xf32>
// CHECK-NOT: hivm.hir.vcos
func.func @test_NormalizeCos_hivm_vcos_f32(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = hivm.hir.vcos ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCos_hivm_vcos_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf16>) -> tensor<4xf16> {
// CHECK: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[CAST_IN:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<4xf16>) outs(%[[EMPTY_F32]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ROUND_INPUT:.*]] = hivm.hir.vadd
// CHECK: %[[ROUND:.*]] = hivm.hir.vcast ins(%[[ROUND_INPUT]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) round_mode = <round> -> tensor<4xf32>
// CHECK: %[[SIGN_FLOOR:.*]] = hivm.hir.vcast
// CHECK-SAME: round_mode = <floor>
// CHECK: %[[EMPTY_F16:.*]] = tensor.empty() : tensor<4xf16>
// CHECK-NEXT: %[[CAST_OUT:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4xf32>) outs(%[[EMPTY_F16]] : tensor<4xf16>) round_mode = <round> -> tensor<4xf16>
// CHECK-NEXT: return %[[CAST_OUT]] : tensor<4xf16>
// CHECK-NOT: hivm.hir.vcos
func.func @test_NormalizeCos_hivm_vcos_f16(%arg0 : tensor<4xf16>) -> tensor<4xf16> {
  %0 = tensor.empty() : tensor<4xf16>
  %1 = hivm.hir.vcos ins(%arg0 : tensor<4xf16>) outs(%0 : tensor<4xf16>) -> tensor<4xf16>
  return %1 : tensor<4xf16>
}
