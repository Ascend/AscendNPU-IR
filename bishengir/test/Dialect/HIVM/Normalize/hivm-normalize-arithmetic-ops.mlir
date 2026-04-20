// RUN: bishengir-opt --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeRSqrt_hivm_rsqrt_to_hivm_sqrt
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf32>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VSQRT:.*]] = hivm.hir.vsqrt ins(%[[ARG0]] : tensor<5x1xf32>) outs(%[[EMPTY1]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VREC:.*]] = hivm.hir.vrec ins(%[[VSQRT]] : tensor<5x1xf32>) outs(%[[EMPTY0]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: return %[[VREC]]
func.func @test_NormalizeRSqrt_hivm_rsqrt_to_hivm_sqrt(%arg0: tensor<5x1xf32>) -> tensor<5x1xf32> {
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = hivm.hir.vrsqrt ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %1 : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeRSqrt_hivm_rsqrt_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[VSQRT:.*]] = hivm.hir.vsqrt ins(%[[ARG0]] : tensor<16xf16>) outs(%[[EMPTY1]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[VREC:.*]] = hivm.hir.vrec ins(%[[VSQRT]] : tensor<16xf16>) outs(%[[EMPTY0]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: return %[[VREC]]
func.func @test_NormalizeRSqrt_hivm_rsqrt_f16(%arg0: tensor<16xf16>) -> tensor<16xf16> {
  %0 = tensor.empty() : tensor<16xf16>
  %1 = hivm.hir.vrsqrt ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xf16>) -> tensor<16xf16>
  return %1 : tensor<16xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeRSqrt_hivm_rsqrt_to_hivm_sqrt_dynshape
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x?xf32>, %[[ARG1:.*]]: index)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[EMPTY0:.*]] = tensor.empty(%[[ARG1]]) : tensor<5x?xf32>
// CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<5x?xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty(%[[DIM]]) : tensor<5x?xf32>
// CHECK: %[[VSQRT:.*]] = hivm.hir.vsqrt ins(%[[ARG0]] : tensor<5x?xf32>) outs(%[[EMPTY1]] : tensor<5x?xf32>) -> tensor<5x?xf32>
// CHECK: %[[VREC:.*]] = hivm.hir.vrec ins(%[[VSQRT]] : tensor<5x?xf32>) outs(%[[EMPTY0]] : tensor<5x?xf32>) -> tensor<5x?xf32>
// CHECK: return %[[VREC]]
func.func @test_NormalizeRSqrt_hivm_rsqrt_to_hivm_sqrt_dynshape(%s: tensor<5x?xf32>, %d : index) -> tensor<5x?xf32> {
  %0 = tensor.empty(%d) : tensor<5x?xf32>
  %1 = hivm.hir.vrsqrt ins(%s : tensor<5x?xf32>) outs(%0 : tensor<5x?xf32>) -> tensor<5x?xf32>
  return %1 : tensor<5x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMulRec_mul_div_by_one
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf16>, %[[ARG1:.*]]: tensor<5x1xf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[DIV0:.*]] = hivm.hir.vdiv ins(%[[ARG1]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY0]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[DIV1:.*]] = hivm.hir.vdiv ins(%[[DIV0]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY1]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[DIV1]]
func.func @test_NormalizeMulRec_mul_div_by_one(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
  %cst = arith.constant 1.000000e+00 : f16
  %0 = tensor.empty() : tensor<5x1xf16>
  %1 = hivm.hir.vdiv ins(%cst, %arg0 : f16, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  %2 = hivm.hir.vmul ins(%1, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  %3 = hivm.hir.vmul ins(%2, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  return %3 : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMulRec_mul_div_by_one_rec
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf16>, %[[ARG1:.*]]: tensor<5x1xf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[DIV0:.*]] = hivm.hir.vdiv ins(%[[ARG1]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY0]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[DIV1:.*]] = hivm.hir.vdiv ins(%[[DIV0]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY1]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[DIV1]]
func.func @test_NormalizeMulRec_mul_div_by_one_rec(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
  %0 = tensor.empty() : tensor<5x1xf16>
  %1 = hivm.hir.vrec ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  %2 = hivm.hir.vmul ins(%1, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  %3 = hivm.hir.vmul ins(%2, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  return %3 : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMulRec_mul_div_by_one_right
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf16>, %[[ARG1:.*]]: tensor<5x1xf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[DIV0:.*]] = hivm.hir.vdiv ins(%[[ARG1]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY0]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[DIV1:.*]] = hivm.hir.vdiv ins(%[[DIV0]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY1]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[DIV1]]
func.func @test_NormalizeMulRec_mul_div_by_one_right(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
  %cst = arith.constant 1.000000e+00 : f16
  %0 = tensor.empty() : tensor<5x1xf16>
  %1 = hivm.hir.vdiv ins(%cst, %arg0 : f16, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  %2 = hivm.hir.vmul ins(%arg1, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  %3 = hivm.hir.vmul ins(%1, %2 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  return %3 : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMulRec_mul_div_by_one_rec_right
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf16>, %[[ARG1:.*]]: tensor<5x1xf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[DIV0:.*]] = hivm.hir.vdiv ins(%[[ARG1]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY0]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[DIV1:.*]] = hivm.hir.vdiv ins(%[[DIV0]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[EMPTY1]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[DIV1]]
func.func @test_NormalizeMulRec_mul_div_by_one_rec_right(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
  %0 = tensor.empty() : tensor<5x1xf16>
  %1 = hivm.hir.vrec ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  %2 = hivm.hir.vmul ins(%arg1, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  %3 = hivm.hir.vmul ins(%1, %2 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  return %3 : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivVSToRec_hivm_div_to_hivm_rec
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf16>, %[[ARG1:.*]]: tensor<5x1xf16>)
// CHECK: %[[REC:.*]] = hivm.hir.vrec ins(%[[ARG0]] : tensor<5x1xf16>) outs(%[[ARG1]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[REC]]
func.func @test_NormalizeDivVSToRec_hivm_div_to_hivm_rec(
  %src : tensor<5x1xf16>, %dst : tensor<5x1xf16>) -> tensor<5x1xf16> {
  %cst = arith.constant 1.000000e+00 : f16
  %ret = hivm.hir.vdiv ins(%cst, %src : f16, tensor<5x1xf16>) outs(%dst : tensor<5x1xf16>) -> tensor<5x1xf16>
  return %ret : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivVSToRec_hivm_div_f32_no_rec
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf32>, %[[ARG1:.*]]: tensor<5x1xf32>)
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NOT: hivm.hir.vrec
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[CST]], %[[ARG0]] : f32, tensor<5x1xf32>) outs(%[[ARG1]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: return %[[DIV]]
func.func @test_NormalizeDivVSToRec_hivm_div_f32_no_rec(
  %src : tensor<5x1xf32>, %dst : tensor<5x1xf32>) -> tensor<5x1xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %ret = hivm.hir.vdiv ins(%cst, %src : f32, tensor<5x1xf32>) outs(%dst : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %ret : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivVSToRec_hivm_div_brc_f16_no_rec
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5x1xf16>, %[[ARG1:.*]]: tensor<5x1xf16>, %[[ARG2:.*]]: tensor<5x1xf16>)
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f16
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[CST]] : f16) outs(%[[ARG1]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK-NOT: hivm.hir.vrec
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[VBRC]], %[[ARG0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[ARG2]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[DIV]]
func.func @test_NormalizeDivVSToRec_hivm_div_brc_f16_no_rec(
  %src : tensor<5x1xf16>, %brcDst : tensor<5x1xf16>, %dst : tensor<5x1xf16>) -> tensor<5x1xf16> {
  %cst = arith.constant 1.000000e+00 : f16
  %brc = hivm.hir.vbrc ins(%cst : f16) outs(%brcDst : tensor<5x1xf16>) -> tensor<5x1xf16>
  %ret = hivm.hir.vdiv ins(%brc, %src : tensor<5x1xf16>, tensor<5x1xf16>) outs(%dst : tensor<5x1xf16>) -> tensor<5x1xf16>
  return %ret : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivVSToRec_hivm_div_dynshape_to_hivm_rec
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x14336xf16>, %[[ARG1:.*]]: tensor<?x14336xf16>)
// CHECK: %[[REC:.*]] = hivm.hir.vrec ins(%[[ARG0]] : tensor<?x14336xf16>) outs(%[[ARG1]] : tensor<?x14336xf16>) -> tensor<?x14336xf16>
// CHECK: return %[[REC]]
func.func @test_NormalizeDivVSToRec_hivm_div_dynshape_to_hivm_rec(
  %src : tensor<?x14336xf16>, %dst : tensor<?x14336xf16>) -> tensor<?x14336xf16> {
  %cst = arith.constant 1.000000e+00 : f16
  %ret = hivm.hir.vdiv ins(%cst, %src : f16, tensor<?x14336xf16>) outs(%dst : tensor<?x14336xf16>) -> tensor<?x14336xf16>
  return %ret : tensor<?x14336xf16>
}
