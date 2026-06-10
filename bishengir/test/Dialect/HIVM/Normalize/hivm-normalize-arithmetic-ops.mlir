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
// CHECK: %[[EMPTY_F16_0:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[EMPTY_F32_0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CAST_IN:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xf16>) outs(%[[EMPTY_F32_0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY_F32_1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CAST_DST:.*]] = hivm.hir.vcast ins(%[[EMPTY_F16_0]] : tensor<16xf16>) outs(%[[EMPTY_F32_1]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY_F32_2:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[SQRT:.*]] = hivm.hir.vsqrt ins(%[[CAST_IN]] : tensor<16xf32>) outs(%[[EMPTY_F32_2]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[REC:.*]] = hivm.hir.vrec ins(%[[SQRT]] : tensor<16xf32>) outs(%[[CAST_DST]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY_F16_1:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[CAST_BACK:.*]] = hivm.hir.vcast ins(%[[REC]] : tensor<16xf32>) outs(%[[EMPTY_F16_1]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: return %[[CAST_BACK]]
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
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<5x1xf16>) outs(%[[EMPTY0]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[REC:.*]] = hivm.hir.vrec ins(%[[CAST0]] : tensor<5x1xf32>) outs(%[[EMPTY1]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[REC]] : tensor<5x1xf32>) outs(%[[EMPTY2]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[CAST1]]
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
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x14336xf16>
// CHECK: %[[EMPTY0:.*]] = tensor.empty(%[[DIM0]]) : tensor<?x14336xf32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<?x14336xf16>) outs(%[[EMPTY0]] : tensor<?x14336xf32>) -> tensor<?x14336xf32>
// CHECK: %[[DIM1:.*]] = tensor.dim %[[CAST0]], %[[C0]] : tensor<?x14336xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty(%[[DIM1]]) : tensor<?x14336xf32>
// CHECK: %[[REC:.*]] = hivm.hir.vrec ins(%[[CAST0]] : tensor<?x14336xf32>) outs(%[[EMPTY1]] : tensor<?x14336xf32>) -> tensor<?x14336xf32>
// CHECK: %[[DIM2:.*]] = tensor.dim %[[REC]], %[[C0]] : tensor<?x14336xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty(%[[DIM2]]) : tensor<?x14336xf16>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[REC]] : tensor<?x14336xf32>) outs(%[[EMPTY2]] : tensor<?x14336xf16>) -> tensor<?x14336xf16>
// CHECK: return %[[CAST1]]
func.func @test_NormalizeDivVSToRec_hivm_div_dynshape_to_hivm_rec(
  %src : tensor<?x14336xf16>, %dst : tensor<?x14336xf16>) -> tensor<?x14336xf16> {
  %cst = arith.constant 1.000000e+00 : f16
  %ret = hivm.hir.vdiv ins(%cst, %src : f16, tensor<?x14336xf16>) outs(%dst : tensor<?x14336xf16>) -> tensor<?x14336xf16>
  return %ret : tensor<?x14336xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVPowi_hivm_vpow_i16
// CHECK: hivm.hir.vcast ins(%arg0 : tensor<4x2x32xi16>) outs({{.*}} : tensor<4x2x32xf32>) round_mode = <trunc> -> tensor<4x2x32xf32>
// CHECK: hivm.hir.vcast ins(%arg1 : tensor<4x2x32xi16>) outs({{.*}} : tensor<4x2x32xf32>) round_mode = <trunc> -> tensor<4x2x32xf32>
// CHECK-NOT: hivm.hir.vpow
  //        ^ VPowi widens to f32 and produces vpow(f32), which is then
  //          further decomposed by NormalizeVPowfOp into exp + log + select.
// CHECK: hivm.hir.vcast ins({{.*}} : tensor<4x2x32xf32>) outs({{.*}} : tensor<4x2x32xi32>) round_mode = <trunc> -> tensor<4x2x32xi32>
// CHECK: hivm.hir.vcast ins({{.*}} : tensor<4x2x32xi32>) outs({{.*}} : tensor<4x2x32xi16>) round_mode = <truncwithoverflow> -> tensor<4x2x32xi16>
// CHECK: return
func.func @test_NormalizeVPowi_hivm_vpow_i16(%arg0 : tensor<4x2x32xi16>, %arg1 : tensor<4x2x32xi16>) -> tensor<4x2x32xi16> {
  %0 = tensor.empty() : tensor<4x2x32xi16>
  %1 = hivm.hir.vpow ins(%arg0, %arg1 : tensor<4x2x32xi16>, tensor<4x2x32xi16>) outs(%0 : tensor<4x2x32xi16>) -> tensor<4x2x32xi16>
  return %1 : tensor<4x2x32xi16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeSubVSToVMulAndVAdd_hivm
// CHECK: %[[NEG1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %{{.*}} = tensor.empty() : tensor<16xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%arg0, %[[NEG1]] : tensor<16xf32>, f32) outs(%[[EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[ADD:.*]] = hivm.hir.vadd ins(%[[MUL]], %arg1 : tensor<16xf32>, f32) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[ADD]]
func.func @test_NormalizeSubVSToVMulAndVAdd_hivm(%arg0: tensor<16xf32>, %arg1: f32) -> tensor<16xf32> {
  %0 = tensor.empty() : tensor<16xf32>
  %1 = tensor.empty() : tensor<16xf32>
  %2 = hivm.hir.vbrc ins(%arg1 : f32) outs(%1 : tensor<16xf32>) -> tensor<16xf32>
  %3 = hivm.hir.vsub ins(%2, %arg0 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  return %3 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVPowi_hivm_vpow_i8
// CHECK: hivm.hir.vcast ins(%arg0 : tensor<4x2x32xi8>) outs({{.*}} : tensor<4x2x32xf16>) -> tensor<4x2x32xf16>
// CHECK: hivm.hir.vcast ins({{.*}} : tensor<4x2x32xf16>) outs({{.*}} : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
// CHECK: hivm.hir.vcast ins(%arg1 : tensor<4x2x32xi8>) outs({{.*}} : tensor<4x2x32xf16>) -> tensor<4x2x32xf16>
// CHECK: hivm.hir.vcast ins({{.*}} : tensor<4x2x32xf16>) outs({{.*}} : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
// CHECK-NOT: hivm.hir.vpow
  //        ^ VPowi widens to f32 and produces vpow(f32), which is then
  //          further decomposed by NormalizeVPowfOp into exp + log + select.
// CHECK: hivm.hir.vcast ins({{.*}} : tensor<4x2x32xf32>) outs({{.*}} : tensor<4x2x32xi32>) round_mode = <trunc> -> tensor<4x2x32xi32>
// CHECK: hivm.hir.vcast ins({{.*}} : tensor<4x2x32xi32>) outs({{.*}} : tensor<4x2x32xi8>) round_mode = <truncwithoverflow> -> tensor<4x2x32xi8>
// CHECK: return
func.func @test_NormalizeVPowi_hivm_vpow_i8(%arg0 : tensor<4x2x32xi8>, %arg1 : tensor<4x2x32xi8>) -> tensor<4x2x32xi8> {
  %0 = tensor.empty() : tensor<4x2x32xi8>
  %1 = hivm.hir.vpow ins(%arg0, %arg1 : tensor<4x2x32xi8>, tensor<4x2x32xi8>) outs(%0 : tensor<4x2x32xi8>) -> tensor<4x2x32xi8>
  return %1 : tensor<4x2x32xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeSubVSToVMulAndVAdd_hivm_zero_scalar_ascend950
// CHECK: %[[NEG1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%arg0, %[[NEG1]] : tensor<16xf32>, f32) outs(%[[EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK-NOT: hivm.hir.vadd
// CHECK: return %[[MUL]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeSubVSToVMulAndVAdd_hivm_zero_scalar_ascend950(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16xf32>
    %1 = tensor.empty() : tensor<16xf32>
    %2 = hivm.hir.vbrc ins(%c0 : f32) outs(%1 : tensor<16xf32>) -> tensor<16xf32>
    %3 = hivm.hir.vsub ins(%2, %arg0 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    return %3 : tensor<16xf32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMulExt_hivm_vmulext_i32_high_bits
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x2xi32>, %[[ARG1:.*]]: tensor<4x2xi32>)
// CHECK: %[[C32:.*]] = arith.constant 32 : i64
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<4x2xi64>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<4x2xi32>) outs(%[[EMPTY0]] : tensor<4x2xi64>) -> tensor<4x2xi64>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<4x2xi64>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[ARG1]] : tensor<4x2xi32>) outs(%[[EMPTY1]] : tensor<4x2xi64>) -> tensor<4x2xi64>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<4x2xi64>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%[[CAST0]], %[[CAST1]] : tensor<4x2xi64>, tensor<4x2xi64>) outs(%[[EMPTY2]] : tensor<4x2xi64>) -> tensor<4x2xi64>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<4x2xi64>
// CHECK: %[[SHR:.*]] = hivm.hir.vshr ins(%[[MUL]], %[[C32]] : tensor<4x2xi64>, i64) outs(%[[EMPTY3]] : tensor<4x2xi64>) -> tensor<4x2xi64>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<4x2xi32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[SHR]] : tensor<4x2xi64>) outs(%[[EMPTY4]] : tensor<4x2xi32>) round_mode = <truncwithoverflow> -> tensor<4x2xi32>
// CHECK: return %[[RES]]
func.func @test_NormalizeMulExt_hivm_vmulext_i32_high_bits(%arg0: tensor<4x2xi32>, %arg1: tensor<4x2xi32>) -> tensor<4x2xi32> {
  %0 = tensor.empty() : tensor<4x2xi32>
  %1 = tensor.empty() : tensor<4x2xi32>
  %2:2 = hivm.hir.vmulext ins(%arg0, %arg1 : tensor<4x2xi32>, tensor<4x2xi32>) outs(%0, %1 : tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>, tensor<4x2xi32>
  return %2#1 : tensor<4x2xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_hivm
// CHECK: %[[LHS_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<48xi32>) outs(%[[LHS_EMPTY]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[RHS_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<48xi32>) outs(%[[RHS_EMPTY]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<48xf32>, tensor<48xf32>) outs(%[[DIV_EMPTY]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES_EMPTY:.*]] = tensor.empty() : tensor<48xi32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<48xf32>) outs(%[[RES_EMPTY]] : tensor<48xi32>) round_mode = <trunc> -> tensor<48xi32>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivSI_hivm(%arg0: tensor<48xi32>, %arg1: tensor<48xi32>) -> tensor<48xi32> {
    %0 = tensor.empty() : tensor<48xi32>
    %1 = hivm.hir.vdiv ins(%arg0, %arg1 : tensor<48xi32>, tensor<48xi32>) outs(%0 : tensor<48xi32>) isSigned = true -> tensor<48xi32>
    return %1 : tensor<48xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMulExt_hivm_vmulext_i32_low_bits
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x2xi32>, %[[ARG1:.*]]: tensor<4x2xi32>)
// CHECK: %[[C32:.*]] = arith.constant 32 : i64
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<4x2xi64>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<4x2xi32>) outs(%[[EMPTY0]] : tensor<4x2xi64>) -> tensor<4x2xi64>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<4x2xi64>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[ARG1]] : tensor<4x2xi32>) outs(%[[EMPTY1]] : tensor<4x2xi64>) -> tensor<4x2xi64>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<4x2xi64>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%[[CAST0]], %[[CAST1]] : tensor<4x2xi64>, tensor<4x2xi64>) outs(%[[EMPTY2]] : tensor<4x2xi64>) -> tensor<4x2xi64>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<4x2xi64>
// CHECK: %[[SHL:.*]] = hivm.hir.vshl ins(%[[MUL]], %[[C32]] : tensor<4x2xi64>, i64) outs(%[[EMPTY3]] : tensor<4x2xi64>) -> tensor<4x2xi64>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<4x2xi64>
// CHECK: %[[SHR:.*]] = hivm.hir.vshr ins(%[[SHL]], %[[C32]] : tensor<4x2xi64>, i64) outs(%[[EMPTY4]] : tensor<4x2xi64>) -> tensor<4x2xi64>
// CHECK: %[[EMPTY5:.*]] = tensor.empty() : tensor<4x2xi32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[SHR]] : tensor<4x2xi64>) outs(%[[EMPTY5]] : tensor<4x2xi32>) round_mode = <truncwithoverflow> -> tensor<4x2xi32>
// CHECK: return %[[RES]]
func.func @test_NormalizeMulExt_hivm_vmulext_i32_low_bits(%arg0: tensor<4x2xi32>, %arg1: tensor<4x2xi32>) -> tensor<4x2xi32> {
  %0 = tensor.empty() : tensor<4x2xi32>
  %1 = tensor.empty() : tensor<4x2xi32>
  %2:2 = hivm.hir.vmulext ins(%arg0, %arg1 : tensor<4x2xi32>, tensor<4x2xi32>) outs(%0, %1 : tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>, tensor<4x2xi32>
  return %2#0 : tensor<4x2xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_vs_hivm
// CHECK: %[[LHS_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<48xi32>) outs(%[[LHS_EMPTY]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[RHS:.*]] = arith.sitofp %arg1 : i32 to f32
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<48xf32>, f32) outs(%[[DIV_EMPTY]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES_EMPTY:.*]] = tensor.empty() : tensor<48xi32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<48xf32>) outs(%[[RES_EMPTY]] : tensor<48xi32>) round_mode = <trunc> -> tensor<48xi32>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivSI_vs_hivm(%arg0: tensor<48xi32>, %arg1: i32) -> tensor<48xi32> {
    %0 = tensor.empty() : tensor<48xi32>
    %1 = hivm.hir.vdiv ins(%arg0, %arg1 : tensor<48xi32>, i32) outs(%0 : tensor<48xi32>) isSigned = true -> tensor<48xi32>
    return %1 : tensor<48xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_sv_hivm
// CHECK: %[[FROM:.*]] = tensor.from_elements %arg0 : tensor<1xi32>
// CHECK: %[[CAST0_EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[FROM]] : tensor<1xi32>) outs(%[[CAST0_EMPTY]] : tensor<1xf32>) round_mode = <trunc> -> tensor<1xf32>
// CHECK: %[[EXTRACT:.*]] = tensor.extract %[[CAST0]][%{{.*}}] : tensor<1xf32>
// CHECK: %[[BRC_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[EXTRACT]] : f32) outs(%[[BRC_EMPTY]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RHS_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<48xi32>) outs(%[[RHS_EMPTY]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[BRC]], %[[RHS]] : tensor<48xf32>, tensor<48xf32>) outs(%[[DIV_EMPTY]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES_EMPTY:.*]] = tensor.empty() : tensor<48xi32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<48xf32>) outs(%[[RES_EMPTY]] : tensor<48xi32>) round_mode = <trunc> -> tensor<48xi32>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivSI_sv_hivm(%arg0: i32, %arg1: tensor<48xi32>) -> tensor<48xi32> {
    %0 = tensor.empty() : tensor<48xi32>
    %1 = tensor.empty() : tensor<48xi32>
    %2 = hivm.hir.vbrc ins(%arg0 : i32) outs(%1 : tensor<48xi32>) -> tensor<48xi32>
    %3 = hivm.hir.vdiv ins(%2, %arg1 : tensor<48xi32>, tensor<48xi32>) outs(%0 : tensor<48xi32>) isSigned = true -> tensor<48xi32>
    return %3 : tensor<48xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_i8_hivm
// CHECK: %[[CSTMIN:.*]] = arith.constant -128 : i16
// CHECK: %[[CSTMAX:.*]] = arith.constant 128 : i16
// CHECK: %[[LHS_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1024xi8>) outs(%[[LHS_EMPTY]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[RHS_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<1024xi8>) outs(%[[RHS_EMPTY]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<1024xi16>, tensor<1024xi16>) outs(%[[DIV_EMPTY]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[MASK:.*]] = hivm.hir.vcmp ins(%[[DIV]], %[[CSTMAX]] : tensor<1024xi16>, i16) outs(%[[MASK_EMPTY:.*]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[MIN_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[MIN:.*]] = hivm.hir.vbrc ins(%[[CSTMIN]] : i16) outs(%[[MIN_EMPTY]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[SELECT:.*]] = hivm.hir.vsel ins(%[[MASK]], %[[MIN]], %[[DIV]] : tensor<1024xi1>, tensor<1024xi16>, tensor<1024xi16>) outs(%[[DIV]] : tensor<1024xi16>) -> tensor<1024xi16>
// CHECK: %[[F16_EMPTY:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[F16:.*]] = hivm.hir.vcast ins(%[[SELECT]] : tensor<1024xi16>) outs(%[[F16_EMPTY]] : tensor<1024xf16>) round_mode = <trunc> -> tensor<1024xf16>
// CHECK: %[[I8_EMPTY:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[F16]] : tensor<1024xf16>) outs(%[[I8_EMPTY]] : tensor<1024xi8>) round_mode = <trunc> -> tensor<1024xi8>
// CHECK: return %[[RES]]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  func.func @test_NormalizeDivSI_i8_hivm(%arg0: tensor<1024xi8>, %arg1: tensor<1024xi8>) -> tensor<1024xi8> {
    %0 = tensor.empty() : tensor<1024xi8>
    %1 = hivm.hir.vdiv ins(%arg0, %arg1 : tensor<1024xi8>, tensor<1024xi8>) outs(%0 : tensor<1024xi8>) isSigned = true -> tensor<1024xi8>
    return %1 : tensor<1024xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivSI_i8_hivm_membase
// CHECK: %[[LHS_F16_EMPTY:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[LHS_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1024xi8>) outs(%[[LHS_F16_EMPTY]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[LHS_F32_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LHS_F32:.*]] = hivm.hir.vcast ins(%[[LHS_F16]] : tensor<1024xf16>) outs(%[[LHS_F32_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RHS_F16_EMPTY:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[RHS_F16:.*]] = hivm.hir.vcast ins(%arg1 : tensor<1024xi8>) outs(%[[RHS_F16_EMPTY]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[RHS_F32_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RHS_F32:.*]] = hivm.hir.vcast ins(%[[RHS_F16]] : tensor<1024xf16>) outs(%[[RHS_F32_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS_F32]], %[[RHS_F32]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[DIV_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[I32_EMPTY:.*]] = tensor.empty() : tensor<1024xi32>
// CHECK: %[[I32:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<1024xf32>) outs(%[[I32_EMPTY]] : tensor<1024xi32>) round_mode = <trunc> -> tensor<1024xi32>
// CHECK: %[[I8_EMPTY:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[I32]] : tensor<1024xi32>) outs(%[[I8_EMPTY]] : tensor<1024xi8>) round_mode = <truncwithoverflow> -> tensor<1024xi8>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivSI_i8_hivm_membase(%arg0: tensor<1024xi8>, %arg1: tensor<1024xi8>) -> tensor<1024xi8> {
    %0 = tensor.empty() : tensor<1024xi8>
    %1 = hivm.hir.vdiv ins(%arg0, %arg1 : tensor<1024xi8>, tensor<1024xi8>) outs(%0 : tensor<1024xi8>) isSigned = true -> tensor<1024xi8>
    return %1 : tensor<1024xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_hivm
// CHECK: %[[LHS_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<48xi32>) outs(%[[LHS_EMPTY]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[RHS_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<48xi32>) outs(%[[RHS_EMPTY]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<48xf32>, tensor<48xf32>) outs(%[[DIV_EMPTY]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES_EMPTY:.*]] = tensor.empty() : tensor<48xi32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<48xf32>) outs(%[[RES_EMPTY]] : tensor<48xi32>) round_mode = <trunc> -> tensor<48xi32>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivUI_hivm(%arg0: tensor<48xi32>, %arg1: tensor<48xi32>) -> tensor<48xi32> {
    %0 = tensor.empty() : tensor<48xi32>
    %1 = hivm.hir.vdiv ins(%arg0, %arg1 : tensor<48xi32>, tensor<48xi32>) outs(%0 : tensor<48xi32>) isSigned = false -> tensor<48xi32>
    return %1 : tensor<48xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_i8_hivm_membase
// CHECK: %[[LHS_F16_EMPTY:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[LHS_F16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1024xi8>) outs(%[[LHS_F16_EMPTY]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[LHS_F32_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LHS_F32:.*]] = hivm.hir.vcast ins(%[[LHS_F16]] : tensor<1024xf16>) outs(%[[LHS_F32_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RHS_F16_EMPTY:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[RHS_F16:.*]] = hivm.hir.vcast ins(%arg1 : tensor<1024xi8>) outs(%[[RHS_F16_EMPTY]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[RHS_F32_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RHS_F32:.*]] = hivm.hir.vcast ins(%[[RHS_F16]] : tensor<1024xf16>) outs(%[[RHS_F32_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS_F32]], %[[RHS_F32]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[DIV_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[I32_EMPTY:.*]] = tensor.empty() : tensor<1024xi32>
// CHECK: %[[I32:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<1024xf32>) outs(%[[I32_EMPTY]] : tensor<1024xi32>) round_mode = <trunc> -> tensor<1024xi32>
// CHECK: %[[I8_EMPTY:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[I32]] : tensor<1024xi32>) outs(%[[I8_EMPTY]] : tensor<1024xi8>) round_mode = <truncwithoverflow> -> tensor<1024xi8>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivUI_i8_hivm_membase(%arg0: tensor<1024xi8>, %arg1: tensor<1024xi8>) -> tensor<1024xi8> {
    %0 = tensor.empty() : tensor<1024xi8>
    %1 = hivm.hir.vdiv ins(%arg0, %arg1 : tensor<1024xi8>, tensor<1024xi8>) outs(%0 : tensor<1024xi8>) isSigned = false -> tensor<1024xi8>
    return %1 : tensor<1024xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_i8_hivm_regbase
// CHECK: %[[LHS_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<1024xi8>) outs(%[[LHS_EMPTY]] : tensor<1024xi16>) cast = <cast_unsigned> -> tensor<1024xi16>
// CHECK: %[[RHS_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<1024xi8>) outs(%[[RHS_EMPTY]] : tensor<1024xi16>) cast = <cast_unsigned> -> tensor<1024xi16>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<1024xi16>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<1024xi16>, tensor<1024xi16>) outs(%[[DIV_EMPTY]] : tensor<1024xi16>) isSigned = false -> tensor<1024xi16>
// CHECK: %[[RES_EMPTY:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<1024xi16>) outs(%[[RES_EMPTY]] : tensor<1024xi8>) round_mode = <trunc> -> tensor<1024xi8>
// CHECK: return %[[RES]]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  func.func @test_NormalizeDivUI_i8_hivm_regbase(%arg0: tensor<1024xi8>, %arg1: tensor<1024xi8>) -> tensor<1024xi8> {
    %0 = tensor.empty() : tensor<1024xi8>
    %1 = hivm.hir.vdiv ins(%arg0, %arg1 : tensor<1024xi8>, tensor<1024xi8>) outs(%0 : tensor<1024xi8>) isSigned = false -> tensor<1024xi8>
    return %1 : tensor<1024xi8>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_vs_hivm
// CHECK: %[[LHS_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[LHS:.*]] = hivm.hir.vcast ins(%arg0 : tensor<48xi32>) outs(%[[LHS_EMPTY]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[RHS:.*]] = arith.sitofp %arg1 : i32 to f32
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LHS]], %[[RHS]] : tensor<48xf32>, f32) outs(%[[DIV_EMPTY]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES_EMPTY:.*]] = tensor.empty() : tensor<48xi32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<48xf32>) outs(%[[RES_EMPTY]] : tensor<48xi32>) round_mode = <trunc> -> tensor<48xi32>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivUI_vs_hivm(%arg0: tensor<48xi32>, %arg1: i32) -> tensor<48xi32> {
    %0 = tensor.empty() : tensor<48xi32>
    %1 = hivm.hir.vdiv ins(%arg0, %arg1 : tensor<48xi32>, i32) outs(%0 : tensor<48xi32>) isSigned = false -> tensor<48xi32>
    return %1 : tensor<48xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeDivUI_sv_hivm
// CHECK: %[[FROM:.*]] = tensor.from_elements %arg0 : tensor<1xi32>
// CHECK: %[[CAST0_EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[FROM]] : tensor<1xi32>) outs(%[[CAST0_EMPTY]] : tensor<1xf32>) round_mode = <trunc> -> tensor<1xf32>
// CHECK: %[[EXTRACT:.*]] = tensor.extract %[[CAST0]][%{{.*}}] : tensor<1xf32>
// CHECK: %[[BRC_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[EXTRACT]] : f32) outs(%[[BRC_EMPTY]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RHS_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[RHS:.*]] = hivm.hir.vcast ins(%arg1 : tensor<48xi32>) outs(%[[RHS_EMPTY]] : tensor<48xf32>) round_mode = <trunc> -> tensor<48xf32>
// CHECK: %[[DIV_EMPTY:.*]] = tensor.empty() : tensor<48xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[BRC]], %[[RHS]] : tensor<48xf32>, tensor<48xf32>) outs(%[[DIV_EMPTY]] : tensor<48xf32>) -> tensor<48xf32>
// CHECK: %[[RES_EMPTY:.*]] = tensor.empty() : tensor<48xi32>
// CHECK: %[[RES:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<48xf32>) outs(%[[RES_EMPTY]] : tensor<48xi32>) round_mode = <trunc> -> tensor<48xi32>
// CHECK: return %[[RES]]
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @test_NormalizeDivUI_sv_hivm(%arg0: i32, %arg1: tensor<48xi32>) -> tensor<48xi32> {
    %0 = tensor.empty() : tensor<48xi32>
    %1 = tensor.empty() : tensor<48xi32>
    %2 = hivm.hir.vbrc ins(%arg0 : i32) outs(%1 : tensor<48xi32>) -> tensor<48xi32>
    %3 = hivm.hir.vdiv ins(%2, %arg1 : tensor<48xi32>, tensor<48xi32>) outs(%0 : tensor<48xi32>) isSigned = false -> tensor<48xi32>
    return %3 : tensor<48xi32>
  }
}
