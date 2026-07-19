// REQUIRES: regbase
// TODO: enable when convert-hfusion-to-hivm will is migrated
// RUN: bishengir-opt --convert-hfusion-to-hivm --hivm-normalize-ops  %s -split-input-file -verify-diagnostics | FileCheck %s
// This file mirrors HFusion scalar-normalize source cases and checks the
// HFusion -> HIVM conversion and HIVM normalization path.

// normalize guard:
// These cases come from HFusion scalar-normalize tests, but their mapped HIVM
// ops have vector-only operand slots. HIVM normalize must keep those operands
// shaped so the converted IR stays legal.

func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_unary_absi(
    %dst: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_unary_absi(
  // CHECK: %[[CST:.+]] = arith.constant dense<5> : tensor<1xi32>
  // CHECK: %[[ABS:.+]] = hivm.hir.vabs ins(%[[CST]] : tensor<1xi32>) outs(%arg0 : tensor<1xi32>) -> tensor<1xi32>
  // CHECK: return %[[ABS]] : tensor<1xi32>
  %src = arith.constant dense<5> : tensor<1xi32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>}
      ins(%src : tensor<1xi32>) outs(%dst : tensor<1xi32>) -> tensor<1xi32>
  return %1 : tensor<1xi32>
}

// -----

// normalize guard: rank-0 shaped input is still shaped in HIVM.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_unary_absi_rank0(
    %dst: tensor<i32>) -> tensor<i32> {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_unary_absi_rank0(
  // CHECK: %[[CST:.+]] = arith.constant 5 : i32
  // CHECK: %[[TMP:.+]] = tensor.empty() : tensor<i32>
  // CHECK: %[[BRC:.+]] = hivm.hir.vbrc ins(%[[CST]] : i32) outs(%[[TMP]] : tensor<i32>) -> tensor<i32>
  // CHECK: %[[ABS:.+]] = hivm.hir.vabs ins(%[[BRC]] : tensor<i32>) outs(%arg0 : tensor<i32>) -> tensor<i32>
  // CHECK: return %[[ABS]] : tensor<i32>
  %src = arith.constant dense<5> : tensor<i32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>}
      ins(%src : tensor<i32>) outs(%dst : tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}

// -----

// normalize guard: operand 0 of hivm.hir.vmin is vector-only.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_binary(
    %src: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_binary(
  // CHECK: %[[CST:.+]] = arith.constant dense<5.000000e-01> : tensor<1xf32>
  // CHECK: %[[OUT:.+]] = tensor.empty() : tensor<1xf32>
  // CHECK: %[[RES:.+]] = hivm.hir.vmin ins(%[[CST]], %arg0 : tensor<1xf32>, tensor<1xf32>) outs(%[[OUT]] : tensor<1xf32>) -> tensor<1xf32>
  // CHECK: return %[[RES]] : tensor<1xf32>
  %cst = arith.constant dense<0.5> : tensor<1xf32>
  %dst = tensor.empty() : tensor<1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>}
      ins(%cst, %src : tensor<1xf32>, tensor<1xf32>)
      outs(%dst : tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----

// normalize case:
// operand 0 of hivm.hir.vmin must stay shaped, while operand 1 may be
// scalarized.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_binary_both_inputs(
    ) -> tensor<1xf32> {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_binary_both_inputs()
  // CHECK: %[[RHS:.+]] = arith.constant 2.500000e-01 : f32
  // CHECK: %[[LHS:.+]] = arith.constant dense<5.000000e-01> : tensor<1xf32>
  // CHECK: %[[OUT:.+]] = tensor.empty() : tensor<1xf32>
  // CHECK: %[[RES:.+]] = hivm.hir.vmin ins(%[[LHS]], %[[RHS]] : tensor<1xf32>, f32) outs(%[[OUT]] : tensor<1xf32>) -> tensor<1xf32>
  // CHECK: return %[[RES]] : tensor<1xf32>
  %lhs = arith.constant dense<0.5> : tensor<1xf32>
  %rhs = arith.constant dense<0.25> : tensor<1xf32>
  %dst = tensor.empty() : tensor<1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>}
      ins(%lhs, %rhs : tensor<1xf32>, tensor<1xf32>)
      outs(%dst : tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----

// normalize no-rewrite guard:
// tensor<1x1xf32> is not treated as scalar-like by the scalar-op template.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_binary_rank2_constant_no_rewrite(
    %src: tensor<1x1xf32>) -> tensor<1x1xf32> {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_binary_rank2_constant_no_rewrite(
  // CHECK: %[[CST:.+]] = arith.constant dense<5.000000e-01> : tensor<1x1xf32>
  // CHECK: %[[OUT:.+]] = tensor.empty() : tensor<1x1xf32>
  // CHECK: %[[RES:.+]] = hivm.hir.vmin ins(%[[CST]], %arg0 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%[[OUT]] : tensor<1x1xf32>) -> tensor<1x1xf32>
  // CHECK: return %[[RES]] : tensor<1x1xf32>
  %cst = arith.constant dense<0.5> : tensor<1x1xf32>
  %dst = tensor.empty() : tensor<1x1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>}
      ins(%cst, %src : tensor<1x1xf32>, tensor<1x1xf32>)
      outs(%dst : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %res : tensor<1x1xf32>
}

// -----

// normalize case:
// operand 1 of hivm.hir.vcmp may be scalarized, and vne is further normalized
// by the comparison patterns in `--hivm-normalize-ops`.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_compare(
    %src: tensor<1xi64>) -> tensor<1xi1> {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_compare(
  // CHECK: %[[CST:.+]] = arith.constant 5 : i64
  // CHECK: %[[OUT0:.+]] = tensor.empty() : tensor<1xi1>
  // CHECK: %[[OUT1:.+]] = tensor.empty() : tensor<1xi1>
  // CHECK: %[[CMP:.+]] = hivm.hir.vcmp ins(%arg0, %[[CST]] : tensor<1xi64>, i64) outs(%[[OUT1]] : tensor<1xi1>) -> tensor<1xi1>
  // CHECK: %[[NOT:.+]] = hivm.hir.vnot ins(%[[CMP]] : tensor<1xi1>) outs(%[[OUT0]] : tensor<1xi1>) -> tensor<1xi1>
  // CHECK: return %[[NOT]] : tensor<1xi1>
  %cst = arith.constant dense<5> : tensor<1xi64>
  %dst = tensor.empty() : tensor<1xi1>
  %res = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
      ins(%src, %cst : tensor<1xi64>, tensor<1xi64>)
      outs(%dst : tensor<1xi1>) -> tensor<1xi1>
  return %res : tensor<1xi1>
}

// -----

// normalize case: scalarize the compare rhs while keeping the lhs shaped.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_compare_veq(
    %src: tensor<1xi64>) -> tensor<1xi1> {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_compare_veq(
  // CHECK: %[[CST:.+]] = arith.constant 7 : i64
  // CHECK: %[[OUT:.+]] = tensor.empty() : tensor<1xi1>
  // CHECK: %[[CMP:.+]] = hivm.hir.vcmp ins(%arg0, %[[CST]] : tensor<1xi64>, i64) outs(%[[OUT]] : tensor<1xi1>) -> tensor<1xi1>
  // CHECK: return %[[CMP]] : tensor<1xi1>
  %cst = arith.constant dense<7> : tensor<1xi64>
  %dst = tensor.empty() : tensor<1xi1>
  %res = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
      ins(%src, %cst : tensor<1xi64>, tensor<1xi64>)
      outs(%dst : tensor<1xi1>) -> tensor<1xi1>
  return %res : tensor<1xi1>
}

// -----

// normalize case: vsel accepts scalar values for the data operands.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_select(
    %src1: memref<1xi1>, %src3: memref<1xi32>, %dst: memref<1xi32>) {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_select(
  // CHECK: %[[CST:.+]] = arith.constant 5 : i32
  // CHECK: hivm.hir.vsel ins(%arg0, %[[CST]], %arg1 : memref<1xi1>, i32, memref<1xi32>) outs(%arg2 : memref<1xi32>)
  // CHECK: return
  %src2 = arith.constant dense<5> : memref<1xi32>
  hfusion.select
      ins(%src1, %src2, %src3 : memref<1xi1>, memref<1xi32>, memref<1xi32>)
      outs(%dst : memref<1xi32>)
  return
}

// -----

// normalize case: both value operands of vsel may be scalarized.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_select_both_values(
    %cond: memref<1xi1>, %dst: memref<1xi32>) {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_select_both_values(
  // CHECK: %[[TRUE:.+]] = arith.constant 5 : i32
  // CHECK: %[[FALSE:.+]] = arith.constant 7 : i32
  // CHECK: hivm.hir.vsel ins(%arg0, %[[TRUE]], %[[FALSE]] : memref<1xi1>, i32, i32) outs(%arg1 : memref<1xi32>)
  // CHECK: return
  %trueValue = arith.constant dense<5> : memref<1xi32>
  %falseValue = arith.constant dense<7> : memref<1xi32>
  hfusion.select
      ins(%cond, %trueValue, %falseValue : memref<1xi1>, memref<1xi32>,
          memref<1xi32>)
      outs(%dst : memref<1xi32>)
  return
}

// -----

// normalize guard: hivm.hir.vcast keeps its source shaped.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_cast(
    ) -> tensor<f16> {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_cast()
  // CHECK: %[[CST:.+]] = arith.constant 0.869565188 : f32
  // CHECK: %[[OUT:.+]] = tensor.empty() : tensor<f16>
  // CHECK: %[[TMP:.+]] = tensor.empty() : tensor<f32>
  // CHECK: %[[BRC:.+]] = hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[TMP]] : tensor<f32>) -> tensor<f32>
  // CHECK: %[[CAST:.+]] = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%[[BRC]] : tensor<f32>) outs(%[[OUT]] : tensor<f16>) round_mode = <trunc> -> tensor<f16>
  // CHECK: return %[[CAST]] : tensor<f16>
  %src = arith.constant dense<0.86956521739130432> : tensor<f32>
  %dst = tensor.empty() : tensor<f16>
  %res = hfusion.cast {enable_overflow = true,
                       round_mode = #hfusion.round_mode<trunc>}
      ins(%src : tensor<f32>) outs(%dst : tensor<f16>) -> tensor<f16>
  return %res : tensor<f16>
}

// -----

// normalize guard: rank-1 cast source is still shaped in HIVM.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_cast_rank1(
    ) -> tensor<1xf16> {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_cast_rank1()
  // CHECK: %[[CST:.+]] = arith.constant dense<1.250000e+00> : tensor<1xf32>
  // CHECK: %[[OUT:.+]] = tensor.empty() : tensor<1xf16>
  // CHECK: %[[CAST:.+]] = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%[[CST]] : tensor<1xf32>) outs(%[[OUT]] : tensor<1xf16>) round_mode = <trunc> -> tensor<1xf16>
  // CHECK: return %[[CAST]] : tensor<1xf16>
  %src = arith.constant dense<1.25> : tensor<1xf32>
  %dst = tensor.empty() : tensor<1xf16>
  %res = hfusion.cast {enable_overflow = true,
                       round_mode = #hfusion.round_mode<trunc>}
      ins(%src : tensor<1xf32>) outs(%dst : tensor<1xf16>) -> tensor<1xf16>
  return %res : tensor<1xf16>
}

// -----

// normalize no-rewrite guard: non-constant rank-0 inputs are not scalarized.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_cast_nonconstant_rank0_no_rewrite(
    %src: tensor<f32>) -> tensor<f16> {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_cast_nonconstant_rank0_no_rewrite(
  // CHECK: %[[OUT:.+]] = tensor.empty() : tensor<f16>
  // CHECK: %[[TMP:.+]] = tensor.empty() : tensor<f32>
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %arg0[] : tensor<f32>
  // CHECK: %[[BRC:.+]] = hivm.hir.vbrc ins(%[[EXTRACT]] : f32) outs(%[[TMP]] : tensor<f32>) -> tensor<f32>
  // CHECK: %[[CAST:.+]] = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%[[BRC]] : tensor<f32>) outs(%[[OUT]] : tensor<f16>) round_mode = <trunc> -> tensor<f16>
  // CHECK: return %[[CAST]] : tensor<f16>
  %dst = tensor.empty() : tensor<f16>
  %res = hfusion.cast {enable_overflow = true,
                       round_mode = #hfusion.round_mode<trunc>}
      ins(%src : tensor<f32>) outs(%dst : tensor<f16>) -> tensor<f16>
  return %res : tensor<f16>
}

// -----

// normalize guard: hivm.hir.vabs keeps its source shaped.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_linalg_elemwise_unary_abs(
    %output: tensor<f32>) -> tensor<f32> {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_linalg_elemwise_unary_abs(
  // CHECK: %[[CST:.+]] = arith.constant 5.100000e+00 : f32
  // CHECK: %[[TMP:.+]] = tensor.empty() : tensor<f32>
  // CHECK: %[[BRC:.+]] = hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[TMP]] : tensor<f32>) -> tensor<f32>
  // CHECK: %[[ABS:.+]] = hivm.hir.vabs ins(%[[BRC]] : tensor<f32>) outs(%arg0 : tensor<f32>) -> tensor<f32>
  // CHECK: return %[[ABS]] : tensor<f32>
  %src = arith.constant dense<5.1> : tensor<f32>
  %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
      ins(%src : tensor<f32>) outs(%output : tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// normalize guard: operand 0 of hivm.hir.vadd is vector-only.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_linalg_elemwise_binary(
    %src: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_linalg_elemwise_binary(
  // CHECK: %[[CST:.+]] = arith.constant dense<5.000000e-01> : tensor<1xf32>
  // CHECK: %[[OUT:.+]] = tensor.empty() : tensor<1xf32>
  // CHECK: %[[RES:.+]] = hivm.hir.vadd ins(%[[CST]], %arg0 : tensor<1xf32>, tensor<1xf32>) outs(%[[OUT]] : tensor<1xf32>) -> tensor<1xf32>
  // CHECK: return %[[RES]] : tensor<1xf32>
  %cst = arith.constant dense<0.5> : tensor<1xf32>
  %dst = tensor.empty() : tensor<1xf32>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%cst, %src : tensor<1xf32>, tensor<1xf32>)
      outs(%dst : tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----

// normalize case:
// operand 0 of hivm.hir.vadd must stay shaped, while operand 1 may be
// scalarized.
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_linalg_elemwise_binary_both_inputs(
    ) -> tensor<1xf32> {
  // CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_linalg_elemwise_binary_both_inputs()
  // CHECK: %[[RHS:.+]] = arith.constant 2.500000e-01 : f32
  // CHECK: %[[LHS:.+]] = arith.constant dense<5.000000e-01> : tensor<1xf32>
  // CHECK: %[[OUT:.+]] = tensor.empty() : tensor<1xf32>
  // CHECK: %[[RES:.+]] = hivm.hir.vadd ins(%[[LHS]], %[[RHS]] : tensor<1xf32>, f32) outs(%[[OUT]] : tensor<1xf32>) -> tensor<1xf32>
  // CHECK: return %[[RES]] : tensor<1xf32>
  %lhs = arith.constant dense<0.5> : tensor<1xf32>
  %rhs = arith.constant dense<0.25> : tensor<1xf32>
  %dst = tensor.empty() : tensor<1xf32>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%lhs, %rhs : tensor<1xf32>, tensor<1xf32>)
      outs(%dst : tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// `linalg.broadcast` cases from the HFusion scalar-normalize test are
// intentionally excluded here. Their current HFusion-to-HIVM lowering creates
// `hivm.hir.vbrc` forms that fail verifier before the HIVM scalar-normalize
// rewrite gets a chance to run, so they do not belong in this verifier-clean
// scalar-normalize path test.
