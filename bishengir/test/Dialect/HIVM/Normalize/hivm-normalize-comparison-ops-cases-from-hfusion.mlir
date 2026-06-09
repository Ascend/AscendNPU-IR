// RUN: bishengir-opt --convert-hfusion-to-hivm --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s
// This test verifies the correctness of HIVM normalization transformations
// applied to HFusion operators after conversion to HIVM dialect.

// CHECK-LABEL: func.func @test_NormalizeCmpVne_normalize_compare_neq_to_Not_eq
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi64>, %[[arg1:.*]]: tensor<1024xi64>, %[[arg2:.*]]: tensor<1024xi1>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[vcmp:.]] = hivm.hir.vcmp ins(%[[arg0]], %[[arg1]] : tensor<1024xi64>, tensor<1024xi64>) outs(%[[empty]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[vnot:.]] = hivm.hir.vnot ins(%[[vcmp]] : tensor<1024xi1>) outs(%[[arg2]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: return %[[vnot]]
func.func @test_NormalizeCmpVne_normalize_compare_neq_to_Not_eq(
  %src1 : tensor<1024xi64>, %src2 : tensor<1024xi64>,  %dst : tensor<1024xi1>) ->  tensor<1024xi1> {
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%src1, %src2 : tensor<1024xi64>, tensor<1024xi64>)
    outs(%dst : tensor<1024xi1>)
    -> tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpVne_normalize_f32_compare_neq_to_Not_eq
// CHECK-SAME: (%[[arg0:.*]]: tensor<128xf32>, %[[arg1:.*]]: tensor<128xf32>, %[[arg2:.*]]: tensor<128xi1>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<128xi1>
// CHECK: %[[vcmp:.]] = hivm.hir.vcmp ins(%[[arg0]], %[[arg1]] : tensor<128xf32>, tensor<128xf32>) outs(%[[empty]] : tensor<128xi1>) -> tensor<128xi1>
// CHECK: %[[vnot:.]] = hivm.hir.vnot ins(%[[vcmp]] : tensor<128xi1>) outs(%[[arg2]] : tensor<128xi1>) -> tensor<128xi1>
// CHECK: return %[[vnot]]
func.func @test_NormalizeCmpVne_normalize_f32_compare_neq_to_Not_eq(
  %src1 : tensor<128xf32>, %src2 : tensor<128xf32>,  %dst : tensor<128xi1>) ->  tensor<128xi1> {
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%src1, %src2 : tensor<128xf32>, tensor<128xf32>)
    outs(%dst : tensor<128xi1>)
    -> tensor<128xi1>
  return %ret : tensor<128xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeXor_xori
// CHECK-SAME: (%[[arg0:.*]]: tensor<512xi16>, %[[arg1:.*]]: tensor<512xi16>)
// CHECK: %[[or_init:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[vor:.*]] = hivm.hir.vor ins(%[[arg0]], %[[arg1]] : tensor<512xi16>, tensor<512xi16>) outs(%[[or_init]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[and_init:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[vand:.*]] = hivm.hir.vand ins(%[[arg0]], %[[arg1]] : tensor<512xi16>, tensor<512xi16>) outs(%[[and_init]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[vnot:.*]] = hivm.hir.vnot ins(%[[vand]] : tensor<512xi16>) outs(%[[vand]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[res_init:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[res:.*]] = hivm.hir.vand ins(%[[vnot]], %[[vor]] : tensor<512xi16>, tensor<512xi16>) outs(%[[res_init]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: return %[[res]]
func.func @test_NormalizeXor_xori(%arg0: tensor<512xi16>, %arg1: tensor<512xi16>) -> tensor<512xi16> {
  %0 = tensor.empty() : tensor<512xi16>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%arg0, %arg1 : tensor<512xi16>, tensor<512xi16>) outs(%0 : tensor<512xi16>) -> tensor<512xi16>
  return %1 : tensor<512xi16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeI8I32Cmp_normalize_i8_hfusion_compare
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi8>, %[[arg1:.*]]: tensor<16xi8>)
// CHECK: %[[lhs_empty:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[lhs_cast:.*]] = hivm.hir.vcast ins(%[[arg0]] : tensor<16xi8>) outs(%[[lhs_empty]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[rhs_empty:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[rhs_cast:.*]] = hivm.hir.vcast ins(%[[arg1]] : tensor<16xi8>) outs(%[[rhs_empty]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[not_empty:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_empty:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[lhs_cast]], %[[rhs_cast]] : tensor<16xf16>, tensor<16xf16>) outs(%[[cmp_empty]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<16xi1>) outs(%[[not_empty]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[not]]
func.func @test_NormalizeI8I32Cmp_normalize_i8_hfusion_compare(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi1> {
  %dst1 = tensor.empty() : tensor<16xi1>
  %dst2 = tensor.empty() : tensor<16xi1>
  %res1 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
    outs(%dst1 : tensor<16xi1>) -> tensor<16xi1>
  return %res1 : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeI8I32Cmp_keep_unsigned_i8_compare
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi8>, %[[arg1:.*]]: tensor<16xi8>)
// CHECK-NOT: hivm.hir.vcast
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[arg0]], %[[arg1]] : tensor<16xi8>, tensor<16xi8>) outs(%{{.*}} : tensor<16xi1>) compare_mode = <lt> is_signed = false -> tensor<16xi1>
// CHECK: return %[[cmp]]
func.func @test_NormalizeI8I32Cmp_keep_unsigned_i8_compare(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi1> {
  %dst = tensor.empty() : tensor<16xi1>
  %res = hfusion.compare {compare_fn = #hfusion.compare_fn<vult>}
    ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
    outs(%dst : tensor<16xi1>) -> tensor<16xi1>
  return %res : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeI8I32Cmp_normalize_i32_hfusion_compare_vlt
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi32>, %[[arg1:.*]]: tensor<16xi32>)
// CHECK: %[[lhs_empty:.*]] = tensor.empty() : tensor<16xi64>
// CHECK: %[[lhs_cast:.*]] = hivm.hir.vcast ins(%[[arg0]] : tensor<16xi32>) outs(%[[lhs_empty]] : tensor<16xi64>) -> tensor<16xi64>
// CHECK: %[[rhs_empty:.*]] = tensor.empty() : tensor<16xi64>
// CHECK: %[[rhs_cast:.*]] = hivm.hir.vcast ins(%[[arg1]] : tensor<16xi32>) outs(%[[rhs_empty]] : tensor<16xi64>) -> tensor<16xi64>
// CHECK: %[[cmp_empty:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[lhs_cast]], %[[rhs_cast]] : tensor<16xi64>, tensor<16xi64>) outs(%[[cmp_empty]] : tensor<16xi1>) compare_mode = <lt> -> tensor<16xi1>
// CHECK: return %[[cmp]]
func.func @test_NormalizeI8I32Cmp_normalize_i32_hfusion_compare_vlt(%arg0: tensor<16xi32>, %arg1: tensor<16xi32>) -> tensor<16xi1> {
  %dst = tensor.empty() : tensor<16xi1>
  %res = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
    ins(%arg0, %arg1 : tensor<16xi32>, tensor<16xi32>)
    outs(%dst : tensor<16xi1>) -> tensor<16xi1>
  return %res : tensor<16xi1>
}

// -----

// Negative copied case:
// HFusion keeps `veq` unchanged here. After conversion to HIVM, this case should
// still stay out of the migrated `i32 -> i64` compare normalization.
// CHECK-LABEL: func.func @test_NormalizeI8I32Cmp_normalize_i32_hfusion_compare
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi32>, %[[arg1:.*]]: tensor<16xi32>)
// CHECK-NOT: hivm.hir.vcast
// CHECK: %[[ret0:.*]] = hivm.hir.vcmp ins(%[[arg0]], %[[arg1]] : tensor<16xi32>, tensor<16xi32>) outs(%{{.*}} : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[cmp_empty:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[arg0]], %[[arg0]] : tensor<16xi32>, tensor<16xi32>) outs(%[[cmp_empty]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<16xi1>) outs(%{{.*}} : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[ret0]], %[[not]]
func.func @test_NormalizeI8I32Cmp_normalize_i32_hfusion_compare(%arg0: tensor<16xi32>, %arg1: tensor<16xi32>) -> (tensor<16xi1>, tensor<16xi1>) {
  %dst1 = tensor.empty() : tensor<16xi1>
  %dst2 = tensor.empty() : tensor<16xi1>
  %res1 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
    ins(%arg0, %arg1 : tensor<16xi32>, tensor<16xi32>)
    outs(%dst1 : tensor<16xi1>) -> tensor<16xi1>
  %res2 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%arg0, %arg0 : tensor<16xi32>, tensor<16xi32>)
    outs(%dst2 : tensor<16xi1>) -> tensor<16xi1>
  return %res1, %res2 : tensor<16xi1>, tensor<16xi1>
}

// -----

// Negative copied case:
// HFusion keeps this dynamic-shape `veq` compare unchanged. After conversion to
// HIVM, it should still stay out of the migrated `i32 -> i64` compare rewrite.
// CHECK-LABEL: func.func @test_NormalizeI8I32Cmp_normalize_i32_hfusion_compare_dynamic(
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.vcmp ins(%[[arg0:.*]], %{{.*}} : tensor<?x?xi32>, tensor<?x?xi32>) outs(%{{.*}} : tensor<?x?xi1>) -> tensor<?x?xi1>
func.func @test_NormalizeI8I32Cmp_normalize_i32_hfusion_compare_dynamic(%arg0: tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<?x?xi1>) attributes {OperatorType = "Default", compute_capability = "", frontend_symbol = {input_0 = ["s93", "s94"], output_0 = ["s93", "s94"], output_1 = ["s93", "s94"]}, hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32319_i32 = arith.constant 32319 : i32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%arg0, %1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %3 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
  %4 = linalg.fill ins(%c32319_i32 : i32) outs(%3 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%2, %4 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%3 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %6 = tensor.empty(%dim, %dim_0) : tensor<?x?xi1>
  %7 = hfusion.compare {fun = #hfusion.compare_fn<veq>} ins(%arg0, %5 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%6 : tensor<?x?xi1>) -> tensor<?x?xi1>
  return %5, %7 : tensor<?x?xi32>, tensor<?x?xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeShiftI8ToI16_i8_shrsi
// CHECK-SAME: (%[[arg0:.*]]: tensor<32xi8>, %[[arg1:.*]]: tensor<32xi8>)
// CHECK: %[[lhs_f16_empty:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[lhs_f16:.*]] = hivm.hir.vcast ins(%[[arg0]] : tensor<32xi8>) outs(%[[lhs_f16_empty]] : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[lhs_empty:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[lhs_cast:.*]] = hivm.hir.vcast ins(%[[lhs_f16]] : tensor<32xf16>) outs(%[[lhs_empty]] : tensor<32xi16>) round_mode = <trunc> -> tensor<32xi16>
// CHECK: %[[rhs_f16_empty:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[rhs_f16:.*]] = hivm.hir.vcast ins(%[[arg1]] : tensor<32xi8>) outs(%[[rhs_f16_empty]] : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[rhs_empty:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[rhs_cast:.*]] = hivm.hir.vcast ins(%[[rhs_f16]] : tensor<32xf16>) outs(%[[rhs_empty]] : tensor<32xi16>) round_mode = <trunc> -> tensor<32xi16>
// CHECK: %[[shift_empty:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[shift:.*]] = hivm.hir.vshr ins(%[[lhs_cast]], %[[rhs_cast]] : tensor<32xi16>, tensor<32xi16>) outs(%[[shift_empty]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[res_empty:.*]] = tensor.empty() : tensor<32xi8>
// CHECK: %[[res:.*]] = hivm.hir.vcast ins(%[[shift]] : tensor<32xi16>) outs(%[[res_empty]] : tensor<32xi8>) round_mode = <truncwithoverflow> -> tensor<32xi8>
// CHECK: return %[[res]]
func.func @test_NormalizeShiftI8ToI16_i8_shrsi(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  %0 = tensor.empty() : tensor<32xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%arg0, %arg1 : tensor<32xi8>, tensor<32xi8>) outs(%0 : tensor<32xi8>) -> tensor<32xi8>
  return %1 : tensor<32xi8>
}
