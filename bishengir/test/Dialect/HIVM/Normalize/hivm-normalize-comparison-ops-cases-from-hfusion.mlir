// RUN: bishengir-opt --convert-hfusion-to-hivm --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s
// This file copies HFusion source cases for the comparison rewrites migrated to
// HIVM. It contains both:
// - positive copied cases: conversion to HIVM succeeds and the migrated HIVM
//   normalize pattern should rewrite the converted input.
// - negative copied cases: conversion to HIVM succeeds, but the converted input
//   should stay unchanged for the migrated HIVM normalize pattern.
//
// Source HFusion cases intentionally omitted from this file:
// - `test_NormalizeShiftI8ToI16_i8_shrui`
//   HFusion `shrui` is logical right shift. `--convert-hfusion-to-hivm` maps it
//   to `hivm.hir.vshr`, and the migrated HIVM normalize path for `vshr` is the
//   arithmetic right shift path. The converted input is therefore not the same
//   rewrite semantics as the HFusion source case.
// - `test_NormalizeShiftI8ToI16_i8_shift`
//   The HFusion source input is vector-vector `shli`. After conversion it becomes
//   vector-vector `hivm.hir.vshl`. Current HIVM `vshl` only accepts scalar rhs,
//   so the converted input is not a legal positive HIVM normalize input.
// - `test_NormalizeI8I32Cmp_select_i1_to_i16_compare`
//   The HFusion source testcase keeps this historical name, but the IR body is
//   a `hfusion.select` testcase. It does not exercise the migrated HIVM
//   comparison rewrites covered by this file.

// CHECK-LABEL: func.func @test_NormalizeCmpOp_compare_i1
// CHECK-SAME: (%[[arg0:.*]]: tensor<16x32xi1>, %[[arg1:.*]]: tensor<16x32xi1>, %[[arg2:.*]]: tensor<16x32xi1>) -> tensor<16x32xi1>
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<16x32xf16>
// CHECK: %[[cast0:.*]] = hivm.hir.vcast ins(%[[arg0]] : tensor<16x32xi1>) outs(%[[empty0]] : tensor<16x32xf16>) round_mode = <trunc> -> tensor<16x32xf16>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<16x32xf16>
// CHECK: %[[cast1:.*]] = hivm.hir.vcast ins(%[[arg1]] : tensor<16x32xi1>) outs(%[[empty1]] : tensor<16x32xf16>) round_mode = <trunc> -> tensor<16x32xf16>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast0]], %[[cast1]] : tensor<16x32xf16>, tensor<16x32xf16>) outs(%[[arg2]] : tensor<16x32xi1>) compare_mode = <lt> -> tensor<16x32xi1>
// CHECK: return %[[cmp]] : tensor<16x32xi1>
func.func @test_NormalizeCmpOp_compare_i1(%arg0: tensor<16x32xi1>,%arg1: tensor<16x32xi1>,  %dst : tensor<16x32xi1>) -> (tensor<16x32xi1>) {
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vlt>}
    ins(%arg0, %arg1 : tensor<16x32xi1>, tensor<16x32xi1>)
    outs(%dst : tensor<16x32xi1>)
    -> tensor<16x32xi1>
  return %ret : tensor<16x32xi1>
}

// CHECK-LABEL: func.func @test_NormalizeCmpToCast_hfusion_compare_neq_ops
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi64>, %[[arg1:.*]]: tensor<1024xi1>)
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[empty_f32:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[cast_src:.*]] = hivm.hir.vcast ins(%[[arg0]] : tensor<1024xi64>) outs(%[[empty_f32]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[empty_f32_2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[zero:.*]] = hivm.hir.vbrc ins(%[[cst]] : f32) outs(%[[empty_f32_2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[empty_i1_a:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[empty_i1_b:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast_src]], %[[zero]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[empty_i1_b]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<1024xi1>) outs(%[[empty_i1_a]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: return %[[not]]
func.func @test_NormalizeCmpToCast_hfusion_compare_neq_ops(
  %src1 : tensor<1024xi64>,  %dst : tensor<1024xi1>) ->  tensor<1024xi1> {
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty() : tensor<1024xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1024xi64>) -> tensor<1024xi64>
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%src1, %1 : tensor<1024xi64>, tensor<1024xi64>)
    outs(%dst : tensor<1024xi1>)
    -> tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpToCast_hfusion_compare_neq_ops_left_zero
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi64>, %[[arg1:.*]]: tensor<1024xi1>)
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[empty_f32:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[cast_src:.*]] = hivm.hir.vcast ins(%[[arg0]] : tensor<1024xi64>) outs(%[[empty_f32]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[empty_f32_2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[zero:.*]] = hivm.hir.vbrc ins(%[[cst]] : f32) outs(%[[empty_f32_2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[empty_i1_a:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[empty_i1_b:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast_src]], %[[zero]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[empty_i1_b]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<1024xi1>) outs(%[[empty_i1_a]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: return %[[not]]
func.func @test_NormalizeCmpToCast_hfusion_compare_neq_ops_left_zero(
  %src1 : tensor<1024xi64>,  %dst : tensor<1024xi1>) ->  tensor<1024xi1> {
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty() : tensor<1024xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1024xi64>) -> tensor<1024xi64>
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%1, %src1 : tensor<1024xi64>, tensor<1024xi64>)
    outs(%dst : tensor<1024xi1>)
    -> tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpVne_triton_where_hfusion_compare_select
// CHECK: %[[cast_arg3:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<8x8x4xi8>) outs(%{{.*}} : tensor<8x8x4xf16>) -> tensor<8x8x4xf16>
// CHECK: %[[zero_cast:.*]] = hivm.hir.vcast
// CHECK: %[[extracted:.*]] = tensor.extract %[[zero_cast]]
// CHECK: %[[zero_brc:.*]] = hivm.hir.vbrc ins(%[[extracted]]
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<8x8x4xi1>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<8x8x4xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast_arg3]], %[[zero_brc]] : tensor<8x8x4xf16>, tensor<8x8x4xf16>) outs(%[[empty2]] : tensor<8x8x4xi1>) -> tensor<8x8x4xi1>
// CHECK: %[[mask:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<8x8x4xi1>) outs(%[[empty1]] : tensor<8x8x4xi1>) -> tensor<8x8x4xi1>
// CHECK: %[[sel:.*]] = hivm.hir.vsel ins(%[[mask]], %{{.*}}, %{{.*}} : tensor<8x8x4xi1>, tensor<8x8x4xf16>, tensor<8x8x4xf16>) outs(%{{.*}} : tensor<8x8x4xf16>) -> tensor<8x8x4xf16>
// CHECK: bufferization.materialize_in_destination
func.func @test_NormalizeCmpVne_triton_where_hfusion_compare_select(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xi8>, %arg4: memref<?xi8>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {global_kernel = "local"} {
  %c0_i8 = arith.constant 0 : i8
  %0 = tensor.empty() : tensor<8x8x4xi8>
  %1 = linalg.fill ins(%c0_i8 : i8) outs(%0 : tensor<8x8x4xi8>) -> tensor<8x8x4xi8>
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8, 8, 4], strides: [32, 4, 1] : memref<?xi8> to memref<8x8x4xi8, strided<[32, 4, 1]>>
  %alloc = memref.alloc() : memref<8x8x4xi8>
  memref.copy %reinterpret_cast, %alloc : memref<8x8x4xi8, strided<[32, 4, 1]>> to memref<8x8x4xi8>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<8x8x4xi8>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8, 8, 4], strides: [32, 4, 1] : memref<?xi8> to memref<8x8x4xi8, strided<[32, 4, 1]>>
  %alloc_1 = memref.alloc() : memref<8x8x4xi8>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<8x8x4xi8, strided<[32, 4, 1]>> to memref<8x8x4xi8>
  %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<8x8x4xi8>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8, 8, 4], strides: [32, 4, 1] : memref<?xi8> to memref<8x8x4xi8, strided<[32, 4, 1]>>
  %alloc_3 = memref.alloc() : memref<8x8x4xi8>
  memref.copy %reinterpret_cast_2, %alloc_3 : memref<8x8x4xi8, strided<[32, 4, 1]>> to memref<8x8x4xi8>
  %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8x4xi8>
  %5 = tensor.empty() : tensor<8x8x4xi1>
  %6 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%4, %1 : tensor<8x8x4xi8>, tensor<8x8x4xi8>) outs(%5 : tensor<8x8x4xi1>) -> tensor<8x8x4xi1>
  %7 = hfusion.select ins(%6, %2, %3 : tensor<8x8x4xi1>, tensor<8x8x4xi8>, tensor<8x8x4xi8>) outs(%0 : tensor<8x8x4xi8>) -> tensor<8x8x4xi8>
  %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 8, 4], strides: [32, 4, 1] : memref<?xi8> to memref<8x8x4xi8, strided<[32, 4, 1]>>
  %cast = memref.cast %reinterpret_cast_4 : memref<8x8x4xi8, strided<[32, 4, 1]>> to memref<8x8x4xi8, strided<[?, ?, ?], offset: ?>>
  bufferization.materialize_in_destination %7 in writable %cast : (tensor<8x8x4xi8>, memref<8x8x4xi8, strided<[?, ?, ?], offset: ?>>) -> ()
  return
}

// -----

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

// CHECK-LABEL: func.func @test_NormalizeCmpOp_remove_overflow_annotation
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[arg0:.*]], %[[arg1:.*]] : tensor<8xf32>, tensor<8xf32>) outs(%[[arg2:.*]] : tensor<8xi1>) compare_mode = <lt> -> tensor<8xi1>
// CHECK-NOT: annotation.mark
// CHECK: return %[[cmp]] : tensor<8xi1>
func.func @test_NormalizeCmpOp_remove_overflow_annotation(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %dst: tensor<8xi1>) -> tensor<8xi1> {
  %ret = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%dst : tensor<8xi1>)
    -> tensor<8xi1>
  annotation.mark %ret {overflow_mode = "trunc"} : tensor<8xi1>
  return %ret : tensor<8xi1>
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
