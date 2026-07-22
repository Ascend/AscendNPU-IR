// RUN: bishengir-opt --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeCmpVneOp_neq_to_Not_eq
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi64>, %[[arg1:.*]]: tensor<1024xi64>, %[[arg2:.*]]: tensor<1024xi1>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[veq:.*]] = hivm.hir.vcmp ins(%[[arg0]], %[[arg1]] : tensor<1024xi64>, tensor<1024xi64>) outs(%[[empty]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[notOp:.*]] = hivm.hir.vnot ins(%[[veq]] : tensor<1024xi1>) outs(%[[arg2]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: return %[[notOp]]
func.func @test_NormalizeCmpVneOp_neq_to_Not_eq(
  %src1 : tensor<1024xi64>, %src2 : tensor<1024xi64>, %dst : tensor<1024xi1>) -> tensor<1024xi1> {
  %ret = hivm.hir.vcmp ins(%src1, %src2 : tensor<1024xi64>, tensor<1024xi64>)
    outs(%dst : tensor<1024xi1>)
    compare_mode = #hivm.compare_mode<ne> -> tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpVneOp_neq_f32
// CHECK-SAME: (%[[arg0:.*]]: tensor<256xf32>, %[[arg1:.*]]: tensor<256xf32>, %[[arg2:.*]]: tensor<256xi1>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<256xi1>
// CHECK: %[[veq:.*]] = hivm.hir.vcmp ins(%[[arg0]], %[[arg1]] : tensor<256xf32>, tensor<256xf32>) outs(%[[empty]] : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[notOp:.*]] = hivm.hir.vnot ins(%[[veq]] : tensor<256xi1>) outs(%[[arg2]] : tensor<256xi1>) -> tensor<256xi1>
// CHECK: return %[[notOp]]
func.func @test_NormalizeCmpVneOp_neq_f32(
  %src1 : tensor<256xf32>, %src2 : tensor<256xf32>, %dst : tensor<256xi1>) -> tensor<256xi1> {
  %ret = hivm.hir.vcmp ins(%src1, %src2 : tensor<256xf32>, tensor<256xf32>)
    outs(%dst : tensor<256xi1>)
    compare_mode = #hivm.compare_mode<ne> -> tensor<256xi1>
  return %ret : tensor<256xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpVneOp_eq_unchanged
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi64>, %[[arg1:.*]]: tensor<1024xi64>, %[[arg2:.*]]: tensor<1024xi1>)
// CHECK: %[[ret:.*]] = hivm.hir.vcmp ins(%[[arg0]], %[[arg1]] : tensor<1024xi64>, tensor<1024xi64>) outs(%[[arg2]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: return %[[ret]]
func.func @test_NormalizeCmpVneOp_eq_unchanged(
  %src1 : tensor<1024xi64>, %src2 : tensor<1024xi64>, %dst : tensor<1024xi1>) -> tensor<1024xi1> {
  %ret = hivm.hir.vcmp ins(%src1, %src2 : tensor<1024xi64>, tensor<1024xi64>)
    outs(%dst : tensor<1024xi1>)
    compare_mode = #hivm.compare_mode<eq> -> tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpVneOp_lt_unchanged
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi64>, %[[arg1:.*]]: tensor<1024xi64>, %[[arg2:.*]]: tensor<1024xi1>)
// CHECK: %[[ret:.*]] = hivm.hir.vcmp ins(%[[arg0]], %[[arg1]] : tensor<1024xi64>, tensor<1024xi64>) outs(%[[arg2]] : tensor<1024xi1>) compare_mode = <lt> -> tensor<1024xi1>
// CHECK: return %[[ret]]
func.func @test_NormalizeCmpVneOp_lt_unchanged(
  %src1 : tensor<1024xi64>, %src2 : tensor<1024xi64>, %dst : tensor<1024xi1>) -> tensor<1024xi1> {
  %ret = hivm.hir.vcmp ins(%src1, %src2 : tensor<1024xi64>, tensor<1024xi64>)
    outs(%dst : tensor<1024xi1>)
    compare_mode = #hivm.compare_mode<lt> -> tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpVneOp_neq_2d
// CHECK-SAME: (%[[arg0:.*]]: tensor<16x32xf16>, %[[arg1:.*]]: tensor<16x32xf16>, %[[arg2:.*]]: tensor<16x32xi1>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<16x32xi1>
// CHECK: %[[veq:.*]] = hivm.hir.vcmp ins(%[[arg0]], %[[arg1]] : tensor<16x32xf16>, tensor<16x32xf16>) outs(%[[empty]] : tensor<16x32xi1>) -> tensor<16x32xi1>
// CHECK: %[[notOp:.*]] = hivm.hir.vnot ins(%[[veq]] : tensor<16x32xi1>) outs(%[[arg2]] : tensor<16x32xi1>) -> tensor<16x32xi1>
// CHECK: return %[[notOp]]
func.func @test_NormalizeCmpVneOp_neq_2d(
  %src1 : tensor<16x32xf16>, %src2 : tensor<16x32xf16>, %dst : tensor<16x32xi1>) -> tensor<16x32xi1> {
  %ret = hivm.hir.vcmp ins(%src1, %src2 : tensor<16x32xf16>, tensor<16x32xf16>)
    outs(%dst : tensor<16x32xi1>)
    compare_mode = #hivm.compare_mode<ne> -> tensor<16x32xi1>
  return %ret : tensor<16x32xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpOp_remove_overflow_annotation
// CHECK: %[[ret:.*]] = hivm.hir.vcmp ins(%[[arg0:.*]], %[[arg1:.*]] : tensor<8xi32>, tensor<8xi32>) outs(%[[arg2:.*]] : tensor<8xi1>) -> tensor<8xi1>
// CHECK-NOT: annotation.mark
// CHECK: return %[[ret]] : tensor<8xi1>
func.func @test_NormalizeCmpOp_remove_overflow_annotation(
  %src1 : tensor<8xi32>, %src2 : tensor<8xi32>, %dst : tensor<8xi1>) -> tensor<8xi1> {
  %ret = hivm.hir.vcmp ins(%src1, %src2 : tensor<8xi32>, tensor<8xi32>)
    outs(%dst : tensor<8xi1>)
    compare_mode = #hivm.compare_mode<eq> -> tensor<8xi1>
  annotation.mark %ret {overflow_mode = "trunc"} : tensor<8xi1>
  return %ret : tensor<8xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpOp_without_overflow_annotation_unchanged
// CHECK: %[[ret:.*]] = hivm.hir.vcmp ins(%[[arg0:.*]], %[[arg1:.*]] : tensor<8xf32>, tensor<8xf32>) outs(%[[arg2:.*]] : tensor<8xi1>) compare_mode = <lt> -> tensor<8xi1>
// CHECK-NOT: annotation.mark
// CHECK: return %[[ret]] : tensor<8xi1>
func.func @test_NormalizeCmpOp_without_overflow_annotation_unchanged(
  %src1 : tensor<8xf32>, %src2 : tensor<8xf32>, %dst : tensor<8xi1>) -> tensor<8xi1> {
  %ret = hivm.hir.vcmp ins(%src1, %src2 : tensor<8xf32>, tensor<8xf32>)
    outs(%dst : tensor<8xi1>)
    compare_mode = #hivm.compare_mode<lt> -> tensor<8xi1>
  return %ret : tensor<8xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpToCast_i32_zero_tensor
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi32>, %[[arg1:.*]]: tensor<16xi1>)
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[empty_f32:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast_src:.*]] = hivm.hir.vcast ins(%[[arg0]] : tensor<16xi32>) outs(%[[empty_f32]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty_f32_2:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[zero:.*]] = hivm.hir.vbrc ins(%[[cst]] : f32) outs(%[[empty_f32_2]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty_i1_a:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[empty_i1_b:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast_src]], %[[zero]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty_i1_b]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<16xi1>) outs(%[[empty_i1_a]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[not]]
func.func @test_NormalizeCmpToCast_i32_zero_tensor(
  %src : tensor<16xi32>, %dst : tensor<16xi1>) -> tensor<16xi1> {
  %c0 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<16xi32>
  %zero = hivm.hir.vbrc ins(%c0 : i32) outs(%empty : tensor<16xi32>) -> tensor<16xi32>
  %ret = hivm.hir.vcmp ins(%src, %zero : tensor<16xi32>, tensor<16xi32>)
    outs(%dst : tensor<16xi1>)
    compare_mode = #hivm.compare_mode<ne> -> tensor<16xi1>
  return %ret : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpToCast_i64_zero_tensor
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi64>, %[[arg1:.*]]: tensor<16xi1>)
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[empty_f32:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast_src:.*]] = hivm.hir.vcast ins(%[[arg0]] : tensor<16xi64>) outs(%[[empty_f32]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty_f32_2:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[zero:.*]] = hivm.hir.vbrc ins(%[[cst]] : f32) outs(%[[empty_f32_2]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty_i1_a:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[empty_i1_b:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast_src]], %[[zero]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty_i1_b]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<16xi1>) outs(%[[empty_i1_a]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[not]]
func.func @test_NormalizeCmpToCast_i64_zero_tensor(
  %src : tensor<16xi64>, %dst : tensor<16xi1>) -> tensor<16xi1> {
  %c0 = arith.constant 0 : i64
  %empty = tensor.empty() : tensor<16xi64>
  %zero = hivm.hir.vbrc ins(%c0 : i64) outs(%empty : tensor<16xi64>) -> tensor<16xi64>
  %ret = hivm.hir.vcmp ins(%src, %zero : tensor<16xi64>, tensor<16xi64>)
    outs(%dst : tensor<16xi1>)
    compare_mode = #hivm.compare_mode<ne> -> tensor<16xi1>
  return %ret : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpToCast_i64_left_zero_tensor
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi64>, %[[arg1:.*]]: tensor<16xi1>)
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[empty_f32:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast_src:.*]] = hivm.hir.vcast ins(%[[arg0]] : tensor<16xi64>) outs(%[[empty_f32]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty_f32_2:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[zero:.*]] = hivm.hir.vbrc ins(%[[cst]] : f32) outs(%[[empty_f32_2]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty_i1_a:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[empty_i1_b:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast_src]], %[[zero]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty_i1_b]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<16xi1>) outs(%[[empty_i1_a]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[not]]
func.func @test_NormalizeCmpToCast_i64_left_zero_tensor(
  %src : tensor<16xi64>, %dst : tensor<16xi1>) -> tensor<16xi1> {
  %c0 = arith.constant 0 : i64
  %empty = tensor.empty() : tensor<16xi64>
  %zero = hivm.hir.vbrc ins(%c0 : i64) outs(%empty : tensor<16xi64>) -> tensor<16xi64>
  %ret = hivm.hir.vcmp ins(%zero, %src : tensor<16xi64>, tensor<16xi64>)
    outs(%dst : tensor<16xi1>)
    compare_mode = #hivm.compare_mode<ne> -> tensor<16xi1>
  return %ret : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeI8Cmp_ne_to_f16
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi8>, %[[arg1:.*]]: tensor<16xi8>, %[[arg2:.*]]: tensor<16xi1>)
// CHECK: %[[lhs_empty:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[lhs_cast:.*]] = hivm.hir.vcast ins(%[[arg0]] : tensor<16xi8>) outs(%[[lhs_empty]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[rhs_empty:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[rhs_cast:.*]] = hivm.hir.vcast ins(%[[arg1]] : tensor<16xi8>) outs(%[[rhs_empty]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[not_empty:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_empty:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[lhs_cast]], %[[rhs_cast]] : tensor<16xf16>, tensor<16xf16>) outs(%[[cmp_empty]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<16xi1>) outs(%[[not_empty]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[not]]
func.func @test_NormalizeI8Cmp_ne_to_f16(
  %src1 : tensor<16xi8>, %src2 : tensor<16xi8>, %dst : tensor<16xi1>) -> tensor<16xi1> {
  %ret = hivm.hir.vcmp ins(%src1, %src2 : tensor<16xi8>, tensor<16xi8>)
    outs(%dst : tensor<16xi1>)
    compare_mode = #hivm.compare_mode<ne> -> tensor<16xi1>
  return %ret : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeI8Cmp_eq_to_f16
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi8>, %[[arg1:.*]]: tensor<16xi8>, %[[arg2:.*]]: tensor<16xi1>)
// CHECK: %[[lhs_empty:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[lhs_cast:.*]] = hivm.hir.vcast ins(%[[arg0]] : tensor<16xi8>) outs(%[[lhs_empty]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[rhs_empty:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[rhs_cast:.*]] = hivm.hir.vcast ins(%[[arg1]] : tensor<16xi8>) outs(%[[rhs_empty]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[cmp_empty:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[lhs_cast]], %[[rhs_cast]] : tensor<16xf16>, tensor<16xf16>) outs(%[[cmp_empty]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[cmp]]
func.func @test_NormalizeI8Cmp_eq_to_f16(
  %src1 : tensor<16xi8>, %src2 : tensor<16xi8>, %dst : tensor<16xi1>) -> tensor<16xi1> {
  %ret = hivm.hir.vcmp ins(%src1, %src2 : tensor<16xi8>, tensor<16xi8>)
    outs(%dst : tensor<16xi1>)
    compare_mode = #hivm.compare_mode<eq> -> tensor<16xi1>
  return %ret : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeI32Cmp_lt_to_i64
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi32>, %[[arg1:.*]]: tensor<16xi32>, %[[arg2:.*]]: tensor<16xi1>)
// CHECK: %[[lhs_empty:.*]] = tensor.empty() : tensor<16xi64>
// CHECK: %[[lhs_cast:.*]] = hivm.hir.vcast ins(%[[arg0]] : tensor<16xi32>) outs(%[[lhs_empty]] : tensor<16xi64>) -> tensor<16xi64>
// CHECK: %[[rhs_empty:.*]] = tensor.empty() : tensor<16xi64>
// CHECK: %[[rhs_cast:.*]] = hivm.hir.vcast ins(%[[arg1]] : tensor<16xi32>) outs(%[[rhs_empty]] : tensor<16xi64>) -> tensor<16xi64>
// CHECK: %[[cmp_empty:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[lhs_cast]], %[[rhs_cast]] : tensor<16xi64>, tensor<16xi64>) outs(%[[cmp_empty]] : tensor<16xi1>) compare_mode = <lt> -> tensor<16xi1>
// CHECK: return %[[cmp]]
func.func @test_NormalizeI32Cmp_lt_to_i64(
  %src1 : tensor<16xi32>, %src2 : tensor<16xi32>, %dst : tensor<16xi1>) -> tensor<16xi1> {
  %ret = hivm.hir.vcmp ins(%src1, %src2 : tensor<16xi32>, tensor<16xi32>)
    outs(%dst : tensor<16xi1>)
    compare_mode = #hivm.compare_mode<lt> -> tensor<16xi1>
  return %ret : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeI32Cmp_eq_unchanged_no_i64_cast
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi32>, %[[arg1:.*]]: tensor<16xi32>, %[[arg2:.*]]: tensor<16xi1>)
// CHECK-NOT: hivm.hir.vcast
// CHECK: %[[ret:.*]] = hivm.hir.vcmp ins(%[[arg0]], %[[arg1]] : tensor<16xi32>, tensor<16xi32>) outs(%[[arg2]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[ret]]
func.func @test_NormalizeI32Cmp_eq_unchanged_no_i64_cast(
  %src1 : tensor<16xi32>, %src2 : tensor<16xi32>, %dst : tensor<16xi1>) -> tensor<16xi1> {
  %ret = hivm.hir.vcmp ins(%src1, %src2 : tensor<16xi32>, tensor<16xi32>)
    outs(%dst : tensor<16xi1>)
    compare_mode = #hivm.compare_mode<eq> -> tensor<16xi1>
  return %ret : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeI32Cmp_ne_no_i64_cast
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xi32>, %[[arg1:.*]]: tensor<16xi32>, %[[arg2:.*]]: tensor<16xi1>)
// CHECK-NOT: hivm.hir.vcast
// CHECK: %[[cmp_empty:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[arg0]], %[[arg1]] : tensor<16xi32>, tensor<16xi32>) outs(%[[cmp_empty]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<16xi1>) outs(%[[arg2]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[not]]
func.func @test_NormalizeI32Cmp_ne_no_i64_cast(
  %src1 : tensor<16xi32>, %src2 : tensor<16xi32>, %dst : tensor<16xi1>) -> tensor<16xi1> {
  %ret = hivm.hir.vcmp ins(%src1, %src2 : tensor<16xi32>, tensor<16xi32>)
    outs(%dst : tensor<16xi1>)
    compare_mode = #hivm.compare_mode<ne> -> tensor<16xi1>
  return %ret : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeShiftRightI8_to_i16
// CHECK-SAME: (%[[arg0:.*]]: tensor<32xi8>, %[[arg1:.*]]: tensor<32xi8>, %[[arg2:.*]]: tensor<32xi8>)
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
func.func @test_NormalizeShiftRightI8_to_i16(
  %src : tensor<32xi8>, %shift : tensor<32xi8>, %dst : tensor<32xi8>) -> tensor<32xi8> {
  %ret = hivm.hir.vshr ins(%src, %shift : tensor<32xi8>, tensor<32xi8>)
    outs(%dst : tensor<32xi8>) round : true -> tensor<32xi8>
  return %ret : tensor<32xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeIsInf_f32
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xf32>, %[[arg1:.*]]: tensor<16xi1>)
// CHECK-DAG: %[[c0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[cneginf:.*]] = arith.constant -2139095040 : i32
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[cneg1:.*]] = arith.constant -1 : i32
// CHECK-DAG: %[[cmask:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[maskempty:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[mask:.*]] = hivm.hir.vbrc ins(%[[cmask]] : i32) outs(%[[maskempty]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[bits:.*]] = hivm.hir.bitcast %[[arg0]] : tensor<16xf32> -> tensor<16xi32>
// CHECK: %[[andempty:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[vand:.*]] = hivm.hir.vand ins(%[[bits]], %[[mask]] : tensor<16xi32>, tensor<16xi32>) outs(%[[andempty]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[addempty:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[vadd0:.*]] = hivm.hir.vadd ins(%[[vand]], %[[cneginf]] : tensor<16xi32>, i32) outs(%[[addempty]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[floatbits:.*]] = hivm.hir.bitcast %[[vadd0]] : tensor<16xi32> -> tensor<16xf32>
// CHECK: %[[absempty:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs:.*]] = hivm.hir.vabs ins(%[[floatbits]] : tensor<16xf32>) outs(%[[absempty]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[absbits:.*]] = hivm.hir.bitcast %[[abs]] : tensor<16xf32> -> tensor<16xi32>
// CHECK: %[[minempty:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[min:.*]] = hivm.hir.vmin ins(%[[absbits]], %[[c1]] : tensor<16xi32>, i32) outs(%[[minempty]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[mul:.*]] = hivm.hir.vmul ins(%[[min]], %[[cneg1]] : tensor<16xi32>, i32) outs(%[[min]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[add1:.*]] = hivm.hir.vadd ins(%[[mul]], %[[c1]] : tensor<16xi32>, i32) outs(%[[mul]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[castempty:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast:.*]] = hivm.hir.vcast ins(%[[add1]] : tensor<16xi32>) outs(%[[castempty]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[cmpempty:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast]], %[[c0]] : tensor<16xf32>, f32) outs(%[[cmpempty]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<16xi1>) outs(%[[arg1]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[not]]
func.func @test_NormalizeIsInf_f32(
  %src : tensor<16xf32>, %dst : tensor<16xi1>) -> tensor<16xi1> {
  %ret = hivm.hir.visinf ins(%src : tensor<16xf32>)
    outs(%dst : tensor<16xi1>) -> tensor<16xi1>
  return %ret : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeIsInf_f16
// CHECK-SAME: (%[[arg0:.*]]: tensor<32xf16>, %[[arg1:.*]]: tensor<32xi1>)
// CHECK-DAG: %[[c0:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG: %[[cneginf:.*]] = arith.constant -31744 : i16
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i16
// CHECK-DAG: %[[cneg1:.*]] = arith.constant -1 : i16
// CHECK-DAG: %[[cmask:.*]] = arith.constant 32767 : i16
// CHECK: %[[maskempty:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[mask:.*]] = hivm.hir.vbrc ins(%[[cmask]] : i16) outs(%[[maskempty]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[bits:.*]] = hivm.hir.bitcast %[[arg0]] : tensor<32xf16> -> tensor<32xi16>
// CHECK: %[[andempty:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[vand:.*]] = hivm.hir.vand ins(%[[bits]], %[[mask]] : tensor<32xi16>, tensor<32xi16>) outs(%[[andempty]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[addempty:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[vadd0:.*]] = hivm.hir.vadd ins(%[[vand]], %[[cneginf]] : tensor<32xi16>, i16) outs(%[[addempty]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[floatbits:.*]] = hivm.hir.bitcast %[[vadd0]] : tensor<32xi16> -> tensor<32xf16>
// CHECK: %[[absempty:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[abs:.*]] = hivm.hir.vabs ins(%[[floatbits]] : tensor<32xf16>) outs(%[[absempty]] : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[absbits:.*]] = hivm.hir.bitcast %[[abs]] : tensor<32xf16> -> tensor<32xi16>
// CHECK: %[[minempty:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[min:.*]] = hivm.hir.vmin ins(%[[absbits]], %[[c1]] : tensor<32xi16>, i16) outs(%[[minempty]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[mul:.*]] = hivm.hir.vmul ins(%[[min]], %[[cneg1]] : tensor<32xi16>, i16) outs(%[[min]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[add1:.*]] = hivm.hir.vadd ins(%[[mul]], %[[c1]] : tensor<32xi16>, i16) outs(%[[mul]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[castempty:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[cast:.*]] = hivm.hir.vcast ins(%[[add1]] : tensor<32xi16>) outs(%[[castempty]] : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[cmpempty:.*]] = tensor.empty() : tensor<32xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast]], %[[c0]] : tensor<32xf16>, f16) outs(%[[cmpempty]] : tensor<32xi1>) -> tensor<32xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<32xi1>) outs(%[[arg1]] : tensor<32xi1>) -> tensor<32xi1>
// CHECK: return %[[not]]
func.func @test_NormalizeIsInf_f16(
  %src : tensor<32xf16>, %dst : tensor<32xi1>) -> tensor<32xi1> {
  %ret = hivm.hir.visinf ins(%src : tensor<32xf16>)
    outs(%dst : tensor<32xi1>) -> tensor<32xi1>
  return %ret : tensor<32xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeIsNan_f32
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xf32>, %[[arg1:.*]]: tensor<16xi1>)
// CHECK-DAG: %[[c0f:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[cneginf:.*]] = arith.constant -2139095040 : i32
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[cmask:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[maskempty:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[mask:.*]] = hivm.hir.vbrc ins(%[[cmask]] : i32) outs(%[[maskempty]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[bits:.*]] = hivm.hir.bitcast %[[arg0]] : tensor<16xf32> -> tensor<16xi32>
// CHECK: %[[andempty:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[vand:.*]] = hivm.hir.vand ins(%[[bits]], %[[mask]] : tensor<16xi32>, tensor<16xi32>) outs(%[[andempty]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[addempty:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[vadd:.*]] = hivm.hir.vadd ins(%[[vand]], %[[cneginf]] : tensor<16xi32>, i32) outs(%[[addempty]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[min:.*]] = hivm.hir.vmin ins(%[[vadd]], %[[c1]] : tensor<16xi32>, i32) outs(%[[vadd]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[max:.*]] = hivm.hir.vmax ins(%[[min]], %[[c0]] : tensor<16xi32>, i32) outs(%[[min]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[castempty:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast:.*]] = hivm.hir.vcast ins(%[[max]] : tensor<16xi32>) outs(%[[castempty]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[cmpempty:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast]], %[[c0f]] : tensor<16xf32>, f32) outs(%[[cmpempty]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<16xi1>) outs(%[[arg1]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[not]]
func.func @test_NormalizeIsNan_f32(
  %src : tensor<16xf32>, %dst : tensor<16xi1>) -> tensor<16xi1> {
  %ret = hivm.hir.visnan ins(%src : tensor<16xf32>)
    outs(%dst : tensor<16xi1>) -> tensor<16xi1>
  return %ret : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeIsNan_f16
// CHECK-SAME: (%[[arg0:.*]]: tensor<32xf16>, %[[arg1:.*]]: tensor<32xi1>)
// CHECK-DAG: %[[c0f:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG: %[[cneginf:.*]] = arith.constant -31744 : i16
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : i16
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i16
// CHECK-DAG: %[[cmask:.*]] = arith.constant 32767 : i16
// CHECK: %[[maskempty:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[mask:.*]] = hivm.hir.vbrc ins(%[[cmask]] : i16) outs(%[[maskempty]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[bits:.*]] = hivm.hir.bitcast %[[arg0]] : tensor<32xf16> -> tensor<32xi16>
// CHECK: %[[andempty:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[vand:.*]] = hivm.hir.vand ins(%[[bits]], %[[mask]] : tensor<32xi16>, tensor<32xi16>) outs(%[[andempty]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[addempty:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[vadd:.*]] = hivm.hir.vadd ins(%[[vand]], %[[cneginf]] : tensor<32xi16>, i16) outs(%[[addempty]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[min:.*]] = hivm.hir.vmin ins(%[[vadd]], %[[c1]] : tensor<32xi16>, i16) outs(%[[vadd]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[max:.*]] = hivm.hir.vmax ins(%[[min]], %[[c0]] : tensor<32xi16>, i16) outs(%[[min]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[castempty:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[cast:.*]] = hivm.hir.vcast ins(%[[max]] : tensor<32xi16>) outs(%[[castempty]] : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[cmpempty:.*]] = tensor.empty() : tensor<32xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast]], %[[c0f]] : tensor<32xf16>, f16) outs(%[[cmpempty]] : tensor<32xi1>) -> tensor<32xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<32xi1>) outs(%[[arg1]] : tensor<32xi1>) -> tensor<32xi1>
// CHECK: return %[[not]]
func.func @test_NormalizeIsNan_f16(
  %src : tensor<32xf16>, %dst : tensor<32xi1>) -> tensor<32xi1> {
  %ret = hivm.hir.visnan ins(%src : tensor<32xf16>)
    outs(%dst : tensor<32xi1>) -> tensor<32xi1>
  return %ret : tensor<32xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeXor_vxor
// CHECK-SAME: (%[[arg0:.*]]: tensor<512xi16>, %[[arg1:.*]]: tensor<512xi16>)
// CHECK: %[[or_init:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[vor:.*]] = hivm.hir.vor ins(%[[arg0]], %[[arg1]] : tensor<512xi16>, tensor<512xi16>) outs(%[[or_init]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[and_init:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[vand:.*]] = hivm.hir.vand ins(%[[arg0]], %[[arg1]] : tensor<512xi16>, tensor<512xi16>) outs(%[[and_init]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[vnot:.*]] = hivm.hir.vnot ins(%[[vand]] : tensor<512xi16>) outs(%[[vand]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[res_init:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[res:.*]] = hivm.hir.vand ins(%[[vnot]], %[[vor]] : tensor<512xi16>, tensor<512xi16>) outs(%[[res_init]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: return %[[res]]
func.func @test_NormalizeXor_vxor(%arg0: tensor<512xi16>, %arg1: tensor<512xi16>) -> tensor<512xi16> {
  %0 = tensor.empty() : tensor<512xi16>
  %1 = hivm.hir.vxor ins(%arg0, %arg1 : tensor<512xi16>, tensor<512xi16>)
    outs(%0 : tensor<512xi16>) -> tensor<512xi16>
  return %1 : tensor<512xi16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeXor_vxor_i1
// CHECK-SAME: (%[[arg0:.*]]: tensor<512xi1>, %[[arg1:.*]]: tensor<512xi1>)
// CHECK: %[[or_init:.*]] = tensor.empty() : tensor<512xi1>
// CHECK: %[[vor:.*]] = hivm.hir.vor ins(%[[arg0]], %[[arg1]] : tensor<512xi1>, tensor<512xi1>) outs(%[[or_init]] : tensor<512xi1>) -> tensor<512xi1>
// CHECK: %[[and_init:.*]] = tensor.empty() : tensor<512xi1>
// CHECK: %[[vand:.*]] = hivm.hir.vand ins(%[[arg0]], %[[arg1]] : tensor<512xi1>, tensor<512xi1>) outs(%[[and_init]] : tensor<512xi1>) -> tensor<512xi1>
// CHECK: %[[vnot:.*]] = hivm.hir.vnot ins(%[[vand]] : tensor<512xi1>) outs(%[[vand]] : tensor<512xi1>) -> tensor<512xi1>
// CHECK: %[[res_init:.*]] = tensor.empty() : tensor<512xi1>
// CHECK: %[[res:.*]] = hivm.hir.vand ins(%[[vnot]], %[[vor]] : tensor<512xi1>, tensor<512xi1>) outs(%[[res_init]] : tensor<512xi1>) -> tensor<512xi1>
// CHECK: return %[[res]]
func.func @test_NormalizeXor_vxor_i1(%arg0: tensor<512xi1>, %arg1: tensor<512xi1>) -> tensor<512xi1> {
  %0 = tensor.empty() : tensor<512xi1>
  %1 = hivm.hir.vxor ins(%arg0, %arg1 : tensor<512xi1>, tensor<512xi1>)
    outs(%0 : tensor<512xi1>) -> tensor<512xi1>
  return %1 : tensor<512xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeXor_vxor_ui8
// CHECK-SAME: (%[[arg0:.*]]: tensor<512xui8>, %[[arg1:.*]]: tensor<512xui8>)
// CHECK: %[[or_init:.*]] = tensor.empty() : tensor<512xui8>
// CHECK: %[[vor:.*]] = hivm.hir.vor ins(%[[arg0]], %[[arg1]] : tensor<512xui8>, tensor<512xui8>) outs(%[[or_init]] : tensor<512xui8>) -> tensor<512xui8>
// CHECK: %[[and_init:.*]] = tensor.empty() : tensor<512xui8>
// CHECK: %[[vand:.*]] = hivm.hir.vand ins(%[[arg0]], %[[arg1]] : tensor<512xui8>, tensor<512xui8>) outs(%[[and_init]] : tensor<512xui8>) -> tensor<512xui8>
// CHECK: %[[vnot:.*]] = hivm.hir.vnot ins(%[[vand]] : tensor<512xui8>) outs(%[[vand]] : tensor<512xui8>) -> tensor<512xui8>
// CHECK: %[[res_init:.*]] = tensor.empty() : tensor<512xui8>
// CHECK: %[[res:.*]] = hivm.hir.vand ins(%[[vnot]], %[[vor]] : tensor<512xui8>, tensor<512xui8>) outs(%[[res_init]] : tensor<512xui8>) -> tensor<512xui8>
// CHECK: return %[[res]]
func.func @test_NormalizeXor_vxor_ui8(%arg0: tensor<512xui8>, %arg1: tensor<512xui8>) -> tensor<512xui8> {
  %0 = tensor.empty() : tensor<512xui8>
  %1 = hivm.hir.vxor ins(%arg0, %arg1 : tensor<512xui8>, tensor<512xui8>)
    outs(%0 : tensor<512xui8>) -> tensor<512xui8>
  return %1 : tensor<512xui8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeXor_vxor_ui64
// CHECK-SAME: (%[[arg0:.*]]: tensor<512xui64>, %[[arg1:.*]]: tensor<512xui64>)
// CHECK: %[[or_init:.*]] = tensor.empty() : tensor<512xui64>
// CHECK: %[[vor:.*]] = hivm.hir.vor ins(%[[arg0]], %[[arg1]] : tensor<512xui64>, tensor<512xui64>) outs(%[[or_init]] : tensor<512xui64>) -> tensor<512xui64>
// CHECK: %[[and_init:.*]] = tensor.empty() : tensor<512xui64>
// CHECK: %[[vand:.*]] = hivm.hir.vand ins(%[[arg0]], %[[arg1]] : tensor<512xui64>, tensor<512xui64>) outs(%[[and_init]] : tensor<512xui64>) -> tensor<512xui64>
// CHECK: %[[vnot:.*]] = hivm.hir.vnot ins(%[[vand]] : tensor<512xui64>) outs(%[[vand]] : tensor<512xui64>) -> tensor<512xui64>
// CHECK: %[[res_init:.*]] = tensor.empty() : tensor<512xui64>
// CHECK: %[[res:.*]] = hivm.hir.vand ins(%[[vnot]], %[[vor]] : tensor<512xui64>, tensor<512xui64>) outs(%[[res_init]] : tensor<512xui64>) -> tensor<512xui64>
// CHECK: return %[[res]]
func.func @test_NormalizeXor_vxor_ui64(%arg0: tensor<512xui64>, %arg1: tensor<512xui64>) -> tensor<512xui64> {
  %0 = tensor.empty() : tensor<512xui64>
  %1 = hivm.hir.vxor ins(%arg0, %arg1 : tensor<512xui64>, tensor<512xui64>)
    outs(%0 : tensor<512xui64>) -> tensor<512xui64>
  return %1 : tensor<512xui64>
}
