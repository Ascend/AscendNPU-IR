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
