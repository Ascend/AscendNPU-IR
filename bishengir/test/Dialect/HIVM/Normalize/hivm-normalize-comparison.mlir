// RUN: bishengir-opt --hivm-normalize-ops -split-input-file %s | FileCheck %s

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