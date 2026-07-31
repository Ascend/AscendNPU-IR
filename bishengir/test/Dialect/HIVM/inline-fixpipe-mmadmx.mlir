// RUN: bishengir-opt -hivm-insert-fixpipe %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
// Elementwise bias (full C) with init=false: InsertFixpipe should refuse to
// insert because the op will later decompose to mmad+vadd.
// CHECK-LABEL: func.func @test_mmadmx_fixpipe_skip_elemwise_bias
func.func @test_mmadmx_fixpipe_skip_elemwise_bias(%bias: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c16 = arith.constant 16 : index
  %false = arith.constant false
  %a = tensor.empty() : tensor<16x16xf8E5M2>
  %b = tensor.empty() : tensor<16x16xf8E5M2>
  %sa = tensor.empty() : tensor<1xui8>
  %sb = tensor.empty() : tensor<1xui8>
  // CHECK: hivm.hir.mmadmxL1
  // CHECK-NOT: hivm.hir.fixpipe
  %mad = hivm.hir.mmadmxL1
    ins(%a, %b, %sa, %sb, %false, %c16, %c16, %c16 :
        tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
        tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
    outs(%bias : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mad : tensor<16x16xf32>
}

// -----
// Clean empty init: fixpipe should be inserted.
// CHECK-LABEL: func.func @test_mmadmx_fixpipe_insert_on_empty
func.func @test_mmadmx_fixpipe_insert_on_empty() -> tensor<16x16xf32> {
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %a = tensor.empty() : tensor<16x16xf8E5M2>
  %b = tensor.empty() : tensor<16x16xf8E5M2>
  %sa = tensor.empty() : tensor<1xui8>
  %sb = tensor.empty() : tensor<1xui8>
  %c = tensor.empty() : tensor<16x16xf32>
  // CHECK: hivm.hir.mmadmxL1
  // CHECK: hivm.hir.fixpipe
  %mad = hivm.hir.mmadmxL1
    ins(%a, %b, %sa, %sb, %true, %c16, %c16, %c16 :
        tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
        tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
    outs(%c : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %mad : tensor<16x16xf32>
}
