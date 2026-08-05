// RUN: bishengir-opt -hivm-normalize-matmul %s -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// 4D fractal B with b_transpose exercises extractRealMKN B-transpose branch.

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_mmadmx_set_real_mkn_b_transpose_fractal
  func.func @test_mmadmx_set_real_mkn_b_transpose_fractal() -> tensor<64x64xf32> {
    %false = arith.constant false
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    // A: 2D ND
    %a = tensor.empty() : tensor<64x32xf8E5M2>
    // B: 4D fractal (K1,N1,16,16) with b_transpose
    %b = tensor.empty() : tensor<2x4x16x16xf8E5M2>
    %sa = tensor.empty() : tensor<64x1xui8>
    %sb = tensor.empty() : tensor<64x1xui8>
    %c = tensor.empty() : tensor<64x64xf32>
    // CHECK: hivm.hir.mmadmxL1
    // CHECK-SAME: already_set_real_mkn
    // CHECK-SAME: b_transpose
    %mad = hivm.hir.mmadmxL1 {b_transpose}
      ins(%a, %b, %sa, %sb, %false, %c64, %c32, %c64 :
          tensor<64x32xf8E5M2>, tensor<2x4x16x16xf8E5M2>,
          tensor<64x1xui8>, tensor<64x1xui8>, i1, index, index, index)
      outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %mad : tensor<64x64xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_mmadmx_already_set_real_mkn_noop
  // Attr already present → SetRealMKNPattern early-exit branch.
  func.func @test_mmadmx_already_set_real_mkn_noop() -> tensor<16x16xf32> {
    %false = arith.constant false
    %c16 = arith.constant 16 : index
    %a = tensor.empty() : tensor<16x16xf8E5M2>
    %b = tensor.empty() : tensor<16x16xf8E5M2>
    %sa = tensor.empty() : tensor<1xui8>
    %sb = tensor.empty() : tensor<1xui8>
    %c = tensor.empty() : tensor<16x16xf32>
    // CHECK: hivm.hir.mmadmxL1
    // CHECK-SAME: already_set_real_mkn
    %mad = hivm.hir.mmadmxL1 {already_set_real_mkn}
      ins(%a, %b, %sa, %sb, %false, %c16, %c16, %c16 :
          tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
          tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
      outs(%c : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %mad : tensor<16x16xf32>
  }
}
