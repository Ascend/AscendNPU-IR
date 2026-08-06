// RUN: bishengir-opt -convert-hfusion-to-hivm %s -split-input-file -verify-diagnostics | FileCheck %s

// Exercise getFormattedI8Source failure paths (pattern does not inline).

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_matmulscale_format_element_mismatch
  // i8 bitcast to e5m2 but format says e4m3 -> do not inline.
  // CHECK: hivm.hir.bitcast
  // CHECK: hivm.hir.mmadmxL1
  // CHECK-SAME: f8E5M2
  func.func @test_matmulscale_format_element_mismatch(
      %arg0: tensor<4x8xi8>, %arg1: tensor<8x16xi8>,
      %arg2: tensor<4x1xi8>, %arg3: tensor<16x1xi8>)
      -> tensor<4x16xf32> {
    %a_empty = tensor.empty() : tensor<4x8xf8E5M2>
    %a_fp8 = hfusion.bitcast ins(%arg0 : tensor<4x8xi8>)
      outs(%a_empty : tensor<4x8xf8E5M2>) -> tensor<4x8xf8E5M2>
    %b_empty = tensor.empty() : tensor<8x16xf8E5M2>
    %b_fp8 = hfusion.bitcast ins(%arg1 : tensor<8x16xi8>)
      outs(%b_empty : tensor<8x16xf8E5M2>) -> tensor<8x16xf8E5M2>
    %acc = tensor.empty() : tensor<4x16xf32>
    %res = hfusion.matmul_mx {lhsFormat = 2 : i32, rhsFormat = 2 : i32}
      ins(%a_fp8, %b_fp8, %arg2, %arg3 :
          tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>,
          tensor<4x1xi8>, tensor<16x1xi8>)
      outs(%acc : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %res : tensor<4x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_matmulscale_format_without_bitcast
  // Format set but inputs are plain fp8 (no i8 bitcast) -> do not inline.
  // CHECK: hivm.hir.mmadmxL1
  // CHECK-SAME: f8E5M2
  func.func @test_matmulscale_format_without_bitcast(
      %arg0: tensor<4x8xf8E5M2>, %arg1: tensor<8x16xf8E5M2>,
      %arg2: tensor<4x1xi8>, %arg3: tensor<16x1xi8>)
      -> tensor<4x16xf32> {
    %acc = tensor.empty() : tensor<4x16xf32>
    %res = hfusion.matmul_mx {lhsFormat = 1 : i32, rhsFormat = 1 : i32}
      ins(%arg0, %arg1, %arg2, %arg3 :
          tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>,
          tensor<4x1xi8>, tensor<16x1xi8>)
      outs(%acc : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %res : tensor<4x16xf32>
  }
}
