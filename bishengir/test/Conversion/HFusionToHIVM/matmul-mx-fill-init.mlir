// RUN: bishengir-opt -convert-hfusion-to-hivm %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_matmulscale_fill_init
  // Zero fill is converted to vbrc and treated as a non-empty init (accumulate).
  // CHECK: %[[FALSE:.*]] = arith.constant false
  // CHECK: hivm.hir.vbrc
  // CHECK: hivm.hir.mmadmxL1
  // CHECK-SAME: %[[FALSE]]
  func.func @test_matmulscale_fill_init(
      %arg0: tensor<4x8xf8E5M2>, %arg1: tensor<8x16xf8E5M2>,
      %arg2: tensor<4x1xi8>, %arg3: tensor<16x1xi8>)
      -> tensor<4x16xf32> {
    %cst = arith.constant 0.0 : f32
    %acc0 = tensor.empty() : tensor<4x16xf32>
    %acc = linalg.fill ins(%cst : f32) outs(%acc0 : tensor<4x16xf32>) -> tensor<4x16xf32>
    %res = hfusion.matmul_mx
      ins(%arg0, %arg1, %arg2, %arg3 :
          tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>,
          tensor<4x1xi8>, tensor<16x1xi8>)
      outs(%acc : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %res : tensor<4x16xf32>
  }
}
