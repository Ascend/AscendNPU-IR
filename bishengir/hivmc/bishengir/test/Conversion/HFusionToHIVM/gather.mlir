// RUN: bishengir-opt -convert-hfusion-to-hivm %s | FileCheck %s
// RUN: bishengir-opt -convert-to-hivm-pipeline %s | FileCheck %s
// CHECK-LABEL: test_vgather
func.func @test_vgather(%src:tensor<16x16xf16>, %idx:tensor<16x4xi32>) -> tensor<16x4xf16>{
  %init = tensor.empty() : tensor<16x4xf16>
  // CHECK: vgather
  %res = hfusion.gather ins(%src, %idx : tensor<16x16xf16>, tensor<16x4xi32>) outs(%init:tensor<16x4xf16>) axis = 1 -> tensor<16x4xf16>
  return %res : tensor<16x4xf16>
}
