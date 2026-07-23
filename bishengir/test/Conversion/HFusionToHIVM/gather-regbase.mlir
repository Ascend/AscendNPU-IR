// RUN: bishengir-opt %s --hfusion-decompose="hfusion-decompose-phase=after-hfusion-flatten" -convert-hfusion-to-hivm -split-input-file -verify-diagnostics | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_gather_tail_axis_regbase
  func.func @test_gather_tail_axis_regbase(
      %src: tensor<5x11xf16>, %idx: tensor<5x7xi32>) -> tensor<5x7xf16> {
    %init = tensor.empty() : tensor<5x7xf16>
    // CHECK: hivm.hir.vgather {{.*}}gather_axis = 1
    %res = hfusion.gather ins(%src, %idx : tensor<5x11xf16>, tensor<5x7xi32>)
        outs(%init : tensor<5x7xf16>) axis = 1 -> tensor<5x7xf16>
    return %res : tensor<5x7xf16>
  }

  // CHECK-LABEL: func.func @test_gather_non_tail_axis_regbase
  func.func @test_gather_non_tail_axis_regbase(
      %src: tensor<5x11xf16>, %idx: tensor<7x11xi32>) -> tensor<7x11xf16> {
    %init = tensor.empty() : tensor<7x11xf16>
    // CHECK: hivm.hir.vgather {{.*}}gather_axis = 0
    %res = hfusion.gather ins(%src, %idx : tensor<5x11xf16>, tensor<7x11xi32>)
        outs(%init : tensor<7x11xf16>) axis = 0 -> tensor<7x11xf16>
    return %res : tensor<7x11xf16>
  }
}
