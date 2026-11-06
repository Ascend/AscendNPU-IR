// RUN: bishengir-opt --hivm-verify-capability=strict=false --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @unknown_still_printed
// CHECK: hivm.hir.fixpipe
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @unknown_still_printed(%src: tensor<16x16xf32>, %dst: tensor<16x16xf32>) -> tensor<16x16xf32> {
    // expected-remark @below {{result is unknown; src or dst memory scope is not on this operation and was not set for this query}}
    %0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<normal>}
      ins(%src : tensor<16x16xf32>) outs(%dst : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}
