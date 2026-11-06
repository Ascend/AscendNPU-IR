// REQUIRES: bishengir-hifloat8
// RUN: bishengir-opt %s --execution-engine-convert-hivm-to-upstream | FileCheck %s
// CHECK-LABEL: func.func @hybrid
// CHECK: hfusion.cast
// CHECK-SAME: round_mode = #hfusion.round_mode<hybrid>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @hybrid(%src: tensor<64xf32>, %dst: tensor<64xf8HIF>) -> tensor<64xf8HIF> {
  %0 = hivm.hir.vcast ins(%src : tensor<64xf32>) outs(%dst : tensor<64xf8HIF>) round_mode = #hivm.round_mode<hybrid> -> tensor<64xf8HIF>
  return %0 : tensor<64xf8HIF>
}
}
