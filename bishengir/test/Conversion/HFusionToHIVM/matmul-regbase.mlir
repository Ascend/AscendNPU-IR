// RUN: bishengir-opt -convert-hfusion-to-hivm="mm-map-mode=macro_instr" -canonicalize-ext %s -split-input-file -verify-diagnostics | FileCheck %s

// On reg-based arches this conversion does not absorb a linalg.transpose into
// the mmad transpose flags; it emits a hivm.hir.vtranspose and NormalizeMatmul
// folds it later. See normalize-matmul.mlir for the folding itself.

// CHECK-LABEL: func.func @test_batchMmadL1_with_transpose
// CHECK: hivm.hir.vtranspose
// CHECK: hivm.hir.batchMmadL1
// CHECK-NOT: batchMmadL1 {a_transpose}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @test_batchMmadL1_with_transpose(%mb: tensor<2x128x256xf16>) -> tensor<2x256x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %mc = tensor.empty() : tensor<2x256x256xf32>
  %mc_fill = linalg.fill ins(%cst : f32) outs(%mc : tensor<2x256x256xf32>) -> tensor<2x256x256xf32>
  %ma = tensor.empty() : tensor<2x128x256xf16>
  %ma_transpose_init = tensor.empty() : tensor<2x256x128xf16>
  %ma_transpose_res = linalg.transpose ins(%ma : tensor<2x128x256xf16>)
                                       outs(%ma_transpose_init : tensor<2x256x128xf16>) permutation = [0, 2, 1]
  %ret = linalg.batch_matmul ins(%ma_transpose_res, %mb : tensor<2x256x128xf16>, tensor<2x128x256xf16>)
                             outs(%mc_fill: tensor<2x256x256xf32>) -> tensor<2x256x256xf32>
  return %ret : tensor<2x256x256xf32>
}
}
