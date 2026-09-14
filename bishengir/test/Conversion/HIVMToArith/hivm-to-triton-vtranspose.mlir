// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s | FileCheck %s

// CHECK-LABEL: func.func @vtranspose
func.func @vtranspose(%arg0: tensor<4x8xf32>) -> tensor<8x4xf32> {
  %empty = tensor.empty() : tensor<8x4xf32>
  // CHECK: tt.trans
  %0 = hivm.hir.vtranspose ins(%arg0 : tensor<4x8xf32>) outs(%empty : tensor<8x4xf32>) permutation = [1, 0] -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}
