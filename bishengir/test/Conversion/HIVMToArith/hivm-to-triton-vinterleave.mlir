// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s | FileCheck %s

// CHECK-LABEL: func.func @vinterleave
func.func @vinterleave(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x8xf32> {
  %empty = tensor.empty() : tensor<2x8xf32>
  // CHECK: tt.join
  // CHECK: tt.reshape
  %0 = hivm.hir.vinterleave ins(%arg0, %arg1 : tensor<2x4xf32>, tensor<2x4xf32>) outs(%empty : tensor<2x8xf32>) interleave_channel_nums = 2 -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
