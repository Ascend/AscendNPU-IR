// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s | FileCheck %s

// CHECK-LABEL: func.func @vdeinterleave_channel0
func.func @vdeinterleave_channel0(%arg0: tensor<2x8xf32>) -> tensor<2x4xf32> {
  %empty = tensor.empty() : tensor<2x4xf32>
  // CHECK: tt.reshape
  // CHECK: %[[LHS:.*]], %{{.*}} = tt.split
  // CHECK: return %[[LHS]]
  %0 = hivm.hir.vdeinterleave ins(%arg0 : tensor<2x8xf32>) outs(%empty : tensor<2x4xf32>) channel_num = 2 index_mode = <CHANNEL_0> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
