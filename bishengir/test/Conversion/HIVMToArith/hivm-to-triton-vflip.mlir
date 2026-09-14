// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s | FileCheck %s

// CHECK-LABEL: func.func @vflip
func.func @vflip(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<8xf32>
  // CHECK: tt.make_range
  // CHECK: arith.subi
  // CHECK: tt.gather
  %0 = hivm.hir.vflip ins(%arg0 : tensor<8xf32>) outs(%empty : tensor<8xf32>) flip_axis = 0 -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
