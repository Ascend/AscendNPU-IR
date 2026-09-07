// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s | FileCheck %s

// CHECK-LABEL: func.func @vgather
func.func @vgather(%arg0: tensor<4x8xf32>, %idx: tensor<4x8xi32>) -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  // CHECK: tt.gather
  %0 = hivm.hir.vgather ins(%arg0 : tensor<4x8xf32>) indices(%idx : tensor<4x8xi32>) outs(%empty : tensor<4x8xf32>) gather_axis = 1 -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
