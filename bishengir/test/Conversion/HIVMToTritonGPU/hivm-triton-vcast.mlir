// RUN: bishengir-opt %s --convert-hivm-to-tritongpu="allow-return-value=true" | FileCheck %s

// CHECK-LABEL: func.func @vcast_i64_to_i32(
// CHECK-NOT: hivm.hir.vcast
// CHECK: %[[CAST:.*]] = arith.trunci %arg0 {round_mode = #hivm.round_mode<truncwithoverflow>} : tensor<32xi64> to tensor<32xi32>
// CHECK: return %[[CAST]] : tensor<32xi32>
func.func @vcast_i64_to_i32(%arg0: tensor<32xi64>) -> tensor<32xi32> {
  %empty = tensor.empty() : tensor<32xi32>
  %cast = hivm.hir.vcast ins(%arg0 : tensor<32xi64>)
      outs(%empty : tensor<32xi32>) round_mode = <truncwithoverflow>
      cast = <cast_signed> -> tensor<32xi32>
  return %cast : tensor<32xi32>
}
