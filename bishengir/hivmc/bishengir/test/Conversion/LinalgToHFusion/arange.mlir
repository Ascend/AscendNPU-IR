// RUN: bishengir-opt -convert-linalg-to-hfusion %s -split-input-file -verify-diagnostics | FileCheck %s

#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @test_arange
// CHECK-SAME:    (%[[ARG:.*]]: tensor<6xi32>) -> tensor<6xi32> {
func.func @test_arange(%arg0 : tensor<6xi32>) -> tensor<6xi32> {
  // CHECK-NEXT: %[[CONST1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[CONST0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[RET:.*]] = hfusion.arange offset[%[[CONST0]]] strides[%[[CONST1]]] outs(%[[ARG]] : tensor<6xi32>) -> tensor<6xi32>
  %ret = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg0 : tensor<6xi32>) {
    ^bb0(%out: i32):
      %0 = linalg.index 0 : index
      %1 = arith.index_cast %0 : index to i32
      linalg.yield %1 : i32
    } -> (tensor<6xi32>)
  return %ret : tensor<6xi32>
}
