// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops='always-inline=true' -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_graph_a_0(
// CHECK-SAME: attributes
// CHECK-SAME: hacc.always_inline
func.func @test_graph_a(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.matmul ins(%arg0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%3 : tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %7 = linalg.matmul ins(%arg0, %5 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %9 = linalg.matmul ins(%arg0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %11 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%5, %9 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %11, %7 : tensor<?x?xf32>, tensor<?x?xf32>
}

