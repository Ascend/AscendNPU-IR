// RUN: bishengir-opt -test-can-fuse -split-input-file -verify-diagnostics %s

// expected-remark@below {{This function is fusible!}}
func.func @model_0(%arg0: tensor<5x1xf32>) -> tensor<5x1xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%1 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %cst : tensor<5x1xf32>, f32) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %3 : f32, tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %4 : tensor<5x1xf32>
}

// -----

// expected-remark@below {{This function is not fusible!}}
func.func @test_unknown(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%arg2 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %5 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg2, %3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %9 = linalg.matmul ins(%arg2, %7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %11 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%7 : tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = tensor.empty(%0, %0, %1) : tensor<?x?x?xf32>
  %13 = linalg.broadcast ins(%arg2 : tensor<?x?xf32>) outs(%12: tensor<?x?x?xf32>) dimensions = [0]
  %14 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %15 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %16 = tensor.empty(%0, %0, %1, %0) : tensor<?x?x?x?xf32>
  %17 = linalg.broadcast ins(%13 : tensor<?x?x?xf32>) outs(%16: tensor<?x?x?x?xf32>) dimensions = [3]
  %18 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %19 = linalg.transpose ins(%15 : tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) permutation = [0, 1]
  return %arg1, %9, %11, %13, %15, %17, %19 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?xf32>
}
