// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="SHALLOW_CV" -hfusion-fuse-ops -split-input-file %s | FileCheck %s --check-prefix=SHALLOW-CV

// SHALLOW-CV-LABEL: func.func @testA_0(
// SHALLOW-CV: linalg.elemwise_unary
// SHALLOW-CV: linalg.elemwise_binary
// SHALLOW-CV: linalg.elemwise_unary
// SHALLOW-CV: linalg.matmul
// SHALLOW-CV: linalg.elemwise_unary
// SHALLOW-CV-LABEL: func.func @testA(
// SHALLOW-CV: call @testA_0(
func.func @testA(%arg0: tensor<7x7xf32>, %arg1: tensor<7x7xf32>, %arg2: tensor<7x7xf32>) -> (tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7xf32>
  %2 = tensor.empty() : tensor<7x7xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%arg2 : tensor<7x7xf32>) outs(%2 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %4 = tensor.empty() : tensor<7x7xf32>
  %5 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg2, %3 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%4 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %6 = tensor.empty() : tensor<7x7xf32>
  %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<7x7xf32>) outs(%6 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %8 = tensor.empty() : tensor<7x7xf32>
  %9 = linalg.matmul ins(%arg2, %7 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%8 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %10 = tensor.empty() : tensor<7x7xf32>
  %11 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%7 : tensor<7x7xf32>) outs(%10 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %12 = tensor.empty() : tensor<7x7x7xf32>
  %13 = linalg.broadcast ins(%arg2 : tensor<7x7xf32>) outs(%12: tensor<7x7x7xf32>) dimensions = [0]
  %14 = tensor.empty() : tensor<7x7xf32>
  %15 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<7x7xf32>) outs(%14 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %16 = tensor.empty() : tensor<7x7x7x7xf32>
  %17 = linalg.broadcast ins(%13 : tensor<7x7x7xf32>) outs(%16: tensor<7x7x7x7xf32>) dimensions = [3]
  %18 = tensor.empty() : tensor<7x7xf32>
  %19 = linalg.transpose ins(%15 : tensor<7x7xf32>) outs(%18 : tensor<7x7xf32>) permutation = [0, 1]
  return %arg1, %9, %11, %13, %15, %17, %19 : tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7xf32>
}

// -----


// SHALLOW-CV-LABEL: func.func @testB_0(
// SHALLOW-CV: linalg.matmul
// SHALLOW-CV: linalg.broadcast
// SHALLOW-CV: linalg.elemwise_unary
// SHALLOW-CV: linalg.matmul
// SHALLOW-CV: linalg.elemwise_binary
// SHALLOW-CV-LABEL: func.func @testB(
// SHALLOW-CV: call @testB_0(
func.func @testB(%arg0: tensor<7x7xf32>, %arg1: tensor<7x7xf32>, %arg2: tensor<7x7xf32>) -> (tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7xf32>
  %2 = tensor.empty() : tensor<7x7xf32>
  %3 = linalg.transpose ins(%arg2 : tensor<7x7xf32>) outs(%2 : tensor<7x7xf32>) permutation = [0, 1]
  %4 = tensor.empty() : tensor<7x7xf32>
  %5 = linalg.transpose ins(%arg2 : tensor<7x7xf32>) outs(%4 : tensor<7x7xf32>) permutation = [0, 1]
  %6 = tensor.empty() : tensor<7x7xf32>
  %7 = linalg.matmul ins(%5, %5 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%6 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %8 = tensor.empty() : tensor<7x7x7xf32>
  %9 = linalg.broadcast ins(%arg2 : tensor<7x7xf32>) outs(%8: tensor<7x7x7xf32>) dimensions = [2]
  %10 = tensor.empty() : tensor<7x7xf32>
  %11 = linalg.transpose ins(%7 : tensor<7x7xf32>) outs(%10 : tensor<7x7xf32>) permutation = [0, 1]
  %12 = tensor.empty() : tensor<7x7x7xf32>
  %13 = linalg.broadcast ins(%7 : tensor<7x7xf32>) outs(%12: tensor<7x7x7xf32>) dimensions = [2]
  %14 = tensor.empty() : tensor<7x7xf32>
  %15 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg2 : tensor<7x7xf32>) outs(%14 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %16 = tensor.empty() : tensor<7x7xf32>
  %17 = linalg.matmul ins(%15, %7 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%16 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %18 = tensor.empty() : tensor<7x7xf32>
  %19 = linalg.elemwise_binary {min_signed, fun = #linalg.binary_fn<min_signed>} ins(%5, %15 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%18 : tensor<7x7xf32>) -> tensor<7x7xf32>
  return %arg0, %arg1, %3, %9, %11, %13, %17, %19 : tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>
}

// -----

// SHALLOW-CV-LABEL: func.func @testC_0(
// SHALLOW-CV: linalg.matmul
// SHALLOW-CV: linalg.broadcast
// SHALLOW-CV: linalg.broadcast
// SHALLOW-CV: linalg.elemwise_binary
// SHALLOW-CV: linalg.elemwise_unary
// SHALLOW-CV-LABEL: func.func @testC(
// SHALLOW-CV: call @testC_0(
func.func @testC(%arg0: tensor<7x7xf32>, %arg1: tensor<7x7xf32>, %arg2: tensor<7x7xf32>) -> (tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7xf32>)  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7xf32>
  %2 = tensor.empty() : tensor<7x7xf32>
  %3 = linalg.matmul ins(%arg1, %arg2 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%2 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %4 = tensor.empty() : tensor<7x7xf32>
  %5 = linalg.transpose ins(%arg2 : tensor<7x7xf32>) outs(%4 : tensor<7x7xf32>) permutation = [0, 1]
  %6 = tensor.empty() : tensor<7x7x7xf32>
  %7 = linalg.broadcast ins(%3 : tensor<7x7xf32>) outs(%6: tensor<7x7x7xf32>) dimensions = [2]
  %8 = tensor.empty() : tensor<7x7x7xf32>
  %9 = linalg.broadcast ins(%5 : tensor<7x7xf32>) outs(%8: tensor<7x7x7xf32>) dimensions = [2]
  %10 = tensor.empty() : tensor<7x7xf32>
  %11 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<7x7xf32>) outs(%10 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %12 = tensor.empty() : tensor<7x7x7xf32>
  %13 = linalg.elemwise_binary {min_signed, fun = #linalg.binary_fn<min_signed>} ins(%9, %7 : tensor<7x7x7xf32>, tensor<7x7x7xf32>) outs(%12 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  %14 = tensor.empty() : tensor<7x7xf32>
  %15 = linalg.reduce {arith.addf} ins(%9 : tensor<7x7x7xf32>) outs(%14: tensor<7x7xf32>) dimensions = [0]
  %16 = tensor.empty() : tensor<7x7x7x7xf32>
  %17 = linalg.broadcast ins(%9 : tensor<7x7x7xf32>) outs(%16: tensor<7x7x7x7xf32>) dimensions = [0]
  %18 = tensor.empty() : tensor<7x7x7xf32>
  %19 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%9 : tensor<7x7x7xf32>) outs(%18 : tensor<7x7x7xf32>) -> tensor<7x7x7xf32>
  return %arg0, %3, %11, %13, %15, %17, %19 : tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7x7xf32>, tensor<7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7xf32>
}
