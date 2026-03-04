// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops -split-input-file %s | FileCheck %s

// CHECK-NOT: func.func @test_function_no_definition_0(
func.func @test_function_no_definition(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) attributes {} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg1 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%3 : tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %9 = linalg.elemwise_binary {sub, fun = #linalg.binary_fn<sub>} ins(%7, %3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %11 = linalg.matmul ins(%7, %5 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %13 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%11 : tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %14 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %15 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg2, %7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %16 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %17 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %18 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %19 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%13, %5 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %arg0, %7, %9, %11, %15, %17, %19 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

// -----


// CHECK-LABEL: func.func @test_function_defined_as_host_0(
// CHECK-SAME: hacc.function_kind<DEVICE>
// CHECK-LABEL: func.func @test_function_defined_as_host_1(
// CHECK-SAME: hacc.function_kind<DEVICE>
// CHECK-LABEL: func.func @test_function_defined_as_host(
// CHECK-SAME: hacc.function_kind<HOST>
func.func @test_function_defined_as_host(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg1 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%3 : tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %9 = linalg.elemwise_binary {sub, fun = #linalg.binary_fn<sub>} ins(%7, %3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %11 = linalg.matmul ins(%7, %5 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %13 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%11 : tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %14 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %15 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg2, %7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %16 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %17 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %18 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %19 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%13, %5 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %arg0, %7, %9, %11, %15, %17, %19 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}
