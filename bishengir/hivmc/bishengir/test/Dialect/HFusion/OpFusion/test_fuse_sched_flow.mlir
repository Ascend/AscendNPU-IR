// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops='output-mode=single' %s | bishengir-opt -hfusion-auto-schedule | FileCheck %s
// CHECK-LABEL: @add_mul_reduce_0
func.func @add_mul_reduce(%arg0: tensor<?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<f32> attributes {hacc.function_kind = #hacc.function_kind<HOST>}
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?xf32>
  %2 = tensor.empty(%0) : tensor<?xf32>
  %3 = linalg.elemwise_binary { mul, fun = #linalg.binary_fn<mul> } ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%2 : tensor<?xf32>) -> tensor<?xf32>

  %4 = tensor.empty(%0) : tensor<?xf32>

  %5 = linalg.elemwise_binary { add, fun = #linalg.binary_fn<add> } ins(%3, %arg2 : tensor<?xf32>, tensor<?xf32>) outs(%4 : tensor<?xf32>) -> tensor<?xf32>
  // CHECK: call @add_mul_reduce_0
  %6 = tensor.empty() : tensor<f32>
  %7 = linalg.reduce { arith.addf } ins(%5 : tensor<?xf32>) outs(%6 : tensor<f32>) dimensions = [0]

  return %7 : tensor<f32>
}
