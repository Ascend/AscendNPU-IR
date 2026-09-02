// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9589 --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
// CHECK-LABEL: @test_simpify_vsub
// CHECK-NEXT: %cst = arith.constant -1.000000e+00 : f32
// CHECK-NEXT: %0 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT: %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %cst : tensor<1024xf32>, f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT: return %1 : tensor<1024xf32>
func.func @test_simpify_vsub(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1024xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%cst, %arg0 : f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
  return %1 : tensor<1024xf32>
}
