// RUN: bishengir-opt %s -hfusion-remove-cache-io -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_remove_cache_io_0
// CHECK-NOT: load
// CHECK-NOT: store
func.func @test_remove_cache_io_0(%arg0: tensor<1x2048xi32>) -> tensor<2048xi32> {
  %0 = tensor.empty() : tensor<2048xi32>
  %1 = tensor.empty() : tensor<2047x2047xf32>
  %2 = tensor.empty() : tensor<2048xi32>
  %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x2048xi32> into tensor<2048xi32>
  %3 = hfusion.load ins(%collapsed : tensor<2048xi32>) outs(%2 : tensor<2048xi32>) -> tensor<2048xi32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %3 : tensor<2048xi32>, tensor<2048xi32>) outs(%0 : tensor<2048xi32>) -> tensor<2048xi32>
  %5 = hfusion.store ins(%4 : tensor<2048xi32>) outs(%2 : tensor<2048xi32>) -> tensor<2048xi32>
  return %5 : tensor<2048xi32>
}
