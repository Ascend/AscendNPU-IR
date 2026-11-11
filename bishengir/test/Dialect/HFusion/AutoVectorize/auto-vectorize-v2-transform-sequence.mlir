// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="emit-transform-sequence" -hfusion-auto-vectorize-interpreter -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @kernel_a
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK-LABEL: func.func @kernel_b
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
module {
  func.func @kernel_a(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, parallel_mode = "simd"} {
    %empty = tensor.empty() : tensor<16xf32>
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func @kernel_b(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32> attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, parallel_mode = "simd"} {
    %empty = tensor.empty() : tensor<32xf32>
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
    return %0 : tensor<32xf32>
  }
}
