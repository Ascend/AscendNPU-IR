// RUN: bishengir-opt %s --transform-interpreter --cse --canonicalize-ext | FileCheck %s

// CHECK: scf.for
// CHECK: scf.for
// CHECK: bufferization.to_tensor
module {
  func.func @to_tensor(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = memref.alloc() : memref<2x3xf32>
    %1 = bufferization.to_tensor %0 : memref<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["bufferization.to_tensor"]} in %0 : (!transform.any_op) -> !transform.op<"bufferization.to_tensor">
      %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [1, 2] : (!transform.op<"bufferization.to_tensor">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}