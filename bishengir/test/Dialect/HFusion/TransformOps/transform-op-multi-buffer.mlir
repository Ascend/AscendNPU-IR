// RUN: bishengir-opt -transform-interpreter -split-input-file -allow-unregistered-dialect %s | FileCheck %s

module attributes { transform.with_named_sequence } {
  func.func @multi_buffer_tensor(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %0 = linalg.exp ins(%arg0: tensor<4x8x16xf32>) outs(%arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    // CHECK: annotation.mark %0 {hfusion.multi_buffer = 2 : i32} : tensor<4x8x16xf32>

    return %0 : tensor<4x8x16xf32>
  }

  func.func @multi_buffer_two_result() -> (memref<1xf32>, memref<2xf32>) {
    // CHECK: %0:2 = "test.source"() : () -> (memref<1xf32>, memref<2xf32>)
    // CHECK: annotation.mark %0#0 {hfusion.multi_buffer = 2 : i32} : memref<1xf32>
    // CHECK: annotation.mark %0#1 {hfusion.multi_buffer = 2 : i32} : memref<2xf32>

    %0, %1 = "test.source"() : () ->  (memref<1xf32>, memref<2xf32>)

    return %0, %1 : memref<1xf32>, memref<2xf32>
  }

  func.func @multi_buffer_three_result() -> (i32, f32, f32) {
    // CHECK: %[[result1:.*]], %[[result2:.*]], %[[result3:.*]] = "test.three_result"() <{kind = 1 : i64}> : () -> (i32, f32, f32)
    // CHECK: annotation.mark %[[result1]] {hfusion.multi_buffer = 2 : i32} : i32
    // CHECK: annotation.mark %[[result2]] {hfusion.multi_buffer = 2 : i32} : f32
    // CHECK: annotation.mark %[[result3]] {hfusion.multi_buffer = 2 : i32} : f32

    %0:3 = "test.three_result"() {kind = 1} : () ->  (i32, f32, f32)

    return %0#0, %0#1, %0#2 : i32, f32, f32
  }

  func.func @multi_buffer_tensor_factor3(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %0 = linalg.abs ins(%arg0: tensor<4x8x16xf32>) outs(%arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    // CHECK: annotation.mark %0 {hfusion.multi_buffer = 3 : i32} : tensor<4x8x16xf32>

    return %0 : tensor<4x8x16xf32>
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["test.source", "test.three_result"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.multi_buffer %0, %1 : !transform.any_op, !transform.any_op

    %2 = transform.structured.match ops{["linalg.abs"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.multi_buffer %2 factor = 3 : !transform.any_op

    transform.yield
  }
}

