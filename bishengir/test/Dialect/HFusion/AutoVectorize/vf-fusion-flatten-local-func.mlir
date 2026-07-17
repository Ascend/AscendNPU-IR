// RUN: bishengir-opt %s --lower-hfusion-pipeline="enable-triton-kernel-compile enable-vf-fusion target=Ascend950PR_9589" --mlir-print-ir-after=hfusion-flatten-ops --mlir-print-ir-after-change 2>&1 | FileCheck %s

// Regression test for the layernorm case.
// The extra flatten pass in the SIMD VFFusion path should flatten each
// VFFusion-produced local function before subsequent auto-vectorization.

// CHECK: IR Dump After FlattenOps
// CHECK-LABEL: func.func private @layernorm_vf_fused(
// CHECK-SAME: %[[ARG2:arg[0-9]+]]: tensor<4x64x16xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG2]] {{\[\[}}0], [1, 2]] : tensor<4x64x16xf32> into tensor<4x1024xf32>
// CHECK-NOT: dimensions = [1, 2]
// CHECK: linalg.reduce ins(%[[COLLAPSED]] : tensor<4x1024xf32>) outs(%{{.*}} : tensor<4xf32>) dimensions = [1]
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9589">
} {
  func.func private @layernorm_vf_fused(
      %arg0: f32,
      %arg1: tensor<4xf32>,
      %arg2: tensor<4x64x16xf32>)
      -> (tensor<4xf32>, tensor<4xf32>)
      attributes {
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        mix_mode = "aiv",
        parallel_mode = "simd"
      } {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<4xf32>) -> tensor<4xf32>
    %reduced = linalg.reduce ins(%arg2 : tensor<4x64x16xf32>) outs(%0 : tensor<4xf32>) dimensions = [1, 2]
      (%in: f32, %init: f32) {
        %1 = arith.addf %in, %init : f32
        linalg.yield %1 : f32
      }
    return %0, %reduced : tensor<4xf32>, tensor<4xf32>
  }

  func.func @caller(
      %arg0: f32,
      %arg1: tensor<4xf32>,
      %arg2: tensor<4x64x16xf32>)
      -> (tensor<4xf32>, tensor<4xf32>)
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        mix_mode = "aiv",
        parallel_mode = "simd"
      } {
    %0:2 = func.call @layernorm_vf_fused(%arg0, %arg1, %arg2)
        : (f32, tensor<4xf32>, tensor<4x64x16xf32>) -> (tensor<4xf32>, tensor<4xf32>)
    return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }
}
