// RUN: bishengir-opt --hfusion-fuse-ops --split-input-file %s | FileCheck %s


// CHECK: func.func @test_zero_dimension_0
// CHECK: hfusion.cast
// CHECK: linalg.elemwise_binary
// CHECK: linalg.fill
// CHECK: linalg.reduce
// CHECK: hfusion.cast
// CHECK: tensor.expand_shape
// CHECK: func.func @test_zero_dimension
func.func @test_zero_dimension(%arg0: tensor<1536xbf16>) -> tensor<1xbf16> attributes {OperatorType = "Reduce", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>, mindspore_kernel, process = "aicore"} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1536xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<1536xbf16>) outs(%0 : tensor<1536xf32>) -> tensor<1536xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %1 : tensor<1536xf32>, tensor<1536xf32>) outs(%0 : tensor<1536xf32>) -> tensor<1536xf32>
  %3 = tensor.empty() : tensor<f32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
  %reduced = linalg.reduce ins(%2 : tensor<1536xf32>) outs(%4 : tensor<f32>) dimensions = [0] 
    (%in: f32, %init: f32) {
      %7 = arith.addf %in, %init : f32
      linalg.yield %7 : f32
    }
  %5 = tensor.empty() : tensor<bf16>
  %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%reduced : tensor<f32>) outs(%5 : tensor<bf16>) -> tensor<bf16>
  %expanded = tensor.expand_shape %6 [] output_shape [1] : tensor<bf16> into tensor<1xbf16>
  return %expanded : tensor<1xbf16>
}

// -----

// CHECK-LABEL: @fuse_zero_rank_0(
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: tensor.collapse_shape
// CHECK: linalg.fill
// CHECK: linalg.reduce
// CHECK: arith.sitofp
// CHECK: linalg.elemwise_binary
// CHECK: linalg.elemwise_binary
// CHECK-LABEL: @fuse_zero_rank(
// CHECK: call @fuse_zero_rank
module {
  func.func @fuse_zero_rank(%arg0: tensor<?x2xf32>, %arg1: i64) -> tensor<f32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x2xf32> into tensor<?xf32>
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<f32>) -> tensor<f32>
    %reduced = linalg.reduce ins(%collapsed : tensor<?xf32>) outs(%1 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %5 = arith.addf %in, %init : f32
        linalg.yield %5 : f32
      }
    %2 = arith.sitofp %arg1 : i64 to f32
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced, %2 : tensor<f32>, f32) outs(%0 : tensor<f32>) -> tensor<f32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%3, %cst : tensor<f32>, f32) outs(%0 : tensor<f32>) -> tensor<f32>
    return %4 : tensor<f32>
  }
}