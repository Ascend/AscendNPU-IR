// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20" -split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @test_static_min_tiling_size_dim_size_tiling_function
// CHECK: %[[c4:.*]] = arith.constant 4 : i64
// CHECK: return {{.*}}, %[[c4]]
func.func @test_static_min_tiling_size_dim_size(%arg0: tensor<4xf32>, %arg1: tensor<4xbf16>) -> (tensor<f32>, tensor<f32>) 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %cst = arith.constant 4.000000e+02 : f32
  %0 = tensor.empty() : tensor<4xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<4xbf16>) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %1 : tensor<4xf32>, tensor<4xf32>) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
  %3 = tensor.empty() : tensor<f32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
  %reduced = linalg.reduce ins(%2 : tensor<4xf32>) outs(%4 : tensor<f32>) dimensions = [0] 
    (%in: f32, %init: f32) {
      %8 = arith.addf %in, %init : f32
      linalg.yield %8 : f32
    }
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%reduced, %cst : tensor<f32>, f32) outs(%3 : tensor<f32>) -> tensor<f32>
  return %reduced, %6 : tensor<f32>, tensor<f32>
}

// -----

// CHECK-LABEL: func.func @test_transpose_size_alignment_0_tiling_function
// CHECK: %[[size0:.*]] = arith.constant 16 : i64
// CHECK: return {{.*}}, %[[size0]]
func.func @test_transpose_size_alignment_0(%arg0: tensor<3072x3072xf32>) -> tensor<3072x3072xbf16> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %0 = tensor.empty() : tensor<3072x3072xf32>
  %1 = tensor.empty() : tensor<3072x3072xbf16>
  %transposed = linalg.transpose ins(%arg0 : tensor<3072x3072xf32>) outs(%0 : tensor<3072x3072xf32>) permutation = [1, 0] 
  %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%transposed : tensor<3072x3072xf32>) outs(%1 : tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
  return %2 : tensor<3072x3072xbf16>
}

// -----

// CHECK-LABEL: func.func @mlir_fused_native_layer_norm_0_tiling_function(
module {
  func.func @mlir_fused_native_layer_norm_0(%arg0: tensor<2048x2048xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<2048x1xf32>, tensor<2048x1xf32>, tensor<2048x2048xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %cst = arith.constant 9.99999974E-6 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 2.048000e+03 : f32
    %0 = tensor.empty() : tensor<2048xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
    %2 = tensor.empty() : tensor<2048x2048xf32>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<2048xf32>) outs(%2 : tensor<2048x2048xf32>) dimensions = [0]
    %broadcasted_3 = linalg.broadcast ins(%arg2 : tensor<2048xf32>) outs(%2 : tensor<2048x2048xf32>) dimensions = [0]
    %reduced = linalg.reduce ins(%arg0 : tensor<2048x2048xf32>) outs(%1 : tensor<2048xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced, %cst_2 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
    %broadcasted_4 = linalg.broadcast ins(%3 : tensor<2048xf32>) outs(%2 : tensor<2048x2048xf32>) dimensions = [1]
    %expanded = tensor.expand_shape %3 [[0, 1]] output_shape [2048, 1] : tensor<2048xf32> into tensor<2048x1xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %broadcasted_4 : tensor<2048x2048xf32>, tensor<2048x2048xf32>) outs(%2 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %4 : tensor<2048x2048xf32>, tensor<2048x2048xf32>) outs(%2 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    %reduced_5 = linalg.reduce ins(%5 : tensor<2048x2048xf32>) outs(%1 : tensor<2048xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced_5, %cst_2 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%6, %cst : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
    %8 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%7 : tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
    %9 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%8 : tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
    %broadcasted_6 = linalg.broadcast ins(%9 : tensor<2048xf32>) outs(%2 : tensor<2048x2048xf32>) dimensions = [1]
    %expanded_7 = tensor.expand_shape %9 [[0, 1]] output_shape [2048, 1] : tensor<2048xf32> into tensor<2048x1xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %broadcasted_6 : tensor<2048x2048xf32>, tensor<2048x2048xf32>) outs(%2 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %broadcasted : tensor<2048x2048xf32>, tensor<2048x2048xf32>) outs(%2 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%11, %broadcasted_3 : tensor<2048x2048xf32>, tensor<2048x2048xf32>) outs(%2 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    return %expanded, %expanded_7, %12 : tensor<2048x1xf32>, tensor<2048x1xf32>, tensor<2048x2048xf32>
  }
}

// -----

// CHECK-LABEL: func.func @test_tile_size_greater_than_dim_size_tiling_function(
// CHECK-NOT:arith.constant 3 : index
// CHECK-NOT:arith.constant 5 : index
// CHECK-NOT:arith.constant 14 : index
// CHECK: func.func @test_tile_size_greater_than_dim_size(
func.func @test_tile_size_greater_than_dim_size(%arg0: tensor<3x5x5x7x2x1xi8>) -> tensor<5x7x2x5x3xi8> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1], [2], [3, 4, 5]] : tensor<3x5x5x7x2x1xi8> into tensor<3x5x5x14xi8>
  %0 = tensor.empty() : tensor<14x5x5x3xi8>
  %transposed = linalg.transpose ins(%collapsed : tensor<3x5x5x14xi8>) outs(%0 : tensor<14x5x5x3xi8>) permutation = [3, 1, 2, 0] 
  %1 = tensor.empty() : tensor<5x14x5x3xi8>
  %transposed_0 = linalg.transpose ins(%transposed : tensor<14x5x5x3xi8>) outs(%1 : tensor<5x14x5x3xi8>) permutation = [1, 0, 2, 3] 
  %expanded = tensor.expand_shape %transposed_0 [[0], [1, 2], [3], [4]] output_shape [5, 7, 2, 5, 3] : tensor<5x14x5x3xi8> into tensor<5x7x2x5x3xi8>
  return %expanded : tensor<5x7x2x5x3xi8>
}