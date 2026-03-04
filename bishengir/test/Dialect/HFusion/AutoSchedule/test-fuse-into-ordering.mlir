// RUN: bishengir-opt %s -hfusion-auto-schedule -split-input-file | FileCheck %s

// CHECK: mlir_fused_add_native_layer_norm_native_layer_norm_backward_13
module {
  func.func @mlir_fused_add_native_layer_norm_native_layer_norm_backward_13(%arg0: tensor<6000x384xf32>, %arg1: tensor<4x1500x384xf32>, %arg2: tensor<6000x384xf32>, %arg3: tensor<384xf32>, %arg4: tensor<4x384x1500xf32>, %arg5: tensor<1500x384xf32>, %arg6: tensor<4x1500x1xf32>, %arg7: tensor<4x1500x1xf32>, %arg8: tensor<4x1500x384xf32>) -> (tensor<4x1500x384xf32>, tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 3.840000e+02 : f32
    %0 = tensor.empty() : tensor<4x1500x384xf32>
    %1 = tensor.empty() : tensor<4x1500xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<4x1500xf32>) -> tensor<4x1500xf32>
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2]] output_shape [4, 1500, 384] : tensor<6000x384xf32> into tensor<4x1500x384xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%expanded, %arg1 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    %expanded_1 = tensor.expand_shape %arg2 [[0, 1], [2]] output_shape [4, 1500, 384] : tensor<6000x384xf32> into tensor<4x1500x384xf32>
    %broadcasted = linalg.broadcast ins(%arg3 : tensor<384xf32>) outs(%0 : tensor<4x1500x384xf32>) dimensions = [0, 1]
    %transposed = linalg.transpose ins(%arg4 : tensor<4x384x1500xf32>) outs(%0 : tensor<4x1500x384xf32>) permutation = [0, 2, 1]
    %broadcasted_2 = linalg.broadcast ins(%arg5 : tensor<1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) dimensions = [0]
    %collapsed = tensor.collapse_shape %arg6 [[0], [1, 2]] : tensor<4x1500x1xf32> into tensor<4x1500xf32>
    %broadcasted_3 = linalg.broadcast ins(%collapsed : tensor<4x1500xf32>) outs(%0 : tensor<4x1500x384xf32>) dimensions = [2]
    %collapsed_4 = tensor.collapse_shape %arg7 [[0], [1, 2]] : tensor<4x1500x1xf32> into tensor<4x1500xf32>
    %broadcasted_5 = linalg.broadcast ins(%collapsed_4 : tensor<4x1500xf32>) outs(%0 : tensor<4x1500x384xf32>) dimensions = [2]
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%collapsed_4, %cst_0 : tensor<4x1500xf32>, f32) outs(%1 : tensor<4x1500xf32>) -> tensor<4x1500xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %expanded_1 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%transposed, %broadcasted_2 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    %broadcasted_6 = linalg.broadcast ins(%4 : tensor<4x1500xf32>) outs(%0 : tensor<4x1500x384xf32>) dimensions = [2]
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %broadcasted : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%6, %broadcasted_3 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%7, %cst_0 : tensor<4x1500x384xf32>, f32) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%8, %broadcasted_5 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%7, %10 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %10 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    %reduced:2 = linalg.reduce ins(%7, %11 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%2, %2 : tensor<4x1500xf32>, tensor<4x1500xf32>) dimensions = [2]  {hfusion.reduce_composed = ""}
      (%in: f32, %in_9: f32, %init: f32, %init_10: f32) {
        %18 = arith.addf %in, %init : f32
        %19 = arith.addf %in_9, %init_10 : f32
        linalg.yield %18, %19 : f32, f32
      }
    %broadcasted_7 = linalg.broadcast ins(%reduced#0 : tensor<4x1500xf32>) outs(%0 : tensor<4x1500x384xf32>) dimensions = [2]
    %broadcasted_8 = linalg.broadcast ins(%reduced#1 : tensor<4x1500xf32>) outs(%0 : tensor<4x1500x384xf32>) dimensions = [2]
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%9, %broadcasted_7 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %broadcasted_8 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%13, %14 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_6, %15 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg8, %16 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>) outs(%0 : tensor<4x1500x384xf32>) -> tensor<4x1500x384xf32>
    return %12, %5, %17 : tensor<4x1500x384xf32>, tensor<4x1500x384xf32>, tensor<4x1500x384xf32>
  }
}