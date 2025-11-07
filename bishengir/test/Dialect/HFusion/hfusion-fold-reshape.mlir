// RUN: bishengir-opt %s --canonicalize-ext --cse --split-input-file | FileCheck %s

// CHECK: multi_reshape_args(%[[ARG0:.*]]: tensor<32x1024x1x1xf32>)
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0], [1, 2, 3]] : tensor<32x1024x1x1xf32> into tensor<32x1024xf32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[COLLAPSED]] {{\[\[}}0], [1, 2]] output_shape [32, 2, 512] : tensor<32x1024xf32> into tensor<32x2x512xf32>
// CHECK: %{{.*}} = linalg.reduce ins(%[[EXPANDED]] : tensor<32x2x512xf32>)
// CHECK: return
module {
  func.func @multi_reshape_args(%arg0: tensor<32x1024x1x1xf32>) -> tensor<32x2x1x512xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %cst = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<32x2x512xf32>
    %1 = tensor.empty() : tensor<32x512xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<32x1024x1x1xf32> into tensor<32x1024xf32>
    %expanded = tensor.expand_shape %collapsed [[0], [1, 2, 3]] output_shape [32, 1, 2, 512] : tensor<32x1024xf32> into tensor<32x1x2x512xf32>
    %collapsed_0 = tensor.collapse_shape %expanded [[0, 1], [2], [3]] : tensor<32x1x2x512xf32> into tensor<32x2x512xf32>
    %expanded_1 = tensor.expand_shape %collapsed [[0], [1, 2, 3, 4]] output_shape [32, 1, 2, 1, 512] : tensor<32x1024xf32> into tensor<32x1x2x1x512xf32>
    %collapsed_2 = tensor.collapse_shape %expanded_1 [[0, 1], [2, 3], [4]] : tensor<32x1x2x1x512xf32> into tensor<32x2x512xf32>
    %reduced = linalg.reduce ins(%collapsed_0 : tensor<32x2x512xf32>) outs(%2 : tensor<32x512xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %5 = arith.maximumf %in, %init : f32
        linalg.yield %5 : f32
      }
    %broadcasted = linalg.broadcast ins(%reduced : tensor<32x512xf32>) outs(%0 : tensor<32x2x512xf32>) dimensions = [1]
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%collapsed_2, %broadcasted : tensor<32x2x512xf32>, tensor<32x2x512xf32>) outs(%0 : tensor<32x2x512xf32>) -> tensor<32x2x512xf32>
    %4 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%3 : tensor<32x2x512xf32>) outs(%0 : tensor<32x2x512xf32>) -> tensor<32x2x512xf32>
    %expanded_3 = tensor.expand_shape %4 [[0], [1, 2], [3]] output_shape [32, 2, 1, 512] : tensor<32x2x512xf32> into tensor<32x2x1x512xf32>
    return %expanded_3 : tensor<32x2x1x512xf32>
  }
}