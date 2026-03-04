// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20" -split-input-file | FileCheck %s

// CHECK-DAG: func.func @main_multi_SHALLOW_CV_0_0_{{.*}} {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}, {{.*}} {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>}, {{.*}} {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<2>}
// CHECK-DAG: func.func @main_multi_SHALLOW_CV_0({{.*}} {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}, {{.*}} {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>}, {{.*}} {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<2>}
module {
  func.func @main_multi_SHALLOW_CV_0(%arg0: tensor<2x128x4096xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<12288x4096xf32>) -> (tensor<2x128xf32>, tensor<2x128x1xf32>, tensor<2x128x12288xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_CV>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 4.096000e+03 : f32
    %cst_2 = arith.constant 9.99999974E-6 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<256x4096xf32>
    %1 = tensor.empty() : tensor<256xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<256xf32>) -> tensor<256xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<256xf32>) -> tensor<256xf32>
    %4 = tensor.empty() : tensor<256x12288xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x128x4096xf32> into tensor<256x4096xf32>
    %5 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%collapsed, %cst_0 : tensor<256x4096xf32>, f32) outs(%0 : tensor<256x4096xf32>) -> tensor<256x4096xf32>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<4096xf32>) outs(%0 : tensor<256x4096xf32>) dimensions = [0] 
    %expanded = tensor.expand_shape %2 [[0, 1]] output_shape [2, 128] : tensor<256xf32> into tensor<2x128xf32>
    %expanded_4 = tensor.expand_shape %3 [[0, 1, 2]] output_shape [2, 128, 1] : tensor<256xf32> into tensor<2x128x1xf32>
    %reduced = linalg.reduce { arith.addf } ins(%5 : tensor<256x4096xf32>) outs(%2 : tensor<256xf32>) dimensions = [1] 
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced, %3 : tensor<256xf32>, tensor<256xf32>) outs(%1 : tensor<256xf32>) -> tensor<256xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%6, %cst_2 : tensor<256xf32>, f32) outs(%1 : tensor<256xf32>) -> tensor<256xf32>
    %8 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%7 : tensor<256xf32>) outs(%1 : tensor<256xf32>) -> tensor<256xf32>
    %9 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%8 : tensor<256xf32>) outs(%1 : tensor<256xf32>) -> tensor<256xf32>
    %broadcasted_5 = linalg.broadcast ins(%9 : tensor<256xf32>) outs(%0 : tensor<256x4096xf32>) dimensions = [1] 
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed, %broadcasted_5 : tensor<256x4096xf32>, tensor<256x4096xf32>) outs(%0 : tensor<256x4096xf32>) -> tensor<256x4096xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %broadcasted : tensor<256x4096xf32>, tensor<256x4096xf32>) outs(%0 : tensor<256x4096xf32>) -> tensor<256x4096xf32>
    %12 = linalg.matmul_transpose_b ins(%11, %arg2 : tensor<256x4096xf32>, tensor<12288x4096xf32>) outs(%4 : tensor<256x12288xf32>) -> tensor<256x12288xf32>
    %expanded_6 = tensor.expand_shape %12 [[0, 1], [2]] output_shape [2, 128, 12288] : tensor<256x12288xf32> into tensor<2x128x12288xf32>
    return %expanded, %expanded_4, %expanded_6 : tensor<2x128xf32>, tensor<2x128x1xf32>, tensor<2x128x12288xf32>
  }
}