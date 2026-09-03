// REQUIRES: asserts
// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=40" -split-input-file | FileCheck %s

// CHECK: @forward
func.func @forward(%arg0: tensor<2x10xf32>, %cst : tensor<2x20xf32> ,%cst_0 : tensor<20x10xf32>, %cst_1 : tensor<20xf32>, %cst_2: tensor<20x20xf32>, %cst_3: tensor<20xf32>, %cst_4: tensor<10x20xf32>, %cst_5: tensor<10xf32>) -> tensor<2x10xf32>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_CV>} {
  %0 = tensor.empty() : tensor<2x20xf32>
  %1 = linalg.matmul_transpose_b ins(%arg0, %cst_0 : tensor<2x10xf32>, tensor<20x10xf32>) outs(%0 : tensor<2x20xf32>) -> tensor<2x20xf32>
  %2 = tensor.empty() : tensor<2x20xf32>
  %broadcasted = linalg.broadcast ins(%cst_1 : tensor<20xf32>) outs(%2 : tensor<2x20xf32>) dimensions = [0]
  %3 = tensor.empty() : tensor<2x20xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %broadcasted : tensor<2x20xf32>, tensor<2x20xf32>) outs(%3 : tensor<2x20xf32>) -> tensor<2x20xf32>
  %5 = tensor.empty() : tensor<2x20xf32>
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%4, %cst : tensor<2x20xf32>, tensor<2x20xf32>) outs(%5 : tensor<2x20xf32>) -> tensor<2x20xf32>
  %7 = tensor.empty() : tensor<2x20xf32>
  %8 = linalg.matmul_transpose_b ins(%6, %cst_2 : tensor<2x20xf32>, tensor<20x20xf32>) outs(%7 : tensor<2x20xf32>) -> tensor<2x20xf32>
  %9 = tensor.empty() : tensor<2x20xf32>
  %broadcasted_6 = linalg.broadcast ins(%cst_3 : tensor<20xf32>) outs(%9 : tensor<2x20xf32>) dimensions = [0]
  %10 = tensor.empty() : tensor<2x20xf32>
  %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%8, %broadcasted_6 : tensor<2x20xf32>, tensor<2x20xf32>) outs(%10 : tensor<2x20xf32>) -> tensor<2x20xf32>
  %12 = tensor.empty() : tensor<2x20xf32>
  %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%11, %cst : tensor<2x20xf32>, tensor<2x20xf32>) outs(%12 : tensor<2x20xf32>) -> tensor<2x20xf32>
  %14 = tensor.empty() : tensor<2x10xf32>
  %15 = linalg.matmul_transpose_b ins(%13, %cst_4 : tensor<2x20xf32>, tensor<10x20xf32>) outs(%14 : tensor<2x10xf32>) -> tensor<2x10xf32>
  %16 = tensor.empty() : tensor<2x10xf32>
  %broadcasted_7 = linalg.broadcast ins(%cst_5 : tensor<10xf32>) outs(%16 : tensor<2x10xf32>) dimensions = [0]
  %17 = tensor.empty() : tensor<2x10xf32>
  %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%15, %broadcasted_7 : tensor<2x10xf32>, tensor<2x10xf32>) outs(%17 : tensor<2x10xf32>) -> tensor<2x10xf32>
  return %18 : tensor<2x10xf32>
}

// -----

module {
// CHECK: shallow_cv_multiple_vv
// CHECK: {{.*}}: tensor<1x256x1152xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>},
// CHECK: {{.*}}: tensor<8x1152xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>},
// CHECK: {{.*}}: tensor<1152x1152xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<2>},
// CHECK: {{.*}}: tensor<1152xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<3>},
// CHECK: {{.*}}: tensor<8x1152xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<4>},
// CHECK: {{.*}}: tensor<8x256x1152xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<5>},
// CHECK: {{.*}}: tensor<8x256x1152xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>},
// CHECK: {{.*}}: tensor<8x1152xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>},
  func.func @shallow_cv_multiple_vv(%arg0: tensor<1x256x1152xf32>, %arg1: tensor<8x1152xf32>, %arg2: tensor<1152x1152xf32>, %arg3: tensor<1152xf32>, %arg4: tensor<8x1152xf32>, %arg5: tensor<8x256x1152xf32>) -> (tensor<8x256x1152xf32>, tensor<8x1152xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_CV>} {
    %0 = tensor.empty() : tensor<8x294912xf32>
    %1 = tensor.empty() : tensor<8x1152xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x256x1152xf32> into tensor<294912xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<294912xf32>) outs(%0 : tensor<8x294912xf32>) dimensions = [0]
    %2 = linalg.matmul_transpose_b ins(%arg1, %arg2 : tensor<8x1152xf32>, tensor<1152x1152xf32>) outs(%1 : tensor<8x1152xf32>) -> tensor<8x1152xf32>
    %broadcasted_0 = linalg.broadcast ins(%arg3 : tensor<1152xf32>) outs(%1 : tensor<8x1152xf32>) dimensions = [0]
    %collapsed_1 = tensor.collapse_shape %arg5 [[0], [1, 2]] : tensor<8x256x1152xf32> into tensor<8x294912xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%collapsed_1, %broadcasted : tensor<8x294912xf32>, tensor<8x294912xf32>) outs(%0 : tensor<8x294912xf32>) -> tensor<8x294912xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%broadcasted_0, %2 : tensor<8x1152xf32>, tensor<8x1152xf32>) outs(%1 : tensor<8x1152xf32>) -> tensor<8x1152xf32>
    %expanded = tensor.expand_shape %3 [[0], [1, 2]] output_shape [8, 256, 1152] : tensor<8x294912xf32> into tensor<8x256x1152xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%4, %arg4 : tensor<8x1152xf32>, tensor<8x1152xf32>) outs(%1 : tensor<8x1152xf32>) -> tensor<8x1152xf32>
    return %expanded, %5 : tensor<8x256x1152xf32>, tensor<8x1152xf32>
  }
}

// -----

module {
// CHECK: shallow_cv_with_reshape_output
// CHECK: [[ARG5:.*]]: tensor<2x256x4096xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>},
  func.func @shallow_cv_with_reshape_output(%arg0: tensor<2x256x11008xf16>, %arg1: tensor<2x256x11008xf16>, %arg2: tensor<2x256x11008xf16>, %arg3: tensor<4096x11008xf16>, %arg4: tensor<2x256x4096xf16>) -> tensor<2x256x4096xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_CV>} {
    %cst = arith.constant -1.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<512x4096xf16>
    %1 = tensor.empty() : tensor<512x11008xf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x256x11008xf16> into tensor<512x11008xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed, %cst : tensor<512x11008xf16>, f16) outs(%1 : tensor<512x11008xf16>) -> tensor<512x11008xf16>
    %collapsed_1 = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<2x256x11008xf16> into tensor<512x11008xf16>
    %collapsed_2 = tensor.collapse_shape %arg2 [[0, 1], [2]] : tensor<2x256x11008xf16> into tensor<512x11008xf16>
    %collapsed_3 = tensor.collapse_shape %arg4 [[0, 1], [2]] : tensor<2x256x4096xf16> into tensor<512x4096xf16>
    %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%2 : tensor<512x11008xf16>) outs(%1 : tensor<512x11008xf16>) -> tensor<512x11008xf16>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %cst_0 : tensor<512x11008xf16>, f16) outs(%1 : tensor<512x11008xf16>) -> tensor<512x11008xf16>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%collapsed_1, %4 : tensor<512x11008xf16>, tensor<512x11008xf16>) outs(%1 : tensor<512x11008xf16>) -> tensor<512x11008xf16>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %collapsed : tensor<512x11008xf16>, tensor<512x11008xf16>) outs(%1 : tensor<512x11008xf16>) -> tensor<512x11008xf16>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %collapsed_2 : tensor<512x11008xf16>, tensor<512x11008xf16>) outs(%1 : tensor<512x11008xf16>) -> tensor<512x11008xf16>
    %8 = linalg.matmul_transpose_b ins(%7, %arg3 : tensor<512x11008xf16>, tensor<4096x11008xf16>) outs(%0 : tensor<512x4096xf16>) -> tensor<512x4096xf16>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%collapsed_3, %8 : tensor<512x4096xf16>, tensor<512x4096xf16>) outs(%0 : tensor<512x4096xf16>) -> tensor<512x4096xf16>
    %expanded = tensor.expand_shape %9 [[0, 1], [2]] output_shape [2, 256, 4096] : tensor<512x4096xf16> into tensor<2x256x4096xf16>
    return %expanded : tensor<2x256x4096xf16>
  }
}
