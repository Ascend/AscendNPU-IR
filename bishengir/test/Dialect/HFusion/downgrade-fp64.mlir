// RUN: bishengir-opt -hfusion-downgrade-fp64 %s -verify-diagnostics -split-input-file | FileCheck %s
module {
// CHECK-LABEL: func.func @mlir_fused_add_convert_element_type_4
// CHECK: %cst = arith.constant 1.62760422E-4 : f32
// CHECK: %cst_0 = arith.constant 1.000000e+00 : f32
// CHECK: %cst_1 = arith.constant 1.000000e+00 : bf16
// CHECK-NOT: arith.truncf
  func.func @mlir_fused_add_convert_element_type_4(%arg0: tensor<24x256x32x24xbf16>, %arg1: tensor<24x256x768xbf16>, %arg2: tensor<24x32xf32>, %arg3: tensor<256xf32>, %arg4: tensor<24x256x768xf32>, %arg5: tensor<24x32xf32>, %arg6: tensor<24x32xf32>, %arg7: tensor<24x32xf32>) -> tensor<24x256x768xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 1.6276041666666666E-4 : f64
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : bf16
    %collapsed = tensor.collapse_shape %arg0 [[0], [1], [2, 3]] : tensor<24x256x32x24xbf16> into tensor<24x256x768xbf16>
    %0 = tensor.empty() : tensor<24x256x768xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<24x256x768xbf16>) outs(%0 : tensor<24x256x768xf32>) -> tensor<24x256x768xf32>
    %expanded = tensor.expand_shape %1 [[0], [1, 2], [3]] output_shape [24, 32, 8, 768] : tensor<24x256x768xf32> into tensor<24x32x8x768xf32>
    %expanded_2 = tensor.expand_shape %arg2 [[0], [1, 2]] output_shape [24, 32, 1] : tensor<24x32xf32> into tensor<24x32x1xf32>
    %2 = tensor.empty() : tensor<24x32x8xf32>
    %collapsed_3 = tensor.collapse_shape %expanded_2 [[0], [1, 2]] : tensor<24x32x1xf32> into tensor<24x32xf32>
    %broadcasted = linalg.broadcast ins(%collapsed_3 : tensor<24x32xf32>) outs(%2 : tensor<24x32x8xf32>) dimensions = [2] 
    %expanded_4 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [1, 32, 8] : tensor<256xf32> into tensor<1x32x8xf32>
    %collapsed_5 = tensor.collapse_shape %expanded_4 [[0, 1], [2]] : tensor<1x32x8xf32> into tensor<32x8xf32>
    %broadcasted_6 = linalg.broadcast ins(%collapsed_5 : tensor<32x8xf32>) outs(%2 : tensor<24x32x8xf32>) dimensions = [0] 
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %broadcasted_6 : tensor<24x32x8xf32>, tensor<24x32x8xf32>) outs(%2 : tensor<24x32x8xf32>) -> tensor<24x32x8xf32>
    %expanded_7 = tensor.expand_shape %3 [[0], [1], [2, 3]] output_shape [24, 32, 8, 1] : tensor<24x32x8xf32> into tensor<24x32x8x1xf32>
    %4 = tensor.empty() : tensor<24x32x8x768xf32>
    %collapsed_8 = tensor.collapse_shape %expanded_7 [[0], [1], [2, 3]] : tensor<24x32x8x1xf32> into tensor<24x32x8xf32>
    %broadcasted_9 = linalg.broadcast ins(%collapsed_8 : tensor<24x32x8xf32>) outs(%4 : tensor<24x32x8x768xf32>) dimensions = [3] 
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded, %broadcasted_9 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%4 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %expanded_10 = tensor.expand_shape %arg4 [[0], [1, 2], [3]] output_shape [24, 32, 8, 768] : tensor<24x256x768xf32> into tensor<24x32x8x768xf32>
    %6 = tensor.empty() : tensor<24x32xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg5, %arg6 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %8 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg7, %8 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%7, %9 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %arg2 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%11, %arg2 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %arg2 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %14 = arith.truncf %cst : f64 to f32
    %15 = linalg.fill ins(%14 : f32) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %15 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %expanded_11 = tensor.expand_shape %16 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %collapsed_12 = tensor.collapse_shape %expanded_11 [[0], [1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<24x32xf32>
    %broadcasted_13 = linalg.broadcast ins(%collapsed_12 : tensor<24x32xf32>) outs(%4 : tensor<24x32x8x768xf32>) dimensions = [2, 3] 
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_10, %broadcasted_13 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%4 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %18 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%17, %18 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%4 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %19 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%4 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %21 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%16 : tensor<24x32xf32>) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%21, %arg6 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg5, %arg2 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%23, %15 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%24, %8 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%22, %25 : tensor<24x32xf32>, tensor<24x32xf32>) outs(%6 : tensor<24x32xf32>) -> tensor<24x32xf32>
    %expanded_14 = tensor.expand_shape %26 [[0], [1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<24x32xf32> into tensor<24x32x1x1xf32>
    %collapsed_15 = tensor.collapse_shape %expanded_14 [[0], [1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<24x32xf32>
    %broadcasted_16 = linalg.broadcast ins(%collapsed_15 : tensor<24x32xf32>) outs(%4 : tensor<24x32x8x768xf32>) dimensions = [2, 3] 
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_16, %18 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%4 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%20, %27 : tensor<24x32x8x768xf32>, tensor<24x32x8x768xf32>) outs(%4 : tensor<24x32x8x768xf32>) -> tensor<24x32x8x768xf32>
    %collapsed_17 = tensor.collapse_shape %28 [[0], [1, 2], [3]] : tensor<24x32x8x768xf32> into tensor<24x256x768xf32>
    %29 = tensor.empty() : tensor<24x256x768xbf16>
    %30 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_17 : tensor<24x256x768xf32>) outs(%29 : tensor<24x256x768xbf16>) -> tensor<24x256x768xbf16>
    %31 = linalg.fill ins(%cst_1 : bf16) outs(%29 : tensor<24x256x768xbf16>) -> tensor<24x256x768xbf16>
    %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%30, %31 : tensor<24x256x768xbf16>, tensor<24x256x768xbf16>) outs(%29 : tensor<24x256x768xbf16>) -> tensor<24x256x768xbf16>
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%collapsed, %32 : tensor<24x256x768xbf16>, tensor<24x256x768xbf16>) outs(%29 : tensor<24x256x768xbf16>) -> tensor<24x256x768xbf16>
    return %33 : tensor<24x256x768xbf16>
  }
}

// -----

// CHECK-LABEL: func.func @mlir_fused_add_sum_108
// CHECK: %cst = arith.constant 9.99999968E-21 : f32
func.func @mlir_fused_add_sum_108(%arg0: tensor<4096x8xbf16>) -> (tensor<4096xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 9.9999999999999995E-21 : f64
  %0 = tensor.empty() : tensor<4096x8xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4096x8xbf16>) outs(%0 : tensor<4096x8xf32>) -> tensor<4096x8xf32>
  %2 = tensor.empty() : tensor<4096xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4096xf32>) -> tensor<4096xf32>
  %reduced = linalg.reduce ins(%1 : tensor<4096x8xf32>) outs(%3 : tensor<4096xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %10 = arith.addf %in, %init : f32
      linalg.yield %10 : f32
    }
  %4 = tensor.empty() : tensor<4096xbf16>
  %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%reduced : tensor<4096xf32>) outs(%4 : tensor<4096xbf16>) -> tensor<4096xbf16>
  %6 = arith.truncf %cst_0 : f64 to bf16
  %7 = arith.extf %6 : bf16 to f32
  %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%reduced, %7 : tensor<4096xf32>, f32) outs(%2 : tensor<4096xf32>) -> tensor<4096xf32>
  return %8 : tensor<4096xf32>
}
