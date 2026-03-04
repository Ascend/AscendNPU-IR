// RUN: bishengir-compile -enable-hfusion-compile=true -enable-hivm-compile=true -enable-lir-compile=false %s | FileCheck %s

// -----
// CHECK: define dso_local void @return_arg_directly_after_optimize
module {
  func.func @return_arg_directly_after_optimize(%arg0: tensor<64x70x768xf16>, %arg1: tensor<64x70x768xf16>, %arg2: tensor<768xf16>, %arg3: tensor<768xf16>) -> (tensor<64x70x768xf16>, tensor<64x70x768xf16>, tensor<64x70x1xf32>, tensor<64x70x1xf32>, tensor<64x70x768xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 7.680000e+02 : f32
    %0 = tensor.empty() : tensor<64x70x768xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<64x70x768xf32>) -> tensor<64x70x768xf32>
    %2 = tensor.empty() : tensor<64x70x768xf16>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<64x70x768xf32>) outs(%2 : tensor<64x70x768xf16>) -> tensor<64x70x768xf16>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %3 : tensor<64x70x768xf16>, tensor<64x70x768xf16>) outs(%2 : tensor<64x70x768xf16>) -> tensor<64x70x768xf16>
    %5 = tensor.empty() : tensor<64x70x768xi64>
    %6 = linalg.fill ins(%c1_i64 : i64) outs(%5 : tensor<64x70x768xi64>) -> tensor<64x70x768xi64>
    %7 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%6 : tensor<64x70x768xi64>) outs(%2 : tensor<64x70x768xf16>) -> tensor<64x70x768xf16>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %7 : tensor<64x70x768xf16>, tensor<64x70x768xf16>) outs(%2 : tensor<64x70x768xf16>) -> tensor<64x70x768xf16>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%4, %8 : tensor<64x70x768xf16>, tensor<64x70x768xf16>) outs(%2 : tensor<64x70x768xf16>) -> tensor<64x70x768xf16>
    %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%9 : tensor<64x70x768xf16>) outs(%0 : tensor<64x70x768xf32>) -> tensor<64x70x768xf32>
    %11 = tensor.empty() : tensor<64x70xf32>
    %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<64x70xf32>) -> tensor<64x70xf32>
    %reduced = linalg.reduce ins(%10 : tensor<64x70x768xf32>) outs(%12 : tensor<64x70xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %39 = arith.addf %in, %init : f32
        linalg.yield %39 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0], [1, 2]] output_shape [64, 70, 1] : tensor<64x70xf32> into tensor<64x70x1xf32>
    %13 = tensor.empty() : tensor<64x70x1xf32>
    %14 = linalg.fill ins(%cst_2 : f32) outs(%13 : tensor<64x70x1xf32>) -> tensor<64x70x1xf32>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %14 : tensor<64x70x1xf32>, tensor<64x70x1xf32>) outs(%13 : tensor<64x70x1xf32>) -> tensor<64x70x1xf32>
    %collapsed = tensor.collapse_shape %15 [[0], [1, 2]] : tensor<64x70x1xf32> into tensor<64x70xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<64x70xf32>) outs(%0 : tensor<64x70x768xf32>) dimensions = [2]
    %16 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%6 : tensor<64x70x768xi64>) outs(%0 : tensor<64x70x768xf32>) -> tensor<64x70x768xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %16 : tensor<64x70x768xf32>, tensor<64x70x768xf32>) outs(%0 : tensor<64x70x768xf32>) -> tensor<64x70x768xf32>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%10, %17 : tensor<64x70x768xf32>, tensor<64x70x768xf32>) outs(%0 : tensor<64x70x768xf32>) -> tensor<64x70x768xf32>
    %19 = linalg.fill ins(%c2_i64 : i64) outs(%5 : tensor<64x70x768xi64>) -> tensor<64x70x768xi64>
    %20 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%19 : tensor<64x70x768xi64>) outs(%0 : tensor<64x70x768xf32>) -> tensor<64x70x768xf32>
    %21 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%18, %20 : tensor<64x70x768xf32>, tensor<64x70x768xf32>) outs(%0 : tensor<64x70x768xf32>) -> tensor<64x70x768xf32>
    %reduced_3 = linalg.reduce ins(%21 : tensor<64x70x768xf32>) outs(%12 : tensor<64x70xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %39 = arith.addf %in, %init : f32
        linalg.yield %39 : f32
      }
    %expanded_4 = tensor.expand_shape %reduced_3 [[0], [1, 2]] output_shape [64, 70, 1] : tensor<64x70xf32> into tensor<64x70x1xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded_4, %14 : tensor<64x70x1xf32>, tensor<64x70x1xf32>) outs(%13 : tensor<64x70x1xf32>) -> tensor<64x70x1xf32>
    %23 = arith.truncf %cst_0 : f64 to f32
    %24 = linalg.fill ins(%23 : f32) outs(%13 : tensor<64x70x1xf32>) -> tensor<64x70x1xf32>
    %25 = tensor.empty() : tensor<64x70x1xi64>
    %26 = linalg.fill ins(%c1_i64 : i64) outs(%25 : tensor<64x70x1xi64>) -> tensor<64x70x1xi64>
    %27 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%26 : tensor<64x70x1xi64>) outs(%13 : tensor<64x70x1xf32>) -> tensor<64x70x1xf32>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%24, %27 : tensor<64x70x1xf32>, tensor<64x70x1xf32>) outs(%13 : tensor<64x70x1xf32>) -> tensor<64x70x1xf32>
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%22, %28 : tensor<64x70x1xf32>, tensor<64x70x1xf32>) outs(%13 : tensor<64x70x1xf32>) -> tensor<64x70x1xf32>
    %30 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%29 : tensor<64x70x1xf32>) outs(%13 : tensor<64x70x1xf32>) -> tensor<64x70x1xf32>
    %collapsed_5 = tensor.collapse_shape %30 [[0], [1, 2]] : tensor<64x70x1xf32> into tensor<64x70xf32>
    %broadcasted_6 = linalg.broadcast ins(%collapsed_5 : tensor<64x70xf32>) outs(%0 : tensor<64x70x768xf32>) dimensions = [2]
    %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%18, %broadcasted_6 : tensor<64x70x768xf32>, tensor<64x70x768xf32>) outs(%0 : tensor<64x70x768xf32>) -> tensor<64x70x768xf32>
    %32 = tensor.empty() : tensor<768xf32>
    %33 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg2 : tensor<768xf16>) outs(%32 : tensor<768xf32>) -> tensor<768xf32>
    %broadcasted_7 = linalg.broadcast ins(%33 : tensor<768xf32>) outs(%0 : tensor<64x70x768xf32>) dimensions = [0, 1]
    %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%31, %broadcasted_7 : tensor<64x70x768xf32>, tensor<64x70x768xf32>) outs(%0 : tensor<64x70x768xf32>) -> tensor<64x70x768xf32>
    %35 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg3 : tensor<768xf16>) outs(%32 : tensor<768xf32>) -> tensor<768xf32>
    %broadcasted_8 = linalg.broadcast ins(%35 : tensor<768xf32>) outs(%0 : tensor<64x70x768xf32>) dimensions = [0, 1]
    %36 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_8, %16 : tensor<64x70x768xf32>, tensor<64x70x768xf32>) outs(%0 : tensor<64x70x768xf32>) -> tensor<64x70x768xf32>
    %37 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%34, %36 : tensor<64x70x768xf32>, tensor<64x70x768xf32>) outs(%0 : tensor<64x70x768xf32>) -> tensor<64x70x768xf32>
    %38 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%37 : tensor<64x70x768xf32>) outs(%2 : tensor<64x70x768xf16>) -> tensor<64x70x768xf16>
    return %4, %9, %15, %30, %38 : tensor<64x70x768xf16>, tensor<64x70x768xf16>, tensor<64x70x1xf32>, tensor<64x70x1xf32>, tensor<64x70x768xf16>
  }
}