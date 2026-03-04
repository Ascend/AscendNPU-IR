// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20" -split-input-file | FileCheck %s

// -----
// CHECK-LABEL: @test_hfusion_load
// CHECK: hfusion.load
// CHECK: hfusion.load
// CHECK: hfusion.load
module {
  func.func @test_hfusion_load(%arg0: tensor<6912xf32>, %arg1: tensor<1xf32>, %arg2: tensor<f32>) -> tensor<6912xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<6912xf32>
    %collapsed = tensor.collapse_shape %arg1 [] : tensor<1xf32> into tensor<f32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<f32>) outs(%0 : tensor<6912xf32>) dimensions = [0] 
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %broadcasted : tensor<6912xf32>, tensor<6912xf32>) outs(%0 : tensor<6912xf32>) -> tensor<6912xf32>
    %extracted = tensor.extract %arg2[] : tensor<f32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %extracted : tensor<6912xf32>, f32) outs(%0 : tensor<6912xf32>) -> tensor<6912xf32>
    return %2 : tensor<6912xf32>
  }
}

// -----
// CHECK-LABEL: @test_recache_unaligned_access
// CHECK: hfusion.load
// CHECK: hfusion.load
module {
  func.func @test_recache_unaligned_access(%arg0: tensor<1x2048xi32>) -> tensor<1x2047x2047xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<2047x2047xi32>
    %1 = tensor.empty() : tensor<2047x2047xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x2048xi32> into tensor<2048xi32>
    %extracted_slice = tensor.extract_slice %collapsed[1] [2047] [1] : tensor<2048xi32> to tensor<2047xi32>
    %broadcasted = linalg.broadcast ins(%extracted_slice : tensor<2047xi32>) outs(%0 : tensor<2047x2047xi32>) dimensions = [1] 
    %extracted_slice_0 = tensor.extract_slice %collapsed[0] [2047] [1] : tensor<2048xi32> to tensor<2047xi32>
    %broadcasted_1 = linalg.broadcast ins(%extracted_slice_0 : tensor<2047xi32>) outs(%0 : tensor<2047x2047xi32>) dimensions = [0] 
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%broadcasted, %broadcasted_1 : tensor<2047x2047xi32>, tensor<2047x2047xi32>) outs(%0 : tensor<2047x2047xi32>) -> tensor<2047x2047xi32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%2 : tensor<2047x2047xi32>) outs(%1 : tensor<2047x2047xf32>) -> tensor<2047x2047xf32>
    %expanded = tensor.expand_shape %3 [[0, 1], [2]] output_shape [1, 2047, 2047] : tensor<2047x2047xf32> into tensor<1x2047x2047xf32>
    return %expanded : tensor<1x2047x2047xf32>
  }
}

// -----
// CHECK-LABEL: @test_aggressive_bubble_up_and_recache
// CHECK: hfusion.load
// CHECK: hfusion.load
// CHECK: hfusion.load
module {
  func.func @test_aggressive_bubble_up_and_recache(%arg0: tensor<1x2047xi64>) -> (tensor<1x2047x2047xf32>, tensor<1x2047x2047xi64>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %cst = arith.constant 3.32225919 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c128_i64 = arith.constant 128 : i64
    %0 = tensor.empty() : tensor<2047xi32>
    %1 = tensor.empty() : tensor<2047x2047xi32>
    %2 = tensor.empty() : tensor<2047x2047xf32>
    %3 = tensor.empty() : tensor<2047x2047xi64>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x2047xi64> into tensor<2047xi64>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<2047xi64>) outs(%0 : tensor<2047xi32>) -> tensor<2047xi32>
    %extracted_slice = tensor.extract_slice %4[2046] [1] [1] : tensor<2047xi32> to tensor<1xi32>
    %extracted_slice_1 = tensor.extract_slice %4[1] [2046] [1] : tensor<2047xi32> to tensor<2046xi32>
    %concat = tensor.concat dim(0) %extracted_slice_1, %extracted_slice : (tensor<2046xi32>, tensor<1xi32>) -> tensor<2047xi32>
    %broadcasted = linalg.broadcast ins(%concat : tensor<2047xi32>) outs(%1 : tensor<2047x2047xi32>) dimensions = [1] 
    %broadcasted_2 = linalg.broadcast ins(%4 : tensor<2047xi32>) outs(%1 : tensor<2047x2047xi32>) dimensions = [0] 
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%broadcasted, %broadcasted_2 : tensor<2047x2047xi32>, tensor<2047x2047xi32>) outs(%1 : tensor<2047x2047xi32>) -> tensor<2047x2047xi32>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%5 : tensor<2047x2047xi32>) outs(%2 : tensor<2047x2047xf32>) -> tensor<2047x2047xf32>
    %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%6 : tensor<2047x2047xf32>) outs(%2 : tensor<2047x2047xf32>) -> tensor<2047x2047xf32>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%7, %cst_0 : tensor<2047x2047xf32>, f32) outs(%2 : tensor<2047x2047xf32>) -> tensor<2047x2047xf32>
    %expanded = tensor.expand_shape %8 [[0, 1], [2]] output_shape [1, 2047, 2047] : tensor<2047x2047xf32> into tensor<1x2047x2047xf32>
    %9 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%8 : tensor<2047x2047xf32>) outs(%2 : tensor<2047x2047xf32>) -> tensor<2047x2047xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%9, %cst : tensor<2047x2047xf32>, f32) outs(%2 : tensor<2047x2047xf32>) -> tensor<2047x2047xf32>
    %11 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%10 : tensor<2047x2047xf32>) outs(%3 : tensor<2047x2047xi64>) -> tensor<2047x2047xi64>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%11, %c0_i64 : tensor<2047x2047xi64>, i64) outs(%3 : tensor<2047x2047xi64>) -> tensor<2047x2047xi64>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%12, %c128_i64 : tensor<2047x2047xi64>, i64) outs(%3 : tensor<2047x2047xi64>) -> tensor<2047x2047xi64>
    %expanded_3 = tensor.expand_shape %13 [[0, 1], [2]] output_shape [1, 2047, 2047] : tensor<2047x2047xi64> into tensor<1x2047x2047xi64>
    return %expanded, %expanded_3 : tensor<1x2047x2047xf32>, tensor<1x2047x2047xi64>
  }
}