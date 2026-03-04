// RUN: bishengir-opt --hfusion-infer-func-fusion-kind -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @model_0
// CHECK-SAME: #hfusion.fusion_kind<PURE_ELEMWISE>
func.func @model_0(%arg0: tensor<5x1xf32>) -> tensor<5x1xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%1 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %cst : tensor<5x1xf32>, f32) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %3 : f32, tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %4 : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @model_0
// CHECK: #hfusion.fusion_kind<PURE_ELEMWISE>
func.func @model_0(%arg0: tensor<5x1xf32>) -> tensor<5x1xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%1 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %cst : tensor<5x1xf32>, f32) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %3 : f32, tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %4 : tensor<5x1xf32>
}

// -----
// CHECK-LABEL: func.func @test_any_pb
// CHECK-SAME: #hfusion.fusion_kind<ANY_PB>
func.func @test_any_pb(%arg0: tensor<7x7x7xf32>, %arg1: tensor<7x7x7x7xf32>, %arg2: tensor<7x7x7x7xf32>, %arg3: tensor<7x7x7x7xf32>) -> (tensor<7x7x7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7x7xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %broadcasted = linalg.broadcast ins(%arg0 : tensor<7x7x7xf32>) outs(%arg1 : tensor<7x7x7x7xf32>) dimensions = [3]
  %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%broadcasted : tensor<7x7x7x7xf32>) outs(%arg2 : tensor<7x7x7x7xf32>) -> tensor<7x7x7x7xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%0 : tensor<7x7x7x7xf32>) outs(%arg3 : tensor<7x7x7x7xf32>) -> tensor<7x7x7x7xf32>
  return %broadcasted, %0, %1 : tensor<7x7x7x7xf32>, tensor<7x7x7x7xf32>, tensor<7x7x7x7xf32>
}

// -----
// CHECK-LABEL: func.func @test_unknown
// CHECK-SAME: #hfusion.fusion_kind<UNKNOWN>
func.func @test_unknown(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%arg2 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %5 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg2, %3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %9 = linalg.matmul ins(%arg2, %7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %11 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%7 : tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = tensor.empty(%0, %0, %1) : tensor<?x?x?xf32>
  %13 = linalg.broadcast ins(%arg2 : tensor<?x?xf32>) outs(%12: tensor<?x?x?xf32>) dimensions = [0]
  %14 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %15 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %16 = tensor.empty(%0, %0, %1, %0) : tensor<?x?x?x?xf32>
  %17 = linalg.broadcast ins(%13 : tensor<?x?x?xf32>) outs(%16: tensor<?x?x?x?xf32>) dimensions = [3]
  %18 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %19 = linalg.transpose ins(%15 : tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) permutation = [0, 1]
  return %arg1, %9, %11, %13, %15, %17, %19 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @Fused_With_Reshape
// CHECK-LABEL: #hfusion.fusion_kind<ANY_PB>
func.func @Fused_With_Reshape(%arg0: tensor<2x1x1152xf32>, %arg1: tensor<2x3072x1152xf32>, %arg2: tensor<2x1x1152xf32>, %arg3: tensor<2x3072x1152xf32>) -> tensor<2x3072x1152xf32> attributes {OperatorType = "Broadcast", compute_capability = "", frontend_symbol = {input_0 = ["2", "1", "1152"], input_1 = ["2", "3072", "1152"], input_2 = ["2", "1", "1152"], output_0 = ["2", "3072", "1152"]}, hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %cst = arith.constant dense<1.000000e+00> : tensor<1xf32>
  %collapsed = tensor.collapse_shape %cst [] : tensor<1xf32> into tensor<f32>
  %0 = tensor.empty() : tensor<2x1x1152xf32>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<f32>) outs(%0 : tensor<2x1x1152xf32>) dimensions = [0, 1, 2]
  %1 = tensor.empty() : tensor<2x1x1152xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%broadcasted, %arg0 : tensor<2x1x1152xf32>, tensor<2x1x1152xf32>) outs(%1 : tensor<2x1x1152xf32>) -> tensor<2x1x1152xf32>
  %collapsed_0 = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<2x1x1152xf32> into tensor<2x1152xf32>
  %3 = tensor.empty() : tensor<2x3072x1152xf32>
  %broadcasted_1 = linalg.broadcast ins(%collapsed_0 : tensor<2x1152xf32>) outs(%3 : tensor<2x3072x1152xf32>) dimensions = [1]
  %4 = tensor.empty() : tensor<2x3072x1152xf32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %broadcasted_1 : tensor<2x3072x1152xf32>, tensor<2x3072x1152xf32>) outs(%4 : tensor<2x3072x1152xf32>) -> tensor<2x3072x1152xf32>
  %collapsed_2 = tensor.collapse_shape %arg2 [[0, 1], [2]] : tensor<2x1x1152xf32> into tensor<2x1152xf32>
  %6 = tensor.empty() : tensor<2x3072x1152xf32>
  %broadcasted_3 = linalg.broadcast ins(%collapsed_2 : tensor<2x1152xf32>) outs(%6 : tensor<2x3072x1152xf32>) dimensions = [1]
  %7 = tensor.empty() : tensor<2x3072x1152xf32>
  %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %broadcasted_3 : tensor<2x3072x1152xf32>, tensor<2x3072x1152xf32>) outs(%arg3 : tensor<2x3072x1152xf32>) -> tensor<2x3072x1152xf32>
  return %8 : tensor<2x3072x1152xf32>
}

// -----
// CHECK-LABEL: func.func @horizontal_func
// CHECK-SAME: LAST_AXIS_PBR
// CHECK: return
func.func @horizontal_func(%arg0: tensor<24x128x256x192xbf16>, %arg1: tensor<24x128x256x192xf32>, %arg2: tensor<24x128x256x192xf16>, %arg3: tensor<24x32x1x1xf32>, %arg4: tensor<24x32x1x1xf32>) -> (tensor<24x128x256x192xf32>, tensor<24x128x256x192xf16>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e-05 : f64
  %cst_1 = arith.constant 1.966080e+05 : f32
  %cst_2 = arith.constant 1.000000e+00 : f32
  %cst_3 = arith.constant 2.000000e+00 : f32
  %collapsed = tensor.collapse_shape %arg4 [[0, 1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<768xf32>
  %collapsed_4 = tensor.collapse_shape %arg3 [[0, 1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<768xf32>
  %collapsed_5 = tensor.collapse_shape %arg2 [[0, 1, 2, 3]] : tensor<24x128x256x192xf16> into tensor<150994944xf16>
  %collapsed_6 = tensor.collapse_shape %arg1 [[0, 1, 2, 3]] : tensor<24x128x256x192xf32> into tensor<150994944xf32>
  %collapsed_7 = tensor.collapse_shape %arg0 [[0, 1, 2, 3]] : tensor<24x128x256x192xbf16> into tensor<150994944xbf16>
  %collapsed_8 = tensor.collapse_shape %arg3 [[0, 1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<768xf32>
  %collapsed_9 = tensor.collapse_shape %arg1 [[0, 1, 2, 3]] : tensor<24x128x256x192xf32> into tensor<150994944xf32>
  %expanded = tensor.expand_shape %collapsed_9 [[0, 1]] output_shape [768, 196608] : tensor<150994944xf32> into tensor<768x196608xf32>
  %expanded_10 = tensor.expand_shape %collapsed_9 [[0, 1]] output_shape [768, 196608] : tensor<150994944xf32> into tensor<768x196608xf32>
  %collapsed_11 = tensor.collapse_shape %arg0 [[0, 1, 2, 3]] : tensor<24x128x256x192xbf16> into tensor<150994944xbf16>
  %expanded_12 = tensor.expand_shape %collapsed_11 [[0, 1]] output_shape [768, 196608] : tensor<150994944xbf16> into tensor<768x196608xbf16>
  %expanded_13 = tensor.expand_shape %collapsed_11 [[0, 1]] output_shape [768, 196608] : tensor<150994944xbf16> into tensor<768x196608xbf16>
  %0 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_7 : tensor<150994944xbf16>) outs(%collapsed_6 : tensor<150994944xf32>) -> tensor<150994944xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded_12 : tensor<768x196608xbf16>) outs(%expanded : tensor<768x196608xf32>) -> tensor<768x196608xf32>
  %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded_13 : tensor<768x196608xbf16>) outs(%expanded_10 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
  %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%0 : tensor<150994944xf32>) outs(%collapsed_5 : tensor<150994944xf16>) -> tensor<150994944xf16>
  %4 = tensor.empty() : tensor<24x32x1x1xf32>
  %collapsed_14 = tensor.collapse_shape %4 [[0, 1, 2, 3]] : tensor<24x32x1x1xf32> into tensor<768xf32>
  %5 = tensor.empty() : tensor<768xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<768xf32>) -> tensor<768xf32>
  %reduced = linalg.reduce ins(%1 : tensor<768x196608xf32>) outs(%6 : tensor<768xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %22 = arith.addf %in, %init : f32
      linalg.yield %22 : f32
    }
  %reduced_15 = linalg.reduce ins(%2 : tensor<768x196608xf32>) outs(%collapsed_14 : tensor<768xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %22 = arith.addf %in, %init : f32
      linalg.yield %22 : f32
    }
  %7 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<768xf32>) -> tensor<768xf32>
  %8 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<768xf32>) -> tensor<768xf32>
  %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced, %7 : tensor<768xf32>, tensor<768xf32>) outs(%collapsed_8 : tensor<768xf32>) -> tensor<768xf32>
  %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced_15, %8 : tensor<768xf32>, tensor<768xf32>) outs(%collapsed_4 : tensor<768xf32>) -> tensor<768xf32>
  %11 = tensor.empty() : tensor<768x196608xf32>
  %broadcasted = linalg.broadcast ins(%9 : tensor<768xf32>) outs(%11 : tensor<768x196608xf32>) dimensions = [1]
  %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %cst_2 : tensor<768x196608xf32>, f32) outs(%11 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
  %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%2, %12 : tensor<768x196608xf32>, tensor<768x196608xf32>) outs(%11 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
  %14 = linalg.fill ins(%cst_3 : f32) outs(%11 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
  %15 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%13, %14 : tensor<768x196608xf32>, tensor<768x196608xf32>) outs(%11 : tensor<768x196608xf32>) -> tensor<768x196608xf32>
  %reduced_16 = linalg.reduce ins(%15 : tensor<768x196608xf32>) outs(%collapsed_14 : tensor<768xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %22 = arith.addf %in, %init : f32
      linalg.yield %22 : f32
    }
  %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced_16, %8 : tensor<768xf32>, tensor<768xf32>) outs(%5 : tensor<768xf32>) -> tensor<768xf32>
  %17 = arith.truncf %cst_0 : f64 to f32
  %18 = linalg.fill ins(%17 : f32) outs(%5 : tensor<768xf32>) -> tensor<768xf32>
  %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%18, %cst_2 : tensor<768xf32>, f32) outs(%5 : tensor<768xf32>) -> tensor<768xf32>
  %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%16, %19 : tensor<768xf32>, tensor<768xf32>) outs(%5 : tensor<768xf32>) -> tensor<768xf32>
  %21 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%20 : tensor<768xf32>) outs(%collapsed : tensor<768xf32>) -> tensor<768xf32>
  %expanded_17 = tensor.expand_shape %0 [[0, 1, 2, 3]] output_shape [24, 128, 256, 192] : tensor<150994944xf32> into tensor<24x128x256x192xf32>
  %expanded_18 = tensor.expand_shape %3 [[0, 1, 2, 3]] output_shape [24, 128, 256, 192] : tensor<150994944xf16> into tensor<24x128x256x192xf16>
  %expanded_19 = tensor.expand_shape %reduced_15 [[0, 1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<768xf32> into tensor<24x32x1x1xf32>
  %expanded_20 = tensor.expand_shape %10 [[0, 1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<768xf32> into tensor<24x32x1x1xf32>
  %expanded_21 = tensor.expand_shape %reduced_16 [[0, 1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<768xf32> into tensor<24x32x1x1xf32>
  %expanded_22 = tensor.expand_shape %21 [[0, 1, 2, 3]] output_shape [24, 32, 1, 1] : tensor<768xf32> into tensor<24x32x1x1xf32>
  return %expanded_17, %expanded_18, %expanded_19, %expanded_20, %expanded_21, %expanded_22 : tensor<24x128x256x192xf32>, tensor<24x128x256x192xf16>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>, tensor<24x32x1x1xf32>
}

// -----
// CHECK-LABEL: single_op_last_pbr
// CHECK-SAME: ANY_PB
func.func @single_op_last_pbr(%arg0: tensor<24xf32>) -> tensor<24xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  %broadcasted = linalg.broadcast ins(%cst : tensor<f32>) outs(%arg0 : tensor<24xf32>) dimensions = [0]
  return %broadcasted : tensor<24xf32>
}

// -----
// CHECK-LABEL: func.func @test_last_axis_pbr
// CHECK-NOT: LAST_AXIS_PBR
func.func @test_last_axis_pbr(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> tensor<1x2x3xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = tensor.empty() : tensor<2x3xf32>
  %1 = linalg.reduce ins(%arg0 : tensor<2x3x4xf32>) outs(%0 : tensor<2x3xf32>) dimensions = [2]
    (%in: f32, %init: f32) {
      %2 = arith.addf %in, %init : f32
      linalg.yield %2 : f32
    }
  %2 = tensor.empty() : tensor<1x2x3xf32>
  %3 = linalg.broadcast ins(%1 : tensor<2x3xf32>) outs(%2 : tensor<1x2x3xf32>) dimensions = [0]
  %4 = linalg.reduce ins(%arg1 : tensor<1x2x3x4xf32>) outs(%2 : tensor<1x2x3xf32>) dimensions = [3]
    (%in: f32, %init: f32) {
      %5 = arith.addf %in, %init : f32
      linalg.yield %5 : f32
    }
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %4: tensor<1x2x3xf32>, tensor<1x2x3xf32>) outs(%2 : tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  return %5 : tensor<1x2x3xf32>
}

// -----
module {
// CHECK-LABEL: func.func @mlir_fused_convert_element_type_0
// CHECK-SAME: ANY_PB
  func.func @mlir_fused_convert_element_type_0(%arg0: tensor<1024xf32>) -> tensor<1024xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    return %arg0 : tensor<1024xf32>
  }
}

// -----
module {
// CHECK-LABEL: func.func @mlir_fused_convert_element_type_0
// CHECK-SAME: ANY_PB
  func.func @mlir_fused_convert_element_type_0(%arg0: tensor<1x1024xf32>) -> tensor<1024xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x1024xf32> into tensor<1024xf32>
    return %collapsed : tensor<1024xf32>
  }
}

// -----
// CHECK-LABEL: func.func @forward
// CHECK-SAME: fusion_kind<SHALLOW_VV>
module {
  func.func @forward(%arg0: tensor<1024xf16>) -> (tensor<1024x1500xf16>, tensor<1024x3000xf16>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %0 = tensor.empty() : tensor<1024x1500xf16>
    %1 = tensor.empty() : tensor<1024x3000xf16>
    %broadcasted = linalg.broadcast ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024x1500xf16>) dimensions = [1]
    %broadcasted_0 = linalg.broadcast ins(%arg0 : tensor<1024xf16>) outs(%1 : tensor<1024x3000xf16>) dimensions = [1]
    return %broadcasted, %broadcasted_0 : tensor<1024x1500xf16>, tensor<1024x3000xf16>
  }
}

// -----
// CHECK-LABEL: @mlir_fused_mul_npu_dtype_cast_2
// CHECK-SAME: PURE_ELEMWISE
func.func @mlir_fused_mul_npu_dtype_cast_2(%arg0: tensor<1xf32>, %arg1: tensor<128xi64>, %arg2: tensor<1xf32>) -> (tensor<1x128xf32>, tensor<1x128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant -9.2103403719761836 : f64
  %c1000_i64 = arith.constant 1000 : i64
  %c128_i64 = arith.constant 128 : i64
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c1000_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%1 : tensor<1xi64>) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %3 : tensor<1xf32>, tensor<1xf32>) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %5 = tensor.empty() : tensor<128xf32>
  %extracted = tensor.extract %4[%c0] : tensor<1xf32>
  %6 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<128xi64>) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
  %7 = tensor.empty() : tensor<128xi64>
  %8 = arith.truncf %cst : f64 to f32
  %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %8 : tensor<128xf32>, f32) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
  %10 = linalg.fill ins(%c128_i64 : i64) outs(%7 : tensor<128xi64>) -> tensor<128xi64>
  %11 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%10 : tensor<128xi64>) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
  %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%9, %11 : tensor<128xf32>, tensor<128xf32>) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
  %13 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%12 : tensor<128xf32>) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
  %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %extracted : tensor<128xf32>, f32) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
  %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg2, %3 : tensor<1xf32>, tensor<1xf32>) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %extracted_0 = tensor.extract %15[%c0] : tensor<1xf32>
  %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %extracted_0 : tensor<128xf32>, f32) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
  %expanded = tensor.expand_shape %14 [[0, 1]] output_shape [1, 128] : tensor<128xf32> into tensor<1x128xf32>
  %expanded_1 = tensor.expand_shape %16 [[0, 1]] output_shape [1, 128] : tensor<128xf32> into tensor<1x128xf32>
  return %expanded, %expanded_1 : tensor<1x128xf32>, tensor<1x128xf32>
}

// -----

// CHECK-LABEL: @main_mix_cv
// CHECK-SAME: fusion_kind<UNKNOWN>
module{
  func.func @main_mix_cv(%arg0: tensor<?x4096xf16>, %arg1: tensor<?x32x128xf16>, %arg2: tensor<?x4096xf16>, %arg3: tensor<4096x4096xf16>) -> (tensor<?x4096xf16>, tensor<?x4096xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 2.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
    %0 = tensor.empty(%dim) : tensor<?x4096xf16>
    %dim_0 = tensor.dim %arg1, %c0 : tensor<?x32x128xf16>
    %1 = tensor.empty(%dim_0) : tensor<?x4096xf16>
    %2 = linalg.matmul_transpose_b ins(%arg2, %arg3 : tensor<?x4096xf16>, tensor<4096x4096xf16>) outs(%1 : tensor<?x4096xf16>) -> tensor<?x4096xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %2 : tensor<?x4096xf16>, tensor<?x4096xf16>) outs(%0 : tensor<?x4096xf16>) -> tensor<?x4096xf16>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%3, %cst : tensor<?x4096xf16>, f16) outs(%0 : tensor<?x4096xf16>) -> tensor<?x4096xf16>
    %5 = tensor.empty(%dim) : tensor<?x4096xf32>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<?x4096xf16>) outs(%5 : tensor<?x4096xf32>) -> tensor<?x4096xf32>
    return %3, %6 : tensor<?x4096xf16>, tensor<?x4096xf32>
  }
}

// -----
// CHECK-LABEL: @single_cube
// CHECK-SAME: fusion_kind<SINGLE_CUBE>
func.func @single_cube(%arg0: tensor<?x4096xf16>, %arg1: tensor<6144x4096xf16>, %arg2: tensor<?x6144xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> tensor<?x6144xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<?x4096xf16>, tensor<6144x4096xf16>) outs(%arg2 : tensor<?x6144xf16>) -> tensor<?x6144xf16>
  return %0 : tensor<?x6144xf16>
}
// -----

// CHECK-LABEL: @shallow_connected_with_scalar_hfusion
// CHECK-SAME: fusion_kind<SHALLOW_VV>
module {
  func.func @shallow_connected_with_scalar_hfusion(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>, %arg5: tensor<16xf32>, %arg6: tensor<16xf32>) -> (tensor<16x16xf32>, tensor<16xf32>) attributes {debug_instruction_number = 190 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %cst = arith.constant {debug_instruction_number = 6 : i32} 0.099999999999999978 : f64
    %cst_0 = arith.constant {debug_instruction_number = 8 : i32} 1.000000e+02 : f32
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = tensor.empty() : tensor<f32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<f32>) -> tensor<f32>
    %3 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%2 : tensor<f32>) outs(%1 : tensor<f32>) -> tensor<f32>
    %4 = linalg.elemwise_binary {debug_instruction_number = 14 : i32, fun = #linalg.binary_fn<mul>} ins(%arg0, %0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = linalg.elemwise_binary {debug_instruction_number = 17 : i32, fun = #linalg.binary_fn<sub>} ins(%arg1, %4 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %6 = arith.truncf %cst {debug_instruction_number = 18 : i32} : f64 to f32
    %7 = linalg.fill {debug_instruction_number = 20 : i32} ins(%6 : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %8 = linalg.elemwise_binary {debug_instruction_number = 23 : i32, fun = #linalg.binary_fn<mul>} ins(%5, %7 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %9 = tensor.empty() {debug_instruction_number = 107 : i32} : tensor<16xf32>
    %10 = tensor.empty() : tensor<16xf32>
    %broadcasted = linalg.broadcast ins(%3 : tensor<f32>) outs(%10 : tensor<16xf32>) dimensions = [0]
    %11 = linalg.elemwise_binary {debug_instruction_number = 112 : i32, fun = #linalg.binary_fn<mul>} ins(%arg5, %broadcasted : tensor<16xf32>, tensor<16xf32>) outs(%9 : tensor<16xf32>) -> tensor<16xf32>
    %12 = linalg.elemwise_binary {debug_instruction_number = 115 : i32, fun = #linalg.binary_fn<sub>} ins(%arg6, %11 : tensor<16xf32>, tensor<16xf32>) outs(%9 : tensor<16xf32>) -> tensor<16xf32>
    return {debug_instruction_number = 189 : i32} %8, %12 : tensor<16x16xf32>, tensor<16xf32>
  }
}

// -----
// CHECK: func.func @LAST_PBR(
// CHECK-SAME: LAST_AXIS_PBR
// CHECK: return
module {
  func.func @LAST_PBR(%arg0: tensor<384x3072xbf16>) -> tensor<1xbf16> attributes {OperatorType = "Reduce", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
    %cst = arith.constant 0.000000e+00 : f32
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<384x3072xbf16> into tensor<1179648xbf16>
    %0 = tensor.empty() : tensor<1179648xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<1179648xbf16>) outs(%0 : tensor<1179648xf32>) -> tensor<1179648xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %1 : tensor<1179648xf32>, tensor<1179648xf32>) outs(%0 : tensor<1179648xf32>) -> tensor<1179648xf32>
    %3 = tensor.empty() : tensor<f32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
    %reduced = linalg.reduce ins(%2 : tensor<1179648xf32>) outs(%4 : tensor<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %7 = arith.addf %in, %init : f32
        linalg.yield %7 : f32
      }
    %5 = tensor.empty() : tensor<bf16>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%reduced : tensor<f32>) outs(%5 : tensor<bf16>) -> tensor<bf16>
    %expanded = tensor.expand_shape %6 [] output_shape [1] : tensor<bf16> into tensor<1xbf16>
    return %expanded : tensor<1xbf16>
  }
}
