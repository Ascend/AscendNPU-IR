// RUN: bishengir-opt %s --hfusion-compose-multi-reduce --split-input-file | FileCheck %s
// RUN: bishengir-opt %s --hfusion-compose-multi-reduce="aggressive=true" --split-input-file | FileCheck %s --check-prefix=AGGR

// CHECK-LABEL: @reduction_tile(
// CHECK: linalg.reduce
// CHECK: linalg.reduce
// CHECK: return
func.func @reduction_tile(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %reduced = linalg.reduce ins(%arg0 : tensor<?x?xf32>) outs(%arg2 : tensor<?xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %0 = arith.addf %in, %init : f32
      linalg.yield %0 : f32
    }
  %reduced_0 = linalg.reduce ins(%arg1 : tensor<?x?xf32>) outs(%arg3 : tensor<?xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %0 = arith.addf %in, %init : f32
      linalg.yield %0 : f32
    }
  return %reduced, %reduced_0 : tensor<?xf32>, tensor<?xf32>
}

// -----
// CHECK-LABEL: @reduction_tile_2
// CHECK: linalg.reduce
// CHECK-SAME: 4x5
// CHECK: linalg.reduce
// CHECK-SAME: 4x6
// CHECK-NOT: linalg.reduce
// CHECK: return
func.func @reduction_tile_2(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>, %arg4: tensor<4x5xf16>, %arg5: tensor<4x5xf16>, %arg6: tensor<4x6xf32>, %arg7: tensor<4x6xf32>, %arg8: tensor<4xf32>, %arg9: tensor<4xf32>, %arg10: tensor<4x6xf16>, %arg11: tensor<4x6xf16>, %arg12: tensor<4xf16>, %arg13: tensor<4xf16>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) {
    %reduced = linalg.reduce ins(%arg0 : tensor<4x5xf32>) outs(%arg2 : tensor<4xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %0 = arith.addf %in, %init : f32
        linalg.yield %0 : f32
      }
    %reduced_0 = linalg.reduce ins(%arg1 : tensor<4x5xf32>) outs(%arg3 : tensor<4xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %0 = arith.addf %in, %init : f32
        linalg.yield %0 : f32
      }
    %reduced_1 = linalg.reduce ins(%arg4 : tensor<4x5xf16>) outs(%arg12 : tensor<4xf16>) dimensions = [1]
      (%in: f16, %init: f16) {
        %0 = arith.addf %in, %init : f16
        linalg.yield %0 : f16
      }
    %reduced_2 = linalg.reduce ins(%arg10 : tensor<4x6xf16>) outs(%arg12 : tensor<4xf16>) dimensions = [1]
      (%in: f16, %init: f16) {
        %0 = arith.addf %in, %init : f16
        %1 = arith.mulf %in, %0 : f16
        linalg.yield %1 : f16
      }
    %reduced_3 = linalg.reduce ins(%arg4 : tensor<4x5xf16>) outs(%arg12 : tensor<4xf16>) dimensions = [1]
      (%in: f16, %init: f16) {
        %0 = arith.addf %in, %init : f16
        %1 = arith.mulf %in, %0 : f16
        linalg.yield %1 : f16
      }
    %reduced_4:4 = linalg.reduce ins(%arg0, %arg1, %arg4, %arg4 : tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf16>, tensor<4x5xf16>) outs(%arg2, %arg3, %arg12, %arg12 : tensor<4xf32>, tensor<4xf32>, tensor<4xf16>, tensor<4xf16>) dimensions = [1]  {hfusion.reduce_composed = ""}
      (%in: f32, %in_6: f32, %in_7: f16, %in_8: f16, %init: f32, %init_9: f32, %init_10: f16, %init_11: f16) {
        %0 = arith.mulf %in, %init : f32
        %1 = arith.addf %in_6, %init_9 : f32
        %2 = arith.subf %in_7, %init_10 : f16
        %3 = arith.addf %in_8, %init_11 : f16
        linalg.yield %0, %1, %2, %3 : f32, f32, f16, f16
      }
    %reduced_5:4 = linalg.reduce ins(%arg6, %arg7, %arg10, %arg10 : tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf16>, tensor<4x6xf16>) outs(%arg8, %arg9, %arg12, %arg12 : tensor<4xf32>, tensor<4xf32>, tensor<4xf16>, tensor<4xf16>) dimensions = [1]  {hfusion.reduce_composed = ""}
      (%in: f32, %in_6: f32, %in_7: f16, %in_8: f16, %init: f32, %init_9: f32, %init_10: f16, %init_11: f16) {
        %0 = arith.mulf %in, %init : f32
        %1 = arith.addf %in_6, %init_9 : f32
        %2 = arith.subf %in_7, %init_10 : f16
        %3 = arith.addf %in_8, %init_11 : f16
        linalg.yield %0, %1, %2, %3 : f32, f32, f16, f16
      }
    return %reduced, %reduced_0, %reduced_1, %reduced_3, %reduced_4#2 : tensor<4xf32>, tensor<4xf32>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>
}

// -----
// CHECK-LABEL: mlir_fused_native_group_norm_backward_0
// CHECK: %[[reduced:.*]]:2 = linalg.reduce ins(
// CHECK-NOT: linalg.reduce
// CHECK: return
module {
  func.func @mlir_fused_native_group_norm_backward_0(%arg0: tensor<24x128x256x192xbf16>, %arg1: tensor<24x128x256x192xbf16>, %arg2: tensor<24x128x1x1xbf16>, %arg3: tensor<24x128x1x1xbf16>, %arg4: tensor<24x128x256x192xf32>, %arg5: tensor<24x32xf32>, %arg6: tensor<24x32xf32>, %arg7: tensor<24x128xf32>, %arg8: tensor<24x128xf32>, %arg9: tensor<24x32x4xf32>) -> (tensor<24x128xf32>, tensor<24x128xf32>, tensor<24x32x4xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant -1.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : bf16
    %cst_1 = arith.constant 1.000000e+00 : f32
    %collapsed = tensor.collapse_shape %arg6 [[0, 1]] : tensor<24x32xf32> into tensor<768xf32>
    %collapsed_2 = tensor.collapse_shape %arg5 [[0, 1]] : tensor<24x32xf32> into tensor<768xf32>
    %expanded = tensor.expand_shape %arg8 [[0], [1, 2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<24x128xf32> into tensor<24x32x4x1x1xf32>
    %collapsed_3 = tensor.collapse_shape %expanded [[0, 1], [2, 3, 4]] : tensor<24x32x4x1x1xf32> into tensor<768x4xf32>
    %collapsed_4 = tensor.collapse_shape %arg9 [[0, 1], [2]] : tensor<24x32x4xf32> into tensor<768x4xf32>
    %expanded_5 = tensor.expand_shape %arg1 [[0], [1, 2, 3, 4], [5, 6, 7], [8]] output_shape [24, 32, 4, 1, 1, 1, 1, 256, 192] : tensor<24x128x256x192xbf16> into tensor<24x32x4x1x1x1x1x256x192xbf16>
    %collapsed_6 = tensor.collapse_shape %expanded_5 [[0, 1], [2, 3, 4], [5, 6], [7, 8]] : tensor<24x32x4x1x1x1x1x256x192xbf16> into tensor<768x4x1x49152xbf16>
    %expanded_7 = tensor.expand_shape %arg0 [[0], [1, 2, 3, 4], [5, 6, 7], [8]] output_shape [24, 32, 4, 1, 1, 1, 1, 256, 192] : tensor<24x128x256x192xbf16> into tensor<24x32x4x1x1x1x1x256x192xbf16>
    %collapsed_8 = tensor.collapse_shape %expanded_7 [[0, 1], [2, 3, 4], [5, 6], [7, 8]] : tensor<24x32x4x1x1x1x1x256x192xbf16> into tensor<768x4x1x49152xbf16>
    %expanded_9 = tensor.expand_shape %arg4 [[0], [1, 2, 3, 4], [5, 6, 7], [8]] output_shape [24, 32, 4, 1, 1, 1, 1, 256, 192] : tensor<24x128x256x192xf32> into tensor<24x32x4x1x1x1x1x256x192xf32>
    %collapsed_10 = tensor.collapse_shape %expanded_9 [[0, 1], [2, 3, 4], [5, 6], [7, 8]] : tensor<24x32x4x1x1x1x1x256x192xf32> into tensor<768x4x1x49152xf32>
    %expanded_11 = tensor.expand_shape %arg7 [[0], [1, 2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<24x128xf32> into tensor<24x32x4x1x1xf32>
    %collapsed_12 = tensor.collapse_shape %expanded_11 [[0, 1], [2, 3, 4]] : tensor<24x32x4x1x1xf32> into tensor<768x4xf32>
    %0 = tensor.empty() : tensor<768x4x1x49152xbf16>
    %expanded_13 = tensor.expand_shape %arg2 [[0], [1, 2], [3], [4]] output_shape [24, 32, 4, 1, 1] : tensor<24x128x1x1xbf16> into tensor<24x32x4x1x1xbf16>
    %collapsed_14 = tensor.collapse_shape %expanded_13 [[0, 1], [2, 3, 4]] : tensor<24x32x4x1x1xbf16> into tensor<768x4xbf16>
    %1 = tensor.empty() : tensor<768x4xf32>
    %2 = tensor.empty() : tensor<768x4x1xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_14 : tensor<768x4xbf16>) outs(%1 : tensor<768x4xf32>) -> tensor<768x4xf32>
    %4 = tensor.empty() : tensor<768x4x1x49152xf32>
    %broadcasted = linalg.broadcast ins(%3 : tensor<768x4xf32>) outs(%4 : tensor<768x4x1x49152xf32>) dimensions = [2, 3]
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_6 : tensor<768x4x1x49152xbf16>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %broadcasted : tensor<768x4x1x49152xf32>, tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %expanded_15 = tensor.expand_shape %arg3 [[0], [1, 2, 3, 4], [5], [6]] output_shape [24, 32, 4, 1, 1, 1, 1] : tensor<24x128x1x1xbf16> into tensor<24x32x4x1x1x1x1xbf16>
    %collapsed_16 = tensor.collapse_shape %expanded_15 [[0, 1], [2, 3, 4], [5, 6]] : tensor<24x32x4x1x1x1x1xbf16> into tensor<768x4x1xbf16>
    %7 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_16 : tensor<768x4x1xbf16>) outs(%2 : tensor<768x4x1xf32>) -> tensor<768x4x1xf32>
    %broadcasted_17 = linalg.broadcast ins(%7 : tensor<768x4x1xf32>) outs(%4 : tensor<768x4x1x49152xf32>) dimensions = [3]
    %8 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<768x4x1x49152xbf16>) -> tensor<768x4x1x49152xbf16>
    %9 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%8 : tensor<768x4x1x49152xbf16>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_17, %9 : tensor<768x4x1x49152xf32>, tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%6, %10 : tensor<768x4x1x49152xf32>, tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%11, %cst : tensor<768x4x1x49152xf32>, f32) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %13 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%12 : tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%13, %cst_1 : tensor<768x4x1x49152xf32>, f32) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %15 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%14 : tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %16 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%15, %9 : tensor<768x4x1x49152xf32>, tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%16, %17 : tensor<768x4x1x49152xf32>, tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%11, %18 : tensor<768x4x1x49152xf32>, tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%9, %9 : tensor<768x4x1x49152xf32>, tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%19, %20 : tensor<768x4x1x49152xf32>, tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%15, %21 : tensor<768x4x1x49152xf32>, tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %23 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_8 : tensor<768x4x1x49152xbf16>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%23, %22 : tensor<768x4x1x49152xf32>, tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%24, %broadcasted : tensor<768x4x1x49152xf32>, tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %reduced = linalg.reduce ins(%25 : tensor<768x4x1x49152xf32>) outs(%collapsed_12 : tensor<768x4xf32>) dimensions = [2, 3]
      (%in: f32, %init: f32) {
        %31 = arith.addf %in, %init : f32
        linalg.yield %31 : f32
      }
    %expanded_18 = tensor.expand_shape %reduced [[0, 1], [2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<768x4xf32> into tensor<24x32x4x1x1xf32>
    %collapsed_19 = tensor.collapse_shape %expanded_18 [[0], [1, 2, 3, 4]] : tensor<24x32x4x1x1xf32> into tensor<24x128xf32>
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%25, %collapsed_10 : tensor<768x4x1x49152xf32>, tensor<768x4x1x49152xf32>) outs(%4 : tensor<768x4x1x49152xf32>) -> tensor<768x4x1x49152xf32>
    %reduced_20 = linalg.reduce ins(%26 : tensor<768x4x1x49152xf32>) outs(%collapsed_3 : tensor<768x4xf32>) dimensions = [2, 3]
      (%in: f32, %init: f32) {
        %31 = arith.addf %in, %init : f32
        linalg.yield %31 : f32
      }
    %expanded_21 = tensor.expand_shape %reduced_20 [[0, 1], [2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<768x4xf32> into tensor<24x32x4x1x1xf32>
    %collapsed_22 = tensor.collapse_shape %expanded_21 [[0], [1, 2, 3, 4]] : tensor<24x32x4x1x1xf32> into tensor<24x128xf32>
    %broadcasted_23 = linalg.broadcast ins(%collapsed_2 : tensor<768xf32>) outs(%1 : tensor<768x4xf32>) dimensions = [1]
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%reduced, %broadcasted_23 : tensor<768x4xf32>, tensor<768x4xf32>) outs(%1 : tensor<768x4xf32>) -> tensor<768x4xf32>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%27, %cst_1 : tensor<768x4xf32>, f32) outs(%1 : tensor<768x4xf32>) -> tensor<768x4xf32>
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%reduced_20, %28 : tensor<768x4xf32>, tensor<768x4xf32>) outs(%1 : tensor<768x4xf32>) -> tensor<768x4xf32>
    %broadcasted_24 = linalg.broadcast ins(%collapsed : tensor<768xf32>) outs(%1 : tensor<768x4xf32>) dimensions = [1]
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%29, %broadcasted_24 : tensor<768x4xf32>, tensor<768x4xf32>) outs(%collapsed_4 : tensor<768x4xf32>) -> tensor<768x4xf32>
    %expanded_25 = tensor.expand_shape %30 [[0, 1], [2]] output_shape [24, 32, 4] : tensor<768x4xf32> into tensor<24x32x4xf32>
    return %collapsed_19, %collapsed_22, %expanded_25 : tensor<24x128xf32>, tensor<24x128xf32>, tensor<24x32x4xf32>
  }
}

// -----
// AGGR-LABEL: mlir_fused_native_group_norm_backward_308
// AGGR: linalg.reduce
// AGGR-NOT: linalg.reduce
// AGGR: return
module {
  func.func @mlir_fused_native_group_norm_backward_308(%arg0: tensor<24x1024xf32>, %arg1: tensor<24x1024xf32>, %arg2: tensor<24x32xf32>, %arg3: tensor<24x32xf32>) -> (tensor<32x32xf32>, tensor<1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [24, 32, 32] : tensor<24x1024xf32> into tensor<24x32x32xf32>
    %expanded_0 = tensor.expand_shape %arg1 [[0], [1, 2]] output_shape [24, 32, 32] : tensor<24x1024xf32> into tensor<24x32x32xf32>
    %0 = tensor.empty() : tensor<24x32x32xf32>
    %broadcasted = linalg.broadcast ins(%arg2 : tensor<24x32xf32>) outs(%0 : tensor<24x32x32xf32>) dimensions = [2]
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_0, %broadcasted : tensor<24x32x32xf32>, tensor<24x32x32xf32>) outs(%0 : tensor<24x32x32xf32>) -> tensor<24x32x32xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%expanded, %1 : tensor<24x32x32xf32>, tensor<24x32x32xf32>) outs(%0 : tensor<24x32x32xf32>) -> tensor<24x32x32xf32>
    %broadcasted_1 = linalg.broadcast ins(%arg3 : tensor<24x32xf32>) outs(%0 : tensor<24x32x32xf32>) dimensions = [2]
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %broadcasted_1 : tensor<24x32x32xf32>, tensor<24x32x32xf32>) outs(%0 : tensor<24x32x32xf32>) -> tensor<24x32x32xf32>
    %4 = tensor.empty() : tensor<32x32xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %reduced = linalg.reduce ins(%3 : tensor<24x32x32xf32>) outs(%5 : tensor<32x32xf32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %8 = arith.addf %in, %init : f32
        linalg.yield %8 : f32
      }
    %6 = tensor.empty() : tensor<1024xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<1024xf32>) -> tensor<1024xf32>
    %reduced_2 = linalg.reduce ins(%arg1 : tensor<24x1024xf32>) outs(%7 : tensor<1024xf32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %8 = arith.addf %in, %init : f32
        linalg.yield %8 : f32
      }
    return %reduced, %reduced_2 : tensor<32x32xf32>, tensor<1024xf32>
  }
}