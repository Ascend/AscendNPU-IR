// RUN: bishengir-opt -lower-hfusion-pipeline --split-input-file %s | FileCheck %s
// RUN: bishengir-opt -lower-hfusion-pipeline="block-dim=40" --split-input-file %s | FileCheck %s -check-prefix=CHECK-BLOCK-DIM

module {
// CHECK-DAG: test_0_0
// CHECK-NOT: test_0_tiling_func

// CHECK-BLOCK-DIM: hacc.block_dim = 40
  func.func @test(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>) -> tensor<8xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %0 = tensor.empty() : tensor<8xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, mul} ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %2 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<8xf32>, tensor<8xf32>) outs(%arg3 : tensor<8xf32>) -> tensor<8xf32>
    return %2 : tensor<8xf32>
  }
}

// -----
// This is when multi dimension is true, else, fold more careful
// MULTI-DYN-TRUE-LABEL: func @broadcast_test
// MULTI-DYN-TRUE: linalg.broadcast
// MULTI-DYN-TRUE-SAME: tensor<4x?xf32>
// MULTI-DYN-TRUE-SAME: tensor<4x3x?x14xf32>
// MULTI-DYN-TRUE-SAME: dimensions = {{\[}}1, 3]
// MULTI-DYN-TRUE: return
// MULTI-DYN-TRUE-SAME: 1x4x3x8x1x1x?x3x1x1x?x1x14xf32


// CHECK-LABEL: func @broadcast_test
// CHECK: linalg.broadcast
// CHECK-SAME: tensor<4x?x?xf32>
// CHECK-SAME: tensor<4x3x?x?x14xf32>
// CHECK-SAME: dimensions = {{\[}}1, 4]
// CHECK: return
// CHECK-SAME: <1x4x3x8x1x1x?x3x1x1x?x1x14xf32>
func.func @broadcast_test(%arg0: tensor<4x8x?x3x?xf32>) -> tensor<1x4x3x8x1x1x?x3x1x1x?x1x14xf32> {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %dim_2 = tensor.dim %arg0, %c2 : tensor<4x8x?x3x?xf32>
    %dim_4 = tensor.dim %arg0, %c4 : tensor<4x8x?x3x?xf32>

    %init = tensor.empty(%dim_2, %dim_4) : tensor<1x4x3x8x1x1x?x3x1x1x?x1x14xf32>
// linalg.broadcast ins(%collapsed : tensor<4x?xf32>) outs(%1 : tensor<4x3x?x14xf32>) dimensions = [1, 3]
    %0 = linalg.broadcast ins(%arg0 : tensor<4x8x?x3x?xf32>)
                         outs(%init : tensor<1x4x3x8x1x1x?x3x1x1x?x1x14xf32>)
                         //                  0 1 2 3 4 5 6 7 8 9 0 1 2
                         //                    4   8     ? 3     ?
                         //                  U   X   U U     U U   U X
                         dimensions = [0, 2, 4, 5, 8, 9, 11, 12]
    return %0 : tensor<1x4x3x8x1x1x?x3x1x1x?x1x14xf32>
}

// -----

// CHECK: func.func @mlir_fused_add_full_mul_npu_dtype_cast_sigmoid_sub_sum_0(
// CHECK-SAME: LAST_AXIS_PBR
 module {
  func.func @mlir_fused_add_full_mul_npu_dtype_cast_sigmoid_sub_sum_0(%arg0: tensor<24x512x16x12xbf16>, %arg1: tensor<24x512x16x12xbf16>, %arg2: tensor<24x512x1x1xbf16>, %arg3: tensor<24x512x1x1xbf16>, %arg4: tensor<24x512x16x12xf32>) -> (tensor<24x512xf32>, tensor<24x512xf32>, tensor<24x512x1x1xf32>, tensor<24x512x1x1xf32>, tensor<24x512x1x1xbf16>, tensor<24x512x1x1xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant dense<1> : tensor<i64>
    %c1_i64 = arith.constant 1 : i64
    %cst_0 = arith.constant 1.000000e+00 : bf16
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<24x512x16x12xbf16>
    %collapsed = tensor.collapse_shape %arg2 [[0], [1, 2, 3]] : tensor<24x512x1x1xbf16> into tensor<24x512xbf16>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<24x512xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) dimensions = [2, 3]
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %broadcasted : tensor<24x512x16x12xbf16>, tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %collapsed_2 = tensor.collapse_shape %arg3 [[0], [1, 2, 3]] : tensor<24x512x1x1xbf16> into tensor<24x512xbf16>
    %broadcasted_3 = linalg.broadcast ins(%collapsed_2 : tensor<24x512xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) dimensions = [2, 3]
    %2 = tensor.empty() : tensor<24x512x16x12xi64>
    %3 = linalg.fill ins(%c1_i64 : i64) outs(%2 : tensor<24x512x16x12xi64>) -> tensor<24x512x16x12xi64>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%3 : tensor<24x512x16x12xi64>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_3, %4 : tensor<24x512x16x12xbf16>, tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %5 : tensor<24x512x16x12xbf16>, tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%6 : tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%7 : tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%8, %cst_0 : tensor<24x512x16x12xbf16>, bf16) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst_0, %9 : bf16, tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %11 = arith.sitofp %cst : tensor<i64> to tensor<bf16>
    %broadcasted_4 = linalg.broadcast ins(%11 : tensor<bf16>) outs(%0 : tensor<24x512x16x12xbf16>) dimensions = [0, 1, 2, 3]
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %4 : tensor<24x512x16x12xbf16>, tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%broadcasted_4, %12 : tensor<24x512x16x12xbf16>, tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %13 : tensor<24x512x16x12xbf16>, tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %4 : tensor<24x512x16x12xbf16>, tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%14, %15 : tensor<24x512x16x12xbf16>, tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %16 : tensor<24x512x16x12xbf16>, tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %17 : tensor<24x512x16x12xbf16>, tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%18, %broadcasted : tensor<24x512x16x12xbf16>, tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %20 = tensor.empty() : tensor<24x512x16x12xf32>
    %21 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%19 : tensor<24x512x16x12xbf16>) outs(%20 : tensor<24x512x16x12xf32>) -> tensor<24x512x16x12xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%21, %arg4 : tensor<24x512x16x12xf32>, tensor<24x512x16x12xf32>) outs(%20 : tensor<24x512x16x12xf32>) -> tensor<24x512x16x12xf32>
    %collapsed_5 = tensor.collapse_shape %22 [[0], [1], [2, 3]] : tensor<24x512x16x12xf32> into tensor<24x512x192xf32>
    %23 = tensor.empty() : tensor<24x512xf32>
    %24 = linalg.fill ins(%cst_1 : f32) outs(%23 : tensor<24x512xf32>) -> tensor<24x512xf32>
    %reduced = linalg.reduce ins(%collapsed_5 : tensor<24x512x192xf32>) outs(%24 : tensor<24x512xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %31 = arith.addf %in, %init : f32
        linalg.yield %31 : f32
      }
    %collapsed_6 = tensor.collapse_shape %21 [[0], [1], [2, 3]] : tensor<24x512x16x12xf32> into tensor<24x512x192xf32>
    %reduced_7 = linalg.reduce ins(%collapsed_6 : tensor<24x512x192xf32>) outs(%24 : tensor<24x512xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %31 = arith.addf %in, %init : f32
        linalg.yield %31 : f32
      }
    %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%18, %arg1 : tensor<24x512x16x12xbf16>, tensor<24x512x16x12xbf16>) outs(%0 : tensor<24x512x16x12xbf16>) -> tensor<24x512x16x12xbf16>
    %26 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%25 : tensor<24x512x16x12xbf16>) outs(%20 : tensor<24x512x16x12xf32>) -> tensor<24x512x16x12xf32>
    %reduced_8 = linalg.reduce ins(%26 : tensor<24x512x16x12xf32>) outs(%24 : tensor<24x512xf32>) dimensions = [2, 3]
      (%in: f32, %init: f32) {
        %31 = arith.addf %in, %init : f32
        linalg.yield %31 : f32
      }
    %expanded = tensor.expand_shape %reduced_8 [[0], [1, 2, 3]] output_shape [24, 512, 1, 1] : tensor<24x512xf32> into tensor<24x512x1x1xf32>
    %27 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%18 : tensor<24x512x16x12xbf16>) outs(%20 : tensor<24x512x16x12xf32>) -> tensor<24x512x16x12xf32>
    %reduced_9 = linalg.reduce ins(%27 : tensor<24x512x16x12xf32>) outs(%24 : tensor<24x512xf32>) dimensions = [2, 3]
      (%in: f32, %init: f32) {
        %31 = arith.addf %in, %init : f32
        linalg.yield %31 : f32
      }
    %expanded_10 = tensor.expand_shape %reduced_9 [[0], [1, 2, 3]] output_shape [24, 512, 1, 1] : tensor<24x512xf32> into tensor<24x512x1x1xf32>
    %28 = tensor.empty() : tensor<24x512x1x1xbf16>
    %29 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded : tensor<24x512x1x1xf32>) outs(%28 : tensor<24x512x1x1xbf16>) -> tensor<24x512x1x1xbf16>
    %30 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded_10 : tensor<24x512x1x1xf32>) outs(%28 : tensor<24x512x1x1xbf16>) -> tensor<24x512x1x1xbf16>
    return %reduced, %reduced_7, %expanded, %expanded_10, %29, %30 : tensor<24x512xf32>, tensor<24x512xf32>, tensor<24x512x1x1xf32>, tensor<24x512x1x1xf32>, tensor<24x512x1x1xbf16>, tensor<24x512x1x1xbf16>
  }
}

// -----
// CHECK-LABEL: func.func @mlir_fused_add_0_0_0(
// CHECK: <24x512xf32>
// CHECK: return
module {
  func.func @mlir_fused_add_0(%arg0: tensor<24x1024xbf16>) -> tensor<24x512x1x1xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %cst = arith.constant 1.000000e+00 : bf16
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2, 3]] output_shape [24, 1024, 1, 1] : tensor<24x1024xbf16> into tensor<24x1024x1x1xbf16>
    %extracted_slice = tensor.extract_slice %expanded[0, 0, 0, 0] [24, 512, 1, 1] [1, 1, 1, 1] : tensor<24x1024x1x1xbf16> to tensor<24x512x1x1xbf16>
    %0 = tensor.empty() : tensor<24x512x1x1xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<24x512x1x1xbf16>) -> tensor<24x512x1x1xbf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %1 : tensor<24x512x1x1xbf16>, tensor<24x512x1x1xbf16>) outs(%0 : tensor<24x512x1x1xbf16>) -> tensor<24x512x1x1xbf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%extracted_slice, %2 : tensor<24x512x1x1xbf16>, tensor<24x512x1x1xbf16>) outs(%0 : tensor<24x512x1x1xbf16>) -> tensor<24x512x1x1xbf16>
    return %3 : tensor<24x512x1x1xbf16>
  }
}

// -----

// CHECK-LABEL: Fused_Select_split_9728772994794235039_0_1(
// CHECK-SAME: ANY_PB
// CHECK: tensor<8x884736xf32>
// CHECK: return
module {
  func.func @Fused_Select_split_9728772994794235039(%arg0: tensor<2x4x1x1xi1>, %arg1: tensor<2x4x768x1152xf32>, %arg2: tensor<2x4x768x1152xf32>) -> tensor<2x4x768x1152xf32> attributes {OperatorType = "Default", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
    %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<2x4x1x1xi1> into tensor<2x4xi1>
    %0 = tensor.empty() : tensor<2x4x768x1152xi1>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<2x4xi1>) outs(%0 : tensor<2x4x768x1152xi1>) dimensions = [2, 3]
    %1 = tensor.empty() : tensor<2x4x768x1152xf32>
    %2 = hfusion.select ins(%broadcasted, %arg1, %arg2 : tensor<2x4x768x1152xi1>, tensor<2x4x768x1152xf32>, tensor<2x4x768x1152xf32>) outs(%1 : tensor<2x4x768x1152xf32>) -> tensor<2x4x768x1152xf32>
    return %2 : tensor<2x4x768x1152xf32>
  }
}