// REQUIRES: asserts
// RUN: bishengir-opt %s -hfusion-auto-schedule -debug-only="hfusion-auto-schedule" 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG

module {
  func.func @Fused_Add_Mul_Mul_Cast_ReduceSum_split_14035710275516055285_0(%arg0: tensor<1x224x3072xbf16>, %arg1: tensor<1x1x3072xbf16>, %arg2: tensor<1x224x3072xbf16>, %arg3: tensor<1x224x3072xbf16>) -> (tensor<1x1x3072xf32>, tensor<1x224x3072xbf16>, tensor<1x224x3072xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<224x3072xbf16>
    %1 = tensor.empty() : tensor<224x3072xf32>
    %2 = tensor.empty() : tensor<3072xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<3072xf32>) -> tensor<3072xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x224x3072xbf16> into tensor<224x3072xbf16>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<224x3072xbf16>) outs(%1 : tensor<224x3072xf32>) -> tensor<224x3072xf32>
    %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1, 2]] : tensor<1x1x3072xbf16> into tensor<3072xbf16>
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_0 : tensor<3072xbf16>) outs(%2 : tensor<3072xf32>) -> tensor<3072xf32>
    %collapsed_1 = tensor.collapse_shape %arg2 [[0, 1], [2]] : tensor<1x224x3072xbf16> into tensor<224x3072xbf16>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_1 : tensor<224x3072xbf16>) outs(%1 : tensor<224x3072xf32>) -> tensor<224x3072xf32>
    %collapsed_2 = tensor.collapse_shape %arg3 [[0, 1], [2]] : tensor<1x224x3072xbf16> into tensor<224x3072xbf16>
    %7 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_2 : tensor<224x3072xbf16>) outs(%1 : tensor<224x3072xf32>) -> tensor<224x3072xf32>
    %broadcasted = linalg.broadcast ins(%5 : tensor<3072xf32>) outs(%1 : tensor<224x3072xf32>) dimensions = [0]
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%7, %6 : tensor<224x3072xf32>, tensor<224x3072xf32>) outs(%1 : tensor<224x3072xf32>) -> tensor<224x3072xf32>
    %9 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%8 : tensor<224x3072xf32>) outs(%0 : tensor<224x3072xbf16>) -> tensor<224x3072xbf16>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %8 : tensor<224x3072xf32>, tensor<224x3072xf32>) outs(%1 : tensor<224x3072xf32>) -> tensor<224x3072xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %8 : tensor<224x3072xf32>, tensor<224x3072xf32>) outs(%1 : tensor<224x3072xf32>) -> tensor<224x3072xf32>
    %expanded = tensor.expand_shape %9 [[0, 1], [2]] output_shape [1, 224, 3072] : tensor<224x3072xbf16> into tensor<1x224x3072xbf16>
    %12 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%10 : tensor<224x3072xf32>) outs(%0 : tensor<224x3072xbf16>) -> tensor<224x3072xbf16>
    %reduced = linalg.reduce ins(%11 : tensor<224x3072xf32>) outs(%3 : tensor<3072xf32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    %expanded_3 = tensor.expand_shape %12 [[0, 1], [2]] output_shape [1, 224, 3072] : tensor<224x3072xbf16> into tensor<1x224x3072xbf16>
    %expanded_4 = tensor.expand_shape %reduced [[0, 1, 2]] output_shape [1, 1, 3072] : tensor<3072xf32> into tensor<1x1x3072xf32>
    return %expanded_4, %expanded, %expanded_3 : tensor<1x1x3072xf32>, tensor<1x224x3072xbf16>, tensor<1x224x3072xbf16>
  }
}

// -----

func.func @rank_increase_and_decrease(%arg0: tensor<10x16x4x128xf16>, %arg1: tensor<10x16x4xf16>, %arg2: tensor<10x256x4x128xf16>, %arg3: tensor<10x256x4xf16>) -> tensor<10x16x256x4xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %0 = tensor.empty() : tensor<10x16x4x128xf16>
  %1 = tensor.empty() : tensor<10x256x4x128xf16>
  %2 = tensor.empty() : tensor<10x16x256x4x128xf16>
  %3 = tensor.empty() : tensor<10x16x256x4xf16>
  %4 = tensor.empty() : tensor<10x16x256x4x128xf32>
  %5 = tensor.empty() : tensor<10x16x256x4xf32>
  %broadcasted = linalg.broadcast ins(%arg1 : tensor<10x16x4xf16>) outs(%0 : tensor<10x16x4x128xf16>) dimensions = [3] 
  %broadcasted_0 = linalg.broadcast ins(%arg3 : tensor<10x256x4xf16>) outs(%1 : tensor<10x256x4x128xf16>) dimensions = [3] 
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %broadcasted : tensor<10x16x4x128xf16>, tensor<10x16x4x128xf16>) outs(%0 : tensor<10x16x4x128xf16>) -> tensor<10x16x4x128xf16>
  %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg2, %broadcasted_0 : tensor<10x256x4x128xf16>, tensor<10x256x4x128xf16>) outs(%1 : tensor<10x256x4x128xf16>) -> tensor<10x256x4x128xf16>
  %broadcasted_1 = linalg.broadcast ins(%6 : tensor<10x16x4x128xf16>) outs(%2 : tensor<10x16x256x4x128xf16>) dimensions = [2] 
  %broadcasted_2 = linalg.broadcast ins(%7 : tensor<10x256x4x128xf16>) outs(%2 : tensor<10x16x256x4x128xf16>) dimensions = [1] 
  %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_1, %broadcasted_2 : tensor<10x16x256x4x128xf16>, tensor<10x16x256x4x128xf16>) outs(%2 : tensor<10x16x256x4x128xf16>) -> tensor<10x16x256x4x128xf16>
  %9 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%8 : tensor<10x16x256x4x128xf16>) outs(%4 : tensor<10x16x256x4x128xf32>) -> tensor<10x16x256x4x128xf32>
  %reduced = linalg.reduce ins(%9 : tensor<10x16x256x4x128xf32>) outs(%5 : tensor<10x16x256x4xf32>) dimensions = [4] 
    (%in: f32, %init: f32) {
      %11 = arith.addf %in, %init : f32
      linalg.yield %11 : f32
    }
  %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%reduced : tensor<10x16x256x4xf32>) outs(%3 : tensor<10x16x256x4xf16>) -> tensor<10x16x256x4xf16>
  return %10 : tensor<10x16x256x4xf16>
}

// -----

// CHECK-DEBUG-DAG: affine_map<()[s0, s1] -> ((s0 * s1) * 16)>
// CHECK-DEBUG-DAG: affine_map<()[s0, s1] -> (s0 * s1)>
func.func @dim_rank_from_collapse_shape(%arg0: tensor<?x?x16x?x?xf16>, %arg1: tensor<?x?x16x?x?xf16>, %arg2: tensor<?x?x16x?x?xf16>) -> tensor<?x?x16x?x?xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?x16x?x?xf16>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?x16x?x?xf16>
  %dim3 = tensor.dim %arg0, %c3 : tensor<?x?x16x?x?xf16>
  %dim4 = tensor.dim %arg0, %c4 : tensor<?x?x16x?x?xf16>
  %collapsed = tensor.collapse_shape %arg2 [[0, 1, 2], [3, 4]] : tensor<?x?x16x?x?xf16> into tensor<?x?xf16>
  %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1, 2], [3, 4]] : tensor<?x?x16x?x?xf16> into tensor<?x?xf16>
  %collapsed_1 = tensor.collapse_shape %arg0 [[0, 1, 2], [3, 4]] : tensor<?x?x16x?x?xf16> into tensor<?x?xf16>
  %0 = tensor.empty(%dim0, %dim1, %dim3, %dim4) : tensor<?x?x16x?x?xf16>
  %collapsed_2 = tensor.collapse_shape %0 [[0, 1, 2], [3, 4]] : tensor<?x?x16x?x?xf16> into tensor<?x?xf16>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed_1, %collapsed_0 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%collapsed_2 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %collapsed : tensor<?x?xf16>, tensor<?x?xf16>) outs(%collapsed_2 : tensor<?x?xf16>) -> tensor<?x?xf16>
  %expanded = tensor.expand_shape %2 [[0, 1, 2], [3, 4]] output_shape [%dim0, %dim1, 16, %dim3, %dim4] : tensor<?x?xf16> into tensor<?x?x16x?x?xf16>
  return %expanded : tensor<?x?x16x?x?xf16>
}

// -----

// CHECK-DEBUG: affine_map<()[s0] -> (s0 + 20)>
func.func @max_dynamic_anchor_dim(%arg0: tensor<1x?x2048xf32>) -> tensor<1x?x2048xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %0 = tensor.empty() : tensor<20x2048xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<20x2048xf32>) -> tensor<20x2048xf32>
  %dim = tensor.dim %arg0, %c1 : tensor<1x?x2048xf32>
  %2 = tensor.empty(%dim) : tensor<?x2048xf16>
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x?x2048xf32> into tensor<?x2048xf32>
  %concat = tensor.concat dim(0) %collapsed, %1 : (tensor<?x2048xf32>, tensor<20x2048xf32>) -> tensor<?x2048xf32>
  %extracted_slice = tensor.extract_slice %concat[0, 0] [%dim, 2048] [1, 1] : tensor<?x2048xf32> to tensor<?x2048xf32>
  %expanded = tensor.expand_shape %extracted_slice [[0, 1], [2]] output_shape [1, %dim, 2048] : tensor<?x2048xf32> into tensor<1x?x2048xf32>
  return %expanded : tensor<1x?x2048xf32>
}

// -----

// CHECK-LABEL: func.func @test_max_shape_anchor_tiling_function(
// CHECK: %[[c4096:.*]] = arith.constant 4096 : i64
// CHECK: return %{{.*}}, %{{.*}}, %[[c4096]]
func.func @test_max_shape_anchor(%arg0: tensor<2048xf32>) -> tensor<64x4096xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<64x2048xf32>
  %broadcasted = linalg.broadcast ins(%arg0 : tensor<2048xf32>) outs(%0 : tensor<64x2048xf32>) dimensions = [0] 
  %padded = tensor.pad %broadcasted low[0, 0] high[0, 2048] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %cst : f32
  } : tensor<64x2048xf32> to tensor<64x4096xf32>
  return %padded : tensor<64x4096xf32>
}