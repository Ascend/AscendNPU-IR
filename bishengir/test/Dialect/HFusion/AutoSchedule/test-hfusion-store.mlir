// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20" -split-input-file | FileCheck %s

// -----
// CHECK-LABEL: @test_same_store
// CHECK: hfusion.store
// CHECK: hfusion.store
// CHECK: hfusion.store
module {
  func.func @test_same_store(%arg0: tensor<24x128xbf16>) -> (tensor<128xbf16>, tensor<128xf32>, tensor<128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<24x128xf32>
    %1 = tensor.empty() : tensor<128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<128xf32>) -> tensor<128xf32>
    %3 = tensor.empty() : tensor<128xbf16>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<24x128xbf16>) outs(%0 : tensor<24x128xf32>) -> tensor<24x128xf32>
    %reduced = linalg.reduce ins(%4 : tensor<24x128xf32>) outs(%2 : tensor<128xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %6 = arith.addf %in, %init : f32
        linalg.yield %6 : f32
      }
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%reduced : tensor<128xf32>) outs(%3 : tensor<128xbf16>) -> tensor<128xbf16>
    return %5, %reduced, %reduced : tensor<128xbf16>, tensor<128xf32>, tensor<128xf32>
  }
}

// -----

// CHECK-LABEL: @test_complicated_duplicate_store
// CHECK: hfusion.store
// CHECK: hfusion.store
// CHECK: hfusion.store
// CHECK: hfusion.store
// CHECK: hfusion.store
// CHECK: hfusion.store
module {
  func.func @test_complicated_duplicate_store(%arg0: tensor<24x32x16x48xf32>, %arg1: tensor<24x32x16x48xf32>, %arg2: tensor<24x32x16x48xf32>, %arg3: tensor<24x32x16x48xf32>) -> (tensor<24x512xf32>, tensor<24x32x16x48xf32>, tensor<24x512x8x6xbf16>, tensor<24x512xf32>, tensor<24x512xf32>, tensor<24x512xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %cst = arith.constant 0.00130208337 : f32
    %cst_0 = arith.constant -1.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<24x32x16x48xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<24x32x16x48xf32>) -> tensor<24x32x16x48xf32>
    %2 = tensor.empty() : tensor<24x32x16xf32>
    %3 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<24x32x16xf32>) -> tensor<24x32x16xf32>
    %4 = tensor.empty() : tensor<24x32x16x48xbf16>
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg2, %arg3 : tensor<24x32x16x48xf32>, tensor<24x32x16x48xf32>) outs(%0 : tensor<24x32x16x48xf32>) -> tensor<24x32x16x48xf32>
    %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<24x32x16x48xf32>, tensor<24x32x16x48xf32>) outs(%0 : tensor<24x32x16x48xf32>) -> tensor<24x32x16x48xf32>
    %35 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%34 : tensor<24x32x16x48xf32>) outs(%4 : tensor<24x32x16x48xbf16>) -> tensor<24x32x16x48xbf16>
    %reduced:2 = linalg.reduce ins(%arg0, %34 : tensor<24x32x16x48xf32>, tensor<24x32x16x48xf32>) outs(%3, %3 : tensor<24x32x16xf32>, tensor<24x32x16xf32>) dimensions = [3]  {hfusion.reduce_composed = ""}
      (%in: f32, %in_18: f32, %init: f32, %init_19: f32) {
        %36 = arith.addf %in, %init : f32
        %37 = arith.addf %in_18, %init_19 : f32
        linalg.yield %36, %37 : f32, f32
      }
    %expanded_14 = tensor.expand_shape %35 [[0], [1], [2], [3, 4]] output_shape [24, 32, 16, 8, 6] : tensor<24x32x16x48xbf16> into tensor<24x32x16x8x6xbf16>
    %collapsed_15 = tensor.collapse_shape %expanded_14 [[0], [1, 2], [3], [4]] : tensor<24x32x16x8x6xbf16> into tensor<24x512x8x6xbf16>
    %collapsed_16 = tensor.collapse_shape %reduced#0 [[0], [1, 2]] : tensor<24x32x16xf32> into tensor<24x512xf32>
    %collapsed_17 = tensor.collapse_shape %reduced#1 [[0], [1, 2]] : tensor<24x32x16xf32> into tensor<24x512xf32>
    return %collapsed_16, %33, %collapsed_15, %collapsed_17, %collapsed_16, %collapsed_17 : tensor<24x512xf32>, tensor<24x32x16x48xf32>, tensor<24x512x8x6xbf16>, tensor<24x512xf32>, tensor<24x512xf32>, tensor<24x512xf32>
  }
}