// REQUIRES: asserts
// RUN: bishengir-opt --split-input-file --bisheng-segmenter="segment-size=3" %s | FileCheck %s --check-prefix="SEGMENTER"
// RUN: bishengir-opt --split-input-file --propagate-reshape --test-dimension-analyzer %s --debug 2>&1 | FileCheck %s --check-prefix="DIM-ANALYZER"
// RUN: bishengir-opt --split-input-file --instruction-marker %s | FileCheck %s --check-prefix="MARKER"

// SEGMENTER: utils_check_seg0
// SEGMENTER: utils_check_seg1
// SEGMENTER: utils_check_seg2
// SEGMENTER: utils_check_seg3
// SEGMENTER: utils_check_seg4
// DIM-ANALYZER: Max Rank is [4, 6, 32, 12]
// MARKER: return {debug = 19 : index}
module {
  func.func @utils_check(%arg0: tensor<24x32x12xf32>, %arg1: tensor<24x32x12xf32>, %arg2: tensor<24x12xf32>, %arg3: tensor<24x12xf32>) -> tensor<4x6x12xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %0 = tensor.empty() : tensor<24x32x12xf32>
    %1 = tensor.empty() : tensor<4x6x12xf32>
    %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%arg0 : tensor<24x32x12xf32>) outs(%0 : tensor<24x32x12xf32>) -> tensor<24x32x12xf32>
    %3 = tensor.empty() : tensor<f32>
    %reduced:2 = linalg.reduce ins(%arg1, %2 : tensor<24x32x12xf32>, tensor<24x32x12xf32>) outs(%arg2, %arg3 : tensor<24x12xf32>, tensor<24x12xf32>) dimensions = [1]  {hfusion.reduce_composed = ""}
      (%in: f32, %in_0: f32, %init: f32, %init_1: f32) {
        %6 = arith.addf %in, %init : f32
        %7 = arith.addf %in_0, %init_1 : f32
        linalg.yield %6, %7 : f32, f32
      }
    %expanded = tensor.expand_shape %reduced#1 [[0, 1], [2]] output_shape [4, 6, 12] : tensor<24x12xf32> into tensor<4x6x12xf32>
    %4 = tensor.empty() : tensor<1xi32>
    %5 = linalg.elemwise_unary ins(%expanded : tensor<4x6x12xf32>) outs(%1 : tensor<4x6x12xf32>) -> tensor<4x6x12xf32>
    %6 = linalg.elemwise_unary ins(%5 : tensor<4x6x12xf32>) outs(%1 : tensor<4x6x12xf32>) -> tensor<4x6x12xf32>
    return %6 : tensor<4x6x12xf32>
  }
}
