// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=1" -split-input-file | FileCheck %s

// CHECK-LABEL: mlir_fused_add_7
// CHECK-NOT: error
func.func @mlir_fused_add_7(%arg0: tensor<24x1024xbf16>, %arg1: tensor<24x512x1x1xbf16>) -> tensor<24x512x1x1xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %cst = arith.constant 1.000000e+00 : bf16
    %collapsed = tensor.collapse_shape %arg1 [[0, 1, 2, 3]] : tensor<24x512x1x1xbf16> into tensor<12288xbf16>
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2, 3]] output_shape [24, 1024, 1, 1] : tensor<24x1024xbf16> into tensor<24x1024x1x1xbf16>
    %collapsed_0 = tensor.collapse_shape %expanded [[0, 1], [2], [3]] : tensor<24x1024x1x1xbf16> into tensor<24576x1x1xbf16>
    %extracted_slice = tensor.extract_slice %collapsed_0[0, 0, 0] [12288, 1, 1] [1, 1, 1] : tensor<24576x1x1xbf16> to tensor<12288xbf16>
    %0 = tensor.empty() : tensor<12288xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<12288xbf16>) -> tensor<12288xbf16>
    %2 = tensor.empty() : tensor<12288xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%extracted_slice : tensor<12288xbf16>) outs(%2 : tensor<12288xf32>) -> tensor<12288xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %3 : tensor<12288xf32>, tensor<12288xf32>) outs(%2 : tensor<12288xf32>) -> tensor<12288xf32>
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%extracted_slice : tensor<12288xbf16>) outs(%2 : tensor<12288xf32>) -> tensor<12288xf32>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %4 : tensor<12288xf32>, tensor<12288xf32>) outs(%2 : tensor<12288xf32>) -> tensor<12288xf32>
    %7 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%6 : tensor<12288xf32>) outs(%collapsed : tensor<12288xbf16>) -> tensor<12288xbf16>
    %expanded_1 = tensor.expand_shape %7 [[0, 1, 2, 3]] output_shape [24, 512, 1, 1] : tensor<12288xbf16> into tensor<24x512x1x1xbf16>
    return %expanded_1 : tensor<24x512x1x1xbf16>
}