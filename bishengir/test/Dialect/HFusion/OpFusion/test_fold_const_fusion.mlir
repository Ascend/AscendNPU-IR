// RUN: bishengir-opt %s --hfusion-normalize-ops --hfusion-fuse-ops --split-input-file | FileCheck %s

// CHECK: @smallest_0(
// CHECK-SAME: ANY_PB
func.func @smallest(%arg0: tensor<8x768x16x1xf32>, %arg1: tensor<8x768x16x1xf32>) -> tensor<8x768x16x1xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %2 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%arg0 : tensor<8x768x16x1xf32>) outs(%arg1 : tensor<8x768x16x1xf32>) -> tensor<8x768x16x1xf32>
    return %2 : tensor<8x768x16x1xf32>
}