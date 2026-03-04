// REQUIRES: asserts
// RUN: bishengir-opt %s -hfusion-auto-schedule -split-input-file -debug-only="fusible-producer-analyzer" 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG

// CHECK-DEBUG: Analyzing reduction producers for consumer #0
// CHECK-DEBUG: linalg.reduce
// CHECK-DEBUG: Collecting fusible producers that share reduction axis: 2. W.r.t to the anchor the axis is 2
// CHECK-DEBUG: nextOperation {{.*}} hfusion.cast {{.*}} is fusible
// CHECK-DEBUG: nextOperation {{.*}} linalg.elemwise_binary {{.*}} is fusible
// CHECK-DEBUG: nextOperation {{.*}} hfusion.load {{.*}} is fusible
// CHECK-DEBUG: nextOperation {{.*}} tensor.collapse_shape {{.*}} is not fusible
// CHECK-DEBUG: nextOperation {{.*}} linalg.broadcast {{.*}} is fusible
// CHECK-DEBUG: nextOperation {{.*}} linalg.elemwise_binary {{.*}} is fusible
// CHECK-DEBUG: nextOperation {{.*}} hfusion.load {{.*}} is fusible
// CHECK-DEBUG: nextOperation {{.*}} tensor.expand_shape {{.*}} is not fusible
// CHECK-DEBUG: nextOperation {{.*}} linalg.broadcast {{.*}} is fusible
// CHECK-DEBUG: nextOperation {{.*}} hfusion.load {{.*}} is fusible
// CHECK-DEBUG: nextOperation {{.*}} tensor.expand_shape {{.*}} is not fusible

// CHECK-DEBUG: Collecting fusible producers that share reduction axis: 3. W.r.t to the anchor the axis is 3
// CHECK-DEBUG: nextOperation {{.*}} hfusion.cast {{.*}} is fusible
// CHECK-DEBUG: nextOperation {{.*}} linalg.elemwise_binary{{.*}} is fusible
// CHECK-DEBUG: nextOperation {{.*}} hfusion.load {{.*}} is fusible
// CHECK-DEBUG: nextOperation {{.*}} tensor.collapse_shape {{.*}} is not fusible
// CHECK-DEBUG: nextOperation {{.*}} linalg.broadcast {{.*}} is fusible
// CHECK-DEBUG: nextOperation {{.*}} linalg.elemwise_binary {{.*}} ins({{.*}}) outs({{.*}}) -> tensor<2x32x40xf16> is not fusible because it failed the test.

// CHECK-DEBUG: Analyzing reduction producers for consumer #0
// CHECK-DEBUG: hfusion.store

// CHECK-DEBUG: Analyzing reduction producers for consumer #1
// CHECK-DEBUG: hfusion.store

module {
  func.func @mlir_fused_native_group_norm_35(%arg0: tensor<2x1280x16x16xf16>, %arg1: tensor<2x1280xf16>, %arg2: tensor<1280xf16>) -> (tensor<2x32x1x1xf32>, tensor<2x32x1x1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %cst = arith.constant 1.024000e+04 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x32x40xf16>
    %1 = tensor.empty() : tensor<2x32x40x256xf16>
    %2 = tensor.empty() : tensor<2x32x40x256xf32>
    %3 = tensor.empty() : tensor<2x32xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<2x32xf32>) -> tensor<2x32xf32>
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2, 3, 4], [5], [6]] output_shape [2, 32, 1, 1, 40, 16, 16] : tensor<2x1280x16x16xf16> into tensor<2x32x1x1x40x16x16xf16>
    %collapsed = tensor.collapse_shape %expanded [[0], [1, 2, 3], [4], [5, 6]] : tensor<2x32x1x1x40x16x16xf16> into tensor<2x32x40x256xf16>
    %expanded_1 = tensor.expand_shape %arg1 [[0], [1, 2]] output_shape [2, 32, 40] : tensor<2x1280xf16> into tensor<2x32x40xf16>
    %expanded_2 = tensor.expand_shape %arg2 [[0, 1]] output_shape [32, 40] : tensor<1280xf16> into tensor<32x40xf16>
    %broadcasted = linalg.broadcast ins(%expanded_2 : tensor<32x40xf16>) outs(%0 : tensor<2x32x40xf16>) dimensions = [0]
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%expanded_1, %broadcasted : tensor<2x32x40xf16>, tensor<2x32x40xf16>) outs(%0 : tensor<2x32x40xf16>) -> tensor<2x32x40xf16>
    %broadcasted_3 = linalg.broadcast ins(%5 : tensor<2x32x40xf16>) outs(%1 : tensor<2x32x40x256xf16>) dimensions = [3]
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%collapsed, %broadcasted_3 : tensor<2x32x40x256xf16>, tensor<2x32x40x256xf16>) outs(%1 : tensor<2x32x40x256xf16>) -> tensor<2x32x40x256xf16>
    %7 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%6 : tensor<2x32x40x256xf16>) outs(%2 : tensor<2x32x40x256xf32>) -> tensor<2x32x40x256xf32>
    %reduced = linalg.reduce ins(%7 : tensor<2x32x40x256xf32>) outs(%4 : tensor<2x32xf32>) dimensions = [2, 3]
      (%in: f32, %init: f32) {
        %11 = arith.addf %in, %init : f32
        linalg.yield %11 : f32
      }
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced, %cst : tensor<2x32xf32>, f32) outs(%3 : tensor<2x32xf32>) -> tensor<2x32xf32>
    %broadcasted_4 = linalg.broadcast ins(%8 : tensor<2x32xf32>) outs(%2 : tensor<2x32x40x256xf32>) dimensions = [2, 3]
    %expanded_5 = tensor.expand_shape %8 [[0], [1, 2, 3]] output_shape [2, 32, 1, 1] : tensor<2x32xf32> into tensor<2x32x1x1xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%7, %broadcasted_4 : tensor<2x32x40x256xf32>, tensor<2x32x40x256xf32>) outs(%2 : tensor<2x32x40x256xf32>) -> tensor<2x32x40x256xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%9, %9 : tensor<2x32x40x256xf32>, tensor<2x32x40x256xf32>) outs(%2 : tensor<2x32x40x256xf32>) -> tensor<2x32x40x256xf32>
    %reduced_6 = linalg.reduce ins(%10 : tensor<2x32x40x256xf32>) outs(%4 : tensor<2x32xf32>) dimensions = [2, 3]
      (%in: f32, %init: f32) {
        %11 = arith.addf %in, %init : f32
        linalg.yield %11 : f32
      }
    %expanded_7 = tensor.expand_shape %reduced_6 [[0], [1, 2, 3]] output_shape [2, 32, 1, 1] : tensor<2x32xf32> into tensor<2x32x1x1xf32>
    return %expanded_5, %expanded_7 : tensor<2x32x1x1xf32>, tensor<2x32x1x1xf32>
  }
}
