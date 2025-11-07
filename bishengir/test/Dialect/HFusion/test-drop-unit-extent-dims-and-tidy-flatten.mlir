// RUN: bishengir-opt -split-input-file %s                        \
// RUN:   -pass-pipeline="builtin.module(                         \
// RUN:     func.func(linalg-fold-unit-extent-dims,               \
// RUN:     canonicalize-ext,cse,propagate-reshape,cse,               \
// RUN:     canonicalize-ext,hfusion-flatten-ops{flatten-mode=tidy},  \
// RUN:     fold-tensor-empty,cse,canonicalize-ext))" | FileCheck %s

// CHECK-LABEL: main_multi_LAST_AXIS_PBR_0(
// CHECK-NOT: ?x1x1x256
func.func @main_multi_LAST_AXIS_PBR_0(%arg0: tensor<?x256xf32>, %arg1: tensor<256xf32>) -> (tensor<?xf32>, tensor<?x1xf32>, tensor<?x256xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %cst = arith.constant 9.99999974E-6 : f32
  %cst_0 = arith.constant 2.560000e+02 : f32
  %cst_1 = arith.constant 2.000000e+00 : f32
  %cst_2 = arith.constant 1.000000e+00 : f32
  %cst_3 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x256xf32>
  %0 = tensor.empty(%dim) : tensor<?x256xf32>
  %1 = tensor.empty(%dim) : tensor<?x1x1x256xf32>
  %2 = tensor.empty(%dim) : tensor<?xf32>
  %3 = linalg.fill ins(%cst_3 : f32) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
  %4 = tensor.empty(%dim) : tensor<?x1xf32>
  %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<?x1xf32>) -> tensor<?x1xf32>
  %6 = tensor.empty(%dim) : tensor<?x1x1xf32>
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2], [3]] output_shape [%dim, 1, 1, 256] : tensor<?x256xf32> into tensor<?x1x1x256xf32>
  %7 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %cst_1 : tensor<?x256xf32>, f32) outs(%0 : tensor<?x256xf32>) -> tensor<?x256xf32>
  %broadcasted = linalg.broadcast ins(%arg1 : tensor<256xf32>) outs(%1 : tensor<?x1x1x256xf32>) dimensions = [0, 1, 2]
  %reduced = linalg.reduce { arith.addf } ins(%7 : tensor<?x256xf32>) outs(%3 : tensor<?xf32>) dimensions = [1]
  %broadcasted_4 = linalg.broadcast ins(%reduced : tensor<?xf32>) outs(%4 : tensor<?x1xf32>) dimensions = [1]
  %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%broadcasted_4, %5 : tensor<?x1xf32>, tensor<?x1xf32>) outs(%4 : tensor<?x1xf32>) -> tensor<?x1xf32>
  %broadcasted_5 = linalg.broadcast ins(%8 : tensor<?x1xf32>) outs(%6 : tensor<?x1x1xf32>) dimensions = [2]
  %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%broadcasted_5, %cst : tensor<?x1x1xf32>, f32) outs(%6 : tensor<?x1x1xf32>) -> tensor<?x1x1xf32>
  %10 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%9 : tensor<?x1x1xf32>) outs(%6 : tensor<?x1x1xf32>) -> tensor<?x1x1xf32>
  %11 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%10 : tensor<?x1x1xf32>) outs(%6 : tensor<?x1x1xf32>) -> tensor<?x1x1xf32>
  %broadcasted_6 = linalg.broadcast ins(%11 : tensor<?x1x1xf32>) outs(%1 : tensor<?x1x1x256xf32>) dimensions = [3]
  %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded, %broadcasted_6 : tensor<?x1x1x256xf32>, tensor<?x1x1x256xf32>) outs(%1 : tensor<?x1x1x256xf32>) -> tensor<?x1x1x256xf32>
  %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %broadcasted : tensor<?x1x1x256xf32>, tensor<?x1x1x256xf32>) outs(%1 : tensor<?x1x1x256xf32>) -> tensor<?x1x1x256xf32>
  %collapsed = tensor.collapse_shape %13 [[0, 1, 2], [3]] : tensor<?x1x1x256xf32> into tensor<?x256xf32>
  return %3, %5, %collapsed : tensor<?xf32>, tensor<?x1xf32>, tensor<?x256xf32>
}
