// RUN: bishengir-opt %s -canonicalize-ext -split-input-file | FileCheck %s

// CHECK-LABEL: @expand_transpose_fold(
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: linalg.transpose
// CHECK-NOT: tensor.collapse_shape
func.func @expand_transpose_fold(%arg0: tensor<24x48x48xbf16>) -> tensor<24x48x48xbf16> {
  %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [24, 48, 1, 48] : tensor<24x48x48xbf16> into tensor<24x48x1x48xbf16>
  %0 = tensor.empty() : tensor<24x48x48x1xbf16>
  %transposed = linalg.transpose ins(%expanded : tensor<24x48x1x48xbf16>) outs(%0 : tensor<24x48x48x1xbf16>) permutation = [0, 1, 3, 2] 
  %collapsed = tensor.collapse_shape %transposed [[0], [1], [2, 3]] : tensor<24x48x48x1xbf16> into tensor<24x48x48xbf16>
  return %collapsed : tensor<24x48x48xbf16>
}

// -----

// CHECK-LABEL: @test_broadcast_from_dense(
// CHECK: %[[cst:.*]] = arith.constant 1 : i8
// CHECK: %[[fill:.*]] = linalg.fill ins(%[[cst]] : i8) outs({{.*}} : tensor<2x200xi8>)
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[fill]], {{.*}} : tensor<2x200xi8>, tensor<2x200xi8>)
func.func @test_broadcast_from_dense(%arg0: tensor<2x200xi8>) -> tensor<2x200xi8> {
  %cst = arith.constant dense<1> : tensor<i8>
  %0 = tensor.empty() : tensor<2x200xi8>
  %broadcasted = linalg.broadcast ins(%cst : tensor<i8>) outs(%0 : tensor<2x200xi8>) dimensions = [0, 1] 
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%broadcasted, %arg0 : tensor<2x200xi8>, tensor<2x200xi8>) outs(%0 : tensor<2x200xi8>) -> tensor<2x200xi8>
  return %1 : tensor<2x200xi8>
}

// -----

// CHECK-LABEL: func.func @refactor_redundant_reduces
// CHECK-NOT: linalg.reduce
// CHECK: arith.addf %arg[[a0:.*]], %[[v0:.*]] : tensor<4xf32>
// CHECK: arith.addf %arg[[a1:.*]], %[[v1:.*]] : tensor<4xf32>
// CHECK: %[[var0:.*]] = arith.addf %arg[[a2:.*]], %[[var0:.*]] : tensor<4xf32>
// CHECK: arith.mulf %[[a2:.*]], %[[var1:.*]] : tensor<4xf32>
// CHECK: arith.subf %arg[[a3:.*]], %[[v3:.*]] : tensor<4xf32>
// CHECK: linalg.reduce
// CHECK-NOT: linalg.reduce
// CHECK: return
func.func @refactor_redundant_reduces(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>, %arg4: tensor<4x6xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %empty_0 = tensor.empty() : tensor<4xf32>
  %empty_1 = tensor.empty() : tensor<4xf32>
  %empty_2 = tensor.empty() : tensor<4xf32>
  %empty_3 = tensor.empty() : tensor<4xf32>
  %empty_4 = tensor.empty() : tensor<4xf32>
  %reduced:4 = linalg.reduce ins(%arg0, %arg1, %arg2, %arg3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%empty_0, %empty_1, %empty_2, %empty_3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) dimensions = []  {hfusion.reduce_composed = ""}
    (%in: f32, %in_1: f32, %in_2: f32, %in_3: f32, %init: f32, %init_4: f32, %init_5: f32, %init_6: f32) {
      %0 = arith.addf %in, %init : f32
      %1 = arith.addf %in_1, %init_4 : f32
      %2 = arith.addf %in_2, %init_5 : f32
      %3 = arith.mulf %in_2, %2 : f32
      %4 = arith.subf %in_3, %init_6 : f32
      linalg.yield %0, %1, %3, %4 : f32, f32, f32, f32
    }
  %reduced_0 = linalg.reduce ins(%arg4 : tensor<4x6xf32>) outs(%empty_4 : tensor<4xf32>) dimensions = [1] 
    (%in: f32, %init: f32) {
      %0 = arith.addf %in, %init : f32
      %1 = arith.mulf %in, %0 : f32
      linalg.yield %1 : f32
    }
  return %reduced_0, %reduced#0, %reduced#1, %reduced#2, %reduced#3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}