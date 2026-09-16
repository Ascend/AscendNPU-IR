// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops --split-input-file %s | FileCheck %s
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="LAST_AXIS_PBR" -hfusion-fuse-ops='output-mode=multi' --split-input-file %s | FileCheck %s --check-prefix=LASTAXIS
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops='output-mode=multi' --split-input-file %s | FileCheck %s
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops='output-mode=single' --split-input-file %s | FileCheck %s -check-prefix=Single
// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops='output-mode=single-aggr' --split-input-file %s | FileCheck %s -check-prefix=Single-Aggr
// RUN: bishengir-opt -hfusion-outline-single-op --split-input-file %s | FileCheck %s -check-prefix=SINGLE-OUTLINE

// CHECK-LABEL: func.func @add_mul_reduce_0(
// CHECK: %[[ELEM1:.*]] = linalg.elemwise_binary
// CHECK: %[[ELEM2:.*]] = linalg.elemwise_binary
// CHECK: %[[ELEM3:.*]] = linalg.elemwise_binary
// CHECK: return {{.*}} : tensor<?xf32>, tensor<?xf32>

// Single-LABEL: func.func @add_mul_reduce(
// Single-SAME: -> (tensor<f32>, tensor<f32>)

// Single-Aggr-LABEL: func.func @add_mul_reduce_0(
// Single-Aggr: mul
// Single-Aggr: add
// Single-Aggr: return {{.*}} : tensor<?xf32>
// Single-Aggr-LABEL: func.func @add_mul_reduce_1(
// Single-Aggr: mul
// Single-Aggr: sub
// Single-Aggr: return {{.*}} : tensor<?xf32>

// SINGLE-OUTLINE-LABEL: func.func @add_mul_reduce_single_outlined_0
// SINGLE-OUTLINE: linalg.elemwise_binary
// SINGLE-OUTLINE-LABEL: func.func @add_mul_reduce_single_outlined_1
// SINGLE-OUTLINE: linalg.elemwise_binary
// SINGLE-OUTLINE-LABEL: func.func @add_mul_reduce_single_outlined_2
// SINGLE-OUTLINE: linalg.elemwise_binary
// SINGLE-OUTLINE-LABEL: func.func @add_mul_reduce_single_outlined_3
// SINGLE-OUTLINE: linalg.reduce
// SINGLE-OUTLINE-LABEL: func.func @add_mul_reduce_single_outlined_4
// SINGLE-OUTLINE: linalg.reduce
// SINGLE-OUTLINE-LABEL: func.func @add_mul_reduce(
// SINGLE-OUTLINE-NOT: linalg
// SINGLE-OUTLINE: return
func.func @add_mul_reduce(%arg0: tensor<?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> (tensor<f32>, tensor<f32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-LABEL: func.func @add_mul_reduce(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>, %[[ARG2:.*]]: tensor<?xf32>)

// LASTAXIS-LABEL: func.func @add_mul_reduce_0(
// LASTAXIS-SAME: LAST_AXIS_PBR
// LASTAXIS-LABEL: func.func @add_mul_reduce(
// LASTAXIS: %[[BUFFER1:.*]] = tensor.empty
// LASTAXIS: %[[BUFFER2:.*]] = tensor.empty
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?xf32>
  %2 = tensor.empty(%0) : tensor<?xf32>
  %3 = linalg.elemwise_binary { mul, fun = #linalg.binary_fn<mul> } ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%2 : tensor<?xf32>) -> tensor<?xf32>

  %4 = tensor.empty(%0) : tensor<?xf32>

  // Single-NOT: call
  // Single: return

  // Single-Aggr: call @add_mul_reduce_1
  // single-Aggr: call @add_mul_reduce_0
  %5 = linalg.elemwise_binary { add, fun = #linalg.binary_fn<add> } ins(%3, %arg2 : tensor<?xf32>, tensor<?xf32>) outs(%4 : tensor<?xf32>) -> tensor<?xf32>

  %6 = tensor.empty(%0) : tensor<?xf32>
  %7 = linalg.elemwise_binary { sub, fun = #linalg.binary_fn<sub> } ins(%3, %5 : tensor<?xf32>, tensor<?xf32>) outs(%6 : tensor<?xf32>) -> tensor<?xf32>

// CHECK: %[[CALL1:.*]]:2 = call @add_mul_reduce_0(
// CHECK-SAME: -> (tensor<?xf32>, tensor<?xf32>)
// LASTAXIS: %[[CALL1:.*]]:2 = call @add_mul_reduce_0(
// LASTAXIS-SAME: %[[BUFFER1]]
// LASTAXIS-SAME: %[[BUFFER2]]
// LASTAXIS-SAME: -> (tensor<f32>, tensor<f32>)
// LASTAXIS: return %[[CALL1]]#0, %[[CALL1]]#1 : tensor<f32>, tensor<f32>
  %8 = tensor.empty() : tensor<f32>
  %9 = linalg.reduce { arith.addf } ins(%5 : tensor<?xf32>) outs(%8 : tensor<f32>) dimensions = [0]

  %10 = tensor.empty() : tensor<f32>
  %11 = linalg.reduce { arith.addf } ins(%7 : tensor<?xf32>) outs(%10 : tensor<f32>) dimensions = [0]

  return %9, %11 : tensor<f32>, tensor<f32>
}

// -----
// CHECK-LABEL: func.func @model_0(
// CHECK-NOT: linalg
// CHECK: return
func.func @model_0(%arg0: tensor<5x1xf32>) -> tensor<5x1xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%1 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %cst : tensor<5x1xf32>, f32) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %3 : f32, tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %4 : tensor<5x1xf32>
}

// -----
// CHECK-LABEL: multi_end_alias_0_0(
// CHECK: expand_shape
// CHECK: return
// CHECK-LABEL: func.func @multi_end_alias_0(
// CHECK: call
// CHECK-NEXT: return
func.func @multi_end_alias_0(%arg0: tensor<1x512xbf16>, %arg1: tensor<1x512xbf16>, %arg2: tensor<512xbf16>) -> (tensor<512xbf16>, tensor<1x512xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x512xbf16> into tensor<512xbf16>
    %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x512xbf16> into tensor<512xbf16>
    %0 = tensor.empty() : tensor<512xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_0 : tensor<512xbf16>) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<512xbf16>) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %2 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%3 : tensor<512xf32>) outs(%arg2 : tensor<512xbf16>) -> tensor<512xbf16>
    %expanded = tensor.expand_shape %4 [[0, 1]] output_shape [1, 512] : tensor<512xbf16> into tensor<1x512xbf16>
    return %4, %expanded : tensor<512xbf16>, tensor<1x512xbf16>
}
