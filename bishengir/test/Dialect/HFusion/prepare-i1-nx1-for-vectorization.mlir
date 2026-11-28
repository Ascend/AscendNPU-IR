// RUN: bishengir-opt %s -hfusion-prepare-i1-nx1-for-vectorization -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#scalar_map = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func.func @collapse_i1_nx1_generic(
// CHECK-SAME: %[[ARG0:.*]]: tensor<64xi1>
// CHECK-DAG: %[[TRUE_VAL:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG: %[[FALSE_VAL:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[COLLAPSED_OUT:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1{{\]\]}} : tensor<64x1xf32> into tensor<64xf32>
// CHECK: %[[GENERIC:.*]] = linalg.generic {indexing_maps = [#{{[A-Za-z0-9_]+}}, #{{[A-Za-z0-9_]+}}, #{{[A-Za-z0-9_]+}}, #{{[A-Za-z0-9_]+}}], iterator_types = ["parallel"]} ins(%[[ARG0]], %[[TRUE_VAL]], %[[FALSE_VAL]] : tensor<64xi1>, f32, f32) outs(%[[COLLAPSED_OUT]] : tensor<64xf32>)
// CHECK: ^bb0(%[[IN:.*]]: i1, %[[TRUE_IN:.*]]: f32, %[[FALSE_IN:.*]]: f32, %{{.*}}: f32):
// CHECK:   %[[SEL:.*]] = arith.select %[[IN]]
// CHECK:   linalg.yield %[[SEL]] : f32
// CHECK: %[[EXPANDED_RES:.*]] = tensor.expand_shape %[[GENERIC]]
// CHECK-SAME: tensor<64xf32> into tensor<64x1xf32>
// CHECK: return %[[EXPANDED_RES]] : tensor<64x1xf32>
func.func @collapse_i1_nx1_generic(%arg0: tensor<64xi1>) -> tensor<64x1xf32> {
  %true_val = arith.constant 1.000000e+00 : f32
  %false_val = arith.constant 0.000000e+00 : f32
  %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [64, 1] : tensor<64xi1> into tensor<64x1xi1>
  %empty = tensor.empty() : tensor<64x1xf32>
  %result = linalg.generic {
      indexing_maps = [#map, #scalar_map, #scalar_map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%expanded, %true_val, %false_val : tensor<64x1xi1>, f32, f32)
      outs(%empty : tensor<64x1xf32>) {
  ^bb0(%in: i1, %true_in: f32, %false_in: f32, %out: f32):
    %selected = arith.select %in, %true_in, %false_in : f32
    linalg.yield %selected : f32
  } -> tensor<64x1xf32>
  return %result : tensor<64x1xf32>
}

// -----

// The i1 mask can also be on the *result* side: a compare whose operands are
// the expanded ones, with its `Nx1` i1 result collapsed straight back. This is
// what Triton's `offs[:, None] < n` lowers to. Leaving the round trip in place
// gives the mask a 256-element UB row stride, so reading its single column
// needs a gather the hardware has no i1 variant of.

#map = affine_map<(d0, d1) -> (d0, d1)>
#scalar_map = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func.func @collapse_i1_nx1_generic_from_collapsed_result(
// CHECK-SAME: %[[ARG0:.*]]: tensor<32xi32>, %[[ARG1:.*]]: i32
// The rebuilt op is rank 1, and both reshapes of the round trip cancel.
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<32x1xi1>
// CHECK: %[[INIT_1D:.*]] = tensor.collapse_shape %[[INIT]] {{\[\[}}0, 1]] : tensor<32x1xi1> into tensor<32xi1>
// CHECK: %[[RES:.*]] = linalg.generic
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<32xi32>, i32)
// CHECK-SAME: outs(%[[INIT_1D]] : tensor<32xi1>)
// CHECK: return %[[RES]] : tensor<32xi1>
func.func @collapse_i1_nx1_generic_from_collapsed_result(%arg0: tensor<32xi32>, %bound: i32) -> tensor<32xi1> {
  %init = tensor.empty() : tensor<32x1xi1>
  %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [32, 1] : tensor<32xi32> into tensor<32x1xi32>
  %cmp = linalg.generic {indexing_maps = [#map, #scalar_map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded, %bound : tensor<32x1xi32>, i32) outs(%init : tensor<32x1xi1>) {
  ^bb0(%in: i32, %b: i32, %out: i1):
    %lt = arith.cmpi slt, %in, %b : i32
    linalg.yield %lt : i1
  } -> tensor<32x1xi1>
  %collapsed = tensor.collapse_shape %cmp [[0, 1]] : tensor<32x1xi1> into tensor<32xi1>
  return %collapsed : tensor<32xi1>
}

// -----

// Two readers of the `Nx1` result keep that shape alive regardless, so the
// rewrite would only add churn. Left alone.

#map = affine_map<(d0, d1) -> (d0, d1)>
#scalar_map = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func.func @nx1_i1_result_with_two_users_is_left_alone(
// CHECK: linalg.generic
// CHECK-SAME: outs(%{{.*}} : tensor<32x1xi1>)
// CHECK: tensor.collapse_shape
func.func @nx1_i1_result_with_two_users_is_left_alone(%arg0: tensor<32xi32>, %bound: i32) -> (tensor<32xi1>, tensor<32x1xi1>) {
  %init = tensor.empty() : tensor<32x1xi1>
  %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [32, 1] : tensor<32xi32> into tensor<32x1xi32>
  %cmp = linalg.generic {indexing_maps = [#map, #scalar_map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded, %bound : tensor<32x1xi32>, i32) outs(%init : tensor<32x1xi1>) {
  ^bb0(%in: i32, %b: i32, %out: i1):
    %lt = arith.cmpi slt, %in, %b : i32
    linalg.yield %lt : i1
  } -> tensor<32x1xi1>
  %collapsed = tensor.collapse_shape %cmp [[0, 1]] : tensor<32x1xi1> into tensor<32xi1>
  return %collapsed, %cmp : tensor<32xi1>, tensor<32x1xi1>
}
