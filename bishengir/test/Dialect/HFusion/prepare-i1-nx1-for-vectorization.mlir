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
