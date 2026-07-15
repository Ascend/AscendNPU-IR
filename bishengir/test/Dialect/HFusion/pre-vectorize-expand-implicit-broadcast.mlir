// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion -mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: func @expand_rank1_implicit_broadcast
// CHECK-NOT:     tensor.expand_shape
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
// CHECK-SAME:      ins(%arg0 : tensor<2xi32>)
func.func @expand_rank1_implicit_broadcast(%arg0: tensor<2xi32>, %out: tensor<2x3x1xi32>) -> tensor<2x3x1xi32> {
  %ex = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [2, 1, 1] : tensor<2xi32> into tensor<2x1x1xi32>
  %g = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                       iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%ex : tensor<2x1x1xi32>) outs(%out : tensor<2x3x1xi32>) {
  ^bb0(%in: i32, %o: i32):
    linalg.yield %in : i32
  } -> tensor<2x3x1xi32>
  return %g : tensor<2x3x1xi32>
}

// CHECK-LABEL: func @expand_all_unit_group_to_zero
// CHECK-NOT:     tensor.expand_shape
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
// CHECK-SAME:      ins(%arg0 : tensor<2x1xi32>)
func.func @expand_all_unit_group_to_zero(%arg0: tensor<2x1xi32>, %out: tensor<2x3x1x1xi32>) -> tensor<2x3x1x1xi32> {
  %ex = tensor.expand_shape %arg0 [[0], [1, 2, 3]] output_shape [2, 1, 1, 1] : tensor<2x1xi32> into tensor<2x1x1x1xi32>
  %g = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
       ins(%ex : tensor<2x1x1x1xi32>) outs(%out : tensor<2x3x1x1xi32>) {
  ^bb0(%in: i32, %o: i32):
    linalg.yield %in : i32
  } -> tensor<2x3x1x1xi32>
  return %g : tensor<2x3x1x1xi32>
}

// CHECK-LABEL: func @genuine_reshape_not_folded
// CHECK:         tensor.expand_shape
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%{{.*}} : tensor<2x1x3xi32>)
func.func @genuine_reshape_not_folded(%arg0: tensor<6xi32>, %out: tensor<2x4x3xi32>) -> tensor<2x4x3xi32> {
  %ex = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [2, 1, 3] : tensor<6xi32> into tensor<2x1x3xi32>
  %g = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                       iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%ex : tensor<2x1x3xi32>) outs(%out : tensor<2x4x3xi32>) {
  ^bb0(%in: i32, %o: i32):
    linalg.yield %in : i32
  } -> tensor<2x4x3xi32>
  return %g : tensor<2x4x3xi32>
}

// CHECK-LABEL: func @const0_on_non_unit_dim_not_folded
// CHECK:         tensor.expand_shape
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%{{.*}} : tensor<1x4xi32>)
func.func @const0_on_non_unit_dim_not_folded(%arg0: tensor<4xi32>, %out: tensor<1x4xi32>) -> tensor<1x4xi32> {
  %ex = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, 4] : tensor<4xi32> into tensor<1x4xi32>
  %g = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>,
                                        affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
       ins(%ex : tensor<1x4xi32>) outs(%out : tensor<1x4xi32>) {
  ^bb0(%in: i32, %o: i32):
    linalg.yield %in : i32
  } -> tensor<1x4xi32>
  return %g : tensor<1x4xi32>
}
