// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true multi-dynamic-shape=false" | FileCheck %s

// Dropped dimensions must not hide a source collapse boundary between retained
// dimensions. Otherwise the result is collapsed to tensor<4xi32> while the
// rebuilt slice still has sizes [2, 2].
// CHECK-LABEL: func.func @rank_reducing_slice_boundary
// CHECK-SAME: %[[ARG:.*]]: tensor<2x1x2x2xi32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ARG]][0, 0, 0, 0] [2, 1, 1, 2] [1, 1, 1, 1]
// CHECK-SAME: to tensor<2x2xi32>
// CHECK: return %[[SLICE]] : tensor<2x2xi32>
func.func @rank_reducing_slice_boundary(%arg: tensor<2x1x2x2xi32>) -> tensor<2x2xi32> {
  %slice = tensor.extract_slice %arg[0, 0, 0, 0] [2, 1, 1, 2] [1, 1, 1, 1] : tensor<2x1x2x2xi32> to tensor<2x2xi32>
  return %slice : tensor<2x2xi32>
}
