// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion -split-input-file | FileCheck %s

// CHECK-DAG: #map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @test_pre_vectorize_gather_1d
// CHECK-SAME: (%[[ARG0:.*]]: tensor<320xf16>, %[[ARG1:.*]]: tensor<1145xi32>) -> tensor<1145xf16> {
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1145xf16>
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
// CHECK-SAME: ins(%[[ARG1]] : tensor<1145xi32>) outs(%[[EMPTY]] : tensor<1145xf16>) {
// CHECK: ^bb0(%[[IN:.*]]: i32, %[[OUT:.*]]: f16):
// CHECK: %[[IDX:.*]] = arith.index_castui %[[IN]] : i32 to index
// CHECK: %[[E:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[IDX]]{{\]}} : tensor<320xf16>
// CHECK: linalg.yield %[[E]] : f16
// CHECK: } -> tensor<1145xf16>
// CHECK: return %[[GENERIC]] : tensor<1145xf16>
// CHECK: }

func.func @test_pre_vectorize_gather_1d(%src:tensor<320xf16>, %idx:tensor<1145xi32>) -> tensor<1145xf16>{
  %init = tensor.empty() : tensor<1145xf16>
  %res = hfusion.gather ins(%src, %idx : tensor<320xf16>, tensor<1145xi32>) outs(%init:tensor<1145xf16>) axis = 0 -> tensor<1145xf16>
  return %res : tensor<1145xf16>
}

// CHECK-LABEL: func.func @test_pre_vectorize_gather_2d
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x16xf16>, %[[ARG1:.*]]: tensor<16x4xi32>) -> tensor<16x4xf16> {
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16x4xf16>
// CHECK: %[[RES:.*]] = hfusion.gather
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<16x16xf16>, tensor<16x4xi32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<16x4xf16>)
// CHECK-SAME: axis = 1 -> tensor<16x4xf16>
// CHECK: return %[[RES]] : tensor<16x4xf16>
// CHECK: }

// -----
func.func @test_pre_vectorize_gather_2d(%src:tensor<16x16xf16>, %idx:tensor<16x4xi32>) -> tensor<16x4xf16>{
  %init = tensor.empty() : tensor<16x4xf16>
  %res = hfusion.gather ins(%src, %idx : tensor<16x16xf16>, tensor<16x4xi32>) outs(%init:tensor<16x4xf16>) axis = 1 -> tensor<16x4xf16>
  return %res : tensor<16x4xf16>
}

