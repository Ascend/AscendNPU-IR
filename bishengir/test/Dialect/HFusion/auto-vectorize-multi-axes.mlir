// RUN: bishengir-opt %s -hfusion-auto-vectorize="max-vectorize-axes=2" -lower-vector-mask -canonicalize | FileCheck %s

// check tile and outlined transpose, should only tile one axis, even though "max-vectorize-axes=2"
// CHECK-LABEL: func.func @tile_multi_axes_0_outlined_vf_0(
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[c16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[c384:.*]] = arith.constant 384 : index
// CHECK: scf.for %[[axis0:.*]] = %[[c0]] to %[[c16]] step %[[c1]]
// CHECK: scf.for %[[axis1:.*]] = %[[c0]] to %[[c384]] step %[[c64]]
// CHECK-DAG: tensor.extract_slice {{.*}}[%[[axis1]], %[[axis0]]] [64, 1]
// CHECK-DAG: tensor.extract_slice {{.*}}[%[[axis0]], %[[axis1]]] [1, 64]

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @tile_multi_axes_0(%arg0: tensor<384x16xf32>, %arg1: tensor<384x16xf32>) -> tensor<16x384xf32> 
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.353553385 : f32
    %0 = tensor.empty() : tensor<384x16xf32>
    %1 = tensor.empty() : tensor<16x384xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<384x16xf32>, tensor<384x16xf32>) outs(%0 : tensor<384x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.addf %in, %in_0 : f32
      %4 = arith.mulf %3, %cst : f32
      linalg.yield %4 : f32
    } -> tensor<384x16xf32>
    %transposed = linalg.transpose ins(%2 : tensor<384x16xf32>) outs(%1 : tensor<16x384xf32>) permutation = [1, 0] 
    return %transposed : tensor<16x384xf32>
  }
}

// -----

// make sure the input of outer_transpose is tiled to continuous memory
// CHECK-LABEL: func.func @tile_multi_axes_outer_transpose_outlined_vf_
// CHECK: vector.transfer_read {{.*}} tensor<1x8x16xf16>, vector<1x8x16xf16>
// CHECK: vector.transpose {{.*}} [1, 0, 2] : vector<1x8x16xf16> to vector<8x1x16xf16>
// CHECK: vector.transfer_write {{.*}} vector<8x1x16xf16>, tensor<8x1x16xf16>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @tile_multi_axes_outer_transpose(%arg0: tensor<64x8x16xf16>) -> tensor<8x64x16xf16>
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() : tensor<64x8x16xf16>
    %1 = tensor.empty() : tensor<64x8x16xf16>
    %2 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} 
         ins(%arg0 : tensor<64x8x16xf16>) outs(%1 : tensor<64x8x16xf16>) {
    ^bb0(%in: f16, %out: f16):
      %4 = arith.addf %in, %in : f16
      linalg.yield %4 : f16
    } -> tensor<64x8x16xf16>
    %3 = tensor.empty() : tensor<8x64x16xf16>
    %transposed = linalg.transpose ins(%2 : tensor<64x8x16xf16>) outs(%3 : tensor<8x64x16xf16>) permutation = [1, 0, 2] 
    return %transposed : tensor<8x64x16xf16>
  }
}

// -----

module {
  func.func @cast_vsstb_tiling_pattern(%arg0: tensor<64x8x16xf32>) -> tensor<8x64x16xf16> attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() : tensor<64x8x16xf16>
    %1 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<64x8x16xf32>) outs(%0 : tensor<64x8x16xf16>) {
    ^bb0(%in: f32, %out: f16):
      %3 = arith.truncf %in {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : f32 to f16
      linalg.yield %3 : f16
    } -> tensor<64x8x16xf16>
    %2 = tensor.empty() : tensor<8x64x16xf16>
    %transposed = linalg.transpose ins(%1 : tensor<64x8x16xf16>) outs(%2 : tensor<8x64x16xf16>) permutation = [1, 0, 2] 
    return %transposed : tensor<8x64x16xf16>
  }
}