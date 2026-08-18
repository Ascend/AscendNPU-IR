// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true" -remove-redundant-write-and-read-pair --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @closest_user_non_fusable
// CHECK:         %[[fused:.*]]:2 = scf.for
// CHECK:           %[[add:.*]] = arith.addf
// CHECK:           %[[mul:.*]] = arith.mulf %[[add]]
// CHECK:         } {"outlined-loop-target-1"}
// CHECK:         %[[collapsed:.*]] = tensor.collapse_shape %[[fused]]#1
// CHECK:         return %[[fused]]#0, %[[collapsed]]

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @closest_user_non_fusable(%arg0: tensor<64x128xf16>, %arg1: tensor<64x128xf16>) -> (tensor<64x128xf16>, tensor<8192xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>, mix_mode = "aiv", parallel_mode = "simd"} {
    %empty = tensor.empty() : tensor<64x128xf16>
    %producer = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<64x128xf16>, tensor<64x128xf16>) outs(%empty : tensor<64x128xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.addf %in, %in_0 : f16
      linalg.yield %0 : f16
    } -> tensor<64x128xf16>
    %collapsed = tensor.collapse_shape %producer [[0, 1]] : tensor<64x128xf16> into tensor<8192xf16>
    %leaf = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%producer, %arg1 : tensor<64x128xf16>, tensor<64x128xf16>) outs(%empty : tensor<64x128xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %0 = arith.mulf %in, %in_0 : f16
      linalg.yield %0 : f16
    } -> tensor<64x128xf16>
    return %leaf, %collapsed : tensor<64x128xf16>, tensor<8192xf16>
  }
}

// -----


// CHECK-LABEL: func.func @closest_user_non_fusable
// CHECK:         %[[fused:.*]]:2 = scf.for
// CHECK:           arith.addf
// CHECK:           arith.mulf
// CHECK:         } {"outlined-loop-target-
//
// CHECK:         %[[reduce:.*]] = scf.for
// CHECK:         } {"outlined-loop-target-
//
// CHECK:         %[[transpose:.*]] = scf.for
// CHECK-DAG:       tensor.extract_slice %[[fused]]#1
// CHECK-DAG:       tensor.extract_slice %[[reduce]]
// CHECK:           arith.subf
// CHECK:           tensor.expand_shape
// CHECK:         } {"outlined-loop-target-
//
// CHECK:         return %[[fused]]#0, %[[transpose]]

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @closest_user_non_fusable(%arg0: tensor<64x128xf16>, %arg1: tensor<64x128xf16>, %arg2: tensor<64xf16>) -> (tensor<64x128x16xf16>, tensor<8x64x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>, mix_mode = "aiv", parallel_mode = "simd"} {
    %empty = tensor.empty() : tensor<64x128xf16>
    %empty_reduce = tensor.empty() : tensor<64xf16>
    %empty_transposed = tensor.empty() : tensor<8x64x16xf16>
    %empty_brc_to_3 = tensor.empty() : tensor<64x128x16xf16>

    %producer = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<64x128xf16>, tensor<64x128xf16>) outs(%empty : tensor<64x128xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %1 = arith.addf %in, %in_1 : f16
      linalg.yield %1 : f16
    } -> tensor<64x128xf16>

    // user 1 - non fusable
    %expanded = tensor.expand_shape %producer [[0], [1, 2]] output_shape [64, 16, 8] : tensor<64x128xf16> into tensor<64x8x16xf16>
    %reduce = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction", "reduction"]} ins(%expanded : tensor<64x8x16xf16>) outs(%empty_reduce : tensor<64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %1 = arith.addf %in, %out : f16
      linalg.yield %1 : f16
    } -> tensor<64xf16>

    // user 2 - conflict
    %sub = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%producer, %reduce : tensor<64x128xf16>, tensor<64xf16>) outs(%empty : tensor<64x128xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %2 = arith.subf %in, %in_1 : f16
      linalg.yield %2 : f16
    } -> tensor<64x128xf16>
    %vsstb_expanded = tensor.expand_shape %sub [[0], [1, 2]] output_shape [64, 8, 16] : tensor<64x128xf16> into tensor<64x8x16xf16>
    %transposed = linalg.transpose ins(%vsstb_expanded : tensor<64x8x16xf16>) outs(%empty_transposed : tensor<8x64x16xf16>) permutation = [1, 0, 2]
    
    // user 3 - ok
    %mul = linalg.generic {indexing_maps = [#map4, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%producer, %arg2 : tensor<64x128xf16>, tensor<64xf16>) outs(%empty_brc_to_3 : tensor<64x128x16xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %4 = arith.mulf %in, %in_1 : f16
      linalg.yield %4 : f16
    } -> tensor<64x128x16xf16>
    return %mul, %transposed : tensor<64x128x16xf16>, tensor<8x64x16xf16>
  }
}
