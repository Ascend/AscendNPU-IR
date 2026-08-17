// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2 2>&1 | FileCheck %s

// CHECK-NOT: AutoVectorizeV2 failed;

#map0 = affine_map<(d0, d1) -> (d0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @two_rank2_transpose(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>) -> (tensor<16x1xf32>, tensor<2x16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %106 = tensor.empty() : tensor<16x1xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<1x16xf32>) outs(%106 : tensor<16x1xf32>) permutation = [1, 0]
    %transposed_21 = linalg.transpose ins(%arg1 : tensor<1x16xf32>) outs(%106 : tensor<16x1xf32>) permutation = [1, 0]
    %108 = tensor.empty() : tensor<16x32xf16>
    %109 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%transposed_21 : tensor<16x1xf32>) outs(%108 : tensor<16x32xf16>) {
    ^bb0(%in: f32, %out: f16):
      %e = math.exp %in : f32
      %c = arith.truncf %e : f32 to f16
      linalg.yield %c : f16
    } -> tensor<16x32xf16>
    %expanded = tensor.expand_shape %109 [[0], [1, 2]] output_shape [16, 2, 16] : tensor<16x32xf16> into tensor<16x2x16xf16>
    %110 = tensor.empty() : tensor<2x16x16xf16>
    %transposed_22 = linalg.transpose ins(%expanded : tensor<16x2x16xf16>) outs(%110 : tensor<2x16x16xf16>) permutation = [1, 0, 2]
    return %transposed, %transposed_22 : tensor<16x1xf32>, tensor<2x16x16xf16>
  }
}
