// RUN: bishengir-opt --split-input-file -hfusion-auto-vectorize-v2 %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @fill_reduce(%arg0: tensor<256x64x128xf32>) -> tensor<256x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<256x64xf32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<256x64xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<256x64xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<256x64x128xf32>) outs(%1 : tensor<256x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %out, %in : f32
      linalg.yield %3 : f32
    } -> tensor<256x64xf32>
    return %2 : tensor<256x64xf32>
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @elementwise_reduce(%arg0: tensor<256x64x128xf32>, %input: tensor<256x64xf32>) -> tensor<256x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<256x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%input, %input : tensor<256x64xf32>, tensor<256x64xf32>) outs(%0 : tensor<256x64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %3 = arith.addf %in, %in_2 : f32
      linalg.yield %3 : f32
    } -> tensor<256x64xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<256x64x128xf32>) outs(%1 : tensor<256x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %out, %in : f32
      linalg.yield %3 : f32
    } -> tensor<256x64xf32>
    return %2 : tensor<256x64xf32>
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
  func.func @elementwise_elementwise(%arg0: tensor<256xf32>, %input: tensor<256xf32>) -> tensor<256xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%input, %input : tensor<256xf32>, tensor<256xf32>) outs(%0 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %3 = arith.addf %in, %in_2 : f32
      linalg.yield %3 : f32
    } -> tensor<256xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<256xf32>) outs(%1 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %out, %in : f32
      linalg.yield %3 : f32
    } -> tensor<256xf32>
    return %2 : tensor<256xf32>
  }
}
