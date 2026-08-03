// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 -split-input-file | FileCheck %s

#off = affine_map<()[s0] -> (s0 * 510)>
#sz = affine_map<()[s0] -> (-s0 + 510)>
#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @dyn_multi_consumer_merge
// CHECK: %[[INIT:.*]] = tensor.empty(%[[TILE:.*]]) : tensor<1x?xf32>
// CHECK: tensor.extract_slice %[[INIT]][%{{.*}}, 0] [1, %[[TILE]]] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
// CHECK-NOT: tensor.extract_slice %{{.*}} [1, 0] [1, 1] : tensor<1x?xf32>
func.func @dyn_multi_consumer_merge(%a: tensor<1x1019xf32>, %b: tensor<1x1019xf32>, %out1: tensor<1x1019xf32>, %out2: tensor<1x1019xf32>) -> (tensor<1x1019xf32>, tensor<1x1019xf32>)
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // Outer "blockify" loop, splitting 1019 into dynamic tiles.
  %r:2 = scf.for %i = %c0 to %c2 step %c1 iter_args(%o1 = %out1, %o2 = %out2) -> (tensor<1x1019xf32>, tensor<1x1019xf32>) {
    %off = affine.apply #off()[%i]
    %tile = affine.apply #sz()[%i]
    %as = tensor.extract_slice %a[0, %off] [1, %tile] [1, 1] : tensor<1x1019xf32> to tensor<1x?xf32>
    %bs = tensor.extract_slice %b[0, %off] [1, %tile] [1, 1] : tensor<1x1019xf32> to tensor<1x?xf32>
    %init = tensor.empty(%tile) : tensor<1x?xf32>
    // Producer with a dynamic dim, used by both consumers below.
    %p = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%as : tensor<1x?xf32>) outs(%init : tensor<1x?xf32>) {
    ^bb0(%in: f32, %o: f32):
      %e = math.exp %in : f32
      linalg.yield %e : f32
    } -> tensor<1x?xf32>
    %add = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%p, %bs : tensor<1x?xf32>, tensor<1x?xf32>) outs(%init : tensor<1x?xf32>) {
    ^bb0(%in: f32, %in1: f32, %o: f32):
      %v = arith.addf %in, %in1 : f32
      linalg.yield %v : f32
    } -> tensor<1x?xf32>
    %mul = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%p, %bs : tensor<1x?xf32>, tensor<1x?xf32>) outs(%init : tensor<1x?xf32>) {
    ^bb0(%in: f32, %in1: f32, %o: f32):
      %v = arith.mulf %in, %in1 : f32
      linalg.yield %v : f32
    } -> tensor<1x?xf32>
    %o1n = tensor.insert_slice %add into %o1[0, %off] [1, %tile] [1, 1] : tensor<1x?xf32> into tensor<1x1019xf32>
    %o2n = tensor.insert_slice %mul into %o2[0, %off] [1, %tile] [1, 1] : tensor<1x?xf32> into tensor<1x1019xf32>
    scf.yield %o1n, %o2n : tensor<1x1019xf32>, tensor<1x1019xf32>
  } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
  return %r#0, %r#1 : tensor<1x1019xf32>, tensor<1x1019xf32>
}
