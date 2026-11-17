// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="tree-reduce=true emit-transform-sequence=true" | FileCheck %s --check-prefix=TRANSFORM
// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="tree-reduce=true" -canonicalize | FileCheck %s --check-prefix=VECTOR --implicit-check-not=linalg.generic

#identity = affine_map<(d0, d1) -> (d0, d1)>
#reduce = affine_map<(d0, d1) -> (d1)>

// Three inputs limit the first tree to four leaves, so sixteen rows require
// a rolled grouping stage. A computed input keeps this on the reshape-based
// path instead of the direct register-tree path.
// TRANSFORM-LABEL: auto_vectorize_v2.transform.grouped_computed_reduce
// TRANSFORM: %tiled_linalg_op, %[[OUTER:.*]] = transform.structured.tile_using_for
// TRANSFORM: transform.structured.split_reduction {{.*}}split_factor = 4
// TRANSFORM: _stage_0_group_fill
// TRANSFORM: %{{.*}}, %[[GROUP:.*]] = transform.structured.fuse
// TRANSFORM: transform.loop.unroll {{.*}}factor = 2
// TRANSFORM: apply_patterns to %[[GROUP]] {
// TRANSFORM-NEXT: transform.apply_patterns.hfusion.forward_extract_slice
// TRANSFORM: _finish_combine
// TRANSFORM-NEXT: apply_patterns to %[[OUTER]] {
// TRANSFORM-NEXT: transform.apply_patterns.hfusion.forward_extract_slice
// VECTOR-LABEL: func.func @grouped_computed_reduce
// VECTOR: arith.mulf {{.*}} : vector<1x1x1x64xf32>
// VECTOR: arith.addf {{.*}} : vector<1x64xf32>
// VECTOR: return
func.func @grouped_computed_reduce(%a: tensor<16x64xf32>, %b: tensor<16x64xf32>, %c: tensor<16x64xf32>, %init: tensor<64xf32>) -> tensor<64xf32>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %result = linalg.generic {
      indexing_maps = [#identity, #identity, #identity, #reduce],
      iterator_types = ["reduction", "parallel"]
    } ins(%a, %b, %c : tensor<16x64xf32>, tensor<16x64xf32>, tensor<16x64xf32>) outs(%init : tensor<64xf32>) {
  ^bb0(%x: f32, %y: f32, %z: f32, %acc: f32):
    %product = arith.mulf %x, %y : f32
    %value = arith.addf %product, %z : f32
    %sum = arith.addf %value, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<64xf32>
  return %result : tensor<64xf32>
}

// A non-power-of-two computed reduction folds its tail into eight partials
// before reducing them with a fully unrolled pairwise tree.
// TRANSFORM-LABEL: auto_vectorize_v2.transform.tail_computed_reduce
// TRANSFORM: transform.structured.tile_reduction_using_for {{.*}}tile_sizes = [8, 0]
// TRANSFORM: _tail_reduce
// TRANSFORM: transform.structured.split_reduction
// TRANSFORM: transform.loop.unroll {{.*}}factor = 2
// TRANSFORM: _finish_combine
// TRANSFORM: transform.apply_patterns.hfusion.forward_extract_slice
// VECTOR-LABEL: func.func @tail_computed_reduce
// VECTOR: arith.mulf {{.*}} : vector<1x64xf32>
// VECTOR: arith.addf {{.*}} : vector<1x64xf32>
// VECTOR: return
func.func @tail_computed_reduce(%input: tensor<15x64xf32>, %init: tensor<64xf32>) -> tensor<64xf32>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %result = linalg.generic {
      indexing_maps = [#identity, #reduce],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : tensor<15x64xf32>) outs(%init : tensor<64xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %square = arith.mulf %x, %x : f32
    %sum = arith.addf %square, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<64xf32>
  return %result : tensor<64xf32>
}
