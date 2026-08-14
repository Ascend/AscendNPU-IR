// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="emit-transform-sequence=true tree-reduce=true" | FileCheck %s

// A zero indexing expression broadcasts an operand dimension.  The current
// reshape-based split-reduction transform cannot split such maps, so keep this
// reduction on the established tile_using_for path.

// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.tree_reduce_zero_broadcast_fallback
// CHECK: transform.structured.tile_using_for
// CHECK-NOT: transform.structured.tile_reduction_using_for
// CHECK-NOT: transform.structured.split_reduction

// The tree size comes from the iteration domain, so projected permutations are
// supported even when input dimension 0 is not reduction dimension 0.

// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.tree_reduce_permuted_input
// CHECK: transform.structured.split_reduction

// A dynamic reduction extent cannot define the compile-time pairwise tree.

// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.tree_reduce_dynamic_extent_fallback
// CHECK: transform.structured.tile_using_for
// CHECK-NOT: transform.structured.split_reduction

#input_map = affine_map<(d0, d1) -> (d0, 0)>
#output_map = affine_map<(d0, d1) -> (d1)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
#permuted = affine_map<(d0, d1) -> (d1, d0)>

module {
  func.func @tree_reduce_zero_broadcast_fallback(
      %arg0: tensor<32x1xf32>) -> tensor<8xf32>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>
      } {
    %c0 = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<8xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty : tensor<8xf32>)
        -> tensor<8xf32>
    %result = linalg.generic {
        indexing_maps = [#input_map, #output_map],
        iterator_types = ["reduction", "parallel"]
      } ins(%arg0 : tensor<32x1xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }

  func.func @tree_reduce_permuted_input(
      %arg0: tensor<8x32xf32>) -> tensor<8xf32>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>
      } {
    %c0 = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<8xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty : tensor<8xf32>)
        -> tensor<8xf32>
    %result = linalg.generic {
        indexing_maps = [#permuted, #output_map],
        iterator_types = ["reduction", "parallel"]
      } ins(%arg0 : tensor<8x32xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }

  func.func @tree_reduce_dynamic_extent_fallback(
      %arg0: tensor<?x8xf32>) -> tensor<8xf32>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>
      } {
    %c0 = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<8xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty : tensor<8xf32>)
        -> tensor<8xf32>
    %result = linalg.generic {
        indexing_maps = [#identity, #output_map],
        iterator_types = ["reduction", "parallel"]
      } ins(%arg0 : tensor<?x8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
