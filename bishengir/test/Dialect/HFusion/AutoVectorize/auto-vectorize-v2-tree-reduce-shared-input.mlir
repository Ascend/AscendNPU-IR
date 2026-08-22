// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="emit-transform-sequence=true tree-reduce=true" | FileCheck %s

// The reductions share a source but expose opposite fixed parallel loop nests.
// A function containing both RA and AR reductions stays in the established
// mixed-reduction scope: dim-0 uses ordinary tiling and dim-1 keeps its normal
// tile_reduction_using_for path.

// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.shared_reduction_input
// CHECK: transform.structured.tile_using_for
// CHECK: transform.structured.tile_reduction_using_for
// CHECK-NOT: "hfusion.register_tree_reduction"

// A standalone canonical dim-0 sum uses the direct register tree.

// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.standalone_reduction_input
// CHECK: annotate {{.*}} "hfusion.register_tree_reduction"

#identity = affine_map<(d0, d1) -> (d0, d1)>
#dim0 = affine_map<(d0, d1) -> (d1)>
#dim1 = affine_map<(d0, d1) -> (d0)>

module {
  func.func @shared_reduction_input(%arg0: tensor<16x16xf32>)
      -> (tensor<16xf32>, tensor<16xf32>)
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>
      } {
    %c0 = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<16xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty : tensor<16xf32>)
        -> tensor<16xf32>
    %dim0_sum = linalg.generic {
        indexing_maps = [#identity, #dim0],
        iterator_types = ["reduction", "parallel"]
      } ins(%arg0 : tensor<16x16xf32>) outs(%init : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<16xf32>
    %dim1_sum = linalg.generic {
        indexing_maps = [#identity, #dim1],
        iterator_types = ["parallel", "reduction"]
      } ins(%arg0 : tensor<16x16xf32>) outs(%init : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<16xf32>
    return %dim0_sum, %dim1_sum : tensor<16xf32>, tensor<16xf32>
  }

  func.func @standalone_reduction_input(%arg0: tensor<16x16xf32>)
      -> tensor<16xf32>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>
      } {
    %c0 = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<16xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty : tensor<16xf32>)
        -> tensor<16xf32>
    %sum = linalg.generic {
        indexing_maps = [#identity, #dim0],
        iterator_types = ["reduction", "parallel"]
      } ins(%arg0 : tensor<16x16xf32>) outs(%init : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %next = arith.addf %in, %out : f32
      linalg.yield %next : f32
    } -> tensor<16xf32>
    return %sum : tensor<16xf32>
  }
}
