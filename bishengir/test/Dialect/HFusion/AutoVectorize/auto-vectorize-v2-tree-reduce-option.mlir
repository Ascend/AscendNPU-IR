// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="emit-transform-sequence=true tree-reduce=false" | FileCheck %s --check-prefix=DISABLED
// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="emit-transform-sequence=true tree-reduce=true" | FileCheck %s --check-prefix=ENABLED

// DISABLED-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.tree_reduce_option
// DISABLED: transform.structured.tile_using_for
// DISABLED-NOT: transform.structured.tile_reduction_using_for
// DISABLED-NOT: transform.structured.split_reduction

// ENABLED-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.tree_reduce_option
// ENABLED: transform.structured.tile_reduction_using_for
// ENABLED: transform.structured.split_reduction

#input_map = affine_map<(d0, d1) -> (d0, d1)>
#output_map = affine_map<(d0, d1) -> (d1)>

module {
  func.func @tree_reduce_option(%arg0: tensor<3x8xf32>) -> tensor<8xf32>
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
      } ins(%arg0 : tensor<3x8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
