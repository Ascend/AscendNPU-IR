// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="emit-transform-sequence=true tree-reduce=true" --split-input-file | FileCheck %s

// A module whose cost decision selected the compact regular route must not
// re-enter the materialized tree after producer fusion turns a canonical
// reduction into a computed payload.
// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.regular_scope_computed
// CHECK: transform.structured.tile_using_for
// CHECK-NOT: transform.structured.tile_reduction_using_for
// CHECK-NOT: transform.structured.split_reduction

#identity = affine_map<(d0, d1) -> (d0, d1)>
#project_first = affine_map<(d0, d1) -> (d1)>

module attributes {hfusion.regular_tree_reduction_scope} {
  func.func @regular_scope_computed(%arg0: tensor<16x64xf32>)
      -> tensor<64xf32>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>
      } {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<64xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64xf32>)
        -> tensor<64xf32>
    %reduced = linalg.generic {
        indexing_maps = [#identity, #project_first],
        iterator_types = ["reduction", "parallel"]
      } ins(%arg0 : tensor<16x64xf32>) outs(%init : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %square = arith.mulf %in, %in : f32
      %sum = arith.addf %square, %out : f32
      linalg.yield %sum : f32
    } -> tensor<64xf32>
    return %reduced : tensor<64xf32>
  }
}

// -----

// Canonicalization may create a structurally eligible reduction after
// VFFusion froze its scope decision. An unmarked late op must stay on the
// regular path and, crucially, must not rescan concurrently rewritten sibling
// functions to recompute the cost.
// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.frozen_unmarked
// CHECK: transform.structured.tile_using_for
// CHECK-NOT: transform.structured.tile_reduction_using_for
// CHECK-NOT: transform.structured.split_reduction

module attributes {hfusion.tree_reduction_selection_frozen} {
  func.func @frozen_unmarked(%arg0: tensor<32x32xf32>)
      -> tensor<32xf32>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>
      } {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32xf32>)
        -> tensor<32xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<32x32xf32>)
        outs(%init : tensor<32xf32>) dimensions = [0]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
    return %reduced : tensor<32xf32>
  }
}
