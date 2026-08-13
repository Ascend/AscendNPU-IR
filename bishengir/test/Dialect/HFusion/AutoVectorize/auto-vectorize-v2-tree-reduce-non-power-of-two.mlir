// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="emit-transform-sequence=true tree-reduce=true" | FileCheck %s

// A non-power-of-two reduction is padded with the neutral element to the next
// power of two.  This keeps the generated floating-point reduction tree
// balanced instead of merging a separately reduced tail.

// CHECK-LABEL: func.func @tree_reduce_3
// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.tree_reduce_3
// CHECK: transform.structured.tile_reduction_using_for {{.*}} tile_sizes = [4, 0]
// CHECK: transform.structured.split_reduction {{.*}} {inner_parallel, split_factor = 2 : i64}

#input_map = affine_map<(d0, d1) -> (d0, d1)>
#output_map = affine_map<(d0, d1) -> (d1)>

module {
  func.func @tree_reduce_3(%arg0: tensor<3x8xf32>) -> tensor<8xf32>
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

// -----

// Keep reductions up to 64 elements as one global pairwise tree.  In
// particular, the first stage of a 64-element reduction must combine the two
// 32-element halves instead of independently reducing four 16-element blocks.

// CHECK-LABEL: func.func @tree_reduce_64
// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.tree_reduce_64
// CHECK-NOT: split_factor = 4 : i64}
// CHECK: transform.structured.split_reduction {{.*}} {inner_parallel, split_factor = 32 : i64}
// CHECK: transform.structured.split_reduction {{.*}} {inner_parallel, split_factor = 16 : i64}

module {
  func.func @tree_reduce_64(%arg0: tensor<64x8xf32>) -> tensor<8xf32>
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
      } ins(%arg0 : tensor<64x8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
