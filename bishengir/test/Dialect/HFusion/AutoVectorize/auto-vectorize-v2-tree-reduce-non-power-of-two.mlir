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

// Sixteen elements is the largest reduction kept on the new pairwise tree.

// CHECK-LABEL: func.func @tree_reduce_16
// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.tree_reduce_16
// CHECK: transform.structured.split_reduction {{.*}} {inner_parallel, split_factor = 8 : i64}
// CHECK: transform.structured.split_reduction {{.*}} {inner_parallel, split_factor = 4 : i64}

module {
  func.func @tree_reduce_16(%arg0: tensor<16x8xf32>) -> tensor<8xf32>
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
      } ins(%arg0 : tensor<16x8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}

// -----

// Seventeen elements is the first reduction size routed to regular tiling.

// CHECK-LABEL: func.func @tree_reduce_17
// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.tree_reduce_17
// CHECK: transform.structured.tile_using_for
// CHECK-NOT: transform.structured.tile_reduction_using_for
// CHECK-NOT: transform.structured.split_reduction

module {
  func.func @tree_reduce_17(%arg0: tensor<17x8xf32>) -> tensor<8xf32>
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
      } ins(%arg0 : tensor<17x8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}

// -----

// Power-of-two reductions above the cutoff use the same regular path.

// CHECK-LABEL: func.func @tree_reduce_64
// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.tree_reduce_64
// CHECK: transform.structured.tile_using_for
// CHECK-NOT: transform.structured.tile_reduction_using_for
// CHECK-NOT: transform.structured.split_reduction

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

// -----

// Larger reductions also use regular tiling.  The radix hierarchy in the new
// tree materializes every pairwise level and is not profitable.

// CHECK-LABEL: func.func @tree_reduce_65
// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.tree_reduce_65
// CHECK: transform.structured.tile_using_for
// CHECK-NOT: transform.structured.tile_reduction_using_for
// CHECK-NOT: transform.structured.split_reduction

module {
  func.func @tree_reduce_65(%arg0: tensor<65x8xf32>) -> tensor<8xf32>
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
      } ins(%arg0 : tensor<65x8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func @tree_reduce_128
// CHECK-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.tree_reduce_128
// CHECK: transform.structured.tile_using_for
// CHECK-NOT: transform.structured.split_reduction

module {
  func.func @tree_reduce_128(%arg0: tensor<128x8xf32>) -> tensor<8xf32>
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
      } ins(%arg0 : tensor<128x8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
