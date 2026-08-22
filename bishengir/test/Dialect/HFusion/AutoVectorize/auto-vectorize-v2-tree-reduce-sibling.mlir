// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="tree-reduce=true" -outline-vector-function -hfusion-auto-vectorize-verifier 2>&1 | FileCheck %s
// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="emit-transform-sequence=true tree-reduce=true" | FileCheck %s --check-prefix=TRANSFORM

// A direct register tree gets a dedicated containing loop.  An independent
// elementwise sibling must remain outside it because the post-vectorization
// rewrite replaces that whole loop.

// CHECK-NOT: operations cannot be fused
// CHECK-NOT: AutoVectorizeV2 failed
// CHECK-NOT: unexpected vector operation outside vector function
// CHECK: func.func @tree_reduce_with_elementwise_sibling_outlined_vf_

// The two reductions expose opposite fixed parallel loop nests. They must be
// outlined separately even though their full iteration shapes match.
// CHECK-COUNT-2: func.func @opposite_reduction_siblings_outlined_vf_

// Multiple direct-tree candidates in one vector function stay on the compact
// regular lowering so horizontal fusion is preserved.  Both reductions share
// one outer loop and one parallel-tile loop with two iter_args/results.
// CHECK-LABEL: func.func @parallel_tree_reduction_siblings_outlined_vf_1
// CHECK: %[[PAR_C1:[0-9A-Za-z_]+]] = arith.constant 1 : index
// CHECK: %[[PAR_C16:[0-9A-Za-z_]+]] = arith.constant 16 : index
// CHECK: %[[PAR_C64:[0-9A-Za-z_]+]] = arith.constant 64 : index
// CHECK: %[[PAR_OUTER:[0-9A-Za-z_]+]]:2 = scf.for {{.*}} to %[[PAR_C16]] step %[[PAR_C1]] iter_args({{.*}}) -> (tensor<64xf32>, tensor<64xf32>) {
// CHECK: %{{[0-9A-Za-z_]+}}:2 = scf.for {{.*}} to %[[PAR_C64]] step %[[PAR_C64]] iter_args({{.*}}) -> (tensor<64xf32>, tensor<64xf32>) {
// CHECK: return %[[PAR_OUTER]]#0, %[[PAR_OUTER]]#1 : tensor<64xf32>, tensor<64xf32>

// The same invariant applies when a node acquires one tree through producer
// fusion and then considers a second tree producer for the same consumer.
// CHECK-LABEL: func.func @producer_tree_reduction_sibling_outlined_vf_1
// CHECK: %[[PROD_C64:[0-9A-Za-z_]+]] = arith.constant 64 : index
// CHECK: %[[PROD_C1:[0-9A-Za-z_]+]] = arith.constant 1 : index
// CHECK: scf.for {{.*}} to %[[PROD_C64]] step %[[PROD_C1]] iter_args({{.*}}) -> (tensor<64x64xf32>) {
// CHECK: %[[PROD_CONSUMER:[0-9A-Za-z_]+]] = scf.for {{.*}} to %[[PROD_C64]] step %[[PROD_C1]] iter_args({{.*}}) -> (tensor<64x64xf32>) {
// CHECK: return %[[PROD_CONSUMER]] : tensor<64x64xf32>

// A direct add(input, accumulator) is selected for the register tree.
// TRANSFORM-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.tree_reduce_with_elementwise_sibling
// TRANSFORM: annotate {{.*}} "hfusion.register_tree_reduction"
// TRANSFORM-NOT: transform.structured.split_reduction

// A value computed inside the reduction body is supported: the extent comes
// from the iteration domain rather than from a particular input operand.
// TRANSFORM-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.computed_reduction_value
// TRANSFORM: transform.structured.split_reduction

#identity = affine_map<(d0, d1) -> (d0, d1)>
#identity_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#project_middle = affine_map<(d0, d1, d2) -> (d0, d2)>
#project_first = affine_map<(d0, d1) -> (d1)>
#project_second = affine_map<(d0, d1) -> (d0)>

module {
  func.func @tree_reduce_with_elementwise_sibling(
      %arg0: tensor<16x64xf32>, %arg1: tensor<16x64xf32>)
      -> (tensor<16x64xf32>, tensor<64xf32>)
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>
      } {
    %zero = arith.constant 0.0 : f32
    %matrix_empty = tensor.empty() : tensor<16x64xf32>
    %vector_empty = tensor.empty() : tensor<64xf32>
    %init = linalg.fill ins(%zero : f32) outs(%vector_empty : tensor<64xf32>)
        -> tensor<64xf32>
    %elementwise = linalg.generic {
        indexing_maps = [#identity, #identity],
        iterator_types = ["parallel", "parallel"]
      } ins(%arg0 : tensor<16x64xf32>)
        outs(%matrix_empty : tensor<16x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<16x64xf32>
    %reduced = linalg.generic {
        indexing_maps = [#identity, #project_first],
        iterator_types = ["reduction", "parallel"]
      } ins(%arg1 : tensor<16x64xf32>) outs(%init : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<64xf32>
    return %elementwise, %reduced : tensor<16x64xf32>, tensor<64xf32>
  }

  func.func @computed_reduction_value(%arg0: tensor<16x64xf32>)
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

  func.func @opposite_reduction_siblings(
      %arg0: tensor<16x64xf32>, %arg1: tensor<16x64xf32>)
      -> (tensor<64xf32>, tensor<16xf32>)
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>
      } {
    %zero = arith.constant 0.0 : f32
    %ra_empty = tensor.empty() : tensor<64xf32>
    %ar_empty = tensor.empty() : tensor<16xf32>
    %ra_init = linalg.fill ins(%zero : f32)
        outs(%ra_empty : tensor<64xf32>) -> tensor<64xf32>
    %ar_init = linalg.fill ins(%zero : f32)
        outs(%ar_empty : tensor<16xf32>) -> tensor<16xf32>
    %ra = linalg.generic {
        indexing_maps = [#identity, #project_first],
        iterator_types = ["reduction", "parallel"]
      } ins(%arg0 : tensor<16x64xf32>) outs(%ra_init : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<64xf32>
    %ar = linalg.generic {
        indexing_maps = [#identity, #project_second],
        iterator_types = ["parallel", "reduction"]
      } ins(%arg1 : tensor<16x64xf32>) outs(%ar_init : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<16xf32>
    return %ra, %ar : tensor<64xf32>, tensor<16xf32>
  }

  func.func @parallel_tree_reduction_siblings(
      %arg0: tensor<16x64xf32>, %arg1: tensor<16x64xf32>)
      -> (tensor<64xf32>, tensor<64xf32>)
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>
      } {
    %zero = arith.constant 0.0 : f32
    %empty0 = tensor.empty() : tensor<64xf32>
    %empty1 = tensor.empty() : tensor<64xf32>
    %init0 = linalg.fill ins(%zero : f32)
        outs(%empty0 : tensor<64xf32>) -> tensor<64xf32>
    %init1 = linalg.fill ins(%zero : f32)
        outs(%empty1 : tensor<64xf32>) -> tensor<64xf32>
    %sum0 = linalg.generic {
        indexing_maps = [#identity, #project_first],
        iterator_types = ["reduction", "parallel"]
      } ins(%arg0 : tensor<16x64xf32>) outs(%init0 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<64xf32>
    %sum1 = linalg.generic {
        indexing_maps = [#identity, #project_first],
        iterator_types = ["reduction", "parallel"]
      } ins(%arg1 : tensor<16x64xf32>) outs(%init1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<64xf32>
    return %sum0, %sum1 : tensor<64xf32>, tensor<64xf32>
  }

  func.func @producer_tree_reduction_sibling(
      %arg0: tensor<64x16x64xf32>, %arg1: tensor<64x16x64xf32>)
      -> tensor<64x64xf32>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>
      } {
    %zero = arith.constant 0.0 : f32
    %empty0 = tensor.empty() : tensor<64x64xf32>
    %empty1 = tensor.empty() : tensor<64x64xf32>
    %consumer_empty = tensor.empty() : tensor<64x64xf32>
    %init0 = linalg.fill ins(%zero : f32)
        outs(%empty0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %init1 = linalg.fill ins(%zero : f32)
        outs(%empty1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %sum0 = linalg.generic {
        indexing_maps = [#identity_3d, #project_middle],
        iterator_types = ["parallel", "reduction", "parallel"]
      } ins(%arg0 : tensor<64x16x64xf32>)
        outs(%init0 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<64x64xf32>
    %sum1 = linalg.generic {
        indexing_maps = [#identity_3d, #project_middle],
        iterator_types = ["parallel", "reduction", "parallel"]
      } ins(%arg1 : tensor<64x16x64xf32>)
        outs(%init1 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<64x64xf32>
    %consumer = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel", "parallel"]
      } ins(%sum0, %sum1 : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%consumer_empty : tensor<64x64xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<64x64xf32>
    return %consumer : tensor<64x64xf32>
  }
}
