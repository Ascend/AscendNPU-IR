// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="emit-transform-sequence=true tree-reduce=true" | FileCheck %s --check-prefix=TRANSFORM
// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="tree-reduce=true" -outline-vector-function -hfusion-auto-vectorize-verifier 2>&1 | FileCheck %s --check-prefix=VERIFY

// The first input is projected and has extent 64 at dimension 0, while the
// reduction loop extent is 16.  The tree size must come from the iteration
// domain.  This also covers the multi-input computed payload used by dot-like
// reductions: (%lhs * %rhs) + %accumulator.

// TRANSFORM-LABEL: transform.sequence {{.*}}auto_vectorize_v2.transform.multi_input_projected_tree_reduction
// TRANSFORM: transform.structured.split_reduction

// VERIFY-NOT: failed to apply
// VERIFY-NOT: unexpected vector operation outside vector function
// VERIFY: func.func @multi_input_projected_tree_reduction_outlined_vf_

#projected = affine_map<(d0, d1) -> (d1)>
#identity = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @multi_input_projected_tree_reduction(
      %lhs: tensor<64xf32>, %rhs: tensor<16x64xf32>) -> tensor<64xf32>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>
      } {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<64xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64xf32>)
        -> tensor<64xf32>
    %result = linalg.generic {
        indexing_maps = [#projected, #identity, #projected],
        iterator_types = ["reduction", "parallel"]
      } ins(%lhs, %rhs : tensor<64xf32>, tensor<16x64xf32>)
        outs(%init : tensor<64xf32>) {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %product = arith.mulf %x, %y : f32
      %sum = arith.addf %product, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<64xf32>
    return %result : tensor<64xf32>
  }
}
