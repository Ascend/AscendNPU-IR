// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion="enable-new-tree-reduce-policy=true" | FileCheck %s

// Verify that a producer containing linalg.index is not fused into the
// reduction when the new tree-reduction policy is enabled.
// AutoVectorizeV2 does not support tree-reduce when the fused
// generic contains an linalg.index operation.
// CHECK-LABEL: func.func @keep_index_producer_separate
// CHECK: linalg.generic
// CHECK: linalg.index
// CHECK: linalg.generic
// CHECK-NOT: linalg.index
// CHECK: return
func.func @keep_index_producer_separate(%input: tensor<128x8xf32>,
                                        %init: tensor<8xf32>) -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<128x8xf32>
  %producer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%input : tensor<128x8xf32>) outs(%empty : tensor<128x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %i = linalg.index 0 : index
      %i32 = arith.index_cast %i : index to i32
      %if = arith.sitofp %i32 : i32 to f32
      %sum = arith.addf %in, %if : f32
      linalg.yield %sum : f32
  } -> tensor<128x8xf32>
  %reduced = linalg.reduce ins(%producer : tensor<128x8xf32>)
      outs(%init : tensor<8xf32>) dimensions = [0]
      (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    linalg.yield %sum : f32
  }
  return %reduced : tensor<8xf32>
}
