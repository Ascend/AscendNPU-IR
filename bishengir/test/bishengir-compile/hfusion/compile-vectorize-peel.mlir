// RUN: bishengir-opt %s -hfusion-auto-vectorize='peel-loops=true' -cse -lower-vector-mask -canonicalize -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

module {
  func.func @test_vectorization(%arg1: tensor<65536x23333xf32>, %arg2: tensor<65536x23333xf32>) 
    -> tensor<65536x23333xf32> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} 
{
    %1 = tensor.empty() : tensor<65536x23333xf32>
    %mul = linalg.elemwise_binary { fun = #linalg.binary_fn<mul> } ins(%arg1, %arg2 : tensor<65536x23333xf32>, tensor<65536x23333xf32>) outs(%1 : tensor<65536x23333xf32>) -> tensor<65536x23333xf32>
    return %mul : tensor<65536x23333xf32>
  }
}

// Checking the first part of the loop contains no mask, so that loop peeling is useful
// After the first part of the loop, there will be masks for the tail blocks.

// CHECK: scf.for
// CHECK: scf.for
// CHECK-NOT: mask
// CHECK: }
// CHECK: mask
// CHECK: }
