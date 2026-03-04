// RUN: bishengir-opt %s -hfusion-auto-schedule -cse -canonicalize -hfusion-pre-vectorization-fusion -hfusion-auto-vectorize -cse -canonicalize -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

module {
  func.func @test_vectorization(%arg1: tensor<65536x32768xf32>, %arg2: tensor<65536x32768xf32>, %arg3: tensor<65536x32768xf32>) 
    -> tensor<65536x32768xf32> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} 
{
    %0 = tensor.empty() : tensor<32768xf32>
    %1 = tensor.empty() : tensor<65536x32768xf32>
    %2 = tensor.empty() : tensor<65536x32768xf32>
    %mul = linalg.elemwise_binary { fun = #linalg.binary_fn<mul> } ins(%arg1, %arg2 : tensor<65536x32768xf32>, tensor<65536x32768xf32>) outs(%1 : tensor<65536x32768xf32>) -> tensor<65536x32768xf32>
    %reduced = linalg.reduce { arith.addf } ins(%mul : tensor<65536x32768xf32>) outs(%0 : tensor<32768xf32>) dimensions = [0] { reduce }
    %bc_after_reduce = linalg.broadcast ins(%reduced : tensor<32768xf32>) outs(%1 : tensor<65536x32768xf32>) dimensions = [0]
    return %bc_after_reduce : tensor<65536x32768xf32>
  }
}

// CHECK: outlined_vf_0
// CHECK: outlined_vf_1
// CHECK: outlined_vf_2
// We should do fusion to have have at most 3 VF here
// CHECK-NOT: outlined_vf_3

// All Linalg op in the original function should be converted to vector
// CHECK: test_vectorization
// CHECK-NOT: elemwise_binary
// CHECK-NOT: reduce
// CHECK-NOT: broadcast
