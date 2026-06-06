// RUN: bishengir-opt %s -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries analysis-heuristic=top-down function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map" -split-input-file | FileCheck %s

module {

func.func @callee_loop_with_index(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {hivm.vector_function, no_inline} {
  return %arg0 : tensor<64xf32>
}

// CHECK-LABEL: func.func @
func.func @test_loop_with_index_arg(%arg0: tensor<31744xf32>, %arg1: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c31744 = arith.constant 31744 : index
  scf.for %arg2 = %c0 to %c31744 step %c64  {
    %extracted = tensor.extract_slice %arg0[%arg2] [64] [1] : tensor<31744xf32> to tensor<64xf32>
    // CHECK: fold_offset_into_ptr
    %1 = func.call @callee_loop_with_index(%extracted) {hivm.vector_function, no_inline} : (tensor<64xf32>) -> tensor<64xf32>
    hivm.hir.store ins(%1: tensor<64xf32>) outs(%arg1 : memref<64xf32>)
  }
  return
}

}
