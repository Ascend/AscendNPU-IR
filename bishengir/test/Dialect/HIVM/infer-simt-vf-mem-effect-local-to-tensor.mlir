// RUN: bishengir-opt --infer-simt-vf-memory-effect %s | FileCheck %s

// CHECK: func.func @ignore_local_to_tensor_scope_0(%arg0: memref<?xf32> {hivm.memory_effect = #hivm.memory_effect<read>}, %arg1: memref<8xf32> {hivm.memory_effect = #hivm.memory_effect<write>})
module {
  func.func @ignore_local_to_tensor_scope_0(%arg0: memref<?xf32>, %arg1: memref<8xf32>) attributes {no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %c1_i32 = arith.constant 1 : i32
    %alloc = memref.alloc() : memref<8xi64>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<8xi64>
    %1 = tensor.empty() : tensor<8xf32>
    %2 = hivm.hir.gather_load ins(%arg0 : memref<?xf32>, %0 : tensor<8xi64>, %c1_i32 : i32) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
    hivm.hir.local_store ins(%arg1 : memref<8xf32>, %2 : tensor<8xf32>)
    return
  }
}
