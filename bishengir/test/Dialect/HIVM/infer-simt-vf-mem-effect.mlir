// RUN: bishengir-opt --infer-simt-vf-memory-effect --split-input-file %s | FileCheck %s

// -----

// CHECK: func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<8xi64> {memory.effect = #hivm.memory.effect<read>}, %arg1: memref<?xf32> {memory.effect = #hivm.memory.effect<read>}, %arg2: i32, %arg3: memref<8xf32> {memory.effect = #hivm.memory.effect<write>})
module {
  func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<8xi64>, %arg1: memref<?xf32>, %arg2: i32, %arg3: memref<8xf32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %0 = tensor.empty() : tensor<8xi64>
    %1 = bufferization.to_memref %0 : memref<8xi64>
    hivm.hir.load ins(%arg0 : memref<8xi64>) outs(%1 : memref<8xi64>)
    %2 = hivm.hir.gather_load ins(%arg1 : memref<?xf32>, %0 : tensor<8xi64>, %arg2 : i32) {cache = 1 : i32, evict = #hivm.evictionpolicy<EvictLast>, isVolatile = false} -> tensor<8xf32>
    hivm.hir.store ins(%2 : tensor<8xf32>) outs(%arg3 : memref<8xf32>)
    return
  }
}

// -----

// CHECK: func.func @simple_indirect_store_kernel_scope_0(%arg0: memref<8xf32> {memory.effect = #hivm.memory.effect<read>}, %arg1: memref<8xi64> {memory.effect = #hivm.memory.effect<read>}, %arg2: memref<?xf32> {memory.effect = #hivm.memory.effect<write>}, %arg3: i32)
module {
  func.func @simple_indirect_store_kernel_scope_0(%arg0: memref<8xf32>, %arg1: memref<8xi64>, %arg2: memref<?xf32>, %arg3: i32) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %0 = tensor.empty() : tensor<8xf32>
    %1 = bufferization.to_memref %0 : memref<8xf32>
    hivm.hir.load ins(%arg0 : memref<8xf32>) outs(%1 : memref<8xf32>)
    %2 = tensor.empty() : tensor<8xi64>
    %3 = bufferization.to_memref %2 : memref<8xi64>
    hivm.hir.load ins(%arg1 : memref<8xi64>) outs(%3 : memref<8xi64>)
    hivm.hir.scatter_store ins(%arg2 : memref<?xf32>, %2 : tensor<8xi64>, %0 : tensor<8xf32>, %arg3 : i32) {cache = 1 : i32, evict = #hivm.evictionpolicy<EvictLast>}
    return
  }
}

