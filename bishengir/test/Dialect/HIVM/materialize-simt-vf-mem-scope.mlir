// RUN: bishengir-opt --materialize-simt-vf-memory-scope --split-input-file %s | FileCheck %s

// -----

// CHECK: func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<?xi64, #hivm.address_space<gm>> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<gm>}, %arg1: memref<8xi64, #hivm.address_space<ub>> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}, %arg2: memref<?xf32, #hivm.address_space<gm>> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<gm>}, %arg3: i32, %arg4: memref<8xf32, #hivm.address_space<ub>> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}, %arg5: memref<1024xi8, #hivm.address_space<ub>> {hivm.shared_memory, hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>})
// CHECK: %[[CAST:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<8xi64, strided<[1]>, #hivm.address_space<gm>>
// CHECK: hivm.hir.load ins(%[[CAST]] : memref<8xi64, strided<[1]>, #hivm.address_space<gm>>) outs(%arg1 : memref<8xi64, #hivm.address_space<ub>>)
// CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.local_load ins(%[[SCRATCH]] : memref<8xf32, #hivm.address_space<ub>>) -> tensor<8xf32>
module {
  func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<?xi64> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<gm>}, %arg1: memref<8xi64> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}, %arg2: memref<?xf32> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<gm>}, %arg3: i32, %arg4: memref<8xf32> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}, %arg5: memref<1024xi8> {hivm.shared_memory, hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%arg1 : memref<8xi64>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<8xi64>
    %1 = tensor.empty() : tensor<8xf32>
    %2 = hivm.hir.gather_load ins(%arg2 : memref<?xf32>, %0 : tensor<8xi64>, %arg3 : i32) outs(%1 : tensor<8xf32>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>, isVolatile = false} -> tensor<8xf32>
    hivm.hir.local_store ins(%arg4 : memref<8xf32>, %2 : tensor<8xf32>)
    %scratch = memref.alloc() : memref<8xf32>
    %3 = hivm.hir.local_load ins(%scratch : memref<8xf32>) -> tensor<8xf32>
    hivm.hir.local_store ins(%arg4 : memref<8xf32>, %3 : tensor<8xf32>)
    return
  }
}
