// RUN: bishengir-opt --infer-simt-vf-memory-scope-hint --split-input-file %s | FileCheck %s

// -----

// CHECK: func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<?xi64> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<gm>}, %arg1: memref<8xi64> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}, %arg2: memref<?xf32> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<gm>}, %arg3: i32, %arg4: memref<8xf32> {hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}, %arg5: memref<1024xi8> {hivm.shared_memory, hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>})
module {
  func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<?xi64>, %arg1: memref<8xi64>, %arg2: memref<?xf32>, %arg3: i32, %arg4: memref<8xf32>, %arg5: memref<1024xi8> {hivm.shared_memory}) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%arg1 : memref<8xi64>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<8xi64>
    %1 = tensor.empty() : tensor<8xf32>
    %2 = hivm.hir.gather_load ins(%arg2 : memref<?xf32>, %0 : tensor<8xi64>, %arg3 : i32) outs(%1 : tensor<8xf32>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>, isVolatile = false} -> tensor<8xf32>
    hivm.hir.local_store ins(%arg4 : memref<8xf32>, %2 : tensor<8xf32>)
    return
  }

  func.func @simple_indirect_load_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xi64>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>} {
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %alloc = memref.alloc() : memref<8xi64>
    %alloc_0 = memref.alloc() : memref<8xf32>
    %alloc_1 = memref.alloc() {hivm.shared_memory} : memref<1024xi8>
    call @simple_indirect_load_kernel_scope_0(%arg3, %alloc, %arg2, %c1_i32, %alloc_0, %alloc_1) : (memref<?xi64>, memref<8xi64>, memref<?xf32>, i32, memref<8xf32>, memref<1024xi8>) -> ()
    %2 = bufferization.to_tensor %alloc_0 : memref<8xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    hivm.hir.store ins(%2 : tensor<8xf32>) outs(%reinterpret_cast : memref<8xf32, strided<[1]>>)
    return
  }
}
