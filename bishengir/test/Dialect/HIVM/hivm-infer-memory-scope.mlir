// RUN: bishengir-opt --hivm-infer-mem-scope --split-input-file %s | FileCheck %s

// CHECK: func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<?xi64, #hivm.address_space<gm>>, %arg1: memref<8xi64, #hivm.address_space<ub>>, %arg2: memref<?xf32, #hivm.address_space<gm>>, %arg3: i32, %arg4: memref<8xf32, #hivm.address_space<ub>>)
module {
  func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<?xi64>, %arg1: memref<8xi64>, %arg2: memref<?xf32>, %arg3: i32, %arg4: memref<8xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%arg1 : memref<8xi64>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<8xi64>
    %1 = hivm.hir.gather_load ins(%arg2 : memref<?xf32>, %0 : tensor<8xi64>, %arg3 : i32) {cache = 1 : i32, evict = #hivm.evictionpolicy<EvictLast>, isVolatile = false} -> tensor<8xf32>
    hivm.hir.local_store ins(%arg4 : memref<8xf32>, %1 : tensor<8xf32>)
    return
  }
  func.func @simple_indirect_load_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xi64>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %alloc = memref.alloc() : memref<8xi64>
    %alloc_0 = memref.alloc() : memref<8xf32>
    call @simple_indirect_load_kernel_scope_0(%arg3, %alloc, %arg2, %c1_i32, %alloc_0) : (memref<?xi64>, memref<8xi64>, memref<?xf32>, i32, memref<8xf32>) -> ()
    %2 = bufferization.to_tensor %alloc_0 : memref<8xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    hivm.hir.store ins(%2 : tensor<8xf32>) outs(%reinterpret_cast : memref<8xf32, strided<[1]>>)
    return
  }
}

// -----

// CHECK: func.func @simple_indirect_store_kernel_scope_0(%arg0: memref<?xi64, #hivm.address_space<gm>>, %arg1: memref<8xi64, #hivm.address_space<ub>>, %arg2: memref<8xf32, #hivm.address_space<ub>>, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: i32)
module {
  func.func @simple_indirect_store_kernel_scope_0(%arg0: memref<?xi64>, %arg1: memref<8xi64>, %arg2: memref<8xf32>, %arg3: memref<?xf32>, %arg4: i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%arg1 : memref<8xi64>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<8xi64>
    %1 = hivm.hir.local_load ins(%arg2 : memref<8xf32>) -> tensor<8xf32>
    hivm.hir.scatter_store ins(%arg3 : memref<?xf32>, %0 : tensor<8xi64>, %1 : tensor<8xf32>, %arg4 : i32) {cache = 1 : i32, evict = #hivm.evictionpolicy<EvictLast>}
    return
  }
  func.func @simple_indirect_store_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xi64>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %alloc = memref.alloc() : memref<8xi64>
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    %alloc_0 = memref.alloc() : memref<8xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<8xf32, strided<[1]>>) outs(%alloc_0 : memref<8xf32>) eviction_policy = <EvictFirst>
    call @simple_indirect_store_kernel_scope_0(%arg3, %alloc, %alloc_0, %arg2, %c1_i32) : (memref<?xi64>, memref<8xi64>, memref<8xf32>, memref<?xf32>, i32) -> ()
    return
  }
}
