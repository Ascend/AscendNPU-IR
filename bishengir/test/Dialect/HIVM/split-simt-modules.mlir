// RUN: bishengir-opt --split-simt-module --split-input-file %s | FileCheck %s

// CHECK: module {
// CHECK-NEXT: module {
// CHECK-NEXT: func.func private @simple_indirect_load_kernel_scope_0
// CHECK-NEXT: func.func @simple_indirect_load_kernel
// CHECK: module attributes {hacc.simt_module} {
// CHECK-NEXT: func.func @simple_indirect_load_kernel_scope_0

module {
  func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<?xi64, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>}, %arg1: memref<8xi64, #hivm.address_space<ub>>, %arg2: memref<?xf32, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>}, %arg3: i32, %arg4: memref<8xf32, #hivm.address_space<ub>>, %arg5: memref<1024xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<8xi64, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>, #hivm.address_space<gm>>) outs(%arg1 : memref<8xi64, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<8xi64, #hivm.address_space<ub>>
    %1 = tensor.empty() : tensor<8xf32>
    %2 = hivm.hir.gather_load ins(%arg2 : memref<?xf32, #hivm.address_space<gm>>, %0 : tensor<8xi64>, %arg3 : i32) outs(%1 : tensor<8xf32>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>, isVolatile = false} -> tensor<8xf32>
    hivm.hir.local_store ins(%arg4 : memref<8xf32, #hivm.address_space<ub>>, %2 : tensor<8xf32>)
    return
  }
  func.func @simple_indirect_load_kernel(%arg0: memref<?xi8, #hivm.address_space<gm>>, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xf32, #hivm.address_space<gm>>, %arg3: memref<?xi64, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %alloc = memref.alloc() : memref<8xi64, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() {hivm.shared_memory} : memref<1024xi8, #hivm.address_space<ub>>
    call @simple_indirect_load_kernel_scope_0(%arg3, %alloc, %arg2, %c1_i32, %alloc_0, %alloc_1) : (memref<?xi64, #hivm.address_space<gm>>, memref<8xi64, #hivm.address_space<ub>>, memref<?xf32, #hivm.address_space<gm>>, i32, memref<8xf32, #hivm.address_space<ub>>, memref<1024xi8, #hivm.address_space<ub>>) -> ()
    %2 = bufferization.to_tensor %alloc_0 : memref<8xf32, #hivm.address_space<ub>>
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.store ins(%2 : tensor<8xf32>) outs(%reinterpret_cast : memref<8xf32, strided<[1]>, #hivm.address_space<gm>>)
    return
  }
}

// -----
// CHECK: module {
// CHECK-NEXT: module {
// CHECK-NEXT: func.func private @simple_indirect_store_kernel_scope_0
// CHECK-NEXT: func.func @simple_indirect_store_kernel
// CHECK: module attributes {hacc.simt_module} {
// CHECK-NEXT: func.func @simple_indirect_store_kernel_scope_0
module {
  func.func @simple_indirect_store_kernel_scope_0(%arg0: memref<?xi64, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>}, %arg1: memref<8xi64, #hivm.address_space<ub>>, %arg2: memref<8xf32, #hivm.address_space<ub>>, %arg3: memref<?xf32, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<write>}, %arg4: i32, %arg5: memref<1024xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<8xi64, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>, #hivm.address_space<gm>>) outs(%arg1 : memref<8xi64, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<8xi64, #hivm.address_space<ub>>
    %1 = hivm.hir.local_load ins(%arg2 : memref<8xf32, #hivm.address_space<ub>>) -> tensor<8xf32>
    hivm.hir.scatter_store ins(%0 : tensor<8xi64>, %1 : tensor<8xf32>, %arg4 : i32) outs(%arg3 : memref<?xf32, #hivm.address_space<gm>>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>}
    return
  }
  func.func @simple_indirect_store_kernel(%arg0: memref<?xi8, #hivm.address_space<gm>>, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xf32, #hivm.address_space<gm>>, %arg3: memref<?xi64, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %alloc = memref.alloc() : memref<8xi64, #hivm.address_space<ub>>
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>, #hivm.address_space<gm>>
    %alloc_0 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xf32, strided<[1]>, #hivm.address_space<gm>>) outs(%alloc_0 : memref<8xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
    %alloc_1 = memref.alloc() {hivm.shared_memory} : memref<1024xi8, #hivm.address_space<ub>>
    call @simple_indirect_store_kernel_scope_0(%arg3, %alloc, %alloc_0, %arg2, %c1_i32, %alloc_1) : (memref<?xi64, #hivm.address_space<gm>>, memref<8xi64, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>, memref<?xf32, #hivm.address_space<gm>>, i32, memref<1024xi8, #hivm.address_space<ub>>) -> ()
    return
  }
}
