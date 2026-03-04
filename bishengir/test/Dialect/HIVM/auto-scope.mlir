// RUN: bishengir-opt --auto-scope --split-input-file %s | FileCheck %s

module {
  // CHECK: scope.scope
  // CHECK:   %[[REINTERPRETCAST:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
  // CHECK:   hivm.hir.load ins(%[[REINTERPRETCAST:.*]] : memref<8xi64, strided<[1]>>) outs(%[[ALLOC:.*]] : memref<8xi64>) eviction_policy = <EvictFirst>
  // CHECK:   %[[TOTENSOR:.*]] = bufferization.to_tensor %[[ALLOC:.*]] restrict writable : memref<8xi64>
  // CHECK:   %[[GATHERLOAD:.*]] = hivm.hir.gather_load ins(%[[ARG2:.*]] : memref<?xf32>, %[[TOTENSOR:.*]] : tensor<8xi64>, %[[c:.*]] : i32) {cache = 1 : i32, evict = #hivm.evictionpolicy<EvictLast>, isVolatile = false} -> tensor<8xf32>
  // CHECK:   scope.return %[[GATHERLOAD:.*]] : tensor<8xf32>
  func.func @simple_indirect_load_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xi64>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    %alloc = memref.alloc() : memref<8xi64>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%alloc : memref<8xi64>) eviction_policy = <EvictFirst>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<8xi64>
    %3 = hivm.hir.gather_load ins(%arg2 : memref<?xf32>, %2 : tensor<8xi64>, %c1_i32 : i32) {cache = 1 : i32, evict = #hivm.evictionpolicy<EvictLast>, isVolatile = false} -> tensor<8xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    hivm.hir.store ins(%3 : tensor<8xf32>) outs(%reinterpret_cast_0 : memref<8xf32, strided<[1]>>)
    return
  }
}

// -----

module {
  // CHECK: scope.scope
  // CHECK:   %[[REINTERPRETCAST:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
  // CHECK:   hivm.hir.load ins(%[[REINTERPRETCAST:.*]] : memref<8xi64, strided<[1]>>) outs(%[[ALLOC:.*]] : memref<8xi64>) eviction_policy = <EvictFirst>
  // CHECK:   %[[TOTENSOR:.*]] = bufferization.to_tensor %[[ALLOC:.*]] restrict writable : memref<8xi64>
  // CHECK:   hivm.hir.scatter_store ins(%[[ARG2:.*]] : memref<?xf32>, %[[TOTENSOR:.*]] : tensor<8xi64>, %[[DATA:.*]] : tensor<8xf32>, %[[c:.*]] : i32) {cache = 1 : i32, evict = #hivm.evictionpolicy<EvictLast>}
  // CHECK:   scope.return
  func.func @simple_indirect_store_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xi64>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    %alloc = memref.alloc() : memref<8xi64>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%alloc : memref<8xi64>) eviction_policy = <EvictFirst>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<8xi64>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    %alloc_1 = memref.alloc() : memref<8xf32>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<8xf32, strided<[1]>>) outs(%alloc_1 : memref<8xf32>) eviction_policy = <EvictFirst>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<8xf32>
    hivm.hir.scatter_store ins(%arg2 : memref<?xf32>, %2 : tensor<8xi64>, %3 : tensor<8xf32>, %c1_i32 : i32) {cache = 1 : i32, evict = #hivm.evictionpolicy<EvictLast>}
    return
  }
}