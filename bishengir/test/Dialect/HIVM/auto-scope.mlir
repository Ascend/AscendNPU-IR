// RUN: bishengir-opt --auto-scope --split-input-file %s | FileCheck %s

module {
  // CHECK: scope.scope
  // CHECK-NEXT:   %[[REINTERPRETCAST:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
  // CHECK-NEXT:   hivm.hir.load ins(%[[REINTERPRETCAST:.*]] : memref<8xi64, strided<[1]>>) outs(%[[ALLOC:.*]] : memref<8xi64>) eviction_policy = <EvictFirst>
  // CHECK-NEXT:   %[[TOTENSOR:.*]] = bufferization.to_tensor %[[ALLOC:.*]] restrict writable : memref<8xi64>
  // CHECK-NEXT:   %[[EMPTY:.*]] = tensor.empty() : tensor<8xf32>
  // CHECK-NEXT:   %[[GATHERLOAD:.*]] = hivm.hir.gather_load ins(%[[ARG2:.*]] : memref<?xf32>, %[[TOTENSOR:.*]] : tensor<8xi64>, %[[c:.*]] : i32) outs(%[[EMPTY]] : tensor<8xf32>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>, isVolatile = false} -> tensor<8xf32>
  // CHECK-NEXT:   scope.return %[[GATHERLOAD:.*]] : tensor<8xf32>
  // CHECK-NEXT:  hivm.func_core_type = #hivm.func_core_type<AIV>
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
    %3 = tensor.empty() : tensor<8xf32>
    %4 = hivm.hir.gather_load ins(%arg2 : memref<?xf32>, %2 : tensor<8xi64>, %c1_i32 : i32) outs(%3 : tensor<8xf32>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>, isVolatile = false} -> tensor<8xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    hivm.hir.store ins(%4 : tensor<8xf32>) outs(%reinterpret_cast_0 : memref<8xf32, strided<[1]>>)
    return
  }
}

// -----

module {
  // CHECK: scope.scope
  // CHECK:   %[[BLOCKIDX:.*]] = hivm.hir.get_block_idx -> i64
  // CHECK-NOT: hivm.hir.get_block_idx
  // CHECK:   %[[GATHER:.*]] = hivm.hir.gather_load
  // CHECK:   scope.return %[[GATHER]]
  func.func @dedup_common_simt_dependencies(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xi64>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.divsi %3, %arg9 : i32
    %5 = arith.remsi %4, %arg8 : i32
    %6 = arith.muli %arg9, %arg8 : i32
    %7 = arith.divsi %3, %6 : i32
    %8 = arith.remsi %7, %arg7 : i32
    %9 = arith.muli %8, %c2_i32 : i32
    %10 = tensor.empty() : tensor<2xi32>
    %11 = hivm.hir.varange offset[%c0] strides[%c1] outs(%10 : tensor<2xi32>) -> tensor<2xi32>
    %12 = tensor.empty() : tensor<2xi32>
    %13 = hivm.hir.vadd ins(%11, %9 : tensor<2xi32>, i32) outs(%12 : tensor<2xi32>) -> tensor<2xi32>
    %14 = arith.muli %5, %c4_i32 : i32
    %15 = arith.index_cast %9 : i32 to index
    %16 = arith.index_cast %arg6 : i32 to index
    %17 = arith.index_cast %14 : i32 to index
    %18 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%17, %15, %16]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%18], sizes: [2, 4], strides: [%16, 1] : memref<?xi64> to memref<2x4xi64, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<2x4xi64>
    hivm.hir.load ins(%reinterpret_cast : memref<2x4xi64, strided<[?, 1], offset: ?>>) outs(%alloc : memref<2x4xi64>) eviction_policy = <EvictFirst>
    %19 = bufferization.to_tensor %alloc restrict writable : memref<2x4xi64>
    %20 = tensor.empty() : tensor<2xi32>
    %21 = hivm.hir.vmul ins(%13, %arg5 : tensor<2xi32>, i32) outs(%20 : tensor<2xi32>) -> tensor<2xi32>
    %22 = tensor.empty() : tensor<2xi64>
    %23 = hivm.hir.vcast ins(%21 : tensor<2xi32>) outs(%22 : tensor<2xi64>) -> tensor<2xi64>
    %expanded = tensor.expand_shape %23 [[0, 1]] output_shape [2, 1] : tensor<2xi64> into tensor<2x1xi64>
    %24 = tensor.empty() : tensor<2x4xi64>
    %25 = hivm.hir.vbrc ins(%expanded : tensor<2x1xi64>) outs(%24 : tensor<2x4xi64>) broadcast_dims = [1] -> tensor<2x4xi64>
    %26 = tensor.empty() : tensor<2x4xi64>
    %27 = hivm.hir.vadd ins(%25, %19 : tensor<2x4xi64>, tensor<2x4xi64>) outs(%26 : tensor<2x4xi64>) -> tensor<2x4xi64>
    %28 = tensor.empty() : tensor<2x4xf32>
    %29 = hivm.hir.gather_load ins(%arg2 : memref<?xf32>, %27 : tensor<2x4xi64>, %c1_i32 : i32) outs(%28 : tensor<2x4xf32>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>, isVolatile = false} -> tensor<2x4xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%18], sizes: [2, 4], strides: [%16, 1] : memref<?xf32> to memref<2x4xf32, strided<[?, 1], offset: ?>>
    hivm.hir.store ins(%29 : tensor<2x4xf32>) outs(%reinterpret_cast_0 : memref<2x4xf32, strided<[?, 1], offset: ?>>)
    return
  }
}

// -----

module {
  // CHECK: scope.scope
  // CHECK-NEXT:   %[[REINTERPRETCAST:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
  // CHECK-NEXT:   hivm.hir.load ins(%[[REINTERPRETCAST:.*]] : memref<8xi64, strided<[1]>>) outs(%[[ALLOC:.*]] : memref<8xi64>) eviction_policy = <EvictFirst>
  // CHECK-NEXT:   %[[TOTENSOR:.*]] = bufferization.to_tensor %[[ALLOC:.*]] restrict writable : memref<8xi64>
  // CHECK-NEXT:   hivm.hir.scatter_store ins(%[[TOTENSOR:.*]] : tensor<8xi64>, %[[DATA:.*]] : tensor<8xf32>, %[[c:.*]] : i32) outs(%[[ARG2:.*]] : memref<?xf32>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>}
  // CHECK-NEXT:   scope.return
  // CHECK-NEXT:  hivm.func_core_type = #hivm.func_core_type<AIV>
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
    hivm.hir.scatter_store ins(%2 : tensor<8xi64>, %3 : tensor<8xf32>, %c1_i32 : i32) outs(%arg2 : memref<?xf32>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>}
    return
  }
}

// -----
// CHECK: scope.return
// CHECK-NEXT: hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>
module {
  func.func @explicit_scope_kernel(%arg0: tensor<8xi64>, %arg1 : memref<8xi64, strided<[1]>>) {
    %0 = scope.scope : () -> tensor<8xi64> {
      scope.return %arg0 : tensor<8xi64>
    } {vector_type = "simt"}
    hivm.hir.store ins(%0 : tensor<8xi64>) outs(%arg1 : memref<8xi64, strided<[1]>>)
    return
  }
}