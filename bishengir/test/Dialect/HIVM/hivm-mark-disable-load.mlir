// RUN: bishengir-opt -pass-pipeline="builtin.module(hivm-mark-disable-load)" %s | FileCheck %s

// -----
// Single-function test: load from a func arg that is also written in the same
// function must be marked with disableDCache.

// CHECK-LABEL: func.func @kernel
func.func @kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    // CHECK: memref.load {{.*}}[{{.*}}] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32}
    %2 = memref.load %reinterpret_cast[%c0] : memref<1xf32, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    // CHECK: memref.load {{.*}}[{{.*}}] {markDCacheInvalidatePatternVisited = 0 : i32}
    %3 = memref.load %reinterpret_cast_0[%c0] : memref<1xf32, strided<[1]>>
    %4 = tensor.empty() : tensor<1xf32>
    %inserted = tensor.insert %2 into %4[%c0] : tensor<1xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    bufferization.materialize_in_destination %inserted in writable %reinterpret_cast_1 : (tensor<1xf32>, memref<1xf32, strided<[1]>>) -> ()
    %inserted_2 = tensor.insert %3 into %4[%c0] : tensor<1xf32>
    bufferization.materialize_in_destination %inserted_2 in writable %reinterpret_cast_1 : (tensor<1xf32>, memref<1xf32, strided<[1]>>) -> ()
    %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1]>>
    bufferization.materialize_in_destination %1 in writable %reinterpret_cast_3 : (tensor<32xf32>, memref<32xf32, strided<[1]>>) -> ()
    // CHECK: memref.load {{.*}}[{{.*}}] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32}
    %5 = memref.load %reinterpret_cast[%c0] : memref<1xf32, strided<[1]>>
    %inserted_4 = tensor.insert %5 into %4[%c0] : tensor<1xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    bufferization.materialize_in_destination %inserted_4 in writable %reinterpret_cast_5 : (tensor<1xf32>, memref<1xf32, strided<[1]>>) -> ()
    return
}

// -----
// Cross-function test (mix_aic / mix_aiv split kernel):
//
// my_kernel_mix_aic reads from %arg2 (index 2).
// my_kernel_mix_aiv writes to %arg2 (index 2) via bufferization.materialize_in_destination.
// The load in mix_aic must be marked with disableDCache because the paired
// mix_aiv function writes to the same buffer argument.
//
// my_kernel_mix_aic also reads from %arg3 (index 3), which is not written in
// either function, so that load must NOT be marked with disableDCache.

// CHECK-LABEL: func.func @my_kernel_mix_aic
func.func @my_kernel_mix_aic(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    // arg2 is written by mix_aiv -> must be marked
    // CHECK: memref.load {{.*}}[{{.*}}] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32}
    %0 = memref.load %reinterpret_cast[%c0] : memref<1xf32, strided<[1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    // arg3 is not written anywhere -> must NOT have disableDCache
    // CHECK: memref.load {{.*}}[{{.*}}] {markDCacheInvalidatePatternVisited = 0 : i32}
    %1 = memref.load %reinterpret_cast_1[%c0] : memref<1xf32, strided<[1]>>
    %2 = tensor.empty() : tensor<1xf32>
    %inserted = tensor.insert %0 into %2[%c0] : tensor<1xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    bufferization.materialize_in_destination %inserted in writable %reinterpret_cast_2 : (tensor<1xf32>, memref<1xf32, strided<[1]>>) -> ()
    return
}

// CHECK-LABEL: func.func @my_kernel_mix_aiv
func.func @my_kernel_mix_aiv(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    // mix_aiv writes arg2 -> triggers disableDCache on mix_aic's load from arg2
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    bufferization.materialize_in_destination %1 in writable %reinterpret_cast : (tensor<1xf32>, memref<1xf32, strided<[1]>>) -> ()
    return
}

// -----
// Symmetric cross-function test (mix_aiv reads, mix_aic writes):
//
// sym_kernel_mix_aiv reads from %arg2 (index 2).
// sym_kernel_mix_aic writes to %arg2 (index 2).
// The load in mix_aiv must be marked with disableDCache because the paired
// mix_aic function writes to the same buffer argument.

// CHECK-LABEL: func.func @sym_kernel_mix_aiv
func.func @sym_kernel_mix_aiv(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    // arg2 is written by mix_aic -> must be marked
    // CHECK: memref.load {{.*}}[{{.*}}] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32}
    %0 = memref.load %reinterpret_cast[%c0] : memref<1xf32, strided<[1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    // arg3 is not written anywhere -> must NOT have disableDCache
    // CHECK: memref.load {{.*}}[{{.*}}] {markDCacheInvalidatePatternVisited = 0 : i32}
    %1 = memref.load %reinterpret_cast_1[%c0] : memref<1xf32, strided<[1]>>
    %2 = tensor.empty() : tensor<1xf32>
    %inserted = tensor.insert %0 into %2[%c0] : tensor<1xf32>
    %inserted_1 = tensor.insert %1 into %2[%c0] : tensor<1xf32>
    return
}

// CHECK-LABEL: func.func @sym_kernel_mix_aic
func.func @sym_kernel_mix_aic(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    // mix_aic writes arg2 -> triggers disableDCache on mix_aiv's load from arg2
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    bufferization.materialize_in_destination %1 in writable %reinterpret_cast : (tensor<1xf32>, memref<1xf32, strided<[1]>>) -> ()
    return
}
