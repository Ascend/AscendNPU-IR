// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @test_NormalizeCumOpF16ToF32Type_cumsum_f16
// CHECK: hfusion.cumsum %[[INPUT0:.*]] : tensor<4x64xf32> cum_dims = [0] reverse = true -> tensor<4x32xf32>
func.func @test_NormalizeCumOpF16ToF32Type_cumsum_f16(%arg0: tensor<4x64xf16>, %reverse: tensor<?xi1>) -> tensor<4x32xf16> {
  %0 = tensor.empty() : tensor<4x32xf16>
  %1 = hfusion.cumsum %arg0 : tensor<4x64xf16> cum_dims = [0] reverse = true -> tensor<4x32xf16>
  return %1 : tensor<4x32xf16>
}

// -----

// CHECK-LABEL: @test_NormalizeCumOpF16ToF32Type_cumprod_f16
// CHECK: hfusion.cumprod %[[INPUT0:.*]] : tensor<4x64xf32> cum_dims = [1] reverse = true -> tensor<4x32xf32>
func.func @test_NormalizeCumOpF16ToF32Type_cumprod_f16(%arg0: tensor<4x64xf16>, %reverse: tensor<?xi1>) -> tensor<4x32xf16> {
  %0 = tensor.empty() : tensor<4x32xf16>
  %1 = hfusion.cumprod %arg0 : tensor<4x64xf16> cum_dims = [1] reverse = true -> tensor<4x32xf16>
  return %1 : tensor<4x32xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_Elemwise_atomic_or_i16
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[LHS:.*]] = bufferization.to_tensor %{{.*}} restrict writable : memref<256xi16>
// CHECK: %[[RES:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[LHS]], %{{.*}} : tensor<256xi16>, tensor<256xi16>) outs(%{{.*}} : tensor<256xi16>) -> tensor<256xi16>
// CHECK: bufferization.materialize_in_destination %[[RES]] in writable %{{.*}} : (tensor<256xi16>, memref<256xi16, strided<[1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_Elemwise_atomic_or_i16(%arg0 : memref<?xi16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi16>) {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %c256_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1], offset: ?>>
    %1 = bufferization.to_memref %arg1 : memref<256xi16, strided<[1]>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<or>} ins(%1 : memref<256xi16, strided<[1]>>) outs(%reinterpret_cast : memref<256xi16, strided<[1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_Elemwise_atomic_umax_i16
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[LHS:.*]] = bufferization.to_tensor %{{.*}} restrict writable : memref<256xi16>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<max_unsigned>} ins(%[[LHS]], %{{.*}} : tensor<256xi16>, tensor<256xi16>) outs(%{{.*}} : tensor<256xi16>) -> tensor<256xi16>
// CHECK: bufferization.materialize_in_destination %[[RES]] in writable %{{.*}} : (tensor<256xi16>, memref<256xi16, strided<[1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_Elemwise_atomic_umax_i16(%arg0 : memref<?xi16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi16>) {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %c256_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1], offset: ?>>
    %1 = bufferization.to_memref %arg1 : memref<256xi16, strided<[1]>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<umax>} ins(%1 : memref<256xi16, strided<[1]>>) outs(%reinterpret_cast : memref<256xi16, strided<[1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_Elemwise_atomic_max_i64
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[LHS:.*]] = bufferization.to_tensor %{{.*}} restrict writable : memref<256xi64>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%[[LHS]], %{{.*}} : tensor<256xi64>, tensor<256xi64>) outs(%{{.*}} : tensor<256xi64>) -> tensor<256xi64>
// CHECK: bufferization.materialize_in_destination %[[RES]] in writable %{{.*}} : (tensor<256xi64>, memref<256xi64, strided<[1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_Elemwise_atomic_max_i64(%arg0 : memref<?xi64> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi64>) {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %c256_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256], strides: [1] : memref<?xi64> to memref<256xi64, strided<[1], offset: ?>>
    %1 = bufferization.to_memref %arg1 : memref<256xi64, strided<[1]>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<max>} ins(%1 : memref<256xi64, strided<[1]>>) outs(%reinterpret_cast : memref<256xi64, strided<[1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_Elemwise_atomic_add_f16_keep_original
// CHECK: hfusion.store {atomic_kind = #hfusion.atomic_kind<add>} ins(%[[SRC:.*]] : memref<256xf16, strided<[1]>>) outs(%[[DST:.*]] : memref<256xf16, strided<[1], offset: ?>>)
// CHECK-NOT: hivm.hir.create_sync_block_lock
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_Elemwise_atomic_add_f16_keep_original(%arg0 : memref<?xf16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xf16>) {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %c256_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256], strides: [1] : memref<?xf16> to memref<256xf16, strided<[1], offset: ?>>
    %1 = bufferization.to_memref %arg1 : memref<256xf16, strided<[1]>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<add>} ins(%1 : memref<256xf16, strided<[1]>>) outs(%reinterpret_cast : memref<256xf16, strided<[1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_Elemwise_atomic_and
// CHECK: %[[VAL_9:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[VAL_9]] : memref<1xi64>)
// CHECK: memref.copy %[[VAL_4:.*]], %[[VAL_6:.*]] : memref<256xi16, strided<[1], offset: ?>> to memref<256xi16>
// CHECK: %[[VAL_10:.*]] = bufferization.to_tensor %[[VAL_6]] restrict writable : memref<256xi16>
// CHECK: %[[VAL_11:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[VAL_10]], %{{.*}} : tensor<256xi16>, tensor<256xi16>) outs(%{{.*}} : tensor<256xi16>) -> tensor<256xi16>
// CHECK: bufferization.materialize_in_destination %[[VAL_11]] in writable %[[VAL_4]] : (tensor<256xi16>, memref<256xi16, strided<[1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[VAL_9]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_Elemwise_atomic_and(%arg0 : memref<?xi16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi16>) {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %c256_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1], offset: ?>>
    %1 = bufferization.to_memref %arg1 : memref<256xi16, strided<[1]>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<and>} ins(%1 : memref<256xi16, strided<[1]>>) outs(%reinterpret_cast : memref<256xi16, strided<[1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_Elemwise_atomic_add_i64
// CHECK: %[[VAL_9:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[VAL_9]] : memref<1xi64>)
// CHECK: memref.copy %[[VAL_4:.*]], %[[VAL_6:.*]] : memref<256xi64, strided<[1], offset: ?>> to memref<256xi64>
// CHECK: %[[VAL_10:.*]] = bufferization.to_tensor %[[VAL_6]] restrict writable : memref<256xi64>
// CHECK: %[[VAL_11:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_10]], %{{.*}} : tensor<256xi64>, tensor<256xi64>) outs(%{{.*}} : tensor<256xi64>) -> tensor<256xi64>
// CHECK: bufferization.materialize_in_destination %[[VAL_11]] in writable %[[VAL_4]] : (tensor<256xi64>, memref<256xi64, strided<[1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[VAL_9]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_Elemwise_atomic_add_i64(%arg0 : memref<?xi64> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi64>) {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %c256_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256], strides: [1] : memref<?xi64> to memref<256xi64, strided<[1], offset: ?>>
    %1 = bufferization.to_memref %arg1 : memref<256xi64, strided<[1]>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<add>} ins(%1 : memref<256xi64, strided<[1]>>) outs(%reinterpret_cast : memref<256xi64, strided<[1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_Elemwise_atomic_add_fp8
// CHECK: %[[VAL_9:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[VAL_9]] : memref<1xi64>)
// CHECK: memref.copy %[[VAL_4:.*]], %[[VAL_6:.*]] : memref<256xf8E4M3FN, strided<[1], offset: ?>> to memref<256xf8E4M3FN>
// CHECK: %[[VAL_11:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%{{.*}} : tensor<256xf8E4M3FN>) outs(%{{.*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[VAL_12:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_11]], %{{.*}} : tensor<256xf32>, tensor<256xf32>) outs(%{{.*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[VAL_13:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[VAL_12]] : tensor<256xf32>) outs(%{{.*}} : tensor<256xf8E4M3FN>) -> tensor<256xf8E4M3FN>
// CHECK: bufferization.materialize_in_destination %[[VAL_13]] in writable %[[VAL_4]] : (tensor<256xf8E4M3FN>, memref<256xf8E4M3FN, strided<[1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[VAL_9]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_Elemwise_atomic_add_fp8(%arg0 : memref<?xf8E4M3FN> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xf8E4M3FN>) {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %c256_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256], strides: [1] : memref<?xf8E4M3FN> to memref<256xf8E4M3FN, strided<[1], offset: ?>>
    %1 = bufferization.to_memref %arg1 : memref<256xf8E4M3FN, strided<[1]>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<add>} ins(%1 : memref<256xf8E4M3FN, strided<[1]>>) outs(%reinterpret_cast : memref<256xf8E4M3FN, strided<[1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_Elemwise_atomic_dyn_and
// CHECK: %[[VAL_9:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[VAL_9]] : memref<1xi64>)
// CHECK: memref.copy %[[VAL_4:.*]], %[[VAL_6:.*]] : memref<?x?xi16, strided<[512, 1], offset: ?>> to memref<?x?xi16, strided<[256, 1]>>
// CHECK: %[[VAL_11:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%{{.*}}, %{{.*}} : tensor<12x256xi16>, tensor<12x256xi16>) outs(%{{.*}} : tensor<12x256xi16>) -> tensor<12x256xi16>
// CHECK: %[[VAL_12:.*]] = tensor.extract_slice %[[VAL_11:.*]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : tensor<12x256xi16> to tensor<?x?xi16>
// CHECK: bufferization.materialize_in_destination %[[VAL_12]] in writable %[[VAL_4]] : (tensor<?x?xi16>, memref<?x?xi16, strided<[512, 1], offset: ?>>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[VAL_9]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_Elemwise_atomic_dyn_and(%arg0 : memref<?xi16> {tt.divisibility = 16 : i32}, %arg1 : memref<?xi16> {tt.divisibility = 16 : i32}, %arg2: index, %arg3: index, %arg4: index) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg4], sizes: [12, 256], strides: [512, 1] : memref<?xi16> to memref<12x256xi16, strided<[512, 1], offset: ?>>
    %alloc = memref.alloc() : memref<12x256xi16>
    %subview = memref.subview %reinterpret_cast[0, 0] [%arg2, %arg3] [1, 1] : memref<12x256xi16, strided<[512, 1], offset: ?>> to memref<?x?xi16, strided<[512, 1], offset: ?>>
    %subview_0 = memref.subview %alloc[0, 0] [%arg2, %arg3] [1, 1] : memref<12x256xi16> to memref<?x?xi16, strided<[256, 1]>>
    memref.copy %subview, %subview_0 : memref<?x?xi16, strided<[512, 1], offset: ?>> to memref<?x?xi16, strided<[256, 1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%arg4], sizes: [12, 256], strides: [512, 1] : memref<?xi16> to memref<12x256xi16, strided<[512, 1], offset: ?>>
    %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%arg2, %arg3] [1, 1] : memref<12x256xi16, strided<[512, 1], offset: ?>> to memref<?x?xi16, strided<[512, 1], offset: ?>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<and>} ins(%subview_0 : memref<?x?xi16, strided<[256, 1]>>) outs(%subview_2 : memref<?x?xi16, strided<[512, 1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_CAS_atomic_cas_i16x4
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%{{.*}}, %{{.*}} : tensor<4xi16>, tensor<4xi16>) outs(%{{.*}} : tensor<4xi1>) -> tensor<4xi1>
// CHECK: %[[SEL:.*]] = hfusion.select ins(%[[CMP]], %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xi16>, tensor<4xi16>) outs(%{{.*}} : tensor<4xi16>) -> tensor<4xi16>
// CHECK: bufferization.materialize_in_destination %[[SEL]] in writable %{{.*}} : (tensor<4xi16>, memref<4xi16>) -> ()
// CHECK-NOT: hfusion.cast
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_CAS_atomic_cas_i16x4(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<4xi16>, %arg3: memref<4xi16>, %arg4: memref<4xi16>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    hfusion.atomic_cas ins(%arg3, %arg4 : memref<4xi16>, memref<4xi16>) outs(%arg2 : memref<4xi16>)
    return
  }
}

// -----

// CHECK-LABEL:   func.func @test_NormalizeAtomicOps_CAS_triton_atomic_cas_1D
// CHECK: %[[VAL_10:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[VAL_10]] : memref<1xi64>)
// CHECK: %[[VAL_14:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[VAL_12:.*]], %{{.*}} : tensor<1xi64>, tensor<1xi64>) outs(%{{.*}} : tensor<1xi1>) -> tensor<1xi1>
// CHECK: %[[VAL_15:.*]] = hfusion.select ins(%[[VAL_14]], %{{.*}}, %[[VAL_12]] : tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) outs(%{{.*}} : tensor<1xi64>) -> tensor<1xi64>
// CHECK: bufferization.materialize_in_destination %[[VAL_15]] in writable %{{.*}} : (tensor<1xi64>, memref<1xi64>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[VAL_10]] : memref<1xi64>)
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_CAS_triton_atomic_cas_1D(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<1xi64>, %arg3: memref<1xi64>, %arg4: memref<1xi64>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    hfusion.atomic_cas ins(%arg3, %arg4 : memref<1xi64>, memref<1xi64>) outs(%arg2 : memref<1xi64>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_CAS_triton_atomic_cas_fp8
// CHECK: %[[VAL_10:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[VAL_10]] : memref<1xi64>)
// CHECK-DAG: %[[VAL_12:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[VAL_11:.*]] : tensor<1xf8E4M3FN>) outs(%{{.*}} : tensor<1xf32>) -> tensor<1xf32>
// CHECK-DAG: %[[VAL_13:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%{{.*}} : tensor<1xf8E4M3FN>) outs(%{{.*}} : tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[VAL_14:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[VAL_12]], %[[VAL_13]] : tensor<1xf32>, tensor<1xf32>) outs(%{{.*}} : tensor<1xi1>) -> tensor<1xi1>
// CHECK: %[[VAL_15:.*]] = hfusion.select ins(%[[VAL_14]], %{{.*}}, %[[VAL_11]] : tensor<1xi1>, tensor<1xf8E4M3FN>, tensor<1xf8E4M3FN>) outs(%[[VAL_11]] : tensor<1xf8E4M3FN>) -> tensor<1xf8E4M3FN>
// CHECK: bufferization.materialize_in_destination %[[VAL_15]] in writable %{{.*}} : (tensor<1xf8E4M3FN>, memref<1xf8E4M3FN>) -> ()
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[VAL_10]] : memref<1xi64>)
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_CAS_triton_atomic_cas_fp8(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<1xf8E4M3FN>, %arg3: memref<1xf8E4M3FN>, %arg4: memref<1xf8E4M3FN>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    hfusion.atomic_cas ins(%arg3, %arg4 : memref<1xf8E4M3FN>, memref<1xf8E4M3FN>) outs(%arg2 : memref<1xf8E4M3FN>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_XCHG_atomic_xchg_2d_i16
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy %[[GM:.*]], %[[TMP:.*]] : memref<2x4xi16> to memref<2x4xi16>
// CHECK: memref.copy %[[UB:.*]], %[[GM]] : memref<2x4xi16> to memref<2x4xi16>
// CHECK: memref.copy %[[TMP]], %[[UB]] : memref<2x4xi16> to memref<2x4xi16>
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_XCHG_atomic_xchg_2d_i16(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<2x4xi16>, %arg3: memref<2x4xi16>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    hfusion.atomic_xchg ins(%arg3 : memref<2x4xi16>) outs(%arg2 : memref<2x4xi16>)
    return
  }
}

// -----

// CHECK-LABEL:   func.func @test_NormalizeAtomicOps_XCHG_triton_atomic_xchg_1D
// CHECK: %[[VAL_10:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[VAL_10]] : memref<1xi64>)
// CHECK: memref.copy %[[GM_SRC:.*]], %[[TMP_BUF:.*]] : memref<1xi64> to memref<1xi64>
// CHECK: memref.copy %[[UB_SRC:.*]], %[[GM_SRC]] : memref<1xi64> to memref<1xi64>
// CHECK: memref.copy %[[TMP_BUF]], %[[UB_SRC]] : memref<1xi64> to memref<1xi64>
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[VAL_10]] : memref<1xi64>)
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_XCHG_triton_atomic_xchg_1D(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<1xi64>, %arg3: memref<1xi64>, %arg4: memref<1xi64>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    hfusion.atomic_xchg ins(%arg3: memref<1xi64>) outs(%arg2 : memref<1xi64>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeSort_sort_i8
// CHECK: %[[CAST_F16:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<8xi8>) outs(%{{.*}} : tensor<8xf16>) -> tensor<8xf16>
// CHECK: %[[CAST_F32:.*]] = hfusion.cast {{.*}} ins(%[[CAST_F16]] : tensor<8xf16>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[SORT:.*]] = hfusion.sort ins(%[[CAST_F32]] : tensor<8xf32>) descending = true sort_axis = 0 -> tensor<8xf32>
// CHECK: %[[CAST_I32:.*]] = hfusion.cast {{.*}} ins(%[[SORT]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xi32>) -> tensor<8xi32>
// CHECK: %[[CAST_I8:.*]] = hfusion.cast {{.*}} ins(%[[CAST_I32]] : tensor<8xi32>) outs(%{{.*}} : tensor<8xi8>) -> tensor<8xi8>
func.func @test_NormalizeSort_sort_i8(%arg0 : tensor<8xi8>) -> tensor<8xi8> {
  %0 = hfusion.sort ins(%arg0 : tensor<8xi8>) descending = true sort_axis = 0 -> tensor<8xi8>
  return %0 : tensor<8xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeSort_sort_f16_keep_original
// CHECK: %[[SORT:.*]] = hfusion.sort ins(%arg0 : tensor<8xf16>) descending = false sort_axis = 0 -> tensor<8xf16>
// CHECK-NOT: hfusion.cast
func.func @test_NormalizeSort_sort_f16_keep_original(%arg0 : tensor<8xf16>) -> tensor<8xf16> {
  %0 = hfusion.sort ins(%arg0 : tensor<8xf16>) descending = false sort_axis = 0 -> tensor<8xf16>
  return %0 : tensor<8xf16>
}
