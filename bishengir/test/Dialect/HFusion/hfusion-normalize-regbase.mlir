// RUN: bishengir-opt --hfusion-normalize-ops -split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_broadcast(
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.fill
// CHECK: linalg.fill
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_broadcast(%arg0: tensor<23xi64>, %arg1: tensor<f16>) -> tensor<23xf16> attributes {hacc.entry} {
    %cst = arith.constant 0.86956521739130432 : f64
    %0 = tensor.empty() : tensor<23xf16>
    %1 = tensor.empty() : tensor<23xf32>
    %2 = hfusion.cast {enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<23xi64>) outs(%1 : tensor<23xf32>) -> tensor<23xf32>
    %3 = hfusion.cast {enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%2 : tensor<23xf32>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<f16>) outs(%0 : tensor<23xf16>) dimensions = [0]
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%3, %broadcasted : tensor<23xf16>, tensor<23xf16>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %5 = arith.truncf %cst : f64 to f16
    %6 = linalg.fill ins(%5 : f16) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %6 : tensor<23xf16>, tensor<23xf16>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    return %7 : tensor<23xf16>
  }
}

// -----
// CHECK-LABEL: @test_broadcast_2(
// CHECK-NOT: linalg.broadcast
// CHECK: %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_6:.*]] = tensor.extract %{{.*}}{{\[}}%[[VAL_2]], %[[VAL_2]], %[[VAL_2]]] : tensor<1x1x1xf16>
// CHECK: %[[VAL_7:.*]] = linalg.fill ins(%[[VAL_6]] : f16) outs(%{{.*}} : tensor<1x1x23x1xf16>) -> tensor<1x1x23x1xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_broadcast_2(%arg0: tensor<1x1x23x1xf32>, %arg1: tensor<1x1x1xf16>) -> tensor<1x1x23x1xf16> attributes {hacc.entry} {
    %cst = arith.constant 0.86956521739130432 : f64
    %0 = tensor.empty() : tensor<1x1x23x1xf16>
    %1 = tensor.empty() : tensor<1x1x23x1xf32>
    %3 = hfusion.cast {enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<1x1x23x1xf32>) outs(%0 : tensor<1x1x23x1xf16>) -> tensor<1x1x23x1xf16>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<1x1x1xf16>) outs(%0 : tensor<1x1x23x1xf16>) dimensions = [2]
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%3, %broadcasted : tensor<1x1x23x1xf16>, tensor<1x1x23x1xf16>) outs(%0 : tensor<1x1x23x1xf16>) -> tensor<1x1x23x1xf16>
    %5 = arith.truncf %cst : f64 to f16
    %6 = linalg.fill ins(%5 : f16) outs(%0 : tensor<1x1x23x1xf16>) -> tensor<1x1x23x1xf16>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %6 : tensor<1x1x23x1xf16>, tensor<1x1x23x1xf16>) outs(%0 : tensor<1x1x23x1xf16>) -> tensor<1x1x23x1xf16>
    return %7 : tensor<1x1x23x1xf16>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL:   func.func @triton_atomic_cas_1D
  func.func @triton_atomic_cas_1D(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<1xi64>, %arg3: memref<1xi64>, %arg4: memref<1xi64>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    hfusion.atomic_cas ins(%arg3, %arg4 : memref<1xi64>, memref<1xi64>) outs(%arg2 : memref<1xi64>)
    // CHECK:           %[[VAL_10:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
    // CHECK:           hivm.hir.sync_block_lock lock_var(%[[VAL_10]] : memref<1xi64>)
    // CHECK:           %[[VAL_14:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[VAL_12:.*]], %{{.*}} : tensor<1xi64>, tensor<1xi64>) outs(%{{.*}} : tensor<1xi1>) -> tensor<1xi1>
    // CHECK:           %[[VAL_15:.*]] = hfusion.select ins(%[[VAL_14]], %{{.*}}, %[[VAL_12]] : tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) outs(%{{.*}} : tensor<1xi64>) -> tensor<1xi64>
    // CHECK:           bufferization.materialize_in_destination %[[VAL_15]] in writable %{{.*}} : (tensor<1xi64>, memref<1xi64>) -> ()
    // CHECK:           hivm.hir.sync_block_unlock lock_var(%[[VAL_10]] : memref<1xi64>)
    // CHECK:           return
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL:   func.func @triton_atomic_xchg_1D
  func.func @triton_atomic_xchg_1D(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<1xi64>, %arg3: memref<1xi64>, %arg4: memref<1xi64>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    hfusion.atomic_xchg ins(%arg3: memref<1xi64>) outs(%arg2 : memref<1xi64>)
    // CHECK:           %[[VAL_10:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
    // CHECK:           hivm.hir.sync_block_lock lock_var(%[[VAL_10]] : memref<1xi64>)
    // CHECK:           memref.copy %[[GM_SRC:.*]], %[[TMP_BUF:.*]] : memref<1xi64> to memref<1xi64>
    // CHECK:           memref.copy %[[UB_SRC:.*]], %[[GM_SRC]] : memref<1xi64> to memref<1xi64>
    // CHECK:           memref.copy %[[TMP_BUF]], %[[UB_SRC]] : memref<1xi64> to memref<1xi64>
    // CHECK:           hivm.hir.sync_block_unlock lock_var(%[[VAL_10]] : memref<1xi64>)
    // CHECK:           return
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_atomic_and
  func.func @test_atomic_and(%arg0 : memref<?xi16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi16>) {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %c256_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1], offset: ?>>
    %1 = bufferization.to_memref %arg1 : memref<256xi16, strided<[1]>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<and>} ins(%1 : memref<256xi16, strided<[1]>>) outs(%reinterpret_cast : memref<256xi16, strided<[1], offset: ?>>)

    // CHECK:           %[[VAL_9:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
    // CHECK:           hivm.hir.sync_block_lock lock_var(%[[VAL_9]] : memref<1xi64>)
    // CHECK:           memref.copy %[[VAL_4:.*]], %[[VAL_6:.*]] : memref<256xi16, strided<[1], offset: ?>> to memref<256xi16>
    // CHECK:           %[[VAL_10:.*]] = bufferization.to_tensor %[[VAL_6]] restrict writable : memref<256xi16>
    // CHECK:           %[[VAL_11:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[VAL_10]], %{{.*}} : tensor<256xi16>, tensor<256xi16>) outs(%{{.*}} : tensor<256xi16>) -> tensor<256xi16>
    // CHECK:           bufferization.materialize_in_destination %[[VAL_11]] in writable %[[VAL_4]] : (tensor<256xi16>, memref<256xi16, strided<[1], offset: ?>>) -> ()
    // CHECK:           hivm.hir.sync_block_unlock lock_var(%[[VAL_9]] : memref<1xi64>)
    
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_atomic_add_i64
  func.func @test_atomic_add_i64(%arg0 : memref<?xi64> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi64>) {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %c256_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256], strides: [1] : memref<?xi64> to memref<256xi64, strided<[1], offset: ?>>
    %1 = bufferization.to_memref %arg1 : memref<256xi64, strided<[1]>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<add>} ins(%1 : memref<256xi64, strided<[1]>>) outs(%reinterpret_cast : memref<256xi64, strided<[1], offset: ?>>)

    // CHECK:           %[[VAL_9:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
    // CHECK:           hivm.hir.sync_block_lock lock_var(%[[VAL_9]] : memref<1xi64>)
    // CHECK:           memref.copy %[[VAL_4:.*]], %[[VAL_6:.*]] : memref<256xi64, strided<[1], offset: ?>> to memref<256xi64>
    // CHECK:           %[[VAL_10:.*]] = bufferization.to_tensor %[[VAL_6]] restrict writable : memref<256xi64>
    // CHECK:           %[[VAL_11:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_10]], %{{.*}} : tensor<256xi64>, tensor<256xi64>) outs(%{{.*}} : tensor<256xi64>) -> tensor<256xi64>
    // CHECK:           bufferization.materialize_in_destination %[[VAL_11]] in writable %[[VAL_4]] : (tensor<256xi64>, memref<256xi64, strided<[1], offset: ?>>) -> ()
    // CHECK:           hivm.hir.sync_block_unlock lock_var(%[[VAL_9]] : memref<1xi64>)
    
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_atomic_add_fp8
  func.func @test_atomic_add_fp8(%arg0 : memref<?xf8E4M3FN> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xf8E4M3FN>) {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %c256_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256], strides: [1] : memref<?xf8E4M3FN> to memref<256xf8E4M3FN, strided<[1], offset: ?>>
    %1 = bufferization.to_memref %arg1 : memref<256xf8E4M3FN, strided<[1]>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<add>} ins(%1 : memref<256xf8E4M3FN, strided<[1]>>) outs(%reinterpret_cast : memref<256xf8E4M3FN, strided<[1], offset: ?>>)

    // CHECK:           %[[VAL_9:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
    // CHECK:           hivm.hir.sync_block_lock lock_var(%[[VAL_9]] : memref<1xi64>)
    // CHECK:           memref.copy %[[VAL_4:.*]], %[[VAL_6:.*]] : memref<256xf8E4M3FN, strided<[1], offset: ?>> to memref<256xf8E4M3FN>
    // CHECK:           %[[VAL_11:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%{{.*}} : tensor<256xf8E4M3FN>) outs(%{{.*}} : tensor<256xf32>) -> tensor<256xf32>
    // CHECK:           %[[VAL_12:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_11]], %{{.*}} : tensor<256xf32>, tensor<256xf32>) outs(%{{.*}} : tensor<256xf32>) -> tensor<256xf32>
    // CHECK:           %[[VAL_13:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_12]] : tensor<256xf32>) outs(%{{.*}} : tensor<256xf8E4M3FN>) -> tensor<256xf8E4M3FN>
    // CHECK:           bufferization.materialize_in_destination %[[VAL_13]] in writable %[[VAL_4]] : (tensor<256xf8E4M3FN>, memref<256xf8E4M3FN, strided<[1], offset: ?>>) -> ()
    // CHECK:           hivm.hir.sync_block_unlock lock_var(%[[VAL_9]] : memref<1xi64>)
    
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL:   func.func @triton_atomic_cas_fp8
  func.func @triton_atomic_cas_fp8(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<1xf8E4M3FN>, %arg3: memref<1xf8E4M3FN>, %arg4: memref<1xf8E4M3FN>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    hfusion.atomic_cas ins(%arg3, %arg4 : memref<1xf8E4M3FN>, memref<1xf8E4M3FN>) outs(%arg2 : memref<1xf8E4M3FN>)
    // CHECK:           %[[VAL_10:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
    // CHECK:           hivm.hir.sync_block_lock lock_var(%[[VAL_10]] : memref<1xi64>)
    // CHECK-DAG:       %[[VAL_12:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_11:.*]] : tensor<1xf8E4M3FN>) outs(%{{.*}} : tensor<1xf32>) -> tensor<1xf32>
    // CHECK-DAG:       %[[VAL_13:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%{{.*}} : tensor<1xf8E4M3FN>) outs(%{{.*}} : tensor<1xf32>) -> tensor<1xf32>
    // CHECK:           %[[VAL_14:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[VAL_12]], %[[VAL_13]] : tensor<1xf32>, tensor<1xf32>) outs(%{{.*}} : tensor<1xi1>) -> tensor<1xi1>
    // CHECK:           %[[VAL_15:.*]] = hfusion.select ins(%[[VAL_14]], %{{.*}}, %[[VAL_11]] : tensor<1xi1>, tensor<1xf8E4M3FN>, tensor<1xf8E4M3FN>) outs(%[[VAL_11]] : tensor<1xf8E4M3FN>) -> tensor<1xf8E4M3FN>
    // CHECK:           bufferization.materialize_in_destination %[[VAL_15]] in writable %{{.*}} : (tensor<1xf8E4M3FN>, memref<1xf8E4M3FN>) -> ()
    // CHECK:           hivm.hir.sync_block_unlock lock_var(%[[VAL_10]] : memref<1xi64>)
    // CHECK:           return
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @test_atomic_dyn_and
  func.func @test_atomic_dyn_and(%arg0 : memref<?xi16> {tt.divisibility = 16 : i32}, %arg1 : memref<?xi16> {tt.divisibility = 16 : i32}, %arg2: index, %arg3: index, %arg4: index) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg4], sizes: [12, 256], strides: [512, 1] : memref<?xi16> to memref<12x256xi16, strided<[512, 1], offset: ?>>
    %alloc = memref.alloc() : memref<12x256xi16>
    %subview = memref.subview %reinterpret_cast[0, 0] [%arg2, %arg3] [1, 1] : memref<12x256xi16, strided<[512, 1], offset: ?>> to memref<?x?xi16, strided<[512, 1], offset: ?>>
    %subview_0 = memref.subview %alloc[0, 0] [%arg2, %arg3] [1, 1] : memref<12x256xi16> to memref<?x?xi16, strided<[256, 1]>>
    memref.copy %subview, %subview_0 : memref<?x?xi16, strided<[512, 1], offset: ?>> to memref<?x?xi16, strided<[256, 1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%arg4], sizes: [12, 256], strides: [512, 1] : memref<?xi16> to memref<12x256xi16, strided<[512, 1], offset: ?>>
    %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%arg2, %arg3] [1, 1] : memref<12x256xi16, strided<[512, 1], offset: ?>> to memref<?x?xi16, strided<[512, 1], offset: ?>>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<and>} ins(%subview_0 : memref<?x?xi16, strided<[256, 1]>>) outs(%subview_2 : memref<?x?xi16, strided<[512, 1], offset: ?>>)

    // CHECK:           %[[VAL_9:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
    // CHECK:           hivm.hir.sync_block_lock lock_var(%[[VAL_9]] : memref<1xi64>)
    // CHECK:           memref.copy %[[VAL_4:.*]], %[[VAL_6:.*]] : memref<?x?xi16, strided<[512, 1], offset: ?>> to memref<?x?xi16, strided<[256, 1]>>
    // CHECK:           %[[VAL_11:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%{{.*}}, %{{.*}} : tensor<12x256xi16>, tensor<12x256xi16>) outs(%{{.*}} : tensor<12x256xi16>) -> tensor<12x256xi16>
    // CHECK:           %[[VAL_12:.*]] = tensor.extract_slice %[[VAL_11:.*]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : tensor<12x256xi16> to tensor<?x?xi16>
    // CHECK:           bufferization.materialize_in_destination %[[VAL_12]] in writable %[[VAL_4]] : (tensor<?x?xi16>, memref<?x?xi16, strided<[512, 1], offset: ?>>) -> ()
    // CHECK:           hivm.hir.sync_block_unlock lock_var(%[[VAL_9]] : memref<1xi64>)
    
    return
  }
}

// -----
// CHECK-LABEL: func.func @cast_to_nd_with_overflow
func.func @cast_to_nd_with_overflow(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty() : tensor<300xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<300xi64>) -> tensor<300xi64>
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [300], strides: [1] : memref<?xi64> to memref<300xi64, strided<[1]>>
  %alloc = memref.alloc() : memref<300xi64>
  memref.copy %reinterpret_cast, %alloc : memref<300xi64, strided<[1]>> to memref<300xi64>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<300xi64>
  %3 = tensor.empty() : tensor<300xi1>
  %4 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%2, %1 : tensor<300xi64>, tensor<300xi64>) outs(%3 : tensor<300xi1>) -> tensor<300xi1>
  // CHECK-NOT: annotation.mark
  annotation.mark %4 {overflow_mode = "trunc"} : tensor<300xi1>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [300], strides: [1] : memref<?xi8> to memref<300xi8, strided<[1]>>
  %5 = tensor.empty() : tensor<300xi8>
  %6 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<300xi1>) outs(%5 : tensor<300xi8>) -> tensor<300xi8>
  bufferization.materialize_in_destination %6 in writable %reinterpret_cast_0 : (tensor<300xi8>, memref<300xi8, strided<[1]>>) -> ()
  return
}
