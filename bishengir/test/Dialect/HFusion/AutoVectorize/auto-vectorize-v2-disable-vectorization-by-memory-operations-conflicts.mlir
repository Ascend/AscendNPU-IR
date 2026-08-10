// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 -outline-vector-function -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @fused_recurrent_gated_delta_rule_update_fwd_kernel(%arg0
// CHECK: bufferization.materialize_in_destination
// CHECK: bufferization.materialize_in_destination
// CHECK: memref.copy %reinterpret_cast_17, %alloc_18
func.func @fused_recurrent_gated_delta_rule_update_fwd_kernel(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: i32, %arg3: i32, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1]>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1]>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1]>>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1]>>
  %reinterpret_cast_3 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1]>>
  %reinterpret_cast_4 = memref.reinterpret_cast %arg10 to offset: [0], sizes: [32, 4], strides: [32, 1] : memref<?xf32> to memref<32x4xf32, strided<[32, 1]>>
  %alloc = memref.alloc() : memref<32x4xf32>
  memref.copy %reinterpret_cast_4, %alloc : memref<32x4xf32, strided<[32, 1]>> to memref<32x4xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<32x4xf32>
  %alloc_5 = memref.alloc() : memref<32xf32>
  memref.copy %reinterpret_cast, %alloc_5 : memref<32xf32, strided<[1]>> to memref<32xf32>
  %1 = bufferization.to_tensor %alloc_5 restrict writable : memref<32xf32>
  %alloc_6 = memref.alloc() : memref<32xf32>
  memref.copy %reinterpret_cast_0, %alloc_6 : memref<32xf32, strided<[1]>> to memref<32xf32>
  %2 = bufferization.to_tensor %alloc_6 restrict writable : memref<32xf32>
  %reinterpret_cast_7 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
  %3 = memref.load %reinterpret_cast_7[%c0] : memref<1xf32, strided<[1]>>
  %alloc_8 = memref.alloc() : memref<4xf32>
  memref.copy %reinterpret_cast_1, %alloc_8 : memref<4xf32, strided<[1]>> to memref<4xf32>
  %4 = bufferization.to_tensor %alloc_8 restrict writable : memref<4xf32>
  %alloc_9 = memref.alloc() : memref<4xf32>
  memref.copy %reinterpret_cast_2, %alloc_9 : memref<4xf32, strided<[1]>> to memref<4xf32>
  %5 = bufferization.to_tensor %alloc_9 restrict writable : memref<4xf32>
  %6 = tensor.empty() : tensor<1xf32>
  %inserted = tensor.insert %3 into %6[%c0] : tensor<1xf32>
  %7 = tensor.empty() : tensor<32x4xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %inserted : tensor<32x4xf32>, tensor<1xf32>) outs(%7 : tensor<32x4xf32>) {
  ^bb0(%in: f32, %in_24: f32, %out: f32):
    %25 = math.exp %in_24 : f32
    %26 = arith.mulf %in, %25 : f32
    linalg.yield %26 : f32
  } -> tensor<32x4xf32>
  %9 = tensor.empty() : tensor<4xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%9 : tensor<4xf32>) {
  ^bb0(%out: f32):
    linalg.yield %cst : f32
  } -> tensor<4xf32>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"]} ins(%8, %2 : tensor<32x4xf32>, tensor<32xf32>) outs(%10 : tensor<4xf32>) {
  ^bb0(%in: f32, %in_24: f32, %out: f32):
    %25 = arith.mulf %in, %in_24 : f32
    %26 = arith.addf %25, %out : f32
    linalg.yield %26 : f32
  } -> tensor<4xf32>
  %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %2, %4, %11, %5 : tensor<32x4xf32>, tensor<32xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%7 : tensor<32x4xf32>) {
  ^bb0(%in: f32, %in_24: f32, %in_25: f32, %in_26: f32, %in_27: f32, %out: f32):
    %25 = arith.subf %in_25, %in_26 : f32
    %26 = arith.mulf %25, %in_27 : f32
    %27 = arith.mulf %in_24, %26 : f32
    %28 = arith.addf %in, %27 : f32
    linalg.yield %28 : f32
  } -> tensor<32x4xf32>
  %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"]} ins(%12, %1 : tensor<32x4xf32>, tensor<32xf32>) outs(%10 : tensor<4xf32>) {
  ^bb0(%in: f32, %in_24: f32, %out: f32):
    %25 = arith.mulf %in, %in_24 : f32
    %26 = arith.addf %25, %out : f32
    linalg.yield %26 : f32
  } -> tensor<4xf32>
  bufferization.materialize_in_destination %13 in writable %reinterpret_cast_3 : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
  %14 = arith.index_cast %arg2 : i32 to index
  %reinterpret_cast_10 = memref.reinterpret_cast %arg11 to offset: [%14], sizes: [32, 4], strides: [32, 1] : memref<?xf32> to memref<32x4xf32, strided<[32, 1], offset: ?>>
  bufferization.materialize_in_destination %12 in writable %reinterpret_cast_10 : (tensor<32x4xf32>, memref<32x4xf32, strided<[32, 1], offset: ?>>) -> ()
  %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [128], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: 128>>
  %reinterpret_cast_12 = memref.reinterpret_cast %arg5 to offset: [128], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: 128>>
  %reinterpret_cast_13 = memref.reinterpret_cast %arg9 to offset: [128], sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1], offset: 128>>
  %reinterpret_cast_14 = memref.reinterpret_cast %arg6 to offset: [128], sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1], offset: 128>>
  %reinterpret_cast_15 = memref.reinterpret_cast %arg8 to offset: [128], sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1], offset: 128>>
  %reinterpret_cast_16 = memref.reinterpret_cast %arg7 to offset: [4], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: 4>>
  %15 = arith.index_cast %arg3 : i32 to index
  %reinterpret_cast_17 = memref.reinterpret_cast %arg11 to offset: [%15], sizes: [32, 4], strides: [32, 1] : memref<?xf32> to memref<32x4xf32, strided<[32, 1], offset: ?>>
  %alloc_18 = memref.alloc() : memref<32x4xf32>
  memref.copy %reinterpret_cast_17, %alloc_18 : memref<32x4xf32, strided<[32, 1], offset: ?>> to memref<32x4xf32>
  %16 = bufferization.to_tensor %alloc_18 restrict writable : memref<32x4xf32>
  %alloc_19 = memref.alloc() : memref<32xf32>
  memref.copy %reinterpret_cast_11, %alloc_19 : memref<32xf32, strided<[1], offset: 128>> to memref<32xf32>
  %17 = bufferization.to_tensor %alloc_19 restrict writable : memref<32xf32>
  %alloc_20 = memref.alloc() : memref<32xf32>
  memref.copy %reinterpret_cast_12, %alloc_20 : memref<32xf32, strided<[1], offset: 128>> to memref<32xf32>
  %18 = bufferization.to_tensor %alloc_20 restrict writable : memref<32xf32>
  %19 = memref.load %reinterpret_cast_16[%c0] : memref<1xf32, strided<[1], offset: 4>>
%alloc_21 = memref.alloc() : memref<4xf32>
  memref.copy %reinterpret_cast_14, %alloc_21 : memref<4xf32, strided<[1], offset: 128>> to memref<4xf32>
  %20 = bufferization.to_tensor %alloc_21 restrict writable : memref<4xf32>
  %alloc_22 = memref.alloc() : memref<4xf32>
  memref.copy %reinterpret_cast_15, %alloc_22 : memref<4xf32, strided<[1], offset: 128>> to memref<4xf32>
  %21 = bufferization.to_tensor %alloc_22 restrict writable : memref<4xf32>
  %inserted_23 = tensor.insert %19 into %6[%c0] : tensor<1xf32>
  %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%16, %inserted_23 : tensor<32x4xf32>, tensor<1xf32>) outs(%7 : tensor<32x4xf32>) {
  ^bb0(%in: f32, %in_24: f32, %out: f32):
    %25 = math.exp %in_24 : f32
    %26 = arith.mulf %in, %25 : f32
    linalg.yield %26 : f32
  } -> tensor<32x4xf32>
  %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"]} ins(%22, %18 : tensor<32x4xf32>, tensor<32xf32>) outs(%10 : tensor<4xf32>) {
  ^bb0(%in: f32, %in_24: f32, %out: f32):
    %25 = arith.mulf %in, %in_24 : f32
    %26 = arith.addf %25, %out : f32
    linalg.yield %26 : f32
  } -> tensor<4xf32>
  %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"]} ins(%22, %18, %20, %23, %21, %17 : tensor<32x4xf32>, tensor<32xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<32xf32>) outs(%10 : tensor<4xf32>) {
  ^bb0(%in: f32, %in_24: f32, %in_25: f32, %in_26: f32, %in_27: f32, %in_28: f32, %out: f32):
    %25 = arith.subf %in_25, %in_26 : f32
    %26 = arith.mulf %25, %in_27 : f32
    %27 = arith.mulf %in_24, %26 : f32
    %28 = arith.addf %in, %27 : f32
    %29 = arith.mulf %28, %in_28 : f32
    %30 = arith.addf %29, %out : f32
    linalg.yield %30 : f32
  } -> tensor<4xf32>
  bufferization.materialize_in_destination %24 in writable %reinterpret_cast_13 : (tensor<4xf32>, memref<4xf32, strided<[1], offset: 128>>) -> ()
  return
}
