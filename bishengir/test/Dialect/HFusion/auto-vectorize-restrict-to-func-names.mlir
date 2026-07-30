// RUN: bishengir-opt %s -hfusion-auto-vectorize="restrict-to-func-names=funcB" | FileCheck %s
// RUN: bishengir-opt %s -hfusion-auto-vectorize="restrict-to-func-names=funcB" | FileCheck %s --check-prefix=SCOPED

// Regression test for the "restrict-to-func-names" option added to the
// legacy AutoVectorize pass: without it, AutoVectorizeV2's
// fallback reprocessed *every* vectorizable function in the module with the
// legacy pass, including ones it had already vectorized itself.
//
// With restrict-to-func-names=funcB, @funcA must be left completely untouched
// (still its original linalg.generic, no outlined function for it) while
// @funcB gets vectorized and outlined as usual.

func.func @funcA(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1]>>
  %alloc = memref.alloc() : memref<64xf32>
  memref.copy %reinterpret_cast, %alloc : memref<64xf32, strided<[1]>> to memref<64xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<64xf32>
  %1 = tensor.empty() : tensor<64xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<64xf32>) outs(%1 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %in : f32
    linalg.yield %3 : f32
  } -> tensor<64xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<64xf32>, memref<64xf32, strided<[1]>>) -> ()
  return
}

func.func @funcB(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1]>>
  %alloc = memref.alloc() : memref<64xf32>
  memref.copy %reinterpret_cast, %alloc : memref<64xf32, strided<[1]>> to memref<64xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<64xf32>
  %1 = tensor.empty() : tensor<64xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<64xf32>) outs(%1 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %in : f32
    linalg.yield %3 : f32
  } -> tensor<64xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<64xf32>, memref<64xf32, strided<[1]>>) -> ()
  return
}

// CHECK-LABEL: func.func @funcA(
// CHECK: linalg.generic
// CHECK-LABEL: func.func @funcB_outlined_vf_0(
// CHECK-LABEL: func.func @funcB(

// SCOPED-NOT: funcA_outlined_vf
