// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9589 --hfusion-auto-vectorize %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @gather_auto_vectorize
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @gather_auto_vectorize(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [15, 3], strides: [3, 1] : memref<?xf16> to memref<15x3xf16, strided<[3, 1]>>
    %alloc = memref.alloc() : memref<15x3xf16>
    memref.copy %reinterpret_cast, %alloc : memref<15x3xf16, strided<[3, 1]>> to memref<15x3xf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<15x3xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [15, 2], strides: [2, 1] : memref<?xi64> to memref<15x2xi64, strided<[2, 1]>>
    %alloc_1 = memref.alloc() : memref<15x2xi64>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<15x2xi64, strided<[2, 1]>> to memref<15x2xi64>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<15x2xi64>
    %2 = tensor.empty() : tensor<15x2xf16>
    %3 = tensor.empty() : tensor<15x2xi32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<15x2xi64>) outs(%3 : tensor<15x2xi32>) {
    ^bb0(%in: i64, %out: i32):
      %6 = arith.trunci %in {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : i64 to i32
      linalg.yield %6 : i32
    } -> tensor<15x2xi32>
    %5 = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%0, %4 : tensor<15x3xf16>, tensor<15x2xi32>) outs(%2 : tensor<15x2xf16>) axis = 1 -> tensor<15x2xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [15, 2], strides: [2, 1] : memref<?xf16> to memref<15x2xf16, strided<[2, 1]>>
    bufferization.materialize_in_destination %5 in writable %reinterpret_cast_2 : (tensor<15x2xf16>, memref<15x2xf16, strided<[2, 1]>>) -> ()
    return
  }
}
