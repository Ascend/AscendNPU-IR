// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9589 --hfusion-decompose="hfusion-decompose-phase=after-hfusion-flatten" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: triton_gather_mapfor_not_to_forall
func.func @triton_gather_mapfor_not_to_forall(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i32, WorkspaceArgIdx = 1 : i32, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [5, 11], strides: [11, 1] : memref<?xf16> to memref<5x11xf16, strided<[11, 1]>>
  %alloc = memref.alloc() : memref<5x11xf16>
  memref.copy %reinterpret_cast, %alloc : memref<5x11xf16, strided<[11, 1]>> to memref<5x11xf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<5x11xf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [7, 11], strides: [11, 1] : memref<?xi32> to memref<7x11xi32, strided<[11, 1]>>
  %alloc_1 = memref.alloc() : memref<7x11xi32>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<7x11xi32, strided<[11, 1]>> to memref<7x11xi32>
  %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<7x11xi32>
  %2 = tensor.empty() : tensor<7x11xf16>
  // CHECK:scf.for
  // CHECK-NOT:map_for_to_forall
  %3 = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%0, %1 : tensor<5x11xf16>, tensor<7x11xi32>) outs(%2 : tensor<7x11xf16>) axis = 0 -> tensor<7x11xf16>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [7, 11], strides: [11, 1] : memref<?xf16> to memref<7x11xf16, strided<[11, 1]>>
  bufferization.materialize_in_destination %3 in writable %reinterpret_cast_2 : (tensor<7x11xf16>, memref<7x11xf16, strided<[11, 1]>>) -> ()
  return
}