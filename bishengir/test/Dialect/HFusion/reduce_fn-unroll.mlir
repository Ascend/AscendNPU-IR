// RUN: bishengir-opt -hfusion-generic-unroller %s | FileCheck %s

// CHECK-LABEL: func.func @reduce_fn_and_unroll
// CHECK-NOT: linalg.reduce
func.func @reduce_fn_and_unroll(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1_i64 = arith.constant 1 : i64
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [2, 1], strides: [1, 1] : memref<?xi64> to memref<2x1xi64, strided<[1, 1]>>
  %alloc = memref.alloc() : memref<2x1xi64>
  memref.copy %reinterpret_cast, %alloc : memref<2x1xi64, strided<[1, 1]>> to memref<2x1xi64>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<2x1xi64>
  %1 = tensor.empty() : tensor<1xi64>
  %2 = linalg.fill ins(%c1_i64 : i64) outs(%1 : tensor<1xi64>) -> tensor<1xi64>
  %reduced = linalg.reduce ins(%0 : tensor<2x1xi64>) outs(%2 : tensor<1xi64>) dimensions = [0]
    (%in: i64, %init: i64) {
      %3 = arith.andi %in, %init : i64
      linalg.yield %3 : i64
    }
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
  bufferization.materialize_in_destination %reduced in writable %reinterpret_cast_0 : (tensor<1xi64>, memref<1xi64, strided<[1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @reduce_fn_or_unroll
// CHECK-NOT: linalg.reduce
func.func @reduce_fn_or_unroll(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [2, 1], strides: [1, 1] : memref<?xi64> to memref<2x1xi64, strided<[1, 1]>>
  %alloc = memref.alloc() : memref<2x1xi64>
  memref.copy %reinterpret_cast, %alloc : memref<2x1xi64, strided<[1, 1]>> to memref<2x1xi64>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<2x1xi64>
  %1 = tensor.empty() : tensor<1xi64>
  %2 = linalg.fill ins(%c0_i64 : i64) outs(%1 : tensor<1xi64>) -> tensor<1xi64>
  %reduced = linalg.reduce ins(%0 : tensor<2x1xi64>) outs(%2 : tensor<1xi64>) dimensions = [0]
    (%in: i64, %init: i64) {
      %3 = arith.ori %in, %init : i64
      linalg.yield %3 : i64
    }
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
  bufferization.materialize_in_destination %reduced in writable %reinterpret_cast_0 : (tensor<1xi64>, memref<1xi64, strided<[1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @reduce_fn_mul_unroll
// CHECK-NOT: linalg.reduce
func.func @reduce_fn_mul_unroll(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c0 = arith.constant 0 : index
  %c1_i16 = arith.constant 1 : i16
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [32], strides: [1] : memref<?xi16> to memref<32xi16, strided<[1]>>
  %alloc = memref.alloc() : memref<32xi16>
  memref.copy %reinterpret_cast, %alloc : memref<32xi16, strided<[1]>> to memref<32xi16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<32xi16>
  %1 = bufferization.alloc_tensor() : tensor<i16>
  %2 = linalg.fill ins(%c1_i16 : i16) outs(%1 : tensor<i16>) -> tensor<i16>
  %reduced = linalg.reduce ins(%0 : tensor<32xi16>) outs(%2 : tensor<i16>) dimensions = [0]
    (%in: i16, %init: i16) {
      %4 = arith.muli %in, %init : i16
      linalg.yield %4 : i16
    }
  %extracted = tensor.extract %reduced[] : tensor<i16>
  %3 = tensor.empty() : tensor<1xi16>
  %inserted = tensor.insert %extracted into %3[%c0] : tensor<1xi16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi16> to memref<1xi16, strided<[1]>>
  bufferization.materialize_in_destination %inserted in writable %reinterpret_cast_0 : (tensor<1xi16>, memref<1xi16, strided<[1]>>) -> ()
  return
}
