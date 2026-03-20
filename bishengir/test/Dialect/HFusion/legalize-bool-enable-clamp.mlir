// RUN: bishengir-opt -hfusion-legalize-bool="enable-clamp=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @triton_pw_rdc5d
func.func @triton_pw_rdc5d(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %c0_i8 = arith.constant 0 : i8
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [9, 3, 2, 4, 17], strides: [408, 136, 68, 17, 1] : memref<?xi8> to memref<9x3x2x4x17xi8, strided<[408, 136, 68, 17, 1]>>
  %alloc = memref.alloc() : memref<9x3x2x4x17xi8>
  memref.copy %reinterpret_cast, %alloc {was_bool_to_int8 = true} : memref<9x3x2x4x17xi8, strided<[408, 136, 68, 17, 1]>> to memref<9x3x2x4x17xi8>
  %0 = bufferization.to_tensor %alloc restrict writable {was_bool_to_int8 = true} : memref<9x3x2x4x17xi8>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [9, 3, 2, 4, 17], strides: [408, 136, 68, 17, 1] : memref<?xi8> to memref<9x3x2x4x17xi8, strided<[408, 136, 68, 17, 1]>>
  %alloc_1 = memref.alloc() : memref<9x3x2x4x17xi8>
  memref.copy %reinterpret_cast_0, %alloc_1 {was_bool_to_int8 = true} : memref<9x3x2x4x17xi8, strided<[408, 136, 68, 17, 1]>> to memref<9x3x2x4x17xi8>
  %1 = bufferization.to_tensor %alloc_1 restrict writable {was_bool_to_int8 = true} : memref<9x3x2x4x17xi8>
  
  // CHECK: %[[C0_I32:.*]] = arith.constant 0 : i32
  // CHECK: %[[C0_I8:.*]] = arith.constant 0 : i8
  // CHECK: %[[TENSOR_0:.*]] = bufferization.to_tensor
  // CHECK: %[[TENSOR_1:.*]] = bufferization.to_tensor
  
  // CHECK: %[[ADD:.*]] = arith.addi %[[TENSOR_0]], %[[TENSOR_1]] {is_clamped = true} : tensor<9x3x2x4x17xi8>
  %2 = arith.addi %0, %1 : tensor<9x3x2x4x17xi8>
  
  // CHECK: %[[EXTSI:.*]] = arith.extsi %[[ADD]] : tensor<9x3x2x4x17xi8> to tensor<9x3x2x4x17xi32>
  // CHECK: %[[EMPTY_I32:.*]] = tensor.empty() : tensor<9x3x2x4x17xi32>
  // CHECK: %[[FILL_I32:.*]] = linalg.fill ins(%[[C0_I32]] : i32) outs(%[[EMPTY_I32]] : tensor<9x3x2x4x17xi32>) -> tensor<9x3x2x4x17xi32>
  // CHECK: %[[CMPI:.*]] = arith.cmpi ne, %[[EXTSI]], %[[FILL_I32]] : tensor<9x3x2x4x17xi32>
  
  // CHECK: %[[EXTUI:.*]] = arith.extui %[[CMPI]] {was_bool_to_int8 = true} : tensor<9x3x2x4x17xi1> to tensor<9x3x2x4x17xi8>
  
  %3 = tensor.empty() : tensor<9x3x2x4xi8>
  %4 = linalg.fill ins(%c0_i8 : i8) outs(%3 : tensor<9x3x2x4xi8>) -> tensor<9x3x2x4xi8>
  
  // CHECK: %[[REDUCED:.*]] = linalg.reduce ins(%[[EXTUI]] : tensor<9x3x2x4x17xi8>)
  %reduced = linalg.reduce ins(%2 : tensor<9x3x2x4x17xi8>) outs(%4 : tensor<9x3x2x4xi8>) dimensions = [4] 
    (%in: i8, %init: i8) {
      %5 = arith.xori %in, %init : i8
      linalg.yield %5 : i8
    }
  %expanded = tensor.expand_shape %reduced [[0], [1], [2], [3, 4]] output_shape [9, 3, 2, 4, 1] : tensor<9x3x2x4xi8> into tensor<9x3x2x4x1xi8>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [9, 3, 2, 4, 1], strides: [24, 8, 4, 1, 1] : memref<?xi8> to memref<9x3x2x4x1xi8, strided<[24, 8, 4, 1, 1]>>
  bufferization.materialize_in_destination %expanded in writable %reinterpret_cast_2 : (tensor<9x3x2x4x1xi8>, memref<9x3x2x4x1xi8, strided<[24, 8, 4, 1, 1]>>) -> ()
  return
}