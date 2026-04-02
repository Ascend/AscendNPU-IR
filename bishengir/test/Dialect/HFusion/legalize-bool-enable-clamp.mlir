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

// -----

// CHECK-LABEL: func.func @triton_sum_bool_reduce
func.func @triton_sum_bool_reduce(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [512], strides: [1] : memref<?xi8> to memref<512xi8, strided<[1]>>
  %alloc = memref.alloc() : memref<512xi8>
  memref.copy %reinterpret_cast, %alloc {was_bool_to_int8 = true} : memref<512xi8, strided<[1]>> to memref<512xi8>
  // CHECK: %[[SRC:.*]] = bufferization.to_tensor %alloc restrict writable {was_bool_to_int8 = true} : memref<512xi8>
  %0 = bufferization.to_tensor %alloc restrict writable {was_bool_to_int8 = true} : memref<512xi8>
  %1 = bufferization.alloc_tensor() : tensor<i8>
  %2 = linalg.fill ins(%c0_i8 : i8) outs(%1 : tensor<i8>) -> tensor<i8>
  %reduced = linalg.reduce ins(%0 : tensor<512xi8>) outs(%2 : tensor<i8>) dimensions = [0]
    (%in: i8, %init: i8) {
      // CHECK: %[[ADD:.*]] = arith.addi %in, %init : i8
      // CHECK: linalg.yield %[[ADD]] : i8
      %3 = arith.addi %in, %init : i8
      linalg.yield %3 : i8
    }
  %extracted = tensor.extract %reduced[] : tensor<i8>
  %3 = tensor.empty() : tensor<1xi8>
  %inserted = tensor.insert %extracted into %3[%c0] : tensor<1xi8>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1]>>
  bufferization.materialize_in_destination %inserted in writable %reinterpret_cast_0 : (tensor<1xi8>, memref<1xi8, strided<[1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @triton_sum_bool_reduce_after_math_abs
func.func @triton_sum_bool_reduce_after_math_abs(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [512], strides: [1] : memref<?xi8> to memref<512xi8, strided<[1]>>
  %alloc = memref.alloc() : memref<512xi8>
  memref.copy %reinterpret_cast, %alloc {was_bool_to_int8 = true} : memref<512xi8, strided<[1]>> to memref<512xi8>
  // CHECK: %[[MATH_SRC:.*]] = bufferization.to_tensor %alloc restrict writable {was_bool_to_int8 = true} : memref<512xi8>
  %0 = bufferization.to_tensor %alloc restrict writable {was_bool_to_int8 = true} : memref<512xi8>
  // CHECK: %[[MATH_ABS:.*]] = math.absi %[[MATH_SRC]] : tensor<512xi8>
  %1 = math.absi %0 : tensor<512xi8>
  %2 = bufferization.alloc_tensor() : tensor<i8>
  %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<i8>) -> tensor<i8>
  %reduced = linalg.reduce ins(%1 : tensor<512xi8>) outs(%3 : tensor<i8>) dimensions = [0]
    (%in: i8, %init: i8) {
      // CHECK: %[[MATH_ADD:.*]] = arith.addi %in, %init : i8
      // CHECK: linalg.yield %[[MATH_ADD]] : i8
      %4 = arith.addi %in, %init : i8
      linalg.yield %4 : i8
    }
  %extracted = tensor.extract %reduced[] : tensor<i8>
  %4 = tensor.empty() : tensor<1xi8>
  %inserted = tensor.insert %extracted into %4[%c0] : tensor<1xi8>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1]>>
  bufferization.materialize_in_destination %inserted in writable %reinterpret_cast_0 : (tensor<1xi8>, memref<1xi8, strided<[1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @triton_sum_bool_reduce_after_hfusion_abs
func.func @triton_sum_bool_reduce_after_hfusion_abs(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [512], strides: [1] : memref<?xi8> to memref<512xi8, strided<[1]>>
  %alloc = memref.alloc() : memref<512xi8>
  memref.copy %reinterpret_cast, %alloc {was_bool_to_int8 = true} : memref<512xi8, strided<[1]>> to memref<512xi8>
  // CHECK: %[[HF_SRC:.*]] = bufferization.to_tensor %alloc restrict writable {was_bool_to_int8 = true} : memref<512xi8>
  %0 = bufferization.to_tensor %alloc restrict writable {was_bool_to_int8 = true} : memref<512xi8>
  %1 = bufferization.alloc_tensor() : tensor<512xi8>
  // CHECK: %[[HF_ABS:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%[[HF_SRC]] : tensor<512xi8>) outs(%{{.*}} : tensor<512xi8>) -> tensor<512xi8>
  %2 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%0 : tensor<512xi8>) outs(%1 : tensor<512xi8>) -> tensor<512xi8>
  %3 = bufferization.alloc_tensor() : tensor<i8>
  %4 = linalg.fill ins(%c0_i8 : i8) outs(%3 : tensor<i8>) -> tensor<i8>
  %reduced = linalg.reduce ins(%2 : tensor<512xi8>) outs(%4 : tensor<i8>) dimensions = [0]
    (%in: i8, %init: i8) {
      // CHECK: %[[HF_ADD:.*]] = arith.addi %in, %init : i8
      // CHECK: linalg.yield %[[HF_ADD]] : i8
      %5 = arith.addi %in, %init : i8
      linalg.yield %5 : i8
    }
  %extracted = tensor.extract %reduced[] : tensor<i8>
  %5 = tensor.empty() : tensor<1xi8>
  %inserted = tensor.insert %extracted into %5[%c0] : tensor<1xi8>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1]>>
  bufferization.materialize_in_destination %inserted in writable %reinterpret_cast_0 : (tensor<1xi8>, memref<1xi8, strided<[1]>>) -> ()
  return
}
