// RUN: bishengir-opt --canonicalize-tensor-reshape -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @test_broadcast(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<32xf32>, %[[VAL_1:.*]]: memref<4096xf32>
// CHECK:           %[[VAL_2:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [32], strides: [1] : memref<32xf32> to memref<32xf32, strided<[1]>>
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<32xf32>
// CHECK:           memref.copy %[[VAL_2]], %[[VAL_3]] : memref<32xf32, strided<[1]>> to memref<32xf32>
// CHECK:           %[[VAL_4:.*]] = bufferization.to_tensor %[[VAL_3]] restrict writable : memref<32xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.expand_shape %[[VAL_4]] {{\[}}[0, 1, 2]] output_shape {{\[}}1, 4, 8] : tensor<32xf32> into tensor<1x4x8xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<128x4x8xf32>
// CHECK:           %[[VAL_7:.*]] = tensor.collapse_shape %[[VAL_5]] {{\[}}[0, 1], [2]] : tensor<1x4x8xf32> into tensor<4x8xf32>
// CHECK:           %[[VAL_8:.*]] = linalg.broadcast ins(%[[VAL_7]] : tensor<4x8xf32>) outs(%1 : tensor<128x4x8xf32>) dimensions = [0]
// CHECK:           %[[VAL_9:.*]] = tensor.collapse_shape %[[VAL_8]] {{\[}}[0, 1, 2]] : tensor<128x4x8xf32> into tensor<4096xf32>
// CHECK:           %[[VAL_10:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [4096], strides: [1] : memref<4096xf32> to memref<4096xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[VAL_9]] in writable %[[VAL_10]] : (tensor<4096xf32>, memref<4096xf32, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }
module{
func.func @test_broadcast(%arg0: memref<32xf32> , %arg1: memref<4096xf32>) {
      %c4096_i64 = arith.constant 4096 : i64
      %cst = arith.constant dense<[1, 4, 8]> : tensor<3xi64>
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32], strides: [1] : memref<32xf32> to memref<32xf32, strided<[1]>>
      %alloc = memref.alloc() : memref<32xf32>
      memref.copy %reinterpret_cast, %alloc : memref<32xf32, strided<[1]>> to memref<32xf32>
      %0 = bufferization.to_tensor %alloc restrict writable : memref<32xf32>
      %reshape = tensor.reshape %0(%cst) : (tensor<32xf32>, tensor<3xi64>) -> tensor<1x4x8xf32>
      %1 = tensor.empty() : tensor<128x4x8xf32>
      %collapsed = tensor.collapse_shape %reshape [[0, 1], [2]] : tensor<1x4x8xf32> into tensor<4x8xf32>
      %broadcasted = linalg.broadcast ins(%collapsed : tensor<4x8xf32>) outs(%1 : tensor<128x4x8xf32>) dimensions = [0]
      %2 = tensor.empty() : tensor<1xi64>
      %3 = linalg.fill ins(%c4096_i64 : i64) outs(%2 : tensor<1xi64>) -> tensor<1xi64>
      %reshape_0 = tensor.reshape %broadcasted(%3) : (tensor<128x4x8xf32>, tensor<1xi64>) -> tensor<4096xf32>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4096], strides: [1] : memref<4096xf32> to memref<4096xf32, strided<[1]>>
      bufferization.materialize_in_destination %reshape_0 in writable %reinterpret_cast_1 : (tensor<4096xf32>, memref<4096xf32, strided<[1]>>) -> ()
      return
    }
}




