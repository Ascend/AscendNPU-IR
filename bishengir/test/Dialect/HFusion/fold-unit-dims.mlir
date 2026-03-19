
// RUN: bishengir-opt -hfusion-fold-unit-dims -cse -canonicalize -split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: triton_ldst_and_totensor_01
// CHECK: memref.alloc() : memref<5x50x2xi16>
// CHECK: memref.copy
// CHECK-SAME: memref<5x50x2xi16, strided<[100, 2, 1]>> to memref<5x50x2xi16>
// CHECK: bufferization.to_tensor
// CHECK-SAME: memref<5x50x2xi16>
// CHECK: bufferization.materialize_in_destination
// CHECK-SAME: (tensor<5x50x2xi32>, memref<5x50x2xi32, strided<[100, 2, 1]>>)
func.func @triton_ldst_and_totensor_01(%arg2: memref<?xi16>, %arg3: memref<?xi32>) {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [5, 50, 2, 1], strides: [100, 2, 1, 1] : memref<?xi16> to memref<5x50x2x1xi16, strided<[100, 2, 1, 1]>>
  %alloc = memref.alloc() : memref<5x50x2x1xi16>
  memref.copy %reinterpret_cast, %alloc : memref<5x50x2x1xi16, strided<[100, 2, 1, 1]>> to memref<5x50x2x1xi16>
  %3 = bufferization.to_tensor %alloc restrict writable : memref<5x50x2x1xi16>
  %4 = arith.extsi %3 : tensor<5x50x2x1xi16> to tensor<5x50x2x1xi32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [5, 50, 2, 1], strides: [100, 2, 1, 1] : memref<?xi32> to memref<5x50x2x1xi32, strided<[100, 2, 1, 1]>>
  bufferization.materialize_in_destination %4 in writable %reinterpret_cast_0 : (tensor<5x50x2x1xi32>, memref<5x50x2x1xi32, strided<[100, 2, 1, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: triton_ldst_and_totensor_02
// CHECK: memref.alloc() : memref<5xf32>
// CHECK: memref.subview
// CHECK-SAME: memref<5xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK: memref.copy
// CHECK-SAME: memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK: bufferization.materialize_in_destination
// CHECK-SAME: tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>
func.func @triton_ldst_and_totensor_02(%arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: index, %arg9: index) {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%arg9], sizes: [1, 5], strides: [5, 1] : memref<?xf32> to memref<1x5xf32, strided<[5, 1], offset: ?>>
  %alloc = memref.alloc() : memref<1x5xf32>
  %subview = memref.subview %reinterpret_cast[0, 0] [1, %arg5] [1, 1] : memref<1x5xf32, strided<[5, 1], offset: ?>> to memref<1x?xf32, strided<[5, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[0, 0] [1, %arg5] [1, 1] : memref<1x5xf32> to memref<1x?xf32, strided<[5, 1]>>
  memref.copy %subview, %subview_0 : memref<1x?xf32, strided<[5, 1], offset: ?>> to memref<1x?xf32, strided<[5, 1]>>
  %11 = bufferization.to_tensor %alloc restrict writable : memref<1x5xf32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%arg9], sizes: [1, 5], strides: [5, 1] : memref<?xf32> to memref<1x5xf32, strided<[5, 1], offset: ?>>
  %alloc_2 = memref.alloc() : memref<1x5xf32>
  %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [1, %arg5] [1, 1] : memref<1x5xf32, strided<[5, 1], offset: ?>> to memref<1x?xf32, strided<[5, 1], offset: ?>>
  %subview_4 = memref.subview %alloc_2[0, 0] [1, %arg5] [1, 1] : memref<1x5xf32> to memref<1x?xf32, strided<[5, 1]>>
  memref.copy %subview_3, %subview_4 : memref<1x?xf32, strided<[5, 1], offset: ?>> to memref<1x?xf32, strided<[5, 1]>>
  %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<1x5xf32>
  %13 = arith.addf %11, %12 : tensor<1x5xf32>
  %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%arg5], sizes: [1, 5], strides: [5, 1] : memref<?xf32> to memref<1x5xf32, strided<[5, 1], offset: ?>>
  %extracted_slice = tensor.extract_slice %13[0, 0] [1, %arg5] [1, 1] : tensor<1x5xf32> to tensor<1x?xf32>
  %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [1, %arg5] [1, 1] : memref<1x5xf32, strided<[5, 1], offset: ?>> to memref<1x?xf32, strided<[5, 1], offset: ?>>
  bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<1x?xf32>, memref<1x?xf32, strided<[5, 1], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL: triton_ldst_and_totensor_03
// CHECK: memref.alloc() : memref<1xf32>
// CHECK: memref.copy
// CHECK-SAME: memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
// CHECK: bufferization.to_tensor
// CHECK-SAME: restrict writable : memref<1xf32>
// CHECK: memref.reinterpret_cast
// CHECK-SAME: memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK: memref.copy
// CHECK-SAME: memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
// CHECK: bufferization.materialize_in_destination
// CHECK-SAME: tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>
func.func @triton_ldst_and_totensor_03(%arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg8: i32) {
  %0 = arith.index_cast %arg8 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%0], sizes: [1, 1], strides: [1, 1] : memref<?xf32> to memref<1x1xf32, strided<[1, 1], offset: ?>>
  %alloc = memref.alloc() : memref<1x1xf32>
  memref.copy %reinterpret_cast, %alloc : memref<1x1xf32, strided<[1, 1], offset: ?>> to memref<1x1xf32>
  %1 = bufferization.to_tensor %alloc restrict writable : memref<1x1xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%0], sizes: [1, 1], strides: [1, 1] : memref<?xf32> to memref<1x1xf32, strided<[1, 1], offset: ?>>
  %alloc_1 = memref.alloc() : memref<1x1xf32>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<1x1xf32, strided<[1, 1], offset: ?>> to memref<1x1xf32>
  %2 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x1xf32>
  %3 = math.absf %2 : tensor<1x1xf32>
  %4 = arith.addf %1, %3 : tensor<1x1xf32>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%0], sizes: [1, 1], strides: [1, 1] : memref<?xf32> to memref<1x1xf32, strided<[1, 1], offset: ?>>
  bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<1x1xf32>, memref<1x1xf32, strided<[1, 1], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL: triton_ldst_and_totensor_04
// CHECK: %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG0:.*]][0, 0] [%[[ARG2:.*]], 1] [1, 2] : tensor<341x2xf32> to tensor<?xf32>
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ARG1:.*]][0, 0] [%[[ARG3:.*]], 1] [1, 1] : memref<341x2xf32, strided<[2, 1]>> to memref<?x1xf32, strided<[2, 1]>>
// CHECK: memref.collapse_shape %[[SUBVIEW]] {{\[\[}}0, 1]] : memref<?x1xf32, strided<[2, 1]>> into memref<?xf32, strided<[2]>>
func.func @triton_ldst_and_totensor_04(%arg0: tensor<1x341x2xf32>, %arg1: memref<?xf32>, %arg2: index, %arg3: index) {
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [1, %arg2, 1] [1, 1, 2] : tensor<1x341x2xf32> to tensor<1x?xf32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1, 341, 2], strides: [682, 2, 1] : memref<?xf32> to memref<1x341x2xf32, strided<[682, 2, 1]>>
  %subview_3 = memref.subview %reinterpret_cast_1[0, 0, 0] [1, %arg3, 1] [1, 1, 1] : memref<1x341x2xf32, strided<[682, 2, 1]>> to memref<1x?xf32, strided<[682, 2]>>
  bufferization.materialize_in_destination %extracted_slice in writable %subview_3 : (tensor<1x?xf32>, memref<1x?xf32, strided<[682, 2]>>) -> ()
  return
}

// -----

// CHECK-LABEL: triton_ldst_and_totensor_05
// CHECK: %[[CST1:.*]] = arith.constant 1 : index
// CHECK: %[[VAL8:.*]] = arith.minsi %[[VAL6:.*]], %[[CST1]] : index
// CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [%[[OFFSET2:.*]]], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1], offset: ?>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8xf32>
// CHECK: %[[VAL10:.*]] = arith.muli %[[VAL8]], %[[VAL9:.*]] : index
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[CAST]][0] [%[[VAL10]]] [1] : memref<8xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK: %[[SUBVIEW0:.*]] = memref.subview %[[ALLOC]][0] [%[[VAL10]]] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>
// CHECK: memref.copy %[[SUBVIEW]], %[[SUBVIEW0]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK: %[[TENSOR11:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<8xf32>
// CHECK: %[[CAST1:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [%[[OFFSET2]]], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1], offset: ?>>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[TENSOR11]][0] [%[[VAL10]]] [1] : tensor<8xf32> to tensor<?xf32>
// CHECK: %[[SUBVIEW2:.*]] = memref.subview %[[CAST1]][0] [%[[VAL10]]] [1] : memref<8xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK: bufferization.materialize_in_destination %[[SLICE]] in writable %[[SUBVIEW2]] : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
func.func @triton_ldst_and_totensor_05(%arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: i32, %arg9: i32) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c8_i32 = arith.constant 8 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = arith.cmpi slt, %arg9, %c1_i32 : i32
  %1 = arith.muli %arg9, %c8_i32 : i32
  %2 = arith.index_cast %1 : i32 to index
  %3 = arith.index_cast %arg5 : i32 to index
  %4 = arith.maxsi %3, %c0 : index
  %5 = arith.minsi %4, %c8 : index
  %6 = arith.index_castui %0 : i1 to index
  %7 = arith.muli %6, %c8 : index
  %8 = arith.minsi %6, %c1 : index
  %9 = arith.minsi %5, %7 : index
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%2], sizes: [1, 8], strides: [8, 1] : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
  %alloc_2 = memref.alloc() : memref<1x8xf32>
  %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%8, %9] [1, 1] : memref<1x8xf32, strided<[8, 1], offset: ?>> to memref<?x?xf32, strided<[8, 1], offset: ?>>
  %subview_4 = memref.subview %alloc_2[0, 0] [%8, %9] [1, 1] : memref<1x8xf32> to memref<?x?xf32, strided<[8, 1]>>
  memref.copy %subview_3, %subview_4 : memref<?x?xf32, strided<[8, 1], offset: ?>> to memref<?x?xf32, strided<[8, 1]>>
  %16 = bufferization.to_tensor %alloc_2 restrict writable : memref<1x8xf32>
  %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%2], sizes: [1, 8], strides: [8, 1] : memref<?xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
  %extracted_slice = tensor.extract_slice %16[0, 0] [%8, %9] [1, 1] : tensor<1x8xf32> to tensor<?x?xf32>
  %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%8, %9] [1, 1] : memref<1x8xf32, strided<[8, 1], offset: ?>> to memref<?x?xf32, strided<[8, 1], offset: ?>>
  bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?x?xf32>, memref<?x?xf32, strided<[8, 1], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL: triton_ldst_and_totensor_06
// CHECK: %[[CST1:.*]] = arith.constant 1 : index
// CHECK: %[[VAL0:.*]] = arith.minsi %[[ARG2:.*]], %[[CST1]] : index
// CHECK: %[[VAL1:.*]] = arith.minsi %[[ARG3:.*]], %[[ARG4:.*]] : index
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x10x64xf32>
// CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [%[[ARG1:.*]]], sizes: [2, 10, 64], strides: [640, 64, 1] : memref<?xf32> to memref<2x10x64xf32, strided<[640, 64, 1], offset: ?>>
// CHECK: %[[VAL2:.*]] = arith.muli %[[VAL0]], %[[VAL1]] : index
// CHECK: %[[VAL3:.*]] = arith.muli %[[VAL2]], %[[VAL0]] : index
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[CAST]][0, 0, 0] [%[[VAL3]], 10, 64] [1, 1, 1] : memref<2x10x64xf32, strided<[640, 64, 1], offset: ?>> to memref<?x10x64xf32, strided<[640, 64, 1], offset: ?>>
// CHECK: %[[SUBVIEW0:.*]] = memref.subview %[[ALLOC]][0, 0, 0] [%[[VAL3]], 10, 64] [1, 1, 1] : memref<2x10x64xf32> to memref<?x10x64xf32, strided<[640, 64, 1]>>
// CHECK: memref.copy %[[SUBVIEW]], %[[SUBVIEW0]] : memref<?x10x64xf32, strided<[640, 64, 1], offset: ?>> to memref<?x10x64xf32, strided<[640, 64, 1]>>
func.func @triton_ldst_and_totensor_06(%arg2: memref<?xf32>, %15: index, %19: index, %20: index, %25: index) {
  %c1 = arith.constant 1 : index
  %26 = arith.minsi %19, %c1 : index
  %27 = arith.minsi %20, %25 : index
  %alloc = memref.alloc() : memref<1x2x1x10x64xf32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [1, 2, 1, 10, 64], strides: [1280, 640, 640, 64, 1] : memref<?xf32> to memref<1x2x1x10x64xf32, strided<[1280, 640, 640, 64, 1], offset: ?>>
  %subview = memref.subview %reinterpret_cast_1[0, 0, 0, 0, 0] [%26, %27, %26, 10, 64] [1, 1, 1, 1, 1] : memref<1x2x1x10x64xf32, strided<[1280, 640, 640, 64, 1], offset: ?>> to memref<?x?x?x10x64xf32, strided<[1280, 640, 640, 64, 1], offset: ?>>
  %subview_3 = memref.subview %alloc[0, 0, 0, 0, 0] [%26, %27, %26, 10, 64] [1, 1, 1, 1, 1] : memref<1x2x1x10x64xf32> to memref<?x?x?x10x64xf32, strided<[1280, 640, 640, 64, 1]>>
  memref.copy %subview, %subview_3 : memref<?x?x?x10x64xf32, strided<[1280, 640, 640, 64, 1], offset: ?>> to memref<?x?x?x10x64xf32, strided<[1280, 640, 640, 64, 1]>>
  return
}

// -----

// CHECK-LABEL: triton_ldst_and_totensor_07
// CHECK: %[[CST1:.*]] = arith.constant 1 : index
// CHECK: %[[VAL0:.*]] = arith.minsi %[[ARG2:.*]], %[[CST1]] : index
// CHECK: %[[VAL1:.*]] = arith.minsi %[[ARG3:.*]], %[[ARG4:.*]] : index
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x10x64xf32>
// CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [%[[ARG1:.*]]], sizes: [2, 10, 64], strides: [640, 64, 1] : memref<?xf32> to memref<2x10x64xf32, strided<[640, 64, 1], offset: ?>>
// CHECK: %[[VAL2:.*]] = arith.muli %[[VAL0]], %[[VAL1]] : index
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[CAST]][0, 0, 0] [%[[VAL2]], 10, 64] [1, 1, 1] : memref<2x10x64xf32, strided<[640, 64, 1], offset: ?>> to memref<?x10x64xf32, strided<[640, 64, 1], offset: ?>>
// CHECK: %[[SUBVIEW0:.*]] = memref.subview %[[ALLOC]][0, 0, 0] [%[[VAL2]], 10, 64] [1, 1, 1] : memref<2x10x64xf32> to memref<?x10x64xf32, strided<[640, 64, 1]>>
// CHECK: memref.copy %[[SUBVIEW]], %[[SUBVIEW0]] : memref<?x10x64xf32, strided<[640, 64, 1], offset: ?>> to memref<?x10x64xf32, strided<[640, 64, 1]>>
func.func @triton_ldst_and_totensor_07(%arg2: memref<?xf32>, %15: index, %19: index, %20: index, %25: index) {
  %c1 = arith.constant 1 : index
  %26 = arith.minsi %19, %c1 : index
  %27 = arith.minsi %20, %25 : index
  %alloc = memref.alloc() : memref<1x2x1x10x64xf32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [1, 2, 1, 10, 64], strides: [1280, 640, 640, 64, 1] : memref<?xf32> to memref<1x2x1x10x64xf32, strided<[1280, 640, 640, 64, 1], offset: ?>>
  %subview = memref.subview %reinterpret_cast_1[0, 0, 0, 0, 0] [%26, %27, 1, 10, 64] [1, 1, 1, 1, 1] : memref<1x2x1x10x64xf32, strided<[1280, 640, 640, 64, 1], offset: ?>> to memref<?x?x1x10x64xf32, strided<[1280, 640, 640, 64, 1], offset: ?>>
  %subview_3 = memref.subview %alloc[0, 0, 0, 0, 0] [%26, %27, 1, 10, 64] [1, 1, 1, 1, 1] : memref<1x2x1x10x64xf32> to memref<?x?x1x10x64xf32, strided<[1280, 640, 640, 64, 1]>>
  memref.copy %subview, %subview_3 : memref<?x?x1x10x64xf32, strided<[1280, 640, 640, 64, 1], offset: ?>> to memref<?x?x1x10x64xf32, strided<[1280, 640, 640, 64, 1]>>
  return
}

// -----

// CHECK-LABEL: triton_ldst_and_totensor_08
// CHECK: memref.reinterpret_cast
// CHECK-SAME: memref<1xi16, strided<[1]>>
// CHECK: memref.alloc() : memref<1xi16>
// CHECK: memref.copy
// CHECK-SAME: memref<1xi16>
// CHECK: bufferization.to_tensor
// CHECK-SAME: memref<1xi16>
// CHECK: memref.reinterpret_cast
// CHECK-SAME: memref<1xi32, strided<[1]>>
// CHECK: bufferization.materialize_in_destination
// CHECK-SAME: (tensor<1xi32>, memref<1xi32, strided<[1]>>)
func.func @triton_ldst_and_totensor_08(%arg2: memref<?xi16>, %arg3: memref<?xi32>) {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1, 1, 1, 1], strides: [1, 1, 1, 1] : memref<?xi16> to memref<1x1x1x1xi16, strided<[1, 1, 1, 1]>>
  %alloc = memref.alloc() : memref<1x1x1x1xi16>
  memref.copy %reinterpret_cast, %alloc : memref<1x1x1x1xi16, strided<[1, 1, 1, 1]>> to memref<1x1x1x1xi16>
  %3 = bufferization.to_tensor %alloc restrict writable : memref<1x1x1x1xi16>
  %4 = arith.extsi %3 : tensor<1x1x1x1xi16> to tensor<1x1x1x1xi32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 1, 1, 1], strides: [1, 1, 1, 1] : memref<?xi32> to memref<1x1x1x1xi32, strided<[1, 1, 1, 1]>>
  bufferization.materialize_in_destination %4 in writable %reinterpret_cast_0 : (tensor<1x1x1x1xi32>, memref<1x1x1x1xi32, strided<[1, 1, 1, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: triton_split
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[EXTRACTED]] [] output_shape [1] : tensor<i32> into tensor<1xi32>
// CHECK: bufferization.materialize_in_destination %[[EXPANDED]]
func.func @triton_split(%arg0: tensor<1x1x1x2xi32>, %arg1: memref<?xi32>) {
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 2] : tensor<1x1x1x2xi32> to tensor<1x1x1xi32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1, 1, 1], strides: [1, 1, 1] : memref<?xi32> to memref<1x1x1xi32, strided<[1, 1, 1]>>
  bufferization.materialize_in_destination %extracted_slice in writable %reinterpret_cast_1 : (tensor<1x1x1xi32>, memref<1x1x1xi32, strided<[1, 1, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: triton_extract_insert_slice_01
// CHECK: %[[MUL_RESULT:.*]] = arith.muli %{{.*}}, %{{.*}} : index
// CHECK: %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %{{.*}}[0] [%[[MUL_RESULT]]] [1] : tensor<8192xf32> to tensor<?xf32>
// CHECK: %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[EXTRACTED_SLICE]] into %{{.*}}[0] [%[[MUL_RESULT]]] [1] : tensor<?xf32> into tensor<8192xf32>
func.func @triton_extract_insert_slice_01(%3: i32) -> tensor<8192x1xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i64 = arith.constant 0 : i64
  %c1 = arith.constant 1 : index
  %0 = tensor.empty() : tensor<8192x1xf32>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<8192x1xf32>) -> tensor<8192x1xf32>
  %7 = scf.for %arg12 = %c0_i32 to %3 step %c1_i32 iter_args(%arg13 = %1) -> (tensor<8192x1xf32>) : i32 {
    %24 = arith.index_cast %arg12 : i32 to index
    %25 = arith.minsi %24, %c1 : index
    %30 = math.exp %arg13 : tensor<8192x1xf32>
    %extracted_slice_3 = tensor.extract_slice %30[0, 0] [%24, %25] [1, 1] : tensor<8192x1xf32> to tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %extracted_slice_3 into %arg13[0, 0] [%24, %25] [1, 1] : tensor<?x?xf32> into tensor<8192x1xf32>
    scf.yield %inserted_slice : tensor<8192x1xf32>
  }
  return %7 : tensor<8192x1xf32>
}

// -----

// CHECK-LABEL: triton_extract_insert_slice_02
// CHECK: %[[MUL_RESULT:.*]] = arith.muli %{{.*}}, %{{.*}} : index
// CHECK: %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %{{.*}}[0] [%[[MUL_RESULT]]] [1] : tensor<8192xf32> to tensor<?xf32>
// CHECK: %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[EXTRACTED_SLICE]] into %{{.*}}[0] [%[[MUL_RESULT]]] [1] : tensor<?xf32> into tensor<8192xf32>
func.func @triton_extract_insert_slice_02(%3: i32) -> tensor<1x8192x1xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i64 = arith.constant 0 : i64
  %c1 = arith.constant 1 : index
  %0 = tensor.empty() : tensor<1x8192x1xf32>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1x8192x1xf32>) -> tensor<1x8192x1xf32>
  %7 = scf.for %arg12 = %c0_i32 to %3 step %c1_i32 iter_args(%arg13 = %1) -> (tensor<1x8192x1xf32>) : i32 {
    %24 = arith.index_cast %arg12 : i32 to index
    %25 = arith.minsi %24, %c1 : index
    %30 = math.exp %arg13 : tensor<1x8192x1xf32>
    %extracted_slice_3 = tensor.extract_slice %30[0, 0, 0] [1, %24, %25] [1, 1, 1] : tensor<1x8192x1xf32> to tensor<1x?x?xf32>
    %inserted_slice = tensor.insert_slice %extracted_slice_3 into %arg13[0, 0, 0] [1, %24, %25] [1, 1, 1] : tensor<1x?x?xf32> into tensor<1x8192x1xf32>
    scf.yield %inserted_slice : tensor<1x8192x1xf32>
  }
  return %7 : tensor<1x8192x1xf32>
}

// -----

// CHECK-LABEL: triton_scope
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[NOT_COLLAPSED1:.*]], %[[NOT_COLLAPSED2:.*]] : tensor<1x64xf32>, tensor<1x64xf32>)
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[VAL:.*]] : tensor<1x64xf32>)
func.func @triton_scope(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg4: memref<?xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 5.000000e-01 : f32
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %alloc = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
  scope.scope : () -> () {
    %2 = linalg.matmul {input_precison = "ieee"} ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%2 : tensor<128x128xf32>) outs(%alloc : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
    scope.return
  } {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, noinline}
  %3 = tensor.empty() : tensor<1x64xf32>
  %4 = linalg.fill ins(%cst_1 : f32) outs(%3 : tensor<1x64xf32>) -> tensor<1x64xf32>
  scope.scope : () -> () {
    %memspacecast = memref.memory_space_cast %alloc : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
    %5 = bufferization.to_tensor %memspacecast restrict writable : memref<64x128xf32>
    %8 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg3 = %4) -> (tensor<1x64xf32>)  : i32 {
      %6 = arith.index_cast %arg2 : i32 to index
      %extracted_slice = tensor.extract_slice %5[%6, 0] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
      %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice, %arg3 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%3 : tensor<1x64xf32>) -> tensor<1x64xf32>
      scf.yield %7 : tensor<1x64xf32>
    }
    %10 = scope.scope : () -> (tensor<1x64xf32>) {
      %9 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%8 : tensor<1x64xf32>) outs(%3 : tensor<1x64xf32>) -> tensor<1x64xf32>
      scope.return %9 : tensor<1x64xf32>
    } {noinline, outline = true, vector_mode = "simd"}
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<?xf32> to memref<1x64xf32, strided<[64, 1]>>
    bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<1x64xf32>, memref<1x64xf32, strided<[64, 1]>>) -> ()
    scope.return
  } {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, noinline}
  return
}

// -----

// CHECK-LABEL: func.func @scffor_iterargs
// CHECK-NOT: tensor.expand_shape
// CHECK: scf.for
// CHECK-SAME: (tensor<20x32x4xf32>, tensor<32x4x8xf32>, i32)
// CHECK-NOT: tensor.collapse_shape
func.func @scffor_iterargs(%arg0: tensor<32x4x8xf32>, %arg1: index, %arg2: i32) -> tensor<1x32x4xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c20_i32 = arith.constant 20 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<32x4x8xf32>
  %1 = tensor.empty() : tensor<32x4xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<32x4xf32>) -> tensor<32x4xf32>
  %3 = tensor.empty() : tensor<20x32x4xf32>
  %expanded = tensor.expand_shape %3 [[0], [1], [2, 3]] output_shape [20, 32, 4, 1] : tensor<20x32x4xf32> into tensor<20x32x4x1xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%expanded : tensor<20x32x4x1xf32>) -> tensor<20x32x4x1xf32>
  %5:3 = scf.for %arg3 = %c0_i32 to %c20_i32 step %c1_i32 iter_args(%arg4 = %4, %arg5 = %arg0, %arg6 = %arg2) -> (tensor<20x32x4x1xf32>, tensor<32x4x8xf32>, i32)  : i32 {
    %reduced = linalg.reduce ins(%arg5 : tensor<32x4x8xf32>) outs(%2 : tensor<32x4xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
    %collapsed_0 = tensor.collapse_shape %arg4 [[0], [1], [2, 3]] : tensor<20x32x4x1xf32> into tensor<20x32x4xf32>
    %6 = arith.addi %arg3, %arg6 : i32
    %7 = arith.index_cast %6 : i32 to index
    %inserted_slice = tensor.insert_slice %reduced into %collapsed_0[%7, 0, 0] [1, 32, 4] [1, 1, 1] : tensor<32x4xf32> into tensor<20x32x4xf32>
    %expanded_1 = tensor.expand_shape %inserted_slice [[0], [1], [2, 3]] output_shape [20, 32, 4, 1] : tensor<20x32x4xf32> into tensor<20x32x4x1xf32>
    %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%arg5 : tensor<32x4x8xf32>) outs(%0 : tensor<32x4x8xf32>) -> tensor<32x4x8xf32>
    scf.yield %expanded_1, %8, %6 : tensor<20x32x4x1xf32>, tensor<32x4x8xf32>, i32
  }
  %collapsed = tensor.collapse_shape %5#0 [[0], [1], [2, 3]] : tensor<20x32x4x1xf32> into tensor<20x32x4xf32>
  %extracted_slice = tensor.extract_slice %collapsed[%arg1, 0, 0] [1, 32, 4] [1, 1, 1] : tensor<20x32x4xf32> to tensor<1x32x4xf32>
  return %extracted_slice : tensor<1x32x4xf32>
}

// -----

// CHECK-LABEL: func.func @store_with_memref_cast
// CHECK-NOT: memref.cast
// CHECK: bufferization.materialize_in_destination %[[VAL0:.*]] in writable %[[REINTER_CAST_0:.*]] : (tensor<1xf32>, memref<1xf32, strided<[1]>>)
func.func @store_with_memref_cast(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 1, 1, 1, 1, 1], strides: [1, 1, 1, 1, 1, 1] : memref<?xf32> to memref<1x1x1x1x1x1xf32, strided<[1, 1, 1, 1, 1, 1]>>
  %alloc = memref.alloc() : memref<1x1x1x1x1x1xf32>
  memref.copy %reinterpret_cast, %alloc : memref<1x1x1x1x1x1xf32, strided<[1, 1, 1, 1, 1, 1]>> to memref<1x1x1x1x1x1xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<1x1x1x1x1x1xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1, 1, 1, 1, 1, 1], strides: [1, 1, 1, 1, 1, 1] : memref<?xf32> to memref<1x1x1x1x1x1xf32, strided<[1, 1, 1, 1, 1, 1]>>
  %cast = memref.cast %reinterpret_cast_0 : memref<1x1x1x1x1x1xf32, strided<[1, 1, 1, 1, 1, 1]>> to memref<?x?x?x?x?x?xf32, strided<[1, 1, 1, 1, 1, 1], offset: ?>>
  bufferization.materialize_in_destination %0 in writable %cast : (tensor<1x1x1x1x1x1xf32>, memref<?x?x?x?x?x?xf32, strided<[1, 1, 1, 1, 1, 1], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @store_with_memref_cast2
// CHECK: %[[CAST_0:.*]] = memref.cast
// CHECK: bufferization.materialize_in_destination %[[VAL0:.*]] in writable %[[CAST_0]] : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>)
func.func @store_with_memref_cast2(%arg0: tensor<?x?xf32>, %arg1: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg1, %c0 : memref<?x?xf32>
  %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
  %subview = memref.subview %arg1[0, 0] [%dim, %dim_0] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
  %cast = memref.cast %subview : memref<?x?xf32, strided<[?, 1]>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  %extracted_slice = tensor.extract_slice %arg0[0, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  bufferization.materialize_in_destination %extracted_slice in writable %cast : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @insert_slice_fallback_unit_dims
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<2x3xf32>
func.func @insert_slice_fallback_unit_dims(%arg0: tensor<2x1x2xf32>, %arg1: tensor<2x1x3xf32>) -> tensor<2x1x3xf32> {
  %0 = tensor.insert_slice %arg0 into %arg1[0, 0, 0] [2, 1, 2] [1, 1, 1]
    : tensor<2x1x2xf32> into tensor<2x1x3xf32>
  return %0 : tensor<2x1x3xf32>
}

// -----

// CHECK-LABEL: func.func @insert_slice_dropped_dims
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<2x3xf32>
func.func @insert_slice_dropped_dims(%arg0: tensor<2x2xf32>, %arg1: tensor<2x1x3x1xf32>) -> tensor<2x1x3x1xf32> {
  // dst has unit dim at position 1, src does not - this is a dropped dims case
  %0 = tensor.insert_slice %arg0 into %arg1[0, 0, 0, 0] [2, 1, 2, 1] [1, 1, 1, 1]
    : tensor<2x2xf32> into tensor<2x1x3x1xf32>
  return %0 : tensor<2x1x3x1xf32>
}

// -----

// CHECK-LABEL: func.func @rank_reduce_insert_slice
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 0, 0] [1, 2, 1] [1, 1, 1] : tensor<2xf32> into tensor<3x2x4xf32>
func.func @rank_reduce_insert_slice(%arg0: tensor<1x2x1xf32>, %arg1: tensor<3x2x4xf32>) -> tensor<3x2x4xf32> {
  %0 = tensor.insert_slice %arg0 into %arg1[0, 0, 0] [1, 2, 1] [1, 1, 1]
    : tensor<1x2x1xf32> into tensor<3x2x4xf32>
  return %0 : tensor<3x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @insert_slice_not_support_dynamic_src
// CHECK: tensor.insert_slice %arg0 into %arg1[0, 0, %arg2] [1, %arg3, 1] [1, 1, 1] : tensor<?xf32> into tensor<3x2x4xf32>

module {
  func.func @insert_slice_not_support_dynamic_src(%arg0: tensor<?xf32>, %arg1: tensor<3x2x4xf32>, %arg2: index, %arg3: index) -> tensor<3x2x4xf32> {
    %inserted_slice = tensor.insert_slice %arg0 into %arg1[0, 0, %arg2] [1, %arg3, 1] [1, 1, 1] : tensor<?xf32> into tensor<3x2x4xf32>
    return %inserted_slice : tensor<3x2x4xf32>
  }
}

// -----

// CHECK-LABEL: func.func @should_interleave
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 0, 0, 0] [3, 2, 250, 1] [1, 1, 1, 2] : tensor<3x2x250xf32> into tensor<3x2x250x2xf32>
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 0, 0, 1] [3, 2, 250, 1] [1, 1, 1, 2] : tensor<3x2x250xf32> into tensor<3x2x250x2xf32>
func.func @should_interleave(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %0 = arith.index_cast %arg8 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%0], sizes: [3, 1, 1, 2, 250], strides: [500, 500, 500, 250, 1] : memref<?xf32> to memref<3x1x1x2x250xf32, strided<[500, 500, 500, 250, 1], offset: ?>>
  %alloc = memref.alloc() : memref<3x1x1x2x250xf32>
  memref.copy %reinterpret_cast, %alloc : memref<3x1x1x2x250xf32, strided<[500, 500, 500, 250, 1], offset: ?>> to memref<3x1x1x2x250xf32>
  %1 = bufferization.to_tensor %alloc restrict writable : memref<3x1x1x2x250xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%0], sizes: [3, 1, 1, 2, 250], strides: [500, 500, 500, 250, 1] : memref<?xf32> to memref<3x1x1x2x250xf32, strided<[500, 500, 500, 250, 1], offset: ?>>
  %alloc_1 = memref.alloc() : memref<3x1x1x2x250xf32>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<3x1x1x2x250xf32, strided<[500, 500, 500, 250, 1], offset: ?>> to memref<3x1x1x2x250xf32>
  %2 = bufferization.to_tensor %alloc_1 restrict writable : memref<3x1x1x2x250xf32>
  %3 = tensor.empty() : tensor<3x1x1x2x250x2xf32>
  %inserted_slice = tensor.insert_slice %1 into %3[0, 0, 0, 0, 0, 0] [3, 1, 1, 2, 250, 1] [1, 1, 1, 1, 1, 2] : tensor<3x1x1x2x250xf32> into tensor<3x1x1x2x250x2xf32>
  %inserted_slice_2 = tensor.insert_slice %2 into %inserted_slice[0, 0, 0, 0, 0, 1] [3, 1, 1, 2, 250, 1] [1, 1, 1, 1, 1, 2] : tensor<3x1x1x2x250xf32> into tensor<3x1x1x2x250x2xf32>
  %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%0], sizes: [3, 1, 1, 2, 250, 2], strides: [1000, 1000, 1000, 500, 2, 1] : memref<?xf32> to memref<3x1x1x2x250x2xf32, strided<[1000, 1000, 1000, 500, 2, 1], offset: ?>>
  bufferization.materialize_in_destination %inserted_slice_2 in writable %reinterpret_cast_3 : (tensor<3x1x1x2x250x2xf32>, memref<3x1x1x2x250x2xf32, strided<[1000, 1000, 1000, 500, 2, 1], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @store_with_dropped_dims
// CHECK: %[[DST:.*]] = tensor.extract_slice {{.*}}[0] [{{.*}}] [1] : tensor<64xf32> to tensor<?xf32>
// CHECK: hivm.hir.store ins(%[[DST]] : tensor<?xf32>) outs({{.*}} : memref<?xf32, strided<[1], offset: ?>>) atomic = <add>
func.func @store_with_dropped_dims(%arg0: tensor<1x64xf32>, %arg1: memref<?x?xf32>, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %2 = arith.maxsi %c1, %c64 : index
  %cast_5 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<?x?xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>> 
  %subview = memref.subview %cast_5[0, 0] [%arg2, %2] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1], offset: ?>> 
  %extracted_slice = tensor.extract_slice %arg0[0, 0] [%arg2, %2] [1, 1] : tensor<1x64xf32> to tensor<?x?xf32>
  hivm.hir.store ins(%extracted_slice : tensor<?x?xf32>) outs(%subview : memref<?x?xf32, strided<[64, 1], offset: ?>>) atomic = <add>
  return
}