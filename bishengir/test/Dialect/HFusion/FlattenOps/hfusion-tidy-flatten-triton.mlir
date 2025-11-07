// RUN: bishengir-opt %s                              \
// RUN:   -pass-pipeline="builtin.module(func.func(   \
// RUN:      hfusion-flatten-ops{flatten-mode=tidy}), \
// RUN:      cse, canonicalize-ext)"                      \
// RUN:   -split-input-file | FileCheck %s

// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape
func.func @test_basic__kernel0(%arg0: memref<?x?xf32> {tt.divisibility = 16 : i32}, %arg1: memref<?x?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?x?xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {global_kernel = "local"} {
  %c256 = arith.constant 256 : index
  %c16 = arith.constant 16 : index
  %c256_i32 = arith.constant 256 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = arith.muli %arg6, %c256_i32 : i32
  %1 = arith.muli %arg4, %c16_i32 : i32
  %2 = arith.index_cast %0 : i32 to index
  %3 = arith.index_cast %1 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [256, 16], strides: [1, 1] : memref<?x?xf32> to memref<256x16xf32, strided<[1, 1], offset: ?>>
  %alloc = memref.alloc() : memref<256x16xf32>
  %4 = arith.addi %2, %c256 : index
  %5 = arith.addi %3, %c16 : index
  %6 = arith.index_cast %arg3 : i32 to index
  %7 = arith.index_cast %arg5 : i32 to index
  %8 = arith.maxsi %2, %6 : index
  %9 = arith.maxsi %3, %7 : index
  %10 = arith.minsi %4, %8 : index
  %11 = arith.minsi %5, %9 : index
  %12 = arith.subi %10, %2 : index
  %13 = arith.subi %11, %3 : index
  %subview = memref.subview %reinterpret_cast[0, 0] [%12, %13] [1, 1] : memref<256x16xf32, strided<[1, 1], offset: ?>> to memref<?x?xf32, strided<[1, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[0, 0] [%12, %13] [1, 1] : memref<256x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  memref.copy %subview, %subview_0 : memref<?x?xf32, strided<[1, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
  %14 = bufferization.to_tensor %alloc restrict writable : memref<256x16xf32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [256, 16], strides: [1, 1] : memref<?x?xf32> to memref<256x16xf32, strided<[1, 1], offset: ?>>
  %alloc_2 = memref.alloc() : memref<256x16xf32>
  %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%12, %13] [1, 1] : memref<256x16xf32, strided<[1, 1], offset: ?>> to memref<?x?xf32, strided<[1, 1], offset: ?>>
  %subview_4 = memref.subview %alloc_2[0, 0] [%12, %13] [1, 1] : memref<256x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  memref.copy %subview_3, %subview_4 : memref<?x?xf32, strided<[1, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
  %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<256x16xf32>
  %16 = tensor.empty() : tensor<256x16xf32>
  %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%14, %15 : tensor<256x16xf32>, tensor<256x16xf32>) outs(%16 : tensor<256x16xf32>) -> tensor<256x16xf32>
  %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%2], sizes: [256, 16], strides: [1, 1] : memref<?x?xf32> to memref<256x16xf32, strided<[1, 1], offset: ?>>
  %extracted_slice = tensor.extract_slice %17[0, 0] [%12, %13] [1, 1] : tensor<256x16xf32> to tensor<?x?xf32>
  %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%12, %13] [1, 1] : memref<256x16xf32, strided<[1, 1], offset: ?>> to memref<?x?xf32, strided<[1, 1], offset: ?>>
  bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?x?xf32>, memref<?x?xf32, strided<[1, 1], offset: ?>>) -> ()
  return
}

// -----

// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[}}{{\[}}0, 1]] : tensor<256x16xf32> into tensor<4096xf32>
// CHECK: tensor.expand_shape
// CHECK-SAME {{\[}}{{\[}}0, 1]] output_shape {{\[}}256, 16] : tensor<4096xf32> into tensor<256x16xf32>
func.func @test_basic__kernel0(%arg0: memref<?x?xf32> {tt.divisibility = 16 : i32}, %arg1: memref<?x?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?x?xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {global_kernel = "local"} {
  %c256 = arith.constant 256 : index
  %c16 = arith.constant 16 : index
  %c256_i32 = arith.constant 256 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = arith.muli %arg6, %c256_i32 : i32
  %1 = arith.muli %arg4, %c16_i32 : i32
  %2 = arith.index_cast %0 : i32 to index
  %3 = arith.index_cast %1 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [256, 16], strides: [1, 1] : memref<?x?xf32> to memref<256x16xf32, strided<[1, 1], offset: ?>>
  %alloc = memref.alloc() : memref<256x16xf32>
  %4 = arith.addi %2, %c256 : index
  %5 = arith.addi %3, %c16 : index
  %6 = arith.index_cast %arg3 : i32 to index
  %7 = arith.index_cast %arg5 : i32 to index
  %8 = arith.maxsi %2, %6 : index
  %9 = arith.maxsi %3, %7 : index
  %10 = arith.minsi %4, %8 : index
  %11 = arith.minsi %5, %9 : index
  %12 = arith.subi %10, %2 : index
  %13 = arith.subi %11, %3 : index
  %subview = memref.subview %reinterpret_cast[0, 0] [%12, %13] [1, 1] : memref<256x16xf32, strided<[1, 1], offset: ?>> to memref<?x?xf32, strided<[1, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[0, 0] [%12, %13] [1, 1] : memref<256x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  memref.copy %subview, %subview_0 : memref<?x?xf32, strided<[1, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
  %14 = bufferization.to_tensor %alloc restrict writable : memref<256x16xf32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [256, 16], strides: [1, 1] : memref<?x?xf32> to memref<256x16xf32, strided<[1, 1], offset: ?>>
  %alloc_2 = memref.alloc() : memref<256x16xf32>
  %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%12, %13] [1, 1] : memref<256x16xf32, strided<[1, 1], offset: ?>> to memref<?x?xf32, strided<[1, 1], offset: ?>>
  %subview_4 = memref.subview %alloc_2[0, 0] [%12, %13] [1, 1] : memref<256x16xf32> to memref<?x?xf32, strided<[16, 1]>>
  memref.copy %subview_3, %subview_4 : memref<?x?xf32, strided<[1, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>>
  %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<256x16xf32>
  %16 = tensor.empty() : tensor<256x16xf32>
  %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%14, %15 : tensor<256x16xf32>, tensor<256x16xf32>) outs(%16 : tensor<256x16xf32>) -> tensor<256x16xf32>
  %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%2], sizes: [256, 16], strides: [1, 1] : memref<?x?xf32> to memref<256x16xf32, strided<[1, 1], offset: ?>>
  bufferization.materialize_in_destination %17 in writable %reinterpret_cast_5 : (tensor<256x16xf32>, memref<256x16xf32, strided<[1, 1], offset: ?>>) -> ()
  return
}