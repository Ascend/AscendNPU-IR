// RUN: bishengir-compile -enable-hfusion-compile=true -enable-hivm-inject-barrier-all-sync -enable-triton-kernel-compile %s
// RUN: bishengir-compile -enable-hfusion-compile=true -enable-triton-kernel-compile %s

#map = affine_map<(d0) -> (d0)>
module {
  func.func @test_basic__kernel0(%arg0: memref<?xf32> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {global_kernel = "local"} {
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.muli %arg7, %c256_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<256xf32>
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.addi %2, %c256 : index
    %4 = arith.index_cast %arg3 : i32 to index
    %5 = arith.maxsi %2, %4 : index
    %6 = arith.minsi %3, %5 : index
    %7 = arith.subi %6, %2 : index
    %subview = memref.subview %reinterpret_cast[0] [%7] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%7] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
    %9 = arith.index_cast %0 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%9], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<256xf32>
    %10 = arith.index_cast %0 : i32 to index
    %11 = arith.addi %10, %c256 : index
    %12 = arith.index_cast %arg3 : i32 to index
    %13 = arith.maxsi %10, %12 : index
    %14 = arith.minsi %11, %13 : index
    %15 = arith.subi %14, %10 : index
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%15] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%15] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %16 = bufferization.to_tensor %alloc_2 restrict writable : memref<256xf32>
    %17 = arith.addf %8, %16 : tensor<256xf32>
    %18 = arith.index_cast %0 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %19 = arith.index_cast %0 : i32 to index
    %20 = arith.addi %19, %c256 : index
    %21 = arith.index_cast %arg3 : i32 to index
    %22 = arith.maxsi %19, %21 : index
    %23 = arith.minsi %20, %22 : index
    %24 = arith.subi %23, %19 : index
    %extracted_slice = tensor.extract_slice %17[0] [%24] [1] : tensor<256xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%24] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

