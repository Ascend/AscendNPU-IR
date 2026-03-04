// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -enable-hivm-inject-barrier-all-sync -enable-triton-kernel-compile %s | FileCheck %s
// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -enable-triton-kernel-compile %s | FileCheck %s

// CHECK: LLVMDialectModule
#map = affine_map<(d0) -> (d0)>
module {
  func.func @test_brc_mul__kernel0(%arg0: memref<?xf32> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {global_kernel = "local"} {
    %c256_i64 = arith.constant 256 : i64
    %cst = arith.constant dense<[2, 1, 32]> : tensor<3xi64>
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c4_i32 = arith.constant 4 : i32
    %c64_i32 = arith.constant 64 : i32
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
    %9 = arith.muli %arg7, %c64_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    %11 = arith.divsi %arg3, %c4_i32 : i32
    %alloc_2 = memref.alloc() : memref<64xf32>
    %12 = arith.index_cast %9 : i32 to index
    %13 = arith.addi %12, %c64 : index
    %14 = arith.index_cast %11 : i32 to index
    %15 = arith.maxsi %12, %14 : index
    %16 = arith.minsi %13, %15 : index
    %17 = arith.subi %16, %12 : index
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%17] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%17] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<64xf32>
    %reshape = tensor.reshape %18(%cst) : (tensor<64xf32>, tensor<3xi64>) -> tensor<2x1x32xf32>
    %19 = tensor.empty() : tensor<2x4x32xf32>
    %collapsed = tensor.collapse_shape %reshape [[0], [1, 2]] : tensor<2x1x32xf32> into tensor<2x32xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<2x32xf32>) outs(%19 : tensor<2x4x32xf32>) dimensions = [1]
    %20 = tensor.empty() : tensor<1xi64>
    %21 = linalg.fill ins(%c256_i64 : i64) outs(%20 : tensor<1xi64>) -> tensor<1xi64>
    %reshape_5 = tensor.reshape %broadcasted(%21) : (tensor<2x4x32xf32>, tensor<1xi64>) -> tensor<256xf32>
    %22 = arith.mulf %8, %reshape_5 : tensor<256xf32>
    %23 = arith.index_cast %0 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %24 = arith.index_cast %0 : i32 to index
    %25 = arith.addi %24, %c256 : index
    %26 = arith.index_cast %arg3 : i32 to index
    %27 = arith.maxsi %24, %26 : index
    %28 = arith.minsi %25, %27 : index
    %29 = arith.subi %28, %24 : index
    %extracted_slice = tensor.extract_slice %22[0] [%29] [1] : tensor<256xf32> to tensor<?xf32>
    %subview_7 = memref.subview %reinterpret_cast_6[0] [%29] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_7 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

