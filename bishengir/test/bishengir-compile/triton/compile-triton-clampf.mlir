// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -enable-hivm-inject-barrier-all-sync -enable-triton-kernel-compile %s | FileCheck %s

// CHECK: LLVMDialectModule
module {
  func.func @test_clampf(%arg0: memref<?xf32> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {global_kernel = "local"} {
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %0 = arith.muli %arg7, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1024xf32>
    memref.copy %reinterpret_cast, %alloc : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
    %3 = affine.load %reinterpret_cast_0[0] : memref<1xf32, strided<[1], offset: ?>>
    %4 = tensor.empty() : tensor<1024xf32>
    %5 = linalg.fill ins(%3 : f32) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
    %6 = arith.minnumf %2, %5 : tensor<1024xf32>
    %7 = arith.index_cast %0 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%7], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %6 in writable %reinterpret_cast_1 : (tensor<1024xf32>, memref<1024xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}