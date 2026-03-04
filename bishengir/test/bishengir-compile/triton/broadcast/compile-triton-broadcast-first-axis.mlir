// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -enable-hivm-inject-barrier-all-sync -enable-triton-kernel-compile %s | FileCheck %s
// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -enable-triton-kernel-compile %s | FileCheck %s

// CHECK: LLVMDialectModule
module {
  func.func @test_broadcast_first_axis__kernel0(%arg0: memref<?xf32> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {global_kernel = "local"} {
    %c4096_i64 = arith.constant 4096 : i64
    %cst = arith.constant dense<[1, 4, 8]> : tensor<3xi64>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1]>>
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
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4096], strides: [1] : memref<?xf32> to memref<4096xf32, strided<[1]>>
    bufferization.materialize_in_destination %reshape_0 in writable %reinterpret_cast_1 : (tensor<4096xf32>, memref<4096xf32, strided<[1]>>) -> ()
    return
  }

}

