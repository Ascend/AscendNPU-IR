// RUN: bishengir-compile -enable-triton-kernel-compile -enable-lir-compile=false -enable-hfusion-compile -block-dim=1 -target=Ascend310B4 -enable-direct-hivm-lowering %s
module {
  func.func @_attn_fwd(%arg1: memref<?xf16> {tt.divisibility = 16 : i32}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<64x16xf32> {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "mix"} {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<64x16xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x16xf32>) -> tensor<64x16xf32>

    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [64, 64], strides: [%c64, %c1] : memref<?xf16> to memref<64x64xf16, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() : memref<64x64xf16>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xf16, strided<[?, ?], offset: ?>> to memref<64x64xf16>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>

    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%c0], sizes: [16, 64], strides: [%c64, %c1] : memref<?xf16> to memref<16x64xf16, strided<[?, ?], offset: ?>>
    %alloc_1 = memref.alloc() : memref<16x64xf16>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<16x64xf16, strided<[?, ?], offset: ?>> to memref<16x64xf16>
    %5 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x64xf16>

    %6 = tensor.empty() : tensor<64x16xf16>
    %transposed = linalg.transpose ins(%5 : tensor<16x64xf16>) outs(%6 : tensor<64x16xf16>) permutation = [1, 0]
    %7 = linalg.matmul ins(%4, %transposed : tensor<64x64xf16>, tensor<64x16xf16>) outs(%1 : tensor<64x16xf32>) -> tensor<64x16xf32>
    bufferization.materialize_in_destination %7 in writable %arg4 : (tensor<64x16xf32>, memref<64x16xf32>) -> ()
    return
  }
}

