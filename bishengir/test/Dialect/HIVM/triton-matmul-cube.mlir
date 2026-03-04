// RUN: bishengir-compile -enable-triton-kernel-compile -enable-lir-compile=false -enable-hfusion-compile -block-dim=1 -target=Ascend310B4  %s | FileCheck %s

module {
  func.func @triton_dot(%arg0: memref<?xi8>, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "mix", hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1]>>
    %alloc = memref.alloc() : memref<32x64xf16>
    // CHECK: nd2nz_half
    memref.copy %reinterpret_cast, %alloc : memref<32x64xf16, strided<[64, 1]>> to memref<32x64xf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<32x64xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64, 16], strides: [16, 1] : memref<?xf16> to memref<64x16xf16, strided<[16, 1]>>
    %alloc_1 = memref.alloc() : memref<64x16xf16>
    // CHECK: nd2nz_half
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<64x16xf16, strided<[16, 1]>> to memref<64x16xf16>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<64x16xf16>
    %alloc_3 = tensor.empty() : tensor<32x16xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%alloc_3 : tensor<32x16xf32>) -> tensor<32x16xf32>
    // CHECK: mma_tile_half_to_float
    %3 = linalg.matmul {input_precison = "ieee"} ins(%0, %1 : tensor<32x64xf16>, tensor<64x16xf16>) outs(%2 : tensor<32x16xf32>) -> tensor<32x16xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32, 16], strides: [16, 1] : memref<?xf32> to memref<32x16xf32, strided<[16, 1]>>
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast_4 : (tensor<32x16xf32>, memref<32x16xf32, strided<[16, 1]>>) -> ()
    return
  }
}