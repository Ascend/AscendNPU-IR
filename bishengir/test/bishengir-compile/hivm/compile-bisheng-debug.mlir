// REQUIRES: enable-lir-compile
// RUN: bishengir-compile --target=Ascend950PR_9579 --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=True --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func private @triton_print_0(tensor<1x32xf8E4M3FN>) attributes {hex = false, prefix = " lhs: "}
  func.func @dot_scale_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf8E4M3FN> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf8E4M3FN> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1, 32], strides: [32, 1] : memref<?xf8E4M3FN> to memref<1x32xf8E4M3FN, strided<[32, 1]>>
    %alloc = memref.alloc() : memref<1x32xf8E4M3FN>
    memref.copy %reinterpret_cast, %alloc : memref<1x32xf8E4M3FN, strided<[32, 1]>> to memref<1x32xf8E4M3FN>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<1x32xf8E4M3FN>
    call @triton_print_0(%2) : (tensor<1x32xf8E4M3FN>) -> ()
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<?xi8> to memref<1x1xi8, strided<[1, 1]>>
    %alloc_1 = memref.alloc() : memref<1x1xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<1x1xi8, strided<[1, 1]>> to memref<1x1xi8>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x1xi8>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [32, 1], strides: [1, 1] : memref<?xf8E4M3FN> to memref<32x1xf8E4M3FN, strided<[1, 1]>>
    %alloc_3 = memref.alloc() : memref<32x1xf8E4M3FN>
    memref.copy %reinterpret_cast_2, %alloc_3 : memref<32x1xf8E4M3FN, strided<[1, 1]>> to memref<32x1xf8E4M3FN>
    %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<32x1xf8E4M3FN>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<?xi8> to memref<1x1xi8, strided<[1, 1]>>
    %alloc_5 = memref.alloc() : memref<1x1xi8>
    memref.copy %reinterpret_cast_4, %alloc_5 : memref<1x1xi8, strided<[1, 1]>> to memref<1x1xi8>
    %5 = bufferization.to_tensor %alloc_5 restrict writable : memref<1x1xi8>
    %6 = hfusion.matmul_mx ins(%2, %4, %3, %5 : tensor<1x32xf8E4M3FN>, tensor<32x1xf8E4M3FN>, tensor<1x1xi8>, tensor<1x1xi8>) outs(%1 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<?xf32> to memref<1x1xf32, strided<[1, 1]>>
    bufferization.materialize_in_destination %6 in writable %reinterpret_cast_6 : (tensor<1x1xf32>, memref<1x1xf32, strided<[1, 1]>>) -> ()
    return
  }
}
