// REQUIRES: hivmc
// UNSUPPORTED: bishengir_published
//
// Exercise int8 dot with fractal-C output (int32 L0C->GM delivery), which
// instantiates _mlir_ciface_fixpipe_normal_int32_t_to_int32_t_4d_to_4d_gm from
// bishengir/lib/Template/lib/RegBase/Cube/Fixpipe.cpp; the device link must
// provide the symbol.

// RUN: bishengir-compile %s --target=Ascend950PR_9599 --enable-auto-multi-buffer=True \
// RUN:   --enable-auto-bind-sub-block=True --disable-ffts \
// RUN:   --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop \
// RUN:   --enable-hfusion-compile=true --enable-triton-kernel-compile=true -o %t

module attributes {hacc.target = #hacc.target<"Ascend950PR_9599">, ssbuffer.insertionOptimization} {
  func.func @k_i8(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi32>, %arg3: memref<?xi8>, %arg4: memref<?xi8>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xi8> to memref<32x32xi8, strided<[32, 1]>>
    %alloc = memref.alloc() : memref<32x32xi8>
    memref.copy %reinterpret_cast, %alloc : memref<32x32xi8, strided<[32, 1]>> to memref<32x32xi8>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<32x32xi8>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xi8> to memref<32x32xi8, strided<[32, 1]>>
    %alloc_1 = memref.alloc() : memref<32x32xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<32x32xi8, strided<[32, 1]>> to memref<32x32xi8>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<32x32xi8>
    %2 = tensor.empty() : tensor<32x32xi32>
    %3 = linalg.matmul {input_precision = "ieee"} ins(%0, %1 : tensor<32x32xi8>, tensor<32x32xi8>) outs(%2 : tensor<32x32xi32>) -> tensor<32x32xi32>
    %4 = hivm.hir.convert_layout %3 output_shape [2, 2, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<32x32xi32>) -> tensor<2x2x16x16xi32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [2, 2, 16, 16], strides: [512, 256, 16, 1] : memref<?xi32> to memref<2x2x16x16xi32, strided<[512, 256, 16, 1]>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<2x2x16x16xi32>, memref<2x2x16x16xi32, strided<[512, 256, 16, 1]>>) -> ()
    return
  }
}
