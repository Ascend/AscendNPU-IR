// REQUIRES: hivmc
// UNSUPPORTED: bishengir_published
//
// Exercise fp8 dot with fractal-A operand (direct GM->L1 fractal load), which
// instantiates _mlir_ciface_load_gm_to_cbuf_1d_float8_e4m3_t from
// bishengir/lib/Template/lib/RegBase/Cube/compat/DMA/Cbuf/Copy1D.cpp; the
// device link must provide the symbol.

// RUN: bishengir-compile %s --target=Ascend950PR_9599 --enable-auto-multi-buffer=True \
// RUN:   --enable-auto-bind-sub-block=True --disable-ffts \
// RUN:   --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop \
// RUN:   --enable-hfusion-compile=true --enable-triton-kernel-compile=true -o %t

module attributes {hacc.target = #hacc.target<"Ascend950PR_9599">, ssbuffer.insertionOptimization} {
  func.func @k_fp8(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf8E4M3FN>, %arg4: memref<?xf8E4M3FN>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 2, 16, 32], strides: [1024, 512, 32, 1] : memref<?xf8E4M3FN> to memref<1x2x16x32xf8E4M3FN, strided<[1024, 512, 32, 1]>>
    %alloc = memref.alloc() : memref<1x2x16x32xf8E4M3FN>
    memref.copy %reinterpret_cast, %alloc : memref<1x2x16x32xf8E4M3FN, strided<[1024, 512, 32, 1]>> to memref<1x2x16x32xf8E4M3FN>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<1x2x16x32xf8E4M3FN>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xf8E4M3FN> to memref<32x32xf8E4M3FN, strided<[32, 1]>>
    %alloc_1 = memref.alloc() : memref<32x32xf8E4M3FN>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<32x32xf8E4M3FN, strided<[32, 1]>> to memref<32x32xf8E4M3FN>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<32x32xf8E4M3FN>
    %2 = hivm.hir.convert_layout %0 output_shape [32, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>} : (tensor<1x2x16x32xf8E4M3FN>) -> tensor<32x32xf8E4M3FN>
    %3 = tensor.empty() : tensor<32x32xf32>
    %4 = linalg.matmul {input_precision = "ieee"} ins(%2, %1 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xf32> to memref<32x32xf32, strided<[32, 1]>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<32x32xf32>, memref<32x32xf32, strided<[32, 1]>>) -> ()
    return
  }
}
