// REQUIRES: hivmc
// UNSUPPORTED: bishengir_published
//
// Exercise bf16 dot with a fractal-A operand whose GM view is strided
// (the zN tile does not span the whole packed M1 extent), so the direct
// GM->L1 fractal load cannot be flattened to a single 1D copy. The regbase
// pipeline must decompose it into nested 1D copies that instantiate
// _mlir_ciface_load_gm_to_cbuf_1d_bfloat16_t from
// bishengir/lib/Template/lib/RegBase/Cube/compat/DMA/Cbuf/Copy1D.cpp; the
// device link must provide the symbol.

// RUN: bishengir-compile %s --target=Ascend950PR_9599 --enable-auto-multi-buffer=True \
// RUN:   --enable-auto-bind-sub-block=True --disable-ffts \
// RUN:   --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop \
// RUN:   --enable-hfusion-compile=true --enable-triton-kernel-compile=true -o %t

module attributes {hacc.target = #hacc.target<"Ascend950PR_9599">, ssbuffer.insertionOptimization} {
  func.func @k_bf16_strided_fractal_a(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xbf16>, %arg4: memref<?xbf16>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    // zN tile [2, 2, 16, 16] out of a [2, 4, 16, 16] packed GM tensor:
    // stride0 = 1024 != 2 * 256, so the GM view is genuinely strided.
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 2, 16, 16], strides: [1024, 256, 16, 1] : memref<?xbf16> to memref<2x2x16x16xbf16, strided<[1024, 256, 16, 1]>>
    %alloc = memref.alloc() : memref<2x2x16x16xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<2x2x16x16xbf16, strided<[1024, 256, 16, 1]>> to memref<2x2x16x16xbf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<2x2x16x16xbf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xbf16> to memref<32x32xbf16, strided<[32, 1]>>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<32x32xbf16>
    %2 = hivm.hir.convert_layout %0 output_shape [32, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<2x2x16x16xbf16>) -> tensor<32x32xbf16>
    %3 = tensor.empty() : tensor<32x32xf32>
    %4 = linalg.matmul {input_precision = "ieee"} ins(%2, %1 : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xf32> to memref<32x32xf32, strided<[32, 1]>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<32x32xf32>, memref<32x32xf32, strided<[32, 1]>>) -> ()
    return
  }
}
