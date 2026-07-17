// REQUIRES: enable-lir-compile
// RUN: bishengir-compile %s --target=Ascend950PR_9589 --enable-auto-multi-buffer=true --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-lir-compile=false -o %t


#loc = loc("test_atomic_xchg_op_fp8_1.py":176:0)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @triton_atomic_xchg_5D(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf8E4M3FN> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf8E4M3FN> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: memref<?xf8E4M3FN> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c6144 = arith.constant 6144 : index
    %c48 = arith.constant 48 : index
    %c24576_i32 = arith.constant 24576 : i32
    %c4_i32 = arith.constant 4 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = arith.muli %arg9, %c4_i32 : i32
    %1 = arith.muli %arg10, %c128_i32 : i32
    %2 = arith.muli %arg8, %c24576_i32 : i32
    %3 = arith.index_cast %1 : i32 to index
    %4 = arith.muli %3, %c48 : index
    %5 = arith.index_cast %0 : i32 to index
    %6 = arith.muli %5, %c6144 : index
    %7 = arith.index_cast %2 : i32 to index
    %8 = arith.addi %7, %6 : index
    %9 = arith.addi %8, %4 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [1, 4, 128, 16, 3], strides: [24576, 6144, 48, 3, 1] : memref<?xf8E4M3FN> to memref<1x4x128x16x3xf8E4M3FN, strided<[24576, 6144, 48, 3, 1], offset: ?>>
    %alloc = memref.alloc() : memref<1x4x128x16x3xf8E4M3FN>
    memref.copy %reinterpret_cast, %alloc : memref<1x4x128x16x3xf8E4M3FN, strided<[24576, 6144, 48, 3, 1], offset: ?>> to memref<1x4x128x16x3xf8E4M3FN>
    %10 = bufferization.to_tensor %alloc restrict writable : memref<1x4x128x16x3xf8E4M3FN>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [1, 4, 128, 16, 3], strides: [24576, 6144, 48, 3, 1] : memref<?xf8E4M3FN> to memref<1x4x128x16x3xf8E4M3FN, strided<[24576, 6144, 48, 3, 1], offset: ?>>
    %cast = memref.cast %alloc : memref<1x4x128x16x3xf8E4M3FN> to memref<1x4x128x16x3xf8E4M3FN, strided<[24576, 6144, 48, 3, 1], offset: ?>>
    hfusion.atomic_xchg ins(%cast : memref<1x4x128x16x3xf8E4M3FN, strided<[24576, 6144, 48, 3, 1], offset: ?>>) outs(%reinterpret_cast_0 : memref<1x4x128x16x3xf8E4M3FN, strided<[24576, 6144, 48, 3, 1], offset: ?>>)
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%9], sizes: [1, 4, 128, 16, 3], strides: [24576, 6144, 48, 3, 1] : memref<?xf8E4M3FN> to memref<1x4x128x16x3xf8E4M3FN, strided<[24576, 6144, 48, 3, 1], offset: ?>>
    bufferization.materialize_in_destination %10 in writable %reinterpret_cast_1 : (tensor<1x4x128x16x3xf8E4M3FN>, memref<1x4x128x16x3xf8E4M3FN, strided<[24576, 6144, 48, 3, 1], offset: ?>>) -> ()
    return
  }
}
