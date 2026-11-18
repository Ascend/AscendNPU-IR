// REQUIRES: hivmc
// UNSUPPORTED: bishengir_published
// White-box regression test for issue #503: a 2D B64 GM-to-UB load whose
// logical UB destination is NOT 32B aligned (offset 13 int64 elements) must
// still reach the vector DMA template with pad_mode=PadValue and
// left_padding_num=1, and the B64 padding repair flag fences must be emitted.
//
// RUN: mkdir -p %t.d
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 --save-temps=%t.d -o %t.d/kernel.o
// RUN: FileCheck --input-file=%t.d/kernel.ll %s
//
// CHECK: call void @load_gm_to_ubuf_2d_int64_t({{.*}}, i64 26, i64 39, i64 1, {{.*}}, i64 13, {{.*}}, i64 26, i64 64, i64 1, i32 2, i64 0, i64 1, i32 0)
// CHECK: call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
// CHECK: call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {
  func.func @triton_poi_fused__to_copy_copy_slice_zeros_26(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c39 = arith.constant 39 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x13xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x13xf32>) -> tensor<4x13xf32>
    %2 = tensor.empty() : tensor<4x64xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x64xf32>) -> tensor<4x64xf32>
    %4 = arith.muli %arg11, %arg7 : i32
    %5 = arith.addi %arg7, %c3_i32 : i32
    %6 = arith.divsi %5, %c4_i32 : i32
    %7 = arith.addi %arg7, %4 : i32
    %8 = arith.minsi %7, %arg5 : i32
    scf.for %arg14 = %c0_i32 to %6 step %c1_i32  : i32 {
      %9 = arith.muli %arg14, %c4_i32 : i32
      %10 = arith.addi %4, %9 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.muli %11, %c39 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%12], sizes: [4, 64], strides: [39, 1] : memref<?xi64> to memref<4x64xi64, strided<[39, 1], offset: ?>>
      %alloc = memref.alloc() : memref<4x64xi64>
      %13 = arith.addi %11, %c4 : index
      %14 = arith.index_cast %8 : i32 to index
      %15 = arith.maxsi %11, %14 : index
      %16 = arith.minsi %13, %15 : index
      %17 = arith.subi %16, %11 : index
      %18 = arith.minsi %17, %c4 : index
      %19 = arith.maxsi %18, %c0 : index
      linalg.fill ins(%c0_i64 : i64) outs(%alloc : memref<4x64xi64>)
      %subview = memref.subview %reinterpret_cast[0, 13] [%19, 26] [1, 1] : memref<4x64xi64, strided<[39, 1], offset: ?>> to memref<?x26xi64, strided<[39, 1], offset: ?>>
      %subview_0 = memref.subview %alloc[0, 13] [%19, 26] [1, 1] : memref<4x64xi64> to memref<?x26xi64, strided<[64, 1], offset: 13>>
      memref.copy %subview, %subview_0 : memref<?x26xi64, strided<[39, 1], offset: ?>> to memref<?x26xi64, strided<[64, 1], offset: 13>>
      %20 = bufferization.to_tensor %alloc restrict writable : memref<4x64xi64>
      %21 = arith.sitofp %20 : tensor<4x64xi64> to tensor<4x64xf32>
      %inserted_slice = tensor.insert_slice %1 into %21[0, 0] [4, 13] [1, 1] : tensor<4x13xf32> into tensor<4x64xf32>
      %extracted_slice = tensor.extract_slice %inserted_slice[0, 0] [4, 39] [1, 1] : tensor<4x64xf32> to tensor<4x39xf32>
      %inserted_slice_1 = tensor.insert_slice %extracted_slice into %3[0, 0] [4, 39] [1, 1] : tensor<4x39xf32> into tensor<4x64xf32>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%12], sizes: [4, 64], strides: [39, 1] : memref<?xf32> to memref<4x64xf32, strided<[39, 1], offset: ?>>
      %alloc_3 = memref.alloc() : memref<4x64xf32>
      linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<4x64xf32>)
      %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%19, 13] [1, 1] : memref<4x64xf32, strided<[39, 1], offset: ?>> to memref<?x13xf32, strided<[39, 1], offset: ?>>
      %subview_5 = memref.subview %alloc_3[0, 0] [%19, 13] [1, 1] : memref<4x64xf32> to memref<?x13xf32, strided<[64, 1]>>
      memref.copy %subview_4, %subview_5 : memref<?x13xf32, strided<[39, 1], offset: ?>> to memref<?x13xf32, strided<[64, 1]>>
      %22 = bufferization.to_tensor %alloc_3 restrict writable : memref<4x64xf32>
      %extracted_slice_6 = tensor.extract_slice %22[0, 0] [4, 13] [1, 1] : tensor<4x64xf32> to tensor<4x13xf32>
      %inserted_slice_7 = tensor.insert_slice %extracted_slice_6 into %inserted_slice_1[0, 0] [4, 13] [1, 1] : tensor<4x13xf32> into tensor<4x64xf32>
      %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%12], sizes: [4, 64], strides: [39, 1] : memref<?xf32> to memref<4x64xf32, strided<[39, 1], offset: ?>>
      %extracted_slice_9 = tensor.extract_slice %inserted_slice_7[0, 0] [%19, 39] [1, 1] : tensor<4x64xf32> to tensor<?x39xf32>
      %subview_10 = memref.subview %reinterpret_cast_8[0, 0] [%19, 39] [1, 1] : memref<4x64xf32, strided<[39, 1], offset: ?>> to memref<?x39xf32, strided<[39, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_9 in writable %subview_10 : (tensor<?x39xf32>, memref<?x39xf32, strided<[39, 1], offset: ?>>) -> ()
    }
    return
  }
}
