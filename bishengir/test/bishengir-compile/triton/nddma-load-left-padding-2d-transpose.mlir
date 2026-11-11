// REQUIRES: hivmc
// White-box test for issue #748 (NDDMA 2D load with left padding): a 2D f32
// GM-to-UB load whose GM side is implicitly transposed (strides [1, 39]) and
// whose UB destination is left-padded (offset 13 elements, linalg.fill 1.0
// providing the PadValue) must reach the padded vector DMA template with
// pad_mode=PadValue (2), pad_value=1.0 and left_padding_num=5 (13 floored to
// the 32B block of 8 f32 elements).
//
// Only the call-site arguments are asserted: the saved kernel.ll carries the
// argument-packing wrapper alone, while the dispatch lives in the template
// bitcode linked at device compile time. Post-linking kernel.ll against the
// template bitcode (llvm-link -S) does expose the padded NDDMA path for
// FileCheck, but release CANN toolchains build llvm-link with textual IR
// printing disabled, so that assertion layer cannot run on the CI and is kept
// out of the test.
//
// RUN: mkdir -p %t.d
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 --save-temps=%t.d -o %t.d/kernel.o
// RUN: FileCheck --input-file=%t.d/kernel.ll %s
//
// CHECK: call void @load_gm_to_ubuf_2d_float({{.*}}i64 4, i64 26, i64 1, i64 39, {{.*}}i64 13, i64 4, i64 26, i64 64, i64 1, i32 2, float 1.000000e+00, i64 5, i64 0, i32 0)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {
  func.func @triton_poi_fused__to_copy_copy_slice_zeros_26(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c39 = arith.constant 39 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %4 = arith.muli %arg11, %arg7 : i32
    %5 = arith.addi %arg7, %c3_i32 : i32
    %6 = arith.divsi %5, %c4_i32 : i32
    scf.for %arg14 = %c0_i32 to %6 step %c1_i32 : i32 {
      %9 = arith.muli %arg14, %c4_i32 : i32
      %10 = arith.addi %4, %9 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.muli %11, %c39 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%12], sizes: [26, 39], strides: [1, 39] : memref<?xf32> to memref<26x39xf32, strided<[1, 39], offset: ?>>
      %alloc = memref.alloc() : memref<4x64xf32>
      linalg.fill ins(%cst : f32) outs(%alloc : memref<4x64xf32>)
      %subview = memref.subview %reinterpret_cast[0, 0] [4, 26] [1, 1] : memref<26x39xf32, strided<[1, 39], offset: ?>> to memref<4x26xf32, strided<[1, 39], offset: ?>>
      %subview_0 = memref.subview %alloc[0, 13] [4, 26] [1, 1] : memref<4x64xf32> to memref<4x26xf32, strided<[64, 1], offset: 13>>
      memref.copy %subview, %subview_0 : memref<4x26xf32, strided<[1, 39], offset: ?>> to memref<4x26xf32, strided<[64, 1], offset: 13>>
      %20 = bufferization.to_tensor %alloc restrict writable : memref<4x64xf32>
      %extracted_slice = tensor.extract_slice %20[0, 0] [4, 39] [1, 1] : tensor<4x64xf32> to tensor<4x39xf32>
      %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%12], sizes: [4, 64], strides: [39, 1] : memref<?xf32> to memref<4x64xf32, strided<[39, 1], offset: ?>>
      %subview_10 = memref.subview %reinterpret_cast_8[0, 0] [4, 39] [1, 1] : memref<4x64xf32, strided<[39, 1], offset: ?>> to memref<4x39xf32, strided<[39, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_10 : (tensor<4x39xf32>, memref<4x39xf32, strided<[39, 1], offset: ?>>) -> ()
    }
    return
  }
}
