// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @triton_index_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %c64 = arith.constant 64 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = arith.divsi %arg12, %arg7 : i32
    %3 = arith.addi %arg8, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.muli %arg12, %c8192_i32 : i32
    %6 = arith.index_cast %5 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [64, 128], strides: [128, 1] : memref<?xbf16> to memref<64x128xbf16, strided<[128, 1], offset: ?>>
    %7 = arith.muli %arg12, %c64_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%8], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<64x128xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<64x128xbf16, strided<[128, 1], offset: ?>> to memref<64x128xbf16>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<64x128xbf16>
    %alloc_2 = memref.alloc() : memref<64xf32>
    memref.copy %reinterpret_cast_1, %alloc_2 : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32>
    %10 = bufferization.to_tensor %alloc_2 restrict writable : memref<64xf32>
    %11 = arith.muli %2, %arg8 : i32
    %12 = arith.muli %11, %c128_i32 : i32
    %13 = arith.index_cast %12 : i32 to index
    %broadcasted = linalg.broadcast ins(%10 : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [1]
    %14 = arith.muli %arg12, %arg8 : i32
    %15 = arith.index_cast %14 : i32 to index
    scf.for %arg15 = %c0_i32 to %4 step %c1_i32  : i32 {
      %16 = arith.muli %arg15, %c8192_i32 : i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = arith.addi %13, %17 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%18], sizes: [64, 128], strides: [128, 1] : memref<?xbf16> to memref<64x128xbf16, strided<[128, 1], offset: ?>>
      %19 = arith.muli %arg15, %c64_i32 : i32
      %alloc_4 = memref.alloc() : memref<64x128xbf16>
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.addi %20, %c64 : index
      %22 = arith.index_cast %arg8 : i32 to index
      %23 = arith.maxsi %20, %22 : index
      %24 = arith.minsi %21, %23 : index
      %25 = arith.subi %24, %20 : index
      %26 = arith.cmpi slt, %25, %c64 : index
      scf.if %26 {
        linalg.fill ins(%cst : bf16) outs(%alloc_4 : memref<64x128xbf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_3[0, 0] [%25, 128] [1, 1] : memref<64x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
      %subview_5 = memref.subview %alloc_4[0, 0] [%25, 128] [1, 1] : memref<64x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
      memref.copy %subview, %subview_5 : memref<?x128xbf16, strided<[128, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
      %27 = bufferization.to_tensor %alloc_4 restrict writable : memref<64x128xbf16>
      %28 = tensor.empty() : tensor<128x64xbf16>
      %transposed = linalg.transpose ins(%27 : tensor<64x128xbf16>) outs(%28 : tensor<128x64xbf16>) permutation = [1, 0]
      %29 = linalg.matmul {input_precison = "ieee"} ins(%9, %transposed : tensor<64x128xbf16>, tensor<128x64xbf16>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %30 = arith.maxnumf %29, %1 : tensor<64x64xf32>
      %31 = arith.mulf %30, %broadcasted : tensor<64x64xf32>
      %32 = tensor.empty() : tensor<64xf32>
      %33 = linalg.fill ins(%cst_0 : f32) outs(%32 : tensor<64xf32>) -> tensor<64xf32>
      %reduced = linalg.reduce ins(%31 : tensor<64x64xf32>) outs(%33 : tensor<64xf32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %35 = arith.addf %in, %init : f32
          linalg.yield %35 : f32
        }
      %34 = arith.addi %15, %20 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%34], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
      %extracted_slice = tensor.extract_slice %reduced[0] [%25] [1] : tensor<64xf32> to tensor<?xf32>
      %subview_7 = memref.subview %reinterpret_cast_6[0] [%25] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_7 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}
