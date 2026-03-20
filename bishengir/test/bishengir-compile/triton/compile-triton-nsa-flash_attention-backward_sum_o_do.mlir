// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @backward_sum_o_do(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c128_i32 = arith.constant 128 : i32
    %0 = arith.muli %arg15, %c128_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.index_cast %arg7 : i32 to index
    %3 = arith.muli %1, %2 : index
    %4 = arith.muli %arg16, %arg8 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.addi %3, %5 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [128, 128], strides: [%2, 1] : memref<?xf16> to memref<128x128xf16, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<128x128xf16>
    %7 = arith.addi %1, %c128 : index
    %8 = arith.index_cast %arg5 : i32 to index
    %9 = arith.maxsi %1, %8 : index
    %10 = arith.minsi %7, %9 : index
    %11 = arith.subi %10, %1 : index
    %12 = arith.index_cast %arg6 : i32 to index
    %13 = arith.maxsi %12, %c0 : index
    %14 = arith.minsi %13, %c128 : index
    %15 = arith.minsi %11, %c128 : index
    %16 = arith.minsi %14, %c128 : index
    %17 = arith.cmpi slt, %15, %c128 : index
    %18 = arith.cmpi slt, %16, %c128 : index
    %19 = arith.ori %17, %18 : i1
    scf.if %19 {
      linalg.fill ins(%cst_0 : f16) outs(%alloc : memref<128x128xf16>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0, 0] [%15, %16] [1, 1] : memref<128x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %subview_1 = memref.subview %alloc[0, 0] [%15, %16] [1, 1] : memref<128x128xf16> to memref<?x?xf16, strided<[128, 1]>>
    memref.copy %subview, %subview_1 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
    %20 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf16>
    %21 = arith.extf %20 : tensor<128x128xf16> to tensor<128x128xf32>
    %22 = arith.index_cast %arg9 : i32 to index
    %23 = arith.muli %1, %22 : index
    %24 = arith.muli %arg16, %arg10 : i32
    %25 = arith.index_cast %24 : i32 to index
    %26 = arith.addi %23, %25 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [128, 128], strides: [%22, 1] : memref<?xf16> to memref<128x128xf16, strided<[?, 1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<128x128xf16>
    scf.if %19 {
      linalg.fill ins(%cst_0 : f16) outs(%alloc_3 : memref<128x128xf16>)
    } {hivm.unlikely_condition}
    %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%15, %16] [1, 1] : memref<128x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %subview_5 = memref.subview %alloc_3[0, 0] [%15, %16] [1, 1] : memref<128x128xf16> to memref<?x?xf16, strided<[128, 1]>>
    memref.copy %subview_4, %subview_5 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
    %27 = bufferization.to_tensor %alloc_3 restrict writable : memref<128x128xf16>
    %28 = arith.extf %27 : tensor<128x128xf16> to tensor<128x128xf32>
    %29 = arith.mulf %21, %28 : tensor<128x128xf32>
    %30 = tensor.empty() : tensor<128xf32>
    %31 = linalg.fill ins(%cst : f32) outs(%30 : tensor<128xf32>) -> tensor<128xf32>
    %reduced = linalg.reduce ins(%29 : tensor<128x128xf32>) outs(%31 : tensor<128xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %35 = arith.addf %in, %init : f32
        linalg.yield %35 : f32
      }
    %32 = arith.muli %arg16, %arg11 : i32
    %33 = arith.index_cast %32 : i32 to index
    %34 = arith.addi %33, %1 : index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%34], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %reduced[0] [%11] [1] : tensor<128xf32> to tensor<?xf32>
    %subview_7 = memref.subview %reinterpret_cast_6[0] [%11] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_7 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
