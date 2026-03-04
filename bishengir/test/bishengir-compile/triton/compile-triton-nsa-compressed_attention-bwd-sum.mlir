// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @backward_sum_o_do(%arg0: memref<?xi8> , %arg1: memref<?xi8> , %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} , %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} , %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32} , %arg5: i32 , %arg6: i32 {tt.divisibility = 16 : i32} , %arg7: i32 {tt.divisibility = 16 : i32} , %arg8: i32 {tt.divisibility = 16 : i32} , %arg9: i32 {tt.divisibility = 16 : i32} , %arg10: i32 {tt.divisibility = 16 : i32} , %arg11: i32 , %arg12: i32 {tt.divisibility = 16 : i32} , %arg13: i32 , %arg14: i32 , %arg15: i32 , %arg16: i32 , %arg17: i32 , %arg18: i32 ) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c128_i32 = arith.constant 128 : i32
    %0 = arith.addi %arg16, %arg12 : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.index_cast %arg7 : i32 to index
    %4 = arith.muli %2, %3 : index
    %5 = arith.muli %arg17, %arg8 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.addi %4, %6 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%7], sizes: [128, 128], strides: [%3, 1] : memref<?xf16> to memref<128x128xf16, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<128x128xf16>
    %8 = arith.addi %2, %c128 : index
    %9 = arith.index_cast %arg5 : i32 to index
    %10 = arith.maxsi %2, %9 : index
    %11 = arith.minsi %8, %10 : index
    %12 = arith.subi %11, %2 : index
    %13 = arith.index_cast %arg6 : i32 to index
    %14 = arith.maxsi %13, %c0 : index
    %15 = arith.minsi %14, %c128 : index
    %16 = arith.minsi %12, %c128 : index
    %17 = arith.minsi %15, %c128 : index
    %18 = arith.cmpi slt, %16, %c128 : index
    %19 = arith.cmpi slt, %17, %c128 : index
    %20 = arith.ori %18, %19 : i1
    scf.if %20 {
      linalg.fill ins(%cst_0 : f16) outs(%alloc : memref<128x128xf16>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0, 0] [%16, %17] [1, 1] : memref<128x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %subview_1 = memref.subview %alloc[0, 0] [%16, %17] [1, 1] : memref<128x128xf16> to memref<?x?xf16, strided<[128, 1]>>
    memref.copy %subview, %subview_1 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
    %21 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf16>
    %22 = arith.extf %21 : tensor<128x128xf16> to tensor<128x128xf32>
    %23 = arith.index_cast %arg9 : i32 to index
    %24 = arith.muli %2, %23 : index
    %25 = arith.muli %arg17, %arg10 : i32
    %26 = arith.index_cast %25 : i32 to index
    %27 = arith.addi %24, %26 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [128, 128], strides: [%23, 1] : memref<?xf16> to memref<128x128xf16, strided<[?, 1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<128x128xf16>
    scf.if %20 {
      linalg.fill ins(%cst_0 : f16) outs(%alloc_3 : memref<128x128xf16>)
    } {hivm.unlikely_condition}
    %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%16, %17] [1, 1] : memref<128x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %subview_5 = memref.subview %alloc_3[0, 0] [%16, %17] [1, 1] : memref<128x128xf16> to memref<?x?xf16, strided<[128, 1]>>
    memref.copy %subview_4, %subview_5 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
    %28 = bufferization.to_tensor %alloc_3 restrict writable : memref<128x128xf16>
    %29 = arith.extf %28 : tensor<128x128xf16> to tensor<128x128xf32>
    %30 = arith.mulf %22, %29 : tensor<128x128xf32>
    %31 = tensor.empty() : tensor<128xf32>
    %32 = linalg.fill ins(%cst : f32) outs(%31 : tensor<128xf32>) -> tensor<128xf32>
    %reduced = linalg.reduce ins(%30 : tensor<128x128xf32>) outs(%32 : tensor<128xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %36 = arith.addf %in, %init : f32
        linalg.yield %36 : f32
      }
    %33 = arith.muli %arg17, %arg11 : i32
    %34 = arith.index_cast %33 : i32 to index
    %35 = arith.addi %34, %2 : index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%35], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %reduced[0] [%12] [1] : tensor<128xf32> to tensor<?xf32>
    %subview_7 = memref.subview %reinterpret_cast_6[0] [%12] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_7 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

