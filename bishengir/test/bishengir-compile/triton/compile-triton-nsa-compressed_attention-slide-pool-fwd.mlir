// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @sliding_pool_fwd_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg16 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %1 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %2 = arith.addi %0, %c1 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%2], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %3 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %4 = arith.subi %3, %1 : i32
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %5 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%2], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %6 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %7 = arith.subi %6, %5 : i32
    %8 = arith.cmpi sge, %arg18, %7 : i32
    scf.if %8 {
    } else {
      %9 = arith.muli %1, %arg9 : i32
      %10 = arith.index_cast %9 : i32 to index
      %11 = arith.muli %arg17, %arg10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %13 = arith.addi %10, %12 : index
      %14 = arith.muli %arg18, %arg8 : i32
      %15 = arith.muli %14, %arg9 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = arith.addi %13, %16 : index
      %18 = arith.index_cast %arg9 : i32 to index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%17], sizes: [32, 128], strides: [%18, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %alloc = memref.alloc() : memref<32x128xf16>
      %19 = arith.index_cast %14 : i32 to index
      %20 = arith.addi %19, %c32 : index
      %21 = arith.index_cast %4 : i32 to index
      %22 = arith.maxsi %19, %21 : index
      %23 = arith.minsi %20, %22 : index
      %24 = arith.subi %23, %19 : index
      %25 = arith.index_cast %arg6 : i32 to index
      %26 = arith.maxsi %25, %c0 : index
      %27 = arith.minsi %26, %c128 : index
      %28 = arith.minsi %24, %c32 : index
      %29 = arith.minsi %27, %c128 : index
      %30 = arith.cmpi slt, %28, %c32 : index
      %31 = arith.cmpi slt, %29, %c128 : index
      %32 = arith.ori %30, %31 : i1
      scf.if %32 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<32x128xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_3[0, 0] [%28, %29] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_4 = memref.subview %alloc[0, 0] [%28, %29] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview, %subview_4 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %33 = bufferization.to_tensor %alloc restrict writable : memref<32x128xf16>
      %34 = tensor.empty() : tensor<128xf16>
      %35 = linalg.fill ins(%cst : f16) outs(%34 : tensor<128xf16>) -> tensor<128xf16>
      %reduced = linalg.reduce ins(%33 : tensor<32x128xf16>) outs(%35 : tensor<128xf16>) dimensions = [0]
        (%in: f16, %init: f16) {
          %48 = arith.addf %in, %init : f16
          linalg.yield %48 : f16
        }
      %36 = arith.extf %reduced : tensor<128xf16> to tensor<128xf32>
      %37 = arith.sitofp %arg7 : i32 to f32
      %38 = tensor.empty() : tensor<128xf32>
      %39 = linalg.fill ins(%37 : f32) outs(%38 : tensor<128xf32>) -> tensor<128xf32>
      %40 = arith.divf %36, %39 : tensor<128xf32>
      %41 = arith.addi %5, %arg18 : i32
      %42 = arith.muli %41, %arg11 : i32
      %43 = arith.index_cast %42 : i32 to index
      %44 = arith.muli %arg17, %arg12 : i32
      %45 = arith.index_cast %44 : i32 to index
      %46 = arith.addi %43, %45 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%46], sizes: [128], strides: [1] : memref<?xf16> to memref<128xf16, strided<[1], offset: ?>>
      %47 = arith.truncf %40 : tensor<128xf32> to tensor<128xf16>
      %extracted_slice = tensor.extract_slice %47[0] [%27] [1] : tensor<128xf16> to tensor<?xf16>
      %subview_6 = memref.subview %reinterpret_cast_5[0] [%27] [1] : memref<128xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf16>, memref<?xf16, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}
