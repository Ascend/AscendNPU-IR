// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @mla_prefill_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: f32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %c16 = arith.constant 16 : index
    %cst_0 = arith.constant 0xFF800000 : f32
    %c0_i8 = arith.constant 0 : i8
    %c16_i32 = arith.constant 16 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %2 = tensor.empty() : tensor<16x16xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<16x16xi8>) -> tensor<16x16xi8>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = tensor.empty() : tensor<16xf32>
    %6 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<16xf32>) -> tensor<16xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<16xf32>) -> tensor<16xf32>
    %8 = tensor.empty() : tensor<16x1xf32>
    %9 = linalg.fill ins(%cst_1 : f32) outs(%8 : tensor<16x1xf32>) -> tensor<16x1xf32>
    %10 = arith.muli %arg27, %c16_i32 : i32
    %11 = arith.muli %arg26, %arg8 : i32
    %12 = arith.index_cast %11 : i32 to index
    %13 = arith.index_cast %10 : i32 to index
    %14 = arith.index_cast %arg10 : i32 to index
    %15 = arith.muli %13, %14 : index
    %16 = arith.addi %12, %15 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [16, 64], strides: [%14, 1] : memref<?xbf16> to memref<16x64xbf16, strided<[?, 1], offset: ?>>
    %17 = arith.muli %arg26, %arg11 : i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.muli %arg26, %arg14 : i32
    %20 = arith.index_cast %19 : i32 to index
    %21 = arith.muli %arg26, %arg20 : i32
    %22 = arith.index_cast %21 : i32 to index
    %alloc = memref.alloc() : memref<16x64xbf16>
    %23 = arith.addi %13, %c16 : index
    %24 = arith.maxsi %13, %c16 : index
    %25 = arith.minsi %23, %24 : index
    %26 = arith.subi %25, %13 : index
    %27 = arith.cmpi slt, %26, %c16 : index
    scf.if %27 {
      linalg.fill ins(%cst : bf16) outs(%alloc : memref<16x64xbf16>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0, 0] [%26, 64] [1, 1] : memref<16x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[?, 1], offset: ?>>
    %subview_2 = memref.subview %alloc[0, 0] [%26, 64] [1, 1] : memref<16x64xbf16> to memref<?x64xbf16, strided<[64, 1]>>
    memref.copy %subview, %subview_2 : memref<?x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[64, 1]>>
    %28 = bufferization.to_tensor %alloc restrict writable : memref<16x64xbf16>
    %29 = arith.extf %28 : tensor<16x64xbf16> to tensor<16x64xf32>
    %30 = arith.index_cast %arg13 : i32 to index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%18], sizes: [16, 64], strides: [%30, 1] : memref<?xbf16> to memref<16x64xbf16, strided<[?, 1], offset: ?>>
    %alloc_4 = memref.alloc() : memref<16x64xbf16>
    memref.copy %reinterpret_cast_3, %alloc_4 : memref<16x64xbf16, strided<[?, 1], offset: ?>> to memref<16x64xbf16>
    %31 = bufferization.to_tensor %alloc_4 restrict writable : memref<16x64xbf16>
    %32 = arith.extf %31 : tensor<16x64xbf16> to tensor<16x64xf32>
    %33 = tensor.empty() : tensor<64x16xf32>
    %transposed = linalg.transpose ins(%32 : tensor<16x64xf32>) outs(%33 : tensor<64x16xf32>) permutation = [1, 0]
    %34 = linalg.matmul {input_precison = "ieee"} ins(%29, %transposed : tensor<16x64xf32>, tensor<64x16xf32>) outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %35 = linalg.fill ins(%arg7 : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %36 = arith.mulf %34, %35 : tensor<16x16xf32>
    %37 = arith.index_cast %arg22 : i32 to index
    %38 = arith.muli %13, %37 : index
    %39 = arith.addi %22, %38 : index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%39], sizes: [16, 16], strides: [%37, 1] : memref<?xi8> to memref<16x16xi8, strided<[?, 1], offset: ?>>
    %alloc_6 = memref.alloc() : memref<16x16xi8>
    %40 = arith.minsi %26, %c16 : index
    %41 = arith.cmpi slt, %40, %c16 : index
    scf.if %41 {
      linalg.fill ins(%c0_i8 : i8) outs(%alloc_6 : memref<16x16xi8>)
    } {hivm.unlikely_condition}
    %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%40, 16] [1, 1] : memref<16x16xi8, strided<[?, 1], offset: ?>> to memref<?x16xi8, strided<[?, 1], offset: ?>>
    %subview_8 = memref.subview %alloc_6[0, 0] [%40, 16] [1, 1] : memref<16x16xi8> to memref<?x16xi8, strided<[16, 1]>>
    memref.copy %subview_7, %subview_8 : memref<?x16xi8, strided<[?, 1], offset: ?>> to memref<?x16xi8, strided<[16, 1]>>
    %42 = bufferization.to_tensor %alloc_6 restrict writable : memref<16x16xi8>
    %43 = arith.cmpi ne, %42, %3 : tensor<16x16xi8>
    %44 = arith.select %43, %36, %4 : tensor<16x16xi1>, tensor<16x16xf32>
    %reduced = linalg.reduce ins(%44 : tensor<16x16xf32>) outs(%7 : tensor<16xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %64 = arith.maxnumf %in, %init : f32
        linalg.yield %64 : f32
      }
    %broadcasted = linalg.broadcast ins(%reduced : tensor<16xf32>) outs(%0 : tensor<16x16xf32>) dimensions = [1]
    %45 = arith.subf %44, %broadcasted : tensor<16x16xf32>
    %46 = math.exp %45 : tensor<16x16xf32>
    %47 = arith.subf %7, %reduced : tensor<16xf32>
    %48 = math.exp %47 : tensor<16xf32>
    %49 = arith.mulf %48, %6 : tensor<16xf32>
    %reduced_9 = linalg.reduce ins(%46 : tensor<16x16xf32>) outs(%6 : tensor<16xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %64 = arith.addf %in, %init : f32
        linalg.yield %64 : f32
      }
    %50 = arith.addf %49, %reduced_9 : tensor<16xf32>
    %expanded = tensor.expand_shape %48 [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
    %51 = arith.mulf %expanded, %9 : tensor<16x1xf32>
    %52 = tensor.empty() : tensor<16x64xf32>
    %collapsed = tensor.collapse_shape %51 [[0, 1]] : tensor<16x1xf32> into tensor<16xf32>
    %broadcasted_10 = linalg.broadcast ins(%collapsed : tensor<16xf32>) outs(%52 : tensor<16x64xf32>) dimensions = [1]
    %53 = arith.index_cast %arg16 : i32 to index
    %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%20], sizes: [16, 64], strides: [%53, 1] : memref<?xbf16> to memref<16x64xbf16, strided<[?, 1], offset: ?>>
    %alloc_12 = memref.alloc() : memref<16x64xbf16>
    memref.copy %reinterpret_cast_11, %alloc_12 : memref<16x64xbf16, strided<[?, 1], offset: ?>> to memref<16x64xbf16>
    %54 = bufferization.to_tensor %alloc_12 restrict writable : memref<16x64xbf16>
    %55 = arith.extf %54 : tensor<16x64xbf16> to tensor<16x64xf32>
    %56 = linalg.matmul {input_precison = "ieee"} ins(%46, %55 : tensor<16x16xf32>, tensor<16x64xf32>) outs(%broadcasted_10 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %broadcasted_13 = linalg.broadcast ins(%50 : tensor<16xf32>) outs(%52 : tensor<16x64xf32>) dimensions = [1]
    %57 = arith.divf %56, %broadcasted_13 : tensor<16x64xf32>
    %58 = arith.muli %arg26, %arg17 : i32
    %59 = arith.index_cast %58 : i32 to index
    %60 = arith.index_cast %arg19 : i32 to index
    %61 = arith.muli %13, %60 : index
    %62 = arith.addi %59, %61 : index
    %reinterpret_cast_14 = memref.reinterpret_cast %arg5 to offset: [%62], sizes: [16, 64], strides: [%60, 1] : memref<?xbf16> to memref<16x64xbf16, strided<[?, 1], offset: ?>>
    %63 = arith.truncf %57 : tensor<16x64xf32> to tensor<16x64xbf16>
    %extracted_slice = tensor.extract_slice %63[0, 0] [%26, 64] [1, 1] : tensor<16x64xbf16> to tensor<?x64xbf16>
    %subview_15 = memref.subview %reinterpret_cast_14[0, 0] [%26, 64] [1, 1] : memref<16x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_15 : (tensor<?x64xbf16>, memref<?x64xbf16, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}
