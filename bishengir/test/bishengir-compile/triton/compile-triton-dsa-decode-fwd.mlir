// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-flatten=False --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @dsa_decode_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFF800000 : f32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 0.0721687824 : f32
    %cst_2 = arith.constant 9.983770e-07 : bf16
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x256xf32>
    %1 = linalg.fill ins(%cst_4 : f32) outs(%0 : tensor<1x256xf32>) -> tensor<1x256xf32>
    %2 = tensor.empty() : tensor<16x256xbf16>
    %3 = linalg.fill ins(%cst_3 : bf16) outs(%2 : tensor<16x256xbf16>) -> tensor<16x256xbf16>
    %4 = tensor.empty() : tensor<16x64xbf16>
    %5 = linalg.fill ins(%cst_3 : bf16) outs(%4 : tensor<16x64xbf16>) -> tensor<16x64xbf16>
    %6 = linalg.fill ins(%cst_2 : bf16) outs(%2 : tensor<16x256xbf16>) -> tensor<16x256xbf16>
    %7 = linalg.fill ins(%cst_2 : bf16) outs(%4 : tensor<16x64xbf16>) -> tensor<16x64xbf16>
    %8 = tensor.empty() : tensor<1x16xf32>
    %9 = linalg.fill ins(%cst_4 : f32) outs(%8 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %10 = linalg.fill ins(%cst_1 : f32) outs(%8 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %11 = tensor.empty() : tensor<1xf32>
    %12 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<1xf32>) -> tensor<1xf32>
    %13 = tensor.empty() : tensor<1x64xf32>
    %14 = linalg.fill ins(%cst_4 : f32) outs(%13 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %15 = linalg.fill ins(%cst : f32) outs(%11 : tensor<1xf32>) -> tensor<1xf32>
    %16 = tensor.empty() : tensor<1x1xf32>
    %17 = linalg.fill ins(%cst_4 : f32) outs(%16 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %18 = arith.divsi %arg26, %c128_i32 : i32
    %19 = arith.remsi %arg26, %c128_i32 : i32
    %20 = arith.muli %18, %arg10 : i32
    %21 = arith.muli %19, %arg11 : i32
    %22 = arith.addi %20, %21 : i32
    %23 = arith.index_cast %22 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [128], strides: [1] : memref<?xbf16> to memref<128xbf16, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<128xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<128xbf16, strided<[1], offset: ?>> to memref<128xbf16>
    %24 = bufferization.to_tensor %alloc restrict writable : memref<128xbf16>
    %25 = arith.muli %19, %c256_i32 : i32
    %26 = arith.muli %25, %arg14 : i32
    %27 = arith.index_cast %26 : i32 to index
    %28 = arith.index_cast %arg14 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%27], sizes: [128, 256], strides: [%28, 1] : memref<?xbf16> to memref<128x256xbf16, strided<[?, 1], offset: ?>>
    %alloc_6 = memref.alloc() : memref<128x256xbf16>
    memref.copy %reinterpret_cast_5, %alloc_6 : memref<128x256xbf16, strided<[?, 1], offset: ?>> to memref<128x256xbf16>
    %29 = bufferization.to_tensor %alloc_6 restrict writable : memref<128x256xbf16>
    %expanded = tensor.expand_shape %24 [[0, 1]] output_shape [1, 128] : tensor<128xbf16> into tensor<1x128xbf16>
    %30 = linalg.matmul {input_precision = "ieee"} ins(%expanded, %29 : tensor<1x128xbf16>, tensor<128x256xbf16>) outs(%1 : tensor<1x256xf32>) -> tensor<1x256xf32>
    %31 = arith.muli %arg26, %arg22 : i32
    %32 = arith.index_cast %31 : i32 to index
    %reinterpret_cast_7 = memref.reinterpret_cast %arg9 to offset: [%32], sizes: [1, 256], strides: [256, 1] : memref<?xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
    bufferization.materialize_in_destination %30 in writable %reinterpret_cast_7 : (tensor<1x256xf32>, memref<1x256xf32, strided<[256, 1], offset: ?>>) -> ()
    %33 = arith.muli %18, %arg12 : i32
    %34 = arith.muli %19, %arg13 : i32
    %35 = arith.addi %33, %34 : i32
    %36 = arith.index_cast %35 : i32 to index
    %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%36], sizes: [64], strides: [1] : memref<?xbf16> to memref<64xbf16, strided<[1], offset: ?>>
    %alloc_9 = memref.alloc() : memref<64xbf16>
    memref.copy %reinterpret_cast_8, %alloc_9 : memref<64xbf16, strided<[1], offset: ?>> to memref<64xbf16>
    %37 = bufferization.to_tensor %alloc_9 restrict writable : memref<64xbf16>
    %38 = arith.muli %18, %arg19 : i32
    %39 = arith.index_cast %38 : i32 to index
    %reinterpret_cast_10 = memref.reinterpret_cast %arg7 to offset: [%39], sizes: [16], strides: [1] : memref<?xi32> to memref<16xi32, strided<[1], offset: ?>>
    %alloc_11 = memref.alloc() : memref<16xi32>
    memref.copy %reinterpret_cast_10, %alloc_11 : memref<16xi32, strided<[1], offset: ?>> to memref<16xi32>
    %40 = bufferization.to_tensor %alloc_11 restrict writable : memref<16xi32>
    %alloc_12 = memref.alloc() : memref<1x256xf32>
    memref.copy %reinterpret_cast_7, %alloc_12 : memref<1x256xf32, strided<[256, 1], offset: ?>> to memref<1x256xf32>
    %41 = bufferization.to_tensor %alloc_12 restrict writable : memref<1x256xf32>
    %42 = arith.muli %18, %arg15 : i32
    %43 = arith.muli %18, %arg17 : i32
    %44:2 = scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg30 = %3, %arg31 = %5) -> (tensor<16x256xbf16>, tensor<16x64xbf16>)  : i32 {
      %85 = arith.index_cast %arg29 : i32 to index
      %extracted = tensor.extract %40[%85] : tensor<16xi32>
      %86 = arith.muli %extracted, %arg16 : i32
      %87 = arith.addi %42, %86 : i32
      %88 = arith.index_cast %87 : i32 to index
      %reinterpret_cast_20 = memref.reinterpret_cast %arg5 to offset: [%88], sizes: [256], strides: [1] : memref<?xbf16> to memref<256xbf16, strided<[1], offset: ?>>
      %alloc_21 = memref.alloc() : memref<256xbf16>
      memref.copy %reinterpret_cast_20, %alloc_21 : memref<256xbf16, strided<[1], offset: ?>> to memref<256xbf16>
      %89 = bufferization.to_tensor %alloc_21 restrict writable : memref<256xbf16>
      %expanded_22 = tensor.expand_shape %89 [[0, 1]] output_shape [1, 256] : tensor<256xbf16> into tensor<1x256xbf16>
      %inserted_slice = tensor.insert_slice %expanded_22 into %arg30[%85, 0] [1, 256] [1, 1] : tensor<1x256xbf16> into tensor<16x256xbf16>
      %90 = arith.muli %extracted, %arg18 : i32
      %91 = arith.addi %43, %90 : i32
      %92 = arith.index_cast %91 : i32 to index
      %reinterpret_cast_23 = memref.reinterpret_cast %arg6 to offset: [%92], sizes: [64], strides: [1] : memref<?xbf16> to memref<64xbf16, strided<[1], offset: ?>>
      %alloc_24 = memref.alloc() : memref<64xbf16>
      memref.copy %reinterpret_cast_23, %alloc_24 : memref<64xbf16, strided<[1], offset: ?>> to memref<64xbf16>
      %93 = bufferization.to_tensor %alloc_24 restrict writable : memref<64xbf16>
      %expanded_25 = tensor.expand_shape %93 [[0, 1]] output_shape [1, 64] : tensor<64xbf16> into tensor<1x64xbf16>
      %inserted_slice_26 = tensor.insert_slice %expanded_25 into %arg31[%85, 0] [1, 64] [1, 1] : tensor<1x64xbf16> into tensor<16x64xbf16>
      scf.yield %inserted_slice, %inserted_slice_26 : tensor<16x256xbf16>, tensor<16x64xbf16>
    }
    %45 = arith.addf %44#0, %6 : tensor<16x256xbf16>
    %46 = tensor.empty() : tensor<256x16xbf16>
    %transposed = linalg.transpose ins(%45 : tensor<16x256xbf16>) outs(%46 : tensor<256x16xbf16>) permutation = [1, 0]
    %47 = arith.addf %44#1, %7 : tensor<16x64xbf16>
    %48 = tensor.empty() : tensor<64x16xbf16>
    %transposed_13 = linalg.transpose ins(%47 : tensor<16x64xbf16>) outs(%48 : tensor<64x16xbf16>) permutation = [1, 0]
    %expanded_14 = tensor.expand_shape %37 [[0, 1]] output_shape [1, 64] : tensor<64xbf16> into tensor<1x64xbf16>
    %49 = linalg.matmul {input_precision = "ieee"} ins(%expanded_14, %transposed_13 : tensor<1x64xbf16>, tensor<64x16xbf16>) outs(%9 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %50 = arith.extf %transposed : tensor<256x16xbf16> to tensor<256x16xf32>
    %51 = linalg.matmul {input_precision = "ieee"} ins(%41, %50 : tensor<1x256xf32>, tensor<256x16xf32>) outs(%49 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %52 = arith.mulf %51, %10 : tensor<1x16xf32>
    %reduced = linalg.reduce ins(%52 : tensor<1x16xf32>) outs(%15 : tensor<1xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %85 = arith.maxnumf %in, %init : f32
        linalg.yield %85 : f32
      }
    %broadcasted = linalg.broadcast ins(%reduced : tensor<1xf32>) outs(%8 : tensor<1x16xf32>) dimensions = [1]
    %53 = arith.subf %52, %broadcasted : tensor<1x16xf32>
    %54 = math.exp %53 : tensor<1x16xf32>
    %55 = linalg.fill ins(%cst_4 : f32) outs(%11 : tensor<1xf32>) -> tensor<1xf32>
    %reduced_15 = linalg.reduce ins(%54 : tensor<1x16xf32>) outs(%55 : tensor<1xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %85 = arith.addf %in, %init : f32
        linalg.yield %85 : f32
      }
    %56 = math.log %reduced_15 : tensor<1xf32>
    %57 = arith.addf %reduced, %56 : tensor<1xf32>
    %58 = arith.addf %57, %15 : tensor<1xf32>
    %59 = arith.divf %58, %12 : tensor<1xf32>
    %60 = arith.subf %15, %59 : tensor<1xf32>
    %61 = math.exp %60 : tensor<1xf32>
    %62 = arith.subf %57, %59 : tensor<1xf32>
    %63 = math.exp %62 : tensor<1xf32>
    %64 = arith.addf %61, %63 : tensor<1xf32>
    %65 = math.log %64 : tensor<1xf32>
    %66 = arith.addf %59, %65 : tensor<1xf32>
    %67 = arith.cmpf une, %66, %66 : tensor<1xf32>
    %68 = arith.select %67, %57, %66 : tensor<1xi1>, tensor<1xf32>
    %69 = arith.subf %15, %68 : tensor<1xf32>
    %70 = math.exp %69 : tensor<1xf32>
    %expanded_16 = tensor.expand_shape %70 [[0, 1]] output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>
    %71 = arith.mulf %expanded_16, %17 : tensor<1x1xf32>
    %collapsed = tensor.collapse_shape %71 [[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
    %broadcasted_17 = linalg.broadcast ins(%collapsed : tensor<1xf32>) outs(%0 : tensor<1x256xf32>) dimensions = [1]
    %72 = arith.subf %57, %68 : tensor<1xf32>
    %73 = math.exp %72 : tensor<1xf32>
    %broadcasted_18 = linalg.broadcast ins(%reduced_15 : tensor<1xf32>) outs(%8 : tensor<1x16xf32>) dimensions = [1]
    %74 = arith.divf %54, %broadcasted_18 : tensor<1x16xf32>
    %75 = arith.extf %45 : tensor<16x256xbf16> to tensor<16x256xf32>
    %76 = linalg.matmul {input_precision = "ieee"} ins(%74, %75 : tensor<1x16xf32>, tensor<16x256xf32>) outs(%1 : tensor<1x256xf32>) -> tensor<1x256xf32>
    %broadcasted_19 = linalg.broadcast ins(%73 : tensor<1xf32>) outs(%0 : tensor<1x256xf32>) dimensions = [1]
    %77 = arith.mulf %76, %broadcasted_19 : tensor<1x256xf32>
    %78 = arith.addf %broadcasted_17, %77 : tensor<1x256xf32>
    %79 = arith.addi %25, %c128_i32 : i32
    %80 = arith.muli %79, %arg14 : i32
    %81 = arith.muli %18, %arg20 : i32
    %82 = arith.muli %19, %arg21 : i32
    %83 = arith.addi %81, %82 : i32
    %84 = scf.for %arg29 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg30 = %c0) -> (index)  : i32 {
      %85 = arith.index_cast %80 : i32 to index
      %86 = arith.muli %arg30, %28 : index
      %87 = arith.addi %85, %86 : index
      %reinterpret_cast_20 = memref.reinterpret_cast %arg4 to offset: [%87], sizes: [64, 256], strides: [%28, 1] : memref<?xbf16> to memref<64x256xbf16, strided<[?, 1], offset: ?>>
      %alloc_21 = memref.alloc() : memref<64x256xbf16>
      memref.copy %reinterpret_cast_20, %alloc_21 : memref<64x256xbf16, strided<[?, 1], offset: ?>> to memref<64x256xbf16>
      %88 = bufferization.to_tensor %alloc_21 restrict writable : memref<64x256xbf16>
      %89 = arith.extf %88 : tensor<64x256xbf16> to tensor<64x256xf32>
      %90 = tensor.empty() : tensor<256x64xf32>
      %transposed_22 = linalg.transpose ins(%89 : tensor<64x256xf32>) outs(%90 : tensor<256x64xf32>) permutation = [1, 0]
      %91 = linalg.matmul {input_precision = "ieee"} ins(%78, %transposed_22 : tensor<1x256xf32>, tensor<256x64xf32>) outs(%14 : tensor<1x64xf32>) -> tensor<1x64xf32>
      %92 = arith.index_cast %83 : i32 to index
      %93 = arith.addi %92, %arg30 : index
      %reinterpret_cast_23 = memref.reinterpret_cast %arg8 to offset: [%93], sizes: [1, 64], strides: [64, %c1] : memref<?xf32> to memref<1x64xf32, strided<[64, ?], offset: ?>>
      bufferization.materialize_in_destination %91 in writable %reinterpret_cast_23 : (tensor<1x64xf32>, memref<1x64xf32, strided<[64, ?], offset: ?>>) -> ()
      %94 = arith.addi %arg30, %c64 : index
      scf.yield %94 : index
    }
    return
  }
}