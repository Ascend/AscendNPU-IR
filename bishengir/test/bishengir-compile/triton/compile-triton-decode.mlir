// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-auto-vectorize-v2=false --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @dsa_of_mla_decode_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: f32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c0_i8 = arith.constant 0 : i8
    %c8_i32 = arith.constant 8 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x64xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %2 = tensor.empty() : tensor<64xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<64xi8>) -> tensor<64xi8>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %5 = tensor.empty() : tensor<1xf32>
    %6 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<1xf32>) -> tensor<1xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<1xf32>) -> tensor<1xf32>
    %8 = arith.divsi %arg26, %c8_i32 : i32
    %9 = arith.remsi %arg26, %c8_i32 : i32
    %10 = arith.muli %8, %arg8 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12 = arith.muli %9, %arg9 : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = arith.addi %11, %13 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64], strides: [1] : memref<?xbf16> to memref<64xbf16, strided<[1], offset: ?>>
    %15 = arith.muli %8, %arg11 : i32
    %16 = arith.index_cast %15 : i32 to index
    %17 = arith.muli %9, %arg12 : i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.addi %16, %18 : index
    %20 = arith.muli %8, %arg14 : i32
    %21 = arith.index_cast %20 : i32 to index
    %22 = arith.muli %9, %arg15 : i32
    %23 = arith.index_cast %22 : i32 to index
    %24 = arith.addi %21, %23 : index
    %25 = arith.muli %8, %arg20 : i32
    %26 = arith.muli %9, %arg21 : i32
    %alloc = memref.alloc() : memref<64xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<64xbf16, strided<[1], offset: ?>> to memref<64xbf16>
    %27 = bufferization.to_tensor %alloc restrict writable : memref<64xbf16>
    %28 = arith.extf %27 : tensor<64xbf16> to tensor<64xf32>
    %expanded = tensor.expand_shape %28 [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
    %29 = linalg.fill ins(%arg7 : f32) outs(%0 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %30:3 = scf.for %arg29 = %c0_i32 to %c256_i32 step %c64_i32 iter_args(%arg30 = %1, %arg31 = %6, %arg32 = %7) -> (tensor<1x64xf32>, tensor<1xf32>, tensor<1xf32>)  : i32 {
      %38 = arith.index_cast %arg29 : i32 to index
      %39 = arith.index_cast %arg13 : i32 to index
      %40 = arith.muli %38, %39 : index
      %41 = arith.addi %19, %40 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [64, 64], strides: [%39, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
      %alloc_4 = memref.alloc() : memref<64x64xbf16>
      %42 = arith.addi %38, %c64 : index
      %43 = arith.maxsi %38, %c256 : index
      %44 = arith.minsi %42, %43 : index
      %45 = arith.subi %44, %38 : index
      %46 = arith.cmpi slt, %45, %c64 : index
      scf.if %46 {
        linalg.fill ins(%cst : bf16) outs(%alloc_4 : memref<64x64xbf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_3[0, 0] [%45, 64] [1, 1] : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[?, 1], offset: ?>>
      %subview_5 = memref.subview %alloc_4[0, 0] [%45, 64] [1, 1] : memref<64x64xbf16> to memref<?x64xbf16, strided<[64, 1]>>
      memref.copy %subview, %subview_5 : memref<?x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[64, 1]>>
      %47 = bufferization.to_tensor %alloc_4 restrict writable : memref<64x64xbf16>
      %48 = arith.extf %47 : tensor<64x64xbf16> to tensor<64x64xf32>
      %49 = tensor.empty() : tensor<64x64xf32>
      %transposed = linalg.transpose ins(%48 : tensor<64x64xf32>) outs(%49 : tensor<64x64xf32>) permutation = [1, 0]
      %50 = linalg.matmul {input_precison = "ieee"} ins(%expanded, %transposed : tensor<1x64xf32>, tensor<64x64xf32>) outs(%1 : tensor<1x64xf32>) -> tensor<1x64xf32>
      %51 = arith.mulf %50, %29 : tensor<1x64xf32>
      %52 = arith.index_cast %25 : i32 to index
      %53 = arith.index_cast %26 : i32 to index
      %54 = arith.addi %52, %53 : index
      %55 = arith.addi %54, %38 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%55], sizes: [64], strides: [1] : memref<?xi8> to memref<64xi8, strided<[1], offset: ?>>
      %alloc_7 = memref.alloc() : memref<64xi8>
      scf.if %46 {
        linalg.fill ins(%c0_i8 : i8) outs(%alloc_7 : memref<64xi8>)
      } {hivm.unlikely_condition}
      %subview_8 = memref.subview %reinterpret_cast_6[0] [%45] [1] : memref<64xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
      %subview_9 = memref.subview %alloc_7[0] [%45] [1] : memref<64xi8> to memref<?xi8, strided<[1]>>
      memref.copy %subview_8, %subview_9 : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1]>>
      %56 = bufferization.to_tensor %alloc_7 restrict writable : memref<64xi8>
      %57 = arith.cmpi ne, %56, %3 : tensor<64xi8>
      %expanded_10 = tensor.expand_shape %57 [[0, 1]] output_shape [1, 64] : tensor<64xi1> into tensor<1x64xi1>
      %58 = arith.select %expanded_10, %51, %4 : tensor<1x64xi1>, tensor<1x64xf32>
      %reduced = linalg.reduce ins(%58 : tensor<1x64xf32>) outs(%7 : tensor<1xf32>) dimensions = [1]
        (%in: f32, %init: f32) {
          %73 = arith.maxnumf %in, %init : f32
          linalg.yield %73 : f32
        }
      %59 = arith.maxnumf %arg32, %reduced : tensor<1xf32>
      %broadcasted_11 = linalg.broadcast ins(%59 : tensor<1xf32>) outs(%0 : tensor<1x64xf32>) dimensions = [1]
      %60 = arith.subf %58, %broadcasted_11 : tensor<1x64xf32>
      %61 = math.exp %60 : tensor<1x64xf32>
      %62 = arith.subf %arg32, %59 : tensor<1xf32>
      %63 = math.exp %62 : tensor<1xf32>
      %64 = arith.mulf %arg31, %63 : tensor<1xf32>
      %reduced_12 = linalg.reduce ins(%61 : tensor<1x64xf32>) outs(%6 : tensor<1xf32>) dimensions = [1]
        (%in: f32, %init: f32) {
          %73 = arith.addf %in, %init : f32
          linalg.yield %73 : f32
        }
      %65 = arith.addf %64, %reduced_12 : tensor<1xf32>
      %broadcasted_13 = linalg.broadcast ins(%63 : tensor<1xf32>) outs(%0 : tensor<1x64xf32>) dimensions = [1]
      %66 = arith.mulf %arg30, %broadcasted_13 : tensor<1x64xf32>
      %67 = arith.index_cast %arg16 : i32 to index
      %68 = arith.muli %38, %67 : index
      %69 = arith.addi %24, %68 : index
      %reinterpret_cast_14 = memref.reinterpret_cast %arg4 to offset: [%69], sizes: [64, 64], strides: [%67, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
      %alloc_15 = memref.alloc() : memref<64x64xbf16>
      scf.if %46 {
        linalg.fill ins(%cst : bf16) outs(%alloc_15 : memref<64x64xbf16>)
      } {hivm.unlikely_condition}
      %subview_16 = memref.subview %reinterpret_cast_14[0, 0] [%45, 64] [1, 1] : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[?, 1], offset: ?>>
      %subview_17 = memref.subview %alloc_15[0, 0] [%45, 64] [1, 1] : memref<64x64xbf16> to memref<?x64xbf16, strided<[64, 1]>>
      memref.copy %subview_16, %subview_17 : memref<?x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[64, 1]>>
      %70 = bufferization.to_tensor %alloc_15 restrict writable : memref<64x64xbf16>
      %71 = arith.extf %70 : tensor<64x64xbf16> to tensor<64x64xf32>
      %72 = linalg.matmul {input_precison = "ieee"} ins(%61, %71 : tensor<1x64xf32>, tensor<64x64xf32>) outs(%66 : tensor<1x64xf32>) -> tensor<1x64xf32>
      scf.yield %72, %65, %59 : tensor<1x64xf32>, tensor<1xf32>, tensor<1xf32>
    }
    %broadcasted = linalg.broadcast ins(%30#1 : tensor<1xf32>) outs(%0 : tensor<1x64xf32>) dimensions = [1]
    %31 = arith.divf %30#0, %broadcasted : tensor<1x64xf32>
    %32 = arith.muli %8, %arg17 : i32
    %33 = arith.index_cast %32 : i32 to index
    %34 = arith.muli %9, %arg18 : i32
    %35 = arith.index_cast %34 : i32 to index
    %36 = arith.addi %33, %35 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%36], sizes: [1, 64], strides: [64, 1] : memref<?xbf16> to memref<1x64xbf16, strided<[64, 1], offset: ?>>
    %37 = arith.truncf %31 : tensor<1x64xf32> to tensor<1x64xbf16>
    bufferization.materialize_in_destination %37 in writable %reinterpret_cast_2 : (tensor<1x64xbf16>, memref<1x64xbf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }
}
