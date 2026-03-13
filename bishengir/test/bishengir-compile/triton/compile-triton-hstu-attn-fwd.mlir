// RUN: bishengir-compile %s --target=Ascend950PR_957c --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=False --disable-ffts --enable-hivm-graph-sync-solver=True --enable-mixed-cv=True --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 --enable-lir-compile=true --enable-layout-optimization=true
// RUN: bishengir-compile %s --target=Ascend950PR_957c --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=False --disable-ffts --enable-hivm-graph-sync-solver=True --enable-mixed-cv=True --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 --enable-lir-compile=true
module attributes {hacc.target = #hacc.target<"Ascend950PR_957c">} {
  func.func @_hstu_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf8E4M3FN> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf8E4M3FN> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf8E4M3FN> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32}, %arg9: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: f32, %arg11: f32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f8E4M3FN
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c256_i32 = arith.constant 256 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8_i64 = arith.constant 8 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i64 = arith.constant 2 : i64
    %c64_i64 = arith.constant 64 : i64
    %c512_i64 = arith.constant 512 : i64
    %c256_i64 = arith.constant 256 : i64
    %c128_i64 = arith.constant 128 : i64
    %c127_i32 = arith.constant 127 : i32
    %c1_i64 = arith.constant 1 : i64
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<256x128xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<256x128xf32>) -> tensor<256x128xf32>
    %2 = tensor.empty() : tensor<256x64xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<256x64xf32>) -> tensor<256x64xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256x128xf32>) -> tensor<256x128xf32>
    %5 = arith.cmpi sle, %arg12, %c256_i32 : i32
    %6 = scf.if %5 -> (i64) {
      scf.yield %c2_i64 : i64
    } else {
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [2], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: 2>>
      %26 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: 2>>
      %27 = arith.extsi %26 : i32 to i64
      scf.yield %27 : i64
    }
    %7 = arith.muli %6, %c8_i64 : i64
    %8 = arith.extsi %arg15 : i32 to i64
    %9 = arith.minsi %8, %7 : i64
    %10 = arith.divsi %7, %9 : i64
    %11 = arith.addi %10, %c1_i64 : i64
    %12 = arith.remsi %7, %9 : i64
    %13 = arith.extsi %arg18 : i32 to i64
    %14 = arith.cmpi slt, %13, %9 : i64
    %15 = arith.cmpi slt, %13, %12 : i64
    %16 = arith.muli %13, %11 : i64
    %17 = arith.muli %12, %11 : i64
    %18 = arith.subi %13, %12 : i64
    %19 = arith.muli %18, %10 : i64
    %20 = arith.addi %17, %19 : i64
    %21 = arith.select %15, %16, %20 : i64
    %22 = arith.select %14, %21, %c0_i64 : i64
    %23 = arith.select %15, %11, %10 : i64
    %24 = arith.select %14, %23, %c0_i64 : i64
    %25 = arith.cmpi sge, %13, %9 : i64
    scf.if %25 {
    } else {
      %26 = arith.cmpi sle, %24, %c0_i64 : i64
      scf.if %26 {
      } else {
        %27 = arith.addi %arg13, %c127_i32 : i32
        %28 = arith.divsi %27, %c128_i32 : i32
        %29 = arith.extsi %28 : i32 to i64
        %30 = arith.muli %24, %29 : i64
        %31 = linalg.fill ins(%arg10 : f32) outs(%0 : tensor<256x128xf32>) -> tensor<256x128xf32>
        %32 = linalg.fill ins(%arg11 : f32) outs(%0 : tensor<256x128xf32>) -> tensor<256x128xf32>
        scf.for %arg21 = %c0_i64 to %30 step %c1_i64  : i64 {
          %33 = arith.divsi %arg21, %29 : i64
          %34 = arith.addi %22, %33 : i64
          %35 = arith.remsi %arg21, %29 : i64
          %36 = arith.divsi %34, %6 : i64
          %37 = arith.remsi %34, %6 : i64
          %38:2 = scf.if %5 -> (i64, i64) {
            scf.yield %37, %c0_i64 : i64, i64
          } else {
            %91:2 = scf.for %arg22 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg23 = %c0_i32, %arg24 = %c3_i32) -> (i32, i32)  : i32 {
              %98 = arith.addi %arg23, %arg24 : i32
              %99 = arith.divsi %98, %c2_i32 : i32
              %100 = arith.index_cast %99 : i32 to index
              %reinterpret_cast_18 = memref.reinterpret_cast %arg7 to offset: [%100], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
              %101 = memref.load %reinterpret_cast_18[%c0] : memref<1xi32, strided<[1], offset: ?>>
              %102 = arith.extsi %101 : i32 to i64
              %103 = arith.cmpi sle, %102, %37 : i64
              %104 = arith.select %103, %arg24, %99 : i32
              %105 = scf.if %103 -> (i32) {
                %106 = arith.addi %99, %c1_i32 : i32
                scf.yield %106 : i32
              } else {
                scf.yield %arg23 : i32
              }
              scf.yield %105, %104 : i32, i32
            }
            %92 = arith.subi %91#0, %c1_i32 : i32
            %93 = arith.extsi %92 : i32 to i64
            %94 = arith.index_cast %92 : i32 to index
            %reinterpret_cast_17 = memref.reinterpret_cast %arg7 to offset: [%94], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
            %95 = memref.load %reinterpret_cast_17[%c0] : memref<1xi32, strided<[1], offset: ?>>
            %96 = arith.extsi %95 : i32 to i64
            %97 = arith.subi %37, %96 : i64
            scf.yield %93, %97 : i64, i64
          }
          %39 = arith.divsi %36, %c2_i64 : i64
          %40 = arith.index_cast %38#0 : i64 to index
          %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%40], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %41 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %42 = arith.addi %40, %c1 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%42], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %43 = memref.load %reinterpret_cast_2[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%40], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %44 = memref.load %reinterpret_cast_3[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%42], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %45 = memref.load %reinterpret_cast_4[%c0] : memref<1xi64, strided<[1], offset: ?>>
          %46 = arith.subi %43, %41 : i64
          %47 = arith.subi %45, %44 : i64
          %48 = arith.muli %36, %c64_i64 : i64
          %49 = arith.muli %41, %c512_i64 : i64
          %50 = arith.addi %48, %49 : i64
          %51 = arith.index_cast %50 : i64 to index
          %52 = arith.muli %39, %c64_i64 : i64
          %53 = arith.muli %44, %c256_i64 : i64
          %54 = arith.addi %52, %53 : i64
          %55 = arith.index_cast %54 : i64 to index
          %56 = arith.muli %35, %c128_i64 : i64
          %57 = arith.muli %38#1, %c256_i64 : i64
          %58 = arith.index_cast %57 : i64 to index
          %59 = arith.muli %58, %c512 : index
          %60 = arith.addi %51, %59 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%60], sizes: [256, 64], strides: [512, 1] : memref<?xf8E4M3FN> to memref<256x64xf8E4M3FN, strided<[512, 1], offset: ?>>
          %alloc = memref.alloc() : memref<256x64xf8E4M3FN>
          %61 = arith.addi %58, %c256 : index
          %62 = arith.index_cast %46 : i64 to index
          %63 = arith.maxsi %58, %62 : index
          %64 = arith.minsi %61, %63 : index
          %65 = arith.subi %64, %58 : index
          %66 = arith.cmpi slt, %65, %c256 : index
          scf.if %66 {
            linalg.fill ins(%cst : f8E4M3FN) outs(%alloc : memref<256x64xf8E4M3FN>)
          } {hivm.unlikely_condition}
          %subview = memref.subview %reinterpret_cast_5[0, 0] [%65, 64] [1, 1] : memref<256x64xf8E4M3FN, strided<[512, 1], offset: ?>> to memref<?x64xf8E4M3FN, strided<[512, 1], offset: ?>>
          %subview_6 = memref.subview %alloc[0, 0] [%65, 64] [1, 1] : memref<256x64xf8E4M3FN> to memref<?x64xf8E4M3FN, strided<[64, 1]>>
          memref.copy %subview, %subview_6 : memref<?x64xf8E4M3FN, strided<[512, 1], offset: ?>> to memref<?x64xf8E4M3FN, strided<[64, 1]>>
          %67 = bufferization.to_tensor %alloc restrict writable : memref<256x64xf8E4M3FN>
          %68 = arith.index_cast %56 : i64 to index
          %69 = arith.muli %68, %c256 : index
          %70 = arith.addi %55, %69 : index
          %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%70], sizes: [128, 64], strides: [256, 1] : memref<?xf8E4M3FN> to memref<128x64xf8E4M3FN, strided<[256, 1], offset: ?>>
          %alloc_8 = memref.alloc() : memref<128x64xf8E4M3FN>
          %71 = arith.addi %68, %c128 : index
          %72 = arith.index_cast %47 : i64 to index
          %73 = arith.maxsi %68, %72 : index
          %74 = arith.minsi %71, %73 : index
          %75 = arith.subi %74, %68 : index
          %76 = arith.cmpi slt, %75, %c128 : index
          scf.if %76 {
            linalg.fill ins(%cst : f8E4M3FN) outs(%alloc_8 : memref<128x64xf8E4M3FN>)
          } {hivm.unlikely_condition}
          %subview_9 = memref.subview %reinterpret_cast_7[0, 0] [%75, 64] [1, 1] : memref<128x64xf8E4M3FN, strided<[256, 1], offset: ?>> to memref<?x64xf8E4M3FN, strided<[256, 1], offset: ?>>
          %subview_10 = memref.subview %alloc_8[0, 0] [%75, 64] [1, 1] : memref<128x64xf8E4M3FN> to memref<?x64xf8E4M3FN, strided<[64, 1]>>
          memref.copy %subview_9, %subview_10 : memref<?x64xf8E4M3FN, strided<[256, 1], offset: ?>> to memref<?x64xf8E4M3FN, strided<[64, 1]>>
          %77 = bufferization.to_tensor %alloc_8 restrict writable : memref<128x64xf8E4M3FN>
          %78 = tensor.empty() : tensor<64x128xf8E4M3FN>
          %transposed = linalg.transpose ins(%77 : tensor<128x64xf8E4M3FN>) outs(%78 : tensor<64x128xf8E4M3FN>) permutation = [1, 0] 
          %79 = linalg.matmul {input_precision = "ieee"} ins(%67, %transposed : tensor<256x64xf8E4M3FN>, tensor<64x128xf8E4M3FN>) outs(%1 : tensor<256x128xf32>) -> tensor<256x128xf32>
          %80 = arith.mulf %79, %31 : tensor<256x128xf32>
          %81 = arith.subf %1, %80 : tensor<256x128xf32>
          %82 = math.exp %81 : tensor<256x128xf32>
          %83 = arith.addf %82, %4 : tensor<256x128xf32>
          %84 = arith.divf %4, %83 : tensor<256x128xf32>
          %85 = arith.mulf %80, %84 : tensor<256x128xf32>
          %86 = arith.mulf %85, %32 : tensor<256x128xf32>
          %87 = arith.truncf %86 {round_mode = #hfusion.round_mode<rint>} : tensor<256x128xf32> to tensor<256x128xf8E4M3FN>
          %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%70], sizes: [128, 64], strides: [256, 1] : memref<?xf8E4M3FN> to memref<128x64xf8E4M3FN, strided<[256, 1], offset: ?>>
          %alloc_12 = memref.alloc() : memref<128x64xf8E4M3FN>
          scf.if %76 {
            linalg.fill ins(%cst : f8E4M3FN) outs(%alloc_12 : memref<128x64xf8E4M3FN>)
          } {hivm.unlikely_condition}
          %subview_13 = memref.subview %reinterpret_cast_11[0, 0] [%75, 64] [1, 1] : memref<128x64xf8E4M3FN, strided<[256, 1], offset: ?>> to memref<?x64xf8E4M3FN, strided<[256, 1], offset: ?>>
          %subview_14 = memref.subview %alloc_12[0, 0] [%75, 64] [1, 1] : memref<128x64xf8E4M3FN> to memref<?x64xf8E4M3FN, strided<[64, 1]>>
          memref.copy %subview_13, %subview_14 : memref<?x64xf8E4M3FN, strided<[256, 1], offset: ?>> to memref<?x64xf8E4M3FN, strided<[64, 1]>>
          %88 = bufferization.to_tensor %alloc_12 restrict writable : memref<128x64xf8E4M3FN>
          %89 = linalg.matmul {input_precision = "ieee"} ins(%87, %88 : tensor<256x128xf8E4M3FN>, tensor<128x64xf8E4M3FN>) outs(%3 : tensor<256x64xf32>) -> tensor<256x64xf32>
          %reinterpret_cast_15 = memref.reinterpret_cast %arg9 to offset: [%60], sizes: [256, 64], strides: [512, 1] : memref<?xf16> to memref<256x64xf16, strided<[512, 1], offset: ?>>
          %90 = arith.truncf %89 : tensor<256x64xf32> to tensor<256x64xf16>
          %extracted_slice = tensor.extract_slice %90[0, 0] [%65, 64] [1, 1] : tensor<256x64xf16> to tensor<?x64xf16>
          %subview_16 = memref.subview %reinterpret_cast_15[0, 0] [%65, 64] [1, 1] : memref<256x64xf16, strided<[512, 1], offset: ?>> to memref<?x64xf16, strided<[512, 1], offset: ?>>
          bufferization.materialize_in_destination %extracted_slice in writable %subview_16 : (tensor<?x64xf16>, memref<?x64xf16, strided<[512, 1], offset: ?>>) -> ()
        }
      }
    }
    return
  }
}