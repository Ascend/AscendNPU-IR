// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @_hstu_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32}, %arg9: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg10: f32, %arg11: f32, %arg12: i32, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c192 = arith.constant 192 : index
    %c48 = arith.constant 48 : index
    %c352 = arith.constant 352 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i64 = arith.constant 4 : i64
    %c352_i32 = arith.constant 352 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i64 = arith.constant 128 : i64
    %c512_i64 = arith.constant 512 : i64
    %c48_i64 = arith.constant 48 : i64
    %c192_i64 = arith.constant 192 : i64
    %c352_i64 = arith.constant 352 : i64
    %c2_i64 = arith.constant 2 : i64
    %c351_i32 = arith.constant 351 : i32
    %c1_i64 = arith.constant 1 : i64
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x352xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<64x352xf32>) -> tensor<64x352xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64x352xf32>) -> tensor<64x352xf32>
    %3 = tensor.empty() : tensor<64x48xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%3 : tensor<64x48xf32>) -> tensor<64x48xf32>
    %5 = arith.cmpi sle, %arg12, %c64_i32 : i32
    %6 = scf.if %5 -> (i64) {
      scf.yield %c2_i64 : i64
    } else {
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [2], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: 2>>
      %26 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: 2>>
      scf.yield %26 : i64
    }
    %7 = arith.muli %6, %c4_i64 : i64
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
        %27 = arith.addi %arg13, %c351_i32 : i32
        %28 = arith.divsi %27, %c352_i32 : i32
        %29 = arith.extsi %28 : i32 to i64
        %30 = arith.muli %24, %29 : i64
        %31 = linalg.fill ins(%arg10 : f32) outs(%0 : tensor<64x352xf32>) -> tensor<64x352xf32>
        %32 = linalg.fill ins(%arg11 : f32) outs(%0 : tensor<64x352xf32>) -> tensor<64x352xf32>
        scf.for %arg21 = %c0_i64 to %30 step %c1_i64  : i64 {
          %33 = arith.divsi %arg21, %29 : i64
          %34 = arith.addi %22, %33 : i64
          %35 = arith.remsi %arg21, %29 : i64
          %36 = arith.divsi %34, %6 : i64
          %37 = arith.remsi %34, %6 : i64
          %38:2 = scf.if %5 -> (i64, i64) {
            scf.yield %37, %c0_i64 : i64, i64
          } else {
            %103:2 = scf.for %arg22 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg23 = %c0_i32, %arg24 = %c3_i32) -> (i32, i32)  : i32 {
              %109 = arith.addi %arg23, %arg24 : i32
              %110 = arith.divsi %109, %c2_i32 : i32
              %111 = arith.index_cast %110 : i32 to index
              %reinterpret_cast_18 = memref.reinterpret_cast %arg7 to offset: [%111], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
              %112 = memref.load %reinterpret_cast_18[%c0] : memref<1xi64, strided<[1], offset: ?>>
              %113 = arith.cmpi sle, %112, %37 : i64
              %114 = arith.select %113, %arg24, %110 : i32
              %115 = scf.if %113 -> (i32) {
                %116 = arith.addi %110, %c1_i32 : i32
                scf.yield %116 : i32
              } else {
                scf.yield %arg23 : i32
              }
              scf.yield %115, %114 : i32, i32
            }
            %104 = arith.subi %103#0, %c1_i32 : i32
            %105 = arith.extsi %104 : i32 to i64
            %106 = arith.index_cast %104 : i32 to index
            %reinterpret_cast_17 = memref.reinterpret_cast %arg7 to offset: [%106], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
            %107 = memref.load %reinterpret_cast_17[%c0] : memref<1xi64, strided<[1], offset: ?>>
            %108 = arith.subi %37, %107 : i64
            scf.yield %105, %108 : i64, i64
          }
          %39 = arith.divsi %36, %c4_i64 : i64
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
          %48 = arith.muli %36, %c128_i64 : i64
          %49 = arith.muli %41, %c512_i64 : i64
          %50 = arith.addi %48, %49 : i64
          %51 = arith.index_cast %50 : i64 to index
          %52 = arith.muli %39, %c128_i64 : i64
          %53 = arith.muli %44, %c128_i64 : i64
          %54 = arith.addi %52, %53 : i64
          %55 = arith.index_cast %54 : i64 to index
          %56 = arith.muli %39, %c48_i64 : i64
          %57 = arith.muli %44, %c48_i64 : i64
          %58 = arith.addi %56, %57 : i64
          %59 = arith.index_cast %58 : i64 to index
          %60 = arith.muli %36, %c48_i64 : i64
          %61 = arith.muli %41, %c192_i64 : i64
          %62 = arith.addi %60, %61 : i64
          %63 = arith.index_cast %62 : i64 to index
          %64 = arith.muli %35, %c352_i64 : i64
          %65 = arith.muli %38#1, %c64_i64 : i64
          %66 = arith.index_cast %65 : i64 to index
          %67 = arith.muli %66, %c512 : index
          %68 = arith.addi %51, %67 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%68], sizes: [64, 128], strides: [512, 1] : memref<?xf16> to memref<64x128xf16, strided<[512, 1], offset: ?>>
          %alloc = memref.alloc() : memref<64x128xf16>
          %69 = arith.addi %66, %c64 : index
          %70 = arith.index_cast %46 : i64 to index
          %71 = arith.maxsi %66, %70 : index
          %72 = arith.minsi %69, %71 : index
          %73 = arith.subi %72, %66 : index
          %74 = arith.cmpi slt, %73, %c64 : index
          scf.if %74 {
            linalg.fill ins(%cst : f16) outs(%alloc : memref<64x128xf16>)
          } {hivm.unlikely_condition}
          %subview = memref.subview %reinterpret_cast_5[0, 0] [%73, 128] [1, 1] : memref<64x128xf16, strided<[512, 1], offset: ?>> to memref<?x128xf16, strided<[512, 1], offset: ?>>
          %subview_6 = memref.subview %alloc[0, 0] [%73, 128] [1, 1] : memref<64x128xf16> to memref<?x128xf16, strided<[128, 1]>>
          memref.copy %subview, %subview_6 : memref<?x128xf16, strided<[512, 1], offset: ?>> to memref<?x128xf16, strided<[128, 1]>>
          %75 = bufferization.to_tensor %alloc restrict writable : memref<64x128xf16>
          %76 = arith.index_cast %64 : i64 to index
          %77 = arith.muli %76, %c128 : index
          %78 = arith.addi %55, %77 : index
          %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%78], sizes: [352, 128], strides: [128, 1] : memref<?xf16> to memref<352x128xf16, strided<[128, 1], offset: ?>>
          %alloc_8 = memref.alloc() : memref<352x128xf16>
          %79 = arith.addi %76, %c352 : index
          %80 = arith.index_cast %47 : i64 to index
          %81 = arith.maxsi %76, %80 : index
          %82 = arith.minsi %79, %81 : index
          %83 = arith.subi %82, %76 : index
          %84 = arith.cmpi slt, %83, %c352 : index
          scf.if %84 {
            linalg.fill ins(%cst : f16) outs(%alloc_8 : memref<352x128xf16>)
          } {hivm.unlikely_condition}
          %subview_9 = memref.subview %reinterpret_cast_7[0, 0] [%83, 128] [1, 1] : memref<352x128xf16, strided<[128, 1], offset: ?>> to memref<?x128xf16, strided<[128, 1], offset: ?>>
          %subview_10 = memref.subview %alloc_8[0, 0] [%83, 128] [1, 1] : memref<352x128xf16> to memref<?x128xf16, strided<[128, 1]>>
          memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1], offset: ?>> to memref<?x128xf16, strided<[128, 1]>>
          %85 = bufferization.to_tensor %alloc_8 restrict writable : memref<352x128xf16>
          %86 = tensor.empty() : tensor<128x352xf16>
          %transposed = linalg.transpose ins(%85 : tensor<352x128xf16>) outs(%86 : tensor<128x352xf16>) permutation = [1, 0]
          %87 = linalg.matmul {input_precison = "ieee"} ins(%75, %transposed : tensor<64x128xf16>, tensor<128x352xf16>) outs(%1 : tensor<64x352xf32>) -> tensor<64x352xf32>
          %88 = arith.mulf %87, %31 : tensor<64x352xf32>
          %89 = arith.subf %1, %88 : tensor<64x352xf32>
          %90 = math.exp %89 : tensor<64x352xf32>
          %91 = arith.addf %90, %2 : tensor<64x352xf32>
          %92 = arith.divf %2, %91 : tensor<64x352xf32>
          %93 = arith.mulf %88, %92 : tensor<64x352xf32>
          %94 = arith.mulf %93, %32 : tensor<64x352xf32>
          %95 = arith.truncf %94 : tensor<64x352xf32> to tensor<64x352xf16>
          %96 = arith.muli %76, %c48 : index
          %97 = arith.addi %59, %96 : index
          %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%97], sizes: [352, 48], strides: [48, 1] : memref<?xf16> to memref<352x48xf16, strided<[48, 1], offset: ?>>
          %alloc_12 = memref.alloc() : memref<352x48xf16>
          scf.if %84 {
            linalg.fill ins(%cst : f16) outs(%alloc_12 : memref<352x48xf16>)
          } {hivm.unlikely_condition}
          %subview_13 = memref.subview %reinterpret_cast_11[0, 0] [%83, 48] [1, 1] : memref<352x48xf16, strided<[48, 1], offset: ?>> to memref<?x48xf16, strided<[48, 1], offset: ?>>
          %subview_14 = memref.subview %alloc_12[0, 0] [%83, 48] [1, 1] : memref<352x48xf16> to memref<?x48xf16, strided<[48, 1]>>
          memref.copy %subview_13, %subview_14 : memref<?x48xf16, strided<[48, 1], offset: ?>> to memref<?x48xf16, strided<[48, 1]>>
          %98 = bufferization.to_tensor %alloc_12 restrict writable : memref<352x48xf16>
          %99 = linalg.matmul {input_precison = "ieee"} ins(%95, %98 : tensor<64x352xf16>, tensor<352x48xf16>) outs(%4 : tensor<64x48xf32>) -> tensor<64x48xf32>
          %100 = arith.muli %66, %c192 : index
          %101 = arith.addi %63, %100 : index
          %reinterpret_cast_15 = memref.reinterpret_cast %arg9 to offset: [%101], sizes: [64, 48], strides: [192, 1] : memref<?xf16> to memref<64x48xf16, strided<[192, 1], offset: ?>>
          %102 = arith.truncf %99 : tensor<64x48xf32> to tensor<64x48xf16>
          %subview_16 = memref.subview %reinterpret_cast_15[0, 0] [%73, 48] [1, 1] : memref<64x48xf16, strided<[192, 1], offset: ?>> to memref<?x48xf16, strided<[192, 1], offset: ?>>
          %extracted_slice = tensor.extract_slice %102[0, 0] [%73, 48] [1, 1] : tensor<64x48xf16> to tensor<?x48xf16>
          hivm.hir.store ins(%extracted_slice : tensor<?x48xf16>) outs(%subview_16 : memref<?x48xf16, strided<[192, 1], offset: ?>>) atomic = <add>
        }
      }
    }
    return
  }
}
