// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t
#map = affine_map<(d0) -> (d0)>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @score_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: i32, %arg11: i32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: f32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 1.44269502 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<64xi32>
    %3 = linalg.fill ins(%c1_i32 : i32) outs(%2 : tensor<64xi32>) -> tensor<64xi32>
    %4 = tensor.empty() : tensor<64x64xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %inserted = tensor.insert %arg13 into %0[%c0] : tensor<1xf32>
    %6 = arith.mulf %inserted, %1 : tensor<1xf32>
    %extracted = tensor.extract %6[%c0] : tensor<1xf32>
    %7 = arith.divsi %arg25, %arg10 : i32
    %8 = arith.remsi %arg25, %arg10 : i32
    %9 = arith.addi %arg26, %arg21 : i32
    %10 = arith.index_cast %7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%10], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %11 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %12 = arith.addi %10, %c1 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg8 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %13 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %14 = arith.subi %13, %11 : i32
    %reinterpret_cast_3 = memref.reinterpret_cast %arg9 to offset: [%10], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %15 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg9 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %16 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %17 = arith.subi %16, %15 : i32
    %18 = arith.muli %9, %c64_i32 : i32
    %19 = arith.cmpi sge, %18, %14 : i32
    %20 = arith.muli %arg27, %c64_i32 : i32
    %21 = arith.cmpi sge, %20, %17 : i32
    %22 = arith.ori %19, %21 : i1
    scf.if %22 {
    } else {
      %23 = arith.muli %15, %arg16 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.muli %8, %arg17 : i32
      %26 = arith.index_cast %25 : i32 to index
      %27 = arith.addi %24, %26 : index
      %28 = arith.index_cast %20 : i32 to index
      %29 = arith.index_cast %arg16 : i32 to index
      %30 = arith.muli %28, %29 : index
      %31 = arith.addi %30, %27 : index
      %32 = arith.index_cast %17 : i32 to index
      %33 = arith.index_cast %arg12 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%31], sizes: [64, 128], strides: [%29, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
      %alloc = memref.alloc() : memref<64x128xf16>
      %34 = arith.divsi %30, %29 : index
      %35 = arith.subi %32, %34 : index
      %36 = arith.maxsi %35, %c0 : index
      %37 = arith.minsi %36, %c64 : index
      %38 = arith.remsi %30, %29 : index
      %39 = arith.subi %33, %38 : index
      %40 = arith.maxsi %39, %c0 : index
      %41 = arith.minsi %40, %c128 : index
      %42 = arith.cmpi slt, %37, %c64 : index
      %43 = arith.cmpi slt, %41, %c128 : index
      %44 = arith.ori %42, %43 : i1
      scf.if %44 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<64x128xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_5[0, 0] [%37, %41] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %alloc[0, 0] [%37, %41] [1, 1] : memref<64x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview, %subview_6 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %45 = bufferization.to_tensor %alloc restrict writable : memref<64x128xf16>
      %46 = tensor.empty() : tensor<128x64xf16>
      %transposed = linalg.transpose ins(%45 : tensor<64x128xf16>) outs(%46 : tensor<128x64xf16>) permutation = [1, 0]
      %47 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : tensor<64xi32>) {
      ^bb0(%out: i32):
        %87 = linalg.index 0 : index
        %88 = arith.index_cast %87 : index to i32
        linalg.yield %88 : i32
      } -> tensor<64xi32>
      %48 = linalg.fill ins(%18 : i32) outs(%2 : tensor<64xi32>) -> tensor<64xi32>
      %49 = arith.addi %47, %48 : tensor<64xi32>
      %50 = linalg.fill ins(%20 : i32) outs(%2 : tensor<64xi32>) -> tensor<64xi32>
      %51 = arith.addi %47, %50 : tensor<64xi32>
      %52 = linalg.fill ins(%arg7 : i32) outs(%2 : tensor<64xi32>) -> tensor<64xi32>
      %53 = arith.muli %51, %52 : tensor<64xi32>
      %54 = linalg.fill ins(%arg6 : i32) outs(%2 : tensor<64xi32>) -> tensor<64xi32>
      %55 = arith.addi %53, %54 : tensor<64xi32>
      %56 = arith.subi %55, %3 : tensor<64xi32>
      %57 = tensor.empty() : tensor<64x64xi32>
      %broadcasted = linalg.broadcast ins(%49 : tensor<64xi32>) outs(%57 : tensor<64x64xi32>) dimensions = [1]
      %broadcasted_7 = linalg.broadcast ins(%56 : tensor<64xi32>) outs(%57 : tensor<64x64xi32>) dimensions = [0]
      %58 = arith.cmpi sge, %broadcasted, %broadcasted_7 : tensor<64x64xi32>
      %59 = arith.muli %8, %arg11 : i32
      %60 = arith.muli %11, %arg14 : i32
      %61 = arith.index_cast %60 : i32 to index
      %62 = arith.index_cast %11 : i32 to index
      %63 = linalg.fill ins(%extracted : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %64 = scf.for %arg28 = %c0_i32 to %arg11 step %c1_i32 iter_args(%arg29 = %5) -> (tensor<64x64xf32>)  : i32 {
        %87 = arith.addi %59, %arg28 : i32
        %88 = arith.muli %87, %arg15 : i32
        %89 = arith.index_cast %88 : i32 to index
        %90 = arith.addi %61, %89 : index
        %91 = arith.index_cast %18 : i32 to index
        %92 = arith.index_cast %arg14 : i32 to index
        %93 = arith.muli %91, %92 : index
        %94 = arith.addi %93, %90 : index
        %95 = arith.index_cast %14 : i32 to index
        %reinterpret_cast_10 = memref.reinterpret_cast %arg2 to offset: [%94], sizes: [64, 128], strides: [%92, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
        %96 = arith.muli %87, %arg18 : i32
        %97 = arith.index_cast %96 : i32 to index
        %98 = arith.addi %62, %97 : index
        %99 = arith.index_cast %arg18 : i32 to index
        %100 = arith.maxsi %99, %c1 : index
        %101 = arith.addi %98, %91 : index
        %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%101], sizes: [1, 64], strides: [%100, 1] : memref<?xf32> to memref<1x64xf32, strided<[?, 1], offset: ?>>
        %alloc_12 = memref.alloc() : memref<64x128xf16>
        %102 = arith.divsi %93, %92 : index
        %103 = arith.subi %95, %102 : index
        %104 = arith.maxsi %103, %c0 : index
        %105 = arith.minsi %104, %c64 : index
        %106 = arith.remsi %93, %92 : index
        %107 = arith.subi %33, %106 : index
        %108 = arith.maxsi %107, %c0 : index
        %109 = arith.minsi %108, %c128 : index
        %110 = arith.cmpi slt, %105, %c64 : index
        %111 = arith.cmpi slt, %109, %c128 : index
        %112 = arith.ori %110, %111 : i1
        scf.if %112 {
          linalg.fill ins(%cst : f16) outs(%alloc_12 : memref<64x128xf16>)
        } {hivm.unlikely_condition}
        %subview_13 = memref.subview %reinterpret_cast_10[0, 0] [%105, %109] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_14 = memref.subview %alloc_12[0, 0] [%105, %109] [1, 1] : memref<64x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_13, %subview_14 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %113 = bufferization.to_tensor %alloc_12 restrict writable : memref<64x128xf16>
        %alloc_15 = memref.alloc() : memref<1x64xf32>
        %114 = arith.divsi %91, %100 : index
        %115 = arith.subi %c1, %114 : index
        %116 = arith.maxsi %115, %c0 : index
        %117 = arith.minsi %116, %c1 : index
        %118 = arith.remsi %91, %100 : index
        %119 = arith.subi %95, %118 : index
        %120 = arith.maxsi %119, %c0 : index
        %121 = arith.minsi %120, %c64 : index
        %122 = arith.cmpi slt, %117, %c1 : index
        %123 = arith.cmpi slt, %121, %c64 : index
        %124 = arith.ori %122, %123 : i1
        scf.if %124 {
          linalg.fill ins(%cst_0 : f32) outs(%alloc_15 : memref<1x64xf32>)
        } {hivm.unlikely_condition}
        %subview_16 = memref.subview %reinterpret_cast_11[0, 0] [%117, %121] [1, 1] : memref<1x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_17 = memref.subview %alloc_15[0, 0] [%117, %121] [1, 1] : memref<1x64xf32> to memref<?x?xf32, strided<[64, 1]>>
        memref.copy %subview_16, %subview_17 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
        %125 = bufferization.to_tensor %alloc_15 restrict writable : memref<1x64xf32>
        %126 = tensor.empty() : tensor<64x1xf32>
        %transposed_18 = linalg.transpose ins(%125 : tensor<1x64xf32>) outs(%126 : tensor<64x1xf32>) permutation = [1, 0]
        %127 = linalg.matmul {input_precison = "ieee"} ins(%113, %transposed : tensor<64x128xf16>, tensor<128x64xf16>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %128 = arith.mulf %127, %63 : tensor<64x64xf32>
        %129 = arith.addf %128, %5 : tensor<64x64xf32>
        %collapsed = tensor.collapse_shape %transposed_18 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
        %broadcasted_19 = linalg.broadcast ins(%collapsed : tensor<64xf32>) outs(%4 : tensor<64x64xf32>) dimensions = [1]
        %130 = arith.subf %129, %broadcasted_19 : tensor<64x64xf32>
        %131 = math.exp2 %130 : tensor<64x64xf32>
        %132 = arith.select %58, %131, %5 : tensor<64x64xi1>, tensor<64x64xf32>
        %133 = arith.addf %arg29, %132 : tensor<64x64xf32>
        scf.yield %133 : tensor<64x64xf32>
      }
      %65 = arith.muli %8, %arg19 : i32
      %66 = arith.index_cast %65 : i32 to index
      %67 = arith.muli %11, %arg20 : i32
      %68 = arith.index_cast %67 : i32 to index
      %69 = arith.addi %66, %68 : index
      %70 = arith.index_cast %18 : i32 to index
      %71 = arith.index_cast %arg20 : i32 to index
      %72 = arith.muli %70, %71 : index
      %73 = arith.addi %69, %72 : index
      %74 = arith.addi %73, %28 : index
      %reinterpret_cast_8 = memref.reinterpret_cast %arg5 to offset: [%74], sizes: [64, 64], strides: [%71, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
      %75 = arith.addi %70, %c64 : index
      %76 = arith.index_cast %14 : i32 to index
      %77 = arith.maxsi %70, %76 : index
      %78 = arith.minsi %75, %77 : index
      %79 = arith.subi %78, %70 : index
      %80 = arith.addi %28, %c64 : index
      %81 = arith.index_cast %17 : i32 to index
      %82 = arith.maxsi %28, %81 : index
      %83 = arith.minsi %80, %82 : index
      %84 = arith.subi %83, %28 : index
      %85 = arith.minsi %79, %c64 : index
      %86 = arith.minsi %84, %c64 : index
      %extracted_slice = tensor.extract_slice %64[0, 0] [%85, %86] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
      %subview_9 = memref.subview %reinterpret_cast_8[0, 0] [%85, %86] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_9 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }
}
