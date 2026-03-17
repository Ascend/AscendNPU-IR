// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t
#map = affine_map<(d0) -> (d0)>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @backward_dq(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: f32, %arg16: i1 {tt.divisibility = 16 : i32}, %arg17: i1 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32, %arg25: i32, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32, %arg35: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 1.44269502 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<128x128xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %4 = tensor.empty() : tensor<128x32xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<128x32xf32>) -> tensor<128x32xf32>
    %6 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<128x32xf32>) -> tensor<128x32xf32>
    %inserted = tensor.insert %arg15 into %0[%c0] : tensor<1xf32>
    %7 = arith.mulf %inserted, %1 : tensor<1xf32>
    %extracted = tensor.extract %7[%c0] : tensor<1xf32>
    %8 = scf.if %arg17 -> (i32) {
      %19 = arith.remsi %arg34, %arg11 : i32
      scf.yield %19 : i32
    } else {
      %19 = arith.divsi %arg34, %arg12 : i32
      scf.yield %19 : i32
    }
    %9 = arith.index_cast %arg33 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg9 to offset: [%9], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %10 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %11 = arith.addi %9, %c1 : index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg9 to offset: [%11], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %12 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %13 = arith.subi %12, %10 : i32
    %reinterpret_cast_4 = memref.reinterpret_cast %arg10 to offset: [%9], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %14 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg10 to offset: [%11], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %15 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %16 = arith.subi %15, %14 : i32
    %17 = arith.muli %arg35, %c128_i32 : i32
    %18 = arith.cmpi sge, %17, %13 : i32
    scf.if %18 {
    } else {
      %19 = arith.muli %10, %arg18 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.muli %arg34, %arg19 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.addi %20, %22 : index
      %24 = arith.index_cast %17 : i32 to index
      %25 = arith.index_cast %arg18 : i32 to index
      %26 = arith.muli %24, %25 : index
      %27 = arith.addi %26, %23 : index
      %28 = arith.index_cast %13 : i32 to index
      %29 = arith.index_cast %arg13 : i32 to index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [128, 128], strides: [%25, 1] : memref<?xf16> to memref<128x128xf16, strided<[?, 1], offset: ?>>
      %30 = tensor.empty() : tensor<128xi32>
      %31 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%30 : tensor<128xi32>) {
      ^bb0(%out: i32):
        %139 = linalg.index 0 : index
        %140 = arith.index_cast %139 : index to i32
        linalg.yield %140 : i32
      } -> tensor<128xi32>
      %32 = arith.muli %10, %arg28 : i32
      %33 = arith.index_cast %32 : i32 to index
      %34 = arith.muli %arg34, %arg29 : i32
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.addi %33, %35 : index
      %37 = arith.index_cast %arg28 : i32 to index
      %38 = arith.muli %24, %37 : index
      %39 = arith.addi %36, %38 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%39], sizes: [128, 128], strides: [%37, 1] : memref<?xf16> to memref<128x128xf16, strided<[?, 1], offset: ?>>
      %40 = arith.muli %14, %arg20 : i32
      %41 = arith.index_cast %40 : i32 to index
      %42 = arith.muli %8, %arg21 : i32
      %43 = arith.index_cast %42 : i32 to index
      %44 = arith.addi %41, %43 : index
      %45 = arith.muli %14, %arg22 : i32
      %46 = arith.index_cast %45 : i32 to index
      %47 = arith.muli %8, %arg23 : i32
      %48 = arith.index_cast %47 : i32 to index
      %49 = arith.addi %46, %48 : index
      %50 = arith.muli %10, %arg26 : i32
      %51 = arith.index_cast %50 : i32 to index
      %52 = arith.muli %arg34, %arg27 : i32
      %53 = arith.index_cast %52 : i32 to index
      %54 = arith.addi %51, %53 : index
      %55 = arith.index_cast %arg26 : i32 to index
      %56 = arith.muli %24, %55 : index
      %57 = arith.addi %56, %54 : index
      %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%57], sizes: [128, 128], strides: [%55, 1] : memref<?xf16> to memref<128x128xf16, strided<[?, 1], offset: ?>>
      %58 = arith.index_cast %10 : i32 to index
      %59 = arith.muli %arg34, %arg25 : i32
      %60 = arith.index_cast %59 : i32 to index
      %61 = arith.addi %58, %60 : index
      %62 = arith.index_cast %arg25 : i32 to index
      %63 = arith.maxsi %62, %c1 : index
      %64 = arith.addi %61, %24 : index
      %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%64], sizes: [1, 128], strides: [%63, 1] : memref<?xf32> to memref<1x128xf32, strided<[?, 1], offset: ?>>
      %65 = arith.muli %arg34, %arg24 : i32
      %66 = arith.index_cast %65 : i32 to index
      %67 = arith.addi %58, %66 : index
      %68 = arith.index_cast %arg24 : i32 to index
      %69 = arith.maxsi %68, %c1 : index
      %70 = arith.addi %67, %24 : index
      %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%70], sizes: [1, 128], strides: [%69, 1] : memref<?xf32> to memref<1x128xf32, strided<[?, 1], offset: ?>>
      %71 = linalg.fill ins(%17 : i32) outs(%30 : tensor<128xi32>) -> tensor<128xi32>
      %72 = arith.addi %31, %71 : tensor<128xi32>
      %73 = tensor.empty() : tensor<32xi32>
      %74 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%73 : tensor<32xi32>) {
      ^bb0(%out: i32):
        %139 = linalg.index 0 : index
        %140 = arith.index_cast %139 : index to i32
        linalg.yield %140 : i32
      } -> tensor<32xi32>
      %alloc = memref.alloc() : memref<128x128xf16>
      %75 = arith.divsi %26, %25 : index
      %76 = arith.subi %28, %75 : index
      %77 = arith.maxsi %76, %c0 : index
      %78 = arith.minsi %77, %c128 : index
      %79 = arith.remsi %26, %25 : index
      %80 = arith.subi %29, %79 : index
      %81 = arith.maxsi %80, %c0 : index
      %82 = arith.minsi %81, %c128 : index
      %83 = arith.cmpi slt, %78, %c128 : index
      %84 = arith.cmpi slt, %82, %c128 : index
      %85 = arith.ori %83, %84 : i1
      scf.if %85 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<128x128xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_6[0, 0] [%78, %82] [1, 1] : memref<128x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_11 = memref.subview %alloc[0, 0] [%78, %82] [1, 1] : memref<128x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview, %subview_11 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %86 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf16>
      %alloc_12 = memref.alloc() : memref<128x128xf16>
      %87 = arith.divsi %56, %55 : index
      %88 = arith.subi %28, %87 : index
      %89 = arith.maxsi %88, %c0 : index
      %90 = arith.minsi %89, %c128 : index
      %91 = arith.remsi %56, %55 : index
      %92 = arith.subi %29, %91 : index
      %93 = arith.maxsi %92, %c0 : index
      %94 = arith.minsi %93, %c128 : index
      %95 = arith.cmpi slt, %90, %c128 : index
      %96 = arith.cmpi slt, %94, %c128 : index
      %97 = arith.ori %95, %96 : i1
      scf.if %97 {
        linalg.fill ins(%cst : f16) outs(%alloc_12 : memref<128x128xf16>)
      } {hivm.unlikely_condition}
      %subview_13 = memref.subview %reinterpret_cast_8[0, 0] [%90, %94] [1, 1] : memref<128x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_14 = memref.subview %alloc_12[0, 0] [%90, %94] [1, 1] : memref<128x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview_13, %subview_14 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %98 = bufferization.to_tensor %alloc_12 restrict writable : memref<128x128xf16>
      %alloc_15 = memref.alloc() : memref<1x128xf32>
      %99 = arith.divsi %24, %69 : index
      %100 = arith.subi %c1, %99 : index
      %101 = arith.maxsi %100, %c0 : index
      %102 = arith.minsi %101, %c1 : index
      %103 = arith.remsi %24, %69 : index
      %104 = arith.subi %28, %103 : index
      %105 = arith.maxsi %104, %c0 : index
      %106 = arith.minsi %105, %c128 : index
      %107 = arith.cmpi slt, %102, %c1 : index
      %108 = arith.cmpi slt, %106, %c128 : index
      %109 = arith.ori %107, %108 : i1
      scf.if %109 {
        linalg.fill ins(%cst_1 : f32) outs(%alloc_15 : memref<1x128xf32>)
      } {hivm.unlikely_condition}
      %subview_16 = memref.subview %reinterpret_cast_10[0, 0] [%102, %106] [1, 1] : memref<1x128xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_17 = memref.subview %alloc_15[0, 0] [%102, %106] [1, 1] : memref<1x128xf32> to memref<?x?xf32, strided<[128, 1]>>
      memref.copy %subview_16, %subview_17 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1]>>
      %110 = bufferization.to_tensor %alloc_15 restrict writable : memref<1x128xf32>
      %111 = tensor.empty() : tensor<128x1xf32>
      %transposed = linalg.transpose ins(%110 : tensor<1x128xf32>) outs(%111 : tensor<128x1xf32>) permutation = [1, 0]
      %alloc_18 = memref.alloc() : memref<1x128xf32>
      %112 = arith.divsi %24, %63 : index
      %113 = arith.subi %c1, %112 : index
      %114 = arith.maxsi %113, %c0 : index
      %115 = arith.minsi %114, %c1 : index
      %116 = arith.remsi %24, %63 : index
      %117 = arith.subi %28, %116 : index
      %118 = arith.maxsi %117, %c0 : index
      %119 = arith.minsi %118, %c128 : index
      %120 = arith.cmpi slt, %115, %c1 : index
      %121 = arith.cmpi slt, %119, %c128 : index
      %122 = arith.ori %120, %121 : i1
      scf.if %122 {
        linalg.fill ins(%cst_1 : f32) outs(%alloc_18 : memref<1x128xf32>)
      } {hivm.unlikely_condition}
      %subview_19 = memref.subview %reinterpret_cast_9[0, 0] [%115, %119] [1, 1] : memref<1x128xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_20 = memref.subview %alloc_18[0, 0] [%115, %119] [1, 1] : memref<1x128xf32> to memref<?x?xf32, strided<[128, 1]>>
      memref.copy %subview_19, %subview_20 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1]>>
      %123 = bufferization.to_tensor %alloc_18 restrict writable : memref<1x128xf32>
      %transposed_21 = linalg.transpose ins(%123 : tensor<1x128xf32>) outs(%111 : tensor<128x1xf32>) permutation = [1, 0]
      %124 = scf.if %arg16 -> (i32) {
        %139 = arith.addi %arg35, %c1_i32 : i32
        %140 = arith.muli %139, %c128_i32 : i32
        scf.yield %140 : i32
      } else {
        scf.yield %16 : i32
      }
      %125 = linalg.fill ins(%extracted : f32) outs(%4 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %collapsed = tensor.collapse_shape %transposed [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
      %broadcasted = linalg.broadcast ins(%collapsed : tensor<128xf32>) outs(%4 : tensor<128x32xf32>) dimensions = [1]
      %126 = linalg.fill ins(%arg15 : f32) outs(%4 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %collapsed_22 = tensor.collapse_shape %transposed_21 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
      %broadcasted_23 = linalg.broadcast ins(%collapsed_22 : tensor<128xf32>) outs(%4 : tensor<128x32xf32>) dimensions = [1]
      %127:3 = scf.for %arg36 = %c0_i32 to %124 step %c32_i32 iter_args(%arg37 = %3, %arg38 = %c0_i32, %arg39 = %c0_i32) -> (tensor<128x128xf32>, i32, i32)  : i32 {
        %139 = arith.index_cast %arg39 : i32 to index
        %140 = arith.index_cast %arg22 : i32 to index
        %141 = arith.muli %139, %140 : index
        %142 = arith.addi %141, %49 : index
        %143 = arith.index_cast %16 : i32 to index
        %reinterpret_cast_25 = memref.reinterpret_cast %arg4 to offset: [%142], sizes: [32, 128], strides: [%140, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
        %144 = arith.index_cast %arg38 : i32 to index
        %145 = arith.index_cast %arg20 : i32 to index
        %146 = arith.muli %144, %145 : index
        %147 = arith.addi %146, %44 : index
        %reinterpret_cast_26 = memref.reinterpret_cast %arg3 to offset: [%147], sizes: [32, 128], strides: [%145, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
        %alloc_27 = memref.alloc() : memref<32x128xf16>
        %148 = arith.divsi %146, %145 : index
        %149 = arith.subi %143, %148 : index
        %150 = arith.maxsi %149, %c0 : index
        %151 = arith.minsi %150, %c32 : index
        %152 = arith.remsi %146, %145 : index
        %153 = arith.subi %29, %152 : index
        %154 = arith.maxsi %153, %c0 : index
        %155 = arith.minsi %154, %c128 : index
        %156 = arith.cmpi slt, %151, %c32 : index
        %157 = arith.cmpi slt, %155, %c128 : index
        %158 = arith.ori %156, %157 : i1
        scf.if %158 {
          linalg.fill ins(%cst : f16) outs(%alloc_27 : memref<32x128xf16>)
        } {hivm.unlikely_condition}
        %subview_28 = memref.subview %reinterpret_cast_26[0, 0] [%151, %155] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_29 = memref.subview %alloc_27[0, 0] [%151, %155] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_28, %subview_29 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %159 = bufferization.to_tensor %alloc_27 restrict writable : memref<32x128xf16>
        %alloc_30 = memref.alloc() : memref<32x128xf16>
        %160 = arith.divsi %141, %140 : index
        %161 = arith.subi %143, %160 : index
        %162 = arith.maxsi %161, %c0 : index
        %163 = arith.minsi %162, %c32 : index
        %164 = arith.remsi %141, %140 : index
        %165 = arith.subi %29, %164 : index
        %166 = arith.maxsi %165, %c0 : index
        %167 = arith.minsi %166, %c128 : index
        %168 = arith.cmpi slt, %163, %c32 : index
        %169 = arith.cmpi slt, %167, %c128 : index
        %170 = arith.ori %168, %169 : i1
        scf.if %170 {
          linalg.fill ins(%cst : f16) outs(%alloc_30 : memref<32x128xf16>)
        } {hivm.unlikely_condition}
        %subview_31 = memref.subview %reinterpret_cast_25[0, 0] [%163, %167] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_32 = memref.subview %alloc_30[0, 0] [%163, %167] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_31, %subview_32 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %171 = bufferization.to_tensor %alloc_30 restrict writable : memref<32x128xf16>
        %172 = scf.if %arg16 -> (tensor<128x32xf32>) {
          %187 = linalg.fill ins(%arg36 : i32) outs(%73 : tensor<32xi32>) -> tensor<32xi32>
          %188 = arith.addi %74, %187 : tensor<32xi32>
          %189 = tensor.empty() : tensor<128x32xi32>
          %broadcasted_35 = linalg.broadcast ins(%72 : tensor<128xi32>) outs(%189 : tensor<128x32xi32>) dimensions = [1]
          %broadcasted_36 = linalg.broadcast ins(%188 : tensor<32xi32>) outs(%189 : tensor<128x32xi32>) dimensions = [0]
          %190 = arith.cmpi sge, %broadcasted_35, %broadcasted_36 : tensor<128x32xi32>
          %191 = arith.select %190, %6, %5 : tensor<128x32xi1>, tensor<128x32xf32>
          scf.yield %191 : tensor<128x32xf32>
        } else {
          scf.yield %6 : tensor<128x32xf32>
        }
        %173 = tensor.empty() : tensor<128x32xf16>
        %transposed_33 = linalg.transpose ins(%159 : tensor<32x128xf16>) outs(%173 : tensor<128x32xf16>) permutation = [1, 0]
        %174 = linalg.matmul {input_precison = "ieee"} ins(%86, %transposed_33 : tensor<128x128xf16>, tensor<128x32xf16>) outs(%6 : tensor<128x32xf32>) -> tensor<128x32xf32>
        %175 = arith.mulf %174, %125 : tensor<128x32xf32>
        %176 = arith.addf %172, %175 : tensor<128x32xf32>
        %177 = arith.subf %176, %broadcasted : tensor<128x32xf32>
        %178 = math.exp2 %177 : tensor<128x32xf32>
        %transposed_34 = linalg.transpose ins(%171 : tensor<32x128xf16>) outs(%173 : tensor<128x32xf16>) permutation = [1, 0]
        %179 = linalg.matmul {input_precison = "ieee"} ins(%98, %transposed_34 : tensor<128x128xf16>, tensor<128x32xf16>) outs(%6 : tensor<128x32xf32>) -> tensor<128x32xf32>
        %180 = arith.mulf %126, %178 : tensor<128x32xf32>
        %181 = arith.subf %179, %broadcasted_23 : tensor<128x32xf32>
        %182 = arith.mulf %180, %181 : tensor<128x32xf32>
        %183 = arith.truncf %182 : tensor<128x32xf32> to tensor<128x32xf16>
        %184 = linalg.matmul {input_precison = "ieee"} ins(%183, %159 : tensor<128x32xf16>, tensor<32x128xf16>) outs(%arg37 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %185 = arith.addi %arg38, %c32_i32 : i32
        %186 = arith.addi %arg39, %c32_i32 : i32
        scf.yield %184, %185, %186 : tensor<128x128xf32>, i32, i32
      }
      %128 = arith.truncf %127#0 : tensor<128x128xf32> to tensor<128x128xf16>
      %129 = arith.addi %24, %c128 : index
      %130 = arith.index_cast %13 : i32 to index
      %131 = arith.maxsi %24, %130 : index
      %132 = arith.minsi %129, %131 : index
      %133 = arith.subi %132, %24 : index
      %134 = arith.index_cast %arg13 : i32 to index
      %135 = arith.maxsi %134, %c0 : index
      %136 = arith.minsi %135, %c128 : index
      %137 = arith.minsi %133, %c128 : index
      %138 = arith.minsi %136, %c128 : index
      %extracted_slice = tensor.extract_slice %128[0, 0] [%137, %138] [1, 1] : tensor<128x128xf16> to tensor<?x?xf16>
      %subview_24 = memref.subview %reinterpret_cast_7[0, 0] [%137, %138] [1, 1] : memref<128x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_24 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }
}
