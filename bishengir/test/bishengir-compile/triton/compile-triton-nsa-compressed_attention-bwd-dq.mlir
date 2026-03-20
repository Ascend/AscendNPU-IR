// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t
#map = affine_map<(d0) -> (d0)>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @backward_dq(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg13: i32, %arg14: i32, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: f32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32, %arg35: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c31_i32 = arith.constant 31 : i32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 1.44269502 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<32xi32>
    %3 = linalg.fill ins(%c1_i32 : i32) outs(%2 : tensor<32xi32>) -> tensor<32xi32>
    %4 = tensor.empty() : tensor<32x32xf32>
    %5 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %6 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %7 = tensor.empty() : tensor<32x128xf32>
    %8 = linalg.fill ins(%cst_1 : f32) outs(%7 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %inserted = tensor.insert %arg16 into %0[%c0] : tensor<1xf32>
    %9 = arith.mulf %inserted, %1 : tensor<1xf32>
    %extracted = tensor.extract %9[%c0] : tensor<1xf32>
    %10 = arith.addi %arg35, %arg29 : i32
    %11 = arith.divsi %arg34, %arg14 : i32
    %12 = arith.index_cast %arg33 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg11 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %13 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %14 = arith.addi %12, %c1 : index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg11 to offset: [%14], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %15 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %16 = arith.subi %15, %13 : i32
    %reinterpret_cast_4 = memref.reinterpret_cast %arg12 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %17 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg12 to offset: [%14], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %18 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %19 = arith.subi %18, %17 : i32
    %20 = arith.muli %10, %c32_i32 : i32
    %21 = arith.addi %20, %arg9 : i32
    %22 = arith.subi %21, %c1_i32 : i32
    %23 = arith.cmpi sge, %22, %16 : i32
    scf.if %23 {
    } else {
      %24 = arith.muli %13, %arg17 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = arith.muli %arg34, %arg18 : i32
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.addi %25, %27 : index
      %29 = arith.index_cast %22 : i32 to index
      %30 = arith.index_cast %arg17 : i32 to index
      %31 = arith.muli %29, %30 : index
      %32 = arith.addi %31, %28 : index
      %33 = arith.index_cast %16 : i32 to index
      %34 = arith.index_cast %arg15 : i32 to index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [32, 128], strides: [%30, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %35 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : tensor<32xi32>) {
      ^bb0(%out: i32):
        %151 = linalg.index 0 : index
        %152 = arith.index_cast %151 : index to i32
        linalg.yield %152 : i32
      } -> tensor<32xi32>
      %36 = arith.muli %13, %arg27 : i32
      %37 = arith.index_cast %36 : i32 to index
      %38 = arith.muli %arg34, %arg28 : i32
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.addi %37, %39 : index
      %41 = arith.index_cast %arg27 : i32 to index
      %42 = arith.muli %29, %41 : index
      %43 = arith.addi %40, %42 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%43], sizes: [32, 128], strides: [%41, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %44 = arith.muli %17, %arg19 : i32
      %45 = arith.index_cast %44 : i32 to index
      %46 = arith.muli %11, %arg20 : i32
      %47 = arith.index_cast %46 : i32 to index
      %48 = arith.addi %45, %47 : index
      %49 = arith.muli %17, %arg21 : i32
      %50 = arith.index_cast %49 : i32 to index
      %51 = arith.muli %11, %arg22 : i32
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.addi %50, %52 : index
      %54 = arith.muli %13, %arg25 : i32
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.muli %arg34, %arg26 : i32
      %57 = arith.index_cast %56 : i32 to index
      %58 = arith.addi %55, %57 : index
      %59 = arith.index_cast %arg25 : i32 to index
      %60 = arith.muli %29, %59 : index
      %61 = arith.addi %60, %58 : index
      %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%61], sizes: [32, 128], strides: [%59, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %62 = arith.index_cast %13 : i32 to index
      %63 = arith.muli %arg34, %arg24 : i32
      %64 = arith.index_cast %63 : i32 to index
      %65 = arith.addi %62, %64 : index
      %66 = arith.index_cast %arg24 : i32 to index
      %67 = arith.maxsi %66, %c1 : index
      %68 = arith.addi %65, %29 : index
      %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%68], sizes: [1, 32], strides: [%67, 1] : memref<?xf32> to memref<1x32xf32, strided<[?, 1], offset: ?>>
      %69 = arith.muli %arg34, %arg23 : i32
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.addi %62, %70 : index
      %72 = arith.index_cast %arg23 : i32 to index
      %73 = arith.maxsi %72, %c1 : index
      %74 = arith.addi %71, %29 : index
      %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%74], sizes: [1, 32], strides: [%73, 1] : memref<?xf32> to memref<1x32xf32, strided<[?, 1], offset: ?>>
      %75 = linalg.fill ins(%22 : i32) outs(%2 : tensor<32xi32>) -> tensor<32xi32>
      %76 = arith.addi %35, %75 : tensor<32xi32>
      %77 = linalg.fill ins(%arg10 : i32) outs(%2 : tensor<32xi32>) -> tensor<32xi32>
      %78 = arith.muli %35, %77 : tensor<32xi32>
      %79 = linalg.fill ins(%arg9 : i32) outs(%2 : tensor<32xi32>) -> tensor<32xi32>
      %80 = arith.addi %78, %79 : tensor<32xi32>
      %81 = arith.subi %80, %3 : tensor<32xi32>
      %alloc = memref.alloc() : memref<32x128xf16>
      %82 = arith.divsi %31, %30 : index
      %83 = arith.subi %33, %82 : index
      %84 = arith.maxsi %83, %c0 : index
      %85 = arith.minsi %84, %c32 : index
      %86 = arith.remsi %31, %30 : index
      %87 = arith.subi %34, %86 : index
      %88 = arith.maxsi %87, %c0 : index
      %89 = arith.minsi %88, %c128 : index
      %90 = arith.cmpi slt, %85, %c32 : index
      %91 = arith.cmpi slt, %89, %c128 : index
      %92 = arith.ori %90, %91 : i1
      scf.if %92 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<32x128xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_6[0, 0] [%85, %89] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_11 = memref.subview %alloc[0, 0] [%85, %89] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview, %subview_11 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %93 = bufferization.to_tensor %alloc restrict writable : memref<32x128xf16>
      %alloc_12 = memref.alloc() : memref<32x128xf16>
      %94 = arith.divsi %60, %59 : index
      %95 = arith.subi %33, %94 : index
      %96 = arith.maxsi %95, %c0 : index
      %97 = arith.minsi %96, %c32 : index
      %98 = arith.remsi %60, %59 : index
      %99 = arith.subi %34, %98 : index
      %100 = arith.maxsi %99, %c0 : index
      %101 = arith.minsi %100, %c128 : index
      %102 = arith.cmpi slt, %97, %c32 : index
      %103 = arith.cmpi slt, %101, %c128 : index
      %104 = arith.ori %102, %103 : i1
      scf.if %104 {
        linalg.fill ins(%cst : f16) outs(%alloc_12 : memref<32x128xf16>)
      } {hivm.unlikely_condition}
      %subview_13 = memref.subview %reinterpret_cast_8[0, 0] [%97, %101] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_14 = memref.subview %alloc_12[0, 0] [%97, %101] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview_13, %subview_14 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %105 = bufferization.to_tensor %alloc_12 restrict writable : memref<32x128xf16>
      %alloc_15 = memref.alloc() : memref<1x32xf32>
      %106 = arith.divsi %29, %73 : index
      %107 = arith.subi %c1, %106 : index
      %108 = arith.maxsi %107, %c0 : index
      %109 = arith.minsi %108, %c1 : index
      %110 = arith.remsi %29, %73 : index
      %111 = arith.subi %33, %110 : index
      %112 = arith.maxsi %111, %c0 : index
      %113 = arith.minsi %112, %c32 : index
      %114 = arith.cmpi slt, %109, %c1 : index
      %115 = arith.cmpi slt, %113, %c32 : index
      %116 = arith.ori %114, %115 : i1
      scf.if %116 {
        linalg.fill ins(%cst_1 : f32) outs(%alloc_15 : memref<1x32xf32>)
      } {hivm.unlikely_condition}
      %subview_16 = memref.subview %reinterpret_cast_10[0, 0] [%109, %113] [1, 1] : memref<1x32xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_17 = memref.subview %alloc_15[0, 0] [%109, %113] [1, 1] : memref<1x32xf32> to memref<?x?xf32, strided<[32, 1]>>
      memref.copy %subview_16, %subview_17 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1]>>
      %117 = bufferization.to_tensor %alloc_15 restrict writable : memref<1x32xf32>
      %118 = tensor.empty() : tensor<32x1xf32>
      %transposed = linalg.transpose ins(%117 : tensor<1x32xf32>) outs(%118 : tensor<32x1xf32>) permutation = [1, 0]
      %alloc_18 = memref.alloc() : memref<1x32xf32>
      %119 = arith.divsi %29, %67 : index
      %120 = arith.subi %c1, %119 : index
      %121 = arith.maxsi %120, %c0 : index
      %122 = arith.minsi %121, %c1 : index
      %123 = arith.remsi %29, %67 : index
      %124 = arith.subi %33, %123 : index
      %125 = arith.maxsi %124, %c0 : index
      %126 = arith.minsi %125, %c32 : index
      %127 = arith.cmpi slt, %122, %c1 : index
      %128 = arith.cmpi slt, %126, %c32 : index
      %129 = arith.ori %127, %128 : i1
      scf.if %129 {
        linalg.fill ins(%cst_1 : f32) outs(%alloc_18 : memref<1x32xf32>)
      } {hivm.unlikely_condition}
      %subview_19 = memref.subview %reinterpret_cast_9[0, 0] [%122, %126] [1, 1] : memref<1x32xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_20 = memref.subview %alloc_18[0, 0] [%122, %126] [1, 1] : memref<1x32xf32> to memref<?x?xf32, strided<[32, 1]>>
      memref.copy %subview_19, %subview_20 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[32, 1]>>
      %130 = bufferization.to_tensor %alloc_18 restrict writable : memref<1x32xf32>
      %transposed_21 = linalg.transpose ins(%130 : tensor<1x32xf32>) outs(%118 : tensor<32x1xf32>) permutation = [1, 0]
      %131 = arith.addi %21, %c31_i32 : i32
      %132 = arith.subi %131, %arg9 : i32
      %133 = arith.divsi %132, %arg10 : i32
      %134 = arith.addi %133, %c1_i32 : i32
      %135 = arith.minsi %19, %134 : i32
      %136 = tensor.empty() : tensor<32x32xi32>
      %broadcasted = linalg.broadcast ins(%76 : tensor<32xi32>) outs(%136 : tensor<32x32xi32>) dimensions = [1]
      %137 = linalg.fill ins(%extracted : f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %collapsed = tensor.collapse_shape %transposed [[0, 1]] : tensor<32x1xf32> into tensor<32xf32>
      %broadcasted_22 = linalg.broadcast ins(%collapsed : tensor<32xf32>) outs(%4 : tensor<32x32xf32>) dimensions = [1]
      %138 = linalg.fill ins(%arg16 : f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %collapsed_23 = tensor.collapse_shape %transposed_21 [[0, 1]] : tensor<32x1xf32> into tensor<32xf32>
      %broadcasted_24 = linalg.broadcast ins(%collapsed_23 : tensor<32xf32>) outs(%4 : tensor<32x32xf32>) dimensions = [1]
      %139:3 = scf.for %arg36 = %c0_i32 to %135 step %c32_i32 iter_args(%arg37 = %8, %arg38 = %c0_i32, %arg39 = %c0_i32) -> (tensor<32x128xf32>, i32, i32)  : i32 {
        %151 = arith.index_cast %arg39 : i32 to index
        %152 = arith.index_cast %arg21 : i32 to index
        %153 = arith.muli %151, %152 : index
        %154 = arith.addi %153, %53 : index
        %155 = arith.index_cast %19 : i32 to index
        %reinterpret_cast_26 = memref.reinterpret_cast %arg4 to offset: [%154], sizes: [32, 128], strides: [%152, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
        %156 = arith.index_cast %arg38 : i32 to index
        %157 = arith.index_cast %arg19 : i32 to index
        %158 = arith.muli %156, %157 : index
        %159 = arith.addi %158, %48 : index
        %reinterpret_cast_27 = memref.reinterpret_cast %arg3 to offset: [%159], sizes: [32, 128], strides: [%157, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
        %alloc_28 = memref.alloc() : memref<32x128xf16>
        %160 = arith.divsi %158, %157 : index
        %161 = arith.subi %155, %160 : index
        %162 = arith.maxsi %161, %c0 : index
        %163 = arith.minsi %162, %c32 : index
        %164 = arith.remsi %158, %157 : index
        %165 = arith.subi %34, %164 : index
        %166 = arith.maxsi %165, %c0 : index
        %167 = arith.minsi %166, %c128 : index
        %168 = arith.cmpi slt, %163, %c32 : index
        %169 = arith.cmpi slt, %167, %c128 : index
        %170 = arith.ori %168, %169 : i1
        scf.if %170 {
          linalg.fill ins(%cst : f16) outs(%alloc_28 : memref<32x128xf16>)
        } {hivm.unlikely_condition}
        %subview_29 = memref.subview %reinterpret_cast_27[0, 0] [%163, %167] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_30 = memref.subview %alloc_28[0, 0] [%163, %167] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_29, %subview_30 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %171 = bufferization.to_tensor %alloc_28 restrict writable : memref<32x128xf16>
        %alloc_31 = memref.alloc() : memref<32x128xf16>
        %172 = arith.divsi %153, %152 : index
        %173 = arith.subi %155, %172 : index
        %174 = arith.maxsi %173, %c0 : index
        %175 = arith.minsi %174, %c32 : index
        %176 = arith.remsi %153, %152 : index
        %177 = arith.subi %34, %176 : index
        %178 = arith.maxsi %177, %c0 : index
        %179 = arith.minsi %178, %c128 : index
        %180 = arith.cmpi slt, %175, %c32 : index
        %181 = arith.cmpi slt, %179, %c128 : index
        %182 = arith.ori %180, %181 : i1
        scf.if %182 {
          linalg.fill ins(%cst : f16) outs(%alloc_31 : memref<32x128xf16>)
        } {hivm.unlikely_condition}
        %subview_32 = memref.subview %reinterpret_cast_26[0, 0] [%175, %179] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_33 = memref.subview %alloc_31[0, 0] [%175, %179] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_32, %subview_33 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %183 = bufferization.to_tensor %alloc_31 restrict writable : memref<32x128xf16>
        %184 = tensor.empty() : tensor<128x32xf16>
        %transposed_34 = linalg.transpose ins(%183 : tensor<32x128xf16>) outs(%184 : tensor<128x32xf16>) permutation = [1, 0]
        %185 = arith.muli %arg36, %arg10 : i32
        %186 = linalg.fill ins(%185 : i32) outs(%2 : tensor<32xi32>) -> tensor<32xi32>
        %187 = arith.addi %186, %81 : tensor<32xi32>
        %broadcasted_35 = linalg.broadcast ins(%187 : tensor<32xi32>) outs(%136 : tensor<32x32xi32>) dimensions = [0]
        %188 = arith.cmpi sge, %broadcasted, %broadcasted_35 : tensor<32x32xi32>
        %189 = arith.select %188, %5, %6 : tensor<32x32xi1>, tensor<32x32xf32>
        %190 = arith.addf %189, %5 : tensor<32x32xf32>
        %transposed_36 = linalg.transpose ins(%171 : tensor<32x128xf16>) outs(%184 : tensor<128x32xf16>) permutation = [1, 0]
        %191 = linalg.matmul {input_precison = "ieee"} ins(%93, %transposed_36 : tensor<32x128xf16>, tensor<128x32xf16>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %192 = arith.mulf %191, %137 : tensor<32x32xf32>
        %193 = arith.addf %190, %192 : tensor<32x32xf32>
        %194 = arith.subf %193, %broadcasted_22 : tensor<32x32xf32>
        %195 = math.exp2 %194 : tensor<32x32xf32>
        %196 = linalg.matmul {input_precison = "ieee"} ins(%105, %transposed_34 : tensor<32x128xf16>, tensor<128x32xf16>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %197 = arith.mulf %138, %195 : tensor<32x32xf32>
        %198 = arith.subf %196, %broadcasted_24 : tensor<32x32xf32>
        %199 = arith.mulf %197, %198 : tensor<32x32xf32>
        %200 = arith.truncf %199 : tensor<32x32xf32> to tensor<32x32xf16>
        %201 = linalg.matmul {input_precison = "ieee"} ins(%200, %171 : tensor<32x32xf16>, tensor<32x128xf16>) outs(%arg37 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %202 = arith.addi %arg38, %c32_i32 : i32
        %203 = arith.addi %arg39, %c32_i32 : i32
        scf.yield %201, %202, %203 : tensor<32x128xf32>, i32, i32
      }
      %140 = arith.truncf %139#0 : tensor<32x128xf32> to tensor<32x128xf16>
      %141 = arith.addi %29, %c32 : index
      %142 = arith.index_cast %16 : i32 to index
      %143 = arith.maxsi %29, %142 : index
      %144 = arith.minsi %141, %143 : index
      %145 = arith.subi %144, %29 : index
      %146 = arith.index_cast %arg15 : i32 to index
      %147 = arith.maxsi %146, %c0 : index
      %148 = arith.minsi %147, %c128 : index
      %149 = arith.minsi %145, %c32 : index
      %150 = arith.minsi %148, %c128 : index
      %extracted_slice = tensor.extract_slice %140[0, 0] [%149, %150] [1, 1] : tensor<32x128xf16> to tensor<?x?xf16>
      %subview_25 = memref.subview %reinterpret_cast_7[0, 0] [%149, %150] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_25 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }
}
