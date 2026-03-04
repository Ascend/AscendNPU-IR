// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1
#map = affine_map<(d0) -> (d0)>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @backward_dq(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: i32, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: f32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i32, %arg37: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %c-1_i32 = arith.constant -1 : i32
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 1.44269502 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<8xi32>
    %3 = linalg.fill ins(%c1_i32 : i32) outs(%2 : tensor<8xi32>) -> tensor<8xi32>
    %4 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<8xi32>) -> tensor<8xi32>
    %5 = tensor.empty() : tensor<1x64xf32>
    %6 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %8 = tensor.empty() : tensor<16x64xf32>
    %9 = linalg.fill ins(%cst_1 : f32) outs(%8 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %10 = tensor.empty() : tensor<16x128xf32>
    %11 = linalg.fill ins(%cst_1 : f32) outs(%10 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %inserted = tensor.insert %arg16 into %0[%c0] : tensor<1xf32>
    %12 = arith.mulf %inserted, %1 : tensor<1xf32>
    %extracted = tensor.extract %12[%c0] : tensor<1xf32>
    %13 = arith.addi %arg37, %arg31 : i32
    %14 = arith.muli %arg36, %arg13 : i32
    %15 = arith.index_cast %arg35 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg10 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %16 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %17 = arith.addi %15, %c1 : index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg10 to offset: [%17], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %18 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %19 = arith.subi %18, %16 : i32
    %reinterpret_cast_4 = memref.reinterpret_cast %arg11 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %20 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg11 to offset: [%17], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %21 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %22 = arith.subi %21, %20 : i32
    %23 = arith.cmpi sge, %13, %19 : i32
    scf.if %23 {
    } else {
      %24 = arith.subi %19, %13 : i32
      %25 = arith.minsi %24, %c1_i32 : i32
      %26 = arith.muli %arg36, %arg23 : i32
      %27 = arith.muli %14, %arg18 : i32
      %28 = arith.muli %14, %arg30 : i32
      %29 = arith.muli %20, %arg19 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.muli %arg36, %arg20 : i32
      %32 = arith.index_cast %31 : i32 to index
      %33 = arith.addi %30, %32 : index
      %34 = arith.index_cast %arg19 : i32 to index
      %35 = arith.index_cast %22 : i32 to index
      %36 = arith.index_cast %arg14 : i32 to index
      %37 = arith.muli %20, %arg21 : i32
      %38 = arith.index_cast %37 : i32 to index
      %39 = arith.muli %arg36, %arg22 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = arith.addi %38, %40 : index
      %42 = arith.index_cast %arg21 : i32 to index
      %43 = arith.muli %14, %arg28 : i32
      %44 = arith.muli %14, %arg26 : i32
      %45 = arith.muli %14, %arg25 : i32
      %46 = tensor.empty() : tensor<64xi32>
      %47 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%46 : tensor<64xi32>) {
      ^bb0(%out: i32):
        %50 = linalg.index 0 : index
        %51 = arith.index_cast %50 : index to i32
        linalg.yield %51 : i32
      } -> tensor<64xi32>
      %48 = linalg.fill ins(%extracted : f32) outs(%8 : tensor<16x64xf32>) -> tensor<16x64xf32>
      %49 = linalg.fill ins(%arg16 : f32) outs(%8 : tensor<16x64xf32>) -> tensor<16x64xf32>
      scf.for %arg38 = %c0_i32 to %25 step %c1_i32  : i32 {
        %50 = arith.addi %13, %arg38 : i32
        %51 = arith.addi %16, %50 : i32
        %52 = arith.muli %51, %arg24 : i32
        %53 = arith.index_cast %52 : i32 to index
        %54 = arith.index_cast %26 : i32 to index
        %55 = arith.addi %53, %54 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%55], sizes: [8], strides: [1] : memref<?xi32> to memref<8xi32, strided<[1], offset: ?>>
        %alloc = memref.alloc() : memref<8xi32>
        %56 = arith.index_cast %arg15 : i32 to index
        %57 = arith.maxsi %56, %c0 : index
        %58 = arith.minsi %57, %c8 : index
        %59 = arith.cmpi slt, %58, %c8 : index
        scf.if %59 {
          linalg.fill ins(%c-1_i32 : i32) outs(%alloc : memref<8xi32>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast_6[0] [%58] [1] : memref<8xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
        %subview_7 = memref.subview %alloc[0] [%58] [1] : memref<8xi32> to memref<?xi32, strided<[1]>>
        memref.copy %subview, %subview_7 : memref<?xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1]>>
        %60 = bufferization.to_tensor %alloc restrict writable : memref<8xi32>
        %61 = arith.cmpi sge, %60, %4 : tensor<8xi32>
        %62 = arith.divsi %50, %c64_i32 : i32
        %63 = linalg.fill ins(%62 : i32) outs(%2 : tensor<8xi32>) -> tensor<8xi32>
        %64 = arith.cmpi sle, %60, %63 : tensor<8xi32>
        %65 = arith.andi %61, %64 : tensor<8xi1>
        %66 = arith.select %65, %3, %4 : tensor<8xi1>, tensor<8xi32>
        %67 = bufferization.alloc_tensor() : tensor<i32>
        %68 = linalg.fill ins(%c0_i32 : i32) outs(%67 : tensor<i32>) -> tensor<i32>
        %reduced = linalg.reduce ins(%66 : tensor<8xi32>) outs(%68 : tensor<i32>) dimensions = [0] 
          (%in: i32, %init: i32) {
            %171 = arith.addi %in, %init : i32
            linalg.yield %171 : i32
          }
        %extracted_8 = tensor.extract %reduced[] : tensor<i32>
        %69 = arith.muli %51, %arg17 : i32
        %70 = arith.index_cast %69 : i32 to index
        %71 = arith.index_cast %27 : i32 to index
        %72 = arith.addi %70, %71 : index
        %73 = arith.index_cast %arg18 : i32 to index
        %74 = arith.index_cast %arg13 : i32 to index
        %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%72], sizes: [16, 128], strides: [%73, 1] : memref<?xf16> to memref<16x128xf16, strided<[?, 1], offset: ?>>
        %75 = arith.muli %51, %arg29 : i32
        %76 = arith.index_cast %75 : i32 to index
        %77 = arith.index_cast %28 : i32 to index
        %78 = arith.addi %76, %77 : index
        %79 = arith.index_cast %arg30 : i32 to index
        %reinterpret_cast_10 = memref.reinterpret_cast %arg9 to offset: [%78], sizes: [16, 128], strides: [%79, 1] : memref<?xf16> to memref<16x128xf16, strided<[?, 1], offset: ?>>
        %80 = arith.muli %51, %arg27 : i32
        %81 = arith.index_cast %80 : i32 to index
        %82 = arith.index_cast %43 : i32 to index
        %83 = arith.addi %81, %82 : index
        %84 = arith.index_cast %arg28 : i32 to index
        %reinterpret_cast_11 = memref.reinterpret_cast %arg8 to offset: [%83], sizes: [16, 128], strides: [%84, 1] : memref<?xf16> to memref<16x128xf16, strided<[?, 1], offset: ?>>
        %85 = arith.index_cast %51 : i32 to index
        %86 = arith.index_cast %44 : i32 to index
        %87 = arith.addi %85, %86 : index
        %88 = arith.index_cast %arg26 : i32 to index
        %reinterpret_cast_12 = memref.reinterpret_cast %arg7 to offset: [%87], sizes: [16, 1], strides: [%88, 1] : memref<?xf32> to memref<16x1xf32, strided<[?, 1], offset: ?>>
        %89 = arith.index_cast %45 : i32 to index
        %90 = arith.addi %85, %89 : index
        %91 = arith.index_cast %arg25 : i32 to index
        %reinterpret_cast_13 = memref.reinterpret_cast %arg6 to offset: [%90], sizes: [16, 1], strides: [%91, 1] : memref<?xf32> to memref<16x1xf32, strided<[?, 1], offset: ?>>
        %alloc_14 = memref.alloc() : memref<16x128xf16>
        %92 = arith.divsi %c0, %73 : index
        %93 = arith.subi %74, %92 : index
        %94 = arith.maxsi %93, %c0 : index
        %95 = arith.minsi %94, %c16 : index
        %96 = arith.remsi %c0, %73 : index
        %97 = arith.subi %36, %96 : index
        %98 = arith.maxsi %97, %c0 : index
        %99 = arith.minsi %98, %c128 : index
        %100 = arith.minsi %95, %c0 : index
        %101 = arith.subi %95, %100 : index
        %102 = arith.minsi %99, %c0 : index
        %103 = arith.subi %99, %102 : index
        %104 = arith.cmpi slt, %101, %c16 : index
        %105 = arith.cmpi slt, %103, %c128 : index
        %106 = arith.ori %104, %105 : i1
        scf.if %106 {
          linalg.fill ins(%cst : f16) outs(%alloc_14 : memref<16x128xf16>)
        } {hivm.unlikely_condition}
        %subview_15 = memref.subview %reinterpret_cast_9[0, 0] [%101, %103] [1, 1] : memref<16x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_16 = memref.subview %alloc_14[%100, %102] [%101, %103] [1, 1] : memref<16x128xf16> to memref<?x?xf16, strided<[128, 1], offset: ?>>
        memref.copy %subview_15, %subview_16 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1], offset: ?>>
        %107 = bufferization.to_tensor %alloc_14 restrict writable : memref<16x128xf16>
        %alloc_17 = memref.alloc() : memref<16x128xf16>
        %108 = arith.divsi %c0, %84 : index
        %109 = arith.subi %74, %108 : index
        %110 = arith.maxsi %109, %c0 : index
        %111 = arith.minsi %110, %c16 : index
        %112 = arith.remsi %c0, %84 : index
        %113 = arith.subi %36, %112 : index
        %114 = arith.maxsi %113, %c0 : index
        %115 = arith.minsi %114, %c128 : index
        %116 = arith.minsi %111, %c0 : index
        %117 = arith.subi %111, %116 : index
        %118 = arith.minsi %115, %c0 : index
        %119 = arith.subi %115, %118 : index
        %120 = arith.cmpi slt, %117, %c16 : index
        %121 = arith.cmpi slt, %119, %c128 : index
        %122 = arith.ori %120, %121 : i1
        scf.if %122 {
          linalg.fill ins(%cst : f16) outs(%alloc_17 : memref<16x128xf16>)
        } {hivm.unlikely_condition}
        %subview_18 = memref.subview %reinterpret_cast_11[0, 0] [%117, %119] [1, 1] : memref<16x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_19 = memref.subview %alloc_17[%116, %118] [%117, %119] [1, 1] : memref<16x128xf16> to memref<?x?xf16, strided<[128, 1], offset: ?>>
        memref.copy %subview_18, %subview_19 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1], offset: ?>>
        %123 = bufferization.to_tensor %alloc_17 restrict writable : memref<16x128xf16>
        %alloc_20 = memref.alloc() : memref<16x1xf32>
        %124 = arith.divsi %c0, %91 : index
        %125 = arith.subi %74, %124 : index
        %126 = arith.maxsi %125, %c0 : index
        %127 = arith.minsi %126, %c16 : index
        %128 = arith.remsi %c0, %91 : index
        %129 = arith.subi %c1, %128 : index
        %130 = arith.maxsi %129, %c0 : index
        %131 = arith.minsi %130, %c1 : index
        %132 = arith.minsi %127, %c0 : index
        %133 = arith.subi %127, %132 : index
        %134 = arith.minsi %131, %c0 : index
        %135 = arith.subi %131, %134 : index
        %136 = arith.cmpi slt, %133, %c16 : index
        %137 = arith.cmpi slt, %135, %c1 : index
        %138 = arith.ori %136, %137 : i1
        scf.if %138 {
          linalg.fill ins(%cst_1 : f32) outs(%alloc_20 : memref<16x1xf32>)
        } {hivm.unlikely_condition}
        %subview_21 = memref.subview %reinterpret_cast_13[0, 0] [%133, %135] [1, 1] : memref<16x1xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_22 = memref.subview %alloc_20[%132, %134] [%133, %135] [1, 1] : memref<16x1xf32> to memref<?x?xf32, strided<[1, 1], offset: ?>>
        memref.copy %subview_21, %subview_22 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[1, 1], offset: ?>>
        %139 = bufferization.to_tensor %alloc_20 restrict writable : memref<16x1xf32>
        %alloc_23 = memref.alloc() : memref<16x1xf32>
        %140 = arith.divsi %c0, %88 : index
        %141 = arith.subi %74, %140 : index
        %142 = arith.maxsi %141, %c0 : index
        %143 = arith.minsi %142, %c16 : index
        %144 = arith.remsi %c0, %88 : index
        %145 = arith.subi %c1, %144 : index
        %146 = arith.maxsi %145, %c0 : index
        %147 = arith.minsi %146, %c1 : index
        %148 = arith.minsi %143, %c0 : index
        %149 = arith.subi %143, %148 : index
        %150 = arith.minsi %147, %c0 : index
        %151 = arith.subi %147, %150 : index
        %152 = arith.cmpi slt, %149, %c16 : index
        %153 = arith.cmpi slt, %151, %c1 : index
        %154 = arith.ori %152, %153 : i1
        scf.if %154 {
          linalg.fill ins(%cst_1 : f32) outs(%alloc_23 : memref<16x1xf32>)
        } {hivm.unlikely_condition}
        %subview_24 = memref.subview %reinterpret_cast_12[0, 0] [%149, %151] [1, 1] : memref<16x1xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_25 = memref.subview %alloc_23[%148, %150] [%149, %151] [1, 1] : memref<16x1xf32> to memref<?x?xf32, strided<[1, 1], offset: ?>>
        memref.copy %subview_24, %subview_25 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[1, 1], offset: ?>>
        %155 = bufferization.to_tensor %alloc_23 restrict writable : memref<16x1xf32>
        %156 = linalg.fill ins(%50 : i32) outs(%46 : tensor<64xi32>) -> tensor<64xi32>
        %collapsed = tensor.collapse_shape %139 [[0, 1]] : tensor<16x1xf32> into tensor<16xf32>
        %broadcasted = linalg.broadcast ins(%collapsed : tensor<16xf32>) outs(%8 : tensor<16x64xf32>) dimensions = [1] 
        %collapsed_26 = tensor.collapse_shape %155 [[0, 1]] : tensor<16x1xf32> into tensor<16xf32>
        %broadcasted_27 = linalg.broadcast ins(%collapsed_26 : tensor<16xf32>) outs(%8 : tensor<16x64xf32>) dimensions = [1] 
        %157 = scf.for %arg39 = %c0_i32 to %extracted_8 step %c1_i32 iter_args(%arg40 = %11) -> (tensor<16x128xf32>)  : i32 {
          %171 = arith.index_cast %arg39 : i32 to index
          %172 = arith.addi %55, %171 : index
          %reinterpret_cast_29 = memref.reinterpret_cast %arg5 to offset: [%172], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
          %173 = memref.load %reinterpret_cast_29[%c0] : memref<1xi32, strided<[1], offset: ?>>
          %174 = arith.muli %173, %c64_i32 : i32
          %175 = arith.index_cast %174 : i32 to index
          %176 = arith.muli %175, %34 : index
          %177 = arith.addi %176, %33 : index
          %reinterpret_cast_30 = memref.reinterpret_cast %arg3 to offset: [%177], sizes: [64, 128], strides: [%34, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
          %alloc_31 = memref.alloc() : memref<64x128xf16>
          %178 = arith.divsi %176, %34 : index
          %179 = arith.subi %35, %178 : index
          %180 = arith.maxsi %179, %c0 : index
          %181 = arith.minsi %180, %c64 : index
          %182 = arith.remsi %176, %34 : index
          %183 = arith.subi %36, %182 : index
          %184 = arith.maxsi %183, %c0 : index
          %185 = arith.minsi %184, %c128 : index
          %186 = arith.cmpi slt, %181, %c64 : index
          %187 = arith.cmpi slt, %185, %c128 : index
          %188 = arith.ori %186, %187 : i1
          scf.if %188 {
            linalg.fill ins(%cst : f16) outs(%alloc_31 : memref<64x128xf16>)
          } {hivm.unlikely_condition}
          %subview_32 = memref.subview %reinterpret_cast_30[0, 0] [%181, %185] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %subview_33 = memref.subview %alloc_31[0, 0] [%181, %185] [1, 1] : memref<64x128xf16> to memref<?x?xf16, strided<[128, 1]>>
          memref.copy %subview_32, %subview_33 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
          %189 = bufferization.to_tensor %alloc_31 restrict writable : memref<64x128xf16>
          %190 = arith.muli %175, %42 : index
          %191 = arith.addi %190, %41 : index
          %reinterpret_cast_34 = memref.reinterpret_cast %arg4 to offset: [%191], sizes: [64, 128], strides: [%42, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
          %alloc_35 = memref.alloc() : memref<64x128xf16>
          %192 = arith.divsi %190, %42 : index
          %193 = arith.subi %35, %192 : index
          %194 = arith.maxsi %193, %c0 : index
          %195 = arith.minsi %194, %c64 : index
          %196 = arith.remsi %190, %42 : index
          %197 = arith.subi %36, %196 : index
          %198 = arith.maxsi %197, %c0 : index
          %199 = arith.minsi %198, %c128 : index
          %200 = arith.cmpi slt, %195, %c64 : index
          %201 = arith.cmpi slt, %199, %c128 : index
          %202 = arith.ori %200, %201 : i1
          scf.if %202 {
            linalg.fill ins(%cst : f16) outs(%alloc_35 : memref<64x128xf16>)
          } {hivm.unlikely_condition}
          %subview_36 = memref.subview %reinterpret_cast_34[0, 0] [%195, %199] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %subview_37 = memref.subview %alloc_35[0, 0] [%195, %199] [1, 1] : memref<64x128xf16> to memref<?x?xf16, strided<[128, 1]>>
          memref.copy %subview_36, %subview_37 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
          %203 = bufferization.to_tensor %alloc_35 restrict writable : memref<64x128xf16>
          %204 = tensor.empty() : tensor<128x64xf16>
          %transposed = linalg.transpose ins(%203 : tensor<64x128xf16>) outs(%204 : tensor<128x64xf16>) permutation = [1, 0] 
          %205 = linalg.fill ins(%174 : i32) outs(%46 : tensor<64xi32>) -> tensor<64xi32>
          %206 = arith.addi %205, %47 : tensor<64xi32>
          %207 = arith.cmpi sge, %156, %206 : tensor<64xi32>
          %expanded = tensor.expand_shape %207 [[0, 1]] output_shape [1, 64] : tensor<64xi1> into tensor<1x64xi1>
          %208 = arith.select %expanded, %6, %7 : tensor<1x64xi1>, tensor<1x64xf32>
          %209 = arith.addf %208, %6 : tensor<1x64xf32>
          %collapsed_38 = tensor.collapse_shape %209 [[0, 1]] : tensor<1x64xf32> into tensor<64xf32>
          %broadcasted_39 = linalg.broadcast ins(%collapsed_38 : tensor<64xf32>) outs(%8 : tensor<16x64xf32>) dimensions = [0] 
          %transposed_40 = linalg.transpose ins(%189 : tensor<64x128xf16>) outs(%204 : tensor<128x64xf16>) permutation = [1, 0] 
          %210 = linalg.matmul {input_precison = "ieee"} ins(%107, %transposed_40 : tensor<16x128xf16>, tensor<128x64xf16>) outs(%9 : tensor<16x64xf32>) -> tensor<16x64xf32>
          %211 = arith.mulf %210, %48 : tensor<16x64xf32>
          %212 = arith.addf %broadcasted_39, %211 : tensor<16x64xf32>
          %213 = arith.subf %212, %broadcasted : tensor<16x64xf32>
          %214 = math.exp2 %213 : tensor<16x64xf32>
          %215 = linalg.matmul {input_precison = "ieee"} ins(%123, %transposed : tensor<16x128xf16>, tensor<128x64xf16>) outs(%9 : tensor<16x64xf32>) -> tensor<16x64xf32>
          %216 = arith.mulf %49, %214 : tensor<16x64xf32>
          %217 = arith.subf %215, %broadcasted_27 : tensor<16x64xf32>
          %218 = arith.mulf %216, %217 : tensor<16x64xf32>
          %219 = arith.truncf %218 : tensor<16x64xf32> to tensor<16x64xf16>
          %220 = linalg.matmul {input_precison = "ieee"} ins(%219, %189 : tensor<16x64xf16>, tensor<64x128xf16>) outs(%arg40 : tensor<16x128xf32>) -> tensor<16x128xf32>
          scf.yield %220 : tensor<16x128xf32>
        }
        %158 = arith.truncf %157 : tensor<16x128xf32> to tensor<16x128xf16>
        %159 = arith.divsi %c0, %79 : index
        %160 = arith.subi %74, %159 : index
        %161 = arith.maxsi %160, %c0 : index
        %162 = arith.minsi %161, %c16 : index
        %163 = arith.remsi %c0, %79 : index
        %164 = arith.subi %36, %163 : index
        %165 = arith.maxsi %164, %c0 : index
        %166 = arith.minsi %165, %c128 : index
        %167 = arith.minsi %162, %c0 : index
        %168 = arith.subi %162, %167 : index
        %169 = arith.minsi %166, %c0 : index
        %170 = arith.subi %166, %169 : index
        %extracted_slice = tensor.extract_slice %158[%167, %169] [%168, %170] [1, 1] : tensor<16x128xf16> to tensor<?x?xf16>
        %subview_28 = memref.subview %reinterpret_cast_10[0, 0] [%168, %170] [1, 1] : memref<16x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_28 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}