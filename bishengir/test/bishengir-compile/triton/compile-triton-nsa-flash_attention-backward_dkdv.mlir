// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @backward_dkdv(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: f32, %arg17: i1 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32, %arg25: i32, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32, %arg35: i32, %arg36: i32, %arg37: i32, %arg38: i32, %arg39: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 1.44269502 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<32x128xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %4 = tensor.empty() : tensor<32x32xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %inserted = tensor.insert %arg16 into %0[%c0] : tensor<1xf32>
    %6 = arith.mulf %inserted, %1 : tensor<1xf32>
    %extracted = tensor.extract %6[%c0] : tensor<1xf32>
    %7:2 = scf.if %arg17 -> (i32, i32) {
      %18 = arith.remsi %arg38, %arg13 : i32
      %19 = arith.divsi %arg38, %arg13 : i32
      scf.yield %18, %19 : i32, i32
    } else {
      %18 = arith.divsi %arg38, %arg13 : i32
      %19 = arith.remsi %arg38, %arg13 : i32
      scf.yield %18, %19 : i32, i32
    }
    %8 = arith.index_cast %arg37 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg10 to offset: [%8], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %9 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %10 = arith.addi %8, %c1 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg10 to offset: [%10], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %11 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %12 = arith.subi %11, %9 : i32
    %reinterpret_cast_3 = memref.reinterpret_cast %arg11 to offset: [%8], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %13 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg11 to offset: [%10], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %14 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %15 = arith.subi %14, %13 : i32
    %16 = arith.muli %arg39, %c32_i32 : i32
    %17 = arith.cmpi sge, %16, %15 : i32
    scf.if %17 {
    } else {
      %18 = arith.muli %13, %arg20 : i32
      %19 = arith.index_cast %18 : i32 to index
      %20 = arith.muli %7#0, %arg21 : i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.addi %19, %21 : index
      %23 = arith.index_cast %16 : i32 to index
      %24 = arith.index_cast %arg20 : i32 to index
      %25 = arith.muli %23, %24 : index
      %26 = arith.addi %25, %22 : index
      %27 = arith.index_cast %15 : i32 to index
      %28 = arith.index_cast %arg14 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [32, 128], strides: [%24, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %29 = arith.muli %13, %arg29 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.muli %7#0, %arg30 : i32
      %32 = arith.index_cast %31 : i32 to index
      %33 = arith.addi %30, %32 : index
      %34 = arith.muli %7#1, %arg28 : i32
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.addi %33, %35 : index
      %37 = arith.index_cast %arg29 : i32 to index
      %38 = arith.muli %23, %37 : index
      %39 = arith.addi %36, %38 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg8 to offset: [%39], sizes: [32, 128], strides: [%37, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %40 = arith.muli %13, %arg22 : i32
      %41 = arith.index_cast %40 : i32 to index
      %42 = arith.muli %7#0, %arg23 : i32
      %43 = arith.index_cast %42 : i32 to index
      %44 = arith.addi %41, %43 : index
      %45 = arith.index_cast %arg22 : i32 to index
      %46 = arith.muli %23, %45 : index
      %47 = arith.addi %46, %44 : index
      %48 = arith.index_cast %arg15 : i32 to index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [32, 128], strides: [%45, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %49 = arith.muli %13, %arg32 : i32
      %50 = arith.index_cast %49 : i32 to index
      %51 = arith.muli %7#0, %arg33 : i32
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.addi %50, %52 : index
      %54 = arith.muli %7#1, %arg31 : i32
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.addi %53, %55 : index
      %57 = arith.index_cast %arg32 : i32 to index
      %58 = arith.muli %23, %57 : index
      %59 = arith.addi %56, %58 : index
      %reinterpret_cast_8 = memref.reinterpret_cast %arg9 to offset: [%59], sizes: [32, 128], strides: [%57, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %alloc = memref.alloc() : memref<32x128xf16>
      %60 = arith.divsi %25, %24 : index
      %61 = arith.subi %27, %60 : index
      %62 = arith.maxsi %61, %c0 : index
      %63 = arith.minsi %62, %c32 : index
      %64 = arith.remsi %25, %24 : index
      %65 = arith.subi %28, %64 : index
      %66 = arith.maxsi %65, %c0 : index
      %67 = arith.minsi %66, %c128 : index
      %68 = arith.cmpi slt, %63, %c32 : index
      %69 = arith.cmpi slt, %67, %c128 : index
      %70 = arith.ori %68, %69 : i1
      scf.if %70 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<32x128xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_5[0, 0] [%63, %67] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_9 = memref.subview %alloc[0, 0] [%63, %67] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview, %subview_9 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %71 = bufferization.to_tensor %alloc restrict writable : memref<32x128xf16>
      %alloc_10 = memref.alloc() : memref<32x128xf16>
      %72 = arith.divsi %46, %45 : index
      %73 = arith.subi %27, %72 : index
      %74 = arith.maxsi %73, %c0 : index
      %75 = arith.minsi %74, %c32 : index
      %76 = arith.remsi %46, %45 : index
      %77 = arith.subi %48, %76 : index
      %78 = arith.maxsi %77, %c0 : index
      %79 = arith.minsi %78, %c128 : index
      %80 = arith.cmpi slt, %75, %c32 : index
      %81 = arith.cmpi slt, %79, %c128 : index
      %82 = arith.ori %80, %81 : i1
      scf.if %82 {
        linalg.fill ins(%cst : f16) outs(%alloc_10 : memref<32x128xf16>)
      } {hivm.unlikely_condition}
      %subview_11 = memref.subview %reinterpret_cast_7[0, 0] [%75, %79] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_12 = memref.subview %alloc_10[0, 0] [%75, %79] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview_11, %subview_12 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %83 = bufferization.to_tensor %alloc_10 restrict writable : memref<32x128xf16>
      %84 = arith.muli %9, %arg18 : i32
      %85 = arith.index_cast %84 : i32 to index
      %86 = arith.muli %arg38, %arg19 : i32
      %87 = arith.index_cast %86 : i32 to index
      %88 = arith.addi %85, %87 : index
      %89 = arith.muli %9, %arg26 : i32
      %90 = arith.index_cast %89 : i32 to index
      %91 = arith.muli %arg38, %arg27 : i32
      %92 = arith.index_cast %91 : i32 to index
      %93 = arith.addi %90, %92 : index
      %94 = arith.index_cast %9 : i32 to index
      %95 = arith.muli %arg38, %arg25 : i32
      %96 = arith.index_cast %95 : i32 to index
      %97 = arith.addi %94, %96 : index
      %98 = arith.muli %arg38, %arg24 : i32
      %99 = arith.index_cast %98 : i32 to index
      %100 = arith.addi %94, %99 : index
      %101 = tensor.empty() : tensor<128x32xf16>
      %transposed = linalg.transpose ins(%71 : tensor<32x128xf16>) outs(%101 : tensor<128x32xf16>) permutation = [1, 0] 
      %102 = linalg.fill ins(%extracted : f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %transposed_13 = linalg.transpose ins(%83 : tensor<32x128xf16>) outs(%101 : tensor<128x32xf16>) permutation = [1, 0] 
      %103 = linalg.fill ins(%arg16 : f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %104:6 = scf.for %arg40 = %c0_i32 to %12 step %c32_i32 iter_args(%arg41 = %3, %arg42 = %3, %arg43 = %c0_i32, %arg44 = %c0_i32, %arg45 = %c0_i32, %arg46 = %c0_i32) -> (tensor<32x128xf32>, tensor<32x128xf32>, i32, i32, i32, i32)  : i32 {
        %121 = arith.index_cast %arg46 : i32 to index
        %122 = arith.addi %121, %97 : index
        %123 = arith.index_cast %12 : i32 to index
        %reinterpret_cast_17 = memref.reinterpret_cast %arg6 to offset: [%122], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
        %124 = arith.index_cast %arg45 : i32 to index
        %125 = arith.addi %124, %100 : index
        %reinterpret_cast_18 = memref.reinterpret_cast %arg5 to offset: [%125], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
        %126 = arith.index_cast %arg44 : i32 to index
        %127 = arith.index_cast %arg26 : i32 to index
        %128 = arith.muli %126, %127 : index
        %129 = arith.addi %128, %93 : index
        %reinterpret_cast_19 = memref.reinterpret_cast %arg7 to offset: [%129], sizes: [32, 128], strides: [%127, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
        %130 = arith.index_cast %arg43 : i32 to index
        %131 = arith.index_cast %arg18 : i32 to index
        %132 = arith.muli %130, %131 : index
        %133 = arith.addi %132, %88 : index
        %reinterpret_cast_20 = memref.reinterpret_cast %arg2 to offset: [%133], sizes: [32, 128], strides: [%131, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
        %alloc_21 = memref.alloc() : memref<32x128xf16>
        %134 = arith.divsi %132, %131 : index
        %135 = arith.subi %123, %134 : index
        %136 = arith.maxsi %135, %c0 : index
        %137 = arith.minsi %136, %c32 : index
        %138 = arith.remsi %132, %131 : index
        %139 = arith.subi %28, %138 : index
        %140 = arith.maxsi %139, %c0 : index
        %141 = arith.minsi %140, %c128 : index
        %142 = arith.cmpi slt, %137, %c32 : index
        %143 = arith.cmpi slt, %141, %c128 : index
        %144 = arith.ori %142, %143 : i1
        scf.if %144 {
          linalg.fill ins(%cst : f16) outs(%alloc_21 : memref<32x128xf16>)
        } {hivm.unlikely_condition}
        %subview_22 = memref.subview %reinterpret_cast_20[0, 0] [%137, %141] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_23 = memref.subview %alloc_21[0, 0] [%137, %141] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_22, %subview_23 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %145 = bufferization.to_tensor %alloc_21 restrict writable : memref<32x128xf16>
        %alloc_24 = memref.alloc() : memref<32x128xf16>
        %146 = arith.divsi %128, %127 : index
        %147 = arith.subi %123, %146 : index
        %148 = arith.maxsi %147, %c0 : index
        %149 = arith.minsi %148, %c32 : index
        %150 = arith.remsi %128, %127 : index
        %151 = arith.subi %48, %150 : index
        %152 = arith.maxsi %151, %c0 : index
        %153 = arith.minsi %152, %c128 : index
        %154 = arith.cmpi slt, %149, %c32 : index
        %155 = arith.cmpi slt, %153, %c128 : index
        %156 = arith.ori %154, %155 : i1
        scf.if %156 {
          linalg.fill ins(%cst : f16) outs(%alloc_24 : memref<32x128xf16>)
        } {hivm.unlikely_condition}
        %subview_25 = memref.subview %reinterpret_cast_19[0, 0] [%149, %153] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_26 = memref.subview %alloc_24[0, 0] [%149, %153] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_25, %subview_26 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %157 = bufferization.to_tensor %alloc_24 restrict writable : memref<32x128xf16>
        %alloc_27 = memref.alloc() : memref<32xf32>
        %158 = arith.subi %123, %124 : index
        %159 = arith.maxsi %158, %c0 : index
        %160 = arith.minsi %159, %c32 : index
        %161 = arith.cmpi slt, %160, %c32 : index
        scf.if %161 {
          linalg.fill ins(%cst_0 : f32) outs(%alloc_27 : memref<32xf32>)
        } {hivm.unlikely_condition}
        %subview_28 = memref.subview %reinterpret_cast_18[0] [%160] [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %subview_29 = memref.subview %alloc_27[0] [%160] [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
        memref.copy %subview_28, %subview_29 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
        %162 = bufferization.to_tensor %alloc_27 restrict writable : memref<32xf32>
        %alloc_30 = memref.alloc() : memref<32xf32>
        %163 = arith.subi %123, %121 : index
        %164 = arith.maxsi %163, %c0 : index
        %165 = arith.minsi %164, %c32 : index
        %166 = arith.cmpi slt, %165, %c32 : index
        scf.if %166 {
          linalg.fill ins(%cst_0 : f32) outs(%alloc_30 : memref<32xf32>)
        } {hivm.unlikely_condition}
        %subview_31 = memref.subview %reinterpret_cast_17[0] [%165] [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %subview_32 = memref.subview %alloc_30[0] [%165] [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
        memref.copy %subview_31, %subview_32 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
        %167 = bufferization.to_tensor %alloc_30 restrict writable : memref<32xf32>
        %168 = linalg.matmul {input_precison = "ieee"} ins(%145, %transposed : tensor<32x128xf16>, tensor<128x32xf16>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %169 = arith.mulf %168, %102 : tensor<32x32xf32>
        %170 = arith.addf %169, %5 : tensor<32x32xf32>
        %broadcasted = linalg.broadcast ins(%162 : tensor<32xf32>) outs(%4 : tensor<32x32xf32>) dimensions = [1] 
        %171 = arith.subf %170, %broadcasted : tensor<32x32xf32>
        %172 = math.exp2 %171 : tensor<32x32xf32>
        %173 = linalg.matmul {input_precison = "ieee"} ins(%157, %transposed_13 : tensor<32x128xf16>, tensor<128x32xf16>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %174 = arith.mulf %103, %172 : tensor<32x32xf32>
        %broadcasted_33 = linalg.broadcast ins(%167 : tensor<32xf32>) outs(%4 : tensor<32x32xf32>) dimensions = [1] 
        %175 = arith.subf %173, %broadcasted_33 : tensor<32x32xf32>
        %176 = arith.mulf %174, %175 : tensor<32x32xf32>
        %177 = arith.truncf %172 : tensor<32x32xf32> to tensor<32x32xf16>
        %178 = arith.truncf %176 : tensor<32x32xf32> to tensor<32x32xf16>
        %179 = tensor.empty() : tensor<32x32xf16>
        %transposed_34 = linalg.transpose ins(%178 : tensor<32x32xf16>) outs(%179 : tensor<32x32xf16>) permutation = [1, 0] 
        %180 = linalg.matmul {input_precison = "ieee"} ins(%transposed_34, %145 : tensor<32x32xf16>, tensor<32x128xf16>) outs(%arg41 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %transposed_35 = linalg.transpose ins(%177 : tensor<32x32xf16>) outs(%179 : tensor<32x32xf16>) permutation = [1, 0] 
        %181 = linalg.matmul {input_precison = "ieee"} ins(%transposed_35, %157 : tensor<32x32xf16>, tensor<32x128xf16>) outs(%arg42 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %182 = arith.addi %arg43, %c32_i32 : i32
        %183 = arith.addi %arg44, %c32_i32 : i32
        %184 = arith.addi %arg45, %c32_i32 : i32
        %185 = arith.addi %arg46, %c32_i32 : i32
        scf.yield %180, %181, %182, %183, %184, %185 : tensor<32x128xf32>, tensor<32x128xf32>, i32, i32, i32, i32
      }
      %105 = arith.truncf %104#0 : tensor<32x128xf32> to tensor<32x128xf16>
      %106 = arith.addi %23, %c32 : index
      %107 = arith.index_cast %15 : i32 to index
      %108 = arith.maxsi %23, %107 : index
      %109 = arith.minsi %106, %108 : index
      %110 = arith.subi %109, %23 : index
      %111 = arith.index_cast %arg14 : i32 to index
      %112 = arith.maxsi %111, %c0 : index
      %113 = arith.minsi %112, %c128 : index
      %114 = arith.minsi %110, %c32 : index
      %115 = arith.minsi %113, %c128 : index
      %extracted_slice = tensor.extract_slice %105[0, 0] [%114, %115] [1, 1] : tensor<32x128xf16> to tensor<?x?xf16>
      %subview_14 = memref.subview %reinterpret_cast_6[0, 0] [%114, %115] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_14 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
      %116 = arith.truncf %104#1 : tensor<32x128xf32> to tensor<32x128xf16>
      %117 = arith.index_cast %arg15 : i32 to index
      %118 = arith.maxsi %117, %c0 : index
      %119 = arith.minsi %118, %c128 : index
      %120 = arith.minsi %119, %c128 : index
      %extracted_slice_15 = tensor.extract_slice %116[0, 0] [%114, %120] [1, 1] : tensor<32x128xf16> to tensor<?x?xf16>
      %subview_16 = memref.subview %reinterpret_cast_8[0, 0] [%114, %120] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_15 in writable %subview_16 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }
}