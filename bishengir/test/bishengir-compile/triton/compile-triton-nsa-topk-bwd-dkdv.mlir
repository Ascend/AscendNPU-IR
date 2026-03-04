// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1
#map = affine_map<(d0) -> (d0)>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @backward_dkdv(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg13: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg14: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg15: i32, %arg16: i32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32, %arg19: f32, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32 {tt.divisibility = 16 : i32}, %arg38: i32 {tt.divisibility = 16 : i32}, %arg39: i32, %arg40: i32, %arg41: i32, %arg42: i32, %arg43: i32, %arg44: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c32_i32 = arith.constant 32 : i32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %cst_2 = arith.constant 1.44269502 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<32x64xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %5 = tensor.empty() : tensor<64x128xf32>
    %6 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %7 = tensor.empty() : tensor<32x128xf16>
    %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<32x128xf16>) -> tensor<32x128xf16>
    %9 = tensor.empty() : tensor<32x1xf32>
    %10 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %inserted = tensor.insert %arg19 into %0[%c0] : tensor<1xf32>
    %11 = arith.mulf %inserted, %1 : tensor<1xf32>
    %extracted = tensor.extract %11[%c0] : tensor<1xf32>
    %12 = arith.divsi %arg43, %arg16 : i32
    %13 = arith.remsi %arg43, %arg16 : i32
    %14 = arith.addi %arg44, %arg38 : i32
    %15 = arith.index_cast %arg42 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg11 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %16 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg12 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %17 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %18 = arith.addi %15, %c1 : index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg12 to offset: [%18], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %19 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %20 = arith.subi %19, %17 : i32
    %21 = arith.muli %14, %c64_i32 : i32
    %22 = arith.cmpi sge, %21, %20 : i32
    scf.if %22 {
    } else {
      %reinterpret_cast_5 = memref.reinterpret_cast %arg13 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %23 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %24 = arith.muli %12, %arg27 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = arith.addi %23, %14 : i32
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.addi %25, %27 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg14 to offset: [%28], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %29 = memref.load %reinterpret_cast_6[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %30 = arith.addi %26, %c1_i32 : i32
      %31 = arith.index_cast %30 : i32 to index
      %32 = arith.addi %25, %31 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg14 to offset: [%32], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %33 = memref.load %reinterpret_cast_7[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %34 = arith.subi %33, %29 : i32
      %35 = arith.muli %12, %arg26 : i32
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.index_cast %29 : i32 to index
      %38 = arith.addi %36, %37 : index
      %39 = arith.muli %17, %arg22 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = arith.muli %12, %arg23 : i32
      %42 = arith.index_cast %41 : i32 to index
      %43 = arith.addi %40, %42 : index
      %44 = arith.index_cast %21 : i32 to index
      %45 = arith.index_cast %arg22 : i32 to index
      %46 = arith.muli %44, %45 : index
      %47 = arith.addi %46, %43 : index
      %48 = arith.index_cast %20 : i32 to index
      %49 = arith.index_cast %arg17 : i32 to index
      %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%47], sizes: [64, 128], strides: [%45, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
      %50 = arith.muli %17, %arg33 : i32
      %51 = arith.index_cast %50 : i32 to index
      %52 = arith.muli %12, %arg34 : i32
      %53 = arith.index_cast %52 : i32 to index
      %54 = arith.addi %51, %53 : index
      %55 = arith.muli %13, %arg32 : i32
      %56 = arith.index_cast %55 : i32 to index
      %57 = arith.addi %54, %56 : index
      %58 = arith.index_cast %arg33 : i32 to index
      %59 = arith.muli %44, %58 : index
      %60 = arith.addi %59, %57 : index
      %reinterpret_cast_9 = memref.reinterpret_cast %arg9 to offset: [%60], sizes: [64, 128], strides: [%58, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
      %61 = arith.muli %17, %arg24 : i32
      %62 = arith.index_cast %61 : i32 to index
      %63 = arith.muli %12, %arg25 : i32
      %64 = arith.index_cast %63 : i32 to index
      %65 = arith.addi %62, %64 : index
      %66 = arith.index_cast %arg24 : i32 to index
      %67 = arith.muli %44, %66 : index
      %68 = arith.addi %67, %65 : index
      %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%68], sizes: [64, 128], strides: [%66, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
      %69 = arith.muli %17, %arg36 : i32
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.muli %12, %arg37 : i32
      %72 = arith.index_cast %71 : i32 to index
      %73 = arith.addi %70, %72 : index
      %74 = arith.muli %13, %arg35 : i32
      %75 = arith.index_cast %74 : i32 to index
      %76 = arith.addi %73, %75 : index
      %77 = arith.index_cast %arg36 : i32 to index
      %78 = arith.muli %44, %77 : index
      %79 = arith.addi %78, %76 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg10 to offset: [%79], sizes: [64, 128], strides: [%77, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
      %80 = tensor.empty() : tensor<64xi32>
      %81 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%80 : tensor<64xi32>) {
      ^bb0(%out: i32):
        %148 = linalg.index 0 : index
        %149 = arith.index_cast %148 : index to i32
        linalg.yield %149 : i32
      } -> tensor<64xi32>
      %82 = linalg.fill ins(%21 : i32) outs(%80 : tensor<64xi32>) -> tensor<64xi32>
      %83 = arith.addi %81, %82 : tensor<64xi32>
      %alloc = memref.alloc() : memref<64x128xf16>
      %84 = arith.divsi %46, %45 : index
      %85 = arith.subi %48, %84 : index
      %86 = arith.maxsi %85, %c0 : index
      %87 = arith.minsi %86, %c64 : index
      %88 = arith.remsi %46, %45 : index
      %89 = arith.subi %49, %88 : index
      %90 = arith.maxsi %89, %c0 : index
      %91 = arith.minsi %90, %c128 : index
      %92 = arith.cmpi slt, %87, %c64 : index
      %93 = arith.cmpi slt, %91, %c128 : index
      %94 = arith.ori %92, %93 : i1
      scf.if %94 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<64x128xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_8[0, 0] [%87, %91] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_12 = memref.subview %alloc[0, 0] [%87, %91] [1, 1] : memref<64x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview, %subview_12 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %95 = bufferization.to_tensor %alloc restrict writable : memref<64x128xf16>
      %alloc_13 = memref.alloc() : memref<64x128xf16>
      %96 = arith.divsi %67, %66 : index
      %97 = arith.subi %48, %96 : index
      %98 = arith.maxsi %97, %c0 : index
      %99 = arith.minsi %98, %c64 : index
      %100 = arith.remsi %67, %66 : index
      %101 = arith.subi %49, %100 : index
      %102 = arith.maxsi %101, %c0 : index
      %103 = arith.minsi %102, %c128 : index
      %104 = arith.cmpi slt, %99, %c64 : index
      %105 = arith.cmpi slt, %103, %c128 : index
      %106 = arith.ori %104, %105 : i1
      scf.if %106 {
        linalg.fill ins(%cst : f16) outs(%alloc_13 : memref<64x128xf16>)
      } {hivm.unlikely_condition}
      %subview_14 = memref.subview %reinterpret_cast_10[0, 0] [%99, %103] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_15 = memref.subview %alloc_13[0, 0] [%99, %103] [1, 1] : memref<64x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview_14, %subview_15 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %107 = bufferization.to_tensor %alloc_13 restrict writable : memref<64x128xf16>
      %108 = arith.muli %16, %arg20 : i32
      %109 = arith.extsi %108 : i32 to i64
      %110 = arith.muli %arg43, %arg21 : i32
      %111 = arith.extsi %110 : i32 to i64
      %112 = arith.addi %109, %111 : i64
      %113 = arith.muli %16, %arg30 : i32
      %114 = arith.extsi %113 : i32 to i64
      %115 = arith.muli %arg43, %arg31 : i32
      %116 = arith.extsi %115 : i32 to i64
      %117 = arith.addi %114, %116 : i64
      %118 = arith.extsi %16 : i32 to i64
      %119 = arith.muli %arg43, %arg29 : i32
      %120 = arith.extsi %119 : i32 to i64
      %121 = arith.addi %118, %120 : i64
      %122 = arith.muli %arg43, %arg28 : i32
      %123 = arith.extsi %122 : i32 to i64
      %124 = arith.addi %118, %123 : i64
      %125 = tensor.empty() : tensor<32x64xi32>
      %broadcasted = linalg.broadcast ins(%83 : tensor<64xi32>) outs(%125 : tensor<32x64xi32>) dimensions = [0]
      %126 = tensor.empty() : tensor<128x64xf16>
      %transposed = linalg.transpose ins(%95 : tensor<64x128xf16>) outs(%126 : tensor<128x64xf16>) permutation = [1, 0]
      %127 = linalg.fill ins(%extracted : f32) outs(%2 : tensor<32x64xf32>) -> tensor<32x64xf32>
      %transposed_16 = linalg.transpose ins(%107 : tensor<64x128xf16>) outs(%126 : tensor<128x64xf16>) permutation = [1, 0]
      %128 = linalg.fill ins(%arg19 : f32) outs(%2 : tensor<32x64xf32>) -> tensor<32x64xf32>
      %129:2 = scf.for %arg45 = %c0_i32 to %34 step %c32_i32 iter_args(%arg46 = %6, %arg47 = %6) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
        %148 = arith.subi %34, %arg45 : i32
        %149 = arith.index_cast %arg45 : i32 to index
        %150 = arith.addi %38, %149 : index
        %reinterpret_cast_20 = memref.reinterpret_cast %arg5 to offset: [%150], sizes: [32], strides: [1] : memref<?xi32> to memref<32xi32, strided<[1], offset: ?>>
        %alloc_21 = memref.alloc() : memref<32xi32>
        %151 = arith.index_cast %148 : i32 to index
        %152 = arith.maxsi %151, %c0 : index
        %153 = arith.minsi %152, %c32 : index
        %154 = arith.cmpi slt, %153, %c32 : index
        scf.if %154 {
          linalg.fill ins(%c0_i32 : i32) outs(%alloc_21 : memref<32xi32>)
        } {hivm.unlikely_condition}
        %subview_22 = memref.subview %reinterpret_cast_20[0] [%153] [1] : memref<32xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
        %subview_23 = memref.subview %alloc_21[0] [%153] [1] : memref<32xi32> to memref<?xi32, strided<[1]>>
        memref.copy %subview_22, %subview_23 : memref<?xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1]>>
        %155 = bufferization.to_tensor %alloc_21 restrict writable : memref<32xi32>
        %alloc_24 = memref.alloc() : memref<32xi32>
        scf.if %154 {
          linalg.fill ins(%c0_i32 : i32) outs(%alloc_24 : memref<32xi32>)
        } {hivm.unlikely_condition}
        %subview_25 = memref.subview %alloc_24[0] [%153] [1] : memref<32xi32> to memref<?xi32, strided<[1]>>
        memref.copy %subview_22, %subview_25 : memref<?xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1]>>
        %156 = bufferization.to_tensor %alloc_24 restrict writable : memref<32xi32>
        %157 = arith.minsi %153, %c32 : index
        %alloc_26 = memref.alloc() : memref<32x128xf16>
        scf.for %arg48 = %c0 to %157 step %c1 {
          %extracted_41 = tensor.extract %155[%arg48] {DiscreteMemAccess} : tensor<32xi32>
          %183 = arith.muli %extracted_41, %arg20 : i32
          %184 = arith.index_cast %112 : i64 to index
          %185 = arith.index_cast %183 : i32 to index
          %186 = arith.addi %184, %185 : index
          %reinterpret_cast_42 = memref.reinterpret_cast %arg2 to offset: [%186], sizes: [1, 128], strides: [128, 1] : memref<?xf16> to memref<1x128xf16, strided<[128, 1], offset: ?>>
          %subview_43 = memref.subview %alloc_26[%arg48, 0] [1, 128] [1, 1] : memref<32x128xf16> to memref<1x128xf16, strided<[128, 1], offset: ?>>
          memref.copy %reinterpret_cast_42, %subview_43 : memref<1x128xf16, strided<[128, 1], offset: ?>> to memref<1x128xf16, strided<[128, 1], offset: ?>>
        } {ExtractedLoadOrStore, hivm.parallel_loop}
        %158 = bufferization.to_tensor %alloc_26 restrict writable : memref<32x128xf16>
        %159 = arith.index_cast %arg17 : i32 to index
        %160 = arith.maxsi %159, %c0 : index
        %161 = arith.minsi %160, %c128 : index
        %162 = arith.minsi %161, %c128 : index
        %extracted_slice_27 = tensor.extract_slice %158[0, 0] [%157, %162] [1, 1] : tensor<32x128xf16> to tensor<?x?xf16>
        %inserted_slice = tensor.insert_slice %extracted_slice_27 into %8[0, 0] [%157, %162] [1, 1] : tensor<?x?xf16> into tensor<32x128xf16>
        %alloc_28 = memref.alloc() : memref<32x128xf16>
        scf.for %arg48 = %c0 to %157 step %c1 {
          %extracted_41 = tensor.extract %155[%arg48] {DiscreteMemAccess} : tensor<32xi32>
          %183 = arith.muli %extracted_41, %arg30 : i32
          %184 = arith.index_cast %117 : i64 to index
          %185 = arith.index_cast %183 : i32 to index
          %186 = arith.addi %184, %185 : index
          %reinterpret_cast_42 = memref.reinterpret_cast %arg8 to offset: [%186], sizes: [1, 128], strides: [128, 1] : memref<?xf16> to memref<1x128xf16, strided<[128, 1], offset: ?>>
          %subview_43 = memref.subview %alloc_28[%arg48, 0] [1, 128] [1, 1] : memref<32x128xf16> to memref<1x128xf16, strided<[128, 1], offset: ?>>
          memref.copy %reinterpret_cast_42, %subview_43 : memref<1x128xf16, strided<[128, 1], offset: ?>> to memref<1x128xf16, strided<[128, 1], offset: ?>>
        } {ExtractedLoadOrStore, hivm.parallel_loop}
        %163 = bufferization.to_tensor %alloc_28 restrict writable : memref<32x128xf16>
        %extracted_slice_29 = tensor.extract_slice %163[0, 0] [%157, %162] [1, 1] : tensor<32x128xf16> to tensor<?x?xf16>
        %inserted_slice_30 = tensor.insert_slice %extracted_slice_29 into %8[0, 0] [%157, %162] [1, 1] : tensor<?x?xf16> into tensor<32x128xf16>
        %164 = scf.for %arg48 = %c0 to %153 step %c1 iter_args(%arg49 = %9) -> (tensor<32x1xf32>) {
          %extracted_41 = tensor.extract %155[%arg48] {DiscreteMemAccess} : tensor<32xi32>
          %183 = arith.extsi %extracted_41 : i32 to i64
          %184 = arith.addi %124, %183 : i64
          %185 = arith.index_cast %184 : i64 to index
          %reinterpret_cast_42 = memref.reinterpret_cast %arg6 to offset: [%185], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
          %186 = memref.load %reinterpret_cast_42[%c0] : memref<1xf32, strided<[1], offset: ?>>
          %inserted_43 = tensor.insert %186 into %arg49[%arg48, %c0] : tensor<32x1xf32>
          scf.yield %inserted_43 : tensor<32x1xf32>
        } {ExtractedLoadOrStore}
        %extracted_slice_31 = tensor.extract_slice %164[0, 0] [%153, 1] [1, 1] : tensor<32x1xf32> to tensor<?x1xf32>
        %inserted_slice_32 = tensor.insert_slice %extracted_slice_31 into %10[0, 0] [%153, 1] [1, 1] : tensor<?x1xf32> into tensor<32x1xf32>
        %165 = scf.for %arg48 = %c0 to %153 step %c1 iter_args(%arg49 = %9) -> (tensor<32x1xf32>) {
          %extracted_41 = tensor.extract %155[%arg48] {DiscreteMemAccess} : tensor<32xi32>
          %183 = arith.extsi %extracted_41 : i32 to i64
          %184 = arith.addi %121, %183 : i64
          %185 = arith.index_cast %184 : i64 to index
          %reinterpret_cast_42 = memref.reinterpret_cast %arg7 to offset: [%185], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
          %186 = memref.load %reinterpret_cast_42[%c0] : memref<1xf32, strided<[1], offset: ?>>
          %inserted_43 = tensor.insert %186 into %arg49[%arg48, %c0] : tensor<32x1xf32>
          scf.yield %inserted_43 : tensor<32x1xf32>
        } {ExtractedLoadOrStore}
        %extracted_slice_33 = tensor.extract_slice %165[0, 0] [%153, 1] [1, 1] : tensor<32x1xf32> to tensor<?x1xf32>
        %inserted_slice_34 = tensor.insert_slice %extracted_slice_33 into %10[0, 0] [%153, 1] [1, 1] : tensor<?x1xf32> into tensor<32x1xf32>
        %broadcasted_35 = linalg.broadcast ins(%156 : tensor<32xi32>) outs(%125 : tensor<32x64xi32>) dimensions = [1]
        %166 = arith.cmpi sge, %broadcasted_35, %broadcasted : tensor<32x64xi32>
        %167 = arith.select %166, %3, %4 : tensor<32x64xi1>, tensor<32x64xf32>
        %168 = arith.addf %167, %3 : tensor<32x64xf32>
        %169 = linalg.matmul {input_precison = "ieee"} ins(%inserted_slice, %transposed : tensor<32x128xf16>, tensor<128x64xf16>) outs(%3 : tensor<32x64xf32>) -> tensor<32x64xf32>
        %170 = arith.mulf %169, %127 : tensor<32x64xf32>
        %171 = arith.addf %168, %170 : tensor<32x64xf32>
        %collapsed = tensor.collapse_shape %inserted_slice_32 [[0, 1]] : tensor<32x1xf32> into tensor<32xf32>
        %broadcasted_36 = linalg.broadcast ins(%collapsed : tensor<32xf32>) outs(%2 : tensor<32x64xf32>) dimensions = [1]
        %172 = arith.subf %171, %broadcasted_36 : tensor<32x64xf32>
        %173 = math.exp2 %172 : tensor<32x64xf32>
        %174 = linalg.matmul {input_precison = "ieee"} ins(%inserted_slice_30, %transposed_16 : tensor<32x128xf16>, tensor<128x64xf16>) outs(%3 : tensor<32x64xf32>) -> tensor<32x64xf32>
        %175 = arith.mulf %128, %173 : tensor<32x64xf32>
        %collapsed_37 = tensor.collapse_shape %inserted_slice_34 [[0, 1]] : tensor<32x1xf32> into tensor<32xf32>
        %broadcasted_38 = linalg.broadcast ins(%collapsed_37 : tensor<32xf32>) outs(%2 : tensor<32x64xf32>) dimensions = [1]
        %176 = arith.subf %174, %broadcasted_38 : tensor<32x64xf32>
        %177 = arith.mulf %175, %176 : tensor<32x64xf32>
        %178 = arith.truncf %173 : tensor<32x64xf32> to tensor<32x64xf16>
        %179 = arith.truncf %177 : tensor<32x64xf32> to tensor<32x64xf16>
        %180 = tensor.empty() : tensor<64x32xf16>
        %transposed_39 = linalg.transpose ins(%179 : tensor<32x64xf16>) outs(%180 : tensor<64x32xf16>) permutation = [1, 0]
        %181 = linalg.matmul {input_precison = "ieee"} ins(%transposed_39, %inserted_slice : tensor<64x32xf16>, tensor<32x128xf16>) outs(%arg46 : tensor<64x128xf32>) -> tensor<64x128xf32>
        %transposed_40 = linalg.transpose ins(%178 : tensor<32x64xf16>) outs(%180 : tensor<64x32xf16>) permutation = [1, 0]
        %182 = linalg.matmul {input_precison = "ieee"} ins(%transposed_40, %inserted_slice_30 : tensor<64x32xf16>, tensor<32x128xf16>) outs(%arg47 : tensor<64x128xf32>) -> tensor<64x128xf32>
        scf.yield %181, %182 : tensor<64x128xf32>, tensor<64x128xf32>
      }
      %130 = arith.truncf %129#0 : tensor<64x128xf32> to tensor<64x128xf16>
      %131 = arith.divsi %59, %58 : index
      %132 = arith.subi %48, %131 : index
      %133 = arith.maxsi %132, %c0 : index
      %134 = arith.minsi %133, %c64 : index
      %135 = arith.remsi %59, %58 : index
      %136 = arith.subi %49, %135 : index
      %137 = arith.maxsi %136, %c0 : index
      %138 = arith.minsi %137, %c128 : index
      %extracted_slice = tensor.extract_slice %130[0, 0] [%134, %138] [1, 1] : tensor<64x128xf16> to tensor<?x?xf16>
      %subview_17 = memref.subview %reinterpret_cast_9[0, 0] [%134, %138] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_17 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
      %139 = arith.truncf %129#1 : tensor<64x128xf32> to tensor<64x128xf16>
      %140 = arith.divsi %78, %77 : index
      %141 = arith.subi %48, %140 : index
      %142 = arith.maxsi %141, %c0 : index
      %143 = arith.minsi %142, %c64 : index
      %144 = arith.remsi %78, %77 : index
      %145 = arith.subi %49, %144 : index
      %146 = arith.maxsi %145, %c0 : index
      %147 = arith.minsi %146, %c128 : index
      %extracted_slice_18 = tensor.extract_slice %139[0, 0] [%143, %147] [1, 1] : tensor<64x128xf16> to tensor<?x?xf16>
      %subview_19 = memref.subview %reinterpret_cast_11[0, 0] [%143, %147] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_18 in writable %subview_19 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }
}