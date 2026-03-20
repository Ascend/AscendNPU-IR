// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t
#map = affine_map<(d0) -> (d0)>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @backward_dkdv(%arg0: memref<?xi8> , %arg1: memref<?xi8> , %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} , %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} , %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} , %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} , %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} , %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} , %arg8: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32} , %arg9: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32} , %arg10: i32 {tt.divisibility = 16 : i32} , %arg11: i32 {tt.divisibility = 16 : i32} , %arg12: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} , %arg13: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} , %arg14: i32 , %arg15: i32 , %arg16: i32 {tt.divisibility = 16 : i32} , %arg17: f32 , %arg18: i32 {tt.divisibility = 16 : i32} , %arg19: i32 {tt.divisibility = 16 : i32} , %arg20: i32 {tt.divisibility = 16 : i32} , %arg21: i32 {tt.divisibility = 16 : i32} , %arg22: i32 {tt.divisibility = 16 : i32} , %arg23: i32 {tt.divisibility = 16 : i32} , %arg24: i32 , %arg25: i32 , %arg26: i32 {tt.divisibility = 16 : i32} , %arg27: i32 {tt.divisibility = 16 : i32} , %arg28: i32 {tt.divisibility = 16 : i32} , %arg29: i32 {tt.divisibility = 16 : i32} , %arg30: i32 {tt.divisibility = 16 : i32} , %arg31: i32 {tt.divisibility = 16 : i32} , %arg32: i32 {tt.divisibility = 16 : i32} , %arg33: i32 {tt.divisibility = 16 : i32} , %arg34: i32 , %arg35: i32 , %arg36: i32 , %arg37: i32 , %arg38: i32 , %arg39: i32 loc("/home/f00935178/0204tmp/triton-ops-master/nsa/xx/test_compressed_attention.py":212:0)) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
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
    %inserted = tensor.insert %arg17 into %0[%c0] : tensor<1xf32>
    %9 = arith.mulf %inserted, %1 : tensor<1xf32>
    %extracted = tensor.extract %9[%c0] : tensor<1xf32>
    %10 = arith.divsi %arg38, %arg15 : i32
    %11 = arith.remsi %arg38, %arg15 : i32
    %12 = arith.index_cast %arg37 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg12 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %13 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %14 = arith.addi %12, %c1 : index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg12 to offset: [%14], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %15 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %16 = arith.subi %15, %13 : i32
    %reinterpret_cast_4 = memref.reinterpret_cast %arg13 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %17 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg13 to offset: [%14], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %18 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %19 = arith.subi %18, %17 : i32
    %20 = arith.muli %arg39, %c32_i32 : i32
    %21 = arith.cmpi sge, %20, %19 : i32
    scf.if %21 {
    } else {
      %22 = arith.muli %17, %arg20 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = arith.muli %10, %arg21 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = arith.addi %23, %25 : index
      %27 = arith.index_cast %20 : i32 to index
      %28 = arith.index_cast %arg20 : i32 to index
      %29 = arith.muli %27, %28 : index
      %30 = arith.addi %29, %26 : index
      %31 = arith.index_cast %19 : i32 to index
      %32 = arith.index_cast %arg16 : i32 to index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%30], sizes: [32, 128], strides: [%28, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %33 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : tensor<32xi32>) {
      ^bb0(%out: i32) :
        %131 = linalg.index 0 : index
        %132 = arith.index_cast %131 : index to i32
        linalg.yield %132 : i32
      } -> tensor<32xi32>
      %34 = arith.muli %17, %arg29 : i32
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.muli %10, %arg30 : i32
      %37 = arith.index_cast %36 : i32 to index
      %38 = arith.addi %35, %37 : index
      %39 = arith.muli %11, %arg28 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = arith.addi %38, %40 : index
      %42 = arith.index_cast %arg29 : i32 to index
      %43 = arith.muli %27, %42 : index
      %44 = arith.addi %41, %43 : index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%44], sizes: [32, 128], strides: [%42, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %45 = arith.muli %17, %arg22 : i32
      %46 = arith.index_cast %45 : i32 to index
      %47 = arith.muli %10, %arg23 : i32
      %48 = arith.index_cast %47 : i32 to index
      %49 = arith.addi %46, %48 : index
      %50 = arith.index_cast %arg22 : i32 to index
      %51 = arith.muli %27, %50 : index
      %52 = arith.addi %51, %49 : index
      %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%52], sizes: [32, 128], strides: [%50, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %53 = arith.muli %17, %arg32 : i32
      %54 = arith.index_cast %53 : i32 to index
      %55 = arith.muli %10, %arg33 : i32
      %56 = arith.index_cast %55 : i32 to index
      %57 = arith.addi %54, %56 : index
      %58 = arith.muli %11, %arg31 : i32
      %59 = arith.index_cast %58 : i32 to index
      %60 = arith.addi %57, %59 : index
      %61 = arith.index_cast %arg32 : i32 to index
      %62 = arith.muli %27, %61 : index
      %63 = arith.addi %60, %62 : index
      %reinterpret_cast_9 = memref.reinterpret_cast %arg9 to offset: [%63], sizes: [32, 128], strides: [%61, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %64 = arith.muli %20, %arg11 : i32
      %65 = linalg.fill ins(%arg11 : i32) outs(%2 : tensor<32xi32>) -> tensor<32xi32>
      %66 = arith.muli %33, %65 : tensor<32xi32>
      %67 = linalg.fill ins(%64 : i32) outs(%2 : tensor<32xi32>) -> tensor<32xi32>
      %68 = arith.addi %67, %66 : tensor<32xi32>
      %69 = linalg.fill ins(%arg10 : i32) outs(%2 : tensor<32xi32>) -> tensor<32xi32>
      %70 = arith.addi %68, %69 : tensor<32xi32>
      %71 = arith.subi %70, %3 : tensor<32xi32>
      %alloc = memref.alloc() : memref<32x128xf16>
      %72 = arith.divsi %29, %28 : index
      %73 = arith.subi %31, %72 : index
      %74 = arith.maxsi %73, %c0 : index
      %75 = arith.minsi %74, %c32 : index
      %76 = arith.remsi %29, %28 : index
      %77 = arith.subi %32, %76 : index
      %78 = arith.maxsi %77, %c0 : index
      %79 = arith.minsi %78, %c128 : index
      %80 = arith.cmpi slt, %75, %c32 : index
      %81 = arith.cmpi slt, %79, %c128 : index
      %82 = arith.ori %80, %81 : i1
      scf.if %82 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<32x128xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_6[0, 0] [%75, %79] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_10 = memref.subview %alloc[0, 0] [%75, %79] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview, %subview_10 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %83 = bufferization.to_tensor %alloc restrict writable : memref<32x128xf16>
      %alloc_11 = memref.alloc() : memref<32x128xf16>
      %84 = arith.divsi %51, %50 : index
      %85 = arith.subi %31, %84 : index
      %86 = arith.maxsi %85, %c0 : index
      %87 = arith.minsi %86, %c32 : index
      %88 = arith.remsi %51, %50 : index
      %89 = arith.subi %32, %88 : index
      %90 = arith.maxsi %89, %c0 : index
      %91 = arith.minsi %90, %c128 : index
      %92 = arith.cmpi slt, %87, %c32 : index
      %93 = arith.cmpi slt, %91, %c128 : index
      %94 = arith.ori %92, %93 : i1
      scf.if %94 {
        linalg.fill ins(%cst : f16) outs(%alloc_11 : memref<32x128xf16>)
      } {hivm.unlikely_condition}
      %subview_12 = memref.subview %reinterpret_cast_8[0, 0] [%87, %91] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_13 = memref.subview %alloc_11[0, 0] [%87, %91] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview_12, %subview_13 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %95 = bufferization.to_tensor %alloc_11 restrict writable : memref<32x128xf16>
      %96 = arith.addi %64, %arg10 : i32
      %97 = arith.subi %96, %c1_i32 : i32
      %98 = arith.muli %13, %arg18 : i32
      %99 = arith.index_cast %98 : i32 to index
      %100 = arith.muli %arg38, %arg19 : i32
      %101 = arith.index_cast %100 : i32 to index
      %102 = arith.addi %99, %101 : index
      %103 = arith.muli %13, %arg26 : i32
      %104 = arith.index_cast %103 : i32 to index
      %105 = arith.muli %arg38, %arg27 : i32
      %106 = arith.index_cast %105 : i32 to index
      %107 = arith.addi %104, %106 : index
      %108 = arith.index_cast %13 : i32 to index
      %109 = arith.muli %arg38, %arg25 : i32
      %110 = arith.index_cast %109 : i32 to index
      %111 = arith.addi %108, %110 : index
      %112 = arith.muli %arg38, %arg24 : i32
      %113 = arith.index_cast %112 : i32 to index
      %114 = arith.addi %108, %113 : index
      %115 = tensor.empty() : tensor<32x32xi32>
      %broadcasted = linalg.broadcast ins(%71 : tensor<32xi32>) outs(%115 : tensor<32x32xi32>) dimensions = [1]
      %116 = linalg.fill ins(%extracted : f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %117 = linalg.fill ins(%arg17 : f32) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %118:6 = scf.for %arg40 = %97 to %16 step %c32_i32 iter_args(%arg41 = %8, %arg42 = %8, %arg43 = %97, %arg44 = %97, %arg45 = %97, %arg46 = %97) -> (tensor<32x128xf32>, tensor<32x128xf32>, i32, i32, i32, i32)  : i32 {
        %131 = arith.index_cast %arg46 : i32 to index
        %132 = arith.addi %131, %111 : index
        %133 = arith.index_cast %16 : i32 to index
        %reinterpret_cast_17 = memref.reinterpret_cast %arg6 to offset: [%132], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
        %134 = arith.index_cast %arg45 : i32 to index
        %135 = arith.addi %134, %114 : index
        %reinterpret_cast_18 = memref.reinterpret_cast %arg5 to offset: [%135], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
        %136 = arith.index_cast %arg44 : i32 to index
        %137 = arith.index_cast %arg26 : i32 to index
        %138 = arith.muli %136, %137 : index
        %139 = arith.addi %138, %107 : index
        %reinterpret_cast_19 = memref.reinterpret_cast %arg7 to offset: [%139], sizes: [32, 128], strides: [%137, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
        %140 = arith.index_cast %arg43 : i32 to index
        %141 = arith.index_cast %arg18 : i32 to index
        %142 = arith.muli %140, %141 : index
        %143 = arith.addi %142, %102 : index
        %reinterpret_cast_20 = memref.reinterpret_cast %arg2 to offset: [%143], sizes: [32, 128], strides: [%141, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
        %alloc_21 = memref.alloc() : memref<32x128xf16>
        %144 = arith.divsi %142, %141 : index
        %145 = arith.subi %133, %144 : index
        %146 = arith.maxsi %145, %c0 : index
        %147 = arith.minsi %146, %c32 : index
        %148 = arith.remsi %142, %141 : index
        %149 = arith.subi %32, %148 : index
        %150 = arith.maxsi %149, %c0 : index
        %151 = arith.minsi %150, %c128 : index
        %152 = arith.cmpi slt, %147, %c32 : index
        %153 = arith.cmpi slt, %151, %c128 : index
        %154 = arith.ori %152, %153 : i1
        scf.if %154 {
          linalg.fill ins(%cst : f16) outs(%alloc_21 : memref<32x128xf16>)
        } {hivm.unlikely_condition}
        %subview_22 = memref.subview %reinterpret_cast_20[0, 0] [%147, %151] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_23 = memref.subview %alloc_21[0, 0] [%147, %151] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_22, %subview_23 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %155 = bufferization.to_tensor %alloc_21 restrict writable : memref<32x128xf16>
        %156 = tensor.empty() : tensor<128x32xf16>
        %transposed = linalg.transpose ins(%155 : tensor<32x128xf16>) outs(%156 : tensor<128x32xf16>) permutation = [1, 0]
        %alloc_24 = memref.alloc() : memref<32x128xf16>
        %157 = arith.divsi %138, %137 : index
        %158 = arith.subi %133, %157 : index
        %159 = arith.maxsi %158, %c0 : index
        %160 = arith.minsi %159, %c32 : index
        %161 = arith.remsi %138, %137 : index
        %162 = arith.subi %32, %161 : index
        %163 = arith.maxsi %162, %c0 : index
        %164 = arith.minsi %163, %c128 : index
        %165 = arith.cmpi slt, %160, %c32 : index
        %166 = arith.cmpi slt, %164, %c128 : index
        %167 = arith.ori %165, %166 : i1
        scf.if %167 {
          linalg.fill ins(%cst : f16) outs(%alloc_24 : memref<32x128xf16>)
        } {hivm.unlikely_condition}
        %subview_25 = memref.subview %reinterpret_cast_19[0, 0] [%160, %164] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_26 = memref.subview %alloc_24[0, 0] [%160, %164] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_25, %subview_26 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %168 = bufferization.to_tensor %alloc_24 restrict writable : memref<32x128xf16>
        %transposed_27 = linalg.transpose ins(%168 : tensor<32x128xf16>) outs(%156 : tensor<128x32xf16>) permutation = [1, 0]
        %alloc_28 = memref.alloc() : memref<32xf32>
        %169 = arith.subi %133, %134 : index
        %170 = arith.maxsi %169, %c0 : index
        %171 = arith.minsi %170, %c32 : index
        %172 = arith.cmpi slt, %171, %c32 : index
        scf.if %172 {
          linalg.fill ins(%cst_1 : f32) outs(%alloc_28 : memref<32xf32>)
        } {hivm.unlikely_condition}
        %subview_29 = memref.subview %reinterpret_cast_18[0] [%171] [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %subview_30 = memref.subview %alloc_28[0] [%171] [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
        memref.copy %subview_29, %subview_30 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
        %173 = bufferization.to_tensor %alloc_28 restrict writable : memref<32xf32>
        %alloc_31 = memref.alloc() : memref<32xf32>
        %174 = arith.subi %133, %131 : index
        %175 = arith.maxsi %174, %c0 : index
        %176 = arith.minsi %175, %c32 : index
        %177 = arith.cmpi slt, %176, %c32 : index
        scf.if %177 {
          linalg.fill ins(%cst_1 : f32) outs(%alloc_31 : memref<32xf32>)
        } {hivm.unlikely_condition}
        %subview_32 = memref.subview %reinterpret_cast_17[0] [%176] [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %subview_33 = memref.subview %alloc_31[0] [%176] [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
        memref.copy %subview_32, %subview_33 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
        %178 = bufferization.to_tensor %alloc_31 restrict writable : memref<32xf32>
        %179 = linalg.fill ins(%arg40 : i32) outs(%2 : tensor<32xi32>) -> tensor<32xi32>
        %180 = arith.addi %33, %179 : tensor<32xi32>
        %broadcasted_34 = linalg.broadcast ins(%180 : tensor<32xi32>) outs(%115 : tensor<32x32xi32>) dimensions = [0]
        %181 = arith.cmpi sle, %broadcasted, %broadcasted_34 : tensor<32x32xi32>
        %182 = arith.select %181, %5, %6 : tensor<32x32xi1>, tensor<32x32xf32>
        %183 = linalg.matmul {input_precison = "ieee"} ins(%83, %transposed : tensor<32x128xf16>, tensor<128x32xf16>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %184 = arith.mulf %183, %116 : tensor<32x32xf32>
        %185 = arith.addf %182, %184 : tensor<32x32xf32>
        %broadcasted_35 = linalg.broadcast ins(%173 : tensor<32xf32>) outs(%4 : tensor<32x32xf32>) dimensions = [0]
        %186 = arith.subf %185, %broadcasted_35 : tensor<32x32xf32>
        %187 = math.exp2 %186 : tensor<32x32xf32>
        %188 = linalg.matmul {input_precison = "ieee"} ins(%95, %transposed_27 : tensor<32x128xf16>, tensor<128x32xf16>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %189 = arith.mulf %117, %187 : tensor<32x32xf32>
        %broadcasted_36 = linalg.broadcast ins(%178 : tensor<32xf32>) outs(%4 : tensor<32x32xf32>) dimensions = [0]
        %190 = arith.subf %188, %broadcasted_36 : tensor<32x32xf32>
        %191 = arith.mulf %189, %190 : tensor<32x32xf32>
        %192 = arith.truncf %187 : tensor<32x32xf32> to tensor<32x32xf16>
        %193 = arith.truncf %191 : tensor<32x32xf32> to tensor<32x32xf16>
        %194 = linalg.matmul {input_precison = "ieee"} ins(%193, %155 : tensor<32x32xf16>, tensor<32x128xf16>) outs(%arg41 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %195 = linalg.matmul {input_precison = "ieee"} ins(%192, %168 : tensor<32x32xf16>, tensor<32x128xf16>) outs(%arg42 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %196 = arith.addi %arg43, %c32_i32 : i32
        %197 = arith.addi %arg44, %c32_i32 : i32
        %198 = arith.addi %arg45, %c32_i32 : i32
        %199 = arith.addi %arg46, %c32_i32 : i32
        scf.yield %194, %195, %196, %197, %198, %199 : tensor<32x128xf32>, tensor<32x128xf32>, i32, i32, i32, i32
      }
      %119 = arith.truncf %118#0 : tensor<32x128xf32> to tensor<32x128xf16>
      %120 = arith.addi %27, %c32 : index
      %121 = arith.index_cast %19 : i32 to index
      %122 = arith.maxsi %27, %121 : index
      %123 = arith.minsi %120, %122 : index
      %124 = arith.subi %123, %27 : index
      %125 = arith.index_cast %arg16 : i32 to index
      %126 = arith.maxsi %125, %c0 : index
      %127 = arith.minsi %126, %c128 : index
      %128 = arith.minsi %124, %c32 : index
      %129 = arith.minsi %127, %c128 : index
      %extracted_slice = tensor.extract_slice %119[0, 0] [%128, %129] [1, 1] : tensor<32x128xf16> to tensor<?x?xf16>
      %subview_14 = memref.subview %reinterpret_cast_7[0, 0] [%128, %129] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_14 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
      %130 = arith.truncf %118#1 : tensor<32x128xf32> to tensor<32x128xf16>
      %extracted_slice_15 = tensor.extract_slice %130[0, 0] [%128, %129] [1, 1] : tensor<32x128xf16> to tensor<?x?xf16>
      %subview_16 = memref.subview %reinterpret_cast_9[0, 0] [%128, %129] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_15 in writable %subview_16 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }
}
