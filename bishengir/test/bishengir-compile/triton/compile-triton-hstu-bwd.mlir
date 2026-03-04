// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @_hstu_attn_bwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg9: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg10: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg11: memref<?xi32> {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: f32, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32, %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32, %arg38: i32, %arg39: i32, %arg40: i32, %arg41: i32, %arg42: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<16x16xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = tensor.empty() : tensor<16x64xf32>
    %6 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %7 = arith.divsi %arg41, %arg31 : i32
    %8 = arith.remsi %arg41, %arg31 : i32
    %9 = arith.divsi %8, %arg32 : i32
    %10 = arith.extsi %8 : i32 to i64
    %11 = arith.extsi %9 : i32 to i64
    %12 = arith.index_cast %7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%12], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
    %13 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>>
    %14 = arith.addi %12, %c1 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%14], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
    %15 = memref.load %reinterpret_cast_2[%c0] : memref<1xi64, strided<[1], offset: ?>>
    %16 = arith.subi %15, %13 : i64
    %17 = arith.trunci %16 : i64 to i32
    %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%12], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
    %18 = memref.load %reinterpret_cast_3[%c0] : memref<1xi64, strided<[1], offset: ?>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%14], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
    %19 = memref.load %reinterpret_cast_4[%c0] : memref<1xi64, strided<[1], offset: ?>>
    %20 = arith.subi %19, %18 : i64
    %21 = arith.trunci %20 : i64 to i32
    %22 = arith.extsi %arg12 : i32 to i64
    %23 = arith.muli %13, %22 : i64
    %24 = arith.extsi %arg13 : i32 to i64
    %25 = arith.muli %10, %24 : i64
    %26 = arith.addi %23, %25 : i64
    %27 = arith.index_cast %26 : i64 to index
    %28 = arith.extsi %arg14 : i32 to i64
    %29 = arith.muli %18, %28 : i64
    %30 = arith.extsi %arg15 : i32 to i64
    %31 = arith.muli %11, %30 : i64
    %32 = arith.addi %29, %31 : i64
    %33 = arith.index_cast %32 : i64 to index
    %34 = arith.extsi %arg16 : i32 to i64
    %35 = arith.muli %18, %34 : i64
    %36 = arith.extsi %arg17 : i32 to i64
    %37 = arith.muli %11, %36 : i64
    %38 = arith.addi %35, %37 : i64
    %39 = arith.index_cast %38 : i64 to index
    %40 = arith.extsi %arg18 : i32 to i64
    %41 = arith.muli %13, %40 : i64
    %42 = arith.extsi %arg19 : i32 to i64
    %43 = arith.muli %10, %42 : i64
    %44 = arith.addi %41, %43 : i64
    %45 = arith.index_cast %44 : i64 to index
    %46 = arith.extsi %arg20 : i32 to i64
    %47 = arith.muli %13, %46 : i64
    %48 = arith.extsi %arg21 : i32 to i64
    %49 = arith.muli %10, %48 : i64
    %50 = arith.addi %47, %49 : i64
    %51 = arith.index_cast %50 : i64 to index
    %52 = arith.extsi %arg22 : i32 to i64
    %53 = arith.muli %18, %52 : i64
    %54 = arith.extsi %arg23 : i32 to i64
    %55 = arith.muli %11, %54 : i64
    %56 = arith.addi %53, %55 : i64
    %57 = arith.index_cast %56 : i64 to index
    %58 = arith.extsi %arg24 : i32 to i64
    %59 = arith.muli %18, %58 : i64
    %60 = arith.extsi %arg25 : i32 to i64
    %61 = arith.muli %11, %60 : i64
    %62 = arith.addi %59, %61 : i64
    %63 = arith.index_cast %62 : i64 to index
    %64 = arith.index_cast %arg12 : i32 to index
    %65 = arith.index_cast %arg20 : i32 to index
    %66 = arith.index_cast %arg18 : i32 to index
    %67 = linalg.fill ins(%arg26 : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %68 = arith.sitofp %arg33 : i32 to f32
    %inserted = tensor.insert %68 into %0[%c0] : tensor<1xf32>
    %69 = arith.divf %1, %inserted : tensor<1xf32>
    %extracted = tensor.extract %69[%c0] : tensor<1xf32>
    %70 = linalg.fill ins(%extracted : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %71 = linalg.fill ins(%arg26 : f32) outs(%5 : tensor<16x64xf32>) -> tensor<16x64xf32>
    scf.for %arg43 = %c0_i32 to %21 step %c16_i32  : i32 {
      %72 = arith.index_cast %arg43 : i32 to index
      %73 = arith.index_cast %arg14 : i32 to index
      %74 = arith.muli %72, %73 : index
      %75 = arith.addi %33, %74 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%75], sizes: [16, 64], strides: [%73, 1] : memref<?xf16> to memref<16x64xf16, strided<[?, 1], offset: ?>>
      %76 = arith.index_cast %arg16 : i32 to index
      %77 = arith.muli %72, %76 : index
      %78 = arith.addi %39, %77 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%78], sizes: [16, 64], strides: [%76, 1] : memref<?xf16> to memref<16x64xf16, strided<[?, 1], offset: ?>>
      %alloc = memref.alloc() : memref<16x64xf16>
      %79 = arith.addi %72, %c16 : index
      %80 = arith.index_cast %21 : i32 to index
      %81 = arith.maxsi %72, %80 : index
      %82 = arith.minsi %79, %81 : index
      %83 = arith.subi %82, %72 : index
      %84 = arith.cmpi slt, %83, %c16 : index
      scf.if %84 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<16x64xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_5[0, 0] [%83, 64] [1, 1] : memref<16x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[?, 1], offset: ?>>
      %subview_7 = memref.subview %alloc[0, 0] [%83, 64] [1, 1] : memref<16x64xf16> to memref<?x64xf16, strided<[64, 1]>>
      memref.copy %subview, %subview_7 : memref<?x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1]>>
      %85 = bufferization.to_tensor %alloc restrict writable : memref<16x64xf16>
      %alloc_8 = memref.alloc() : memref<16x64xf16>
      scf.if %84 {
        linalg.fill ins(%cst : f16) outs(%alloc_8 : memref<16x64xf16>)
      } {hivm.unlikely_condition}
      %subview_9 = memref.subview %reinterpret_cast_6[0, 0] [%83, 64] [1, 1] : memref<16x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[?, 1], offset: ?>>
      %subview_10 = memref.subview %alloc_8[0, 0] [%83, 64] [1, 1] : memref<16x64xf16> to memref<?x64xf16, strided<[64, 1]>>
      memref.copy %subview_9, %subview_10 : memref<?x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1]>>
      %86 = bufferization.to_tensor %alloc_8 restrict writable : memref<16x64xf16>
      %87:2 = scf.for %arg44 = %c0_i32 to %17 step %c16_i32 iter_args(%arg45 = %6, %arg46 = %6) -> (tensor<16x64xf32>, tensor<16x64xf32>)  : i32 {
        %97 = arith.muli %arg44, %arg12 : i32
        %98 = arith.index_cast %97 : i32 to index
        %99 = arith.addi %27, %98 : index
        %reinterpret_cast_16 = memref.reinterpret_cast %arg2 to offset: [%99], sizes: [16, 64], strides: [%64, 1] : memref<?xf16> to memref<16x64xf16, strided<[?, 1], offset: ?>>
        %alloc_17 = memref.alloc() : memref<16x64xf16>
        %100 = arith.index_cast %arg44 : i32 to index
        %101 = arith.addi %100, %c16 : index
        %102 = arith.index_cast %17 : i32 to index
        %103 = arith.maxsi %100, %102 : index
        %104 = arith.minsi %101, %103 : index
        %105 = arith.subi %104, %100 : index
        %106 = arith.cmpi slt, %105, %c16 : index
        scf.if %106 {
          linalg.fill ins(%cst : f16) outs(%alloc_17 : memref<16x64xf16>)
        } {hivm.unlikely_condition}
        %subview_18 = memref.subview %reinterpret_cast_16[0, 0] [%105, 64] [1, 1] : memref<16x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[?, 1], offset: ?>>
        %subview_19 = memref.subview %alloc_17[0, 0] [%105, 64] [1, 1] : memref<16x64xf16> to memref<?x64xf16, strided<[64, 1]>>
        memref.copy %subview_18, %subview_19 : memref<?x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1]>>
        %107 = bufferization.to_tensor %alloc_17 restrict writable : memref<16x64xf16>
        %108 = tensor.empty() : tensor<64x16xf16>
        %transposed = linalg.transpose ins(%107 : tensor<16x64xf16>) outs(%108 : tensor<64x16xf16>) permutation = [1, 0] 
        %109 = linalg.matmul {input_precison = "ieee"} ins(%85, %transposed : tensor<16x64xf16>, tensor<64x16xf16>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
        %110 = arith.mulf %109, %67 : tensor<16x16xf32>
        %111 = arith.subf %3, %110 : tensor<16x16xf32>
        %112 = math.exp %111 : tensor<16x16xf32>
        %113 = arith.addf %112, %4 : tensor<16x16xf32>
        %114 = arith.divf %4, %113 : tensor<16x16xf32>
        %115 = arith.mulf %110, %114 : tensor<16x16xf32>
        %116 = arith.mulf %115, %70 : tensor<16x16xf32>
        %117 = arith.truncf %116 : tensor<16x16xf32> to tensor<16x16xf16>
        %118 = arith.muli %arg44, %arg18 : i32
        %119 = arith.index_cast %118 : i32 to index
        %120 = arith.addi %45, %119 : index
        %reinterpret_cast_20 = memref.reinterpret_cast %arg7 to offset: [%120], sizes: [16, 64], strides: [%66, 1] : memref<?xf16> to memref<16x64xf16, strided<[?, 1], offset: ?>>
        %alloc_21 = memref.alloc() : memref<16x64xf16>
        scf.if %106 {
          linalg.fill ins(%cst : f16) outs(%alloc_21 : memref<16x64xf16>)
        } {hivm.unlikely_condition}
        %subview_22 = memref.subview %reinterpret_cast_20[0, 0] [%105, 64] [1, 1] : memref<16x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[?, 1], offset: ?>>
        %subview_23 = memref.subview %alloc_21[0, 0] [%105, 64] [1, 1] : memref<16x64xf16> to memref<?x64xf16, strided<[64, 1]>>
        memref.copy %subview_22, %subview_23 : memref<?x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1]>>
        %121 = bufferization.to_tensor %alloc_21 restrict writable : memref<16x64xf16>
        %122 = linalg.matmul {input_precison = "ieee"} ins(%117, %121 : tensor<16x16xf16>, tensor<16x64xf16>) outs(%arg46 : tensor<16x64xf32>) -> tensor<16x64xf32>
        %transposed_24 = linalg.transpose ins(%121 : tensor<16x64xf16>) outs(%108 : tensor<64x16xf16>) permutation = [1, 0] 
        %123 = linalg.matmul {input_precison = "ieee"} ins(%86, %transposed_24 : tensor<16x64xf16>, tensor<64x16xf16>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
        %124 = arith.mulf %123, %114 : tensor<16x16xf32>
        %125 = arith.subf %4, %114 : tensor<16x16xf32>
        %126 = arith.mulf %110, %125 : tensor<16x16xf32>
        %127 = arith.addf %126, %4 : tensor<16x16xf32>
        %128 = arith.mulf %124, %127 : tensor<16x16xf32>
        %129 = arith.mulf %128, %70 : tensor<16x16xf32>
        %130 = arith.extf %transposed : tensor<64x16xf16> to tensor<64x16xf32>
        %transposed_25 = linalg.transpose ins(%130 : tensor<64x16xf32>) outs(%5 : tensor<16x64xf32>) permutation = [1, 0] 
        %131 = arith.truncf %129 : tensor<16x16xf32> to tensor<16x16xf16>
        %132 = linalg.matmul {input_precison = "ieee"} ins(%129, %transposed_25 : tensor<16x16xf32>, tensor<16x64xf32>) outs(%arg45 : tensor<16x64xf32>) -> tensor<16x64xf32>
        %133 = tensor.empty() : tensor<16x16xf16>
        %transposed_26 = linalg.transpose ins(%131 : tensor<16x16xf16>) outs(%133 : tensor<16x16xf16>) permutation = [1, 0] 
        %134 = linalg.matmul {input_precison = "ieee"} ins(%transposed_26, %85 : tensor<16x16xf16>, tensor<16x64xf16>) outs(%6 : tensor<16x64xf32>) -> tensor<16x64xf32>
        %135 = arith.mulf %134, %71 : tensor<16x64xf32>
        %136 = arith.muli %arg44, %arg20 : i32
        %137 = arith.index_cast %136 : i32 to index
        %138 = arith.addi %51, %137 : index
        %reinterpret_cast_27 = memref.reinterpret_cast %arg8 to offset: [%138], sizes: [16, 64], strides: [%65, 1] : memref<?xf32> to memref<16x64xf32, strided<[?, 1], offset: ?>>
        %subview_28 = memref.subview %reinterpret_cast_27[0, 0] [%105, 64] [1, 1] : memref<16x64xf32, strided<[?, 1], offset: ?>> to memref<?x64xf32, strided<[?, 1], offset: ?>>
        %extracted_slice_29 = tensor.extract_slice %135[0, 0] [%105, 64] [1, 1] : tensor<16x64xf32> to tensor<?x64xf32>
        hivm.hir.store ins(%extracted_slice_29 : tensor<?x64xf32>) outs(%subview_28 : memref<?x64xf32, strided<[?, 1], offset: ?>>) atomic = <add>
        scf.yield %132, %122 : tensor<16x64xf32>, tensor<16x64xf32>
      } {tt.divisibility_arg1 = dense<16> : tensor<1xi32>}
      %88 = arith.index_cast %arg24 : i32 to index
      %89 = arith.muli %72, %88 : index
      %90 = arith.addi %63, %89 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg10 to offset: [%90], sizes: [16, 64], strides: [%88, 1] : memref<?xf16> to memref<16x64xf16, strided<[?, 1], offset: ?>>
      %91 = arith.index_cast %arg22 : i32 to index
      %92 = arith.muli %72, %91 : index
      %93 = arith.addi %57, %92 : index
      %reinterpret_cast_12 = memref.reinterpret_cast %arg9 to offset: [%93], sizes: [16, 64], strides: [%91, 1] : memref<?xf16> to memref<16x64xf16, strided<[?, 1], offset: ?>>
      %94 = arith.mulf %87#0, %71 : tensor<16x64xf32>
      %95 = arith.truncf %87#1 : tensor<16x64xf32> to tensor<16x64xf16>
      %subview_13 = memref.subview %reinterpret_cast_11[0, 0] [%83, 64] [1, 1] : memref<16x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[?, 1], offset: ?>>
      %extracted_slice = tensor.extract_slice %95[0, 0] [%83, 64] [1, 1] : tensor<16x64xf16> to tensor<?x64xf16>
      hivm.hir.store ins(%extracted_slice : tensor<?x64xf16>) outs(%subview_13 : memref<?x64xf16, strided<[?, 1], offset: ?>>) atomic = <add>
      %96 = arith.truncf %94 : tensor<16x64xf32> to tensor<16x64xf16>
      %subview_14 = memref.subview %reinterpret_cast_12[0, 0] [%83, 64] [1, 1] : memref<16x64xf16, strided<[?, 1], offset: ?>> to memref<?x64xf16, strided<[?, 1], offset: ?>>
      %extracted_slice_15 = tensor.extract_slice %96[0, 0] [%83, 64] [1, 1] : tensor<16x64xf16> to tensor<?x64xf16>
      hivm.hir.store ins(%extracted_slice_15 : tensor<?x64xf16>) outs(%subview_14 : memref<?x64xf16, strided<[?, 1], offset: ?>>) atomic = <add>
    }
    return
  }
}