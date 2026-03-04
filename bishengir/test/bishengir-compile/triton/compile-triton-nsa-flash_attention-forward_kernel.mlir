// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @forward_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: f32, %arg14: i1 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 1.44269502 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<32x32xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = tensor.empty() : tensor<1x32xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %6 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %7 = tensor.empty() : tensor<32x128xf32>
    %8 = linalg.fill ins(%cst_1 : f32) outs(%7 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %9 = tensor.empty() : tensor<32xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<32xf32>) -> tensor<32xf32>
    %inserted = tensor.insert %arg13 into %0[%c0] : tensor<1xf32>
    %11 = arith.mulf %inserted, %1 : tensor<1xf32>
    %extracted = tensor.extract %11[%c0] : tensor<1xf32>
    %12 = scf.if %arg14 -> (i32) {
      %23 = arith.remsi %arg28, %arg9 : i32
      scf.yield %23 : i32
    } else {
      %23 = arith.divsi %arg28, %arg10 : i32
      scf.yield %23 : i32
    }
    %13 = arith.index_cast %arg27 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%13], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %14 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %15 = arith.addi %13, %c1 : index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %16 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %17 = arith.subi %16, %14 : i32
    %reinterpret_cast_4 = memref.reinterpret_cast %arg8 to offset: [%13], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %18 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg8 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %19 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %20 = arith.subi %19, %18 : i32
    %21 = arith.muli %arg29, %c32_i32 : i32
    %22 = arith.cmpi sge, %21, %17 : i32
    scf.if %22 {
    } else {
      %23 = arith.muli %14, %arg15 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.muli %arg28, %arg16 : i32
      %26 = arith.index_cast %25 : i32 to index
      %27 = arith.addi %24, %26 : index
      %28 = arith.index_cast %21 : i32 to index
      %29 = arith.index_cast %arg15 : i32 to index
      %30 = arith.muli %28, %29 : index
      %31 = arith.addi %30, %27 : index
      %32 = arith.index_cast %17 : i32 to index
      %33 = arith.index_cast %arg11 : i32 to index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [32, 128], strides: [%29, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %alloc = memref.alloc() : memref<32x128xf16>
      %34 = arith.divsi %30, %29 : index
      %35 = arith.subi %32, %34 : index
      %36 = arith.maxsi %35, %c0 : index
      %37 = arith.minsi %36, %c32 : index
      %38 = arith.remsi %30, %29 : index
      %39 = arith.subi %33, %38 : index
      %40 = arith.maxsi %39, %c0 : index
      %41 = arith.minsi %40, %c128 : index
      %42 = arith.cmpi slt, %37, %c32 : index
      %43 = arith.cmpi slt, %41, %c128 : index
      %44 = arith.ori %42, %43 : i1
      scf.if %44 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<32x128xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_6[0, 0] [%37, %41] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_7 = memref.subview %alloc[0, 0] [%37, %41] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview, %subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %45 = bufferization.to_tensor %alloc restrict writable : memref<32x128xf16>
      %46 = arith.muli %18, %arg17 : i32
      %47 = arith.index_cast %46 : i32 to index
      %48 = arith.muli %12, %arg18 : i32
      %49 = arith.index_cast %48 : i32 to index
      %50 = arith.addi %47, %49 : index
      %51 = arith.muli %18, %arg19 : i32
      %52 = arith.index_cast %51 : i32 to index
      %53 = arith.muli %12, %arg20 : i32
      %54 = arith.index_cast %53 : i32 to index
      %55 = arith.addi %52, %54 : index
      %56 = linalg.fill ins(%extracted : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %57:3 = scf.for %arg30 = %c0_i32 to %20 step %c32_i32 iter_args(%arg31 = %8, %arg32 = %10, %arg33 = %10) -> (tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>)  : i32 {
        %85 = arith.index_cast %arg30 : i32 to index
        %86 = arith.index_cast %arg17 : i32 to index
        %87 = arith.muli %85, %86 : index
        %88 = arith.addi %87, %50 : index
        %89 = arith.index_cast %20 : i32 to index
        %reinterpret_cast_13 = memref.reinterpret_cast %arg3 to offset: [%88], sizes: [32, 128], strides: [%86, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
        %90 = arith.index_cast %arg19 : i32 to index
        %91 = arith.muli %85, %90 : index
        %92 = arith.addi %91, %55 : index
        %93 = arith.index_cast %arg12 : i32 to index
        %reinterpret_cast_14 = memref.reinterpret_cast %arg4 to offset: [%92], sizes: [32, 128], strides: [%90, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
        %alloc_15 = memref.alloc() : memref<32x128xf16>
        %94 = arith.divsi %87, %86 : index
        %95 = arith.subi %89, %94 : index
        %96 = arith.maxsi %95, %c0 : index
        %97 = arith.minsi %96, %c32 : index
        %98 = arith.remsi %87, %86 : index
        %99 = arith.subi %33, %98 : index
        %100 = arith.maxsi %99, %c0 : index
        %101 = arith.minsi %100, %c128 : index
        %102 = arith.cmpi slt, %97, %c32 : index
        %103 = arith.cmpi slt, %101, %c128 : index
        %104 = arith.ori %102, %103 : i1
        scf.if %104 {
          linalg.fill ins(%cst : f16) outs(%alloc_15 : memref<32x128xf16>)
        } {hivm.unlikely_condition}
        %subview_16 = memref.subview %reinterpret_cast_13[0, 0] [%97, %101] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_17 = memref.subview %alloc_15[0, 0] [%97, %101] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_16, %subview_17 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %105 = bufferization.to_tensor %alloc_15 restrict writable : memref<32x128xf16>
        %106 = tensor.empty() : tensor<128x32xf16>
        %transposed = linalg.transpose ins(%105 : tensor<32x128xf16>) outs(%106 : tensor<128x32xf16>) permutation = [1, 0] 
        %107 = arith.subi %20, %arg30 : i32
        %108 = arith.index_cast %107 : i32 to index
        %109 = arith.maxsi %108, %c0 : index
        %110 = arith.minsi %109, %c32 : index
        %extracted_slice_18 = tensor.extract_slice %6[0, 0] [1, %110] [1, 1] : tensor<1x32xf32> to tensor<1x?xf32>
        %inserted_slice = tensor.insert_slice %extracted_slice_18 into %5[0, 0] [1, %110] [1, 1] : tensor<1x?xf32> into tensor<1x32xf32>
        %111 = arith.addf %inserted_slice, %6 : tensor<1x32xf32>
        %collapsed = tensor.collapse_shape %111 [[0, 1]] : tensor<1x32xf32> into tensor<32xf32>
        %broadcasted_19 = linalg.broadcast ins(%collapsed : tensor<32xf32>) outs(%2 : tensor<32x32xf32>) dimensions = [0] 
        %112 = linalg.matmul {input_precison = "ieee"} ins(%45, %transposed : tensor<32x128xf16>, tensor<128x32xf16>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %113 = arith.mulf %112, %56 : tensor<32x32xf32>
        %114 = arith.addf %broadcasted_19, %113 : tensor<32x32xf32>
        %reduced = linalg.reduce ins(%114 : tensor<32x32xf32>) outs(%10 : tensor<32xf32>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %141 = arith.maxnumf %in, %init : f32
            linalg.yield %141 : f32
          }
        %115 = arith.maxnumf %arg32, %reduced : tensor<32xf32>
        %broadcasted_20 = linalg.broadcast ins(%115 : tensor<32xf32>) outs(%2 : tensor<32x32xf32>) dimensions = [1] 
        %116 = arith.subf %114, %broadcasted_20 : tensor<32x32xf32>
        %117 = math.exp2 %116 : tensor<32x32xf32>
        %118 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<32xf32>) -> tensor<32xf32>
        %reduced_21 = linalg.reduce ins(%117 : tensor<32x32xf32>) outs(%118 : tensor<32xf32>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %141 = arith.addf %in, %init : f32
            linalg.yield %141 : f32
          }
        %119 = arith.subf %arg32, %115 : tensor<32xf32>
        %120 = math.exp2 %119 : tensor<32xf32>
        %broadcasted_22 = linalg.broadcast ins(%120 : tensor<32xf32>) outs(%7 : tensor<32x128xf32>) dimensions = [1] 
        %121 = arith.mulf %arg31, %broadcasted_22 : tensor<32x128xf32>
        %alloc_23 = memref.alloc() : memref<32x128xf16>
        %122 = arith.divsi %91, %90 : index
        %123 = arith.subi %89, %122 : index
        %124 = arith.maxsi %123, %c0 : index
        %125 = arith.minsi %124, %c32 : index
        %126 = arith.remsi %91, %90 : index
        %127 = arith.subi %93, %126 : index
        %128 = arith.maxsi %127, %c0 : index
        %129 = arith.minsi %128, %c128 : index
        %130 = arith.cmpi slt, %125, %c32 : index
        %131 = arith.cmpi slt, %129, %c128 : index
        %132 = arith.ori %130, %131 : i1
        scf.if %132 {
          linalg.fill ins(%cst : f16) outs(%alloc_23 : memref<32x128xf16>)
        } {hivm.unlikely_condition}
        %subview_24 = memref.subview %reinterpret_cast_14[0, 0] [%125, %129] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_25 = memref.subview %alloc_23[0, 0] [%125, %129] [1, 1] : memref<32x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_24, %subview_25 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %133 = bufferization.to_tensor %alloc_23 restrict writable : memref<32x128xf16>
        %134 = arith.truncf %117 : tensor<32x32xf32> to tensor<32x32xf16>
        %135 = linalg.matmul {input_precison = "ieee"} ins(%134, %133 : tensor<32x32xf16>, tensor<32x128xf16>) outs(%121 : tensor<32x128xf32>) -> tensor<32x128xf32>
        %136 = arith.subf %arg33, %115 : tensor<32xf32>
        %137 = math.exp2 %136 : tensor<32xf32>
        %138 = arith.addf %137, %reduced_21 : tensor<32xf32>
        %139 = math.log2 %138 : tensor<32xf32>
        %140 = arith.addf %115, %139 : tensor<32xf32>
        scf.yield %135, %115, %140 : tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>
      } {tt.divisibility_arg1 = dense<32> : tensor<1xi32>}
      %58 = arith.subf %57#1, %57#2 : tensor<32xf32>
      %59 = math.exp2 %58 : tensor<32xf32>
      %broadcasted = linalg.broadcast ins(%59 : tensor<32xf32>) outs(%7 : tensor<32x128xf32>) dimensions = [1] 
      %60 = arith.mulf %57#0, %broadcasted : tensor<32x128xf32>
      %61 = arith.muli %14, %arg21 : i32
      %62 = arith.index_cast %61 : i32 to index
      %63 = arith.muli %arg28, %arg22 : i32
      %64 = arith.index_cast %63 : i32 to index
      %65 = arith.addi %62, %64 : index
      %66 = arith.index_cast %arg21 : i32 to index
      %67 = arith.muli %28, %66 : index
      %68 = arith.addi %65, %67 : index
      %reinterpret_cast_8 = memref.reinterpret_cast %arg5 to offset: [%68], sizes: [32, 128], strides: [%66, 1] : memref<?xf16> to memref<32x128xf16, strided<[?, 1], offset: ?>>
      %69 = arith.truncf %60 : tensor<32x128xf32> to tensor<32x128xf16>
      %70 = arith.addi %28, %c32 : index
      %71 = arith.index_cast %17 : i32 to index
      %72 = arith.maxsi %28, %71 : index
      %73 = arith.minsi %70, %72 : index
      %74 = arith.subi %73, %28 : index
      %75 = arith.index_cast %arg12 : i32 to index
      %76 = arith.maxsi %75, %c0 : index
      %77 = arith.minsi %76, %c128 : index
      %78 = arith.minsi %74, %c32 : index
      %79 = arith.minsi %77, %c128 : index
      %extracted_slice = tensor.extract_slice %69[0, 0] [%78, %79] [1, 1] : tensor<32x128xf16> to tensor<?x?xf16>
      %subview_9 = memref.subview %reinterpret_cast_8[0, 0] [%78, %79] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_9 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
      %80 = arith.index_cast %14 : i32 to index
      %81 = arith.muli %arg28, %arg23 : i32
      %82 = arith.index_cast %81 : i32 to index
      %83 = arith.addi %80, %82 : index
      %84 = arith.addi %83, %28 : index
      %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%84], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
      %extracted_slice_11 = tensor.extract_slice %57#2[0] [%74] [1] : tensor<32xf32> to tensor<?xf32>
      %subview_12 = memref.subview %reinterpret_cast_10[0] [%74] [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_11 in writable %subview_12 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}