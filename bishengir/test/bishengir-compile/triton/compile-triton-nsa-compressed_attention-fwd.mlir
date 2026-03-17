// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t
#map = affine_map<(d0) -> (d0)>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @forward_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: i32, %arg12: i32, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: f32, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32, %arg30: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c63_i32 = arith.constant 63 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 1.44269502 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<64xi32>
    %3 = linalg.fill ins(%c1_i32 : i32) outs(%2 : tensor<64xi32>) -> tensor<64xi32>
    %4 = tensor.empty() : tensor<64xf32>
    %5 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<64xf32>) -> tensor<64xf32>
    %6 = tensor.empty() : tensor<64x128xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %8 = tensor.empty() : tensor<64x64xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %10 = linalg.fill ins(%cst_1 : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %inserted = tensor.insert %arg14 into %0[%c0] : tensor<1xf32>
    %11 = arith.mulf %inserted, %1 : tensor<1xf32>
    %extracted = tensor.extract %11[%c0] : tensor<1xf32>
    %12 = arith.addi %arg30, %arg24 : i32
    %13 = arith.divsi %arg29, %arg12 : i32
    %14 = arith.index_cast %arg28 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg9 to offset: [%14], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %15 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %16 = arith.addi %14, %c1 : index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg9 to offset: [%16], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %17 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %18 = arith.subi %17, %15 : i32
    %reinterpret_cast_4 = memref.reinterpret_cast %arg10 to offset: [%14], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %19 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg10 to offset: [%16], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %20 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %21 = arith.subi %20, %19 : i32
    %22 = arith.muli %12, %c64_i32 : i32
    %23 = arith.addi %22, %arg7 : i32
    %24 = arith.subi %23, %c1_i32 : i32
    %25 = arith.cmpi sge, %24, %18 : i32
    scf.if %25 {
    } else {
      %26 = arith.muli %15, %arg15 : i32
      %27 = arith.index_cast %26 : i32 to index
      %28 = arith.muli %arg29, %arg16 : i32
      %29 = arith.index_cast %28 : i32 to index
      %30 = arith.addi %27, %29 : index
      %31 = arith.index_cast %24 : i32 to index
      %32 = arith.index_cast %arg15 : i32 to index
      %33 = arith.muli %31, %32 : index
      %34 = arith.addi %33, %30 : index
      %35 = arith.index_cast %18 : i32 to index
      %36 = arith.index_cast %arg13 : i32 to index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 128], strides: [%32, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
      %37 = arith.muli %19, %arg17 : i32
      %38 = arith.index_cast %37 : i32 to index
      %39 = arith.muli %13, %arg18 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = arith.addi %38, %40 : index
      %42 = arith.muli %19, %arg19 : i32
      %43 = arith.index_cast %42 : i32 to index
      %44 = arith.muli %13, %arg20 : i32
      %45 = arith.index_cast %44 : i32 to index
      %46 = arith.addi %43, %45 : index
      %alloc = memref.alloc() : memref<64x128xf16>
      %47 = arith.divsi %33, %32 : index
      %48 = arith.subi %35, %47 : index
      %49 = arith.maxsi %48, %c0 : index
      %50 = arith.minsi %49, %c64 : index
      %51 = arith.remsi %33, %32 : index
      %52 = arith.subi %36, %51 : index
      %53 = arith.maxsi %52, %c0 : index
      %54 = arith.minsi %53, %c128 : index
      %55 = arith.cmpi slt, %50, %c64 : index
      %56 = arith.cmpi slt, %54, %c128 : index
      %57 = arith.ori %55, %56 : i1
      scf.if %57 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<64x128xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_6[0, 0] [%50, %54] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_7 = memref.subview %alloc[0, 0] [%50, %54] [1, 1] : memref<64x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview, %subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %58 = bufferization.to_tensor %alloc restrict writable : memref<64x128xf16>
      %59 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : tensor<64xi32>) {
      ^bb0(%out: i32):
        %102 = linalg.index 0 : index
        %103 = arith.index_cast %102 : index to i32
        linalg.yield %103 : i32
      } -> tensor<64xi32>
      %60 = linalg.fill ins(%24 : i32) outs(%2 : tensor<64xi32>) -> tensor<64xi32>
      %61 = arith.addi %59, %60 : tensor<64xi32>
      %62 = linalg.fill ins(%arg8 : i32) outs(%2 : tensor<64xi32>) -> tensor<64xi32>
      %63 = arith.muli %59, %62 : tensor<64xi32>
      %64 = linalg.fill ins(%arg7 : i32) outs(%2 : tensor<64xi32>) -> tensor<64xi32>
      %65 = arith.addi %63, %64 : tensor<64xi32>
      %66 = arith.subi %65, %3 : tensor<64xi32>
      %67 = arith.addi %23, %c63_i32 : i32
      %68 = arith.subi %67, %arg7 : i32
      %69 = arith.divsi %68, %arg8 : i32
      %70 = arith.addi %69, %c1_i32 : i32
      %71 = arith.minsi %21, %70 : i32
      %72 = tensor.empty() : tensor<64x64xi32>
      %broadcasted = linalg.broadcast ins(%61 : tensor<64xi32>) outs(%72 : tensor<64x64xi32>) dimensions = [1]
      %73 = linalg.fill ins(%extracted : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %74:5 = scf.for %arg31 = %c0_i32 to %71 step %c64_i32 iter_args(%arg32 = %7, %arg33 = %5, %arg34 = %5, %arg35 = %c0_i32, %arg36 = %c0_i32) -> (tensor<64x128xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32)  : i32 {
        %102 = arith.index_cast %arg36 : i32 to index
        %103 = arith.index_cast %arg19 : i32 to index
        %104 = arith.muli %102, %103 : index
        %105 = arith.addi %104, %46 : index
        %106 = arith.index_cast %21 : i32 to index
        %reinterpret_cast_14 = memref.reinterpret_cast %arg4 to offset: [%105], sizes: [64, 128], strides: [%103, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
        %107 = arith.index_cast %arg35 : i32 to index
        %108 = arith.index_cast %arg17 : i32 to index
        %109 = arith.muli %107, %108 : index
        %110 = arith.addi %109, %41 : index
        %reinterpret_cast_15 = memref.reinterpret_cast %arg3 to offset: [%110], sizes: [64, 128], strides: [%108, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
        %alloc_16 = memref.alloc() : memref<64x128xf16>
        %111 = arith.divsi %109, %108 : index
        %112 = arith.subi %106, %111 : index
        %113 = arith.maxsi %112, %c0 : index
        %114 = arith.minsi %113, %c64 : index
        %115 = arith.remsi %109, %108 : index
        %116 = arith.subi %36, %115 : index
        %117 = arith.maxsi %116, %c0 : index
        %118 = arith.minsi %117, %c128 : index
        %119 = arith.cmpi slt, %114, %c64 : index
        %120 = arith.cmpi slt, %118, %c128 : index
        %121 = arith.ori %119, %120 : i1
        scf.if %121 {
          linalg.fill ins(%cst : f16) outs(%alloc_16 : memref<64x128xf16>)
        } {hivm.unlikely_condition}
        %subview_17 = memref.subview %reinterpret_cast_15[0, 0] [%114, %118] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_18 = memref.subview %alloc_16[0, 0] [%114, %118] [1, 1] : memref<64x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_17, %subview_18 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %122 = bufferization.to_tensor %alloc_16 restrict writable : memref<64x128xf16>
        %123 = tensor.empty() : tensor<128x64xf16>
        %transposed = linalg.transpose ins(%122 : tensor<64x128xf16>) outs(%123 : tensor<128x64xf16>) permutation = [1, 0]
        %124 = arith.muli %arg31, %arg8 : i32
        %125 = linalg.fill ins(%124 : i32) outs(%2 : tensor<64xi32>) -> tensor<64xi32>
        %126 = arith.addi %125, %66 : tensor<64xi32>
        %broadcasted_19 = linalg.broadcast ins(%126 : tensor<64xi32>) outs(%72 : tensor<64x64xi32>) dimensions = [0]
        %127 = arith.cmpi sge, %broadcasted, %broadcasted_19 : tensor<64x64xi32>
        %128 = arith.select %127, %9, %10 : tensor<64x64xi1>, tensor<64x64xf32>
        %129 = arith.addf %128, %9 : tensor<64x64xf32>
        %130 = linalg.matmul {input_precison = "ieee"} ins(%58, %transposed : tensor<64x128xf16>, tensor<128x64xf16>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %131 = arith.mulf %130, %73 : tensor<64x64xf32>
        %132 = arith.addf %129, %131 : tensor<64x64xf32>
        %reduced = linalg.reduce ins(%132 : tensor<64x64xf32>) outs(%5 : tensor<64xf32>) dimensions = [1]
          (%in: f32, %init: f32) {
            %161 = arith.maxnumf %in, %init : f32
            linalg.yield %161 : f32
          }
        %133 = arith.maxnumf %arg33, %reduced : tensor<64xf32>
        %broadcasted_20 = linalg.broadcast ins(%133 : tensor<64xf32>) outs(%8 : tensor<64x64xf32>) dimensions = [1]
        %134 = arith.subf %132, %broadcasted_20 : tensor<64x64xf32>
        %135 = math.exp2 %134 : tensor<64x64xf32>
        %136 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<64xf32>) -> tensor<64xf32>
        %reduced_21 = linalg.reduce ins(%135 : tensor<64x64xf32>) outs(%136 : tensor<64xf32>) dimensions = [1]
          (%in: f32, %init: f32) {
            %161 = arith.addf %in, %init : f32
            linalg.yield %161 : f32
          }
        %137 = arith.subf %arg33, %133 : tensor<64xf32>
        %138 = math.exp2 %137 : tensor<64xf32>
        %broadcasted_22 = linalg.broadcast ins(%138 : tensor<64xf32>) outs(%6 : tensor<64x128xf32>) dimensions = [1]
        %139 = arith.mulf %arg32, %broadcasted_22 : tensor<64x128xf32>
        %alloc_23 = memref.alloc() : memref<64x128xf16>
        %140 = arith.divsi %104, %103 : index
        %141 = arith.subi %106, %140 : index
        %142 = arith.maxsi %141, %c0 : index
        %143 = arith.minsi %142, %c64 : index
        %144 = arith.remsi %104, %103 : index
        %145 = arith.subi %36, %144 : index
        %146 = arith.maxsi %145, %c0 : index
        %147 = arith.minsi %146, %c128 : index
        %148 = arith.cmpi slt, %143, %c64 : index
        %149 = arith.cmpi slt, %147, %c128 : index
        %150 = arith.ori %148, %149 : i1
        scf.if %150 {
          linalg.fill ins(%cst : f16) outs(%alloc_23 : memref<64x128xf16>)
        } {hivm.unlikely_condition}
        %subview_24 = memref.subview %reinterpret_cast_14[0, 0] [%143, %147] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_25 = memref.subview %alloc_23[0, 0] [%143, %147] [1, 1] : memref<64x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_24, %subview_25 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %151 = bufferization.to_tensor %alloc_23 restrict writable : memref<64x128xf16>
        %152 = arith.truncf %135 : tensor<64x64xf32> to tensor<64x64xf16>
        %153 = linalg.matmul {input_precison = "ieee"} ins(%152, %151 : tensor<64x64xf16>, tensor<64x128xf16>) outs(%139 : tensor<64x128xf32>) -> tensor<64x128xf32>
        %154 = arith.subf %arg34, %133 : tensor<64xf32>
        %155 = math.exp2 %154 : tensor<64xf32>
        %156 = arith.addf %155, %reduced_21 : tensor<64xf32>
        %157 = math.log2 %156 : tensor<64xf32>
        %158 = arith.addf %133, %157 : tensor<64xf32>
        %159 = arith.addi %arg35, %c64_i32 : i32
        %160 = arith.addi %arg36, %c64_i32 : i32
        scf.yield %153, %133, %158, %159, %160 : tensor<64x128xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32
      } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
      %75 = arith.subf %74#1, %74#2 : tensor<64xf32>
      %76 = math.exp2 %75 : tensor<64xf32>
      %broadcasted_8 = linalg.broadcast ins(%76 : tensor<64xf32>) outs(%6 : tensor<64x128xf32>) dimensions = [1]
      %77 = arith.mulf %74#0, %broadcasted_8 : tensor<64x128xf32>
      %78 = arith.muli %15, %arg21 : i32
      %79 = arith.index_cast %78 : i32 to index
      %80 = arith.muli %arg29, %arg22 : i32
      %81 = arith.index_cast %80 : i32 to index
      %82 = arith.addi %79, %81 : index
      %83 = arith.index_cast %arg21 : i32 to index
      %84 = arith.muli %31, %83 : index
      %85 = arith.addi %82, %84 : index
      %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%85], sizes: [64, 128], strides: [%83, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
      %86 = arith.truncf %77 : tensor<64x128xf32> to tensor<64x128xf16>
      %87 = arith.addi %31, %c64 : index
      %88 = arith.index_cast %18 : i32 to index
      %89 = arith.maxsi %31, %88 : index
      %90 = arith.minsi %87, %89 : index
      %91 = arith.subi %90, %31 : index
      %92 = arith.index_cast %arg13 : i32 to index
      %93 = arith.maxsi %92, %c0 : index
      %94 = arith.minsi %93, %c128 : index
      %95 = arith.minsi %91, %c64 : index
      %96 = arith.minsi %94, %c128 : index
      %extracted_slice = tensor.extract_slice %86[0, 0] [%95, %96] [1, 1] : tensor<64x128xf16> to tensor<?x?xf16>
      %subview_10 = memref.subview %reinterpret_cast_9[0, 0] [%95, %96] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_10 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
      %97 = arith.index_cast %15 : i32 to index
      %98 = arith.muli %arg29, %arg23 : i32
      %99 = arith.index_cast %98 : i32 to index
      %100 = arith.addi %97, %99 : index
      %101 = arith.addi %100, %31 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg6 to offset: [%101], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
      %extracted_slice_12 = tensor.extract_slice %74#2[0] [%91] [1] : tensor<64xf32> to tensor<?xf32>
      %subview_13 = memref.subview %reinterpret_cast_11[0] [%91] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_12 in writable %subview_13 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}
