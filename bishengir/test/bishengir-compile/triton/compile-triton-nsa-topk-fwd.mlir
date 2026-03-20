// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t
#map = affine_map<(d0) -> (d0)>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @forward_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: i32, %arg11: i32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: f32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32, %arg21: i32, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32, %arg30: i32, %arg31: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %c-1_i32 = arith.constant -1 : i32
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 1.44269502 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<16x64xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %4 = tensor.empty() : tensor<1x64xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %6 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %7 = tensor.empty() : tensor<16x128xf32>
    %8 = linalg.fill ins(%cst_1 : f32) outs(%7 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %9 = tensor.empty() : tensor<16xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<16xf32>) -> tensor<16xf32>
    %11 = tensor.empty() : tensor<8xi32>
    %12 = linalg.fill ins(%c0_i32 : i32) outs(%11 : tensor<8xi32>) -> tensor<8xi32>
    %13 = linalg.fill ins(%c1_i32 : i32) outs(%11 : tensor<8xi32>) -> tensor<8xi32>
    %inserted = tensor.insert %arg13 into %0[%c0] : tensor<1xf32>
    %14 = arith.mulf %inserted, %1 : tensor<1xf32>
    %extracted = tensor.extract %14[%c0] : tensor<1xf32>
    %15 = arith.muli %arg30, %arg11 : i32
    %16 = arith.addi %arg31, %arg25 : i32
    %17 = arith.index_cast %arg29 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%17], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %18 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %19 = arith.addi %17, %c1 : index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg8 to offset: [%19], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %20 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %21 = arith.subi %20, %18 : i32
    %reinterpret_cast_4 = memref.reinterpret_cast %arg9 to offset: [%17], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %22 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg9 to offset: [%19], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %23 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %24 = arith.subi %23, %22 : i32
    %25 = arith.cmpi sge, %16, %21 : i32
    scf.if %25 {
    } else {
      %26 = arith.subi %21, %16 : i32
      %27 = arith.minsi %26, %c1_i32 : i32
      %28 = arith.muli %arg30, %arg20 : i32
      %29 = arith.muli %15, %arg15 : i32
      %30 = arith.muli %22, %arg16 : i32
      %31 = arith.index_cast %30 : i32 to index
      %32 = arith.muli %arg30, %arg17 : i32
      %33 = arith.index_cast %32 : i32 to index
      %34 = arith.addi %31, %33 : index
      %35 = arith.index_cast %arg16 : i32 to index
      %36 = arith.index_cast %24 : i32 to index
      %37 = arith.index_cast %arg12 : i32 to index
      %38 = arith.muli %22, %arg18 : i32
      %39 = arith.index_cast %38 : i32 to index
      %40 = arith.muli %arg30, %arg19 : i32
      %41 = arith.index_cast %40 : i32 to index
      %42 = arith.addi %39, %41 : index
      %43 = arith.index_cast %arg18 : i32 to index
      %44 = tensor.empty() : tensor<64xi32>
      %45 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%44 : tensor<64xi32>) {
      ^bb0(%out: i32):
        %48 = linalg.index 0 : index
        %49 = arith.index_cast %48 : index to i32
        linalg.yield %49 : i32
      } -> tensor<64xi32>
      %46 = linalg.fill ins(%extracted : f32) outs(%2 : tensor<16x64xf32>) -> tensor<16x64xf32>
      %47 = arith.muli %15, %arg23 : i32
      scf.for %arg32 = %c0_i32 to %27 step %c1_i32  : i32 {
        %48 = arith.addi %16, %arg32 : i32
        %49 = arith.addi %18, %48 : i32
        %50 = arith.muli %49, %arg21 : i32
        %51 = arith.index_cast %50 : i32 to index
        %52 = arith.index_cast %28 : i32 to index
        %53 = arith.addi %51, %52 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%53], sizes: [8], strides: [1] : memref<?xi32> to memref<8xi32, strided<[1], offset: ?>>
        %alloc = memref.alloc() : memref<8xi32>
        linalg.fill ins(%c-1_i32 : i32) outs(%alloc : memref<8xi32>)
        %subview = memref.subview %reinterpret_cast_6[0] [5] [1] : memref<8xi32, strided<[1], offset: ?>> to memref<5xi32, strided<[1], offset: ?>>
        %subview_7 = memref.subview %alloc[0] [5] [1] : memref<8xi32> to memref<5xi32, strided<[1]>>
        memref.copy %subview, %subview_7 : memref<5xi32, strided<[1], offset: ?>> to memref<5xi32, strided<[1]>>
        %54 = bufferization.to_tensor %alloc restrict writable : memref<8xi32>
        %55 = arith.cmpi sge, %54, %12 : tensor<8xi32>
        %56 = arith.divsi %48, %c64_i32 : i32
        %57 = linalg.fill ins(%56 : i32) outs(%11 : tensor<8xi32>) -> tensor<8xi32>
        %58 = arith.cmpi sle, %54, %57 : tensor<8xi32>
        %59 = arith.andi %55, %58 : tensor<8xi1>
        %60 = arith.select %59, %13, %12 : tensor<8xi1>, tensor<8xi32>
        %61 = bufferization.alloc_tensor() : tensor<i32>
        %62 = linalg.fill ins(%c0_i32 : i32) outs(%61 : tensor<i32>) -> tensor<i32>
        %reduced = linalg.reduce ins(%60 : tensor<8xi32>) outs(%62 : tensor<i32>) dimensions = [0]
          (%in: i32, %init: i32) {
            %108 = arith.addi %in, %init : i32
            linalg.yield %108 : i32
          }
        %extracted_8 = tensor.extract %reduced[] : tensor<i32>
        %63 = arith.muli %49, %arg14 : i32
        %64 = arith.index_cast %63 : i32 to index
        %65 = arith.index_cast %29 : i32 to index
        %66 = arith.addi %64, %65 : index
        %67 = arith.index_cast %arg15 : i32 to index
        %68 = arith.index_cast %arg11 : i32 to index
        %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%66], sizes: [16, 128], strides: [%67, 1] : memref<?xf16> to memref<16x128xf16, strided<[?, 1], offset: ?>>
        %alloc_10 = memref.alloc() : memref<16x128xf16>
        %69 = arith.divsi %c0, %67 : index
        %70 = arith.subi %68, %69 : index
        %71 = arith.maxsi %70, %c0 : index
        %72 = arith.minsi %71, %c16 : index
        %73 = arith.remsi %c0, %67 : index
        %74 = arith.subi %37, %73 : index
        %75 = arith.maxsi %74, %c0 : index
        %76 = arith.minsi %75, %c128 : index
        %77 = arith.cmpi slt, %72, %c16 : index
        %78 = arith.cmpi slt, %76, %c128 : index
        %79 = arith.ori %77, %78 : i1
        scf.if %79 {
          linalg.fill ins(%cst : f16) outs(%alloc_10 : memref<16x128xf16>)
        } {hivm.unlikely_condition}
        %subview_11 = memref.subview %reinterpret_cast_9[0, 0] [%72, %76] [1, 1] : memref<16x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_12 = memref.subview %alloc_10[0, 0] [%72, %76] [1, 1] : memref<16x128xf16> to memref<?x?xf16, strided<[128, 1]>>
        memref.copy %subview_11, %subview_12 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
        %80 = bufferization.to_tensor %alloc_10 restrict writable : memref<16x128xf16>
        %81 = linalg.fill ins(%48 : i32) outs(%44 : tensor<64xi32>) -> tensor<64xi32>
        %82:3 = scf.for %arg33 = %c0_i32 to %extracted_8 step %c1_i32 iter_args(%arg34 = %8, %arg35 = %10, %arg36 = %10) -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)  : i32 {
          %108 = arith.index_cast %arg33 : i32 to index
          %109 = arith.addi %53, %108 : index
          %reinterpret_cast_18 = memref.reinterpret_cast %arg5 to offset: [%109], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
          %110 = memref.load %reinterpret_cast_18[%c0] : memref<1xi32, strided<[1], offset: ?>>
          %111 = arith.muli %110, %c64_i32 : i32
          %112 = arith.index_cast %111 : i32 to index
          %113 = arith.muli %112, %35 : index
          %114 = arith.addi %113, %34 : index
          %reinterpret_cast_19 = memref.reinterpret_cast %arg3 to offset: [%114], sizes: [64, 128], strides: [%35, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
          %alloc_20 = memref.alloc() : memref<64x128xf16>
          %115 = arith.divsi %113, %35 : index
          %116 = arith.subi %36, %115 : index
          %117 = arith.maxsi %116, %c0 : index
          %118 = arith.minsi %117, %c64 : index
          %119 = arith.remsi %113, %35 : index
          %120 = arith.subi %37, %119 : index
          %121 = arith.maxsi %120, %c0 : index
          %122 = arith.minsi %121, %c128 : index
          %123 = arith.cmpi slt, %118, %c64 : index
          %124 = arith.cmpi slt, %122, %c128 : index
          %125 = arith.ori %123, %124 : i1
          scf.if %125 {
            linalg.fill ins(%cst : f16) outs(%alloc_20 : memref<64x128xf16>)
          } {hivm.unlikely_condition}
          %subview_21 = memref.subview %reinterpret_cast_19[0, 0] [%118, %122] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %subview_22 = memref.subview %alloc_20[0, 0] [%118, %122] [1, 1] : memref<64x128xf16> to memref<?x?xf16, strided<[128, 1]>>
          memref.copy %subview_21, %subview_22 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
          %126 = bufferization.to_tensor %alloc_20 restrict writable : memref<64x128xf16>
          %127 = tensor.empty() : tensor<128x64xf16>
          %transposed = linalg.transpose ins(%126 : tensor<64x128xf16>) outs(%127 : tensor<128x64xf16>) permutation = [1, 0]
          %128 = linalg.fill ins(%111 : i32) outs(%44 : tensor<64xi32>) -> tensor<64xi32>
          %129 = arith.addi %128, %45 : tensor<64xi32>
          %130 = arith.cmpi sge, %81, %129 : tensor<64xi32>
          %expanded = tensor.expand_shape %130 [[0, 1]] output_shape [1, 64] : tensor<64xi1> into tensor<1x64xi1>
          %131 = arith.select %expanded, %6, %5 : tensor<1x64xi1>, tensor<1x64xf32>
          %132 = arith.addf %131, %6 : tensor<1x64xf32>
          %collapsed = tensor.collapse_shape %132 [[0, 1]] : tensor<1x64xf32> into tensor<64xf32>
          %broadcasted_23 = linalg.broadcast ins(%collapsed : tensor<64xf32>) outs(%2 : tensor<16x64xf32>) dimensions = [0]
          %133 = linalg.matmul {input_precison = "ieee"} ins(%80, %transposed : tensor<16x128xf16>, tensor<128x64xf16>) outs(%3 : tensor<16x64xf32>) -> tensor<16x64xf32>
          %134 = arith.mulf %133, %46 : tensor<16x64xf32>
          %135 = arith.addf %broadcasted_23, %134 : tensor<16x64xf32>
          %reduced_24 = linalg.reduce ins(%135 : tensor<16x64xf32>) outs(%10 : tensor<16xf32>) dimensions = [1]
            (%in: f32, %init: f32) {
              %164 = arith.maxnumf %in, %init : f32
              linalg.yield %164 : f32
            }
          %136 = arith.maxnumf %arg35, %reduced_24 : tensor<16xf32>
          %broadcasted_25 = linalg.broadcast ins(%136 : tensor<16xf32>) outs(%2 : tensor<16x64xf32>) dimensions = [1]
          %137 = arith.subf %135, %broadcasted_25 : tensor<16x64xf32>
          %138 = math.exp2 %137 : tensor<16x64xf32>
          %139 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<16xf32>) -> tensor<16xf32>
          %reduced_26 = linalg.reduce ins(%138 : tensor<16x64xf32>) outs(%139 : tensor<16xf32>) dimensions = [1]
            (%in: f32, %init: f32) {
              %164 = arith.addf %in, %init : f32
              linalg.yield %164 : f32
            }
          %140 = arith.subf %arg35, %136 : tensor<16xf32>
          %141 = math.exp2 %140 : tensor<16xf32>
          %broadcasted_27 = linalg.broadcast ins(%141 : tensor<16xf32>) outs(%7 : tensor<16x128xf32>) dimensions = [1]
          %142 = arith.mulf %arg34, %broadcasted_27 : tensor<16x128xf32>
          %143 = arith.muli %112, %43 : index
          %144 = arith.addi %143, %42 : index
          %reinterpret_cast_28 = memref.reinterpret_cast %arg4 to offset: [%144], sizes: [64, 128], strides: [%43, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
          %alloc_29 = memref.alloc() : memref<64x128xf16>
          %145 = arith.divsi %143, %43 : index
          %146 = arith.subi %36, %145 : index
          %147 = arith.maxsi %146, %c0 : index
          %148 = arith.minsi %147, %c64 : index
          %149 = arith.remsi %143, %43 : index
          %150 = arith.subi %37, %149 : index
          %151 = arith.maxsi %150, %c0 : index
          %152 = arith.minsi %151, %c128 : index
          %153 = arith.cmpi slt, %148, %c64 : index
          %154 = arith.cmpi slt, %152, %c128 : index
          %155 = arith.ori %153, %154 : i1
          scf.if %155 {
            linalg.fill ins(%cst : f16) outs(%alloc_29 : memref<64x128xf16>)
          } {hivm.unlikely_condition}
          %subview_30 = memref.subview %reinterpret_cast_28[0, 0] [%148, %152] [1, 1] : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %subview_31 = memref.subview %alloc_29[0, 0] [%148, %152] [1, 1] : memref<64x128xf16> to memref<?x?xf16, strided<[128, 1]>>
          memref.copy %subview_30, %subview_31 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
          %156 = bufferization.to_tensor %alloc_29 restrict writable : memref<64x128xf16>
          %157 = arith.truncf %138 : tensor<16x64xf32> to tensor<16x64xf16>
          %158 = linalg.matmul {input_precison = "ieee"} ins(%157, %156 : tensor<16x64xf16>, tensor<64x128xf16>) outs(%142 : tensor<16x128xf32>) -> tensor<16x128xf32>
          %159 = arith.subf %arg36, %136 : tensor<16xf32>
          %160 = math.exp2 %159 : tensor<16xf32>
          %161 = arith.addf %160, %reduced_26 : tensor<16xf32>
          %162 = math.log2 %161 : tensor<16xf32>
          %163 = arith.addf %136, %162 : tensor<16xf32>
          scf.yield %158, %136, %163 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>
        }
        %83 = arith.subf %82#1, %82#2 : tensor<16xf32>
        %84 = math.exp2 %83 : tensor<16xf32>
        %broadcasted = linalg.broadcast ins(%84 : tensor<16xf32>) outs(%7 : tensor<16x128xf32>) dimensions = [1]
        %85 = arith.mulf %82#0, %broadcasted : tensor<16x128xf32>
        %86 = arith.muli %49, %arg22 : i32
        %87 = arith.index_cast %86 : i32 to index
        %88 = arith.index_cast %47 : i32 to index
        %89 = arith.addi %87, %88 : index
        %90 = arith.index_cast %arg23 : i32 to index
        %reinterpret_cast_13 = memref.reinterpret_cast %arg6 to offset: [%89], sizes: [16, 128], strides: [%90, 1] : memref<?xf16> to memref<16x128xf16, strided<[?, 1], offset: ?>>
        %91 = arith.truncf %85 : tensor<16x128xf32> to tensor<16x128xf16>
        %92 = arith.divsi %c0, %90 : index
        %93 = arith.subi %68, %92 : index
        %94 = arith.maxsi %93, %c0 : index
        %95 = arith.minsi %94, %c16 : index
        %96 = arith.remsi %c0, %90 : index
        %97 = arith.subi %37, %96 : index
        %98 = arith.maxsi %97, %c0 : index
        %99 = arith.minsi %98, %c128 : index
        %extracted_slice = tensor.extract_slice %91[0, 0] [%95, %99] [1, 1] : tensor<16x128xf16> to tensor<?x?xf16>
        %subview_14 = memref.subview %reinterpret_cast_13[0, 0] [%95, %99] [1, 1] : memref<16x128xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview_14 : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
        %100 = arith.index_cast %49 : i32 to index
        %101 = arith.index_cast %15 : i32 to index
        %102 = arith.index_cast %arg24 : i32 to index
        %103 = arith.muli %101, %102 : index
        %104 = arith.addi %100, %103 : index
        %reinterpret_cast_15 = memref.reinterpret_cast %arg7 to offset: [%104], sizes: [16], strides: [%102] : memref<?xf32> to memref<16xf32, strided<[?], offset: ?>>
        %105 = arith.index_cast %arg11 : i32 to index
        %106 = arith.maxsi %105, %c0 : index
        %107 = arith.minsi %106, %c16 : index
        %extracted_slice_16 = tensor.extract_slice %82#2[0] [%107] [1] : tensor<16xf32> to tensor<?xf32>
        %subview_17 = memref.subview %reinterpret_cast_15[0] [%107] [1] : memref<16xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice_16 in writable %subview_17 : (tensor<?xf32>, memref<?xf32, strided<[?], offset: ?>>) -> ()
      }
    }
    return
  }
}
