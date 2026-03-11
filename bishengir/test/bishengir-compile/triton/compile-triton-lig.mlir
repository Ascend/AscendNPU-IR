// RUN: bishengir-compile %s --target=Ascend910_9589 --enable-auto-multi-buffer=false --enable-auto-bind-sub-block=true --disable-ffts --enable-flatten=false --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @lightning_indexer_grad_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg11: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg12: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg13: memref<?xf32> {tt.divisibility = 16 : i32}, %arg14: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32 {tt.divisibility = 16 : i32}, %arg38: i32 {tt.divisibility = 16 : i32}, %arg39: i32 {tt.divisibility = 16 : i32}, %arg40: i32 {tt.divisibility = 16 : i32}, %arg41: i32 {tt.divisibility = 16 : i32}, %arg42: i32, %arg43: i32, %arg44: i32, %arg45: i32, %arg46: i32, %arg47: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c2048 = arith.constant 2048 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c1_i32 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = tensor.empty() : tensor<64x128xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
    %4 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1]>>
    %5 = arith.extsi %arg45 : i32 to i64
    %6 = arith.cmpi sge, %5, %4 : i64
    scf.if %6 {
    } else {
      %reinterpret_cast_1 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
      %7 = memref.load %reinterpret_cast_1[%c0] : memref<1xi64, strided<[1]>>
      %8 = arith.muli %arg45, %arg20 : i32
      %9 = arith.muli %arg45, %arg15 : i32
      %10 = arith.muli %arg46, %arg16 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.muli %arg45, %arg21 : i32
      %13 = arith.muli %arg45, %arg22 : i32
      %14 = arith.muli %arg46, %arg23 : i32
      %15 = arith.addi %13, %14 : i32
      %16 = arith.muli %arg45, %arg24 : i32
      %17 = arith.muli %arg46, %arg25 : i32
      %18 = arith.addi %16, %17 : i32
      %19 = arith.muli %arg45, %arg29 : i32
      %20 = arith.muli %arg46, %arg30 : i32
      %21 = arith.addi %19, %20 : i32
      %22 = arith.muli %arg46, %arg19 : i32
      %23 = arith.muli %arg46, %arg28 : i32
      %24 = arith.muli %arg45, %arg31 : i32
      %25 = arith.muli %arg46, %arg32 : i32
      %26 = arith.addi %24, %25 : i32
      %27 = arith.muli %arg45, %arg37 : i32
      %28 = arith.muli %arg46, %arg38 : i32
      %29 = arith.addi %27, %28 : i32
      %30 = arith.index_cast %8 : i32 to index
      %31 = scf.for %arg48 = %c0_i32 to %c2048_i32 step %c1_i32 iter_args(%arg49 = %c0_i32) -> (i32)  : i32 {
        %33 = arith.index_cast %arg48 : i32 to index
        %34 = arith.addi %30, %33 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%34], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %35 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>>
        %36 = arith.cmpi sge, %35, %c0_i32 : i32
        %37 = scf.if %36 -> (i32) {
          %38 = arith.addi %arg49, %c1_i32 : i32
          scf.yield %38 : i32
        } else {
          scf.yield %arg49 : i32
        }
        scf.yield %37 : i32
      }
      %32 = arith.cmpi eq, %31, %c0_i32 : i32
      scf.if %32 {
      } else {
        %33 = arith.addi %31, %c63_i32 : i32
        %34 = arith.divsi %33, %c64_i32 : i32
        %35:2 = scf.for %arg48 = %c0_i32 to %c2048_i32 step %c1_i32 iter_args(%arg49 = %c0_i32, %arg50 = %c0_i32) -> (i32, i32)  : i32 {
          %43 = arith.index_cast %arg48 : i32 to index
          %44 = arith.addi %30, %43 : index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%44], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
          %45 = memref.load %reinterpret_cast_6[%c0] : memref<1xi32, strided<[1], offset: ?>>
          %46 = arith.cmpi sge, %45, %c0_i32 : i32
          %47:2 = scf.if %46 -> (i32, i32) {
            %48 = arith.extsi %45 : i32 to i64
            %49 = arith.cmpi slt, %48, %7 : i64
            %50 = arith.muli %45, %arg18 : i32
            %51 = arith.addi %22, %50 : i32
            %52 = arith.muli %arg50, %arg33 : i32
            %53 = arith.addi %26, %52 : i32
            scf.for %arg51 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
              %56 = arith.muli %arg51, %c32_i32 : i32
              %57 = arith.index_cast %51 : i32 to index
              %58 = arith.index_cast %56 : i32 to index
              %59 = arith.addi %57, %58 : index
              %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%59], sizes: [1, 32], strides: [32, 1] : memref<?xf16> to memref<1x32xf16, strided<[32, 1], offset: ?>>
              %alloc = memref.alloc() : memref<1x32xf16>
              %60 = arith.index_castui %49 : i1 to index
              %61 = arith.muli %60, %c32 : index
              %62 = arith.addi %58, %c32 : index
              %63 = arith.maxsi %58, %c128 : index
              %64 = arith.minsi %62, %63 : index
              %65 = arith.subi %64, %58 : index
              %66 = arith.minsi %60, %c1 : index
              %67 = arith.minsi %61, %65 : index
              %68 = arith.cmpi slt, %66, %c1 : index
              %69 = arith.cmpi slt, %67, %c32 : index
              %70 = arith.ori %68, %69 : i1
              scf.if %70 {
                linalg.fill ins(%cst : f16) outs(%alloc : memref<1x32xf16>)
              } {hivm.unlikely_condition}
              %subview = memref.subview %reinterpret_cast_7[0, 0] [%66, %67] [1, 1] : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1], offset: ?>>
              %subview_8 = memref.subview %alloc[0, 0] [%66, %67] [1, 1] : memref<1x32xf16> to memref<?x?xf16, strided<[32, 1]>>
              memref.copy %subview, %subview_8 : memref<?x?xf16, strided<[32, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1]>>
              %71 = bufferization.to_tensor %alloc restrict writable : memref<1x32xf16>
              %72 = arith.index_cast %53 : i32 to index
              %73 = arith.addi %72, %58 : index
              %reinterpret_cast_9 = memref.reinterpret_cast %arg12 to offset: [%73], sizes: [1, 32], strides: [32, 1] : memref<?xf16> to memref<1x32xf16, strided<[32, 1], offset: ?>>
              %extracted_slice = tensor.extract_slice %71[0, 0] [1, %65] [1, 1] : tensor<1x32xf16> to tensor<1x?xf16>
              %subview_10 = memref.subview %reinterpret_cast_9[0, 0] [1, %65] [1, 1] : memref<1x32xf16, strided<[32, 1], offset: ?>> to memref<1x?xf16, strided<[32, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview_10 : (tensor<1x?xf16>, memref<1x?xf16, strided<[32, 1], offset: ?>>) -> ()
            }
            %54 = arith.addi %arg49, %c1_i32 : i32
            %55 = arith.addi %arg50, %c1_i32 : i32
            scf.yield %54, %55 : i32, i32
          } else {
            scf.yield %arg49, %arg50 : i32, i32
          }
          scf.yield %47#0, %47#1 : i32, i32
        }
        %36 = arith.index_cast %11 : i32 to index
        %37 = arith.index_cast %arg17 : i32 to index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [64, 128], strides: [%37, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
        %38 = arith.index_cast %15 : i32 to index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%38], sizes: [64, 1], strides: [1, 1] : memref<?xf16> to memref<64x1xf16, strided<[1, 1], offset: ?>>
        %39 = arith.index_cast %21 : i32 to index
        %reinterpret_cast_4 = memref.reinterpret_cast %arg11 to offset: [%39], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
        %40 = arith.index_cast %18 : i32 to index
        %41 = arith.index_cast %arg26 : i32 to index
        %reinterpret_cast_5 = memref.reinterpret_cast %arg9 to offset: [%40], sizes: [64, 128], strides: [%41, 1] : memref<?xf32> to memref<64x128xf32, strided<[?, 1], offset: ?>>
        scf.for %arg48 = %c0_i32 to %34 step %c1_i32  : i32 {
          %43 = arith.muli %arg48, %c64_i32 : i32
          %alloc = memref.alloc() : memref<64x128xf16>
          memref.copy %reinterpret_cast_2, %alloc : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<64x128xf16>
          %44 = bufferization.to_tensor %alloc restrict writable : memref<64x128xf16>
          %45 = arith.index_cast %26 : i32 to index
          %46 = arith.index_cast %43 : i32 to index
          %47 = arith.index_cast %arg33 : i32 to index
          %48 = arith.muli %46, %47 : index
          %49 = arith.addi %45, %48 : index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg12 to offset: [%49], sizes: [64, 128], strides: [%47, 1] : memref<?xf16> to memref<64x128xf16, strided<[?, 1], offset: ?>>
          %alloc_7 = memref.alloc() : memref<64x128xf16>
          memref.copy %reinterpret_cast_6, %alloc_7 : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<64x128xf16>
          %50 = bufferization.to_tensor %alloc_7 restrict writable : memref<64x128xf16>
          %51 = tensor.empty() : tensor<128x64xf16>
          %transposed = linalg.transpose ins(%50 : tensor<64x128xf16>) outs(%51 : tensor<128x64xf16>) permutation = [1, 0]
          %52 = linalg.matmul {input_precison = "ieee"} ins(%44, %transposed : tensor<64x128xf16>, tensor<128x64xf16>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
          %alloc_8 = memref.alloc() : memref<64x1xf16>
          memref.copy %reinterpret_cast_3, %alloc_8 : memref<64x1xf16, strided<[1, 1], offset: ?>> to memref<64x1xf16>
          %53 = bufferization.to_tensor %alloc_8 restrict writable : memref<64x1xf16>
          %54 = arith.index_cast %12 : i32 to index
          %55 = arith.addi %54, %46 : index
          %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%55], sizes: [1, 64], strides: [64, 1] : memref<?xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
          %alloc_10 = memref.alloc() : memref<1x64xf16>
          %56 = arith.addi %46, %c64 : index
          %57 = arith.index_cast %7 : i64 to index
          %58 = arith.maxsi %46, %57 : index
          %59 = arith.minsi %56, %58 : index
          %60 = arith.subi %59, %46 : index
          %61 = arith.cmpi slt, %60, %c64 : index
          scf.if %61 {
            linalg.fill ins(%cst : f16) outs(%alloc_10 : memref<1x64xf16>)
          } {hivm.unlikely_condition}
          %subview = memref.subview %reinterpret_cast_9[0, 0] [1, %60] [1, 1] : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x?xf16, strided<[64, 1], offset: ?>>
          %subview_11 = memref.subview %alloc_10[0, 0] [1, %60] [1, 1] : memref<1x64xf16> to memref<1x?xf16, strided<[64, 1]>>
          memref.copy %subview, %subview_11 : memref<1x?xf16, strided<[64, 1], offset: ?>> to memref<1x?xf16, strided<[64, 1]>>
          %62 = bufferization.to_tensor %alloc_10 restrict writable : memref<1x64xf16>
          %63 = linalg.matmul {input_precison = "ieee"} ins(%53, %62 : tensor<64x1xf16>, tensor<1x64xf16>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
          %64 = arith.cmpf oge, %52, %1 : tensor<64x64xf32>
          %65 = arith.select %64, %63, %1 : tensor<64x64xi1>, tensor<64x64xf32>
          %66 = arith.select %64, %52, %1 : tensor<64x64xi1>, tensor<64x64xf32>
          %alloc_12 = memref.alloc() : memref<1x64xf16>
          %67 = arith.maxsi %46, %c2048 : index
          %68 = arith.minsi %56, %67 : index
          %69 = arith.subi %68, %46 : index
          %70 = arith.cmpi slt, %69, %c64 : index
          scf.if %70 {
            linalg.fill ins(%cst : f16) outs(%alloc_12 : memref<1x64xf16>)
          } {hivm.unlikely_condition}
          %subview_13 = memref.subview %reinterpret_cast_9[0, 0] [1, %69] [1, 1] : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x?xf16, strided<[64, 1], offset: ?>>
          %subview_14 = memref.subview %alloc_12[0, 0] [1, %69] [1, 1] : memref<1x64xf16> to memref<1x?xf16, strided<[64, 1]>>
          memref.copy %subview_13, %subview_14 : memref<1x?xf16, strided<[64, 1], offset: ?>> to memref<1x?xf16, strided<[64, 1]>>
          %71 = bufferization.to_tensor %alloc_12 restrict writable : memref<1x64xf16>
          %72 = arith.extf %71 : tensor<1x64xf16> to tensor<1x64xf32>
          %collapsed = tensor.collapse_shape %72 [[0, 1]] : tensor<1x64xf32> into tensor<64xf32>
          %broadcasted = linalg.broadcast ins(%collapsed : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [0]
          %73 = arith.mulf %66, %broadcasted : tensor<64x64xf32>
          %74 = tensor.empty() : tensor<64xf32>
          %75 = linalg.fill ins(%cst_0 : f32) outs(%74 : tensor<64xf32>) -> tensor<64xf32>
          %reduced = linalg.reduce ins(%73 : tensor<64x64xf32>) outs(%75 : tensor<64xf32>) dimensions = [1]
            (%in: f32, %init: f32) {
              %86 = arith.addf %in, %init : f32
              linalg.yield %86 : f32
            }
          hivm.hir.store ins(%reduced : tensor<64xf32>) outs(%reinterpret_cast_4 : memref<64xf32, strided<[1], offset: ?>>) atomic = <add>
          %76 = arith.truncf %65 : tensor<64x64xf32> to tensor<64x64xf16>
          %alloc_15 = memref.alloc() : memref<64x128xf16>
          memref.copy %reinterpret_cast_6, %alloc_15 : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<64x128xf16>
          %77 = bufferization.to_tensor %alloc_15 restrict writable : memref<64x128xf16>
          %78 = linalg.matmul {input_precison = "ieee"} ins(%76, %77 : tensor<64x64xf16>, tensor<64x128xf16>) outs(%3 : tensor<64x128xf32>) -> tensor<64x128xf32>
          hivm.hir.store ins(%78 : tensor<64x128xf32>) outs(%reinterpret_cast_5 : memref<64x128xf32, strided<[?, 1], offset: ?>>) atomic = <add>
          %alloc_16 = memref.alloc() : memref<64x128xf16>
          memref.copy %reinterpret_cast_2, %alloc_16 : memref<64x128xf16, strided<[?, 1], offset: ?>> to memref<64x128xf16>
          %79 = bufferization.to_tensor %alloc_16 restrict writable : memref<64x128xf16>
          %80 = tensor.empty() : tensor<64x64xf16>
          %transposed_17 = linalg.transpose ins(%76 : tensor<64x64xf16>) outs(%80 : tensor<64x64xf16>) permutation = [1, 0]
          %81 = linalg.matmul {input_precison = "ieee"} ins(%transposed_17, %79 : tensor<64x64xf16>, tensor<64x128xf16>) outs(%3 : tensor<64x128xf32>) -> tensor<64x128xf32>
          %82 = arith.index_cast %29 : i32 to index
          %83 = arith.index_cast %arg39 : i32 to index
          %84 = arith.muli %46, %83 : index
          %85 = arith.addi %82, %84 : index
          %reinterpret_cast_18 = memref.reinterpret_cast %arg14 to offset: [%85], sizes: [64, 128], strides: [%83, 1] : memref<?xf32> to memref<64x128xf32, strided<[?, 1], offset: ?>>
          bufferization.materialize_in_destination %81 in writable %reinterpret_cast_18 : (tensor<64x128xf32>, memref<64x128xf32, strided<[?, 1], offset: ?>>) -> ()
        }
        %42 = scf.for %arg48 = %c0_i32 to %c2048_i32 step %c1_i32 iter_args(%arg49 = %c0_i32) -> (i32)  : i32 {
          %43 = arith.index_cast %arg48 : i32 to index
          %44 = arith.addi %30, %43 : index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%44], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
          %45 = memref.load %reinterpret_cast_6[%c0] : memref<1xi32, strided<[1], offset: ?>>
          %46 = arith.cmpi sge, %45, %c0_i32 : i32
          %47 = scf.if %46 -> (i32) {
            %48 = arith.muli %arg49, %arg39 : i32
            %49 = arith.addi %29, %48 : i32
            %50 = arith.muli %45, %arg27 : i32
            %51 = arith.addi %23, %50 : i32
            scf.for %arg50 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
              %53 = arith.muli %arg50, %c32_i32 : i32
              %54 = arith.index_cast %49 : i32 to index
              %55 = arith.index_cast %53 : i32 to index
              %56 = arith.addi %54, %55 : index
              %reinterpret_cast_7 = memref.reinterpret_cast %arg14 to offset: [%56], sizes: [1, 32], strides: [32, 1] : memref<?xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
              %alloc = memref.alloc() : memref<1x32xf32>
              memref.copy %reinterpret_cast_7, %alloc : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32>
              %57 = arith.index_cast %51 : i32 to index
              %58 = arith.addi %57, %55 : index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg10 to offset: [%58], sizes: [1, 32], strides: [32, 1] : memref<?xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
              %cast = memref.cast %alloc : memref<1x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
              %59 = bufferization.to_tensor %cast restrict writable : memref<1x32xf32, strided<[32, 1], offset: ?>>
              hivm.hir.store ins(%59 : tensor<1x32xf32>) outs(%reinterpret_cast_8 : memref<1x32xf32, strided<[32, 1], offset: ?>>) atomic = <add>
            }
            %52 = arith.addi %arg49, %c1_i32 : i32
            scf.yield %52 : i32
          } else {
            scf.yield %arg49 : i32
          }
          scf.yield %47 : i32
        }
      }
    }
    return
  }
}
