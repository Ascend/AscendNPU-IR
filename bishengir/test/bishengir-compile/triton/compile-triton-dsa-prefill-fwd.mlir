// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 -o %t

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @dsa_prefill_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: f32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %c1024 = arith.constant 1024 : index
    %c16 = arith.constant 16 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 9.99999996E-13 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %c0_i8 = arith.constant 0 : i8
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_3 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %2 = tensor.empty() : tensor<16x16xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<16x16xi8>) -> tensor<16x16xi8>
    %4 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = tensor.empty() : tensor<16x128xf32>
    %6 = linalg.fill ins(%cst_3 : f32) outs(%5 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %7 = tensor.empty() : tensor<16xf32>
    %8 = linalg.fill ins(%cst_1 : f32) outs(%7 : tensor<16xf32>) -> tensor<16xf32>
    %9 = linalg.fill ins(%cst_2 : f32) outs(%7 : tensor<16xf32>) -> tensor<16xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<16xf32>) -> tensor<16xf32>
    %11 = arith.divsi %arg25, %c16_i32 : i32
    %12 = arith.remsi %arg25, %c16_i32 : i32
    %13 = arith.muli %11, %arg7 : i32
    %14 = arith.muli %12, %arg8 : i32
    %15 = arith.addi %13, %14 : i32
    %16 = arith.muli %11, %arg10 : i32
    %17 = arith.muli %12, %arg11 : i32
    %18 = arith.addi %16, %17 : i32
    %19 = linalg.fill ins(%arg21 : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %20 = arith.muli %11, %arg19 : i32
    %21 = arith.muli %11, %arg13 : i32
    %22 = arith.muli %12, %arg14 : i32
    %23 = arith.addi %21, %22 : i32
    %24 = arith.muli %11, %arg16 : i32
    %25 = arith.muli %12, %arg17 : i32
    %26 = arith.addi %24, %25 : i32
    scf.for %arg28 = %c0_i32 to %c1024_i32 step %c16_i32  : i32 {
      %27 = arith.index_cast %15 : i32 to index
      %28 = arith.index_cast %arg28 : i32 to index
      %29 = arith.index_cast %arg9 : i32 to index
      %30 = arith.muli %28, %29 : index
      %31 = arith.addi %27, %30 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [16, 192], strides: [%29, 1] : memref<?xbf16> to memref<16x192xbf16, strided<[?, 1], offset: ?>>
      %alloc = memref.alloc() : memref<16x192xbf16>
      %32 = arith.addi %28, %c16 : index
      %33 = arith.maxsi %28, %c1024 : index
      %34 = arith.minsi %32, %33 : index
      %35 = arith.subi %34, %28 : index
      %36 = arith.cmpi slt, %35, %c16 : index
      scf.if %36 {
        linalg.fill ins(%cst : bf16) outs(%alloc : memref<16x192xbf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast[0, 0] [%35, 192] [1, 1] : memref<16x192xbf16, strided<[?, 1], offset: ?>> to memref<?x192xbf16, strided<[?, 1], offset: ?>>
      %subview_4 = memref.subview %alloc[0, 0] [%35, 192] [1, 1] : memref<16x192xbf16> to memref<?x192xbf16, strided<[192, 1]>>
      memref.copy %subview, %subview_4 : memref<?x192xbf16, strided<[?, 1], offset: ?>> to memref<?x192xbf16, strided<[192, 1]>>
      %37 = bufferization.to_tensor %alloc restrict writable : memref<16x192xbf16>
      %38 = arith.addi %arg28, %c1_i32 : i32
      %39:2 = scf.for %arg29 = %c0_i32 to %38 step %c16_i32 iter_args(%arg30 = %6, %arg31 = %9) -> (tensor<16x128xf32>, tensor<16xf32>)  : i32 {
        %47 = arith.index_cast %18 : i32 to index
        %48 = arith.index_cast %arg29 : i32 to index
        %49 = arith.index_cast %arg12 : i32 to index
        %50 = arith.muli %48, %49 : index
        %51 = arith.addi %47, %50 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%51], sizes: [16, 192], strides: [%49, 1] : memref<?xbf16> to memref<16x192xbf16, strided<[?, 1], offset: ?>>
        %alloc_7 = memref.alloc() : memref<16x192xbf16>
        %52 = arith.addi %48, %c16 : index
        %53 = arith.maxsi %48, %c1024 : index
        %54 = arith.minsi %52, %53 : index
        %55 = arith.subi %54, %48 : index
        %56 = arith.cmpi slt, %55, %c16 : index
        scf.if %56 {
          linalg.fill ins(%cst : bf16) outs(%alloc_7 : memref<16x192xbf16>)
        } {hivm.unlikely_condition}
        %subview_8 = memref.subview %reinterpret_cast_6[0, 0] [%55, 192] [1, 1] : memref<16x192xbf16, strided<[?, 1], offset: ?>> to memref<?x192xbf16, strided<[?, 1], offset: ?>>
        %subview_9 = memref.subview %alloc_7[0, 0] [%55, 192] [1, 1] : memref<16x192xbf16> to memref<?x192xbf16, strided<[192, 1]>>
        memref.copy %subview_8, %subview_9 : memref<?x192xbf16, strided<[?, 1], offset: ?>> to memref<?x192xbf16, strided<[192, 1]>>
        %57 = bufferization.to_tensor %alloc_7 restrict writable : memref<16x192xbf16>
        %58 = tensor.empty() : tensor<192x16xbf16>
        %transposed = linalg.transpose ins(%57 : tensor<16x192xbf16>) outs(%58 : tensor<192x16xbf16>) permutation = [1, 0]
        %59 = linalg.matmul {input_precision = "ieee"} ins(%37, %transposed : tensor<16x192xbf16>, tensor<192x16xbf16>) outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
        %60 = arith.mulf %59, %19 : tensor<16x16xf32>
        %61 = arith.index_cast %20 : i32 to index
        %62 = arith.index_cast %arg20 : i32 to index
        %63 = arith.muli %28, %62 : index
        %64 = arith.addi %61, %63 : index
        %65 = arith.addi %64, %48 : index
        %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%65], sizes: [16, 16], strides: [%62, 1] : memref<?xi8> to memref<16x16xi8, strided<[?, 1], offset: ?>>
        %alloc_11 = memref.alloc() : memref<16x16xi8>
        %66 = arith.minsi %35, %c16 : index
        %67 = arith.minsi %55, %c16 : index
        %68 = arith.cmpi slt, %66, %c16 : index
        %69 = arith.cmpi slt, %67, %c16 : index
        %70 = arith.ori %68, %69 : i1
        scf.if %70 {
          linalg.fill ins(%c0_i8 : i8) outs(%alloc_11 : memref<16x16xi8>)
        } {hivm.unlikely_condition}
        %subview_12 = memref.subview %reinterpret_cast_10[0, 0] [%66, %67] [1, 1] : memref<16x16xi8, strided<[?, 1], offset: ?>> to memref<?x?xi8, strided<[?, 1], offset: ?>>
        %subview_13 = memref.subview %alloc_11[0, 0] [%66, %67] [1, 1] : memref<16x16xi8> to memref<?x?xi8, strided<[16, 1]>>
        memref.copy %subview_12, %subview_13 : memref<?x?xi8, strided<[?, 1], offset: ?>> to memref<?x?xi8, strided<[16, 1]>>
        %71 = bufferization.to_tensor %alloc_11 restrict writable : memref<16x16xi8>
        %72 = arith.cmpi ne, %71, %3 : tensor<16x16xi8>
        %73 = arith.select %72, %60, %4 : tensor<16x16xi1>, tensor<16x16xf32>
        %74 = arith.index_cast %23 : i32 to index
        %75 = arith.index_cast %arg15 : i32 to index
        %76 = arith.muli %48, %75 : index
        %77 = arith.addi %74, %76 : index
        %reinterpret_cast_14 = memref.reinterpret_cast %arg4 to offset: [%77], sizes: [16, 128], strides: [%75, 1] : memref<?xbf16> to memref<16x128xbf16, strided<[?, 1], offset: ?>>
        %alloc_15 = memref.alloc() : memref<16x128xbf16>
        scf.if %56 {
          linalg.fill ins(%cst : bf16) outs(%alloc_15 : memref<16x128xbf16>)
        } {hivm.unlikely_condition}
        %subview_16 = memref.subview %reinterpret_cast_14[0, 0] [%55, 128] [1, 1] : memref<16x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
        %subview_17 = memref.subview %alloc_15[0, 0] [%55, 128] [1, 1] : memref<16x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
        memref.copy %subview_16, %subview_17 : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
        %78 = bufferization.to_tensor %alloc_15 restrict writable : memref<16x128xbf16>
        %reduced = linalg.reduce ins(%73 : tensor<16x16xf32>) outs(%9 : tensor<16xf32>) dimensions = [1]
          (%in: f32, %init: f32) {
            %105 = arith.maxnumf %in, %init : f32
            linalg.yield %105 : f32
          }
        %broadcasted = linalg.broadcast ins(%reduced : tensor<16xf32>) outs(%0 : tensor<16x16xf32>) dimensions = [1]
        %79 = arith.subf %73, %broadcasted : tensor<16x16xf32>
        %80 = math.exp %79 : tensor<16x16xf32>
        %81 = linalg.fill ins(%cst_3 : f32) outs(%7 : tensor<16xf32>) -> tensor<16xf32>
        %reduced_18 = linalg.reduce ins(%80 : tensor<16x16xf32>) outs(%81 : tensor<16xf32>) dimensions = [1]
          (%in: f32, %init: f32) {
            %105 = arith.addf %in, %init : f32
            linalg.yield %105 : f32
          }
        %82 = math.log %reduced_18 : tensor<16xf32>
        %83 = arith.addf %reduced, %82 : tensor<16xf32>
        %84 = math.exp %arg31 : tensor<16xf32>
        %85 = arith.addf %83, %8 : tensor<16xf32>
        %86 = math.exp %85 : tensor<16xf32>
        %87 = arith.addf %84, %86 : tensor<16xf32>
        %88 = math.log %87 : tensor<16xf32>
        %89 = arith.cmpf une, %88, %88 : tensor<16xf32>
        %90 = arith.select %89, %arg31, %88 : tensor<16xi1>, tensor<16xf32>
        %91 = arith.subf %arg31, %90 : tensor<16xf32>
        %92 = math.exp %91 : tensor<16xf32>
        %93 = arith.cmpf oeq, %reduced, %9 : tensor<16xf32>
        %94 = arith.select %93, %10, %92 : tensor<16xi1>, tensor<16xf32>
        %broadcasted_19 = linalg.broadcast ins(%94 : tensor<16xf32>) outs(%5 : tensor<16x128xf32>) dimensions = [1]
        %95 = arith.mulf %arg30, %broadcasted_19 : tensor<16x128xf32>
        %96 = arith.subf %83, %90 : tensor<16xf32>
        %97 = math.exp %96 : tensor<16xf32>
        %broadcasted_20 = linalg.broadcast ins(%reduced_18 : tensor<16xf32>) outs(%0 : tensor<16x16xf32>) dimensions = [1]
        %98 = arith.divf %80, %broadcasted_20 : tensor<16x16xf32>
        %99 = arith.truncf %98 : tensor<16x16xf32> to tensor<16x16xbf16>
        %100 = linalg.matmul {input_precision = "ieee"} ins(%99, %78 : tensor<16x16xbf16>, tensor<16x128xbf16>) outs(%6 : tensor<16x128xf32>) -> tensor<16x128xf32>
        %broadcasted_21 = linalg.broadcast ins(%97 : tensor<16xf32>) outs(%5 : tensor<16x128xf32>) dimensions = [1]
        %101 = arith.mulf %100, %broadcasted_21 : tensor<16x128xf32>
        %102 = arith.cmpf une, %101, %101 : tensor<16x128xf32>
        %103 = arith.select %102, %6, %101 : tensor<16x128xi1>, tensor<16x128xf32>
        %104 = arith.addf %95, %103 : tensor<16x128xf32>
        scf.yield %104, %90 : tensor<16x128xf32>, tensor<16xf32>
      }
      %40 = arith.cmpf une, %39#0, %39#0 : tensor<16x128xf32>
      %41 = arith.select %40, %6, %39#0 : tensor<16x128xi1>, tensor<16x128xf32>
      %42 = arith.index_cast %26 : i32 to index
      %43 = arith.index_cast %arg18 : i32 to index
      %44 = arith.muli %28, %43 : index
      %45 = arith.addi %42, %44 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%45], sizes: [16, 128], strides: [%43, 1] : memref<?xbf16> to memref<16x128xbf16, strided<[?, 1], offset: ?>>
      %46 = arith.truncf %41 : tensor<16x128xf32> to tensor<16x128xbf16>
      bufferization.materialize_in_destination %46 in writable %reinterpret_cast_5 : (tensor<16x128xbf16>, memref<16x128xbf16, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }
}
