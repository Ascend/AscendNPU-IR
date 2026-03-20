// RUN: bishengir-compile %s  --target=Ascend910_9589 --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --disable-ffts --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @sparse_flash_attention_prefill_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32}, %arg5: memref<?xi32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xbf16> {tt.divisibility = 16 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32}, %arg10: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg11: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: f32, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i32, %arg37: i32, %arg38: i32, %arg39: i32, %arg40: i32, %arg41: i32, %arg42: i32, %arg43: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c256_i32 = arith.constant 256 : i32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %c1024 = arith.constant 1024 : index
    %c128 = arith.constant 128 : index
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<16x128xf32>
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %2 = tensor.empty() : tensor<16xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<16xf32>) -> tensor<16xf32>
    %4 = tensor.empty() : tensor<16x256xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<16x256xf32>) -> tensor<16x256xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %7 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<16xf32>) -> tensor<16xf32>
    %8 = arith.muli %arg41, %arg37 : i32
    %9 = arith.addi %arg41, %c1_i32 : i32
    %10 = arith.muli %9, %arg37 : i32
    %11 = arith.minsi %arg33, %10 : i32
    %12 = arith.muli %arg41, %arg28 : i32
    %13 = arith.muli %arg41, %arg30 : i32
    %14 = linalg.fill ins(%arg32 : f32) outs(%0 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %15 = arith.index_cast %12 : i32 to index
    %16 = arith.index_cast %arg29 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg10 to offset: [%15], sizes: [16, 128], strides: [%16, 1] : memref<?xbf16> to memref<16x128xbf16, strided<[?, 1], offset: ?>>
    %17 = arith.index_cast %13 : i32 to index
    %18 = arith.index_cast %arg31 : i32 to index
    scf.for %arg44 = %8 to %11 step %c1_i32  : i32 {
      %19 = arith.muli %arg44, %arg12 : i32
      %20 = arith.muli %arg44, %arg23 : i32
      %21 = arith.muli %arg44, %arg21 : i32
      %22 = arith.index_cast %19 : i32 to index
      %23 = arith.index_cast %arg14 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [16, 576], strides: [%23, 1] : memref<?xbf16> to memref<16x576xbf16, strided<[?, 1], offset: ?>>
      %24 = arith.index_cast %20 : i32 to index
      %25 = arith.index_cast %21 : i32 to index
      %26 = arith.index_cast %arg22 : i32 to index
      %27 = scf.for %arg45 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg46 = %7) -> (tensor<16xf32>)  : i32 {
        %28 = arith.muli %arg45, %c128_i32 : i32
        %alloc = memref.alloc() : memref<16x576xbf16>
        memref.copy %reinterpret_cast_2, %alloc : memref<16x576xbf16, strided<[?, 1], offset: ?>> to memref<16x576xbf16>
        %29 = bufferization.to_tensor %alloc restrict writable : memref<16x576xbf16>
        %30 = arith.index_cast %28 : i32 to index
        %31 = arith.index_cast %arg25 : i32 to index
        %32 = arith.muli %30, %31 : index
        %33 = arith.addi %24, %32 : index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%33], sizes: [128, 576], strides: [%31, 1] : memref<?xbf16> to memref<128x576xbf16, strided<[?, 1], offset: ?>>
        %alloc_4 = memref.alloc() : memref<128x576xbf16>
        memref.copy %reinterpret_cast_3, %alloc_4 : memref<128x576xbf16, strided<[?, 1], offset: ?>> to memref<128x576xbf16>
        %34 = bufferization.to_tensor %alloc_4 restrict writable : memref<128x576xbf16>
        %35 = tensor.empty() : tensor<576x128xbf16>
        %transposed = linalg.transpose ins(%34 : tensor<128x576xbf16>) outs(%35 : tensor<576x128xbf16>) permutation = [1, 0]
        %36 = linalg.matmul {input_precision = "ieee"} ins(%29, %transposed : tensor<16x576xbf16>, tensor<576x128xbf16>) outs(%6 : tensor<16x128xf32>) -> tensor<16x128xf32>
        %37 = arith.mulf %36, %14 : tensor<16x128xf32>
        %38 = arith.addi %30, %c128 : index
        %39 = arith.maxsi %30, %c1024 : index
        %40 = arith.minsi %38, %39 : index
        %41 = arith.subi %40, %30 : index
        %extracted_slice = tensor.extract_slice %37[0, 0] [16, %41] [1, 1] : tensor<16x128xf32> to tensor<16x?xf32>
        %inserted_slice = tensor.insert_slice %extracted_slice into %1[0, 0] [16, %41] [1, 1] : tensor<16x?xf32> into tensor<16x128xf32>
        %reduced = linalg.reduce ins(%inserted_slice : tensor<16x128xf32>) outs(%7 : tensor<16xf32>) dimensions = [1]
          (%in: f32, %init: f32) {
            %65 = arith.maxnumf %in, %init : f32
            linalg.yield %65 : f32
          }
        %broadcasted = linalg.broadcast ins(%reduced : tensor<16xf32>) outs(%0 : tensor<16x128xf32>) dimensions = [1]
        %42 = arith.subf %inserted_slice, %broadcasted : tensor<16x128xf32>
        %43 = math.exp %42 : tensor<16x128xf32>
        %44 = linalg.fill ins(%cst : f32) outs(%2 : tensor<16xf32>) -> tensor<16xf32>
        %reduced_5 = linalg.reduce ins(%43 : tensor<16x128xf32>) outs(%44 : tensor<16xf32>) dimensions = [1]
          (%in: f32, %init: f32) {
            %65 = arith.addf %in, %init : f32
            linalg.yield %65 : f32
          }
        %45 = math.log %reduced_5 : tensor<16xf32>
        %46 = arith.addf %reduced, %45 : tensor<16xf32>
        %47 = arith.addf %arg46, %46 : tensor<16xf32>
        %48 = arith.divf %47, %3 : tensor<16xf32>
        %49 = arith.subf %arg46, %48 : tensor<16xf32>
        %50 = math.exp %49 : tensor<16xf32>
        %51 = arith.subf %46, %48 : tensor<16xf32>
        %52 = math.exp %51 : tensor<16xf32>
        %53 = arith.addf %50, %52 : tensor<16xf32>
        %54 = math.log %53 : tensor<16xf32>
        %55 = arith.addf %48, %54 : tensor<16xf32>
        %56 = arith.cmpf une, %55, %55 : tensor<16xf32>
        %57 = arith.select %56, %46, %55 : tensor<16xi1>, tensor<16xf32>
        %58 = arith.subf %46, %57 : tensor<16xf32>
        %59 = math.exp %58 : tensor<16xf32>
        %broadcasted_6 = linalg.broadcast ins(%reduced_5 : tensor<16xf32>) outs(%0 : tensor<16x128xf32>) dimensions = [1]
        %60 = arith.divf %43, %broadcasted_6 : tensor<16x128xf32>
        %61 = arith.truncf %60 : tensor<16x128xf32> to tensor<16x128xbf16>
        bufferization.materialize_in_destination %61 in writable %reinterpret_cast : (tensor<16x128xbf16>, memref<16x128xbf16, strided<[?, 1], offset: ?>>) -> ()
        %alloc_7 = memref.alloc() : memref<16x128xbf16>
        memref.copy %reinterpret_cast, %alloc_7 : memref<16x128xbf16, strided<[?, 1], offset: ?>> to memref<16x128xbf16>
        %62 = bufferization.to_tensor %alloc_7 restrict writable : memref<16x128xbf16>
        %63 = arith.subf %arg46, %57 : tensor<16xf32>
        %64 = math.exp %63 : tensor<16xf32>
        %broadcasted_8 = linalg.broadcast ins(%64 : tensor<16xf32>) outs(%4 : tensor<16x256xf32>) dimensions = [1]
        %broadcasted_9 = linalg.broadcast ins(%59 : tensor<16xf32>) outs(%4 : tensor<16x256xf32>) dimensions = [1]
        scf.for %arg47 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %65 = arith.muli %arg47, %c256_i32 : i32
          %66 = arith.index_cast %65 : i32 to index
          %67 = arith.addi %33, %66 : index
          %reinterpret_cast_10 = memref.reinterpret_cast %arg7 to offset: [%67], sizes: [128, 256], strides: [%31, 1] : memref<?xbf16> to memref<128x256xbf16, strided<[?, 1], offset: ?>>
          %alloc_11 = memref.alloc() : memref<128x256xbf16>
          memref.copy %reinterpret_cast_10, %alloc_11 : memref<128x256xbf16, strided<[?, 1], offset: ?>> to memref<128x256xbf16>
          %68 = bufferization.to_tensor %alloc_11 restrict writable : memref<128x256xbf16>
          %69 = linalg.matmul {input_precision = "ieee"} ins(%62, %68 : tensor<16x128xbf16>, tensor<128x256xbf16>) outs(%5 : tensor<16x256xf32>) -> tensor<16x256xf32>
          %70 = arith.addi %17, %66 : index
          %reinterpret_cast_12 = memref.reinterpret_cast %arg11 to offset: [%70], sizes: [16, 256], strides: [%18, 1] : memref<?xf32> to memref<16x256xf32, strided<[?, 1], offset: ?>>
          bufferization.materialize_in_destination %69 in writable %reinterpret_cast_12 : (tensor<16x256xf32>, memref<16x256xf32, strided<[?, 1], offset: ?>>) -> ()
          %71 = arith.addi %25, %66 : index
          %reinterpret_cast_13 = memref.reinterpret_cast %arg6 to offset: [%71], sizes: [16, 256], strides: [%26, 1] : memref<?xf32> to memref<16x256xf32, strided<[?, 1], offset: ?>>
          %alloc_14 = memref.alloc() : memref<16x256xf32>
          memref.copy %reinterpret_cast_13, %alloc_14 : memref<16x256xf32, strided<[?, 1], offset: ?>> to memref<16x256xf32>
          %72 = bufferization.to_tensor %alloc_14 restrict writable : memref<16x256xf32>
          %73 = arith.mulf %72, %broadcasted_8 : tensor<16x256xf32>
          %74 = arith.mulf %69, %broadcasted_9 : tensor<16x256xf32>
          %75 = arith.addf %73, %74 : tensor<16x256xf32>
          bufferization.materialize_in_destination %75 in writable %reinterpret_cast_13 : (tensor<16x256xf32>, memref<16x256xf32, strided<[?, 1], offset: ?>>) -> ()
        }
        scf.yield %57 : tensor<16xf32>
      }
    }
    return
  }
}