// REQUIRES: issue893
// RUN: bishengir-compile -enable-triton-kernel-compile -enable-lir-compile=false -enable-hfusion-compile -block-dim=1 -target=Ascend310B4  %s
module {
  func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xf16> {tt.divisibility = 16 : i32}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "mix", hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.72134751 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %cst_3 = arith.constant 1.44269502 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<64xf32>
    %3 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %5 = tensor.empty() : tensor<64x16xf32>
    %6 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<64x16xf32>) -> tensor<64x16xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %8 = tensor.empty() : tensor<64x64xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %10 = arith.muli %arg10, %c64_i32 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12 = arith.muli %11, %c64 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [64, 64], strides: [%c64, %c1] : memref<?xf16> to memref<64x64xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%12], sizes: [64, 64], strides: [%c64, %c1] : memref<?xf16> to memref<64x64xf16, strided<[?, ?], offset: ?>>
    %13 = linalg.fill ins(%arg6 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %14 = arith.mulf %13, %1 : tensor<1xf32>
    %extracted = tensor.extract %14[%c0] : tensor<1xf32>
    %alloc = memref.alloc() : memref<64x64xf16>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xf16, strided<[?, ?], offset: ?>> to memref<64x64xf16>
    %15 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
    %16 = linalg.fill ins(%extracted : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %17 = linalg.fill ins(%extracted : f32) outs(%5 : tensor<64x16xf32>) -> tensor<64x16xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%c0], sizes: [16, 64], strides: [%c64, %c1] : memref<?xf16> to memref<16x64xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg2 to offset: [%c0], sizes: [16, 64], strides: [%c64, %c1] : memref<?xf16> to memref<16x64xf16, strided<[?, ?], offset: ?>>
    %18:9 = scf.for %arg13 = %c0_i32 to %c64_i32 step %c16_i32 iter_args(%arg14 = %3, %arg15 = %9, %arg16 = %4, %arg17 = %reinterpret_cast_5, %arg18 = %reinterpret_cast_6, %arg19 = %c0, %arg20 = %c0, %arg21 = %c0, %arg22 = %c0) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, memref<16x64xf16, strided<[?, ?], offset: ?>>, memref<16x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
      %alloc_8 = memref.alloc() : memref<16x64xf16>
      memref.copy %arg18, %alloc_8 : memref<16x64xf16, strided<[?, ?], offset: ?>> to memref<16x64xf16>
      %23 = bufferization.to_tensor %alloc_8 restrict writable : memref<16x64xf16>
      %24 = tensor.empty() : tensor<64x16xf16>
      %transposed = linalg.transpose ins(%23 : tensor<16x64xf16>) outs(%24 : tensor<64x16xf16>) permutation = [1, 0]
      %25 = linalg.matmul ins(%15, %transposed : tensor<64x64xf16>, tensor<64x16xf16>) outs(%6 : tensor<64x16xf32>) -> tensor<64x16xf32>
      %reduced = linalg.reduce ins(%25 : tensor<64x16xf32>) outs(%2 : tensor<64xf32>) dimensions = [1]
        (%in: f32, %init: f32) {
          %45 = arith.maxnumf %in, %init : f32
          linalg.yield %45 : f32
        }
      %26 = arith.mulf %reduced, %16 : tensor<64xf32>
      %27 = arith.maxnumf %arg16, %26 : tensor<64xf32>
      %28 = arith.mulf %25, %17 : tensor<64x16xf32>
      %broadcasted_9 = linalg.broadcast ins(%27 : tensor<64xf32>) outs(%5 : tensor<64x16xf32>) dimensions = [1]
      %29 = arith.subf %28, %broadcasted_9 : tensor<64x16xf32>
      %30 = math.exp2 %29 : tensor<64x16xf32>
      %reduced_10 = linalg.reduce ins(%30 : tensor<64x16xf32>) outs(%2 : tensor<64xf32>) dimensions = [1]
        (%in: f32, %init: f32) {
          %45 = arith.addf %in, %init : f32
          linalg.yield %45 : f32
        }
      %31 = arith.subf %arg16, %27 : tensor<64xf32>
      %32 = math.exp2 %31 : tensor<64xf32>
      %33 = arith.mulf %arg14, %32 : tensor<64xf32>
      %34 = arith.addf %33, %reduced_10 : tensor<64xf32>
      %broadcasted_11 = linalg.broadcast ins(%32 : tensor<64xf32>) outs(%8 : tensor<64x64xf32>) dimensions = [1]
      %35 = arith.mulf %arg15, %broadcasted_11 : tensor<64x64xf32>
      %alloc_12 = memref.alloc() : memref<16x64xf16>
      memref.copy %arg17, %alloc_12 : memref<16x64xf16, strided<[?, ?], offset: ?>> to memref<16x64xf16>
      %36 = bufferization.to_tensor %alloc_12 restrict writable : memref<16x64xf16>
      %37 = arith.truncf %30 : tensor<64x16xf32> to tensor<64x16xf16>
      %38 = linalg.matmul ins(%37, %36 : tensor<64x16xf16>, tensor<16x64xf16>) outs(%35 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %39 = arith.mulf %27, %16 : tensor<64xf32>
      %40 = arith.divf %39, %7 : tensor<64xf32>
      %41 = arith.addi %arg19, %c1024 : index
      %42 = arith.addi %41, %arg20 : index
      %reinterpret_cast_13 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [16, 64], strides: [%c64, %c1] : memref<?xf16> to memref<16x64xf16, strided<[?, ?], offset: ?>>
      %43 = arith.addi %arg21, %c1024 : index
      %44 = arith.addi %43, %arg22 : index
      %reinterpret_cast_14 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [16, 64], strides: [%c64, %c1] : memref<?xf16> to memref<16x64xf16, strided<[?, ?], offset: ?>>
      scf.yield %34, %38, %40, %reinterpret_cast_13, %reinterpret_cast_14, %42, %c0, %44, %c0 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, memref<16x64xf16, strided<[?, ?], offset: ?>>, memref<16x64xf16, strided<[?, ?], offset: ?>>, index, index, index, index
    }
    %19 = math.log2 %18#0 : tensor<64xf32>
    %20 = arith.addf %18#2, %19 : tensor<64xf32>
    %broadcasted = linalg.broadcast ins(%18#0 : tensor<64xf32>) outs(%8 : tensor<64x64xf32>) dimensions = [1]
    %21 = arith.divf %18#1, %broadcasted : tensor<64x64xf32>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [%11], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %20 in writable %reinterpret_cast_7 : (tensor<64xf32>, memref<64xf32, strided<[1], offset: ?>>) -> ()
    %22 = arith.truncf %21 : tensor<64x64xf32> to tensor<64x64xf16>
    bufferization.materialize_in_destination %22 in writable %reinterpret_cast_4 : (tensor<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>) -> ()
    return
  }
}


