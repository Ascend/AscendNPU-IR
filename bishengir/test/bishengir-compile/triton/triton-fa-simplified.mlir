// REQUIRES: issue893
// RUN: bishengir-compile -enable-triton-kernel-compile -enable-lir-compile=false -enable-hfusion-compile -block-dim=1 -target=Ascend310B4  %s

module {
  func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xf16> {tt.divisibility = 16 : i32}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {WorkspaceArgIdx = 0 : i64, hivm.func_core_type = #hivm.func_core_type<AIC>, global_kernel = "local", mix_mode = "mix"} {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %filled2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %2 = tensor.empty() : tensor<64x16xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<64x16xf32>) -> tensor<64x16xf32>
    %4 = arith.muli %arg10, %c64_i32 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.muli %5, %c64 : index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%6], sizes: [64, 64], strides: [%c64, %c1] : memref<?xf16> to memref<64x64xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%6], sizes: [64], strides: [%c64] : memref<?xf16> to memref<64xf16, strided<[?], offset: ?>>
    %alloc = memref.alloc() : memref<64x64xf16>
    %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%c0], sizes: [64], strides: [%c1] : memref<?xf32> to memref<64xf32, strided<[?], offset: ?>>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xf16, strided<[?, ?], offset: ?>> to memref<64x64xf16>
    %7 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%c0], sizes: [16, 64], strides: [%c64, %c1] : memref<?xf16> to memref<16x64xf16, strided<[?, ?], offset: ?>>
    %8:3 = scf.for %arg13 = %c0_i32 to %c64_i32 step %c16_i32 iter_args(%arg14 = %1, %arg15 = %reinterpret_cast_2, %arg16 = %c0) -> (tensor<64xf32>, memref<16x64xf16, strided<[?, ?], offset: ?>>, index)  : i32 {
      %alloc_3 = memref.alloc() : memref<16x64xf16>
      memref.copy %arg15, %alloc_3 : memref<16x64xf16, strided<[?, ?], offset: ?>> to memref<16x64xf16>
      %10 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x64xf16>
      %11 = tensor.empty() : tensor<64x16xf16>
      %transposed = linalg.transpose ins(%10 : tensor<16x64xf16>) outs(%11 : tensor<64x16xf16>) permutation = [1, 0]
      %12 = linalg.matmul ins(%7, %transposed : tensor<64x64xf16>, tensor<64x16xf16>) outs(%3 : tensor<64x16xf32>) -> tensor<64x16xf32>
      %reduced = linalg.reduce ins(%12 : tensor<64x16xf32>) outs(%filled2 : tensor<64xf32>) dimensions = [1]
        (%in: f32, %init: f32) {
          %15 = arith.maxnumf %in, %init : f32
          linalg.yield %15 : f32
        }
      %13 = arith.maxnumf %arg14, %reduced : tensor<64xf32>
      %14 = arith.addi %arg16, %c1024 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [16, 64], strides: [%c64, %c1] : memref<?xf16> to memref<16x64xf16, strided<[?, ?], offset: ?>>
      scf.yield %13, %reinterpret_cast_4, %14 : tensor<64xf32>, memref<16x64xf16, strided<[?, ?], offset: ?>>, index
    }
    // %9 = arith.truncf %8#0 : tensor<64xf32> to tensor<64xf16>
    // bufferization.materialize_in_destination %9 in writable %reinterpret_cast_1 : (tensor<64xf16>, memref<64xf16, strided<[?], offset: ?>>) -> ()

    // Use reinterpret_cast_10 to specify the constant offset, size, stride for arg4.
    // Alternative method of directly `writable %arg4` should also work,
    // except these three arguments will come in as function argument instead.
    // bufferization.materialize_in_destination %8#0 in writable %arg4 : (tensor<64xf32>, memref<?xf32>) -> ()
    bufferization.materialize_in_destination %8#0 in writable %reinterpret_cast_10 : (tensor<64xf32>, memref<64xf32, strided<[?], offset: ?>>) -> ()
    return
  }
}

