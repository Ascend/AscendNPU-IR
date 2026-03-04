// RUN: bishengir-compile %s --target=Ascend950PR_9589 --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-lir-compile=false

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @_attn_fwd(%arg0: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096xf32>) -> tensor<4096xf32>
    scf.for %arg7 = %arg4 to %c1_i32 step %c32_i32  : i32 {
      %2 = scope.scope : () -> tensor<4096xf32> {
        %4 = scf.for %arg8 = %c0_i32 to %c4096_i32 step %c64_i32 iter_args(%arg9 = %1) -> (tensor<4096xf32>)  : i32 {
          %5 = arith.index_cast %arg8 : i32 to index
          %extracted_slice = tensor.extract_slice %arg9[%5] [64] [1] : tensor<4096xf32> to tensor<64xf32>
          %6 = arith.mulf %extracted_slice, %extracted_slice : tensor<64xf32>
          %inserted_slice = tensor.insert_slice %6 into %arg9[%5] [64] [1] : tensor<64xf32> into tensor<4096xf32>
          scf.yield %inserted_slice : tensor<4096xf32>
        }
        scope.return %4 : tensor<4096xf32>
      } {noinline, outline = true, vector_mode = "simd"}
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096], strides: [1] : memref<?xf16> to memref<4096xf16, strided<[1], offset: ?>>
      %3 = arith.truncf %2 : tensor<4096xf32> to tensor<4096xf16>
      bufferization.materialize_in_destination %3 in writable %reinterpret_cast : (tensor<4096xf16>, memref<4096xf16, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}
