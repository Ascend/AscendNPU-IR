// RUN: bishengir-opt --auto-scope --split-input-file %s
// XFAIL: *
func.func @simple_indirect_load_add_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xi64>, %arg4: memref<?xf32>, %arg5: f32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {parallel_mode = "mix_simd_simt"} {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.muli %arg6, %arg7 : i32
    %1 = arith.muli %0, %arg8 : i32
    %alloc = memref.alloc() : memref<33xi64>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<33xi64>
    %3 = tensor.empty() : tensor<33xf32>
    %4 = hivm.hir.gather_load ins(%arg2 : memref<?xf32>, %2 : tensor<33xi64>, %c1_i32 : i32) outs(%3 : tensor<33xf32>) -> tensor<33xf32>
    %5 = tensor.empty() : tensor<33xf32>
    %6 = hivm.hir.vadd ins(%4, %arg5 : tensor<33xf32>, f32) outs (%5 : tensor<33xf32>) -> tensor<33xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [33], strides: [1] : memref<?xf32> to memref<33xf32, strided<[1]>>
    hivm.hir.store ins(%6 : tensor<33xf32>) outs(%reinterpret_cast_0 : memref<33xf32, strided<[1]>>)
    return
}