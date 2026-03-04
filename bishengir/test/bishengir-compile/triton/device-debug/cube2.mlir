// RUN: bishengir-compile --enable-lir-compile=false --enable-auto-multi-buffer=true --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true %s | FileCheck %s

// CHECK-DAG: @_debug_prefix_0
// CHECK-DAG: declare extern_weak dso_local void @_mlir_ciface_init_debug
// CHECK-DAG: define private void @print_2d_float_gm
// CHECK-DAG: declare extern_weak dso_local void @_mlir_ciface_print_2d_float_gm
// CHECK-DAG: declare extern_weak dso_local void @_mlir_ciface_finish_debug
// CHECK-DAG: define dso_local void @matmul_kernel
module {
  func.func private @triton_print_0(tensor<16x16xf32>) attributes {hex = false, prefix = " accumulator: "}
  func.func @matmul_kernel(%arg0: memref<?xi8>, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "mix"} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %2 = arith.index_cast %arg4 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 16], strides: [%2, 1] : memref<?xf32> to memref<16x16xf32, strided<[?, 1]>>
    %3 = arith.index_cast %arg5 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 16], strides: [%3, 1] : memref<?xf32> to memref<16x16xf32, strided<[?, 1]>>
    %alloc = memref.alloc() : memref<16x16xf32>
    memref.copy %reinterpret_cast, %alloc : memref<16x16xf32, strided<[?, 1]>> to memref<16x16xf32>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
    %alloc_1 = memref.alloc() : memref<16x16xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<16x16xf32, strided<[?, 1]>> to memref<16x16xf32>
    %5 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16xf32>
    %6 = linalg.matmul ins(%4, %5 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    gpu.barrier
    call @triton_print_0(%6) : (tensor<16x16xf32>) -> ()
    %7 = arith.truncf %6 : tensor<16x16xf32> to tensor<16x16xf16>
    %8 = arith.index_cast %arg6 : i32 to index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [%8, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1]>>
    bufferization.materialize_in_destination %7 in writable %reinterpret_cast_2 : (tensor<16x16xf16>, memref<16x16xf16, strided<[?, 1]>>) -> ()
    return
  }
}
