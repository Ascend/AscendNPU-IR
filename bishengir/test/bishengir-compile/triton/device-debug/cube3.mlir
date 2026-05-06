// RUN: bishengir-compile --enable-lir-compile=false --enable-auto-multi-buffer=true --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true %s | FileCheck %s

// CHECK-DAG: @_debug_prefix_0
// CHECK-DAG: declare extern_weak dso_local void @_mlir_ciface_init_debug
// CHECK-DAG: define private void @fixpipe_nz2nd_float_to_bfloat16_t_4d_to_2d_gm
// CHECK-DAG: declare dso_local void @_mlir_ciface_fixpipe_nz2nd_float_to_bfloat16_t_4d_to_2d_gm
// CHECK-DAG: define private void @print_2d_bfloat16_t_gm
// CHECK-DAG: declare extern_weak dso_local void @_mlir_ciface_print_2d_bfloat16_t_gm
// CHECK-DAG: declare extern_weak dso_local void @_mlir_ciface_finish_debug
// CHECK-DAG: define dso_local void @_fwd(
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func private @triton_print_0(tensor<2x2xbf16>) attributes {hex = false, prefix = " do: "}
  func.func @_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i8 = arith.constant 0 : i8
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x2xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x2xf32>) -> tensor<2x2xf32>
    scf.for %arg12 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %2 = arith.muli %arg12, %c4_i32 : i32
      %3 = arith.addi %2, %arg9 : i32
      %4 = arith.divsi %3, %c4_i32 : i32
      %5 = arith.index_cast %4 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%5], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
      %6 = memref.load %reinterpret_cast[%c0] {was_bool_to_int8 = true} : memref<1xi8, strided<[1], offset: ?>>
      %7 = arith.cmpi ne, %6, %c0_i8 : i8
      scf.if %7 {
        %8 = arith.index_cast %2 : i32 to index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [2, 2], strides: [2, 1] : memref<?xbf16> to memref<2x2xbf16, strided<[2, 1], offset: ?>>
        %alloc = memref.alloc() : memref<2x2xbf16>
        memref.copy %reinterpret_cast_0, %alloc : memref<2x2xbf16, strided<[2, 1], offset: ?>> to memref<2x2xbf16>
        %9 = bufferization.to_tensor %alloc restrict writable : memref<2x2xbf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%8], sizes: [2, 2], strides: [2, 1] : memref<?xbf16> to memref<2x2xbf16, strided<[2, 1], offset: ?>>
        %alloc_2 = memref.alloc() : memref<2x2xbf16>
        memref.copy %reinterpret_cast_1, %alloc_2 : memref<2x2xbf16, strided<[2, 1], offset: ?>> to memref<2x2xbf16>
        %10 = bufferization.to_tensor %alloc_2 restrict writable : memref<2x2xbf16>
        %11 = linalg.matmul {input_precision = "ieee"} ins(%9, %10 : tensor<2x2xbf16>, tensor<2x2xbf16>) outs(%1 : tensor<2x2xf32>) -> tensor<2x2xf32>
        %12 = arith.truncf %11 : tensor<2x2xf32> to tensor<2x2xbf16>
        func.call @triton_print_0(%12) : (tensor<2x2xbf16>) -> ()
        %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%8], sizes: [2, 2], strides: [2, 1] : memref<?xbf16> to memref<2x2xbf16, strided<[2, 1], offset: ?>>
        bufferization.materialize_in_destination %12 in writable %reinterpret_cast_3 : (tensor<2x2xbf16>, memref<2x2xbf16, strided<[2, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}