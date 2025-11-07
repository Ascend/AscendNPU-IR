// REQUIRES: asserts
// RUN: bishengir-opt %s --propagate-reshape --cse --canonicalize-ext --valid-propagate --debug-only="propagate-valid-check" -split-input-file | FileCheck %s

// CHECK: Valid
// CHECK-LABEL: @test_collapse_down_memref_strided
func.func @test_collapse_down_memref_strided(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false]> : vector<8xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
  %c0_i32 = arith.constant 0 : i32
  %c1048576_i32 = arith.constant 1048576 : i32
  %c8192_i32 = arith.constant 8192 : i32
  %c128_i32 = arith.constant 128 : i32
  %c32_i32 = arith.constant 32 : i32
  %c48_i32 = arith.constant 48 : i32
  %c8_i32 = arith.constant 8 : i32
  %c384_i32 = arith.constant 384 : i32
  %c256_i32 = arith.constant 256 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.muli %arg7, %arg6 : i32
  %3 = arith.divsi %1, %2 : i32
  %4 = arith.remsi %3, %arg5 : i32
  hivm.hir.set_mask_norm
  %5 = arith.muli %4, %c256_i32 : i32
  scf.for %arg8 = %c0_i32 to %c384_i32 step %c1_i32  : i32 {
    %6 = arith.divsi %arg8, %c384_i32 : i32
    %7 = arith.divsi %arg8, %c8_i32 : i32
    %8 = arith.remsi %7, %c48_i32 : i32
    %9 = arith.remsi %arg8, %c8_i32 : i32
    %10 = arith.muli %9, %c32_i32 : i32
    %11 = arith.addi %5, %10 : i32
    %12 = arith.muli %11, %c128_i32 : i32
    %13 = arith.muli %6, %c48_i32 : i32
    %14 = arith.addi %13, %8 : i32
    %15 = arith.muli %14, %c8192_i32 : i32
    %16 = arith.index_cast %15 : i32 to index
    %17 = arith.index_cast %11 : i32 to index
    %18 = arith.addi %16, %17 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<32xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<32xf32, strided<[1], offset: ?>>) outs(%alloc : memref<32xf32>)
    %19 = bufferization.to_tensor %alloc restrict writable : memref<32xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%18], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<32xf32>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<32xf32, strided<[1], offset: ?>>) outs(%alloc_1 : memref<32xf32>)
    %20 = bufferization.to_tensor %alloc_1 restrict writable : memref<32xf32>
    %21 = arith.muli %14, %c1048576_i32 : i32
    %22 = arith.index_cast %21 : i32 to index
    %23 = arith.index_cast %12 : i32 to index
    %24 = arith.addi %22, %23 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [4096], strides: [1] : memref<?xf32> to memref<4096xf32, strided<[1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<4096xf32>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<4096xf32, strided<[1], offset: ?>>) outs(%alloc_3 : memref<4096xf32>)
    %25 = bufferization.to_tensor %alloc_3 restrict writable : memref<4096xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [4096], strides: [1] : memref<?xf32> to memref<4096xf32, strided<[1], offset: ?>>
    %alloc_5 = memref.alloc() : memref<4096xf32>
    hivm.hir.load ins(%reinterpret_cast_4 : memref<4096xf32, strided<[1], offset: ?>>) outs(%alloc_5 : memref<4096xf32>)
    %26 = bufferization.to_tensor %alloc_5 restrict writable : memref<4096xf32>
    %27 = tensor.empty() : tensor<32xf32>
    %28 = hivm.hir.vexp ins(%20 : tensor<32xf32>) outs(%27 : tensor<32xf32>) -> tensor<32xf32>
    %29 = hivm.hir.vexp ins(%19 : tensor<32xf32>) outs(%27 : tensor<32xf32>) -> tensor<32xf32>
    %30 = hivm.hir.vadd ins(%28, %29 : tensor<32xf32>, tensor<32xf32>) outs(%27 : tensor<32xf32>) -> tensor<32xf32>
    %31 = hivm.hir.vln ins(%30 : tensor<32xf32>) outs(%27 : tensor<32xf32>) -> tensor<32xf32>
    %32 = hivm.hir.vsub ins(%19, %31 : tensor<32xf32>, tensor<32xf32>) outs(%27 : tensor<32xf32>) -> tensor<32xf32>
    %expanded = tensor.expand_shape %32 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    %33 = hivm.hir.vsub ins(%20, %31 : tensor<32xf32>, tensor<32xf32>) outs(%27 : tensor<32xf32>) -> tensor<32xf32>
    %expanded_6 = tensor.expand_shape %33 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    %34 = tensor.empty() : tensor<32x1xf32>
    %35 = hivm.hir.vexp ins(%expanded : tensor<32x1xf32>) outs(%34 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %36 = tensor.empty() : tensor<32x128xf32>
    %37 = hivm.hir.vbrc ins(%35 : tensor<32x1xf32>) outs(%36 : tensor<32x128xf32>) broadcast_dims = [1] -> tensor<32x128xf32>
    %expanded_7 = tensor.expand_shape %25 [[0, 1]] output_shape [32, 128] : tensor<4096xf32> into tensor<32x128xf32>
    %38 = hivm.hir.vmul ins(%37, %expanded_7 : tensor<32x128xf32>, tensor<32x128xf32>) outs(%36 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %39 = hivm.hir.vexp ins(%expanded_6 : tensor<32x1xf32>) outs(%34 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %40 = hivm.hir.vbrc ins(%39 : tensor<32x1xf32>) outs(%36 : tensor<32x128xf32>) broadcast_dims = [1] -> tensor<32x128xf32>
    %expanded_8 = tensor.expand_shape %26 [[0, 1]] output_shape [32, 128] : tensor<4096xf32> into tensor<32x128xf32>
    %41 = hivm.hir.vmul ins(%40, %expanded_8 : tensor<32x128xf32>, tensor<32x128xf32>) outs(%36 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %42 = hivm.hir.vadd ins(%38, %41 : tensor<32x128xf32>, tensor<32x128xf32>) outs(%36 : tensor<32x128xf32>) -> tensor<32x128xf32>
    hivm.hir.store ins(%31 : tensor<32xf32>) outs(%reinterpret_cast : memref<32xf32, strided<[1], offset: ?>>)
    %collapsed = tensor.collapse_shape %42 [[0, 1]] : tensor<32x128xf32> into tensor<4096xf32>
    hivm.hir.store ins(%collapsed : tensor<4096xf32>) outs(%reinterpret_cast_2 : memref<4096xf32, strided<[1], offset: ?>>)
  }
  return
}

// -----

// CHECK: Valid
// CHECK-LABEL: @test_swap_collapse_expand
func.func @test_swap_collapse_expand(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg2: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}) {
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 128, 3], strides: [384, 3, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<16x128x3xbf16, #hivm.address_space<gm>>
  %alloc = memref.alloc() : memref<128x16x3xbf16, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %alloc [[0, 1, 2]] : memref<128x16x3xbf16, #hivm.address_space<ub>> into memref<6144xbf16, #hivm.address_space<ub>>
  %expand_shape = memref.expand_shape %collapse_shape [[0, 1, 2]] output_shape [16, 128, 3] : memref<6144xbf16, #hivm.address_space<ub>> into memref<16x128x3xbf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast : memref<16x128x3xbf16, #hivm.address_space<gm>>) outs(%expand_shape : memref<16x128x3xbf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128, 16, 3], strides: [48, 3, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x16x3xbf16, #hivm.address_space<gm>>
  hivm.hir.store ins(%alloc : memref<128x16x3xbf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<128x16x3xbf16, #hivm.address_space<gm>>)
  return
}