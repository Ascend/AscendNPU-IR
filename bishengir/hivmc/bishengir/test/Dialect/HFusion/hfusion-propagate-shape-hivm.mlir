// RUN: bishengir-opt %s --propagate-reshape="for-hivm=true" --cse --canonicalize -split-input-file | FileCheck %s

// CHECK: diamond_brc_binary(
// CHECK: tensor.expand_shape
// CHECK: tensor<8xi32> into tensor<1x8xi32>
// CHECK: tensor.expand_shape
// CHECK: tensor<8xi32> into tensor<8x1xi32>
// CHECK: return
module {
  func.func @diamond_brc_binary(%arg0: tensor<8xi64>) -> tensor<8x8xi32> {
    %0 = tensor.empty() : tensor<8xi32>
    %1 = tensor.empty() : tensor<8x8xi32>
    %2 = hivm.hir.vcast ins(%arg0 : tensor<8xi64>) outs(%0 : tensor<8xi32>) -> tensor<8xi32>
    %expanded = tensor.expand_shape %2 [[0, 1]] output_shape [1, 8] : tensor<8xi32> into tensor<1x8xi32>
    %3 = hivm.hir.vbrc ins(%expanded : tensor<1x8xi32>) outs(%1 : tensor<8x8xi32>) broadcast_dims = [0] -> tensor<8x8xi32>
    %expanded_0 = tensor.expand_shape %2 [[0, 1]] output_shape [8, 1] : tensor<8xi32> into tensor<8x1xi32>
    %4 = hivm.hir.vbrc ins(%expanded_0 : tensor<8x1xi32>) outs(%1 : tensor<8x8xi32>) broadcast_dims = [1] -> tensor<8x8xi32>
    %5 = hivm.hir.vmul ins(%3, %4 : tensor<8x8xi32>, tensor<8x8xi32>) outs(%1 : tensor<8x8xi32>) -> tensor<8x8xi32>
    return %5 : tensor<8x8xi32>
  }
}

// -----
module {
  func.func @mm_03(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %false = arith.constant false
    %c5_i32 = arith.constant 5 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c640_i32 = arith.constant 640 : i32
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c1280 = arith.constant 1280 : index
    %c64 = arith.constant 64 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %arg8, %arg7 : i32
    %3 = arith.divsi %1, %2 : i32
    %4 = arith.remsi %3, %arg6 : i32
    hivm.hir.set_mask_norm
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1]>>
    %alloc = memref.alloc() : memref<256x64xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<256x64xf16, strided<[64, 1]>>) outs(%alloc : memref<256x64xf16>)
    %5 = bufferization.to_tensor %alloc restrict writable : memref<256x64xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1]>>
    // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<1x64xf32>
    // CHECK: %[[collapse_shape_1:.*]] = memref.collapse_shape %[[alloc_1:.*]] {{\[}}[0, 1]] : memref<1x64xf32> into memref<64xf32>
    // CHECK: hivm.hir.load ins({{.*}} : memref<64xf32, strided<[1]>>) outs(%[[collapse_shape_1:.*]] : memref<64xf32>)
    // CHECK: %[[bias:.*]] = bufferization.to_tensor %[[alloc_1:.*]] restrict writable : memref<1x64xf32>
    %alloc_1 = memref.alloc() : memref<64xf32>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<64xf32, strided<[1]>>) outs(%alloc_1 : memref<64xf32>)
    %6 = bufferization.to_tensor %alloc_1 restrict writable : memref<64xf32>
    %7 = arith.muli %4, %c640_i32 : i32
    %8 = tensor.empty() : tensor<128x64xf32>
    %expanded = tensor.expand_shape %6 [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
    %9 = hivm.hir.vbrc ins(%expanded : tensor<1x64xf32>) outs(%8 : tensor<128x64xf32>) broadcast_dims = [0] -> tensor<128x64xf32>
    scf.for %arg9 = %c0_i32 to %c5_i32 step %c1_i32  : i32 {
      %10 = arith.muli %arg9, %c128_i32 : i32
      %11 = arith.addi %7, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %13 = arith.muli %12, %c256 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [128, 256], strides: [256, 1] : memref<?xf16> to memref<128x256xf16, strided<[256, 1], offset: ?>>
      %alloc_3 = memref.alloc() : memref<128x256xf16>
      %14 = arith.addi %12, %c128 : index
      %15 = arith.maxsi %12, %c1280 : index
      %16 = arith.minsi %14, %15 : index
      %17 = arith.subi %16, %12 : index
      %18 = arith.minsi %17, %c128 : index
      %subview = memref.subview %reinterpret_cast_2[0, 0] [%18, 256] [1, 1] : memref<128x256xf16, strided<[256, 1], offset: ?>> to memref<?x256xf16, strided<[256, 1], offset: ?>>
      %subview_4 = memref.subview %alloc_3[0, 0] [%18, 256] [1, 1] : memref<128x256xf16> to memref<?x256xf16, strided<[256, 1]>>
      hivm.hir.load ins(%subview : memref<?x256xf16, strided<[256, 1], offset: ?>>) outs(%subview_4 : memref<?x256xf16, strided<[256, 1]>>) left_padding_num = %c0 : index
      %19 = bufferization.to_tensor %alloc_3 restrict writable : memref<128x256xf16>
      %20 = hivm.hir.mmadL1 ins(%19, %5, %false, %c0, %c0, %c0 : tensor<128x256xf16>, tensor<256x64xf16>, i1, index, index, index) outs(%9 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %21 = arith.muli %12, %c64 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%21], sizes: [128, 64], strides: [64, 1] : memref<?xf32> to memref<128x64xf32, strided<[64, 1], offset: ?>>
      %extracted_slice = tensor.extract_slice %20[0, 0] [%18, 64] [1, 1] : tensor<128x64xf32> to tensor<?x64xf32>
      %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%18, 64] [1, 1] : memref<128x64xf32, strided<[64, 1], offset: ?>> to memref<?x64xf32, strided<[64, 1], offset: ?>>
      hivm.hir.store ins(%extracted_slice : tensor<?x64xf32>) outs(%subview_6 : memref<?x64xf32, strided<[64, 1], offset: ?>>)
    }
    return
  }
}

// -----
// CHECK: func.func @unit_expand_shape(
func.func @unit_expand_shape(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi32> {tt.divisibility = 16 : i32}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, false, false, false]> : vector<7xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
  %c0_i32 = arith.constant 0 : i32
  %c12 = arith.constant 12 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  %c3_i32 = arith.constant 3 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.remsi %1, %arg6 : i32
  %3 = arith.divsi %1, %arg6 : i32
  %4 = arith.remsi %3, %arg5 : i32
  %5 = arith.muli %arg6, %arg5 : i32
  %6 = arith.divsi %1, %5 : i32
  %7 = arith.remsi %6, %arg4 : i32
  hivm.hir.set_mask_norm
  %8 = arith.muli %7, %c2_i32 : i32
  %9 = arith.muli %4, %c3_i32 : i32
  %10 = arith.muli %2, %c4_i32 : i32
  %11 = arith.index_cast %8 : i32 to index
  %12 = arith.muli %11, %c12 : index
  %13 = arith.index_cast %9 : i32 to index
  %14 = arith.muli %13, %c4 : index
  %15 = arith.index_cast %10 : i32 to index
  %16 = arith.addi %12, %14 : index
  %17 = arith.addi %16, %15 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%17], sizes: [2, 3, 4], strides: [12, 4, 1] : memref<?xi32> to memref<2x3x4xi32, strided<[12, 4, 1], offset: ?>>
  %alloc = memref.alloc() : memref<2x3x4xi32>
  hivm.hir.load ins(%reinterpret_cast : memref<2x3x4xi32, strided<[12, 4, 1], offset: ?>>) outs(%alloc : memref<2x3x4xi32>) init_out_buffer = false
  %18 = bufferization.to_tensor %alloc restrict writable : memref<2x3x4xi32>
  %collapsed = tensor.collapse_shape %18 [[0, 1, 2]] : tensor<2x3x4xi32> into tensor<24xi32>
  %19 = bufferization.alloc_tensor() : tensor<i32>
  %20 = hivm.hir.vbrc ins(%c0_i32 : i32) outs(%19 : tensor<i32>) -> tensor<i32>
  %expanded = tensor.expand_shape %20 [] output_shape [1] : tensor<i32> into tensor<1xi32>
  // CHECK: hivm.hir.vreduce <sum> ins
  // CHECK-SAME: tensor<2x3x4xi32>
  // CHECK-SAME tensor<1x1x1xi32>
  // CHECK-SAME reduce_dims = [0, 1, 2] -> tensor<1x1x1xi32>
  %21 = hivm.hir.vreduce <sum> ins(%collapsed : tensor<24xi32>) outs(%expanded : tensor<1xi32>) unsigned_src = false reduce_dims = [0] -> tensor<1xi32>
  %collapsed_0 = tensor.collapse_shape %21 [] : tensor<1xi32> into tensor<i32>
  %extracted = tensor.extract %collapsed_0[] : tensor<i32>
  %22 = tensor.empty() : tensor<1xi32>
  %23 = hivm.hir.vbrc ins(%extracted : i32) outs(%22 : tensor<1xi32>) -> tensor<1xi32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
  hivm.hir.store ins(%23 : tensor<1xi32>) outs(%reinterpret_cast_1 : memref<1xi32, strided<[1]>>)
  return
}
