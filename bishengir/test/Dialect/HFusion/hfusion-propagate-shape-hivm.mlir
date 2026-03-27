// RUN: bishengir-opt %s --propagate-reshape="for-hivm=true" --cse --canonicalize-ext -split-input-file | FileCheck %s

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
// CHECK-LABEL: @mm_03
func.func @mm_03(%arg0: memref<?xf16>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<256x64xf16>, %arg4: i32, %arg5: i32, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i1, %arg15: index) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
   %0 = bufferization.to_tensor %arg3 restrict writable : memref<256x64xf16>
   %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64], strides: [3] : memref<?xf32> to memref<64xf32, strided<[3]>>
   %alloc = memref.alloc() : memref<64xf32>
   hivm.hir.load ins(%reinterpret_cast : memref<64xf32, strided<[3]>>) outs(%alloc : memref<64xf32>)
   %1 = bufferization.to_tensor %alloc restrict writable : memref<64xf32>
   %2 = arith.muli %arg4, %arg10 : i32
   %3 = tensor.empty() : tensor<128x64xf32>
   %expanded = tensor.expand_shape %1 [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
   %4 = hivm.hir.vbrc ins(%expanded : tensor<1x64xf32>) outs(%3 : tensor<128x64xf32>) broadcast_dims = [0] -> tensor<128x64xf32>
   scf.for %arg16 = %arg11 to %arg13 step %arg5  : i32 {
     %5 = arith.muli %arg16, %arg12 : i32
     %6 = arith.addi %2, %5 : i32
     %7 = arith.index_cast %6 : i32 to index
     %8 = arith.muli %7, %arg9 : index
     %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%8], sizes: [128, 256], strides: [256, 1] : memref<?xf16> to memref<128x256xf16, strided<[256, 1], offset: ?>>
     %alloc_1 = memref.alloc() : memref<128x256xf16>
     %9 = arith.addi %7, %arg8 : index
     %10 = arith.maxsi %7, %arg7 : index
     %11 = arith.minsi %9, %10 : index
     %12 = arith.subi %11, %7 : index
     %13 = arith.minsi %12, %arg8 : index
    // CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<64xf32>
    // CHECK: hivm.hir.load ins({{.*}} : memref<64xf32, strided<[3]>>) outs(%[[alloc_1:.*]] : memref<64xf32>)
    // CHECK: %[[expand:.*]] = memref.expand_shape %[[alloc_1:.*]] {{\[\[0, *1\]\]}} output_shape [1, 64] : memref<64xf32> into memref<1x64xf32> 
    // CHECK: %[[bias:.*]] = bufferization.to_tensor %[[expand:.*]] restrict writable : memref<1x64xf32>
     %subview = memref.subview %reinterpret_cast_0[0, 0] [%13, 256] [1, 1] : memref<128x256xf16, strided<[256, 1], offset: ?>> to memref<?x256xf16, strided<[256, 1], offset: ?>>
     %subview_2 = memref.subview %alloc_1[0, 0] [%13, 256] [1, 1] : memref<128x256xf16> to memref<?x256xf16, strided<[256, 1]>>
     hivm.hir.load ins(%subview : memref<?x256xf16, strided<[256, 1], offset: ?>>) outs(%subview_2 : memref<?x256xf16, strided<[256, 1]>>) left_padding_num = %arg15 : index
     %14 = bufferization.to_tensor %alloc_1 restrict writable : memref<128x256xf16>
     %15 = hivm.hir.mmadL1 ins(%14, %0, %arg14, %arg15, %arg15, %arg15 : tensor<128x256xf16>, tensor<256x64xf16>, i1, index, index, index) outs(%4 : tensor<128x64xf32>) -> tensor<128x64xf32>
     %16 = arith.muli %7, %arg6 : index
     %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%16], sizes: [128, 64], strides: [64, 1] : memref<?xf32> to memref<128x64xf32, strided<[64, 1], offset: ?>>
     %extracted_slice = tensor.extract_slice %15[0, 0] [%13, 64] [1, 1] : tensor<128x64xf32> to tensor<?x64xf32>
     %subview_4 = memref.subview %reinterpret_cast_3[0, 0] [%13, 64] [1, 1] : memref<128x64xf32, strided<[64, 1], offset: ?>> to memref<?x64xf32, strided<[64, 1], offset: ?>>
     hivm.hir.store ins(%extracted_slice : tensor<?x64xf32>) outs(%subview_4 : memref<?x64xf32, strided<[64, 1], offset: ?>>)
   }
   return
 }

// -----
// CHECK-LABEL: func.func @unit_expand_shape(
func.func @unit_expand_shape(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: index) -> tensor<i32> attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, false, false, false]> : vector<7xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg2], sizes: [2, 3, 4], strides: [12, 4, 1] : memref<?xi32> to memref<2x3x4xi32, strided<[12, 4, 1], offset: ?>>
  %alloc = memref.alloc() : memref<2x3x4xi32>
  hivm.hir.load ins(%reinterpret_cast : memref<2x3x4xi32, strided<[12, 4, 1], offset: ?>>) outs(%alloc : memref<2x3x4xi32>) init_out_buffer = false
  %0 = bufferization.to_tensor %alloc restrict writable : memref<2x3x4xi32>
  %collapsed = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<2x3x4xi32> into tensor<24xi32>
  %1 = bufferization.alloc_tensor() : tensor<i32>
  %2 = tensor.empty() : tensor<1xi32>
  // CHECK: hivm.hir.vreduce <sum> ins
  // CHECK-SAME: tensor<2x3x4xi32>
  // CHECK-SAME tensor<1x1x1xi32>
  // CHECK-SAME reduce_dims = [0, 1, 2] -> tensor<1x1x1xi32>
  %3 = hivm.hir.vreduce <sum> ins(%collapsed : tensor<24xi32>) outs(%2 : tensor<1xi32>) reduce_dims = [0] -> tensor<1xi32>
  %collapsed_0 = tensor.collapse_shape %3 [] : tensor<1xi32> into tensor<i32>
  %extracted = tensor.extract %collapsed_0[] : tensor<i32>
  %4 = tensor.empty() : tensor<1xi32>
  %5 = hivm.hir.vbrc ins(%extracted : i32) outs(%4 : tensor<1xi32>) -> tensor<1xi32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
  hivm.hir.store ins(%5 : tensor<1xi32>) outs(%reinterpret_cast_1 : memref<1xi32, strided<[1]>>)
  return %1 : tensor<i32>
}

// -----
// CHECK: func.func @reinterpret_dynamic_stride(
// CHECK: hivm.hir.load
// CHECK-SAME: %{{.*}} : memref<2x2xf32, strided<[2, 1], offset: ?>>
// CHECK: hivm.hir.load
// CHECK: %{{.*}} : memref<2x2xf32, strided<[?, ?], offset: ?>>
func.func @reinterpret_dynamic_stride(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32, %arg3: index, %arg4: index, %arg5: tensor<2x2xf32>, %arg6: i1, %arg7: index) -> tensor<2x2xf32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg3], sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1], offset: ?>>
  %alloc = memref.alloc() : memref<4xf32>
  hivm.hir.load ins(%reinterpret_cast : memref<4xf32, strided<[1], offset: ?>>) outs(%alloc : memref<4xf32>) init_out_buffer = false
  %0 = bufferization.to_tensor %alloc restrict writable : memref<4xf32>
  %expanded = tensor.expand_shape %0 [[0, 1]] output_shape [2, 2] : tensor<4xf32> into tensor<2x2xf32>
  %1 = arith.index_cast %arg2 : i32 to index
  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%arg4], sizes: [4], strides: [%1] : memref<?xf32> to memref<4xf32, strided<[?], offset: ?>>
  %alloc_1 = memref.alloc() : memref<4xf32>
  hivm.hir.load ins(%reinterpret_cast_0 : memref<4xf32, strided<[?], offset: ?>>) outs(%alloc_1 : memref<4xf32>) init_out_buffer = false
  %2 = bufferization.to_tensor %alloc_1 restrict writable : memref<4xf32>
  %expanded_2 = tensor.expand_shape %2 [[0, 1]] output_shape [2, 2] : tensor<4xf32> into tensor<2x2xf32>
  hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 5
  %3 = hivm.hir.mmadL1 ins(%expanded, %expanded_2, %arg6, %arg7, %arg7, %arg7 : tensor<2x2xf32>, tensor<2x2xf32>, i1, index, index, index) outs(%arg5 : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %3 : tensor<2x2xf32>
}

// -----
// CHECK: func.func @alloc_expand_shape_all_one_not_propagate(
// CHECK: hivm.hir.load
// CHECK-SAME: %{{.*}} : memref<4xf32, strided<[1], offset: ?>>
// CHECK: hivm.hir.load
// CHECK: %{{.*}} : memref<4xf32, strided<[?], offset: ?>>
func.func @alloc_expand_shape_all_one_not_propagate(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32, %arg3: index, %arg4: index, %arg5: tensor<1x1xf32>, %arg6: i1, %arg7: index) -> tensor<1x1xf32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg3], sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32, strided<[1], offset: ?>>
  %alloc = memref.alloc() : memref<4xf32>
  hivm.hir.load ins(%reinterpret_cast : memref<4xf32, strided<[1], offset: ?>>) outs(%alloc : memref<4xf32>) init_out_buffer = false
  %0 = bufferization.to_tensor %alloc restrict writable : memref<4xf32>
  %expanded = tensor.expand_shape %0 [[0, 1]] output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>
  %1 = arith.index_cast %arg2 : i32 to index
  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%arg4], sizes: [4], strides: [%1] : memref<?xf32> to memref<4xf32, strided<[?], offset: ?>>
  %alloc_1 = memref.alloc() : memref<4xf32>
  hivm.hir.load ins(%reinterpret_cast_0 : memref<4xf32, strided<[?], offset: ?>>) outs(%alloc_1 : memref<4xf32>) init_out_buffer = false
  %2 = bufferization.to_tensor %alloc_1 restrict writable : memref<4xf32>
  %expanded_2 = tensor.expand_shape %2 [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32>
  hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 5
  %3 = hivm.hir.mmadL1 ins(%expanded, %expanded_2, %arg6, %arg7, %arg7, %arg7 : tensor<1x4xf32>, tensor<4x1xf32>, i1, index, index, index) outs(%arg5 : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %3 : tensor<1x1xf32>
}

// -----
// CHECK-LABEL: func.func @rearrange_cache_with_mask
// CHECK: %[[OUT_I1:.*]] = tensor.empty() : tensor<1x30xi1>
// CHECK: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG0:.*]], %[[ARG0]] : tensor<1x30xi1>, tensor<1x30xi1>) outs(%[[OUT_I1]] : tensor<1x30xi1>) -> tensor<1x30xi1>
// CHECK-NOT: tensor.collapse_shape %[[CMP]]
// CHECK: %[[OUT_F16:.*]] = tensor.empty() : tensor<30xf16>
// CHECK: %[[EXP:.*]] = tensor.expand_shape %[[OUT_F16]] {{\[\[0, *1\]\]}} output_shape [1, 30] : tensor<30xf16> into tensor<1x30xf16>
// CHECK: %[[CAST:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[CMP]] : tensor<1x30xi1>) outs(%[[EXP]] : tensor<1x30xf16>) -> tensor<1x30xf16>
// CHECK: %[[COLL:.*]] = tensor.collapse_shape %[[CAST]] {{\[\[0, *1\]\]}} : tensor<1x30xf16> into tensor<30xf16>
module {
  func.func @rearrange_cache_with_mask(%arg0: tensor<1x30xi1>) -> tensor<30xf16> {
    %0 = tensor.empty() : tensor<1x30xi1>
    %1 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
      ins(%arg0, %arg0 : tensor<1x30xi1>, tensor<1x30xi1>) outs(%0 : tensor<1x30xi1>) -> tensor<1x30xi1>

    %collapsed_in = tensor.collapse_shape %1 [[0, 1]] : tensor<1x30xi1> into tensor<30xi1>
    %2 = tensor.empty() : tensor<30xf16>
    %3 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>}
      ins(%collapsed_in : tensor<30xi1>) outs(%2 : tensor<30xf16>) -> tensor<30xf16>
    return %3 : tensor<30xf16>
  }
}

// -----
// CHECK-LABEL: @test_trivial_subview
// CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[ARG2:.*]] to offset: {{\[}}%[[ARG0:.*]]], sizes: [128, 1], strides: [8, 8] : memref<?xf32> to memref<128x1xf32, strided<[8, 8], offset: ?>>
// CHECK: %[[SUBVIEW_1:.*]] = memref.subview %[[CAST]][0, 0] {{\[}}%[[ARG1:.*]], 1] [1, 1] : memref<128x1xf32, strided<[8, 8], offset: ?>> to memref<?x1xf32, strided<[8, 8], offset: ?>>
// CHECK: %[[SUBVIEW_2:.*]] = memref.subview %[[ALLOC:.*]][0, 0] {{\[}}%[[ARG1]], 1] [1, 1] : memref<128x1xf32> to memref<?x1xf32, strided<[1, 1]>>
// CHECK: hivm.hir.load ins(%[[SUBVIEW_1]] : memref<?x1xf32, strided<[8, 8], offset: ?>>) outs(%[[SUBVIEW_2]] : memref<?x1xf32, strided<[1, 1]>>)
func.func @test_trivial_subview(%arg0: index, %arg1: index, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}) -> tensor<128x1xf32> {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%arg0], sizes: [128], strides: [8] : memref<?xf32> to memref<128xf32, strided<[8], offset: ?>>
  %alloc = memref.alloc() : memref<128x1xf32>
  %collapse_shape = memref.collapse_shape %alloc [[0, 1]] : memref<128x1xf32> into memref<128xf32>
  %subview = memref.subview %reinterpret_cast[0] [%arg1] [1] : memref<128xf32, strided<[8], offset: ?>> to memref<?xf32, strided<[8], offset: ?>>
  %subview_0 = memref.subview %collapse_shape[0] [%arg1] [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
  hivm.hir.load ins(%subview : memref<?xf32, strided<[8], offset: ?>>) outs(%subview_0 : memref<?xf32, strided<[1]>>) may_implicit_transpose_with_last_axis = false
  %0 = bufferization.to_tensor %alloc restrict writable : memref<128x1xf32>
  return %0 : tensor<128x1xf32>
}

// -----
// CHECK-LABEL: func.func @reinterpret_cast
// CHECK-SAME: %[[VAL_0:.*]]: memref<?xf32>,
// CHECK-SAME: %[[VAL_1:.*]]: index) -> tensor<16x16xf32> {
// CHECK: %{{.*}} = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [16, 1], strides: {{\[}}%[[VAL_1]], %[[VAL_1]]] : memref<?xf32> to memref<16x1xf32, strided<[?, ?]>>
func.func @reinterpret_cast(%arg0: memref<?xf32>, %arg1: index) -> tensor<16x16xf32> {
  %alloc = memref.alloc() : memref<16x16xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16], strides: [%arg1] : memref<?xf32> to memref<16xf32, strided<[?]>>
  %expand_shape = memref.expand_shape %reinterpret_cast [[0, 1]] output_shape [16, 1] : memref<16xf32, strided<[?]>> into memref<16x1xf32, strided<[?, ?]>>
  %alloc_0 = memref.alloc() : memref<16x1xf32>
  %collapse_shape = memref.collapse_shape %alloc_0 [[0, 1]] : memref<16x1xf32> into memref<16xf32>
  hivm.hir.load ins(%expand_shape : memref<16x1xf32, strided<[?, ?]>>) outs(%alloc_0 : memref<16x1xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  return %0 : tensor<16x16xf32>
}