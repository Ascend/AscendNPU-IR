// RUN: bishengir-opt %s -hivm-fuse-transpose-into-load | FileCheck %s

// CHECK-LABEL: func.func @fuse_load_with_dyn_size
// CHECK:       %[[res:.*]] = bufferization.to_tensor
// CHECK:       hivm.hir.load ins(%{{.*}} : memref<256x?xbf16, strided<[1, 256], offset: ?>>)
// CHECK-SAME:                outs(%{{.*}} : memref<256x?xbf16, strided<[128, 1], offset: ?>>)
// CHECK-NOT:   transpose
// CHECK:       return %[[res]] : tensor<256x128xbf16>
func.func @fuse_load_with_dyn_size(%arg0: memref<?xbf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: i1) -> tensor<256x128xbf16> {
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %alloc = memref.alloc() : memref<128x256xbf16>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [128, 256], strides: [256, 1] : memref<?xbf16> to memref<128x256xbf16, strided<[256, 1], offset: ?>>
  %subview = memref.subview %reinterpret_cast[0, 0] [%arg3, 256] [1, 1] : memref<128x256xbf16, strided<[256, 1], offset: ?>> to memref<?x256xbf16, strided<[256, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[%arg2, 0] [%arg3, 256] [1, 1] : memref<128x256xbf16> to memref<?x256xbf16, strided<[256, 1], offset: ?>>
  %0 = arith.cmpi slt, %arg3, %c128 : index
  %1 = arith.ori %arg4, %0 : i1
  scf.if %1 {
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<128x256xbf16>)
  }
  hivm.hir.load ins(%subview : memref<?x256xbf16, strided<[256, 1], offset: ?>>) outs(%subview_0 : memref<?x256xbf16, strided<[256, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : bf16
  %2 = bufferization.to_tensor %alloc restrict writable : memref<128x256xbf16>
  %3 = tensor.empty() : tensor<256x128xbf16>
  %transposed = linalg.transpose ins(%2 : tensor<128x256xbf16>) outs(%3 : tensor<256x128xbf16>) permutation = [1, 0]
  return %transposed : tensor<256x128xbf16>
}

// -----

// CHECK-LABEL: func.func @fuse_load_with_min_max_bounded_sizes
// CHECK:       hivm.hir.load ins(%{{.*}} : memref<?x?xf16, strided<[1, 128], offset: ?>>)
// CHECK-SAME:                outs(%{{.*}} : memref<?x?xf16, strided<[32, 1]>>)
// CHECK-NOT:   linalg.transpose
// CHECK:       return %{{.*}} : tensor<32x32xf16>
func.func @fuse_load_with_min_max_bounded_sizes(%arg0: memref<?xf16>, %arg1: index, %arg2: index, %arg3: index) -> tensor<32x32xf16> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<32x32xf16>
  linalg.fill ins(%cst : f16) outs(%alloc : memref<32x32xf16>)
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [32, 32], strides: [128, 1] : memref<?xf16> to memref<32x32xf16, strided<[128, 1], offset: ?>>
  %0 = arith.maxsi %arg2, %c0 : index
  %1 = arith.minsi %0, %c32 : index
  %2 = arith.maxsi %arg3, %c0 : index
  %3 = arith.minsi %2, %c32 : index
  %subview = memref.subview %reinterpret_cast[0, 0] [%1, %3] [1, 1] : memref<32x32xf16, strided<[128, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[0, 0] [%1, %3] [1, 1] : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1]>>
  hivm.hir.load ins(%subview : memref<?x?xf16, strided<[128, 1], offset: ?>>) outs(%subview_0 : memref<?x?xf16, strided<[32, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16
  %4 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf16>
  %5 = tensor.empty() : tensor<32x32xf16>
  %transposed = linalg.transpose ins(%4 : tensor<32x32xf16>) outs(%5 : tensor<32x32xf16>) permutation = [1, 0]
  return %transposed : tensor<32x32xf16>
}

// -----

// CHECK-LABEL: func.func @fuse_loop_tail_rank_reduced_dst_with_bounded_sizes
// CHECK:       scf.for
// CHECK:       %[[ALLOC:.*]] = memref.alloc() : memref<2x32x32xf16>
// CHECK:       %[[LOAD_DST:.*]] = memref.subview %[[ALLOC]][%{{.*}}, 0, 0] [1, %{{.*}}, %{{.*}}] [1, 1, 1] : memref<2x32x32xf16> to memref<?x?xf16, strided<[32, 1], offset: ?>>
// CHECK:       %[[READ:.*]] = memref.subview %[[ALLOC]][%{{.*}}, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
// CHECK:       %[[RETURN:.*]] = bufferization.to_tensor %[[READ]] restrict writable : memref<32x32xf16, strided<[32, 1], offset: ?>>
// CHECK:       hivm.hir.load ins(%{{.*}} : memref<?x?xf16, strided<[1, 128], offset: ?>>)
// CHECK-SAME:                outs(%[[LOAD_DST]] : memref<?x?xf16, strided<[32, 1], offset: ?>>)
// CHECK-NOT:   linalg.transpose
// CHECK:       scf.yield %[[RETURN]] : tensor<32x32xf16>
#map = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>
#map2 = affine_map<()[s0, s1] -> (-s1 - s0 floordiv 128 + 128)>
#map3 = affine_map<()[s0] -> (-s0 + (s0 floordiv 128) * 128 + 128)>
func.func @fuse_loop_tail_rank_reduced_dst_with_bounded_sizes(%arg0: memref<?xf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: tensor<32x32xf16>) -> tensor<32x32xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<2x32x32xf16>
  %0 = scf.for %arg5 = %c0 to %c2 step %c1 iter_args(%arg6 = %arg4) -> (tensor<32x32xf16>) {
    %subview = memref.subview %alloc[%arg5, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
    %1 = affine.apply #map(%arg5)[%arg2]
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.muli %2, %c32_i32 : i32
    %4 = arith.maxsi %3, %c0_i32 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = affine.apply #map1()[%5, %arg1, %arg3]
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%6], sizes: [32, 32], strides: [128, 1] : memref<?xf16> to memref<32x32xf16, strided<[128, 1], offset: ?>>
    %7 = affine.apply #map2()[%5, %arg3]
    %8 = arith.maxsi %7, %c0 : index
    %9 = arith.minsi %8, %c32 : index
    %10 = affine.apply #map3()[%5]
    %11 = arith.maxsi %10, %c0 : index
    %12 = arith.minsi %11, %c32 : index
    %13 = arith.cmpi slt, %9, %c32 : index
    %14 = arith.cmpi slt, %12, %c32 : index
    %15 = arith.ori %13, %14 : i1
    %subview_0 = memref.subview %reinterpret_cast[0, 0] [%9, %12] [1, 1] : memref<32x32xf16, strided<[128, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1], offset: ?>>
    %subview_1 = memref.subview %alloc[%arg5, 0, 0] [1, %9, %12] [1, 1, 1] : memref<2x32x32xf16> to memref<?x?xf16, strided<[32, 1], offset: ?>>
    scf.if %15 {
      linalg.fill ins(%cst : f16) outs(%alloc : memref<2x32x32xf16>)
    }
    hivm.hir.load ins(%subview_0 : memref<?x?xf16, strided<[128, 1], offset: ?>>) outs(%subview_1 : memref<?x?xf16, strided<[32, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f16
    %16 = bufferization.to_tensor %subview restrict writable : memref<32x32xf16, strided<[32, 1], offset: ?>>
    %17 = tensor.empty() : tensor<32x32xf16>
    %transposed = linalg.transpose ins(%16 : tensor<32x32xf16>) outs(%17 : tensor<32x32xf16>) permutation = [1, 0]
    scf.yield %transposed : tensor<32x32xf16>
  }
  return %0 : tensor<32x32xf16>
}

// -----

// CHECK-LABEL: func.func @fuse_load_with_generic_fill
// CHECK:       %[[res:.*]] = bufferization.to_tensor
// CHECK:       hivm.hir.load ins(%{{.*}} : memref<32x32xf16, strided<{{\[}}1, 128{{\]}}, offset: ?>>)
// CHECK-SAME:                outs(%{{.*}} : memref<32x32xf16, strided<{{\[}}32, 1{{\]}}, offset: ?>>)
// CHECK-NOT:   transpose
// CHECK:       return %[[res]] : tensor<32x32xf16>
func.func @fuse_load_with_generic_fill(%arg0: memref<?xf16>, %arg1: index, %arg2: index) -> tensor<32x32xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<2x32x32xf16>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc : memref<2x32x32xf16>) {
  ^bb0(%out: f16):
    linalg.yield %cst : f16
  }
  %subview = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [32, 32], strides: [128, 1] : memref<?xf16> to memref<32x32xf16, strided<[128, 1], offset: ?>>
  hivm.hir.load ins(%reinterpret_cast : memref<32x32xf16, strided<[128, 1], offset: ?>>) outs(%subview : memref<32x32xf16, strided<[32, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f16
  %subview_0 = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %0 = bufferization.to_tensor %subview_0 restrict writable : memref<32x32xf16, strided<[32, 1], offset: ?>>
  %1 = tensor.empty() : tensor<32x32xf16>
  %transposed = linalg.transpose ins(%0 : tensor<32x32xf16>) outs(%1 : tensor<32x32xf16>) permutation = [1, 0]
  return %transposed : tensor<32x32xf16>
}

// -----

// CHECK-LABEL: func.func @fuse_load_dst_with_rank_reduce_subview
// CHECK:       %[[res:.*]] = bufferization.to_tensor
// CHECK:       hivm.hir.load ins(%{{.*}} : memref<32x32xf16, strided<[1, 128], offset: ?>>)
// CHECK-SAME:                outs(%{{.*}} : memref<32x32xf16, strided<[32, 1], offset: ?>>)
// CHECK-NOT:   transpose
// CHECK:       return %[[res]] : tensor<32x32xf16>
func.func @fuse_load_dst_with_rank_reduce_subview(%arg0: memref<?xf16>, %arg1: index, %arg2: index) -> tensor<32x32xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<2x32x32xf16>
  linalg.fill ins(%cst : f16) outs(%alloc : memref<2x32x32xf16>)
  %subview = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [32, 32], strides: [128, 1] : memref<?xf16> to memref<32x32xf16, strided<[128, 1], offset: ?>>
  hivm.hir.load ins(%reinterpret_cast : memref<32x32xf16, strided<[128, 1], offset: ?>>) outs(%subview : memref<32x32xf16, strided<[32, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f16
  %subview_0 = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %0 = bufferization.to_tensor %subview_0 restrict writable : memref<32x32xf16, strided<[32, 1], offset: ?>>
  %1 = tensor.empty() : tensor<32x32xf16>
  %transposed = linalg.transpose ins(%0 : tensor<32x32xf16>) outs(%1 : tensor<32x32xf16>) permutation = [1, 0]
  return %transposed : tensor<32x32xf16>
}

// -----

// CHECK-LABEL: func.func @fuse_load_dst_with_rank_reduce_rectangular
func.func @fuse_load_dst_with_rank_reduce_rectangular(%arg0: memref<?xf16>, %arg1: index, %arg2: index) -> tensor<32x16xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<2x16x32xf16>
  linalg.fill ins(%cst : f16) outs(%alloc : memref<2x16x32xf16>)
  %subview = memref.subview %alloc[%arg2, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xf16> to memref<16x32xf16, strided<[32, 1], offset: ?>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [16, 32], strides: [128, 1] : memref<?xf16> to memref<16x32xf16, strided<[128, 1], offset: ?>>
  hivm.hir.load ins(%reinterpret_cast : memref<16x32xf16, strided<[128, 1], offset: ?>>) outs(%subview : memref<16x32xf16, strided<[32, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f16
  %subview_0 = memref.subview %alloc[%arg2, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xf16> to memref<16x32xf16, strided<[32, 1], offset: ?>>
  %0 = bufferization.to_tensor %subview_0 restrict writable : memref<16x32xf16, strided<[32, 1], offset: ?>>
  %1 = tensor.empty() : tensor<32x16xf16>
  %transposed = linalg.transpose ins(%0 : tensor<16x32xf16>) outs(%1 : tensor<32x16xf16>) permutation = [1, 0]
  return %transposed : tensor<32x16xf16>
}




// -----

// CHECK-LABEL: func.func @fuse_load_rectangular_padded_dst_nested_src_subview
// CHECK:       %[[res:.*]] = bufferization.to_tensor
// CHECK:       hivm.hir.load ins(%{{.*}} : memref<?x?xbf16, strided<[1, 128], offset: ?>>)
// CHECK-SAME:                outs(%{{.*}} : memref<?x?xbf16, strided<[64, 1], offset: ?>>)
// CHECK-NOT:   linalg.transpose
// CHECK:       return %[[res]] : tensor<32x64xbf16>
func.func @fuse_load_rectangular_padded_dst_nested_src_subview(%arg0: memref<?xbf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: i32, %arg5: index) -> tensor<32x64xbf16> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %col_base = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg5]
  %col_limit = affine.apply affine_map<()[s0] -> (s0 * 32 + 32)>()[%arg5]
  %alloc = memref.alloc() : memref<64x32xbf16>
  %row_max = arith.maxsi %arg3, %c0 : index
  %row_extent = arith.minsi %row_max, %c64 : index
  %col_max = arith.maxsi %arg2, %c0 : index
  %col_extent = arith.minsi %col_max, %c64 : index
  %row_pad = arith.subi %c0_i32, %arg4 : i32
  %row_pad_nonneg = arith.maxsi %row_pad, %c0_i32 : i32
  %row_off = arith.index_cast %row_pad_nonneg : i32 to index
  %dst_row = arith.minsi %row_off, %row_extent : index
  %col_pad = arith.minsi %arg5, %col_extent : index
  %col_pad2 = arith.minsi %col_pad, %col_extent : index
  %row_size = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%row_extent, %dst_row]
  %col_size_src = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%col_extent, %col_pad2]
  %src_col_size = arith.minsi %col_base, %col_size_src : index
  %src_col_size2 = affine.apply affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2)>()[%src_col_size, %col_extent, %col_pad2]
  %src_col_size3 = arith.minsi %src_col_size2, %c32 : index
  %dst_col_base = arith.maxsi %col_pad2, %col_base : index
  %dst_col_bound = arith.minsi %col_extent, %col_limit : index
  %dst_col_max = arith.maxsi %dst_col_base, %dst_col_bound : index
  %dst_col_size = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%dst_col_max, %dst_col_base]
  %dst_col_off = affine.apply affine_map<()[s0, s1] -> (s0 - s1 * 32)>()[%dst_col_base, %arg5]
  %needs_fill = arith.cmpi slt, %row_size, %c64 : index
  linalg.fill ins(%cst : bf16) outs(%alloc : memref<64x32xbf16>)
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [64, 64], strides: [128, 1] : memref<?xbf16> to memref<64x64xbf16, strided<[128, 1], offset: ?>>
  %subview = memref.subview %reinterpret_cast[0, %col_base] [64, 32] [1, 1] : memref<64x64xbf16, strided<[128, 1], offset: ?>> to memref<64x32xbf16, strided<[128, 1], offset: ?>>
  %subview_0 = memref.subview %subview[0, 0] [%row_size, %src_col_size3] [1, 1] : memref<64x32xbf16, strided<[128, 1], offset: ?>> to memref<?x?xbf16, strided<[128, 1], offset: ?>>
  %subview_1 = memref.subview %alloc[%dst_row, %dst_col_off] [%row_size, %dst_col_size] [1, 1] : memref<64x32xbf16> to memref<?x?xbf16, strided<[32, 1], offset: ?>>
  scf.if %needs_fill {
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<64x32xbf16>)
  }
  hivm.hir.load ins(%subview_0 : memref<?x?xbf16, strided<[128, 1], offset: ?>>) outs(%subview_1 : memref<?x?xbf16, strided<[32, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : bf16
  %5 = bufferization.to_tensor %alloc restrict writable : memref<64x32xbf16>
  %6 = tensor.empty() : tensor<32x64xbf16>
  %transposed = linalg.transpose ins(%5 : tensor<64x32xbf16>) outs(%6 : tensor<32x64xbf16>) permutation = [1, 0]
  return %transposed : tensor<32x64xbf16>
}

// -----

// CHECK-LABEL: func.func @no_fuse_load_with_padding
// CHECK: linalg.transpose
func.func @no_fuse_load_with_padding(%arg0: memref<?xf16>, %arg1: index, %arg2: index) -> tensor<32x32xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<2x32x32xf16>
  linalg.fill ins(%cst : f16) outs(%alloc : memref<2x32x32xf16>)
  %subview = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [32, 32], strides: [128, 1] : memref<?xf16> to memref<32x32xf16, strided<[128, 1], offset: ?>>
  hivm.hir.load ins(%reinterpret_cast : memref<32x32xf16, strided<[128, 1], offset: ?>>) outs(%subview : memref<32x32xf16, strided<[32, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index
  %subview_0 = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %0 = bufferization.to_tensor %subview_0 restrict writable : memref<32x32xf16, strided<[32, 1], offset: ?>>
  %1 = tensor.empty() : tensor<32x32xf16>
  %transposed = linalg.transpose ins(%0 : tensor<32x32xf16>) outs(%1 : tensor<32x32xf16>) permutation = [1, 0]
  return %transposed : tensor<32x32xf16>
}

// CHECK-LABEL: func.func @no_fuse_multi_use_to_tensor
// CHECK: linalg.transpose
func.func @no_fuse_multi_use_to_tensor(%arg0: memref<?xf16>, %arg1: index, %arg2: index) -> (tensor<32x32xf16>, tensor<32x32xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<2x32x32xf16>
  linalg.fill ins(%cst : f16) outs(%alloc : memref<2x32x32xf16>)
  %subview = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [32, 32], strides: [128, 1] : memref<?xf16> to memref<32x32xf16, strided<[128, 1], offset: ?>>
  hivm.hir.load ins(%reinterpret_cast : memref<32x32xf16, strided<[128, 1], offset: ?>>) outs(%subview : memref<32x32xf16, strided<[32, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f16
  %subview_0 = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %0 = bufferization.to_tensor %subview_0 restrict writable : memref<32x32xf16, strided<[32, 1], offset: ?>>
  %1 = tensor.empty() : tensor<32x32xf16>
  %transposed = linalg.transpose ins(%0 : tensor<32x32xf16>) outs(%1 : tensor<32x32xf16>) permutation = [1, 0]
  return %transposed, %0 : tensor<32x32xf16>, tensor<32x32xf16>
}
