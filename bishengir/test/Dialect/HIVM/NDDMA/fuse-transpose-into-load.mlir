// RUN: bishengir-opt %s -hivm-fuse-transpose-into-load | FileCheck %s

// Dst subview offset %arg2 becomes last-dim offset after last-two permute.
// DMA pad cannot write [0, %arg2); the fill must follow the new dest, but
// only when that offset is not 0 (runtime skip recovers last-dim-only kda).
// CHECK-LABEL: func.func @fuse_load_with_dyn_size
// CHECK:       %[[NEW:.*]] = memref.alloc() : memref<256x128xbf16>
// CHECK:       scf.if
// CHECK:         linalg.fill ins(%{{.*}} : bf16) outs(%[[NEW]] : memref<256x128xbf16>)
// Dynamic last-dim offset: keep fill, do not expand dest to rootLast-offset.
// CHECK-NOT:   arith.subi
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<256x?xbf16, strided<[1, 256], offset: ?>> to memref<256x?xbf16, strided<[128, 1], offset: ?>>
// CHECK:       %[[res:.*]] = bufferization.to_tensor
// CHECK-NOT:   transpose
// CHECK:       return %[[res]] : tensor<256x128xbf16>
func.func @fuse_load_with_dyn_size(%arg0: memref<?xbf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: i1) -> tensor<256x128xbf16> attributes {mix_mode = "aiv"} {
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
  memref.copy %subview, %subview_0 : memref<?x256xbf16, strided<[256, 1], offset: ?>> to memref<?x256xbf16, strided<[256, 1], offset: ?>>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<128x256xbf16>
  %3 = tensor.empty() : tensor<256x128xbf16>
  %transposed = linalg.transpose ins(%2 : tensor<128x256xbf16>) outs(%3 : tensor<256x128xbf16>) permutation = [1, 0]
  return %transposed : tensor<256x128xbf16>
}

// -----

// Same last-dim-only pad as fuse_load_with_dyn_size, but the dest row offset is
// IndexBound-proven 0 (`%arg2 - %arg2`). Drop fill, do not expand last dim.
// CHECK-LABEL: func.func @fuse_load_last_dim_proven_zero_offset
// CHECK:       %[[NEW:.*]] = memref.alloc() : memref<256x128xbf16>
// CHECK-NOT:   arith.cmpi ne
// CHECK:       linalg.fill
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<256x?xbf16, strided<[1, 256], offset: ?>> to memref<256x?xbf16, strided<[128, 1], offset: ?>>
// CHECK-NOT:   transpose
// CHECK:       return %{{.*}} : tensor<256x128xbf16>
func.func @fuse_load_last_dim_proven_zero_offset(%arg0: memref<?xbf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: i1) -> tensor<256x128xbf16> attributes {mix_mode = "aiv"} {
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %alloc = memref.alloc() : memref<128x256xbf16>
  %off = arith.subi %arg2, %arg2 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [128, 256], strides: [256, 1] : memref<?xbf16> to memref<128x256xbf16, strided<[256, 1], offset: ?>>
  %subview = memref.subview %reinterpret_cast[0, 0] [%arg3, 256] [1, 1] : memref<128x256xbf16, strided<[256, 1], offset: ?>> to memref<?x256xbf16, strided<[256, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[%off, 0] [%arg3, 256] [1, 1] : memref<128x256xbf16> to memref<?x256xbf16, strided<[256, 1], offset: ?>>
  %0 = arith.cmpi slt, %arg3, %c128 : index
  %1 = arith.ori %arg4, %0 : i1
  scf.if %1 {
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<128x256xbf16>)
  }
  memref.copy %subview, %subview_0 : memref<?x256xbf16, strided<[256, 1], offset: ?>> to memref<?x256xbf16, strided<[256, 1], offset: ?>>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<128x256xbf16>
  %3 = tensor.empty() : tensor<256x128xbf16>
  %transposed = linalg.transpose ins(%2 : tensor<128x256xbf16>) outs(%3 : tensor<256x128xbf16>) permutation = [1, 0]
  return %transposed : tensor<256x128xbf16>
}

// -----

// CHECK-LABEL: func.func @fuse_load_with_min_max_bounded_sizes
// CHECK:       %[[ALLOC:.*]] = memref.alloc() : memref<32x32xf16>
// CHECK:       linalg.fill ins(%{{.*}} : f16) outs(%[[ALLOC]] : memref<32x32xf16>)
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<?x?xf16, strided<[1, 128], offset: ?>> to memref<?x?xf16, strided<[32, 1]>>
// CHECK-NOT:   linalg.transpose
// CHECK:       return %{{.*}} : tensor<32x32xf16>
func.func @fuse_load_with_min_max_bounded_sizes(%arg0: memref<?xf16>, %arg1: index, %arg2: index, %arg3: index) -> tensor<32x32xf16> attributes {mix_mode = "aiv"} {
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
  memref.copy %subview, %subview_0 : memref<?x?xf16, strided<[128, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1]>>
  %4 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf16>
  %5 = tensor.empty() : tensor<32x32xf16>
  %transposed = linalg.transpose ins(%4 : tensor<32x32xf16>) outs(%5 : tensor<32x32xf16>) permutation = [1, 0]
  return %transposed : tensor<32x32xf16>
}

// -----

// CHECK-LABEL: func.func @fuse_loop_tail_rank_reduced_dst_with_bounded_sizes
// CHECK:       scf.for
// CHECK:       %[[ALLOC:.*]] = memref.alloc() : memref<2x32x32xf16>
// CHECK:       %[[READ:.*]] = memref.subview %[[ALLOC]][%{{.*}}, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<?x?xf16, strided<[1, 128], offset: ?>> to memref<?x?xf16, strided<[32, 1], offset: ?>>
// CHECK:       %[[RETURN:.*]] = bufferization.to_tensor %[[READ]] restrict writable : memref<32x32xf16, strided<[32, 1], offset: ?>>
// CHECK-NOT:   linalg.transpose
// CHECK:       scf.yield %[[RETURN]] : tensor<32x32xf16>
#map = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>
#map2 = affine_map<()[s0, s1] -> (-s1 - s0 floordiv 128 + 128)>
#map3 = affine_map<()[s0] -> (-s0 + (s0 floordiv 128) * 128 + 128)>
func.func @fuse_loop_tail_rank_reduced_dst_with_bounded_sizes(%arg0: memref<?xf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: tensor<32x32xf16>) -> tensor<32x32xf16> attributes {mix_mode = "aiv"} {
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
    memref.copy %subview_0, %subview_1 : memref<?x?xf16, strided<[128, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1], offset: ?>>
    %16 = bufferization.to_tensor %subview restrict writable : memref<32x32xf16, strided<[32, 1], offset: ?>>
    %17 = tensor.empty() : tensor<32x32xf16>
    %transposed = linalg.transpose ins(%16 : tensor<32x32xf16>) outs(%17 : tensor<32x32xf16>) permutation = [1, 0]
    scf.yield %transposed : tensor<32x32xf16>
  }
  return %0 : tensor<32x32xf16>
}

// -----

// Generic fill is not linalg.fill; inspectLoadUses must reject it so the
// transpose is not fused.
// CHECK-LABEL: func.func @fuse_load_with_generic_fill
// CHECK:       linalg.generic
// CHECK:       memref.copy
// CHECK:       linalg.transpose
func.func @fuse_load_with_generic_fill(%arg0: memref<?xf16>, %arg1: index, %arg2: index) -> tensor<32x32xf16> attributes {mix_mode = "aiv"} {
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<2x32x32xf16>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc : memref<2x32x32xf16>) {
  ^bb0(%out: f16):
    linalg.yield %cst : f16
  }
  %subview = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [32, 32], strides: [128, 1] : memref<?xf16> to memref<32x32xf16, strided<[128, 1], offset: ?>>
  memref.copy %reinterpret_cast, %subview : memref<32x32xf16, strided<[128, 1], offset: ?>> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %0 = bufferization.to_tensor %subview_0 restrict writable : memref<32x32xf16, strided<[32, 1], offset: ?>>
  %1 = tensor.empty() : tensor<32x32xf16>
  %transposed = linalg.transpose ins(%0 : tensor<32x32xf16>) outs(%1 : tensor<32x32xf16>) permutation = [1, 0]
  return %transposed : tensor<32x32xf16>
}

// -----

// CHECK-LABEL: func.func @fuse_load_dst_with_rank_reduce_subview
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<32x32xf16, strided<[1, 128], offset: ?>> to memref<32x32xf16, strided<[32, 1], offset: ?>>
// CHECK:       %[[res:.*]] = bufferization.to_tensor
// CHECK-NOT:   transpose
// CHECK:       return %[[res]] : tensor<32x32xf16>
func.func @fuse_load_dst_with_rank_reduce_subview(%arg0: memref<?xf16>, %arg1: index, %arg2: index) -> tensor<32x32xf16> attributes {mix_mode = "aiv"} {
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<2x32x32xf16>
  linalg.fill ins(%cst : f16) outs(%alloc : memref<2x32x32xf16>)
  %subview = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [32, 32], strides: [128, 1] : memref<?xf16> to memref<32x32xf16, strided<[128, 1], offset: ?>>
  memref.copy %reinterpret_cast, %subview : memref<32x32xf16, strided<[128, 1], offset: ?>> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %0 = bufferization.to_tensor %subview_0 restrict writable : memref<32x32xf16, strided<[32, 1], offset: ?>>
  %1 = tensor.empty() : tensor<32x32xf16>
  %transposed = linalg.transpose ins(%0 : tensor<32x32xf16>) outs(%1 : tensor<32x32xf16>) permutation = [1, 0]
  return %transposed : tensor<32x32xf16>
}

// -----

// CHECK-LABEL: func.func @fuse_load_dst_with_rank_reduce_rectangular
func.func @fuse_load_dst_with_rank_reduce_rectangular(%arg0: memref<?xf16>, %arg1: index, %arg2: index) -> tensor<32x16xf16> attributes {mix_mode = "aiv"} {
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<2x16x32xf16>
  linalg.fill ins(%cst : f16) outs(%alloc : memref<2x16x32xf16>)
  %subview = memref.subview %alloc[%arg2, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xf16> to memref<16x32xf16, strided<[32, 1], offset: ?>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [16, 32], strides: [128, 1] : memref<?xf16> to memref<16x32xf16, strided<[128, 1], offset: ?>>
  memref.copy %reinterpret_cast, %subview : memref<16x32xf16, strided<[128, 1], offset: ?>> to memref<16x32xf16, strided<[32, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[%arg2, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xf16> to memref<16x32xf16, strided<[32, 1], offset: ?>>
  %0 = bufferization.to_tensor %subview_0 restrict writable : memref<16x32xf16, strided<[32, 1], offset: ?>>
  %1 = tensor.empty() : tensor<32x16xf16>
  %transposed = linalg.transpose ins(%0 : tensor<16x32xf16>) outs(%1 : tensor<32x16xf16>) permutation = [1, 0]
  return %transposed : tensor<32x16xf16>
}

// -----

// The dest row offset is dynamic, so nothing proves it is zero and the copy
// is left alone - the load form used to fuse this because its pad covered the
// head of the row.
// CHECK-LABEL: func.func @no_fuse_rectangular_dst_dynamic_last_dim_offset
// CHECK:       linalg.transpose
func.func @no_fuse_rectangular_dst_dynamic_last_dim_offset(%arg0: memref<?xbf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: i32, %arg5: index) -> tensor<32x64xbf16> attributes {mix_mode = "aiv"} {
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
  memref.copy %subview_0, %subview_1 : memref<?x?xbf16, strided<[128, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1], offset: ?>>
  %5 = bufferization.to_tensor %alloc restrict writable : memref<64x32xbf16>
  %6 = tensor.empty() : tensor<32x64xbf16>
  %transposed = linalg.transpose ins(%5 : tensor<64x32xbf16>) outs(%6 : tensor<32x64xbf16>) permutation = [1, 0]
  return %transposed : tensor<32x64xbf16>
}

// -----

// CHECK-LABEL: func.func @no_fuse_load_with_padding
// CHECK: linalg.transpose
func.func @no_fuse_load_with_padding(%arg0: memref<?xf16>, %arg1: index, %arg2: index) -> tensor<32x32xf16> attributes {mix_mode = "aiv"} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<2x32x32xf16>
  linalg.fill ins(%cst : f16) outs(%alloc : memref<2x32x32xf16>)
  %subview = memref.subview %alloc[%arg2, 0, 3] [1, 32, 29] [1, 1, 1] : memref<2x32x32xf16> to memref<32x29xf16, strided<[32, 1], offset: ?>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [32, 32], strides: [128, 1] : memref<?xf16> to memref<32x32xf16, strided<[128, 1], offset: ?>>
  %src = memref.subview %reinterpret_cast[0, 0] [32, 29] [1, 1] : memref<32x32xf16, strided<[128, 1], offset: ?>> to memref<32x29xf16, strided<[128, 1], offset: ?>>
  memref.copy %src, %subview : memref<32x29xf16, strided<[128, 1], offset: ?>> to memref<32x29xf16, strided<[32, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %0 = bufferization.to_tensor %subview_0 restrict writable : memref<32x32xf16, strided<[32, 1], offset: ?>>
  %1 = tensor.empty() : tensor<32x32xf16>
  %transposed = linalg.transpose ins(%0 : tensor<32x32xf16>) outs(%1 : tensor<32x32xf16>) permutation = [1, 0]
  return %transposed : tensor<32x32xf16>
}

// CHECK-LABEL: func.func @no_fuse_multi_use_to_tensor
// CHECK: linalg.transpose
func.func @no_fuse_multi_use_to_tensor(%arg0: memref<?xf16>, %arg1: index, %arg2: index) -> (tensor<32x32xf16>, tensor<32x32xf16>) attributes {mix_mode = "aiv"} {
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<2x32x32xf16>
  linalg.fill ins(%cst : f16) outs(%alloc : memref<2x32x32xf16>)
  %subview = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [32, 32], strides: [128, 1] : memref<?xf16> to memref<32x32xf16, strided<[128, 1], offset: ?>>
  memref.copy %reinterpret_cast, %subview : memref<32x32xf16, strided<[128, 1], offset: ?>> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[%arg2, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
  %0 = bufferization.to_tensor %subview_0 restrict writable : memref<32x32xf16, strided<[32, 1], offset: ?>>
  %1 = tensor.empty() : tensor<32x32xf16>
  %transposed = linalg.transpose ins(%0 : tensor<32x32xf16>) outs(%1 : tensor<32x32xf16>) permutation = [1, 0]
  return %transposed, %0 : tensor<32x32xf16>, tensor<32x32xf16>
}

// -----

// Same last-dim offset hole as fuse_load_with_dyn_size; unique CHECKs are the
// unlikely hint and transferred multi_buffer mark (pad gate lives on dyn_size).
// CHECK-LABEL: func.func @fuse_load_with_multibuffer_annotation
// CHECK:       %[[NEW:.*]] = memref.alloc() : memref<128x32xbf16>
// CHECK:       scf.if
// CHECK:         linalg.fill ins(%{{.*}} : bf16) outs(%[[NEW]] : memref<128x32xbf16>)
// CHECK:       } {hivm.unlikely_condition}
// CHECK:       annotation.mark %[[NEW]] {hivm.multi_buffer = 2 : i32}
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<128x?xbf16, strided<[1, 4096], offset: ?>> to memref<128x?xbf16, strided<[32, 1], offset: ?>>
// CHECK:       %[[res:.*]] = bufferization.to_tensor
// CHECK-NOT:   transpose
// CHECK:       return %[[res]] : tensor<128x32xbf16>
func.func @fuse_load_with_multibuffer_annotation(%arg0: memref<?xbf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: i1) -> tensor<128x32xbf16> attributes {mix_mode = "aiv"} {
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %alloc = memref.alloc() : memref<32x128xbf16>
  annotation.mark %alloc {hivm.multi_buffer = 2 : i32} : memref<32x128xbf16>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [32, 128], strides: [4096, 1] : memref<?xbf16> to memref<32x128xbf16, strided<[4096, 1], offset: ?>>
  %subview = memref.subview %reinterpret_cast[0, 0] [%arg3, 128] [1, 1] : memref<32x128xbf16, strided<[4096, 1], offset: ?>> to memref<?x128xbf16, strided<[4096, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[%arg2, 0] [%arg3, 128] [1, 1] : memref<32x128xbf16> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
  %0 = arith.cmpi slt, %arg3, %c32 : index
  %1 = arith.ori %arg4, %0 : i1
  scf.if %1 {
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
  } {hivm.unlikely_condition}
  memref.copy %subview, %subview_0 : memref<?x128xbf16, strided<[4096, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
  %3 = tensor.empty() : tensor<128x32xbf16>
  %transposed = linalg.transpose ins(%2 : tensor<32x128xbf16>) outs(%3 : tensor<128x32xbf16>) permutation = [1, 0]
  return %transposed : tensor<128x32xbf16>
}

// -----

// Last-dim-only static offset 0: drop fill, no expand, no runtime offset gate.
// CHECK-LABEL: func.func @fuse_load_transfer_multi_buffer_mark
// CHECK:       %[[NEW_ALLOC:.*]] = memref.alloc() : memref<32x16xf16>
// CHECK:       annotation.mark %[[NEW_ALLOC]] {hivm.multi_buffer = 2 : i32} : memref<32x16xf16>
// CHECK-NOT:   arith.cmpi ne
// CHECK-NOT:   linalg.fill
// CHECK: memref.copy %{{.*}}, %[[NEW_ALLOC]] : memref<32x16xf16, strided<[1, 128], offset: ?>> to memref<32x16xf16>
// CHECK-NOT:   linalg.transpose
// CHECK:       return %{{.*}} : tensor<32x16xf16>
func.func @fuse_load_transfer_multi_buffer_mark(%arg0: memref<?xf16>, %arg1: index) -> tensor<32x16xf16> attributes {mix_mode = "aiv"} {
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<16x32xf16>
  annotation.mark %alloc {hivm.multi_buffer = 2 : i32} : memref<16x32xf16>
  linalg.fill ins(%cst : f16) outs(%alloc : memref<16x32xf16>)
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [16, 32], strides: [128, 1] : memref<?xf16> to memref<16x32xf16, strided<[128, 1], offset: ?>>
  memref.copy %reinterpret_cast, %alloc : memref<16x32xf16, strided<[128, 1], offset: ?>> to memref<16x32xf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x32xf16>
  %1 = tensor.empty() : tensor<32x16xf16>
  %transposed = linalg.transpose ins(%0 : tensor<16x32xf16>) outs(%1 : tensor<32x16xf16>) permutation = [1, 0]
  return %transposed : tensor<32x16xf16>
}

// -----

// CHECK-LABEL: func.func @fuse_load_transfer_mark_rank_reduced_subview
// CHECK:       %[[NEW_ALLOC:.*]] = memref.alloc() : memref<2x64x32xf16>
// CHECK:       annotation.mark %[[NEW_ALLOC]] {hivm.multi_buffer = 2 : i32} : memref<2x64x32xf16>
// CHECK-NOT:   linalg.fill
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<64x32xf16, strided<[1, 128], offset: ?>> to memref<64x32xf16, strided<[32, 1], offset: ?>>
// CHECK-NOT:   linalg.transpose
// CHECK:       return %{{.*}} : tensor<64x32xf16>
func.func @fuse_load_transfer_mark_rank_reduced_subview(%arg0: memref<?xf16>, %arg1: index, %arg2: index) -> tensor<64x32xf16> attributes {mix_mode = "aiv"} {
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<2x32x64xf16>
  annotation.mark %alloc {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf16>
  linalg.fill ins(%cst : f16) outs(%alloc : memref<2x32x64xf16>)
  %subview = memref.subview %alloc[%arg2, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [32, 64], strides: [128, 1] : memref<?xf16> to memref<32x64xf16, strided<[128, 1], offset: ?>>
  memref.copy %reinterpret_cast, %subview : memref<32x64xf16, strided<[128, 1], offset: ?>> to memref<32x64xf16, strided<[64, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[%arg2, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
  %0 = bufferization.to_tensor %subview_0 restrict writable : memref<32x64xf16, strided<[64, 1], offset: ?>>
  %1 = tensor.empty() : tensor<64x32xf16>
  %transposed = linalg.transpose ins(%0 : tensor<32x64xf16>) outs(%1 : tensor<64x32xf16>) permutation = [1, 0]
  return %transposed : tensor<64x32xf16>
}

// -----

// CHECK-LABEL: func.func @fuse_load_transfer_plain_mark
// CHECK:       %[[NEW_ALLOC:.*]] = memref.alloc() : memref<2x64x32xf16>
// CHECK:       annotation.mark %[[NEW_ALLOC]] {some_other_attr = 1 : i32} : memref<2x64x32xf16>
// CHECK-NOT:   linalg.fill
// CHECK:       memref.copy
// CHECK-NOT:   linalg.transpose
// CHECK:       return %{{.*}} : tensor<64x32xf16>
func.func @fuse_load_transfer_plain_mark(%arg0: memref<?xf16>, %arg1: index) -> tensor<64x32xf16> attributes {mix_mode = "aiv"} {
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<2x32x64xf16>
  annotation.mark %alloc {some_other_attr = 1 : i32} : memref<2x32x64xf16>
  linalg.fill ins(%cst : f16) outs(%alloc : memref<2x32x64xf16>)
  %subview = memref.subview %alloc[%arg1, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [32, 64], strides: [128, 1] : memref<?xf16> to memref<32x64xf16, strided<[128, 1], offset: ?>>
  memref.copy %reinterpret_cast, %subview : memref<32x64xf16, strided<[128, 1], offset: ?>> to memref<32x64xf16, strided<[64, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[%arg1, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
  %0 = bufferization.to_tensor %subview_0 restrict writable : memref<32x64xf16, strided<[64, 1], offset: ?>>
  %1 = tensor.empty() : tensor<64x32xf16>
  %transposed = linalg.transpose ins(%0 : tensor<32x64xf16>) outs(%1 : tensor<64x32xf16>) permutation = [1, 0]
  return %transposed : tensor<64x32xf16>
}

// -----

// GLA k/gk shape: fill the 16x32 dest, DMA a prefix, to_tensor the full
// alloc, then transpose to 32x16. The fill must follow the new dest; otherwise
// pad K lanes are leftover UB.
// CHECK-LABEL: func.func @fuse_load_keep_fill_on_transposed_dst
// CHECK:       %[[NEW:.*]] = memref.alloc() : memref<32x16xbf16>
// CHECK:       linalg.fill ins(%{{.*}} : bf16) outs(%[[NEW]] : memref<32x16xbf16>)
// CHECK-NOT:   arith.cmpi ne
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<?x?xbf16, strided<[1, 32], offset: ?>> to memref<?x?xbf16, strided<[16, 1]>>
// CHECK-NOT:   linalg.transpose
// CHECK:       return %{{.*}} : tensor<32x16xbf16>
func.func @fuse_load_keep_fill_on_transposed_dst(%arg0: memref<?xbf16>, %arg1: index, %arg2: index, %arg3: index) -> tensor<32x16xbf16> attributes {mix_mode = "aiv"} {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %alloc = memref.alloc() : memref<16x32xbf16>
  linalg.fill ins(%cst : bf16) outs(%alloc : memref<16x32xbf16>)
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [16, 32], strides: [32, 1] : memref<?xbf16> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
  %0 = arith.maxsi %arg2, %c0 : index
  %1 = arith.minsi %0, %c16 : index
  %2 = arith.maxsi %arg3, %c0 : index
  %3 = arith.minsi %2, %c32 : index
  %subview = memref.subview %reinterpret_cast[0, 0] [%1, %3] [1, 1] : memref<16x32xbf16, strided<[32, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[0, 0] [%1, %3] [1, 1] : memref<16x32xbf16> to memref<?x?xbf16, strided<[32, 1]>>
  memref.copy %subview, %subview_0 : memref<?x?xbf16, strided<[32, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1]>>
  %4 = bufferization.to_tensor %alloc restrict writable : memref<16x32xbf16>
  %5 = tensor.empty() : tensor<32x16xbf16>
  %transposed = linalg.transpose ins(%4 : tensor<16x32xbf16>) outs(%5 : tensor<32x16xbf16>) permutation = [1, 0]
  return %transposed : tensor<32x16xbf16>
}
// -----

// Same GLA prefix-load shape as fuse_load_keep_fill_on_transposed_dst, but the
// fill is behind ConvertToHIVMOp's then-only scf.if. The fill must still land
// on the transposed dest, and the guard must be kept (full tiles skip it).
// CHECK-LABEL: func.func @fuse_load_keep_fill_in_unlikely_if
// CHECK:       %[[NEW:.*]] = memref.alloc() : memref<32x16xbf16>
// CHECK:       scf.if
// CHECK:         linalg.fill ins(%{{.*}} : bf16) outs(%[[NEW]] : memref<32x16xbf16>)
// CHECK:       } {hivm.unlikely_condition}
// CHECK-NOT:   arith.cmpi ne
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<?x?xbf16, strided<[1, 32], offset: ?>> to memref<?x?xbf16, strided<[16, 1]>>
// CHECK-NOT:   linalg.transpose
// CHECK:       return %{{.*}} : tensor<32x16xbf16>
func.func @fuse_load_keep_fill_in_unlikely_if(%arg0: memref<?xbf16>, %arg1: index, %arg2: index, %arg3: index) -> tensor<32x16xbf16> attributes {mix_mode = "aiv"} {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %alloc = memref.alloc() : memref<16x32xbf16>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [16, 32], strides: [32, 1] : memref<?xbf16> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
  %0 = arith.maxsi %arg2, %c0 : index
  %1 = arith.minsi %0, %c16 : index
  %2 = arith.maxsi %arg3, %c0 : index
  %3 = arith.minsi %2, %c32 : index
  %4 = arith.cmpi slt, %3, %c32 : index
  scf.if %4 {
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<16x32xbf16>)
  } {hivm.unlikely_condition}
  %subview = memref.subview %reinterpret_cast[0, 0] [%1, %3] [1, 1] : memref<16x32xbf16, strided<[32, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[0, 0] [%1, %3] [1, 1] : memref<16x32xbf16> to memref<?x?xbf16, strided<[32, 1]>>
  memref.copy %subview, %subview_0 : memref<?x?xbf16, strided<[32, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1]>>
  %5 = bufferization.to_tensor %alloc restrict writable : memref<16x32xbf16>
  %6 = tensor.empty() : tensor<32x16xbf16>
  %transposed = linalg.transpose ins(%5 : tensor<16x32xbf16>) outs(%6 : tensor<32x16xbf16>) permutation = [1, 0]
  return %transposed : tensor<32x16xbf16>
}

// -----

// perm(0, 2, 1, 3) keeps the last axis in place and permutes two middle axes.
// NDDMA drives loop0 .. loop4, so the whole rank-4 permutation folds into the
// load: dst becomes a 4x8x16x32 alloc and src takes the permuted strides.
// CHECK-LABEL: func.func @fuse_load_rank4_middle_axes
// CHECK:       %[[NEW:.*]] = memref.alloc() : memref<4x8x16x32xf16>
// CHECK: memref.copy %{{.*}}, %[[NEW]] : memref<4x8x16x32xf16, strided<[4096, 32, 256, 1], offset: ?>> to memref<4x8x16x32xf16>
// CHECK-NOT:   linalg.transpose
// CHECK:       return %{{.*}} : tensor<4x8x16x32xf16>
func.func @fuse_load_rank4_middle_axes(%arg0: memref<?xf16>, %arg1: index) -> tensor<4x8x16x32xf16> attributes {mix_mode = "aiv"} {
  %alloc = memref.alloc() : memref<4x16x8x32xf16>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [4, 16, 8, 32], strides: [4096, 256, 32, 1] : memref<?xf16> to memref<4x16x8x32xf16, strided<[4096, 256, 32, 1], offset: ?>>
  memref.copy %reinterpret_cast, %alloc : memref<4x16x8x32xf16, strided<[4096, 256, 32, 1], offset: ?>> to memref<4x16x8x32xf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<4x16x8x32xf16>
  %1 = tensor.empty() : tensor<4x8x16x32xf16>
  %transposed = linalg.transpose ins(%0 : tensor<4x16x8x32xf16>) outs(%1 : tensor<4x8x16x32xf16>) permutation = [0, 2, 1, 3]
  return %transposed : tensor<4x8x16x32xf16>
}

// -----

// A rank-3 cycle perm(2, 0, 1) moves every axis; still one NDDMA descriptor.
// CHECK-LABEL: func.func @fuse_load_rank3_cycle
// CHECK:       %[[NEW:.*]] = memref.alloc() : memref<32x8x16xf16>
// CHECK: memref.copy %{{.*}}, %[[NEW]] : memref<32x8x16xf16, strided<[1, 512, 32], offset: ?>> to memref<32x8x16xf16>
// CHECK-NOT:   linalg.transpose
// CHECK:       return %{{.*}} : tensor<32x8x16xf16>
func.func @fuse_load_rank3_cycle(%arg0: memref<?xf16>, %arg1: index) -> tensor<32x8x16xf16> attributes {mix_mode = "aiv"} {
  %alloc = memref.alloc() : memref<8x16x32xf16>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [8, 16, 32], strides: [512, 32, 1] : memref<?xf16> to memref<8x16x32xf16, strided<[512, 32, 1], offset: ?>>
  memref.copy %reinterpret_cast, %alloc : memref<8x16x32xf16, strided<[512, 32, 1], offset: ?>> to memref<8x16x32xf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<8x16x32xf16>
  %1 = tensor.empty() : tensor<32x8x16xf16>
  %transposed = linalg.transpose ins(%0 : tensor<8x16x32xf16>) outs(%1 : tensor<32x8x16xf16>) permutation = [2, 0, 1]
  return %transposed : tensor<32x8x16xf16>
}

// -----

// A padded load fuses when the permutation keeps the last axis: the pad
// counts still address the dst innermost axis, which NDDMA pads on loop0.
// CHECK-LABEL: func.func @fuse_padded_load_last_axis_kept
// CHECK:       %[[NEW:.*]] = memref.alloc() : memref<8x16x32xf16>
// CHECK:       memref.copy
// CHECK-NOT:   linalg.transpose
// CHECK:       return %{{.*}} : tensor<8x16x32xf16>
func.func @fuse_padded_load_last_axis_kept(%arg0: memref<?xf16>, %arg1: index) -> tensor<8x16x32xf16> attributes {mix_mode = "aiv"} {
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<16x8x32xf16>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [16, 8, 32], strides: [256, 32, 1] : memref<?xf16> to memref<16x8x32xf16, strided<[256, 32, 1], offset: ?>>
  %subview = memref.subview %reinterpret_cast[0, 0, 0] [16, 8, 29] [1, 1, 1] : memref<16x8x32xf16, strided<[256, 32, 1], offset: ?>> to memref<16x8x29xf16, strided<[256, 32, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[0, 0, 3] [16, 8, 29] [1, 1, 1] : memref<16x8x32xf16> to memref<16x8x29xf16, strided<[256, 32, 1], offset: 3>>
  memref.copy %subview, %subview_0 : memref<16x8x29xf16, strided<[256, 32, 1], offset: ?>> to memref<16x8x29xf16, strided<[256, 32, 1], offset: 3>>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x8x32xf16>
  %1 = tensor.empty() : tensor<8x16x32xf16>
  %transposed = linalg.transpose ins(%0 : tensor<16x8x32xf16>) outs(%1 : tensor<8x16x32xf16>) permutation = [1, 0, 2]
  return %transposed : tensor<8x16x32xf16>
}

// -----

// The same padded load with a permutation that moves the last axis is left
// alone: the pad counts would no longer name the dst innermost axis.
// CHECK-LABEL: func.func @no_fuse_padded_load_last_axis_moved
// CHECK:       memref.copy
// CHECK:       linalg.transpose
func.func @no_fuse_padded_load_last_axis_moved(%arg0: memref<?xf16>, %arg1: index) -> tensor<16x32x8xf16> attributes {mix_mode = "aiv"} {
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<16x8x32xf16>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [16, 8, 32], strides: [256, 32, 1] : memref<?xf16> to memref<16x8x32xf16, strided<[256, 32, 1], offset: ?>>
  %subview = memref.subview %reinterpret_cast[0, 0, 0] [16, 8, 29] [1, 1, 1] : memref<16x8x32xf16, strided<[256, 32, 1], offset: ?>> to memref<16x8x29xf16, strided<[256, 32, 1], offset: ?>>
  %subview_0 = memref.subview %alloc[0, 0, 3] [16, 8, 29] [1, 1, 1] : memref<16x8x32xf16> to memref<16x8x29xf16, strided<[256, 32, 1], offset: 3>>
  memref.copy %subview, %subview_0 : memref<16x8x29xf16, strided<[256, 32, 1], offset: ?>> to memref<16x8x29xf16, strided<[256, 32, 1], offset: 3>>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x8x32xf16>
  %1 = tensor.empty() : tensor<16x32x8xf16>
  %transposed = linalg.transpose ins(%0 : tensor<16x8x32xf16>) outs(%1 : tensor<16x32x8xf16>) permutation = [0, 2, 1]
  return %transposed : tensor<16x32x8xf16>
}

// -----

// Rank 6 exceeds the five NDDMA loops, so the lowering peels the outer axes
// into loops around a five-loop descriptor. Collapsing the axes the
// permutation leaves contiguous is hivm-flatten-ops' job and is checked in
// fuse-transpose-then-flatten.mlir.
// CHECK-LABEL: func.func @fuse_rank6_transpose
// CHECK:       %[[NEW:.*]] = memref.alloc() : memref<2x2x2x2x4x2xf16>
// CHECK: memref.copy %{{.*}}, %[[NEW]] : memref<2x2x2x2x4x2xf16, strided<[64, 32, 16, 8, 1, 4], offset: ?>> to memref<2x2x2x2x4x2xf16>
// CHECK-NOT:   linalg.transpose
func.func @fuse_rank6_transpose(%arg0: memref<?xf16>, %arg1: index) -> tensor<2x2x2x2x4x2xf16> attributes {mix_mode = "aiv"} {
  %alloc = memref.alloc() : memref<2x2x2x2x2x4xf16>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [2, 2, 2, 2, 2, 4], strides: [64, 32, 16, 8, 4, 1] : memref<?xf16> to memref<2x2x2x2x2x4xf16, strided<[64, 32, 16, 8, 4, 1], offset: ?>>
  memref.copy %reinterpret_cast, %alloc : memref<2x2x2x2x2x4xf16, strided<[64, 32, 16, 8, 4, 1], offset: ?>> to memref<2x2x2x2x2x4xf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<2x2x2x2x2x4xf16>
  %1 = tensor.empty() : tensor<2x2x2x2x4x2xf16>
  %transposed = linalg.transpose ins(%0 : tensor<2x2x2x2x2x4xf16>) outs(%1 : tensor<2x2x2x2x4x2xf16>) permutation = [0, 1, 2, 3, 5, 4]
  return %transposed : tensor<2x2x2x2x4x2xf16>
}

// -----

// perm(4, 3, 2, 1, 0) reverses every axis, so no adjacent pair stays
// contiguous on both operands and the load keeps all five. It still needs one
// descriptor, which has exactly five loops.
// CHECK-LABEL: func.func @fuse_rank5_reversal_no_collapse
// CHECK-NOT:   memref.collapse_shape
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<6x5x4x3x2xf16, strided<[1, 6, 30, 120, 360], offset: ?>> to memref<6x5x4x3x2xf16>
// CHECK-NOT:   linalg.transpose
// CHECK:       return %{{.*}} : tensor<6x5x4x3x2xf16>
func.func @fuse_rank5_reversal_no_collapse(%arg0: memref<?xf16>, %arg1: index) -> tensor<6x5x4x3x2xf16> attributes {mix_mode = "aiv"} {
  %alloc = memref.alloc() : memref<2x3x4x5x6xf16>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [2, 3, 4, 5, 6], strides: [360, 120, 30, 6, 1] : memref<?xf16> to memref<2x3x4x5x6xf16, strided<[360, 120, 30, 6, 1], offset: ?>>
  memref.copy %reinterpret_cast, %alloc : memref<2x3x4x5x6xf16, strided<[360, 120, 30, 6, 1], offset: ?>> to memref<2x3x4x5x6xf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<2x3x4x5x6xf16>
  %1 = tensor.empty() : tensor<6x5x4x3x2xf16>
  %transposed = linalg.transpose ins(%0 : tensor<2x3x4x5x6xf16>) outs(%1 : tensor<6x5x4x3x2xf16>) permutation = [4, 3, 2, 1, 0]
  return %transposed : tensor<6x5x4x3x2xf16>
}

// -----

// The Triton frontend emits the load as a memref.copy into a fresh contiguous
// alloc and the permute as a separate linalg.transpose; convert-to-hivm-op
// only turns the copy into a hivm.hir.load much later, so this is the form the
// pass meets on real input. Fusing rewrites the copy to read the permuted view
// and drops the transpose.
// CHECK-LABEL: func.func @fuse_memref_copy_rank3
// CHECK:       %[[NEW:.*]] = memref.alloc() : memref<4x16x8xf32>
// CHECK:       memref.copy %{{.*}}, %[[NEW]] : memref<4x16x8xf32, strided<[128, 1, 16]>> to memref<4x16x8xf32>
// CHECK:       bufferization.to_tensor %[[NEW]]
// CHECK-NOT:   linalg.transpose
func.func @fuse_memref_copy_rank3(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {mix_mode = "aiv"} {
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4, 8, 16], strides: [128, 16, 1] : memref<?xf32> to memref<4x8x16xf32, strided<[128, 16, 1]>>
  %alloc = memref.alloc() : memref<4x8x16xf32>
  memref.copy %reinterpret_cast, %alloc : memref<4x8x16xf32, strided<[128, 16, 1]>> to memref<4x8x16xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<4x8x16xf32>
  %1 = tensor.empty() : tensor<4x16x8xf32>
  %transposed = linalg.transpose ins(%0 : tensor<4x8x16xf32>) outs(%1 : tensor<4x16x8xf32>) permutation = [0, 2, 1]
  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4, 16, 8], strides: [128, 8, 1] : memref<?xf32> to memref<4x16x8xf32, strided<[128, 8, 1]>>
  bufferization.materialize_in_destination %transposed in writable %reinterpret_cast_0 : (tensor<4x16x8xf32>, memref<4x16x8xf32, strided<[128, 8, 1]>>) -> ()
  return
}

// -----

// A permutation that moves an inner axis pair but not the innermost one, at a
// rank only NDDMA can drive in one descriptor.
// CHECK-LABEL: func.func @fuse_memref_copy_rank4
// CHECK:       %[[NEW:.*]] = memref.alloc() : memref<2x8x4x16xf32>
// CHECK:       memref.copy %{{.*}}, %[[NEW]] : memref<2x8x4x16xf32, strided<[512, 16, 128, 1]>> to memref<2x8x4x16xf32>
// CHECK:       bufferization.to_tensor %[[NEW]]
// CHECK-NOT:   linalg.transpose
func.func @fuse_memref_copy_rank4(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {mix_mode = "aiv"} {
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2, 4, 8, 16], strides: [512, 128, 16, 1] : memref<?xf32> to memref<2x4x8x16xf32, strided<[512, 128, 16, 1]>>
  %alloc = memref.alloc() : memref<2x4x8x16xf32>
  memref.copy %reinterpret_cast, %alloc : memref<2x4x8x16xf32, strided<[512, 128, 16, 1]>> to memref<2x4x8x16xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<2x4x8x16xf32>
  %1 = tensor.empty() : tensor<2x8x4x16xf32>
  %transposed = linalg.transpose ins(%0 : tensor<2x4x8x16xf32>) outs(%1 : tensor<2x8x4x16xf32>) permutation = [0, 2, 1, 3]
  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [2, 8, 4, 16], strides: [512, 64, 16, 1] : memref<?xf32> to memref<2x8x4x16xf32, strided<[512, 64, 16, 1]>>
  bufferization.materialize_in_destination %transposed in writable %reinterpret_cast_0 : (tensor<2x8x4x16xf32>, memref<2x8x4x16xf32, strided<[512, 64, 16, 1]>>) -> ()
  return
}

// -----

// A copy between two local buffers is not a DMA, so permuting it would buy
// nothing; the transpose stays.
// CHECK-LABEL: func.func @no_fuse_copy_between_locals
// CHECK:       linalg.transpose
func.func @no_fuse_copy_between_locals() -> tensor<4x16x8xf32> attributes {mix_mode = "aiv"} {
  %cst = arith.constant 0.000000e+00 : f32
  %src = memref.alloc() : memref<4x8x16xf32>
  linalg.fill ins(%cst : f32) outs(%src : memref<4x8x16xf32>)
  %alloc = memref.alloc() : memref<4x8x16xf32>
  memref.copy %src, %alloc : memref<4x8x16xf32> to memref<4x8x16xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<4x8x16xf32>
  %1 = tensor.empty() : tensor<4x16x8xf32>
  %transposed = linalg.transpose ins(%0 : tensor<4x8x16xf32>) outs(%1 : tensor<4x16x8xf32>) permutation = [0, 2, 1]
  return %transposed : tensor<4x16x8xf32>
}

// -----

// memref.copy names no address space, so a GM source alone does not make the
// destination UB. In a kernel with a cube half the same shape is a GM->L1
// matmul operand, which NDDMA cannot write, so the transpose has to stay.
// CHECK-LABEL: func.func @no_fuse_gm_copy_in_mix_kernel
// CHECK:       linalg.transpose
func.func @no_fuse_gm_copy_in_mix_kernel(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf32>) attributes {mix_mode = "mix"} {
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1]>>
  %alloc = memref.alloc() : memref<32x32xf16>
  memref.copy %reinterpret_cast, %alloc : memref<32x32xf16, strided<[32, 1]>> to memref<32x32xf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf16>
  %1 = tensor.empty() : tensor<32x32xf16>
  %transposed = linalg.transpose ins(%0 : tensor<32x32xf16>) outs(%1 : tensor<32x32xf16>) permutation = [1, 0]
  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1]>>
  bufferization.materialize_in_destination %transposed in writable %reinterpret_cast_0 : (tensor<32x32xf16>, memref<32x32xf16, strided<[32, 1]>>) -> ()
  return
}

// -----

// convert-to-hivm-op derives left_padding_num from the last-dim offset of the
// destination subview, so this copy becomes a padded load. NDDMA applies the
// count on loop0, which only keeps its meaning while the innermost axis stays
// in place, and permutation [0, 2, 1] moves it - so the transpose has to stay.
// CHECK-LABEL: func.func @no_fuse_left_padded_copy
// CHECK:       linalg.transpose
func.func @no_fuse_left_padded_copy(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {mix_mode = "aiv"} {
  %cst = arith.constant 0.000000e+00 : f32
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4, 8, 16], strides: [128, 16, 1] : memref<?xf32> to memref<4x8x16xf32, strided<[128, 16, 1]>>
  %alloc = memref.alloc() : memref<4x8x16xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<4x8x16xf32>)
  %subview = memref.subview %reinterpret_cast[0, 0, 0] [4, 8, 13] [1, 1, 1] : memref<4x8x16xf32, strided<[128, 16, 1]>> to memref<4x8x13xf32, strided<[128, 16, 1]>>
  %subview_0 = memref.subview %alloc[0, 0, 3] [4, 8, 13] [1, 1, 1] : memref<4x8x16xf32> to memref<4x8x13xf32, strided<[128, 16, 1], offset: 3>>
  memref.copy %subview, %subview_0 : memref<4x8x13xf32, strided<[128, 16, 1]>> to memref<4x8x13xf32, strided<[128, 16, 1], offset: 3>>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<4x8x16xf32>
  %1 = tensor.empty() : tensor<4x16x8xf32>
  %transposed = linalg.transpose ins(%0 : tensor<4x8x16xf32>) outs(%1 : tensor<4x16x8xf32>) permutation = [0, 2, 1]
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4, 16, 8], strides: [128, 8, 1] : memref<?xf32> to memref<4x16x8xf32, strided<[128, 8, 1]>>
  bufferization.materialize_in_destination %transposed in writable %reinterpret_cast_1 : (tensor<4x16x8xf32>, memref<4x16x8xf32, strided<[128, 8, 1]>>) -> ()
  return
}

// -----

// The same shape with the window at offset 0 is the masked-load form: the pad
// is a fill over the whole buffer, carries no count, and permutes with the
// buffer, so it fuses.
// CHECK-LABEL: func.func @fuse_masked_copy_offset_zero
// CHECK:       %[[NEW:.*]] = memref.alloc() : memref<4x16x8xf32>
// CHECK:       linalg.fill ins(%{{.*}} : f32) outs(%[[NEW]] : memref<4x16x8xf32>)
// CHECK:       memref.copy
// CHECK-NOT:   linalg.transpose
func.func @fuse_masked_copy_offset_zero(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {mix_mode = "aiv"} {
  %cst = arith.constant 0.000000e+00 : f32
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4, 8, 16], strides: [128, 16, 1] : memref<?xf32> to memref<4x8x16xf32, strided<[128, 16, 1]>>
  %alloc = memref.alloc() : memref<4x8x16xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<4x8x16xf32>)
  %subview = memref.subview %reinterpret_cast[0, 0, 0] [4, 8, 10] [1, 1, 1] : memref<4x8x16xf32, strided<[128, 16, 1]>> to memref<4x8x10xf32, strided<[128, 16, 1]>>
  %subview_0 = memref.subview %alloc[0, 0, 0] [4, 8, 10] [1, 1, 1] : memref<4x8x16xf32> to memref<4x8x10xf32, strided<[128, 16, 1]>>
  memref.copy %subview, %subview_0 : memref<4x8x10xf32, strided<[128, 16, 1]>> to memref<4x8x10xf32, strided<[128, 16, 1]>>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<4x8x16xf32>
  %1 = tensor.empty() : tensor<4x16x8xf32>
  %transposed = linalg.transpose ins(%0 : tensor<4x8x16xf32>) outs(%1 : tensor<4x16x8xf32>) permutation = [0, 2, 1]
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4, 16, 8], strides: [128, 8, 1] : memref<?xf32> to memref<4x16x8xf32, strided<[128, 8, 1]>>
  bufferization.materialize_in_destination %transposed in writable %reinterpret_cast_1 : (tensor<4x16x8xf32>, memref<4x16x8xf32, strided<[128, 8, 1]>>) -> ()
  return
}

// -----

// A collapse root with a static shape can still carry a dynamic offset or
// dynamic strides. Those read back from the type as the kDynamic sentinel,
// which is not something reinterpret_cast can be handed as a literal, so the
// real values come off the buffer with extract_strided_metadata.
// CHECK-LABEL: func.func @fuse_memref_copy_collapse_root_dynamic_offset
// CHECK:       %[[C:.*]] = memref.collapse_shape
// CHECK:       %{{.*}}, %[[OFF:.*]], %{{.*}}:3, %{{.*}}:3 = memref.extract_strided_metadata %[[C]]
// CHECK:       %[[P:.*]] = memref.reinterpret_cast %[[C]] to offset: [%[[OFF]]], sizes: [8, 16, 8], strides: [128, 1, 16]
// CHECK:       %[[NEW:.*]] = memref.alloc() : memref<8x16x8xf32>
// CHECK:       memref.copy %[[P]], %[[NEW]]
// CHECK-NOT:   linalg.transpose
func.func @fuse_memref_copy_collapse_root_dynamic_offset(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %off: index) attributes {mix_mode = "aiv"} {
  %cast = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [2, 4, 8, 16], strides: [512, 128, 16, 1] : memref<?xf32> to memref<2x4x8x16xf32, strided<[512, 128, 16, 1], offset: ?>>
  %collapsed = memref.collapse_shape %cast [[0, 1], [2], [3]] : memref<2x4x8x16xf32, strided<[512, 128, 16, 1], offset: ?>> into memref<8x8x16xf32, strided<[128, 16, 1], offset: ?>>
  %alloc = memref.alloc() : memref<8x8x16xf32>
  memref.copy %collapsed, %alloc : memref<8x8x16xf32, strided<[128, 16, 1], offset: ?>> to memref<8x8x16xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<8x8x16xf32>
  %1 = tensor.empty() : tensor<8x16x8xf32>
  %transposed = linalg.transpose ins(%0 : tensor<8x8x16xf32>) outs(%1 : tensor<8x16x8xf32>) permutation = [0, 2, 1]
  %out = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8, 16, 8], strides: [128, 8, 1] : memref<?xf32> to memref<8x16x8xf32, strided<[128, 8, 1]>>
  bufferization.materialize_in_destination %transposed in writable %out : (tensor<8x16x8xf32>, memref<8x16x8xf32, strided<[128, 8, 1]>>) -> ()
  return
}
