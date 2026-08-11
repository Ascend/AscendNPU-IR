// RUN: bishengir-opt -hivm-recognize-discontinuous-store -split-input-file %s | FileCheck %s

// CHECK-LABEL: store_brc_f32_collapse_shape_dyn_offsets
// f32 store with dynamic offsets on both src and dst.
// src: subview of collapse_shape (forces fallback path; fast path skips rank-changing ops).
// dst: subview of reinterpret_cast with dynamic offset and stride=96.
// channelNum = 8 (32B / 4B). The static alloc upper bound comes from the
// fallback path (traceToAllocMaxSize returns total element count = 64), so
// needNarrowSubview = true and a narrowing subview is inserted to fit the
// dynamic src size.
//
// CHECK: %[[expand:.*]] = memref.expand_shape
// CHECK: %[[alloc:.*]] = memref.alloc() : memref<64x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
// CHECK: %[[subview2:.*]] = memref.subview %[[alloc]][0, 0] [%arg1, 8] [1, 1]
// CHECK: hivm.hir.vbrc ins(%[[expand]] : {{.*}}) outs(%[[subview2]] : {{.*}}) broadcast_dims = [1]
// CHECK: %[[subview3:.*]] = memref.subview %[[subview2]][0, 0] [%arg1, 1] [1, 1]
// CHECK: %[[collapse:.*]] = memref.collapse_shape %[[subview3]] {{\[\[}}0, 1{{\]\]}}
// CHECK: hivm.hir.store ins(%[[collapse]] : memref<?xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?xf32, strided<[96], offset: ?>, #hivm.address_space<gm>>)
func.func @store_brc_f32_collapse_shape_dyn_offsets(%arg0: index, %arg1: index, %arg2: memref<?xf32, #hivm.address_space<gm>>) {
  %static_alloc_2d = memref.alloc() : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %static_alloc_2d [[0, 1]] : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
  %subview_src = memref.subview %collapse_shape[%arg0] [%arg1] [1] : memref<64xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %reinterpret_cast_dst = memref.reinterpret_cast %arg2 to offset: [%arg0], sizes: [64], strides: [96] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[96], offset: ?>, #hivm.address_space<gm>>
  %subview_dst = memref.subview %reinterpret_cast_dst[0] [%arg1] [1] : memref<64xf32, strided<[96], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[96], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview_src : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_dst : memref<?xf32, strided<[96], offset: ?>, #hivm.address_space<gm>>)
  return
}

// -----

// CHECK-LABEL: func.func @store_brc_f32_dynamic_dst_offset
// CHECK: %[[expand:.*]] = memref.expand_shape
// CHECK: %[[alloc:.*]] = memref.alloc() : memref<64x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
// CHECK: hivm.hir.vbrc ins(%[[expand]] : {{.*}}) outs(%[[alloc]] : {{.*}}) broadcast_dims = [1]
// CHECK: %[[subview:.*]] = memref.subview %[[alloc]][0, 0] [64, 1] [1, 1]
// CHECK: %[[collapse:.*]] = memref.collapse_shape %[[subview]] {{\[\[}}0, 1{{\]\]}}
// CHECK: hivm.hir.store ins(%[[collapse]] : {{.*}}) outs(%{{.*}} : {{.*}})
func.func @store_brc_f32_dynamic_dst_offset(%arg0: index, %arg1: memref<?xf32, #hivm.address_space<gm>>) {
  %static_alloc = memref.alloc() : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>
  %reinterpret_cast_dst = memref.reinterpret_cast %arg1 to offset: [%arg0], sizes: [64], strides: [96] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[96], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%static_alloc : memref<64xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_dst : memref<64xf32, strided<[96], offset: ?>, #hivm.address_space<gm>>)
  return
}

// -----

// Negative: src last stride != 1. Pattern should not fire.
// CHECK-LABEL: store_brc_neg_src_last_stride_not_one
// CHECK-NOT: hivm.hir.vbrc
// CHECK: hivm.hir.store
func.func @store_brc_neg_src_last_stride_not_one(%arg0: memref<64xf32, strided<[96]>, #hivm.address_space<gm>>) {
  %static_alloc = memref.alloc() : memref<64xf32, strided<[2]>, #hivm.address_space<ub>>
  hivm.hir.store ins(%static_alloc : memref<64xf32, strided<[2]>, #hivm.address_space<ub>>) outs(%arg0 : memref<64xf32, strided<[96]>, #hivm.address_space<gm>>)
  return
}