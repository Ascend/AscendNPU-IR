// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend310B4 -hivm-mark-stride-align -split-input-file %s | FileCheck %s

// This file covers the UB-side rank-reduced subview fix in MarkStrideAlign for
// hivm.hir.store. The mark should still be attached to the current IR operand
// used by store.

// -----

// Positive: UB source is a rank-reduced subview from 8x1 -> 8.
// The current IR operand is the subview result, so the mark should be attached
// to the rank-reduced subview result rather than the root alloc.
//
// CHECK-LABEL: func.func @store_rank_reduced_subview_2d_to_1d
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ALLOC:.*]][0, 0] [8, 1] [1, 1] : memref<8x1xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[SUBVIEW]] {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @store_rank_reduced_subview_2d_to_1d(%arg0: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
    %alloc = memref.alloc() : memref<8x1xf32, #hivm.address_space<ub>>
    %subview = memref.subview %alloc[0, 0] [8, 1] [1, 1] : memref<8x1xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
    %gm = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [117] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[117]>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview : memref<8xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%gm : memref<8xf32, strided<[117]>, #hivm.address_space<gm>>)
    return
  }
}

// -----

// Positive: UB source is a rank-reduced subview from 4x8x1 -> 4x8.
// The dropped trailing static-1 dim should not prevent the current 2D subview
// result from being marked for stride alignment.
//
// CHECK-LABEL: func.func @store_rank_reduced_subview_3d_to_2d
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ALLOC:.*]][0, 0, 0] [4, 8, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[SUBVIEW]] {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @store_rank_reduced_subview_3d_to_2d(%arg0: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
    %alloc = memref.alloc() : memref<4x8x1xf32, #hivm.address_space<ub>>
    %subview = memref.subview %alloc[0, 0, 0] [4, 8, 1] [1, 1, 1] : memref<4x8x1xf32, #hivm.address_space<ub>> to memref<4x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %gm = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4, 8], strides: [1394, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x8xf32, strided<[1394, 1]>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview : memref<4x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%gm : memref<4x8xf32, strided<[1394, 1]>, #hivm.address_space<gm>>)
    return
  }
}
