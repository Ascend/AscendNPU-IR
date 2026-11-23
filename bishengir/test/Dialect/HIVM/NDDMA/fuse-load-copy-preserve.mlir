// RUN: bishengir-opt %s --hivm-inline-load-copy -split-input-file | FileCheck %s
// The broadcast path must preserve the original load and retarget its output.
// CHECK-LABEL: func.func @keep_broadcast(
// CHECK-SAME: %[[SRC:[a-zA-Z0-9_]+]]: memref<1x32xf32
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: memref<4x32xf32
// CHECK: hivm.hir.load ins(%[[SRC]]
// CHECK-SAME: outs(%[[OUT]]
// CHECK-SAME: hivm.tcore_type = #hivm.tcore_type<VECTOR>
// CHECK-SAME: eviction_policy = <EvictFirst>
// CHECK-SAME: broadcast_axes = [0]
// CHECK-SAME: core_type = <VECTOR>
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: hivm.hir.copy
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @keep_broadcast(%src: memref<1x32xf32, #hivm.address_space<gm>>, %out: memref<4x32xf32, #hivm.address_space<ub>>) {
  %tile = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%tile : memref<4x32xf32, #hivm.address_space<ub>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} eviction_policy = <EvictFirst> broadcast_axes = [0] core_type = <VECTOR>
  // Explicit no-padding and identity reassociation are not extra copy behavior.
  hivm.hir.copy ins(%tile : memref<4x32xf32, #hivm.address_space<ub>>) outs(%out : memref<4x32xf32, #hivm.address_space<ub>>) pad_mode = <PadNull> collapse_reassociation = [[0], [1]]
  return
 }
}

// -----

// Padding is part of the original load, not a reason to reject a plain copy.
// CHECK-LABEL: func.func @keep_span_padding(
// CHECK-SAME: %[[SRC:.*]]: memref<1x28xf32
// CHECK-SAME: %[[OUT:.*]]: memref<4x32xf32
// CHECK: %[[LP:.*]] = arith.constant 1 : index
// CHECK: %[[RP:.*]] = arith.constant 3 : index
// CHECK: %[[PAD:.*]] = arith.constant
// CHECK: hivm.hir.load ins(%[[SRC]]
// CHECK-SAME: outs(%[[OUT]]
// CHECK-SAME: pad_mode = <PadValue> pad_value = %[[PAD]] : f32
// CHECK-SAME: left_padding_num = %[[LP]] : index
// CHECK-SAME: right_padding_num = %[[RP]] : index
// CHECK-SAME: eviction_policy = <EvictFirst>
// CHECK-SAME: broadcast_axes = [0]
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: hivm.hir.copy
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @keep_span_padding(%src: memref<1x28xf32, #hivm.address_space<gm>>, %out: memref<4x32xf32, #hivm.address_space<ub>>) {
    %lp = arith.constant 1 : index
    %rp = arith.constant 3 : index
    %pad = arith.constant 7.0 : f32
    %tile = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<1x28xf32, #hivm.address_space<gm>>) outs(%tile : memref<4x32xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %pad : f32 left_padding_num = %lp : index right_padding_num = %rp : index eviction_policy = <EvictFirst> broadcast_axes = [0]
    hivm.hir.copy ins(%tile : memref<4x32xf32, #hivm.address_space<ub>>) outs(%out : memref<4x32xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Keep the original rank-3 offset/stride input: a shape match alone does not
// prove destination-layout equivalence. This deliberately retains the chain.
// CHECK-LABEL: func.func @fuse_strided_offset(
// CHECK: hivm.hir.load
// CHECK-SAME: outs(%[[TILE:.*]] : memref<2x4x32xf32
// CHECK-SAME: broadcast_axes = [0]
// CHECK: %[[VIEW:.*]] = memref.subview
// CHECK-SAME: to memref<2x4x32xf32, strided<[512, 64, 1], offset: 577>
// CHECK: hivm.hir.copy ins(%[[TILE]]
// CHECK-SAME: outs(%[[VIEW]]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @fuse_strided_offset(%src: memref<1x4x32xf32, #hivm.address_space<gm>>, %out: memref<3x8x64xf32, #hivm.address_space<ub>>) {
    %tile = memref.alloc() : memref<2x4x32xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<1x4x32xf32, #hivm.address_space<gm>>) outs(%tile : memref<2x4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
    %view = memref.subview %out[1, 1, 1] [2, 4, 32] [1, 1, 1] : memref<3x8x64xf32, #hivm.address_space<ub>> to memref<2x4x32xf32, strided<[512, 64, 1], offset: 577>, #hivm.address_space<ub>>
    hivm.hir.copy ins(%tile : memref<2x4x32xf32, #hivm.address_space<ub>>) outs(%view : memref<2x4x32xf32, strided<[512, 64, 1], offset: 577>, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Padding and reassociation belong to the copy, not to the original load.
// Neither may be silently discarded or transferred to a cloned load.
// CHECK-LABEL: func.func @keep_semantic_copies(
// CHECK: hivm.hir.load
// CHECK-SAME: outs(%[[PAD:.*]] : memref<4x32xf32
// CHECK-SAME: broadcast_axes = [0]
// CHECK: hivm.hir.copy ins(%[[PAD]]
// CHECK-SAME: pad_mode = <PadValue> pad_value =
// CHECK: hivm.hir.load
// CHECK-SAME: outs(%[[RESHAPE:.*]] : memref<4x32xf32
// CHECK-SAME: broadcast_axes = [0]
// CHECK: hivm.hir.copy ins(%[[RESHAPE]]
// CHECK-SAME: collapse_reassociation = {{\[\[0, 1\]\]}}
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @keep_semantic_copies(%src: memref<1x32xf32, #hivm.address_space<gm>>, %out0: memref<4x32xf32, #hivm.address_space<ub>>, %out1: memref<4x32xf32, #hivm.address_space<ub>>) {
    %zero = arith.constant 0.0 : f32
    %pad = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%pad : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
    hivm.hir.copy ins(%pad : memref<4x32xf32, #hivm.address_space<ub>>) outs(%out0 : memref<4x32xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %zero : f32
    %reshape = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%reshape : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
    hivm.hir.copy ins(%reshape : memref<4x32xf32, #hivm.address_space<ub>>) outs(%out1 : memref<4x32xf32, #hivm.address_space<ub>>) collapse_reassociation = [[0, 1]]
    return
  }
}

// -----

// A compatible '?' does not prove equal extents. A static overlapping layout
// is also not an admissible destination for the broadcast template.
// CHECK-LABEL: func.func @keep_unproven_destinations(
// CHECK: hivm.hir.load
// CHECK-SAME: outs(%[[DYNAMIC:.*]] : memref<4x32xf32
// CHECK-SAME: broadcast_axes = [0]
// CHECK: hivm.hir.copy ins(%[[DYNAMIC]]
// CHECK-SAME: outs({{.*}} : memref<?x32xf32
// CHECK: hivm.hir.load
// CHECK-SAME: outs(%[[OVERLAP:.*]] : memref<4x32xf32
// CHECK-SAME: broadcast_axes = [0]
// CHECK: hivm.hir.copy ins(%[[OVERLAP]]
// CHECK-SAME: outs({{.*}} : memref<4x32xf32, strided<[16, 1]>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @keep_unproven_destinations(%src: memref<1x32xf32, #hivm.address_space<gm>>, %out0: memref<?x32xf32, #hivm.address_space<ub>>, %out1: memref<4x32xf32, strided<[16, 1]>, #hivm.address_space<ub>>) {
    %dynamic = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dynamic : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
    hivm.hir.copy ins(%dynamic : memref<4x32xf32, #hivm.address_space<ub>>) outs(%out0 : memref<?x32xf32, #hivm.address_space<ub>>)
    %overlap = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%overlap : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
    hivm.hir.copy ins(%overlap : memref<4x32xf32, #hivm.address_space<ub>>) outs(%out1 : memref<4x32xf32, strided<[16, 1]>, #hivm.address_space<ub>>)
    return
  }
}

// -----

// The existing middle-buffer read check must still run for broadcasts.
// CHECK-LABEL: func.func @keep_live_read(
// CHECK: hivm.hir.load
// CHECK-SAME: outs(%[[TILE:.*]] : memref<4x32xf32
// CHECK-SAME: broadcast_axes = [0]
// CHECK: %[[VALUE:.*]] = memref.load %[[TILE]]
// CHECK: hivm.hir.copy ins(%[[TILE]]
// CHECK: return %[[VALUE]]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
  func.func @keep_live_read(%src: memref<1x32xf32, #hivm.address_space<gm>>, %out: memref<4x32xf32, #hivm.address_space<ub>>) -> f32 {
    %c0 = arith.constant 0 : index
    %tile = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%tile : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
    %value = memref.load %tile[%c0, %c0] : memref<4x32xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%tile : memref<4x32xf32, #hivm.address_space<ub>>) outs(%out : memref<4x32xf32, #hivm.address_space<ub>>)
    return %value : f32
  }
}
