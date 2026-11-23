// RUN: bishengir-opt %s -hivm-flatten-ops -convert-hivm-to-std | FileCheck %s

!gm = memref<7680x8xi16, #hivm.address_space<gm>>
!slice = memref<7680x4xi16, strided<[8, 1], offset: 2>, #hivm.address_space<gm>>
!tile = memref<7680x4xi16, #hivm.address_space<ub>>

module attributes {hacc.target = #hacc.target<"Ascend960">} {
  // X[:, 2:6] has 8-byte rows, a 16-byte GM pitch, and no padding.
  // CHECK-LABEL: func.func @compact_slice(
  // CHECK-SAME: %[[GM:[^:]+]]: {{.*}}, %[[UB:[^:]+]]:
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[NO_PAD:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[VIEW:.*]] = memref.subview %[[GM]][0, 2] [7680, 4] [1, 1]
  // CHECK-DAG: %[[SRC:.*]] = memref.cast %[[VIEW]] : memref<7680x4xi16, strided<[8, 1], offset: 2>, #hivm.address_space<gm>>
  // CHECK-DAG: %[[DST:.*]] = memref.cast %[[UB]] : memref<7680x4xi16, #hivm.address_space<ub>>
  // CHECK: call @load_gm_to_ubuf_2d_int16_t(%[[SRC]], %[[DST]], %[[NO_PAD]], %{{[^,]+}}, %[[ZERO]], %{{[^,)]+}})
  func.func @compact_slice(%src: !gm, %dst: !tile) {
    %slice = memref.subview %src[0, 2] [7680, 4] [1, 1] : !gm to !slice
    hivm.hir.load ins(%slice : !slice) outs(%dst : !tile)
    return
  }

  // When the same output comes from contiguous GM rows, flatten it to 1D.
  // CHECK-LABEL: func.func @contiguous_rows(
  // CHECK: call @load_gm_to_ubuf_1d_int16_t(
  func.func @contiguous_rows(%src: memref<7680x4xi16, #hivm.address_space<gm>>, %dst: !tile) {
    hivm.hir.load ins(%src : memref<7680x4xi16, #hivm.address_space<gm>>) outs(%dst : !tile)
    return
  }
}
