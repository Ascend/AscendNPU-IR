// RUN: bishengir-opt %s -hivm-enable-stride-align | FileCheck %s

// Two collapses of one alloc are views of the same buffer. Stride alignment
// pads the alloc so neither collapse can keep the strided layout, and type
// propagation fails. Both views must be retargeted to one compact alloc.
// A private alloc per collapse leaves the store reading a copy taken before
// the load.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @shared_collapse_after_stride_align
  // CHECK: %[[ALIGNED:.*]] = memref.alloc() : memref<2x2x16x1xbf16, #hivm.address_space<ub>>
  // CHECK: %[[VIEW:.*]] = memref.subview %[[ALIGNED]][0, 0, 0, 0] [2, 2, 8, 1] [1, 1, 1, 1]
  // Both collapses bind the same %[[SHARED]]. A per-user alloc fails this.
  // CHECK: %[[SHARED:.*]] = memref.alloc() : memref<2x2x8xbf16, #hivm.address_space<ub>>
  // CHECK: hivm.hir.copy ins(%[[VIEW]] : memref<2x2x8xbf16, strided<[32, 16, 1]>, #hivm.address_space<ub>>) outs(%[[SHARED]] : memref<2x2x8xbf16, #hivm.address_space<ub>>)
  // CHECK-DAG: %[[FLAT:.*]] = memref.collapse_shape %[[SHARED]] {{\[}}[0, 1, 2]]
  // CHECK-DAG: %[[TILE:.*]] = memref.collapse_shape %[[SHARED]] {{\[}}[0], [1, 2]]
  // CHECK: hivm.hir.load {{.*}}outs(%[[TILE]] : memref<2x16xbf16, #hivm.address_space<ub>>)
  // CHECK: hivm.hir.copy ins(%[[SHARED]] : memref<2x2x8xbf16, #hivm.address_space<ub>>) outs(%[[VIEW]]
  // CHECK: hivm.hir.store ins(%[[FLAT]] : memref<32xbf16, #hivm.address_space<ub>>)
  func.func @shared_collapse_after_stride_align(%src: memref<2x16xbf16, #hivm.address_space<gm>>, %dst: memref<32xbf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %alloc = memref.alloc() : memref<2x2x8xbf16, #hivm.address_space<ub>>
    annotation.mark %alloc {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x2x8xbf16, #hivm.address_space<ub>>
    %flat = memref.collapse_shape %alloc [[0, 1, 2]] : memref<2x2x8xbf16, #hivm.address_space<ub>> into memref<32xbf16, #hivm.address_space<ub>>
    %tile = memref.collapse_shape %alloc [[0], [1, 2]] : memref<2x2x8xbf16, #hivm.address_space<ub>> into memref<2x16xbf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<2x16xbf16, #hivm.address_space<gm>>) outs(%tile : memref<2x16xbf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%flat : memref<32xbf16, #hivm.address_space<ub>>) outs(%dst : memref<32xbf16, #hivm.address_space<gm>>)
    return
  }
}
