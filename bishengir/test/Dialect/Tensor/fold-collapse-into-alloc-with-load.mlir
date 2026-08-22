// RUN: bishengir-opt -fold-collapse-into-alloc-with-load -allow-unregistered-dialect %s -split-input-file | FileCheck %s

// The bug this prevents: InferHIVMDataLayout classifies a GM->cbuf copy by
// its src rank (rank-3 == batch). A scaffolded [d0,1,d2] chain is therefore
// misconverted into a per-row batch ND2NZ loop. These tests pin the decision
// input: after the fold, the copy src is genuinely rank-2, so classification
// takes the single-ND2NZ path. (The numerical consequence is covered by the
// fused_chunk_based_bwd_kernel end-to-end test: error ratio 5723 -> 1.000.)

// Heal case: a rank-3 unit-dim alloc whose only non-copy user is a unit-dim
// collapse gets folded to rank-2; the copy src is collapsed to rank-2 and the
// collapse disappears.
// CHECK-LABEL: func.func @fold_scaffold_chain
func.func @fold_scaffold_chain(%src: memref<16x1x16xf16, strided<[16, 16, 1], offset: ?>>) {
  // CHECK: %[[ALLOC2:.*]] = memref.alloc() : memref<16x16xf16>
  // CHECK: %[[SV2:.*]] = memref.subview %[[ALLOC2]][0, 0] [16, 16] [1, 1]
  // CHECK: %[[SRC2:.*]] = memref.collapse_shape %{{.*}} into memref<16x16xf16, strided<[16, 1], offset: ?>>
  // CHECK: hivm.hir.load ins(%[[SRC2]] {{.*}}) outs(%[[SV2]]
  // CHECK: "use_matrix"(%[[ALLOC2]]) : (memref<16x16xf16>)
  %alloc = memref.alloc() : memref<16x1x16xf16>
  %sv = memref.subview %alloc[0, 0, 0] [16, 1, 16] [1, 1, 1] : memref<16x1x16xf16> to memref<16x1x16xf16>
  hivm.hir.load ins(%src : memref<16x1x16xf16, strided<[16, 16, 1], offset: ?>>) outs(%sv : memref<16x1x16xf16>)
  %c = memref.collapse_shape %alloc [[0, 1], [2]] : memref<16x1x16xf16> into memref<16x16xf16>
  "use_matrix"(%c) : (memref<16x16xf16>) -> ()
  return
}

// -----

// Negative case: a collapse merging two non-unit dims is a genuine reshape,
// the chain must stay untouched.
// CHECK-LABEL: func.func @no_fold_genuine_reshape
func.func @no_fold_genuine_reshape(%src: memref<2x8x8xf16>) {
  // CHECK: memref.collapse_shape %{{.*}} : memref<2x8x8xf16> into memref<2x64xf16>
  %alloc = memref.alloc() : memref<2x8x8xf16>
  %c = memref.collapse_shape %alloc [[0], [1, 2]] : memref<2x8x8xf16> into memref<2x64xf16>
  "use_matrix"(%c) : (memref<2x64xf16>) -> ()
  return
}

// -----

// Negative case: the alloc has a user that is neither the collapse nor a
// subview feeding a copy — the fold must conservatively bail and leave the
// whole chain untouched.
// CHECK-LABEL: func.func @no_fold_unknown_user
func.func @no_fold_unknown_user(%src: memref<16x1x16xf16>) {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16x1x16xf16>
  // CHECK: memref.collapse_shape %[[ALLOC]] {{.*}} into memref<16x16xf16>
  %alloc = memref.alloc() : memref<16x1x16xf16>
  %c = memref.collapse_shape %alloc [[0, 1], [2]] : memref<16x1x16xf16> into memref<16x16xf16>
  "direct_use"(%alloc) : (memref<16x1x16xf16>) -> ()
  "use_matrix"(%c) : (memref<16x16xf16>) -> ()
  return
}
