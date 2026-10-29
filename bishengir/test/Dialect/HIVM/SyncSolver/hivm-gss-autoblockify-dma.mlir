// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v1" %s | FileCheck %s
// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v2" %s | FileCheck %s

// Iterations of an auto-blockify loop handle different logical blocks, so a
// DMA op marked by auto-blockify never conflicts with itself across iterations.

module {
  // CHECK-LABEL: func.func @autoblockify_marked_store
  func.func @autoblockify_marked_store(%gm: memref<16xf32, #hivm.address_space<gm>>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %ub = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
    // CHECK: scf.for
    // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_MTE3>]
    // CHECK: hivm.hir.store
    // CHECK: } {autoblockify.subloop}
    scf.for %arg0 = %c0 to %n step %c1 {
      hivm.hir.store ins(%ub : memref<16xf32, #hivm.address_space<ub>>) outs(%gm : memref<16xf32, #hivm.address_space<gm>>) {autoblockify.dma}
    } {autoblockify.subloop}
    return
  }

  // Stores not marked by auto-blockify keep their cross-iteration sync.
  // CHECK-LABEL: func.func @autoblockify_unmarked_store
  func.func @autoblockify_unmarked_store(%gm: memref<16xf32, #hivm.address_space<gm>>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %ub = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
    // CHECK: scf.for
    // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE3>]
    // CHECK-NEXT: hivm.hir.store
    scf.for %arg0 = %c0 to %n step %c1 {
      hivm.hir.store ins(%ub : memref<16xf32, #hivm.address_space<ub>>) outs(%gm : memref<16xf32, #hivm.address_space<gm>>)
    } {autoblockify.subloop}
    return
  }

  // A loop rebuilt without the auto-blockify attr keeps its sync.
  // CHECK-LABEL: func.func @plain_loop_marked_store
  func.func @plain_loop_marked_store(%gm: memref<16xf32, #hivm.address_space<gm>>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %ub = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
    // CHECK: scf.for
    // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE3>]
    // CHECK-NEXT: hivm.hir.store
    scf.for %arg0 = %c0 to %n step %c1 {
      hivm.hir.store ins(%ub : memref<16xf32, #hivm.address_space<ub>>) outs(%gm : memref<16xf32, #hivm.address_space<gm>>) {autoblockify.dma}
    }
    return
  }

  // The inner loop is not auto-blockify, so its iterations still conflict.
  // CHECK-LABEL: func.func @autoblockify_inner_loop_marked_store
  func.func @autoblockify_inner_loop_marked_store(%gm: memref<16xf32, #hivm.address_space<gm>>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %ub = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE3>]
    // CHECK-NEXT: hivm.hir.store
    scf.for %arg0 = %c0 to %n step %c1 {
      scf.for %arg1 = %c0 to %n step %c1 {
        hivm.hir.store ins(%ub : memref<16xf32, #hivm.address_space<ub>>) outs(%gm : memref<16xf32, #hivm.address_space<gm>>) {autoblockify.dma}
      }
    } {autoblockify.subloop}
    return
  }

  // Different marked stores may still write the same GM across iterations.
  // CHECK-LABEL: func.func @autoblockify_two_marked_stores
  func.func @autoblockify_two_marked_stores(%gm: memref<16xf32, #hivm.address_space<gm>>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %ub = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
    // CHECK: scf.for
    // CHECK-NEXT: hivm.hir.pipe_barrier[<PIPE_MTE3>]
    // CHECK-NEXT: hivm.hir.store
    // CHECK-NEXT: hivm.hir.pipe_barrier[<PIPE_MTE3>]
    // CHECK-NEXT: hivm.hir.store
    scf.for %arg0 = %c0 to %n step %c1 {
      hivm.hir.store ins(%ub : memref<16xf32, #hivm.address_space<ub>>) outs(%gm : memref<16xf32, #hivm.address_space<gm>>) {autoblockify.dma}
      hivm.hir.store ins(%ub : memref<16xf32, #hivm.address_space<ub>>) outs(%gm : memref<16xf32, #hivm.address_space<gm>>) {autoblockify.dma}
    } {autoblockify.subloop}
    return
  }

  // CHECK-LABEL: func.func @autoblockify_marked_fixpipe
  func.func @autoblockify_marked_fixpipe(%gm: memref<16x16xf32, #hivm.address_space<gm>>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %l0c = hivm.hir.pointer_cast(%c0_i64) : memref<16x16xf32, #hivm.address_space<cc>>
    // CHECK: scf.for
    // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK: hivm.hir.fixpipe
    // CHECK: } {autoblockify.subloop}
    scf.for %arg0 = %c0 to %n step %c1 {
      hivm.hir.fixpipe {autoblockify.dma, dma_mode = #hivm.dma_mode<nz2nd>} ins(%l0c : memref<16x16xf32, #hivm.address_space<cc>>) outs(%gm : memref<16x16xf32, #hivm.address_space<gm>>)
    } {autoblockify.subloop}
    return
  }

  // CHECK-LABEL: func.func @autoblockify_unmarked_fixpipe
  func.func @autoblockify_unmarked_fixpipe(%gm: memref<16x16xf32, #hivm.address_space<gm>>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %l0c = hivm.hir.pointer_cast(%c0_i64) : memref<16x16xf32, #hivm.address_space<cc>>
    // CHECK: scf.for
    // CHECK-NEXT: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK-NEXT: hivm.hir.fixpipe
    scf.for %arg0 = %c0 to %n step %c1 {
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%l0c : memref<16x16xf32, #hivm.address_space<cc>>) outs(%gm : memref<16x16xf32, #hivm.address_space<gm>>)
    } {autoblockify.subloop}
    return
  }
}
