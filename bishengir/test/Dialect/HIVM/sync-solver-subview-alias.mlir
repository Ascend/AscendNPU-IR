// RUN: bishengir-opt -hivm-graph-sync-solver -split-input-file %s | FileCheck %s
// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-graph-sync-solver{enable-subview-conflict-refinement=false}))" -split-input-file %s | FileCheck %s --check-prefix=DISABLED

// Sibling subviews that are disjoint in one static dimension must not
// introduce a same-pipe barrier. This also covers a multi-buffer pointer_cast.
// The full-width store still overlaps both subviews and keeps the required
// MTE2-to-MTE3 synchronization.
module {
  // CHECK-LABEL: func.func @disjoint_sibling_subviews
  // DISABLED-LABEL: func.func @disjoint_sibling_subviews
  func.func @disjoint_sibling_subviews(
      %arg0: memref<?x256xf32, #hivm.address_space<gm>>,
      %arg1: memref<?x256xf32, #hivm.address_space<gm>>,
      %arg2: memref<?x512xf32, #hivm.address_space<gm>>,
      %rows: index) {
    %c0_i64 = arith.constant 0 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %ub = hivm.hir.pointer_cast(%c0_i64, %c65536_i64) : memref<32x512xf32, #hivm.address_space<ub>>
    %left = memref.subview %ub[0, 0] [%rows, 256] [1, 1] : memref<32x512xf32, #hivm.address_space<ub>> to memref<?x256xf32, strided<[512, 1]>, #hivm.address_space<ub>>
    %right = memref.subview %ub[0, 256] [%rows, 256] [1, 1] : memref<32x512xf32, #hivm.address_space<ub>> to memref<?x256xf32, strided<[512, 1], offset: 256>, #hivm.address_space<ub>>
    // CHECK: hivm.hir.load
    // DISABLED: hivm.hir.load
    hivm.hir.load ins(%arg0 : memref<?x256xf32, #hivm.address_space<gm>>) outs(%left : memref<?x256xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
    // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_MTE2>]
    // DISABLED: hivm.hir.pipe_barrier[<PIPE_MTE2>]
    // CHECK: hivm.hir.load
    // DISABLED: hivm.hir.load
    hivm.hir.load ins(%arg1 : memref<?x256xf32, #hivm.address_space<gm>>) outs(%right : memref<?x256xf32, strided<[512, 1], offset: 256>, #hivm.address_space<ub>>)
    %full = memref.subview %ub[0, 0] [%rows, 512] [1, 1] : memref<32x512xf32, #hivm.address_space<ub>> to memref<?x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.store
    hivm.hir.store ins(%full : memref<?x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%arg2 : memref<?x512xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

// ValueBounds can prove these subviews disjoint within one iteration by
// cancelling %iv. Keep the conflict because a double-buffer slot is reused and
// the relation may not hold between different iterations.
module {
  // CHECK-LABEL: func.func @multibuffer_loop_dependent_offsets
  func.func @multibuffer_loop_dependent_offsets(
      %arg0: memref<1x256xf32, #hivm.address_space<gm>>,
      %arg1: memref<1x256xf32, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    scf.for %iv = %c0 to %c3 step %c1 {
      %ub = hivm.hir.pointer_cast(%c0_i64, %c65536_i64) : memref<32x512xf32, #hivm.address_space<ub>>
      %row_plus_2 = affine.apply affine_map<(d0) -> (d0 + 2)>(%iv)
      %first = memref.subview %ub[%row_plus_2, 0] [1, 256] [1, 1] : memref<32x512xf32, #hivm.address_space<ub>> to memref<1x256xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
      %second = memref.subview %ub[%iv, 0] [1, 256] [1, 1] : memref<32x512xf32, #hivm.address_space<ub>> to memref<1x256xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: hivm.hir.load
      hivm.hir.load ins(%arg0 : memref<1x256xf32, #hivm.address_space<gm>>) outs(%first : memref<1x256xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE2>]
      // CHECK: hivm.hir.load
      hivm.hir.load ins(%arg1 : memref<1x256xf32, #hivm.address_space<gm>>) outs(%second : memref<1x256xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>)
    }
    return
  }
}

// -----

// Overlapping sibling subviews must retain the original conservative barrier.
module {
  // CHECK-LABEL: func.func @overlapping_sibling_subviews
  func.func @overlapping_sibling_subviews(
      %arg0: memref<32x256xf32, #hivm.address_space<gm>>,
      %arg1: memref<32x256xf32, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %ub = hivm.hir.pointer_cast(%c0_i64) : memref<32x512xf32, #hivm.address_space<ub>>
    %first = memref.subview %ub[0, 0] [32, 256] [1, 1] : memref<32x512xf32, #hivm.address_space<ub>> to memref<32x256xf32, strided<[512, 1]>, #hivm.address_space<ub>>
    %second = memref.subview %ub[0, 128] [32, 256] [1, 1] : memref<32x512xf32, #hivm.address_space<ub>> to memref<32x256xf32, strided<[512, 1], offset: 128>, #hivm.address_space<ub>>
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%arg0 : memref<32x256xf32, #hivm.address_space<gm>>) outs(%first : memref<32x256xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE2>]
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%arg1 : memref<32x256xf32, #hivm.address_space<gm>>) outs(%second : memref<32x256xf32, strided<[512, 1], offset: 128>, #hivm.address_space<ub>>)
    return
  }
}

// -----

// If the relative offset cannot be bounded, keep treating the accesses as
// conflicting.
module {
  // CHECK-LABEL: func.func @unknown_subview_overlap
  func.func @unknown_subview_overlap(
      %arg0: memref<32x256xf32, #hivm.address_space<gm>>,
      %arg1: memref<32x256xf32, #hivm.address_space<gm>>,
      %offset: index) {
    %c0_i64 = arith.constant 0 : i64
    %ub = hivm.hir.pointer_cast(%c0_i64) : memref<32x1024xf32, #hivm.address_space<ub>>
    %first = memref.subview %ub[0, 0] [32, 256] [1, 1] : memref<32x1024xf32, #hivm.address_space<ub>> to memref<32x256xf32, strided<[1024, 1]>, #hivm.address_space<ub>>
    %second = memref.subview %ub[0, %offset] [32, 256] [1, 1] : memref<32x1024xf32, #hivm.address_space<ub>> to memref<32x256xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<ub>>
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%arg0 : memref<32x256xf32, #hivm.address_space<gm>>) outs(%first : memref<32x256xf32, strided<[1024, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE2>]
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%arg1 : memref<32x256xf32, #hivm.address_space<gm>>) outs(%second : memref<32x256xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<ub>>)
    return
  }
}

// -----

// A subview whose source is not a pointer_cast remains on the original
// traceback path. All roots of the selected buffer must still be collected.
module {
  // CHECK-LABEL: func.func @subview_of_selected_buffer
  func.func @subview_of_selected_buffer(
      %arg0: memref<32x256xf32, #hivm.address_space<gm>>,
      %arg1: memref<32x256xf32, #hivm.address_space<gm>>,
      %cond: i1) {
    %c0_i64 = arith.constant 0 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %ub0 = hivm.hir.pointer_cast(%c0_i64) : memref<32x512xf32, #hivm.address_space<ub>>
    %ub1 = hivm.hir.pointer_cast(%c65536_i64) : memref<32x512xf32, #hivm.address_space<ub>>
    %selected = arith.select %cond, %ub0, %ub1 : memref<32x512xf32, #hivm.address_space<ub>>
    %direct = memref.subview %ub0[0, 0] [32, 256] [1, 1] : memref<32x512xf32, #hivm.address_space<ub>> to memref<32x256xf32, strided<[512, 1]>, #hivm.address_space<ub>>
    %selected_view = memref.subview %selected[0, 0] [32, 256] [1, 1] : memref<32x512xf32, #hivm.address_space<ub>> to memref<32x256xf32, strided<[512, 1]>, #hivm.address_space<ub>>
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%arg0 : memref<32x256xf32, #hivm.address_space<gm>>) outs(%direct : memref<32x256xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE2>]
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%arg1 : memref<32x256xf32, #hivm.address_space<gm>>) outs(%selected_view : memref<32x256xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Nested subviews are outside the refinement scope. Even though the outer
// subviews are disjoint, traceback must continue to the root buffer and retain
// the original conservative barrier.
module {
  // CHECK-LABEL: func.func @nested_subviews
  func.func @nested_subviews(
      %arg0: memref<32x256xf32, #hivm.address_space<gm>>,
      %arg1: memref<32x256xf32, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %ub = hivm.hir.pointer_cast(%c0_i64) : memref<32x512xf32, #hivm.address_space<ub>>
    %parent = memref.subview %ub[0, 0] [32, 512] [1, 1] : memref<32x512xf32, #hivm.address_space<ub>> to memref<32x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
    %left = memref.subview %parent[0, 0] [32, 256] [1, 1] : memref<32x512xf32, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x256xf32, strided<[512, 1]>, #hivm.address_space<ub>>
    %right = memref.subview %parent[0, 256] [32, 256] [1, 1] : memref<32x512xf32, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x256xf32, strided<[512, 1], offset: 256>, #hivm.address_space<ub>>
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%arg0 : memref<32x256xf32, #hivm.address_space<gm>>) outs(%left : memref<32x256xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE2>]
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%arg1 : memref<32x256xf32, #hivm.address_space<gm>>) outs(%right : memref<32x256xf32, strided<[512, 1], offset: 256>, #hivm.address_space<ub>>)
    return
  }
}
